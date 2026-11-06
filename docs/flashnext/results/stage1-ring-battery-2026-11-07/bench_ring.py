"""Stage 1 ring ON vs OFF benchmark on synthetic Q4/G32 tensors (battery mode).

Synthetic random weights with real Qwen3.8-Flash-Next expert row sizes
(hidden 2560, intermediate 640, 3,072,000 B per expert), a real
SafeTensorStore, real StreamingSwitchGLU layers, a real ExpertPool and the
real prefill_ring session. MoE layers only: no attention, router or n-gram
work, so this isolates the expert-loading path and its GPU gather_qmm.

Arms (reversed, interleaved per round): off, on, idle (control).
Safety: a watchdog stops the run on battery < 25%, memory pressure >= warn,
new swap-outs, CPU speed limit < 90, or MLX peak + pool over the Metal limit.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

# Paths are configurable: VEYSTRIX_REPO (default: the repository containing this
# file) and BENCH_SCRATCH (default: a directory under the system temp dir).
REPO = os.environ.get(
    "VEYSTRIX_REPO",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
)
SP = os.environ.get("BENCH_SCRATCH", os.path.join(tempfile.gettempdir(), "veystrix_bench"))
os.makedirs(SP, exist_ok=True)
sys.path.insert(0, REPO)

HIDDEN, INTER, EXPERTS = 2560, 640, 512
EXPERT_BYTES = 3_072_000
SHAPES = {
    "gate_proj": (INTER, HIDDEN // 8, HIDDEN // 32),
    "up_proj": (INTER, HIDDEN // 8, HIDDEN // 32),
    "down_proj": (HIDDEN, INTER // 8, INTER // 32),
}
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
# Tokens and distinct routed experts per layer. 334 at ~1K tokens is recorded;
# 400 at 2K is an earlier estimate; 460 at 4K is a hypothesis, not a measurement.
SIZES = {"1k": (1024, 334), "2k": (2048, 400), "4k": (4096, 460)}
KEYS = ("Pageins", "Pageouts", "Swapins", "Swapouts", "Compressions", "Decompressions")

BATTERY_MIN = 25
SPEED_LIMIT_MIN = 90


class BenchAbort(Exception):
    pass


CURRENT = {"size": None, "round": None, "arm": None, "pos": None}


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def prefix_of(layer):
    return f"language_model.model.layers.{layer}.mlp.switch_mlp"


def vm_counters():
    out = {}
    for line in sh("vm_stat").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip().rstrip(".")
            if value.isdigit():
                out[key.strip()] = int(value)
    return out


def free_gb(c):
    return (c.get("Pages free", 0) + c.get("Pages inactive", 0)
            + c.get("Pages speculative", 0)) * 16384 / 1e9


def battery():
    text = sh("pmset", "-g", "batt")
    match = re.search(r"(\d+)%", text)
    return {"percent": int(match.group(1)) if match else None,
            "discharging": "discharging" in text,
            "source": "battery" if "Battery Power" in text else "ac"}


def speed_limit():
    text = sh("pmset", "-g", "therm")
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    warning = "No thermal warning level has been recorded" not in text
    return (int(match.group(1)) if match else 100), warning, text.strip().replace("\n", " | ")


def pressure_level():
    try:
        return int(sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip())
    except ValueError:
        return 0


def low_power():
    return "lowpowermode         1" in sh("pmset", "-g") or "lowpowermode\t1" in sh("pmset", "-g")


def swap_used_mb():
    match = re.search(r"used = ([\d.]+)M", sh("sysctl", "-n", "vm.swapusage"))
    return float(match.group(1)) if match else None


class Watchdog(threading.Thread):
    def __init__(self, baseline_swapouts, interval=2.5):
        super().__init__(daemon=True)
        self.baseline = baseline_swapouts
        self.interval = interval
        self.reason = None
        self.stop = threading.Event()

    def check(self):
        b = battery()
        if b["percent"] is not None and b["discharging"] and b["percent"] < BATTERY_MIN:
            return f"battery {b['percent']}% < {BATTERY_MIN}%"
        level = pressure_level()
        if level >= 2:
            return f"memory pressure level {level}"
        swapouts = vm_counters().get("Swapouts", 0)
        if swapouts > self.baseline:
            return f"swapping began: Swapouts {self.baseline} -> {swapouts}"
        limit, warning, text = speed_limit()
        if limit < SPEED_LIMIT_MIN or warning:
            return f"thermal: CPU_Speed_Limit={limit} warning={warning} ({text})"
        if low_power():
            return "Low Power Mode is on"
        return None

    def run(self):
        while not self.stop.wait(self.interval):
            reason = self.check()
            if reason:
                self.reason = f"{reason} [during {dict(CURRENT)}]"
                return


def make_checkpoint(directory, layers):
    os.makedirs(directory, exist_ok=True)
    header, offset, plan = {}, 0, []
    for layer in range(1, layers + 1):
        for projection, (rows, packed, groups) in SHAPES.items():
            for kind, dtype, tail, size in (
                ("weight", "U32", (rows, packed), 4),
                ("scales", "BF16", (rows, groups), 2),
                ("biases", "BF16", (rows, groups), 2),
            ):
                shape = [EXPERTS, *tail]
                nbytes = int(np.prod(shape)) * size
                name = f"{prefix_of(layer)}.{projection}.{kind}"
                header[name] = {"dtype": dtype, "shape": shape,
                                "data_offsets": [offset, offset + nbytes]}
                plan.append((kind, nbytes))
                offset += nbytes
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    shard = "model-00001-of-00001.safetensors"
    rng = np.random.default_rng(5)
    fd = os.open(os.path.join(directory, shard), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    try:
        os.write(fd, len(blob).to_bytes(8, "little") + blob)
        block = 32 * 1024 * 1024
        for kind, nbytes in plan:
            done = 0
            while done < nbytes:
                piece = min(block, nbytes - done)
                if kind == "weight":
                    os.write(fd, rng.bytes(piece))
                else:
                    # Finite, small bf16 scales/biases (truncated float32 bits).
                    values = (rng.random(piece // 2, dtype=np.float32) * 0.02 - 0.01)
                    bits = (values.view(np.uint32) >> 16).astype("<u2")
                    os.write(fd, bits.tobytes())
                done += piece
    finally:
        os.close(fd)
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
        json.dump({"weight_map": {key: shard for key in header}}, out)
    return offset


def paired_stats(ratios):
    ratios = sorted(ratios)
    n = len(ratios)
    out = {"n": n, "median": round(statistics.median(ratios), 3),
           "min": round(ratios[0], 3), "max": round(ratios[-1], 3),
           "mean": round(statistics.mean(ratios), 3)}
    if n > 1:
        sd = statistics.stdev(ratios)
        out["stdev"] = round(sd, 3)
        # Student t 95% half-width for small n.
        t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36}.get(n, 2.0)
        out["ci95_half_width"] = round(t * sd / n ** 0.5, 3)
    out["on_faster_in"] = sum(1 for r in ratios if r > 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1k")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--file-layers", type=int, default=4)
    ap.add_argument("--arm-layers", type=int, default=4)
    ap.add_argument("--dir", default=os.path.join(SP, "scratch_bench"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-dir", action="store_true")
    ap.add_argument("--design", choices=("alternate", "blocked"), default="alternate")
    ap.add_argument("--block", type=int, default=3)
    args = ap.parse_args()

    c0 = vm_counters()
    b0 = battery()
    print(f"start: battery {b0}, free {free_gb(c0):.2f} GB, pressure {pressure_level()}, "
          f"low_power={low_power()}, speed_limit={speed_limit()[0]}, "
          f"swap_used={swap_used_mb()} MB", flush=True)
    if b0["source"] != "battery" or low_power():
        print("note: expected battery power with Low Power Mode off; recording as found")
    if (b0["percent"] or 0) < BATTERY_MIN + 5 or free_gb(c0) < 5.0 or pressure_level() >= 2:
        print("refusing to start: battery, free memory or pressure not safe")
        return 2
    disk = os.statvfs(SP)
    need = args.file_layers * 1.6e9 + 20e9
    if disk.f_bavail * disk.f_frsize < need:
        print("refusing: not enough disk")
        return 2

    os.environ.update({
        "FLASHNEXT_EXPERT_POOL_GB": "6",
        "FLASHNEXT_PREFILL_RING_EXPERTS": "512",
        "FLASHNEXT_PREFILL_RING_MIN_TOKENS": "0",
    })
    import mlx.core as mx
    import mlx.nn as nn
    from types import SimpleNamespace

    from models.flashnext import expert_cache, expert_pool, prefill_ring
    from models.flashnext.store import SafeTensorStore

    watchdog = Watchdog(c0.get("Swapouts", 0))
    result = {
        "script": "bench_ring.py", "note": "battery-mode preliminary, MoE layers only, synthetic",
        "start": {"battery": b0, "low_power": low_power(), "pressure": pressure_level(),
                  "speed_limit": speed_limit()[0], "swap_used_mb": swap_used_mb(),
                  "free_gb": round(free_gb(c0), 2), "thermal_text": speed_limit()[2]},
        "args": vars(args), "sizes": {}, "aborted": None,
        "mlx": mx.__version__ if hasattr(mx, "__version__") else "0.32.2",
    }
    store = None
    pool = None
    try:
        began = time.perf_counter()
        total = make_checkpoint(args.dir, args.file_layers)
        print(f"checkpoint {total / 1e9:.1f} GB written in {time.perf_counter() - began:.0f}s",
              flush=True)
        result["checkpoint_gb"] = round(total / 1e9, 2)
        store = SafeTensorStore(args.dir)
        glus = [expert_cache.StreamingSwitchGLU(
            store, prefix_of(layer), 32, 4, "affine",
            lambda up, gate: nn.silu(gate) * up, layer_id=layer)
            for layer in range(1, args.file_layers + 1)]
        language = SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(switch_mlp=g)) for g in glus]))
        pool = expert_pool.for_store(store, 32)
        pool_bytes = len(pool._mm)
        limit = prefill_ring.metal_working_set_limit()
        result["metal_limit_gb"] = round(limit / 1e9, 3)
        result["pool_gb"] = round(pool_bytes / 1e9, 3)

        # -- timing instrumentation --------------------------------------------
        timers = {"read": 0.0, "ring_calls": 0, "off_calls": 0}
        original_read = expert_cache.StreamingSwitchGLU._read_weights
        original_stage = prefill_ring.PrefillRing.stage

        def timed_read(self, *a, **k):
            t = time.perf_counter()
            try:
                return original_read(self, *a, **k)
            finally:
                timers["read"] += time.perf_counter() - t
                timers["off_calls"] += 1

        def timed_stage(self, *a, **k):
            t = time.perf_counter()
            try:
                return original_stage(self, *a, **k)
            finally:
                timers["read"] += time.perf_counter() - t
                timers["ring_calls"] += 1

        expert_cache.StreamingSwitchGLU._read_weights = timed_read
        prefill_ring.PrefillRing.stage = timed_stage

        def probe():
            a = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
            mx.eval(a)
            began = time.perf_counter()
            for _ in range(8):
                mx.eval(a @ a)
            return round((time.perf_counter() - began) / 8 * 1000, 2)

        def make_workload(tokens, unique, rng, count):
            work = []
            for _ in range(count):
                chosen = rng.permutation(EXPERTS)[:unique]
                routed = np.concatenate([chosen, rng.choice(chosen, tokens * 10 - unique)])
                work.append(routed.reshape(1, tokens, 10).astype(np.uint32))
            return work

        def run_arm(arm, tokens, unique, layer_ids, workload, x, digests=None, idle_s=4.0, pos=0):
            CURRENT.update(arm=arm, pos=pos)
            if watchdog.reason:
                raise BenchAbort(watchdog.reason)
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            rec = {"arm": arm, "pos": pos, "probe_ms": probe()}
            mx.reset_peak_memory()
            timers.update(read=0.0, ring_calls=0, off_calls=0)
            before = vm_counters()
            bat = battery()
            began = time.perf_counter()
            gpu = 0.0
            staged = 0
            fallbacks = 0
            if arm == "idle":
                time.sleep(idle_s)
            else:
                def body():
                    nonlocal gpu
                    for index, layer in enumerate(layer_ids):
                        if watchdog.reason:
                            raise BenchAbort(watchdog.reason)
                        indices = mx.array(workload[index])
                        y = glus[layer](x, indices)
                        t = time.perf_counter()
                        mx.eval(y)
                        gpu += time.perf_counter() - t
                        if digests is not None:
                            host = np.array(y.view(mx.uint16))
                            digests.append(hashlib.blake2b(memoryview(host).cast("B"), digest_size=16).hexdigest())
                            del host
                        del y

                if arm == "on":
                    with prefill_ring.session(language, tokens) as ring:
                        if ring is None:
                            raise RuntimeError(f"ring refused: {prefill_ring.LAST_REASON[0]}")
                        body()
                        stats = ring.stats()
                        staged = stats["staged_layers"]
                        fallbacks = len(layer_ids) - staged
                        rec["ring_stats"] = {k: stats[k] for k in (
                            "staged_layers", "fallback_capacity", "fallback_busy",
                            "fallback_error", "fallback_memory", "disabled")}
                else:
                    body()
            wall = time.perf_counter() - began
            after = vm_counters()
            delta = {k: after.get(k, 0) - before.get(k, 0) for k in KEYS}
            nbytes = unique * EXPERT_BYTES * len(layer_ids) if arm != "idle" else 0
            peak = mx.get_peak_memory()
            rec.update({
                "wall_s": round(wall, 4), "read_s": round(timers["read"], 4),
                "gpu_s": round(gpu, 4), "other_s": round(max(0.0, wall - timers["read"] - gpu), 4),
                "bytes": nbytes,
                "read_gb_s": round(nbytes / 1e9 / timers["read"], 3) if timers["read"] else None,
                "read_ms_per_layer": round(timers["read"] / len(layer_ids) * 1000, 1) if arm != "idle" else None,
                "gpu_ms_per_layer": round(gpu / len(layer_ids) * 1000, 1) if arm != "idle" else None,
                "staged": staged, "fallbacks": fallbacks, "vm": delta,
                "compressor_pages_before": before.get("Pages occupied by compressor", 0),
                "compressor_pages_after": after.get("Pages occupied by compressor", 0),
                "mlx_peak_mb": round(peak / 1e6, 1),
                "mlx_active_mb_after": round(mx.get_active_memory() / 1e6, 1),
                "mlx_peak_gb_incl_pool": round(peak / 1e9, 3),  # MLX counts the wrapped pool
                "free_gb_after": round(free_gb(after), 2),
                "battery_before": bat["percent"], "battery_after": battery()["percent"],
                "speed_limit": speed_limit()[0], "pressure": pressure_level(),
                "swap_used_mb": swap_used_mb(),
            })
            if arm == "on" and fallbacks:
                raise RuntimeError(f"ring fell back on {fallbacks} layers: {rec.get('ring_stats')}")
            if peak > limit - 200_000_000:
                raise BenchAbort(
                    f"MLX peak (includes pool) = {rec['mlx_peak_gb_incl_pool']} GB is within "
                    f"0.2 GB of the {limit / 1e9:.2f} GB Metal limit")
            return rec

        # -- warm-up (compile kernels and fault in code, untimed, unrecorded) -----
        watchdog.start()
        rng = np.random.default_rng(99)
        x_warm = mx.random.normal((1, 1024, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x_warm)
        warm = make_workload(1024, 100, rng, 1)
        run_arm("off", 1024, 100, [0], warm, x_warm)
        run_arm("on", 1024, 100, [0], warm, x_warm)
        print("warm-up ok", flush=True)

        for name in [s.strip() for s in args.sizes.split(",") if s.strip()]:
            tokens, unique = SIZES[name]
            mx.random.seed(7)
            x = mx.random.normal((1, tokens, HIDDEN)).astype(mx.bfloat16)
            mx.eval(x)
            size_result = {"tokens": tokens, "unique_experts": unique, "rounds": [],
                           "arm_layers": args.arm_layers}
            result["sizes"][name] = size_result
            rng = np.random.default_rng(1000 + tokens)

            CURRENT.update(size=name, round="verify")
            # Bit-identity: identical routes through both arms.
            same = make_workload(tokens, unique, rng, args.arm_layers)
            ids = list(range(args.arm_layers))
            d_off, d_on = [], []
            run_arm("off", tokens, unique, ids, same, x, digests=d_off)
            run_arm("on", tokens, unique, ids, same, x, digests=d_on)
            identical = d_off == d_on
            size_result["bit_identical"] = identical
            size_result["digest_layers"] = len(d_off)
            print(f"[{name}] bit-identical over {len(d_off)} layers: {identical}", flush=True)
            if not identical:
                raise BenchAbort(f"[{name}] ring output is NOT bit-identical; stopping")

            idle_s = 4.0
            counter = 0
            CURRENT.update(size=name)
            for rnd in range(args.rounds):
                CURRENT.update(round=rnd)
                if args.design == "alternate":
                    names = ["off", "on", "idle"] if rnd % 2 == 0 else ["idle", "on", "off"]
                    order = [(a, 0) for a in names]
                else:
                    first, second = ("off", "on") if rnd % 2 == 0 else ("on", "off")
                    order = ([(first, i) for i in range(args.block)]
                             + [(second, i) for i in range(args.block)] + [("idle", 0)])
                record = {"round": rnd, "order": [f"{a}{p}" for a, p in order], "arms": []}
                for arm, pos in order:
                    layer_ids = [(counter + i) % args.file_layers for i in range(args.arm_layers)]
                    counter += args.arm_layers
                    work = make_workload(tokens, unique, rng, args.arm_layers)
                    rec = run_arm(arm, tokens, unique, layer_ids, work, x, idle_s=idle_s, pos=pos)
                    record["arms"].append(rec)
                    if arm == "off":
                        idle_s = rec["wall_s"]
                    print(f"[{name}] r{rnd} {arm:4s}#{pos} wall={rec['wall_s']:6.2f}s "
                          f"read={rec['read_s']:6.2f}s gpu={rec['gpu_s']:5.2f}s "
                          f"{(rec['read_gb_s'] or 0):5.2f}GB/s comp={rec['vm']['Compressions']:>7} "
                          f"decomp={rec['vm']['Decompressions']:>7} swapouts={rec['vm']['Swapouts']} "
                          f"peak={rec['mlx_peak_gb_incl_pool']}GB(incl pool) "
                          f"bat={rec['battery_after']}% pressure={rec['pressure']} probe={rec['probe_ms']}ms "
                          f"free={rec['free_gb_after']}GB", flush=True)
                size_result["rounds"].append(record)

            rounds = size_result["rounds"]
            allarms = [a for r in rounds for a in r["arms"]]

            def pick(r, arm, steady):
                recs = [a for a in r["arms"] if a["arm"] == arm and (a["pos"] >= 1 or not steady)]
                return recs

            def mean(vals):
                vals = list(vals)
                return sum(vals) / len(vals) if vals else None

            summary = {}
            for label, steady in (("all_arms", False), ("steady_state", True)):
                ratios = {"wall": [], "read": [], "gpu": []}
                for r in rounds:
                    off, on = pick(r, "off", steady), pick(r, "on", steady)
                    if not off or not on:
                        continue
                    for key, field in (("wall", "wall_s"), ("read", "read_s"), ("gpu", "gpu_s")):
                        ratios[key].append(mean(a[field] for a in off) / mean(a[field] for a in on))
                if ratios["wall"]:
                    summary[label] = {f"{k}_off_over_on": paired_stats(v) for k, v in ratios.items()}
            for arm in ("off", "on", "idle"):
                for label, steady in (("all", False), ("steady", True)):
                    recs = [a for a in allarms if a["arm"] == arm and (a["pos"] >= 1 or not steady)]
                    if not recs:
                        continue
                    wall = sum(r["wall_s"] for r in recs)
                    entry = {
                        "n_arms": len(recs),
                        "median_wall_s": round(statistics.median(r["wall_s"] for r in recs), 3),
                        "compressions_per_s": round(sum(r["vm"]["Compressions"] for r in recs) / wall),
                        "decompressions_per_s": round(sum(r["vm"]["Decompressions"] for r in recs) / wall),
                        "pageins_per_s": round(sum(r["vm"]["Pageins"] for r in recs) / wall),
                        "swapouts": sum(r["vm"]["Swapouts"] for r in recs),
                        "max_mlx_peak_gb_incl_pool": max(r["mlx_peak_gb_incl_pool"] for r in recs),
                        "min_free_gb": min(r["free_gb_after"] for r in recs),
                        "max_pressure": max(r["pressure"] for r in recs),
                    }
                    if arm != "idle":
                        entry.update({
                            "median_read_ms_per_layer": round(statistics.median(r["read_ms_per_layer"] for r in recs), 1),
                            "median_gpu_ms_per_layer": round(statistics.median(r["gpu_ms_per_layer"] for r in recs), 1),
                            "median_read_gb_s": round(statistics.median(r["read_gb_s"] for r in recs), 2),
                        })
                    summary[f"{arm}_{label}"] = entry
            size_result["summary"] = summary
            print(f"[{name}] summary: " + json.dumps(summary, indent=1), flush=True)
    except BenchAbort as error:
        result["aborted"] = str(error)
        print(f"ABORTED: {error}", flush=True)
    except Exception as error:  # noqa: BLE001
        result["aborted"] = f"error: {error!r}"
        print(f"ERROR: {error!r}", flush=True)
    finally:
        watchdog.stop.set()
        result["watchdog_reason"] = watchdog.reason
        result["end"] = {"battery": battery(), "pressure": pressure_level(),
                         "speed_limit": speed_limit()[0], "swap_used_mb": swap_used_mb(),
                         "free_gb": round(free_gb(vm_counters()), 2),
                         "thermal_text": speed_limit()[2]}
        try:
            expert_cache.StreamingSwitchGLU._read_weights = original_read
            prefill_ring.PrefillRing.stage = original_stage
        except NameError:
            pass
        if store is not None:
            store.close()
        if pool is not None:
            pool.close()
        if not args.keep_dir:
            shutil.rmtree(args.dir, ignore_errors=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1, default=str)
        print("saved", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
