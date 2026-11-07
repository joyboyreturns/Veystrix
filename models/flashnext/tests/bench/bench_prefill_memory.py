#!/usr/bin/env python3
"""End-to-end prefill memory and speed: baseline, scratch pool, ring, cache cap.

Orchestrator and child in one file. The orchestrator builds a reversed,
fresh-process schedule (``prefill_memory_harness.build_schedule``), runs each
arm as ``--child`` under a safety watchdog, and records the lifecycle with
``MeasurementRun``. The child loads the real checkpoint, runs one scenario at
one size, and writes one JSON file. Plan: docs/flashnext/prefetch-integration.md
section 14.

This needs the real checkpoint and a plugged-in Mac, so it must not be run
without approval. ``--dry-run`` prints the schedule, commands and flag overlays
and touches nothing (no child, no results file, no checkpoint).

    python -m models.flashnext.tests.bench.bench_prefill_memory --dry-run
    python -m models.flashnext.tests.bench.bench_prefill_memory \
        --sizes 1k --conditions F0 F2 --rounds 1        # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402
from models.flashnext.tests.bench import prefill_memory_harness as H  # noqa: E402

MODULE = "models.flashnext.tests.bench.bench_prefill_memory"
SOURCE_TEXT = ROOT / "docs" / "flashnext" / "research.md"
FINGERPRINTED = (
    "models/flashnext/prefill.py", "models/flashnext/prefill_ring.py",
    "models/flashnext/prefill_memory.py", "models/flashnext/expert_pool.py",
    "models/flashnext/expert_cache.py", "models/flashnext/pool_ring.py",
    "models/flashnext/tests/bench/bench_prefill_memory.py",
    "models/flashnext/tests/bench/prefill_memory_harness.py",
)
EFFECT_METRICS = ("prefill_tok_s", "decode_tok_s_1_64", "decode_tok_s_65_128",
                  "peak_gb", "ssd_prefill_gb")


# -- child: one scenario in one fresh process ---------------------------------

class RealProbe:
    """Readings the child takes. Real MLX, vm_stat, rusage; faked in tests."""

    def memory(self) -> dict:
        import mlx.core as mx

        return {"active": mx.get_active_memory(), "peak": mx.get_peak_memory(),
                "cache": mx.get_cache_memory()}

    def reset_peak(self) -> None:
        import mlx.core as mx

        mx.reset_peak_memory()

    def metal_limit(self) -> int | None:
        from models.flashnext.prefill_ring import metal_working_set_limit

        return metal_working_set_limit()

    def vm(self) -> dict:
        from models.flashnext.diskio import vm_counters

        return vm_counters()

    def disk_bytes(self) -> int:
        from models.flashnext.diskio import disk_bytes_read

        return disk_bytes_read()

    def free_mb(self) -> float:
        from models.flashnext.diskio import free_memory_mb

        return free_memory_mb()

    def faults(self) -> dict:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"minor": usage.ru_minflt, "major": usage.ru_majflt}

    def rss_bytes(self) -> int | None:
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                 capture_output=True, text=True, timeout=5).stdout
            return int(out.strip()) * 1024
        except Exception:
            return None

    def now(self) -> float:
        return time.perf_counter()


def find_store(language):
    """The checkpoint store the streamed layers share, or None."""
    body = getattr(language, "model", language)
    for layer in getattr(body, "layers", ()):
        glu = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        cache = getattr(getattr(glu, "gate_proj", None), "cache", None)
        store = getattr(cache, "store", None)
        if store is not None:
            return store
    return None


def pool_state(store) -> dict:
    pool = getattr(store, "_flashnext_expert_pool", None) if store is not None else None
    if pool is None:
        return {"exists": False}
    return {"exists": True, "slots": pool.slots, "used": len(pool._slot_of),
            "hits": pool.hits, "misses": pool.misses}


class Instrument:
    """Wraps ``prefill_target``, ``acquire`` and ``release`` for the measured turns.

    It only observes: every wrapper calls the original with the same arguments
    and returns its result unchanged. Memory is read after ``release`` returns,
    so scratch cleanup (G4) is judged on the state the decode pool will see.
    """

    def __init__(self, probe, prefill_module, ring_module, language):
        self.probe, self.prefill, self.ring = probe, prefill_module, ring_module
        self.language = language
        self._saved = {}
        self.reset()

    def reset(self) -> None:
        self.calls, self.cache_max = [], 0
        self.acquired, self.reason, self.ring_stats = False, None, None
        self.acquire_s = self.dispose_s = None
        self.acquire_faults = None

    def _seen(self, reading: dict) -> dict:
        self.cache_max = max(self.cache_max, reading["cache"])
        return reading

    def __enter__(self):
        instrument = self
        original = {"target": self.prefill.prefill_target,
                    "acquire": self.ring.acquire, "release": self.ring.release}
        self._saved = dict(original)

        def prefill_target(language, ids, cache, *args, **kwargs):
            store = find_store(language)
            before = instrument._seen(instrument.probe.memory())
            record = {"tokens": int(ids.shape[1]),
                      "before": {**before, "pool_exists": pool_state(store)["exists"]}}
            began = instrument.probe.now()
            result = original["target"](language, ids, cache, *args, **kwargs)
            record["seconds"] = instrument.probe.now() - began
            after = instrument._seen(instrument.probe.memory())
            record["after"] = {**after, "pool_exists": pool_state(store)["exists"],
                               "retained": len(instrument.ring._RETAINED)}
            instrument.calls.append(record)
            return result

        def acquire(language, tokens):
            faults = instrument.probe.faults()
            began = instrument.probe.now()
            state = original["acquire"](language, tokens)
            instrument.acquire_s = (instrument.acquire_s or 0.0) + instrument.probe.now() - began
            instrument.acquire_faults = {
                key: instrument.probe.faults()[key] - faults[key] for key in faults
            }
            instrument.acquired = state is not None
            instrument.reason = instrument.ring.LAST_REASON[0]
            return state

        def release(state, failed=False):
            if state is not None:
                instrument.ring_stats = dict(state.stats())
            began = instrument.probe.now()
            original["release"](state, failed)
            if state is not None:
                instrument.dispose_s = (instrument.dispose_s or 0.0) + instrument.probe.now() - began

        self.prefill.prefill_target = prefill_target
        self.ring.acquire = acquire
        self.ring.release = release
        return self

    def __exit__(self, *exc):
        self.prefill.prefill_target = self._saved["target"]
        self.ring.acquire = self._saved["acquire"]
        self.ring.release = self._saved["release"]
        return False


def kv_digest(cache) -> str:
    """Hash of every array in the KV cache, widened to float32 (exact for bf16)."""
    try:
        import mlx.core as mx
        import numpy as np

        digest = hashlib.sha256()
        for entry in cache:
            state = getattr(entry, "state", None)
            if state is None:
                continue
            for array in state if isinstance(state, (list, tuple)) else (state,):
                if array is None:
                    continue
                wide = array.astype(mx.float32)
                mx.eval(wide)
                digest.update(np.asarray(wide).tobytes())
        return digest.hexdigest()[:16]
    except Exception:
        return "unavailable"


def run_turn(backend, probe, instrument, store, name: str, feed, decode: int,
             hasher=kv_digest) -> dict:
    """Feed one turn, prefill it, decode, and return every measurement."""
    feed()
    pending = len(backend.pending)
    instrument.reset()
    probe.reset_peak()
    start_mem = probe.memory()
    start_pool = pool_state(store)
    vm_before, disk_before = probe.vm(), probe.disk_bytes()
    free_before = probe.free_mb()
    marks, stamps = {}, []

    def on_prefilled():
        marks["time"] = probe.now()
        marks["mem"] = probe.memory()
        marks["vm"], marks["disk"] = probe.vm(), probe.disk_bytes()
        marks["pool"] = pool_state(store)
        marks["rss"] = probe.rss_bytes()
        marks["first_token"] = None
        marks["kv_hash"] = hasher(backend.cache)

    def on_token(value, piece):
        stamps.append(probe.now())
        if marks.get("first_token") is None:
            marks["first_token"] = int(value)

    _text, stats = backend.generate(decode, on_prefilled=on_prefilled, on_decode_token=on_token)
    end_mem = probe.memory()
    ids = list(backend.tape[-stats.tokens:]) if stats.tokens else []
    prefilled = sum(call["tokens"] for call in instrument.calls)
    prefill_s = stats.prefill_seconds
    vm_end, disk_end = probe.vm(), probe.disk_bytes()
    after_pool = pool_state(store)
    return {
        "name": name,
        "pending_tokens": pending,
        "prefilled_tokens": prefilled,
        "prefill_s": prefill_s,
        "prefill_tok_s": prefilled / prefill_s if prefill_s and prefilled else None,
        "prefill_calls": instrument.calls,
        "mem_start": start_mem,
        "mem_end_prefill": marks["mem"],
        "mem_end_decode": end_mem,
        "cache_max": instrument.cache_max,
        "rss_end_prefill": marks.get("rss"),
        "pool_start": start_pool, "pool_end_prefill": marks["pool"], "pool_end": after_pool,
        "pool_hits_decode": (after_pool.get("hits", 0) - marks["pool"].get("hits", 0)),
        "pool_misses_decode": (after_pool.get("misses", 0) - marks["pool"].get("misses", 0)),
        "ring_acquired": instrument.acquired, "ring_reason": instrument.reason,
        "ring": instrument.ring_stats,
        "scratch": {"acquire_s": instrument.acquire_s, "dispose_s": instrument.dispose_s,
                    "acquire_faults": instrument.acquire_faults},
        "first_token": marks.get("first_token"),
        "kv_hash": marks.get("kv_hash"),
        "digest": H.digest_ids(ids[:decode]),
        "generated_tokens": stats.tokens,
        "decode": {
            "tokens": stats.tokens,
            "first_token_s": (stamps[0] - marks["time"]) if stamps and "time" in marks else None,
            "tok_s_1_64": H.window_rate(stamps, 1, 64),
            "tok_s_65_128": H.window_rate(stamps, 65, 128),
            "tok_s_overall": stats.rate,
        },
        "ssd_prefill_bytes": (marks["disk"] - disk_before) if disk_before >= 0 else None,
        "ssd_decode_bytes": (disk_end - marks["disk"]) if disk_end >= 0 else None,
        "vm": H.vm_delta(vm_before, marks["vm"]),
        "vm_decode": H.vm_delta(marks["vm"], vm_end),
        "free_mb_before": free_before,
        "finish": stats.finish,
    }


def run_scenario(backend, probe, instrument, store, cond: H.Condition, prompts: dict,
                 decode: int, environ, flags: dict | None = None,
                 hasher=kv_digest) -> list[dict]:
    """First-turn: apply the condition, measure turn 1. Second-turn: run turn 1
    under the baseline flags as setup, apply the condition, measure turn 2."""
    turns = []
    if cond.turn == "second":
        H.apply_overlay(environ, H.condition_overlay(H.CONDITIONS["S0"], flags))
        turns.append(run_turn(
            backend, probe, instrument, store, "setup",
            lambda: backend.open_conversation(prompts["system"], prompts["first"],
                                              enable_thinking=False),
            decode, hasher))
        H.apply_overlay(environ, H.condition_overlay(cond, flags))
        turns.append(run_turn(
            backend, probe, instrument, store, "measured",
            lambda: backend.append_user(prompts["later"], enable_thinking=False),
            decode, hasher))
    else:
        H.apply_overlay(environ, H.condition_overlay(cond, flags))
        turns.append(run_turn(
            backend, probe, instrument, store, "measured",
            lambda: backend.open_conversation(prompts["system"], prompts["first"],
                                              enable_thinking=False),
            decode, hasher))
    return turns


def child_main(args) -> int:
    cond = H.CONDITIONS[args.condition]
    H.apply_overlay(os.environ, H.launch_overlay(cond))
    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ["FLASHNEXT_GPU_KEEPWARM"] = "0"
    target = output_path("flashnext", "prefill-memory", f"arm-{args.condition}.json", args.json)
    pins = target.parent / "pins"
    pins.mkdir(exist_ok=True)
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, pins / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(pins / "pins.json")

    from macqwen.backends.flashnext import FlashNextBackend
    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext import prefill, prefill_ring
    from models.flashnext.checkpoint_policy import resolve_resident_experts
    from models.flashnext.routing import checkpoint_identity_for_store

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32
    probe = RealProbe()
    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    store = find_store(backend.language)
    prompts = H.build_prompts(backend.tokenizer, SOURCE_TEXT.read_text(), H.SIZES[args.size])
    evidence = {
        "arm_id": args.arm_id or f"{args.size}-{args.condition}", "size": args.size,
        "turn": cond.turn, "condition": args.condition, "round": args.round,
        "reverse": args.reverse, "checkpoint": checkpoint,
        "checkpoint_identity": checkpoint_identity_for_store(store) if store else None,
        "metal_limit_bytes": probe.metal_limit(),
        "prompts": {key: prompts[key] for key in
                    ("source_hash", "first_hash", "later_hash", "first_ids", "later_ids")},
        "after_load": {**probe.memory(), "rss": probe.rss_bytes(),
                       "pool": pool_state(store)},
        "environment": {k: os.environ.get(k) for k in sorted(os.environ)
                        if k.startswith("FLASHNEXT_")},
        "status": "running", "turns": [],
    }
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    with Instrument(probe, prefill, prefill_ring, backend.language) as instrument:
        evidence["turns"] = run_scenario(
            backend, probe, instrument, store, cond, prompts, args.decode, os.environ)
    evidence["status"] = "completed"
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    print(f"wrote {target}", flush=True)
    return 0


# -- orchestrator -------------------------------------------------------------

def child_command(arm: H.Arm, decode: int, result: Path) -> list[str]:
    return [sys.executable, "-m", MODULE, "--child", "--size", arm.size,
            "--condition", arm.condition, "--decode", str(decode),
            "--arm-id", arm.arm_id, "--round", str(arm.round + 1),
            "--json", str(result)] + (["--reverse"] if arm.reverse else [])


def child_environment(base: dict, cond: H.Condition, flags: dict | None = None) -> dict:
    return H.apply_overlay(dict(base), H.launch_overlay(cond, flags))


def describe_schedule(arms: list[H.Arm], decode: int, flags: dict | None = None) -> list[str]:
    lines = []
    for arm in arms:
        cond = H.CONDITIONS[arm.condition]
        overlay = {k: v for k, v in H.launch_overlay(cond, flags).items() if v is not None}
        later = {k: v for k, v in H.condition_overlay(cond, flags).items() if v is not None}
        lines.append(
            f"{arm.index:3d} {arm.arm_id:14s} {'reversed' if arm.reverse else 'forward ':8s} "
            f"turn={cond.turn:6s} launch_flags={overlay or '{}'}"
            + (f" flags_for_turn_2={later or '{}'}" if cond.turn == "second" else "")
        )
    return lines


def fingerprints() -> dict:
    from macqwen.measurement import source_hash

    return {name: source_hash(ROOT / name) for name in FINGERPRINTED if (ROOT / name).is_file()}


def validate_arms(arms: list[dict]) -> dict:
    """Gates G1 to G5 for the arms collected so far."""
    by_id = {(a["size"], a["condition"], a["round"]): a for a in arms}
    report = {"exactness": H.exactness(arms), "arms": {}, "decode": H.decode_regression(arms)}
    for arm in arms:
        cond = H.CONDITIONS[arm["condition"]]
        base = by_id.get((arm["size"], H.BASELINES[cond.turn], arm["round"]))
        report["arms"][arm["arm_id"]] = {
            "G2": H.ring_health(arm, cond),
            "G3": H.memory_gate(arm),
            "G4": H.scratch_cleanup(arm, cond, base),
        }
    return report


def orchestrate(args, sampler=None, popen=subprocess.Popen, sleep=time.sleep) -> int:
    from macqwen.measurement import MeasurementRun

    arms = H.build_schedule(args.sizes, args.conditions, args.rounds)
    sampler = sampler or H.Sampler(vm=_vm_counters, load=lambda: os.getloadavg()[0])
    policy = H.SafetyPolicy(final=args.final)
    start = sampler.sample()
    watchdog = H.Watchdog(policy, start)
    refusal = watchdog.check(start)
    records = output_path("flashnext", "prefill-memory", "records.jsonl", args.record)
    run = MeasurementRun(records, runtime="flashnext", experiment="prefill-memory", metadata={
        "plan": "docs/flashnext/prefetch-integration.md section 14",
        "preliminary": not args.final or start.get("power_source") != "ac",
        "environment_at_start": start, "sizes": args.sizes, "conditions": args.conditions,
        "rounds": args.rounds, "decode_tokens": args.decode,
        "schedule": [a.arm_id for a in arms], "sources": fingerprints(),
        "sampler": "greedy; closed thinking block",
    })
    run.start()
    if refusal:
        run.failure(refusal, error_type="safety_stop", phase="preflight")
        run.finish("stopped", reason=refusal)
        print(f"refusing to start: {refusal}", flush=True)
        return 3
    finished: list[dict] = []
    skipped_sizes: set[str] = set()
    current = None
    for arm in arms:
        if arm.size in skipped_sizes:
            continue
        cond = H.CONDITIONS[arm.condition]
        if current is not None and arm.size != current:
            worst = H.worst_headroom(finished, current)
            if worst is not None and worst < H.NEXT_SIZE_HEADROOM_BYTES:
                reason = (f"{current} left {worst / 1e9:.2f} GB of headroom, "
                          f"need {H.NEXT_SIZE_HEADROOM_BYTES / 1e9:.0f} GB for the next size")
                run.failure(reason, error_type="size_gate", arm_id=arm.arm_id)
                skipped_sizes.update(a.size for a in arms if SIZE_ORDER[a.size] >= SIZE_ORDER[arm.size])
                continue
        current = arm.size
        guarded = (cond.guarded and finished and any(
            v["status"] == "fail" for v in _gate_statuses(validate_arms(finished))))
        if guarded:
            run.failure("an earlier arm failed a gate; the guarded condition is skipped",
                        error_type="guard", arm_id=arm.arm_id)
            continue
        pre = sampler.sample()
        reason = watchdog.check(pre)
        if reason:
            run.failure(reason, error_type="safety_stop", arm_id=arm.arm_id, phase="before-arm")
            run.finish("stopped", reason=reason, completed_arms=len(finished))
            print(f"stopped before {arm.arm_id}: {reason}", flush=True)
            return 3
        result_path = output_path("flashnext", "prefill-memory", f"arm-{arm.arm_id}.json")
        command = child_command(arm, args.decode, result_path)
        outcome = H.run_child(
            command, child_environment(dict(os.environ), cond), sampler=sampler,
            watchdog=watchdog, vm_start=pre.get("vm"), interval=args.interval,
            timeout=H.TIME_LIMIT_SECONDS[arm.size], popen=popen, sleep=sleep)
        post = sampler.sample()
        if outcome.status != "ok":
            run.failure(outcome.message, arm_id=arm.arm_id, error_type=outcome.status,
                        returncode=outcome.returncode, before=pre, after=post)
            if outcome.status in ("stopped", "timeout", "exit_137"):
                run.finish("stopped", reason=outcome.message, completed_arms=len(finished))
                print(f"stopped at {arm.arm_id}: {outcome.message}", flush=True)
                return 3
            continue
        data = json.loads(result_path.read_text())
        data.update(arm_id=arm.arm_id, round=arm.round, reverse=arm.reverse)
        finished.append(data)
        run.arm(arm_id=arm.arm_id, condition=arm.condition, round_index=arm.round + 1,
                command=command, metrics=_arm_metrics(data), size=arm.size,
                turn=cond.turn, reverse=arm.reverse, order_index=arm.index,
                environment_before=pre, environment_after=post,
                samples=len(outcome.samples), result_file=str(result_path.name))
        print(f"{arm.arm_id}: done", flush=True)
        sleep(args.settle)
    report = validate_arms(finished)
    for arm in finished:
        gates = report["arms"][arm["arm_id"]]
        ok = all(g["status"] in ("pass", "n/a") for g in gates.values())
        run.validation(arm["arm_id"], "pass" if ok else "fail",
                       **{k: v for k, v in gates.items()})
    for key, result in report["exactness"].items():
        run.validation(f"exactness-{key[0]}-{key[1]}", result["status"],
                       mismatches=[list(m) for m in result["mismatches"]])
    effects = H.effects_by_group(finished, EFFECT_METRICS)
    run.finish("completed", arms=len(finished),
               effects={"/".join(k): {kk: vv for kk, vv in v.items() if kk != "differences"}
                        for k, v in effects.items()},
               decode={"/".join(k): v for k, v in report["decode"].items()})
    return 0


SIZE_ORDER = {name: index for index, name in enumerate(sorted(H.SIZES, key=H.SIZES.get))}


def _vm_counters() -> dict:
    from models.flashnext.diskio import vm_counters

    return vm_counters()


def _gate_statuses(report: dict):
    for gates in report["arms"].values():
        yield from (g for g in gates.values() if g["status"] != "n/a")
    yield from (v for v in report["exactness"].values())


def _arm_metrics(data: dict) -> dict:
    turn = H.measured_turn(data)
    return {
        "prefilled_tokens": turn["prefilled_tokens"], "prefill_tok_s": turn["prefill_tok_s"],
        "decode_tok_s_1_64": turn["decode"]["tok_s_1_64"],
        "decode_tok_s_65_128": turn["decode"]["tok_s_65_128"],
        "peak_bytes": turn["mem_end_prefill"]["peak"], "digest": turn["digest"],
        "first_token": turn["first_token"], "kv_hash": turn["kv_hash"],
        "ssd_prefill_bytes": turn["ssd_prefill_bytes"], "vm": turn["vm"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", choices=sorted(H.SIZES), default=["1k"])
    parser.add_argument("--conditions", nargs="+", choices=sorted(H.CONDITIONS),
                        default=list(H.CONDITIONS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--decode", type=int, default=H.DECODE_TOKENS)
    parser.add_argument("--interval", type=float, default=5.0, help="watchdog sample seconds")
    parser.add_argument("--settle", type=float, default=30.0, help="pause between arms")
    parser.add_argument("--final", action="store_true",
                        help="require AC power; otherwise records are marked preliminary")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the schedule and flags; start nothing, write nothing")
    parser.add_argument("--record", help="measurement JSONL path (inside results/)")
    parser.add_argument("--json", help="child result path (inside results/)")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--size", choices=sorted(H.SIZES), help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=sorted(H.CONDITIONS), help=argparse.SUPPRESS)
    parser.add_argument("--arm-id", help=argparse.SUPPRESS)
    parser.add_argument("--round", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--reverse", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        if not (args.size and args.condition):
            parser.error("--child needs --size and --condition")
        args.round -= 1
        return child_main(args)
    if args.dry_run:
        arms = H.build_schedule(args.sizes, args.conditions, args.rounds)
        print(f"{len(arms)} fresh-process arms; sizes {args.sizes}; rounds {args.rounds}")
        print("\n".join(describe_schedule(arms, args.decode)))
        print("example command:", " ".join(
            child_command(arms[0], args.decode, Path("<results>/arm.json"))))
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
