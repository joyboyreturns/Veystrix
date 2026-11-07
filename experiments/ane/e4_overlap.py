"""E4: can the ANE run beside GPU work without slowing it, and does GPU load slow the ANE?

GPU workloads: compute-bound (chained bf16 matmuls) and bandwidth-bound (a matvec over
819 MB, like decode). Side loads: none, ANE predictor paced at 200 Hz, ANE predictor flat
out, ANE saturating matmul, CPU Core ML predictor flat out. Each pairing runs in separate
processes that start together; order is reversed on odd rounds; 2 rounds; 4 s windows."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C

PY, HERE, WINDOW, ROUNDS = sys.executable, Path(__file__).parent, 4.0, 2
ROUTER = str(C.OUT / "router_b256.mlpackage")
BIG = str(C.OUT / "matmul_b512_h2048.mlpackage")
SIDES = [  # label, role, package, interval_ms
    ("none", None, None, 0),
    ("ANE predictor 200 Hz", "ane", ROUTER, 5),
    ("ANE predictor flat out", "ane", ROUTER, 0),
    ("ANE saturating matmul", "ane", BIG, 0),
    ("CPU Core ML predictor flat out", "cpu", ROUTER, 0),
]
GPUS = ["gpu-compute", "gpu-bandwidth"]


def spawn(role, start, package=None, interval=0):
    cmd = [PY, str(HERE / "e4_worker.py"), role, str(start), str(WINDOW)]
    if package:
        cmd += [package, str(interval)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def result(proc):
    out, _ = proc.communicate(timeout=300)
    line = next(l for l in out.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


rows = []
for gpu in GPUS:
    for round_index in range(ROUNDS):
        order = SIDES if round_index % 2 == 0 else list(reversed(SIDES))
        for label, role, package, interval in order:
            battery = C.guard()
            time.sleep(6)                                  # let ANE power and clocks settle
            start = time.time() + 12                       # workers need time to load first
            procs = {"gpu": spawn(gpu, start)}
            if role:
                procs["side"] = spawn(role, start, package, interval)
            got = {k: result(p) for k, p in procs.items()}
            row = {"gpu_workload": gpu, "side": label, "round": round_index, "gpu": got["gpu"],
                   "side_result": got.get("side"), "battery": battery}
            rows.append(row)
            side = got.get("side")
            print(f"{gpu:14s} r{round_index} {label:32s} gpu p50 {got['gpu']['p50_ms']:7.3f} ms "
                  f"({got['gpu']['calls_per_s']:6.1f}/s)"
                  + (f"   side p50 {side['p50_ms']:.3f} ms ({side['calls_per_s']:.0f}/s)" if side else ""),
                  flush=True)
(C.OUT / "e4_overlap.json").write_text(json.dumps(rows, indent=1))
