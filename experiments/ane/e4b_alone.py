"""E4b: the E4 side loads with no GPU work, same worker and harness (control for E4)."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C

PY, HERE, WINDOW = sys.executable, Path(__file__).parent, 4.0
ROUTER, BIG = str(C.OUT / "router_b256.mlpackage"), str(C.OUT / "matmul_b512_h2048.mlpackage")
rows = []
for label, role, package, interval in [("ANE predictor 200 Hz", "ane", ROUTER, 5),
                                       ("ANE predictor flat out", "ane", ROUTER, 0),
                                       ("ANE saturating matmul", "ane", BIG, 0),
                                       ("CPU Core ML predictor flat out", "cpu", ROUTER, 0)]:
    for repeat in range(2):
        C.guard(); time.sleep(6)
        start = time.time() + 8
        p = subprocess.Popen([PY, str(HERE / "e4_worker.py"), role, str(start), str(WINDOW), package, str(interval)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        out, _ = p.communicate(timeout=300)
        r = json.loads(next(l for l in out.splitlines() if l.startswith("RESULT "))[7:])
        rows.append({"side": label, "repeat": repeat, **r})
        print(f"{label:32s} alone p50 {r['p50_ms']:.3f} ms  ({r['calls_per_s']:.0f}/s)", flush=True)
(C.OUT / "e4b_alone.json").write_text(json.dumps(rows, indent=1))
