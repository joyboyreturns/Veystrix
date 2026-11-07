"""One fresh-process case for E2/E3. Prints one JSON line.

usage: e2_child.py <package|mlx> <units cpu|gpu|ane|all|-> <seconds> [batch] [hidden]
Loop window start/end are epoch seconds so the parent can line up its ANE polls.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import common as C

target, units, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1
hidden = int(sys.argv[5]) if len(sys.argv) > 5 else C.HIDDEN
row = {"target": target, "units": units, "batch": batch, "hidden": hidden,
       "neural_before": C.neural_footprint()["neural_footprint"], "rss_before": C.rss_bytes()}
if target == "mlx":
    import mlx.core as mx
    w = mx.random.normal((hidden, hidden)).astype(mx.float16)
    v = mx.random.normal((batch, hidden)).astype(mx.float16)
    mx.eval(w, v)
    call = lambda: mx.eval(v @ w)
    row["load_s"] = 0.0
else:
    began = time.perf_counter()
    model = C.load(Path(target), units)
    arr = {"x": np.random.default_rng(1).standard_normal((batch, hidden)).astype(np.float16)}
    model.predict(arr)                    # first call compiles and loads for the units
    row["load_s"] = time.perf_counter() - began
    call = lambda: model.predict(arr)
row["neural_after_load"] = C.neural_footprint()["neural_footprint"]
row["rss_after_load"] = C.rss_bytes()
for _ in range(5):
    call()
samples, start = [], time.time()
end = time.perf_counter() + seconds
while time.perf_counter() < end:
    t = time.perf_counter(); call(); samples.append(time.perf_counter() - t)
row.update(C.percentiles(samples))
row.update(loop_start=start, loop_end=time.time(),
           neural_after_run=C.neural_footprint()["neural_footprint"],
           rss_after_run=C.rss_bytes())
print("RESULT " + json.dumps(row), flush=True)
