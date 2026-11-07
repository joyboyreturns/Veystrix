"""E4 worker: one role in an overlap test, in its own process (no shared GIL).

usage: e4_worker.py <role> <start_epoch> <seconds> [package] [interval_ms]
roles: gpu-compute | gpu-bandwidth | ane | cpu  (ane/cpu need a Core ML package)
All workers wait for the same start time so their loops overlap exactly."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import common as C

role, start, seconds = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
package = sys.argv[4] if len(sys.argv) > 4 else None
interval = float(sys.argv[5]) / 1e3 if len(sys.argv) > 5 else 0.0

if role == "gpu-compute":
    import mlx.core as mx
    w = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    x = mx.random.normal((512, 4096)).astype(mx.bfloat16)
    mx.eval(w, x)

    def call():
        y = x
        for _ in range(4):
            y = (y @ w) * 0.01
        mx.eval(y)
elif role == "gpu-bandwidth":
    import mlx.core as mx
    w = mx.random.normal((2560, 160000)).astype(mx.float16)      # about 819 MB read per call
    x = mx.random.normal((1, 2560)).astype(mx.float16)
    mx.eval(w, x)
    call = lambda: mx.eval(x @ w)
else:
    model = C.load(Path(package), role)
    meta = model.get_spec().description.input[0]
    batch, hidden = list(meta.type.multiArrayType.shape)
    arr = {"x": np.random.default_rng(1).standard_normal((batch, hidden)).astype(np.float16)}
    model.predict(arr)
    call = lambda: model.predict(arr)
for _ in range(5):
    call()
while time.time() < start:
    time.sleep(0.001)
samples, end = [], start + seconds
tick = time.perf_counter()
while time.time() < end:
    t = time.perf_counter(); call(); samples.append(time.perf_counter() - t)
    if interval:
        tick += interval
        pause = tick - time.perf_counter()
        if pause > 0:
            time.sleep(pause)
        else:
            tick = time.perf_counter()
row = {"role": role, "n": len(samples), "calls_per_s": len(samples) / seconds,
       **C.percentiles(samples)}
print("RESULT " + json.dumps(row), flush=True)
