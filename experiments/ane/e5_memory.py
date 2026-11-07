"""E5 memory child: RSS and ANE footprint as distinct Core ML router models load, one process."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import common as C

units = sys.argv[1]
rows = [{"step": "start", "rss_MB": C.rss_bytes() / 1e6, "neural_MB": C.neural_footprint()["neural_footprint"] / 1e6}]
import coremltools as ct
rows.append({"step": "import coremltools", "rss_MB": C.rss_bytes() / 1e6,
             "neural_MB": C.neural_footprint()["neural_footprint"] / 1e6})
models = []
arr = {"x": np.random.default_rng(1).standard_normal((256, C.HIDDEN)).astype(np.float16)}
for seed in range(4):
    package = C.OUT / f"router_seed{seed}.mlpackage"
    if not package.exists():
        C.build_model(f"router_seed{seed}", "router", 256, seed=seed)
    began = time.perf_counter()
    m = C.load(package, units)
    m.predict(arr)
    models.append(m)
    rows.append({"step": f"load+first predict #{seed + 1}", "seconds": time.perf_counter() - began,
                 "rss_MB": C.rss_bytes() / 1e6, "neural_MB": C.neural_footprint()["neural_footprint"] / 1e6})
print("RESULT " + json.dumps(rows))
