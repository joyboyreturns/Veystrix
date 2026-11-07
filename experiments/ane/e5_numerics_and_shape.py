"""E5: (a) fp16 ANE top-10 agreement with an fp32 reference (random weights, so indicative
only), (b) matmul throughput at the real hidden size on ANE vs CPU vs MLX GPU."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import common as C

C.guard()
PY, HERE = sys.executable, Path(__file__).parent
report = {}
# (a) numerics: same fp16 weights as build_model(seed=0); fp32 reference math
rng = np.random.default_rng(0)
w16 = (rng.standard_normal((C.EXPERTS, C.HIDDEN)) / np.sqrt(C.HIDDEN)).astype(np.float16)
ref_w = w16.astype(np.float32)
xs = np.random.default_rng(7).standard_normal((2048, C.HIDDEN)).astype(np.float16)
ref = xs.astype(np.float32) @ ref_w.T
ref_top = np.argsort(-ref, axis=1)[:, :C.TOPK]
for units in ("cpu", "ane"):
    model = C.load(C.OUT / "router_b256.mlpackage", units)
    out = np.concatenate([next(iter(model.predict({"x": xs[i:i + 256]}).values()))
                          for i in range(0, 2048, 256)]).astype(np.float32)
    top = np.argsort(-out, axis=1)[:, :C.TOPK]
    same_set = np.mean([set(a) == set(b) for a, b in zip(ref_top, top)])
    overlap = np.mean([len(set(a) & set(b)) / C.TOPK for a, b in zip(ref_top, top)])
    report[f"numerics_{units}"] = {"rows": 2048, "identical_top10_set": float(same_set),
                                   "mean_overlap": float(overlap),
                                   "max_abs_logit_diff": float(np.abs(out - ref).max()),
                                   "mean_abs_logit_diff": float(np.abs(out - ref).mean())}
    print(units, report[f"numerics_{units}"], flush=True)
# (b) throughput at hidden 2560, batch 1024 (prefill chunk), square matmul
name = "matmul_b1024_h2560"
if not (C.OUT / f"{name}.mlpackage").exists():
    C.build_model(name, "matmul", 1024, hidden=2560)
flops = 2 * 1024 * 2560 * 2560
for target, units in ((str(C.OUT / f"{name}.mlpackage"), "cpu"), (str(C.OUT / f"{name}.mlpackage"), "ane"),
                      ("mlx", "-")):
    out = subprocess.run([PY, str(HERE / "e2_child.py"), target, units, "2.0", "1024", "2560"],
                         capture_output=True, text=True, timeout=300)
    row = json.loads(next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))[7:])
    label = "mlx-gpu-fp16" if target == "mlx" else f"coreml-{units}"
    report[f"matmul_1024x2560x2560_{label}"] = {"p50_ms": row["p50_ms"], "tflops": flops / (row["p50_ms"] * 1e-3) / 1e12,
                                                "neural_MB": row["neural_after_load"] / 1e6}
    print(label, report[f"matmul_1024x2560x2560_{label}"], flush=True)
# memory (c): each in a fresh process
for units in ("cpu", "ane"):
    out = subprocess.run([PY, str(HERE / "e5_memory.py"), units], capture_output=True, text=True, timeout=600)
    rows = json.loads(next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))[7:])
    report[f"memory_{units}"] = rows
    for r in rows:
        print(units, r["step"], f"rss {r['rss_MB']:.0f} MB  neural {r['neural_MB']:.2f} MB",
              f"{r.get('seconds', 0):.2f}s" if "seconds" in r else "", flush=True)
(C.OUT / "e5.json").write_text(json.dumps(report, indent=1))
