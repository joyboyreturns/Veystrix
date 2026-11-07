"""E3: predictor latency by device, with a placement check for every ANE case.

Core ML cases run in fresh processes (e2_child.py) so the neural footprint is clean;
a footprint of 0 under CPU_AND_NE means Core ML silently ran it elsewhere.
MLX GPU, MLX CPU and numpy (Accelerate) run in this process. Short loops (1 s)."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import coremltools as ct
import mlx.core as mx
import common as C

PY, HERE, SECONDS = sys.executable, Path(__file__).parent, 1.0
MODELS = [("router", 1), ("router3", 3), ("router_topk", 1), ("mlp", 1)]
BATCHES = [1, 16, 256, 1024]


def preferred(package: Path) -> dict:
    compiled = C.compile_path(package)
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(compiled),
                                                               compute_units=ct.ComputeUnit.CPU_AND_NE)
    out = {}
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:
            device = type(usage.preferred_compute_device).__name__.replace("ComputeDevice", "").replace("ML", "")
            out[op.operator_name.split(".")[-1]] = device
    return out


rows = []
for base, look in MODELS:
    kind = base.rstrip("0123456789")
    for batch in BATCHES:
        C.guard()
        name = f"{base}_b{batch}"
        package = C.OUT / f"{name}.mlpackage"
        if not package.exists():
            C.build_model(name, kind, batch, lookahead=look)
        plan = preferred(package)
        for units in ("cpu", "ane"):
            out = subprocess.run([PY, str(HERE / "e2_child.py"), str(package), units, str(SECONDS),
                                  str(batch), str(C.HIDDEN)], capture_output=True, text=True, timeout=300)
            line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line is None:
                rows.append({"model": name, "units": units, "error": out.stderr[-300:]}); continue
            r = json.loads(line[7:])
            rows.append({"model": name, "units": f"coreml-{units}", "batch": batch,
                         "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"], "plan_preferred": plan,
                         "neural_MB": r["neural_after_load"] / 1e6,
                         "rss_MB": (r["rss_after_load"] - r["rss_before"]) / 1e6, "load_s": r["load_s"]})
            print(f"{name:18s} coreml-{units:3s} p50 {r['p50_ms']:8.3f} ms p95 {r['p95_ms']:8.3f}  "
                  f"neural {r['neural_after_load'] / 1e6:6.2f} MB  plan {sorted(set(plan.values()))}", flush=True)

# non-Core-ML references: router (2560 -> 512), same shapes, random weights
rng = np.random.default_rng(0)
for batch in BATCHES:
    w32 = (rng.standard_normal((C.HIDDEN, C.EXPERTS)) / 50).astype(np.float32)
    x32 = rng.standard_normal((batch, C.HIDDEN)).astype(np.float32)
    rows.append({"model": f"router_b{batch}", "units": "numpy-fp32-accelerate", "batch": batch,
                 **C.time_calls(lambda: np.argpartition(x32 @ w32, -C.TOPK, axis=-1), 300)})
    for dtype, tag in ((mx.bfloat16, "bf16"), (mx.float16, "fp16")):
        wm, xm = mx.array(w32).astype(dtype), mx.array(x32).astype(dtype)
        mx.eval(wm, xm)
        for device, dname in ((mx.gpu, "gpu"), (mx.cpu, "cpu")):
            if dname == "cpu" and tag == "fp16":
                continue
            with mx.stream(device):
                rows.append({"model": f"router_b{batch}", "units": f"mlx-{dname}-{tag}", "batch": batch,
                             **C.time_calls(lambda: mx.eval(mx.argpartition(-(xm @ wm), C.TOPK, axis=-1)), 300)})
for r in rows:
    if str(r["units"]).startswith(("numpy", "mlx")) and r["model"].endswith(("b1", "b256")):
        print(f"{r['model']:18s} {r['units']:24s} p50 {r['p50_ms']:8.3f} ms  p95 {r['p95_ms']:8.3f}")
(C.OUT / "e3_latency.json").write_text(json.dumps(rows, indent=1))
