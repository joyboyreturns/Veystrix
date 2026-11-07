"""E1: build synthetic predictors and ask Core ML where each op is planned to run,
then check whether runtime signals (ANE memory footprint, ANE power state) move."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import coremltools as ct
import common as C

C.guard()
VARIANTS = [("trivial", 1), ("router", 1), ("router_topk", 1), ("mlp", 1), ("router", 256)]
report = {"battery": C.battery(), "coremltools": ct.__version__, "variants": []}
for kind, batch in VARIANTS:
    name = f"{kind}_b{batch}"
    hidden = 16 if kind == "trivial" else C.HIDDEN
    package = C.build_model(name, kind, batch, hidden=hidden)
    compiled = C.compile_path(package)
    entry = {"name": name, "ops": {}}
    try:
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
        program = plan.model_structure.program
        main = program.functions["main"]
        for op in main.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:       # const ops carry no device
                continue
            entry["ops"][f"{op.operator_name}:{op.outputs[0].name}"] = {
                "preferred": type(usage.preferred_compute_device).__name__,
                "supported": [type(d).__name__ for d in usage.supported_compute_devices],
            }
    except Exception as error:      # record, never hide
        entry["plan_error"] = repr(error)
    report["variants"].append(entry)
    print(name, json.dumps(entry["ops"], indent=None)[:400], entry.get("plan_error", ""), flush=True)
(C.OUT / "e1_plan.json").write_text(json.dumps(report, indent=1))
