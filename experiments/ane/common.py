"""Shared helpers for the isolated ANE experiments (synthetic tensors only).

Nothing here touches the production inference path, a checkpoint or Q4 weights.
The predictor weights are random: latency, memory and placement do not depend on
their values; predictor *accuracy* cannot be measured with them and is not claimed.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

HIDDEN, EXPERTS, TOPK = 2560, 512, 10     # Qwen3.8-Flash-Next routing shape (docs, code)
OUT = Path(__file__).resolve().parent / "out"


def build_model(name: str, kind: str, batch: int, lookahead: int = 1, seed: int = 0,
                hidden: int = HIDDEN, experts: int = EXPERTS, dtype="fp16"):
    """Core ML program for one synthetic predictor, converted without torch.

    kind: ``router`` (hidden to expert logits), ``router_topk`` (plus top-10),
    ``mlp`` (2560 to 1024 SiLU to 512), ``trivial`` (one tiny op, fixed-cost probe),
    ``matmul`` (square fp16 matmul for throughput; ``hidden`` is the side).
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    rng = np.random.default_rng(seed)
    np_dtype = np.float16 if dtype == "fp16" else np.float32
    mil_dtype = types.fp16 if dtype == "fp16" else types.fp32
    shape = (batch, hidden)

    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=mil_dtype)],
                opset_version=ct.target.iOS17)
    def program(x):
        if kind == "trivial":
            return mb.mul(x=x, y=np.float16(1.0) if dtype == "fp16" else np.float32(1.0))
        if kind == "matmul":
            w = (rng.standard_normal((hidden, hidden)) / np.sqrt(hidden)).astype(np_dtype)
            return mb.matmul(x=x, y=w)
        if kind == "mlp":
            w1 = (rng.standard_normal((1024, hidden)) / np.sqrt(hidden)).astype(np_dtype)
            w2 = (rng.standard_normal((experts, 1024)) / 32).astype(np_dtype)
            return mb.linear(x=mb.silu(x=mb.linear(x=x, weight=w1)), weight=w2)
        w = (rng.standard_normal((experts * lookahead, hidden)) / np.sqrt(hidden)).astype(np_dtype)
        logits = mb.linear(x=x, weight=w)
        if kind == "router":
            return logits
        values, indices = mb.topk(x=logits, k=TOPK, axis=-1)
        return values, indices

    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                       compute_precision=ct.precision.FLOAT16 if dtype == "fp16"
                       else ct.precision.FLOAT32)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.mlpackage"
    model.save(str(path))
    return path


def compile_path(package: Path) -> Path:
    """Compiled .mlmodelc for an .mlpackage (needed by MLComputePlan)."""
    import coremltools as ct

    return Path(ct.utils.compile_model(str(package)))


def load(package: Path, units: str):
    import coremltools as ct

    chosen = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU,
              "ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL}[units]
    return ct.models.MLModel(str(package), compute_units=chosen)


# -- evidence that the ANE ran something -------------------------------------------

_LIBSYSTEM = ctypes.CDLL("/usr/lib/libSystem.dylib")
_V6_FIELDS = ("neural_footprint", "lifetime_max_neural_footprint", "interval_max_neural_footprint")
_NEURAL_OFFSET = 16 + 8 * 44          # ri_uuid, then 44 uint64 fields before ri_neural_footprint (SDK header)


def neural_footprint() -> dict | None:
    """Per-process ANE memory from ``proc_pid_rusage`` (rusage_info_v6), bytes."""
    buffer = (ctypes.c_uint8 * 1024)()
    if _LIBSYSTEM.proc_pid_rusage(os.getpid(), 6, ctypes.byref(buffer)) != 0:
        return None
    raw = bytes(buffer)
    return {name: int.from_bytes(raw[_NEURAL_OFFSET + 8 * i:_NEURAL_OFFSET + 8 * i + 8], "little")
            for i, name in enumerate(_V6_FIELDS)}


def ane_power_state() -> int | None:
    """``DevicePowerState`` of the ANE driver node (0 = powered down, 1 = up)."""
    try:
        out = subprocess.run(["ioreg", "-r", "-c", "H11ANEIn", "-d", "1"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    marker = '"DevicePowerState"='
    index = out.find(marker)
    return int(out[index + len(marker)]) if index >= 0 else None


def rss_bytes() -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout
    return int(out.strip()) * 1024


# -- timing -----------------------------------------------------------------------------

def percentiles(values: list[float]) -> dict:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"n": len(values), "p50_ms": pick(0.5) * 1e3, "p95_ms": pick(0.95) * 1e3,
            "min_ms": ordered[0] * 1e3, "mean_ms": statistics.fmean(values) * 1e3}


def time_calls(call, count: int, warmup: int = 5) -> dict:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(count):
        began = time.perf_counter()
        call()
        samples.append(time.perf_counter() - began)
    return percentiles(samples)


def battery() -> dict:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    import re

    m = re.search(r"(\d+)%", out)
    return {"percent": int(m.group(1)) if m else None, "on_ac": "AC Power" in out}


def guard(min_percent: int = 25):
    state = battery()
    if not state["on_ac"] and state["percent"] is not None and state["percent"] < min_percent:
        raise SystemExit(f"battery {state['percent']}% is below {min_percent}%: stopping")
    return state
