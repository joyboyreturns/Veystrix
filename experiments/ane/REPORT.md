# Apple Neural Engine for Flash-Next: expert-route prediction and other uses

**Status: evaluated and rejected (2026-11-07). ANE development is stopped.** Reasons, all measured or
recorded below: batch-1 predictors never reach the ANE; the predictor is not the bottleneck (about 0.1% of
a 1K prefill); ANE traffic slows bandwidth-bound GPU work by 3% to 17%; the ANE is not bit-exact for routing;
dense offload needs duplicate fp16 weights that do not fit. Known limits: random weights (no accuracy
claim), battery-mode preliminary runs, short loops, Python Core ML call path, no hardware energy counter.
Raw results are in `results/` (paths sanitised). `out/` holds generated models and is git-ignored.

Branch `development3.8`. Experimental code in `experiments/ane/`, **untracked and uncommitted**.
No production file was changed, no model weights were downloaded or read, Q4 is untouched, `main` is untouched.
Tags: **[measured]** here; **[recorded]** from `docs/flashnext/research.md`; **[estimate]** arithmetic, not a result.

## 0. Conditions and limits

**Correction, 2026-11-07:** an earlier version of this report extrapolated over 40 layers (56 ms, 270 MB). The checkpoint's config has 48 layers, so those two figures are 68 ms and 310 MB. The conclusion is unchanged.

* M4 MacBook Air, 16 GB, macOS 27.2, Core ML via coremltools 9.0, MLX 0.32.2, in an isolated venv (the repo `.venv` is untouched).
* **On battery (75% to 68%), Low Power Mode off, no thermal warning.** These are preliminary numbers, not plugged-in numbers. A battery guard (stop below 25%) ran before every case.
* Every predictor uses **random weights**. Latency, placement and memory do not depend on weight values. **Predictor accuracy was not measured**, and nothing below claims it.
* Core ML is called through the Python `coremltools` path, so per-call figures include its marshalling. A native Swift or Objective-C caller may be cheaper. Not measured.
* Short loops (1 to 4 s per case). Cases that compare devices ran in fresh processes. The overlap test ran 2 rounds in reversed order and the rounds agree to within 0.02 ms.

## 1. Does the ANE actually execute? (task 3)

The signals available without root, and what each showed:

| Signal | What it shows | Result |
|---|---|---|
| `MLComputePlan` (static) | device Core ML plans for each op | at batch 1 every op is planned on **CPU** though the ANE is "supported"; at batch 16 and up, linear and silu are planned on the ANE; `topk`'s cast op is CPU-only |
| ANE driver `DevicePowerState`, polled about 20 Hz in the parent while a fresh child loops | whether the ANE is powered | ANE-loaded models: up **100%** of the loop. CPU_ONLY, CPU_AND_GPU, MLX GPU and the batch-1 model under CPU_AND_NE: **0%** |
| `proc_pid_rusage` v6 `ri_neural_footprint` | ANE memory owned by the process | 2.7 to 18.4 MB for ANE-loaded models; **0** for CPU_ONLY, GPU, MLX, and batch-1 CPU_AND_NE |
| Timing | device signature | ANE and CPU/GPU rates differ at large shapes (section 2) |

**Verified:** batch >= 16 linear predictors loaded under `CPU_AND_NE` are placed on, and wake, the Neural Engine. **Verified not on the ANE:** the batch-1 (decode-size) predictor, which silently stays on the CPU under `CPU_AND_NE` (35 us, 0 MB neural footprint).

**Not obtained:** energy counters. IOReport "Energy Model" channels read zero for CPU, GPU and ANE even under a CPU burn, in the sandbox and (with your one-time approval of that exact call) outside it. `powermetrics` needs sudo, so I did not run it; `verify_with_powermetrics.sh` is ready if you want hardware power confirmation. The signals above show the model is loaded on the ANE and the ANE is awake while it runs; they do not prove each individual op executed there, and the `topk` model is a visible CPU/ANE split.

## 2. Latency, ANE vs CPU vs MLX GPU (task 4, 5)

p50 milliseconds per call, router 2560 -> 512 experts (top-10 of 512 as in Qwen3.8-Flash-Next). **[measured]**

| Batch (tokens) | Core ML ANE | Core ML CPU | MLX GPU (bf16) | MLX CPU | numpy fp32 |
|---|---|---|---|---|---|
| 1 (decode) | 0.035 (**ran on CPU**) | 0.036 | 0.22 | 0.019 | 0.010 |
| 16 | 0.173 | 0.072 | 0.20 | 0.080 | 0.041 |
| 256 | **0.409** | 0.421 | 0.49 | 1.28 | 0.99 |
| 1024 | **1.41** | 1.61 | 1.37 | 6.47 | 4.98 |

MLX and numpy rows include a top-10 `argpartition`; the Core ML router rows do not, so only ANE-vs-Core-ML-CPU is like for like.

Larger predictors (Core ML, ANE vs Core ML CPU): look-ahead over 3 layers (2560 -> 1536) 0.61 vs 0.86 ms at batch 256 and 2.22 vs 3.52 ms at 1024; MLP 2560 -> 1024 -> 512 0.50 vs 0.70 and 1.76 vs 2.90. So the ANE is **1.15x to 1.6x faster than Core ML on the CPU only at batch >= 256 and with more than a bare router**. A router with `topk` inside is **slower** on the ANE (1.09 vs 0.53 ms at 256) because the plan splits CPU/ANE.

Fixed costs and transfer **[measured, not separable]**: ANE dispatch costs about 0.15 ms regardless of size (batch 16 is 0.173 ms for 42 MFLOP); Core ML on the CPU about 0.035 ms; an MLX GPU eval about 0.2 ms. A paced ANE predictor (one call every 5 ms, about a layer's cadence) took **1.64 ms**, four times its flat-out 0.41 ms, because the ANE clocks down between calls. Moving a live hidden state from MLX into Core ML also needs a GPU sync and a copy; that was not measured on a live model (**[recorded]** the extra per-layer eval is free in decode: 498.5 vs 499.3 ms).

Throughput at large fp16 matmuls (includes marshalling): 1024x2560x2560 ANE **4.2 TFLOPS**, MLX GPU 3.1, Core ML CPU 2.5; 512x2048x2048 ANE 3.5, MLX GPU 2.8, CPU 2.2.

## 3. Memory (task 5)

* Core ML runtime import: +59 MB RSS (31 -> 91 MB). **[measured]**
* One ANE router model: +25 MB RSS for the first, about +2 MB for each further one, **2.72 MB** ANE footprint each (7.96 to 18.4 MB for larger models). The same model on CPU: +45 MB first, then about +4 to 10 MB each. **[measured]**
* **[estimate]** 48 per-layer routers on the ANE (the checkpoint has 48 layers, all with routed experts): about 130 MB ANE footprint (48 x 2.72 MB) plus about 120 MB RSS plus the 59 MB runtime, roughly **310 MB**. The recorded prefill left about 100 MB free and the guard headroom is about 0.6 GB (`docs/flashnext/prefetch-integration.md` sections 5 and 12.7).

## 4. Overlap with GPU work (task 5)

Separate processes (no shared GIL) start together; times are the GPU worker's p50, mean of two reversed rounds. **[measured]**

| Side load | GPU compute-bound (21.56 ms alone) | GPU bandwidth-bound, decode-like (8.13 ms alone) |
|---|---|---|
| ANE predictor, 200 Hz | +0.3% | **+3.5%** |
| ANE predictor, flat out | -0.2% | **+17.6%** |
| ANE saturating matmul | +0.5% | **+14.9%** |
| CPU Core ML predictor, flat out | +0.2% | +2.2% |

* GPU and ANE **can** run at the same time, and compute-bound GPU work does not notice. Bandwidth-bound GPU work (the shape of decode) **does** slow, and the ANE costs it more than a CPU predictor does. This matches the **[recorded]** finding that memory-controller contention, not compute, limits overlap on this machine.
* The ANE slows under GPU load too: the saturating matmul took 1.22 ms alone (two repeats) and 2.8 to 3.0 ms beside either GPU workload, 2.3x to 2.5x slower. The small predictor moved less (0.407 ms alone, 0.425 to 0.45 beside the GPU).

## 5. Could ANE prediction improve SSD expert prefetch without changing routing? (task 6)

**Routing exactness.** Used only to predict, the ANE cannot change what Qwen routes: the real router still decides, a wrong prediction only wastes a read (design in `prefetch-design.md`). **It must not make the routing decision.** Against an fp32 reference on 2,048 random rows with the same fp16 weights, the ANE's top-10 set was identical on 98.8% of rows (CPU fp16 on 89.9%): not exact, so it would change routing. Random weights make the rate indicative only.

**Why it does not help:**

1. **Decode:** the predictor is not the bottleneck and the ANE does not even run it (batch 1 stays on the CPU, 10 to 35 us). **[recorded]** a lead-1 predictor recovers 72% of top-10 (4.2 wasted reads per layer); a *perfect* predictor with overlapped reads tied the baseline (+0.1%, -2.9%, -0.7%); early submit lost 8.2%; the cause is memory-controller contention. Section 4 shows ANE traffic adds to that contention rather than avoiding it.
2. **Prefill, 2048 tokens and up:** about 400 of 512 experts are routed per layer **[recorded]**, so everything is staged and no prediction is needed.
3. **Prefill, 1K:** about 334 of 512 experts per layer. **[estimate]** the 1024-token router costs 1.4 ms per layer, about 68 ms over 48 layers, 0.14% of the recorded 49.3 s prefill. The cost is reads (25.6 s) and GPU drain (14.9 s), not prediction, and Stage 1 deliberately does not overlap reads with GPU work.
4. Cost of trying: about 310 MB **[estimate]**, a private framework dependency, and a GPU-to-ANE copy per layer.

## 6. Other operations on the ANE (task 7)

* **Dense prefill matmuls:** the ANE is 1.2x to 1.4x faster per FLOP **[measured]**, but Core ML needs its own fp16 copy of the weights (the 3.4 GB dense core cannot be duplicated here), fp16 differs from bf16 (not exact), and the experts are Q4 `gather_qmm`, which Core ML cannot use. **[estimate]** even an ideal, exact, free-memory split of the 14.9 s GPU drain would save at most about 5 s (11% of prefill, with the measured 2.4x concurrent ANE slowdown) to 9 s (17%, ignoring that slowdown), and the real figure is smaller because most of that time is the MoE.
* **Routing itself:** not exact (above).
* **Draft model for speculation:** not tested. **[recorded]** speculative decoding was break-even, and the ANE slows bandwidth-bound GPU work.
* Nothing found that is both useful and low in memory.

## 7. Recommendation

1. **Do not build an ANE expert-route predictor or ANE prefetch for Flash-Next.** At decode it does not run on the ANE and the predictor is already 10 to 35 us; at prefill its cost is 0.1% of the time; and ANE traffic slows bandwidth-bound GPU work by 3% to 17%.
2. **Do not use the ANE for any computation that feeds the model's output** (routing, dense layers): it is not bit-exact, needs duplicate fp16 weights, and the gain is bounded by the I/O-bound prefill.
3. Keep this as a closed route in the research log, alongside the memory-controller contention finding.
4. **Optional, only if you want to keep the question open:** run `sudo experiments/ane/verify_with_powermetrics.sh <python> ane` once for hardware ANE power confirmation, and repeat the timing runs plugged in. I do not expect either to change the recommendation.

## 8. Reproduce

`experiments/ane/`: `e1_placement.py` (compute plan), `e2_verify.py` (placement evidence and speed), `e3_latency.py`, `e4_overlap.py` and `e4b_alone.py`, `e5_numerics_and_shape.py` and `e5_memory.py`, run in that order (later scripts reuse models the earlier ones build into the git-ignored `out/`); raw JSON in `results/` (sanitised copies of the `out/*.json` files). Isolated venv: `python3.12 -m venv V; V/bin/pip install coremltools mlx==0.32.2 numpy`.
