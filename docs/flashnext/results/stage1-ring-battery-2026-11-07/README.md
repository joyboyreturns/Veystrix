# Stage 1 ring benchmark, battery mode, 2026-11-07

Preliminary. MacBook Air M4 16 GB on **battery power, Low Power Mode off**
(battery 92% to 84% over the session). Not the final plugged-in benchmark.
Synthetic random Q4/G32 tensors with real expert row sizes; MoE layers only.
No model weights. See `docs/flashnext/prefetch-integration.md` section 12.

* `blocked_{1k,2k,4k}` - the reported runs: blocked design (3 same-arm runs
  per block, order reversed each round), 3 rounds, 8 layers per arm, one fresh
  process per size. Only these are quoted as results.
* `alternate_1k_aborted_at_2k` - earlier alternating design, 6 rounds at 1K,
  then stopped by the watchdog on memory pressure level 2 at the start of 2K.
  It shows the order effect that motivated the blocked design.
* `smoke_1k` - 2-round smoke test on a 6.3 GB file. Its 1.55x speed-up is
  inflated: the small file stayed in the page cache. Do not quote it.
* `bench_ring.py` - the harness. Run it from the repository root with the
  project's Python 3.12 environment, for example
  `python docs/flashnext/results/stage1-ring-battery-2026-11-07/bench_ring.py --sizes 2k --rounds 3 --design blocked --block 3 --file-layers 24 --arm-layers 8 --out result.json`.
  `VEYSTRIX_REPO` and `BENCH_SCRATCH` override the repository and scratch
  paths. It writes a scratch checkpoint of about 38 GB with `--file-layers 24`
  and deletes it afterwards. Paths in the saved logs and JSON were replaced
  with `<scratch>` and `<repo>`. Heavy: run it plugged in and read the safety
  stops in its header first.
