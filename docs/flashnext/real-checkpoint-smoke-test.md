# Real checkpoint: preflight, launch command and the first safe smoke test

Status: written 2026-11-07 while the checkpoint was downloading. Nothing here has been run
against the real model. Tags: **[verified]** checked in this work from metadata or code;
**[recorded]** from earlier project results; **[estimate]** arithmetic, not a measurement;
**[not yet]** needs the finished download.

## 1. The checkpoint

| | |
|---|---|
| Repository | `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP`, revision `2b170fa6309d5d1ee380b35636075fac7945f286` |
| Same as | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` **[verified]**: identical commit, the same 35 files, identical SHA-256 on all 23 LFS files and identical git blobs on the other 12 |
| Directory | `~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP` (the name the alias `vontra-mtp` resolves to **[verified]**) |
| Size | 113.23 GB in 22 safetensors shards, 3,747 tensors **[verified]** |
| Architecture | `qwen4_exp`; hidden 2560, 48 layers (36 linear attention, 12 full attention), 512 routed experts, top-10, expert width 640, one shared expert, vocab 248,320, one MTP layer **[verified]** |
| Quantization | uniform 4-bit, group 32, affine, no per-module overrides **[verified]** |
| Routers | `mlp.gate` is a plain weight (no scales or biases) in all 48 layers and the MTP layer **[verified]**; this is what keeps routing exact |
| One expert | 3,072,000 bytes from the config; the slab layout stride is the same **[verified]** |

No compatibility problem was found in the metadata. Two observations, neither a blocker: the config says
`ple_layer_ids: [2]` while the PLE tensors sit under layer 1 (the runtime's loader handles this checkpoint; the
checker does not assert it), and 333 vision-tower tensors are present but unused (`keep_vision=False`).

**Resident-experts policy.** The runtime applies 8 resident experts only when the checkpoint's content identity
(a hash of the index, the config and every shard's name, size and header) equals `VONTRA_4BIT_MTP_IDENTITY`. A
bit-identical download matches. Any other copy silently runs with the generic default of 32, which changes memory
and speed. Check it once the download has finished (section 3).

## 2. The preflight checker

`python -m models.flashnext.preflight [PATH|ALIAS]` (module `models/flashnext/preflight.py`).

It reads `config.json`, the safetensors index, the tokenizer files and the directory listing, and calls `stat` on
the shards. **By default it never opens, reads or hashes a weight shard**, so it is safe while a download is
writing them. It does not import MLX and is not imported by the runtime.

What it checks: required files and chat template; model type; the exact Q4 settings (4-bit, group 32, affine,
uniform); the architecture profile above; layer types; the expert record size against the slab layout; shard
names; every backbone and expert tensor named in the index (2,941), every scales/biases tensor having its weight,
routers unquantized, the MTP block whole when present (and the loader's both-sidecar-and-indexed refusal);
shards present at the manifest's size, partial and oversize files, `.incomplete` files in
`.cache/huggingface/download` (mapped to their shard by the SHA-256 in the file name), and free disk against what
is left to download. The manifest is `models/flashnext/preflight_manifests/qwen3.8-flash-next-mlx-4bit-mtp.json`
(sizes and hashes from the Hub for the revision above).

Verdicts and exit status: `READY` 0, `INCOMPLETE` 1 (a download problem: missing, partial or still-downloading
files, including a missing tokenizer), `INCOMPATIBLE` 2 (wrong metadata, other quantization, a quantized router,
missing tensors), `MISSING` 2. `--json` prints the report as JSON; `--no-profile` checks structure only.

Two opt-in checks read **only shard headers** (a few hundred KB per shard) and refuse to run while the download is
incomplete:

* `--headers`: header size against file size, index against headers, uint32-packed 4-bit layout with group 32,
  router dtype, expert tensor shape.
* `--identity`: the content identity and whether the resident-experts policy will apply.

Run against the real metadata (22 shards still missing) every metadata check passed and the verdict was
`INCOMPLETE`, with all 22 shards named as missing **[verified]**.

## 3. After the download finishes

```
cd <repo root>
.venv/bin/python -m models.flashnext.preflight vontra-mtp --headers --identity
```

Expected: `VERDICT: READY`, `content identity ... the resident-experts policy (8) applies` **[not yet]**. If the
identity line says `WARN ... will NOT apply`, stop: the files are not the validated checkpoint. For a byte-level
check, compare each shard's SHA-256 with the `sha256` values in the manifest (this reads the whole 113 GB once, so
run it with nothing else active).

## 4. Launch command (interactive chat, not a benchmark)

```
./chat.sh --model flashnext --checkpoint vontra-mtp
```

(`README.md`; `vontra-mtp` resolves to `~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`. A full path to the directory
also works.) The chat defaults come from `models/flashnext/settings/launch.py` (`CHAT_ENV`), including the 6 GB
expert pool. The Stage 1 ring, scratch pool and cache cap are off unless their `FLASHNEXT_PREFILL_*` variables are
set; the production path is unchanged. Do not run it as a smoke test: it starts inference.

## 5. First safe real-model smoke test: F0 versus F2 at 1K, one round

Purpose: show the harness, the ring and the scratch pool run end to end on the real model and that the outputs
match. **It is not a speed measurement.** Each arm has n = 1, and the first arm reads the checkpoint from SSD
while the second finds part of it in the page cache, so the order alone biases the timing (a page-cache effect
inflated a smoke ratio to 1.55x in section 12).

### 5.1 Before you start

* Download complete; `preflight --headers --identity` is `READY` and the identity matches.
* **Plug the Mac in and turn Low Power Mode off** for numbers you will keep. On battery the run is allowed but
  every record is marked preliminary. Low Power Mode on always stops the run.
* Quit browsers and other large apps. The `--check-ready` step warns below 3 GB free or inactive.
* No other benchmark or download running.

### 5.2 Commands (in order; none of the first three loads the model)

```
.venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory --check-ready \
    --checkpoint vontra-mtp --sizes 1k --conditions F0 F2 --rounds 1
.venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory --dry-run \
    --sizes 1k --conditions F0 F2 --rounds 1
```

`--check-ready` verifies, without starting anything: the checkpoint preflight is `READY`; the child would resolve
the same directory; the production API the harness wraps still exists (a rename would otherwise fail after the
model had loaded); the real tokenizer gives prompts near the target size (measured: turn 1 prefills 1,025 / 2,049
/ 4,097 tokens and turn 2 1,024 / 2,048 / 4,096 for targets 1K / 2K / 4K **[verified]** with the checkpoint's
tokenizer; the harness constants were corrected from 64 and 16 to 36 and 23 to get there); the machine state
passes the watchdog rules; free memory; shell variables that would change the baseline. It exits 0 when `READY`.

Then, only with approval:

```
.venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory \
    --checkpoint vontra-mtp --sizes 1k --conditions F0 F2 --rounds 1 --final
```

Drop `--final` on battery. Two fresh child processes run: F0 (no pool exists during prefill, today's behaviour),
then F2 (`FLASHNEXT_PREFILL_RING_EXPERTS=512`, `FLASHNEXT_PREFILL_RING_SCRATCH=1`). The orchestrator samples power,
thermal, swap and memory pressure every 5 s and before each arm, kills the child on any stop rule, and writes the
lifecycle to `results/flashnext/<run>/records.jsonl` with one `arm-*.json` per child.

**[estimate]** About 5 to 10 minutes: load, a 1,025-token prefill (49.3 s was recorded for 1,007 tokens), 128
decode tokens, and a 30 s pause. MLX peak for F0 is about 6.1 GB **[recorded]**; F2 about 6.7 GB **[estimate]**.

### 5.3 What to read afterwards

| Check | Passing looks like |
|---|---|
| Exactness (G1) | `validation exactness-1k-first` is `pass`: same first token, KV-cache hash and 128-token digest in F0 and F2 |
| Ring health (G2) | F2 `ring_acquired` true, no `fallback_*` counts, no `last_error` |
| Memory (G3) | `memory_gate` pass, headroom at least 0.5 GB under the 12.713 GB Metal limit |
| Scratch cleanup (G4) | no pool at the end of F2's prefill, nothing retained, active memory near F0's |
| No stops | no `safety_stop` failure; no swap-in or swap-out |

**Cold-cache imbalance.** F0 runs first and reads the checkpoint cold from SSD; F2 then finds part of it in the
page cache. F2 can look faster purely from cache warming. The run record says so: `speed_evidence` is `false` for
fewer than 3 rounds, and `--check-ready` and the run itself print the warning. Treat any F0-versus-F2 timing from
this test as uninterpretable.

A `pass` here means the outputs match on one prompt and the mechanisms ran. It supports going on to 1K with 3
rounds in reversed order; it supports no speed claim.

### 5.4 If it stops

Exit status 3 with the partial record kept. The stop reasons are the watchdog's: battery below 25% on battery, Low
Power Mode, memory-pressure level 2 or higher, swap activity (swap-in or swap-out, or 64 MB of growth), thermal
throttling in two samples, exit 137, or a time limit. **Pressure level 2 stopped the earlier battery benchmark;
if the real model reaches it without swapping, report it and decide together before relaxing the rule.** Do not
re-run in a loop on fanless hardware.

## 6. Not yet verified

The finished download (shard hashes, headers, identity); anything on the real model: prefill and decode speed,
memory, the ring and scratch pool, exact outputs; whether pressure level 2 is reached in a normal run. The checker
and the readiness mode are tested with synthetic metadata and fakes only
(`models/flashnext/tests/unit/test_preflight.py`, `test_bench_prefill_memory.py`).

## 7. First real-model attempts (2026-11-07) and the memory diagnostic

**Facts [measured, from `results/flashnext/20261107-120921-...` and `...-121249-...`].** Both attempts were the 1K F0
arm on AC power, Low Power Mode off, pressure level 1, 7 to 9 GB free or inactive, with the production 6 GB decode pool.

| Attempt | Stopped | Why |
|---|---|---|
| 1 | about 5 s in, model loaded (2.79 GB MLX active) | the swap-in counter rose by 4 pages (64 KiB) while swap used stayed 770 MB: the old rule stopped on any swap-in |
| 2 | about 30 s in | memory pressure level 4 (critical) |

Attempt 2, before the stop: seven 5 s samples all at level 1, swap used 770 MB, swap-in 0, swap-out 0. By the stop
the swap file had grown from 2,048 to 3,072 MB, swap used was 2,004 MB, swap-out was +78,976 pages (1.2 GiB) and
the page-in counter was +1.85M pages (about 30 GB). The step from level 1 to level 4 happened inside one 5 s
sample, which is why the 5 s watchdog could not see it coming. No result was written for the arm: `turns` is
filled only when a whole turn (prefill and 128 decode tokens) ends.

**Where in the run [inference, not proof].** Loading is ruled out: attempt 1 had written its after-load record within
about 5 s, and attempt 2 then showed 25 s of level 1. The decode pool is created by the first decode layer, and
the run had read about 30 GB of the roughly 48 GB a 1K prefill streams (at the recorded 1,025-token prefill's 334
of 512 experts per layer), so it was most likely still in prefill. That makes the 6 GB pool a less likely cause
than MLX allocations and page-cache churn during prefill, but the pool has not been excluded. The 1 s trace below
answers it directly.

**Diagnostic run (needs approval; not a speed measurement).** Added to the harness, tested offline:

* The child writes `arm-<id>.trace.jsonl`, one flushed row per second: phase (`load`, `prompts`, `feed`, `prefill`,
  `ring-acquire`, `ring-release`, `post-prefill`, `decode`, `turn-end`, `done`), MLX active, cache and peak, process
  RSS, available memory (free + inactive + speculative), compressor, swap used, pressure level, swap and compression
  counters and the expert pool state. `expert_pool_creating` and `expert_pool_created` are logged as events.
* **Stop before macOS does.** The child exits with code 75 (and writes the reason into the trace) when available
  memory falls below 2 GB (`--min-available-gb`) or MLX active plus cache comes within 0.5 GB of the Metal limit.
  The orchestrator applies the same available-memory floor every 5 s and still stops on pressure level 2 or more,
  any swap-out, swap-in growth over 64 MiB in an arm, thermal throttling and battery. The sample that triggers a
  stop is always in the record.
* `--expert-pool-gb 2` replaces the 6 GB decode pool for the arm; `--ring-min-tokens 64` lets F2's ring serve a
  prompt shorter than its production minimum of 512 tokens. Without it F2 reports "short prompt" and activates
  nothing, so an F0-versus-F2 comparison at 128 or 256 tokens would compare nothing (`--check-ready` warns). Neither
  option changes the Q4 weights or the routing. Neither is a safety guard: the ring's working-set guard stays on.
* Sizes `128` and `256` are diagnostic-only; the planned matrix is still 1K, 2K and 4K.

```
.venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory --check-ready --checkpoint vontra-mtp \
    --sizes 128 --conditions F0 F2 --rounds 1 --expert-pool-gb 2 --ring-min-tokens 64
.venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory --checkpoint vontra-mtp \
    --sizes 128 --conditions F0 F2 --rounds 1 --expert-pool-gb 2 --ring-min-tokens 64 --final
```

Go to 256 only if 128 ends with a comfortable minimum available memory in its trace, then to 1K only if 256 does.
Swap used was 1.58 GB at the last check (it was 0.77 GB before attempt 2 and has not been released); a reboot
before the run would remove that variable.

### 7.1 Result of the 128-token diagnostic (`results/flashnext/20261107-122502-prefill-memory-manual/`)

AC power, Low Power Mode off, 2 GB decode pool, ring minimum 64, 1 s trace. **Both arms completed; pressure stayed at
level 1, swap-out was 0, the lowest available memory was 3.84 GB (F0) and 4.47 GB (F2) against the 2 GB guard.**
Outputs matched: first token 760, KV hash `c3bb3c89b80971ae`, digest `f193ad2f3480ed79`, 35 tokens, finish `stop`
(so the 64-token decode windows are empty). G1 to G4 passed. F2: ring acquired, 48 layers and 9,435 experts staged
(28.98 GB), no fallback, no `last_error`, scratch pool 1.62 GB created in `ring-acquire` and disposed before decode.

Allocations by operation (MLX active, from the trace): load +2.79 GB; F2 scratch pool +1.66 GB at `ring-acquire`
and -1.7 GB when released; decode pool +2.05 GB (3.04 to 5.08 GB, filled in about 2 s) at the first decode layer in
both arms; MLX peak 5.11 GB in both. **Page cache:** F0 read 29.9 GB through the page cache (page-ins +982k pages)
and macOS compressed the process's own memory: process RSS fell from 3.95 GB to 0.88 GB within a second while the
compressor grew from 0.31 to 3.06 GB. F2's reads bypass the cache (page-ins +394 pages, compressions 62k against
404k pages) and its compressor stayed at 0.5 to 0.75 GB. "Available memory" did not show this, because page cache
counts as available. **This run did not reproduce the 1K spike**, so the spike's cause is still open; the F0
compression storm is the leading candidate mechanism, scaling with the 48 GB a 1K prefill reads.

The trace recorded compress and decompress as empty: the parser looked for `Pages compressed`, while `vm_stat`
prints `Compressions` and `Decompressions`. Fixed afterwards (with a test against the live tool) and the trace now also
records file-backed and anonymous pages; the per-arm counters in the result files were correct throughout.
Timing is diagnostic only: n = 1 per arm, and F0 ran first on a colder cache.

### 7.2 Compressor-growth guard, and why F0 compresses and F2 does not (offline)

**Guard.** The child and the orchestrator stop an arm when the compressor grows by more than 2 GiB
(`--max-compressor-growth-gib`, default 2) from that arm's own starting size: the child's first 1 s reading, the
orchestrator's pre-arm sample. The triggering sample, its phase (`during` in the trace's guard row, and `phase` in the
child's `guard_stop` arm file) and its compressor figures are written before the process ends. A stop on a memory rule
is recorded as `classification: resource_stop`, a diagnostic stop and not a correctness failure: no output was
compared. At 128 tokens F0's compressor grew 2.75 GB, so F0 is expected to stop at 256 tokens; the threshold is not to
be raised to finish a run.

**Why F0 reads fill the cache and F2's do not [code, verified].** The baseline reads experts through
`SafeTensorStore._fd`: `os.open(..., O_RDONLY)` with read-ahead cleared (`FLASHNEXT_RDAHEAD=0`) and no `F_NOCACHE`, so
every row goes through the unified buffer cache. The expert pool and the ring read through `ExpertPool._fd`, which
sets `F_NOCACHE`. With no pool during F0's prefill, all of its 29.9 GB passes through the cache.

**What the counters show [measured, 128 tokens].** F0 prefill: page-ins +982k pages (16.1 GB), compressions +404k
pages (6.6 GB), decompressions +234k pages (3.8 GB); at the second the compressor grew by 2.5 GB, the process's RSS
fell from 3.95 to 0.88 GB. F2: page-ins +394, compressions +63k, RSS steady. macOS keeps only 4,000 pages (65 MB)
free (`kern.vm_page_free_target`), so the "available" figure is mostly reclaimable cache, and the VM reclaims
continuously as new cache pages arrive.

**Inference, not yet shown.** To admit the stream of cache pages the VM compressed cold anonymous pages, here the idle
MLX weight buffers (the GPU is quiet while CPU-bound reads run) rather than dropping recently read file pages. Those
pages are decompressed again when the GPU uses them, which is the 3.8 GB of decompressions, and a larger prefill (48 GB
at 1K) plus a 6 GB pool is where compressed memory could overflow into swap. F2 also differs from F0 by holding a 1.62
GB scratch pool during prefill, so F0-versus-F2 alone does not separate the cache effect from the pool.

**Control (test-only, built, not yet run).** Condition `F0u` is F0 with `F_NOCACHE` set on the store's descriptors by
the benchmark child, on its own store object: no environment flag, no pool, no ring, nothing in production changed, and
it is not in the planned matrix. If F0u shows no compressor growth and no RSS drop, the buffered reads are the cause; if
they persist, something else is. It covers reads through `store._fd` (the default `pread` mode); it is not a speed
comparison, since uncached reads lose cache hits.

### 7.3 Result of the 256-token diagnostic (battery, PRELIMINARY)

Battery 76% at launch and at the end, Low Power Mode off, pressure level 1, 2 GB decode pool, ring minimum 64, 1 s
trace, compressor limit 2 GiB, run without `--final` (records marked `preliminary`, battery-powered). F0 and F2 ran as
two separate runs (an F0 stop ends its run): F0 `results/flashnext/20261107-130727-...`, F2 `...-130911-...`.

**F0: diagnostic stop, no output (not a correctness failure).** The child's guard fired 3.16 s in, in `prefill`, about
1.2 s after prefill began: the compressor grew 3.06 GB (0.26 to 3.32 GB), 2.99 GiB against the 2 GiB limit. In that
second anonymous memory fell from 7.19 to 4.39 GB, process RSS from 3.17 to 1.32 GB, compressions +281k pages
(4.6 GB), page-ins +215k pages (3.5 GB). Pressure stayed at 1, swap used stayed 1,332 MB, no swap-in or swap-out,
available memory 5.34 GB. The 5 s orchestrator watchdog took one sample (growth 0): only the 1 s child guard saw it.
At 128 tokens the same jump was 2.75 GB, so the F0 compression is a lump of about 2.5 to 3 GB (the size of the MLX
weights) at the start of prefill and not growth proportional to the prompt: under this limit F0 cannot complete.

**F2: completed.** Digest `eeba4d99e433b2e1`, first token 760, KV hash `789915b197b1c37a`, 45 tokens, finish `stop`;
257 prefilled tokens in 15.2 s (16.9 tok/s, 12.1 s of it waiting on reads), decode 2.07 tok/s overall (diagnostic only).
Ring acquired: 48 layers, 12,209 experts staged (37.5 GB), no fallback, no `last_error`; scratch pool 1.62 GB created at
`ring-acquire` and disposed before decode; decode pool 2.0 GB created at the first decode layer. MLX peak 5.12 GB,
G2 to G4 pass. Compressor growth peaked at 0.88 GB in the second after the scratch pool was created, then drained to
0.19 GB; swap used 1,037 MB throughout, no swap-in or swap-out, pressure 1, lowest available memory 4.03 GB.
F0 has no digest, so exactness of F2 at 256 tokens is not yet compared with the baseline path.

### 7.4 Result of the F0u control at 256 tokens (battery, PRELIMINARY; `results/flashnext/20261107-131700-...`)

Battery 76%, Low Power Mode off, pressure 1, 2 GB pool, compressor limit 2 GiB. **F0u stopped at 2.06 s, in `prefill`,
by the compressor guard: growth 2.99 GB (0.59 to 3.58 GB), RSS 0.37 GB, MLX active 2.80 GB, anonymous memory 3.02 GB,
page-ins +143k pages (2.3 GB), pressure 1, no swap-in or swap-out, 5.2 GB available.** No output, no digest.
Uncached reads were applied: at the stop the store held one shard descriptor and it had `F_NOCACHE` (applied 1, unprotected
none). A separate re-read test on this machine showed the flag works with 3,072,000-byte `pread`s: a re-read of a 1 GiB region
ran at 22 to 23 GB/s buffered (cache) and 4.0 GB/s with `F_NOCACHE`.

**Reading.** The compression happened with uncached expert reads, within about a second of the start of prefill, before
more than one shard had been opened and with only 2.3 GB paged in (essentially the weights). So reads through the file
cache are not the (only) cause: the page-cache hypothesis of 7.2 is not supported. **Caveats:** the run stopped too early to
test the bulk of the 29 GB of reads; and I had read about 4 GiB through the cache in I/O probes just before launch, so
the file cache started at 8.8 GB (5.5 to 5.9 GB in the F2 and F0 256-token runs), a confound that was not controlled.
The `uncached_reads_applied` event at launch lists no descriptors because the store opens shards lazily; the guard-stop
and completion records carry the real report.

**Untested candidates.** (a) The baseline reads with `os.pread`, which returns a fresh 3 MB `bytes` object per row, then
copies it: tens of GB of large anonymous allocations; the ring reads with `preadv` into a fixed buffer. (b) The weights
themselves being compressed right after load. `FLASHNEXT_READ=preadv` (an existing flag) reads into the destination
without the per-row allocation and would test (a); a load-only run (load, idle, no prefill) would test (b).

### 7.5 Result of the 1K F2-only diagnostic (battery, PRELIMINARY; `results/flashnext/20261107-132143-...`)

Battery 75%, Low Power Mode off, pressure 1, 2 GB decode pool, production ring minimum (512), 1 s trace, compressor limit
2 GiB, one round. **F2 did not complete: the child's compressor guard stopped it at 3.08 s, in `prefill`, a diagnostic stop
(no output, so no digest, token count beyond the 988 + template prompt, speed or SSD figure).**

| t | phase | MLX active / cache / peak | RSS | compressor (growth) | anonymous | available |
|---|---|---|---|---|---|---|
| 0.00 | load | 0 | 0.04 GB | 0.69 GB | 4.54 GB | 8.14 GB |
| 1.02 | load | 2.20 / 0 / 2.20 GB | 2.27 GB | 0.69 GB (0) | 6.63 GB | 6.82 GB |
| 1.54 | scratch pool created (1.62 GB, 528 slots) at `ring-acquire` | | | | | |
| 2.05 | prefill | 4.54 / 0.50 / 5.03 GB | 2.32 GB | 2.34 GB (+1.65) | 5.69 GB | 5.55 GB |
| 3.08 | prefill, **stop** | 4.86 / 0.88 / 5.56 GB | 1.52 GB | 3.65 GB (+2.96) | 2.75 GB | 3.78 GB |

Pressure 1, swap used 1,013 MB, no swap-in or swap-out throughout; page-ins +54k pages (the ring bypasses the cache). At
the same instant after the scratch pool (about 2 s) F2 at 256 tokens had compressed +0.12 GB; at 1K +1.65 GB.

**F2 avoided the early compression at 128 and 256 tokens (at most +0.9 GB) but not at 1K (+2.96 GB).** The size of the
jump in every F-arm that stopped (3.06, 2.99, 2.96 GB, and 2.5 GB at 128 tokens) matches the 2.8 GB of resident MLX
weights, and Q4 data barely compresses, so compressor growth is about equal to the anonymous memory that disappears. This
is an inference: pages of the idle weights, not of the expert stream, appear to be what macOS compresses once the
process's memory (weights 2.8 + scratch pool 1.6 + MLX cache and activations 0.9 and rising) outgrows what the VM wants
to keep resident. It was not tested whether it would have reached swap or pressure had the guard not stopped it.

### 7.6 Result of the 1K F2s run: 256-expert scratch ring (battery, PRELIMINARY; `results/flashnext/20261107-132622-...`)

Battery 74%, Low Power Mode off, pressure 1, 2 GB decode pool, production ring minimum 512, MLX cache limit untouched, compressor
limit 2 GiB. **Stopped at 3.08 s in `prefill` by the compressor guard (+3.51 GB, 3.43 GiB): a diagnostic stop, no output.**
Scratch pool as designed: 0.836 GB, 272 slots, created at `ring-acquire` at 1.98 s. No ring statistics exist (they are written on
completion), so staging and fallbacks are unknown; with 334 routed experts on average a layer usually exceeds a 256-expert half
and `stage()` then falls back to the normal read path.

| | F2 512 experts (1.62 GB) | F2s 256 experts (0.84 GB) |
|---|---|---|
| t = 2.05 s: MLX active, RSS, compressor growth | 4.54 GB, 2.32 GB, +1.65 GB | 3.63 GB, 3.17 GB, +0.13 GB |
| stop at 3.08 s: MLX active / cache / peak | 4.86 / 0.88 / 5.56 GB | 4.90 / 0.54 / 4.95 GB |
| RSS, anonymous memory | 1.52 GB, 2.75 GB | 0.83 GB, 4.06 GB |
| compressor growth | +2.96 GB | +3.51 GB |
| available memory, pressure, swap | 3.78 GB, 1, none | 4.57 GB, 1, none |

Halving the ring delayed the first compression by about a second and lowered the MLX peak by 0.6 GB, but the compressor still grew
by 3 GB or more within about a second of the start of prefill, more than with 512 experts. **Scratch-pool capacity is not the
cause.** In all five stopped arms (F0 128-token, F0 256-token, F0u, F2 1K, F2s 1K) pressure stayed at 1, available memory stayed at
3.8 GB or more, and nothing swapped; the one run that was truly unsafe (the first 1K F0 run with the 6 GB pool) also reached
pressure 4 and swapped out 1.2 GiB. The compressor-growth rule therefore stops runs the machine tolerates, and it has so far
prevented the question "does F2 finish a 1K prompt?" from being answered.

### 7.7 Result of the 1K F2 run under the conditional compressor guard (battery, PRELIMINARY; `results/flashnext/20261107-133433-...`)

Guard change (benchmark-only): compressor growth over 2 GiB is a recorded warning; it stops an arm only with growth over 2 GiB
**and** available memory below 3 GiB. Unchanged and mutation-tested: pressure 2 or more, any swap-out, swap-in over 64 MiB, available
memory below 2 GB, Metal headroom, thermal throttling, battery cutoff, Low Power Mode. During prefill the child samples every 0.25 s
(MLX, one `vm_stat` read, about 2 ms) and every 1 s in full (RSS, swap, pressure); the guard is evaluated on every sample.

Run: battery 73% to 72%, Low Power Mode off, original 512-expert scratch ring (1.62 GB), 2 GB decode pool, production ring minimum,
no MLX cache cap, one round. **F2 completed.** 1,025 tokens prefilled in 25.3 s (40.5 tok/s, diagnostic), 77 tokens generated
(finish `stop`), 2.08 tok/s decode on the 2 GB pool (about 0.82 GB read per token, 67% pool misses). Digest `2f84f7d220e782b8`, first
token 760, KV hash `fd83510bd53f9c82`. Ring: acquired, 48 layers, 16,385 experts (50.3 GB), **no fallback of any kind**, no
`last_error`; 16.6 s of the 25.3 s prefill was spent waiting on reads (66%); 53.4 GB read from the SSD. Scratch pool 1.622 GB (528 slots)
created at 1.94 s, absent at the end of prefill, 19 ms to dispose, nothing retained; G2 to G4 pass. Pressure 1 throughout, swap used
1,013 MB, swap-out 0, swap-in 16 pages, no thermal warning.

| phase (t) | MLX active / cache / peak | compressor growth | available | notes |
|---|---|---|---|---|
| load (0 to 1.0 s) | 0.77 / 0 / 0.77 GB | 0 | 8.3 GB | |
| prefill start (2.05 s) | 4.94 / 0.09 / 5.03 GB | +0.07 GB | 6.2 GB | scratch pool up at 1.94 s |
| early burst (3.65 s) | 4.58 / 1.16 / 5.56 GB | **+2.09 GB (warning)** | 4.0 GB | same pattern as F0, smaller |
| prefill, middle (4 to 21 s) | up to 5.6 / 1.3 | drains to +1.26 GB (decompressions 8k to 109k pages) | 4.0 to 4.6 GB | |
| end of prefill (22.9 to 27.2 s) | 5.96 / 1.35 / **6.34 GB** | rises to **+2.58 GB (peak)** | **3.45 GB (minimum)** | file cache 3.2 to 2.5 GB |
| decode (27.5 to 63.5 s) | 5.12 / 0.85 / 6.34 GB | falls to +0.33 GB | 4.1 to 4.6 GB | decode pool 2.0 GB created at 27.5 s |

Reading. F2 did not avoid the early compression at 1K (+2.09 GB within 1.6 s of prefill starting); what differed from the stopped runs is
that it was transient and did not coincide with low available memory. The margin was thin: the peak growth (2.58 GB) came with 3.57 GB
available, and the minimum available memory (3.45 GB, about 3.4 GiB) was about 0.4 GiB above the 3 GiB floor at which the new rule
would have stopped the run. The earlier 1K stops (+2.96 and +3.51 GB at 3 s) were therefore at the very start of this same burst.
This is one preliminary battery run with a 2 GB pool, not the production 6 GB pool, and there is no baseline digest at 1K to compare.
