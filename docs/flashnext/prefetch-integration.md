# Prefetch integration design: staging inside the 6 GB expert pool

Status: design and memory budget only. No production code is changed. Figures
marked (m) are measured and recorded in `research.md` or `results/`; (d) are
derived by arithmetic from measured values; (h) are hypotheses that need a gate.

## 1. The constraint, and why heap staging is out

`prefetch.py` as committed allocates its own staging buffers. For overlap two
layers' experts are live at once: the layer the GPU is computing and the layer
being read.

- Dense layer: 512 x 3,072,000 B = 1.57 GB, so two = 3.15 GB (d).
- Sparse layer at 1,007 tokens: about 334 experts, 1.03 GB, so two = 2.05 GB (d).
- A 1,007-token prefill already peaks at 6.145 GB of MLX memory (m) and the
  chat default adds a 6 GB pool that "would hold about 9.4 GB" of process
  memory at idle (`research.md`, read-path section). Heap staging on top is
  about 6.1 + 6.0 + 3.2 = 15.3 GB (d), which does not fit. It stays as the
  pool-off fallback only.

## 2. Core idea: prefill reads into the pool, in place

The pool is already one anonymous region of 3,072,000-byte expert records,
wrapped once as a Metal buffer (`expert_pool.py`, `ExpertPool.buffer_mx`). The
pool has `(6e9 - 4096) // 3,072,000 = 1,953` slots (d). Prefill does not touch
it today.

Proposal: for a prefill above a token threshold, lease a **ring** of 1,024
slots (two halves of 512, the most distinct experts one layer can route to).
Prefill reads experts straight into ring slots and runs `gather_qmm` on views
of the pool itself. No separate staging buffer and no per-layer
`np.empty` read buffers exist any more.

### 2.1 gather_qmm operands without a copy

Nine `as_strided` views of the pool (gate/up/down x weight/scales/biases),
built once per pool, shape `[slots, rows, cols]`, expert stride = record
stride, base offset = the part's offset in the record. `local` indices become
slot numbers. `expert_record_view` in `store.py` already builds the same strided
NumPy views for writing.

Gate P1 (must pass before anything else): MLX `gather_qmm` must accept an
expert-axis stride larger than one matrix without copying. If MLX makes a
contiguous copy it would copy up to 6 GB and the design is dead; the fallback
is a part-major pool layout, which changes decode kernel addressing and is a
separate project. (h: I recall MLX only requiring the last two dimensions to
be row-contiguous, but MLX is not installed here and I have not verified it.)

### 2.2 Ping-pong makes hazards structural

Ring half A holds layer L, half B holds layer L+1, then they swap.

- At the end of `_one_pass(L)` the previous router sync has drained the GPU,
  so only MoE_L (just enqueued, reading half X) can run. Writes for L+1 go to
  the other half. No pinned-slot tracking is needed.
- By the time the next prefetch targets half X again, layer L+1's score sync
  has drained MoE_L (MLX is lazy: MoE_L does not even dispatch until then).
- Dense mode needs no prediction: slot = half base + expert id, and the
  ticket for L+1 is submitted at the end of `_one_pass(L)`. The reads run
  during layer L+1's `mx.eval(scores)` in `_moe_call`
  (`adaptive_topk.py:99-102`), which is exactly the GPU block that drains
  MoE_L plus attention_{L+1}. This adds no `mx.eval` to the path.

### 2.3 Sparse mode (below about 2K tokens)

Predicted experts are staged by id into the next half. After the real router
runs, the misses are read on demand into free slots of the same half (a half
holds 512, enough for every distinct expert). The predictor (layer L+1's gate
on layer L's hidden state) needs one small eval before submit and is a later
stage. Dense-only is stage 1.

### 2.4 Exactness (the second gate)

Today `local` is first-seen order, so the sorted row order of
`_gather_sort` follows first appearance. With slot or expert-id indices the
runs are ordered by id instead. The bytes per expert are identical, but the
order of runs could in principle change the tile layout. Gate P2: on random
weights with the real shapes (hidden 2560, intermediate 640, Q4/G32, 512
experts, widths up to 10), `mx.array_equal` of `gather_qmm` outputs between
compact first-seen arrays, id-ordered strided pool views, and a shuffled slot
order, at 1K, 2K and 4K rows. The E1 experiment (digest changed at 1,007
tokens, not at 96) shows row grouping can matter, so this is not assumed.
If it fails, sparse mode can allocate slots monotone in first-seen rank, and
dense mode is dropped or needs an index remap.

## 3. Reads

- Reuse `ExpertPool` machinery: `F_NOCACHE` descriptors, so prefill reads
  stop filling the page cache (today 49 GB flows through it with about
  100 MB free (m)). Submit on the existing `_POOL`, so no new worker count.
- A dense part is one contiguous file range of 512 rows. Read it as `preadv`
  with up to 512 iovecs, one per ring slot (rows are page multiples: 819,200 and
  102,400 bytes), split into chunks of about 16 rows so the queue stays full.
  This gives sequential file reads into scattered destinations. E2 failed
  because fewer, larger tasks lowered queue depth; chunking keeps depth.
- Dense reads 1.53x the bytes of a sparse layer at 1K tokens (1.57 vs
  1.03 GB, d) and about 1.2x at 2K. Whether sequential reads at the drive's
  2.9 to 3.0 GB/s isolated rate pay for that is gate P3. The log says 1.9 GB/s
  is what this access pattern reaches (m).

## 4. Pool lease

`ExpertPool.lease_ring(n_slots)` (not yet written):

1. Take the `n_slots` highest slots, drop their `_slot_of` entries and remove
   them from `_free`, so decode LRU cannot assign them. Slots 0..S-1025 keep
   their decode-hot contents untouched.
2. `release_ring()` in a `finally` around prefill (including
   `GenerationCancelled`) returns the slots to `_free`. Optionally register the
   last layers' ring contents as valid `(layer, expert)` entries: they are real
   records and decode's late layers may hit them.
3. Prefill and decode never overlap (one generation thread), so no locking
   beyond the prefetcher's own.

Cost to decode: the pool is about half its size until the freed ring refills.
The replay found 3 and 4 GB pools lost to the page cache and 5 GB tied (m), so
the first tokens after a long prefill run at roughly a 3 GB pool until LRU
refills, about 3.1 GB of reads spread over the following tokens (d). Against a
prefill saving of about 10 to 25 s this is small, but it makes short prompts
a loss: only lease above a token threshold (start at 1,024, tune by gate).

## 5. Memory budget (16 GB, M4)

| Component | Today, pool on, 1K prefill | Pool-in-place |
|---|---|---|
| Dense core | 3.4 GB (m) | 3.4 GB |
| Other MLX transients (activations, sorted rows, logits) | about 1.7 GB (d: 6.145 peak - 3.43 - 1.03) | about 1.7 GB |
| Per-layer expert read buffers | about 1.0 GB sparse (d) | 0 (reads land in the pool) |
| Pool | 6.0 GB if warm (h), 0 if never touched | same 6.0 GB, ring inside it |
| Staging added by prefetch | n/a | 0 |
| **Process total** | **about 12.1 GB warm (d)** | **about 11.1 GB warm; about 8.3 GB if the pool was cold and only the 3.15 GB ring is touched (d)** |

Notes:

- The "today" column has never been measured with the pool on during a
  prefill. That is a gap. If it is near 12 GB it may already be at the edge.
- Open question H1: Metal's recommended working set on a 16 GB M4 is
  commonly about 10.7 to 12 GB (h, not verified). A 6 GB wrapped pool plus
  6 GB of MLX allocations may sit at or over it. Read
  `mx.metal.device_info()` at startup and compare before enabling; this is
  also a reason to clear the MLX cache between layers in the ring path.
- The ring never exceeds 1,024 of the pool's 1,953 slots. The budget flag
  (`FLASHNEXT_PREFILL_PREFETCH_GB`) should now mean "GB of the pool to lease",
  clamped to half the pool and to `safe_budget` over free memory.

## 6. The documented rejection of overlap

`research.md` (about lines 765 to 803 and 958 to 968) records that every
in-process overlap of SSD reads with GPU work lost on this machine and says
not to retry. The mechanism offered is unified-memory contention: about 31%
GPU loss under concurrent SSD traffic here, up to 73% in the flash-moe report.
Two facts keep this design worth one gated test. Issue #45 records that the
contention interval was never validated (about line 3562), and the log itself
says to re-measure overlap here (about line 795). The decode tests overlapped
a few milliseconds per layer; prefill overlaps seconds.

Arithmetic (d), using the 1,007-token profile (reads 25.6 s, GPU 14.9 s,
serial total 40.5 s): overlapped time is about `max(R, C x (1 + s))`. Even
at s = 73% this is `max(25.6, 25.8)`, about 26 s versus 40.5 s serial. A win
survives pessimistic contention, but only if reads do not also slow down, so
this is a hypothesis, not a result.

Gate P4, no model needed: a synthetic MLX loop of `gather_qmm` on random
weights with a concurrent `F_NOCACHE` `preadv` stream of a scratch file on the
same SSD (not model weights), measuring GPU slowdown `s` and read throughput.
Reuse the structure of `bench_native_dma_contention.py`.

## 7. Code changes for later (nothing done now)

| File | Change |
|---|---|
| `expert_pool.py` | `lease_ring`, `release_ring`, strided operand views, scatter `preadv` fill |
| `prefetch.py` | An arena seam: heap arena (today) and pool-ring arena; ticket records slots, not buffers |
| `expert_cache.py` | Pool branch in `_one_pass` for rows > 8; submit L+1 at the end of `_one_pass(L)`; claim at L+1; build views once, not per layer |
| `macqwen/backends/flashnext.py` | Lease at prefill start, `cancel` plus release in `finally` and in the `layer_completed` cancel path |
| `loader.py` | Record the next layer's block for the later predictor (`next_prefix` exists and is unused) |
| `settings/` | Redefine the flag as pool GB to lease; no launch default |

The production path stays untouched while the flag is 0 and while the pool or
the threshold conditions are not met.

## 8. Order of work

1. P1 and P2 (synthetic MLX, bitwise): decide whether the pool views work at
   all. Needs the project venv but no checkpoint.
2. P4 (synthetic contention): decide whether the premise survives.
3. Mock unit tests for `lease_ring` and the arena seam, which need no MLX.
4. Wire it behind the flag, then full-model gates with permission: identical
   digest, paired reversed arms at 1K and 2K, MLX peak and free memory,
   exit-137 watch, and decode rate for 128 tokens after a long prefill.

## 9. Measured results, 2026-11-06 (synthetic, this M4 Air, no checkpoint)

> **Power-state caveat (recorded 2026-11-07).** The machine may have been on
> battery in Low Power Mode when these numbers were taken; the power state was
> not logged. Treat absolute times, GB/s and the ratios between arms as
> unconfirmed until they are repeated plugged in with Low Power Mode off. The
> results are kept as measured. Nothing below is a production speedup claim.

Environment: `.venv` from `pyproject.toml` (MLX 0.32.2, Python 3.12). MLX
reports `max_recommended_working_set_size` 12,713,115,648 B and
`max_buffer_length` 9,534,832,640 B, so H1 in section 5 is confirmed: the
working-set limit is 12.7 GB. Correction to earlier text: the layer shape is
hidden 2560, intermediate 640 (the slab layout's 640x320 weight rows), not
2048 and 800. Scripts are in the session scratchpad and are not committed.

### Experiment A: operands (512 experts per ring half, random Q4/G32 weights)

| Operand layout | Extra memory during `gather_qmm` | Bit-identical to today's compact arrays | Time vs compact |
|---|---|---|---|
| Part-major views inside the pool (9 contiguous slices) | 0 (view build adds 0 bytes; op peak equals compact) | yes, at 64, 256, 1,024, 2,048 and 4,096 tokens | same (235.7 vs 230.8 ms at 2,048; 395.1 vs 397.4 ms at 4,096) |
| Record-layout strided views (design v1, section 2.1) | +1.6 GB at 256 and 1,024 tokens, +1.9 GB at 2,048, +2.3 GB at 4,096 | yes | +40 to 90% (177.5 vs 95.3 ms at 256; 332.5 vs 230.8 ms at 2,048; 537.3 vs 397.4 ms at 4,096) |

- At 64 tokens (640 rows) the strided view does not copy. From about 256
  tokens (2,560 rows) MLX materializes the whole strided expert span, so the
  record-layout view is **rejected**: it is the second 1.6 GB buffer the
  design set out to avoid, and it is slower.
- Part-major placement is invariant to order: id order, a random permutation
  and first-seen rank all gave identical bits. This closes the exactness gate
  of section 2.4 for this kernel and these shapes. Sorted row order is not a
  numerical hazard here.
- An unrequested observation: in a smaller diagnostic (256-expert half, a
  768-slot whole-pool view, gate projection only) the whole-pool strided view
  returned wrong bits at 1,024 and 2,048 rows. I did not investigate. It is a
  further reason to avoid strided views.
- The routed-expert chain alone (gate, up, down on 20,480 rows) took about 236
  ms per layer at 2,048 tokens, or about 11 s over 48 layers: 0.85 TFLOPS
  against a GPU peak of about 4 TFLOPS (peak is an external figure). That is
  a hard ceiling near 180 tok/s for routed-expert compute alone at this kernel
  efficiency.

**Revised architecture:** lease a contiguous slot range and re-lay it out
part-major (nine contiguous arrays of 512 rows, the checkpoint tensors'
layout), not as 3,072,000-byte records. A leased slot is not a decode record
while leased, so the layout is free. Each part is then one contiguous file
range read into one contiguous destination, and `local` is the expert id.

### Experiment B: SSD reads and compute (scratch file of random bytes, 8 layers, 8 workers, F_NOCACHE, reversed rounds)

| Tokens, GPU scale | Dense read ms/layer (1.57 GB) | Sparse read ms/layer (334 experts, 1.03 GB) | Serial, dense | Overlap, dense | Overlap speedup |
|---|---|---|---|---|---|
| 1,024 x1 | 432 | 300 | 505 | 431 | 1.17x |
| 1,024 x2 | 421 | 297 | 578 | 446 | 1.30x |
| 2,048 x1 | 420 | 298 | 538 | 435 | 1.24x |
| 2,048 x2 | 419 | 298 | 711 | 529 | 1.35x |

- Dense reads ran at 3.64 to 3.76 GB/s, sparse (2-row tasks) at 3.42 to 3.45
  GB/s. The recorded production prefill averaged 1.0 GB/s and 1.9 GB/s while
  waiting, on a machine with a 256 GB SSD. This machine's SSD is faster, so
  absolute times do not transfer.
- Reads during overlap: 412 to 414 ms/layer in three cases against 420 to 432
  alone (no slowdown), and 490 against 419 in the heaviest case (+17%).
- Overlap came within 0 to 13% of the ideal `max(read, GPU)` in the
  read-bound cases and about 26% over it in the heaviest case.
- GPU interference is **not resolved**. The GPU-only arm ran 1.5 to 2 times
  slower per layer than the GPU portion inside the serial arm (for example 213
  vs 118 ms at 2,048 tokens), reproducibly in both round orders. That looks
  like a GPU clock or power-state effect, the same family as the keep-warm
  finding, so the printed "GPU slowdown" figures (-51% to +4%) are not valid
  evidence either way. The end-to-end serial-over-overlap ratios are the
  reliable measure, because both arms do identical work.
- Memory: MLX peak 3.3 to 3.5 GB, max RSS 4.36 GB (both ring halves plus
  activations), zero page-outs, swap-outs and compressions, swap use unchanged
  (1,814 to 1,806 MB), free memory 8.8 GB afterwards.

### What the numbers say about a net gain

Dense reading costs 41% more time per layer than the sparse pattern at 334
experts (420 vs 298 ms). Against today's sparse-serial path (derived: serial
minus dense read plus sparse read) dense-overlap came out at 0.87x, 1.02x,
0.96x and 1.12x in the four synthetic cases, i.e. no gain when the stand-in
GPU time per layer is short. Overlap with dense reads beats sparse-serial
whenever GPU time per layer exceeds about 120 ms (and reads stay hidden when it
exceeds 420 ms), provided contention inflates the GPU by less than the margin.
The recorded 1,007-token profile had about 310 ms of GPU time per layer
(14.9 s over 48 layers, old clock) and 626 ms at 2,031 tokens by scaling, which
is where dense-overlap should win (derived 1.4x to 1.6x, a projection, not a
measurement). The synthetic GPU stand-in covers routed experts only, with no
attention, so it understates real per-layer GPU time.

Below roughly 1.4K tokens the dense read (about 420 ms per layer here) exceeds
the GPU time, so overlap there is read-bound and sparse prefetch needs the
predictor. A separate and cheaper lever appeared: sparse reads with
`F_NOCACHE` into resident pool pages reached 3.4 GB/s here against the log's
1.9 GB/s (the machines differ, so this is a hypothesis for the reference
machine).

## 10. Measured results, round two (synthetic, this M4 Air, no checkpoint)

> **Power-state caveat (recorded 2026-11-07).** The machine may have been on
> battery in Low Power Mode when these numbers were taken; the power state was
> not logged. Treat absolute times, GB/s and the ratios between arms as
> unconfirmed until they are repeated plugged in with Low Power Mode off. The
> results are kept as measured. Nothing below is a production speedup claim.

Scripts are in the session scratchpad and are not committed. Same hardware,
`.venv` with MLX 0.32.2, a 28 GB scratch file of random bytes (deleted after),
8 workers, `F_NOCACHE` where stated.

### Read paths (334 experts, 6 layers per arm, 3 reversed rounds)

| Arm | ms/layer (median) | vs today | Compressions (all rounds) |
|---|---|---|---|
| today: buffered pread + copy into fresh arrays | 330 | 1.00x | 94,997 |
| buffered preadv into fresh arrays | 347 | 0.95x | 198,532 |
| F_NOCACHE pread + copy into fresh arrays | 333 | 0.99x | 5,652 |
| F_NOCACHE preadv into fresh arrays | 293 | 1.13x | 2,862 |
| F_NOCACHE preadv into resident ring, first-seen order | 196 | 1.68x | 47,108 |
| same, sorted by expert id | 324 | 1.02x | 9,408 |
| same, sorted, 16 workers | 261 | 1.27x | 0 |

- The gain comes from the resident destination, not from `F_NOCACHE` alone.
  The best arm varied between 1.10 s and 1.59 s across rounds, so read 1.68x as
  about 1.2x to 1.8x.
- Sorting by expert id is slower (324 against 196 ms). I cannot explain it; keep
  the first-seen order the current path already uses.
- Memory compression is explained by the buffered arms: they caused 95,000 to
  199,000 compressions (about 1.5 to 3.2 GB of pages) against 0 to 9,000 for
  the direct arms, because the page cache grows and macOS compresses other
  anonymous memory to make room. The ring arm's 47,000 is unexplained (it may
  be idle ring pages being compressed between arms) and must be re-checked on
  the integrated path. Swap-outs were 0 everywhere.

### Duty-matched GPU contention (median GPU time per layer, ms)

| Case | back to back | after an idle gap | during overlapped reads |
|---|---|---|---|
| 1,024 tokens, 334 experts, 1x GPU | 124 | 66 | 71-77 |
| 1,024, 334, 2x | 254 | 138 | 144-253 |
| 2,048, 400, 1x | 211 | 118 | 114-119 |
| 2,048, 400, 2x | 437 | 274 | 370-466 |
| 1,024, 512, 1x | 140 | 85 | 81-85 |

The same GPU work runs 1.6 to 1.9 times slower back to back than after an idle
gap, with no SSD traffic, so most of the earlier slowdown is a sustained-load
effect, not SSD contention. During overlapped reads the GPU was within 0 to 11%
of the idle-gap time when its duty was low, and approached the back-to-back time
as duty approached 100%.

### Sparse prefetch with overlap against today's sparse-serial path (speed, higher is faster)

| Predictor (recall/precision) | 1K, 334, 1x | 1K, 334, 2x | 2K, 400, 1x | 2K, 400, 2x | 1K, 512, 1x |
|---|---|---|---|---|---|
| 1.0 / 1.0 | 1.65x | 1.38x | 1.58x | 1.37x | 1.43x |
| 0.90 / 0.80 | 1.05x | 1.19x | 1.16x | 1.35x | 1.29x |
| 0.75 / 0.70 | 1.12x | 1.16x | 1.19x | 1.25x | 1.26x |
| 0.50 / 0.50 | 0.95x | 1.03x | 1.18x | 1.20x | 1.43x |

The overlap arms read in sorted order, which the read-path table shows is the
slow order, so these are conservative. Predictor recall on a real prefill is
unmeasured; the only recorded figure is 72% at top-10 for one decode token.

### Consequence for stage 1

Stage 1 reads exactly the routed experts, first-seen order, with `F_NOCACHE`
`preadv` into a resident ring half and runs `gather_qmm` on part-major views:
no prediction, no concurrent SSD and GPU work. It needs one half, not two,
because a layer's reads cannot start before the previous layer's compute has
finished.

## 11. Stage 1 as implemented (opt-in, off by default)

### 11.1 What it does

`FLASHNEXT_PREFILL_RING_EXPERTS=N` (N > 0, with `FLASHNEXT_EXPERT_POOL_GB` set)
makes `prefill_target` hold one ring half of N experts, leased from the top of
the existing pool, for the duration of one prefill (`prefill_ring.session`).
In `StreamingSwitchGLU._one_pass`, for layers with more than 8 rows, the
routed experts (first-seen order, exactly the router's choice) are read with
`F_NOCACHE` `preadv` into the half and handed to `gather_qmm` as nine
zero-copy part-major views. No prediction, no read concurrent with GPU work.
With the flag unset, `prefill_target` calls the unchanged function and
`_PREFILL_RING` stays `None`, so the existing path is byte-for-byte what ran
before.

Settings (research, startup): `FLASHNEXT_PREFILL_RING_EXPERTS` (0),
`..._MIN_TOKENS` (512), `..._MARGIN_GB` (0.5), `..._TRANSIENT_GB` (2.0).

### 11.2 Guards and fallbacks (each makes that layer use the existing path)

| Condition | Behaviour |
|---|---|
| Layer routes more experts than the half holds | fall back, counted |
| Read error or short read | half released, error recorded, fall back |
| Half busy | fall back |
| Live MLX active memory + pool + margin over the Metal limit | ring disabled for the rest of the prefill |
| Prompt shorter than `MIN_TOKENS`, pool off, layouts differ, dtype unsupported, limit unknown | ring not acquired; `LAST_REASON` says why |
| Working set at acquire: pool + active + transient + margin over limit | not acquired; reason prints the four numbers |
| Exception or `KeyboardInterrupt` in the prefill or a read | wait for the GPU (`fence`), release the half, return every slot, clear the hook |

The half is retired at the next layer's `stage` (the router sync before it
proves the previous layer's GPU work finished) or at session end, after the
final evals; `prefill_target` also evaluates anything it returns and fences
before the lease is given back.

### 11.3 A deadlock found while testing: do not call `mx.synchronize`

`mx.synchronize()` keeps the GIL while it waits. MLX's completion handler can
need the GIL to free a numpy-backed operand (the existing read path wraps
numpy buffers with `from_dlpack`), so a synchronize that actually waits can
deadlock. A test hung there; a stack sample showed the main thread in
`waitUntilCompleted` and the completion queue blocked. The ring uses
`prefill_ring.fence()` instead, a one-element `mx.eval`, which releases the
GIL and cannot finish before earlier work on the same stream. Anything else
added around this path should do the same.

### 11.4 Offline test results (2026-11-07, no GPU benchmark)

New: `test_prefill_ring.py` (32 tests), 4 `halves=1` tests in
`test_pool_ring.py`. They use a synthetic checkpoint with real row sizes
(hidden 2560, intermediate 640, Q4/G32), the real `SafeTensorStore`, real
`StreamingSwitchGLU` layers and a real `ExpertPool`. They check: ring output
bitwise equal to the existing path across two layers; staged views equal the
checkpoint rows in first-seen order with the right dtypes; zero-copy
(active memory grows under 1 MB for 8 experts, a copy would be 6 MB or more
per projection); nothing outside the leased range is written; decode-sized
calls never touch the ring; every fallback and refusal in 11.2; slot
restoration on normal exit, exception, cancellation and a second session;
the `prefill_target` hook wiring.

Full offline suites: flashnext 525 tests, one failing assertion in my own new
test (a released lease handle reports `stale`, the half is `free`), fixed and
re-run with `test_pool_ring`, `test_prefill_ring` and `test_settings_registry`
(67 OK); macqwen 427 OK. The complete flashnext suite is re-run before commit.

### 11.5 The memory compression question (first attempt; superseded by section 12)

*Update 2026-11-07: section 12 has the measured answer. The paragraphs below
are the state before those runs and contain one wrong inference, corrected in
12.7.*

Hypothesis from section 10: buffered page-cache reads cause the compressions
(95K and 199K per arm for the buffered arms, 0 to 9K for `F_NOCACHE`), but the
resident-ring arm showed 47K that nothing explained, and it had no idle
control.

An integrated-path run (`exp_e_integrated.py`: real layers, a 12.6 GB random
checkpoint, existing path vs ring vs an idle control) was started on
2026-11-07 and stopped by itself after the first arm. It is **not** a result:

* The existing path's first arm recorded 499K compressions, 417K
  decompressions and 797K page-ins in 10.2 s with MLX peak 7.0 GB, a heavy
  memory-pressure signature on this machine, but at the start free memory was
  7.6 GB and the power state is unknown (Low Power Mode possible).
* The ring then refused to acquire, correctly, on the working-set check
  (`pool + active + transient + margin` over the 12.7 GB limit). The refusal
  message did not print its numbers then; it does now.
* No ring or idle arm completed, so nothing here shows whether the ring
  reduces, matches or increases compression.

Do not enable the ring by default until the run is repeated plugged in, Low
Power Mode off, and the ring arm and the idle control have both completed.
Per-arm compression rates against the idle control are the figure that
matters.

### 11.6 Still to validate on the real Qwen3.8 checkpoint

All of this needs the real checkpoint on the plugged-in, Low Power Mode off
machine, after you approve a benchmark window.

1. **Working-set fit, the main risk.** MLX's peak already includes the
   wrapped 6 GB pool (12.7). The synthetic MoE-only runs peaked at 7.2 to 8.2
   GB without the ring and 6.2 to 6.8 GB with it, but the real model also holds
   the dense core (about 3.4 GB) and transients (about 1.7 GB, section 5). The
   section 5 estimate stands: about 12.1 GB with the pool warm and no ring, so
   about 0.6 GB under the 12.713 GB limit, and about 11.1 GB with the ring
   (the 1 GB of per-layer read buffers disappears). Measure the real peak,
   including attention and KV cache, at 1K, 2K and 4K tokens, with and without
   the ring, and confirm `_over_limit` trips before Metal pressure. The acquire
   check uses `transient` (2.0 GB default) as an estimate: with the dense core
   and pool resident (about 9.4 GB) plus 2.0 plus the 0.5 GB margin it passes
   at about 11.9 GB, which is close to the limit.
2. **Exactness.** Compare the digest of ring-on and ring-off prefill outputs
   (logits or hidden state) token for token on real prompts. Synthetic
   bitwise equality is necessary, not sufficient.
3. **Speed.** Paired, order-reversed arms at 1K and 2K tokens. Report only
   measured numbers.
4. **Compression and free memory** with an idle control (11.5), and the
   exit-137 watch during a 2K-token run.
5. **Decode after prefill.** The pool's cached experts below the ring stay
   valid; check decode rate and hit rate after a ring prefill match the
   existing path.
6. **Ring size.** The default request is the flag value; a layer that routes
   more experts than the half falls back. Check the real distribution
   (about 334 per layer at 1K tokens) against `FLASHNEXT_PREFILL_RING_EXPERTS`.

## 12. Stage 1 benchmark, battery mode (2026-11-07)

**Preliminary.** MacBook Air M4, 16 GB, on battery with Low Power Mode off
(battery 92% to 84% over the whole session), no thermal warning recorded,
`CPU_Speed_Limit` 100 throughout. These are not the final plugged-in
benchmarks. Synthetic random Q4/G32 tensors with the real expert row sizes
(3,072,000 B per expert), a real store, real `StreamingSwitchGLU` layers, a
real `ExpertPool` and the real `prefill_ring` session. **MoE layers only:** no
attention, router, KV cache or n-gram work, and random routing. No model
weights. Raw data, logs and the harness: `docs/flashnext/results/stage1-ring-battery-2026-11-07/`.

### 12.1 Two bugs that only a real-size pool could show

Both were in the first Stage 1 commits (6ec6cc0, c5d3ca9). The unit tests
missed them because their pool is 0.15 GB. Both are fixed in the commit that
follows them and are covered by new tests that fail on the old code (checked in
a separate worktree at c5d3ca9: all five regression tests fail there; the slice
tests also fail, with the original `Slice indices must be 32-bit integers`
error, when only the old slicing is put back under the new guard).

1. **The pool was counted twice in the memory guard.** MLX counts the wrapped
   pool in `get_active_memory` and `get_peak_memory`: creating a 6 GB pool
   raised active memory by exactly 6.000 GB. The acquire and per-layer guards
   added `pool_bytes` on top, so with a 6 GB pool the ring refused every time
   (`6 + 6 + 2 + 0.5` against 12.71 GB). Now `working_set_bytes` counts it once.
2. **bf16 slices overflowed MLX's 32-bit slice index.** A ring sitting above
   4.29 GB in a 6 GB pool needs bf16 element indices above 2^31, and MLX
   raised `Slice indices must be 32-bit integers`. Every layer fell back to the
   old path. Parts are now sliced as uint32 (index below 2^31 up to 8 GiB) and
   reinterpreted as bf16; a part that cannot be 4-byte aligned or lies beyond
   the range falls back.

Also, one unit test was flaky (about 1 failure in 15 runs): it derived a limit
from one active-memory reading while MLX freed earlier tests' arrays
asynchronously. Both guard tests now patch the reading.

### 12.2 Design and safety

Each size ran in a fresh process with a blocked design: three runs of one arm,
three of the other, then an idle control, order reversed every round, 3 rounds,
8 layers per arm, fresh random experts per arm so the page cache cannot favour
either arm. A bit-identity round per size ran both arms on identical routes
and compared a hash of every layer's output. The watchdog stops on battery
below 25%, memory pressure level 2 or higher, any new swap-out, a CPU speed
limit below 90, Low Power Mode on, or an MLX peak within 0.2 GB of the 12.713
GB Metal limit. Result: the final runs never tripped it. The earlier
alternating run did trip it (memory pressure level 2 at the start of its 2K
phase, after 24 arms in one process); nothing swapped.

### 12.3 Results (steady state = runs 2 and 3 of each block; ON means ring on)

| | 1K (334 experts) | 2K (400) | 4K (460) |
|---|---|---|---|
| Bit-identical to the original path | yes, 8 layers | yes, 8 layers | yes, 8 layers |
| Wall speed-up, steady state (median, range, 95% t half-width, n=3) | 1.100 (1.090-1.123, ±0.042) | 1.099 (1.099-1.124, ±0.036) | 1.074 (1.035-1.082, ±0.062) |
| Wall speed-up, all arms | 1.078 (1.074-1.086) | 1.095 (1.055-1.100) | 1.063 (1.000-1.083) |
| ON faster in rounds | 3 of 3 | 3 of 3 | 3 of 3 steady, 2 of 3 all arms |
| Wall per 8-layer arm, OFF to ON | 3.22 s to 2.90 s | 4.23 s to 3.86 s | 5.31 s to 4.99 s |
| Expert read per layer, OFF to ON | 328 ms to 293 ms | 399 ms to 370 ms | 464 ms to 427 ms |
| SSD throughput, OFF to ON | 3.12 to 3.52 GB/s | 3.08 to 3.33 GB/s | 3.05 to 3.32 GB/s |
| GPU gather_qmm per layer, OFF to ON | 72.8 ms to 70.2 ms | 126.1 ms to 111.8 ms | 200.2 ms to 193.8 ms |

The read gain is 8 to 11%. The GPU time is also a few percent lower with the
ring, but the 95% interval on the GPU ratio at 2K is ±0.22, so that part is not
established.

### 12.4 Compression and memory

| | 1K | 2K | 4K |
|---|---|---|---|
| Compressions per second, OFF steady / ON steady | 6,223 / 220 | 13,911 / 285 | 5,460 / 323 |
| Compressions per second, OFF all arms / ON all arms | 29,361 / 3,107 | 28,952 / 2,793 | 26,263 / 2,673 |
| Idle control, compressions per second | 0 | 0 | 0 |
| MLX peak incl. the 6 GB pool, OFF / ON | 7.22 / 6.19 GB | 7.61 / 6.38 GB | 8.18 / 6.76 GB |
| Working memory above the pool, OFF / ON | 1.2 / 0.2 GB | 1.6 / 0.4 GB | 2.2 / 0.8 GB |
| Lowest free+inactive memory after an arm, OFF / ON | 5.9 / 3.0 GB | 5.7 / 2.9 GB | 5.8 / 2.9 GB |
| Max pressure level, swap-outs | 1, 0 | 1, 0 | 1, 0 |

* The idle control shows no background compression, so the compressions
  belong to the arms.
* Compression is mostly a transition effect. The first OFF run after an ON
  block caused 312K to 408K compressions (415K to 622K at the very start); the
  first ON run after an OFF block caused 33K to 59K; an ON run after an idle
  gap or another ON run caused none. This is why the alternating design gave
  an order effect (below).
* The ring holds more memory resident: free+inactive memory after an ON arm
  was about 2.9 GB against 5.7 to 5.9 GB after an OFF arm, because the ring's
  pool pages stay resident as anonymous memory while OFF leaves reclaimable
  page cache. Pressure stayed at level 1 and nothing swapped, but that is the
  number to watch on the real model, where attention, KV cache and the
  non-expert weights also need memory.
* In these MoE-only runs the 12.713 GB Metal limit was never approached: the
  closest was OFF at 4K, 4.5 GB below it. The real model has about 5 GB more
  resident (dense core and transients), so this is not the real headroom; see
  12.7 and 11.6.

### 12.5 What the earlier numbers did and did not show

* The 2-round smoke run read 1.55x on a 6.3 GB file. That is inflated: the
  small file stayed in the page cache. Do not quote it.
* The alternating 1K run (6 rounds) gave a median of 1.13x with a range of
  0.95 to 1.26. By round the ratio was about 1.0 when the OFF arm ran first
  and 1.2 when the ON arm ran first, because ON leaves less free memory and
  the following OFF arm pays for compression. The blocked design is the fairer
  comparison for a prefill, which runs layers back to back.
* Experiment D (section 10) measured the resident ring reading 1.2 to 1.8x
  faster than today's path. The integrated path measures 1.08 to 1.11x. The
  two are not reconciled; D timed reads alone with a pre-touched ring.

### 12.6 How far to trust it

* Three rounds per size, so the intervals are wide (±0.04 to ±0.11) and a
  ratio of 1.00 to 1.10 at 4K cannot be told apart from a small gain.
* MoE layers only. A real prefill also spends time in attention, the router
  sync and n-gram work. For scale, the recorded 1,007-token profile spent 25.6
  s of 49.3 s in `io_wait`; a read gain of 8 to 11% on that share would be
  about 2 to 3 s, roughly 5% end to end. That is a hypothesis, not a
  measurement.
* Random routing, and the expert counts for 2K (400) and 4K (460) are
  estimates; only 334 at 1K is recorded.
* Battery power. The earlier synthetic numbers and these may all move when
  repeated plugged in. The GPU probe stayed near 6 ms with isolated spikes to
  16 to 18 ms in single arms, so there was no sustained throttle.
* The page cache and compressor state carry over between arms and set the
  transition effects; real prefill has its own history.

### 12.7 Corrections to earlier text

* Section 11.5 and an earlier note said the synthetic path's 7.0 GB MLX peak
  "plus the 6 GB pool" was already over the 12.7 GB limit. That was wrong for
  those synthetic runs: MLX's peak already includes the pool, so 7.0 GB meant
  about 1 GB of working memory above it.
* Section 5 is **not** affected. Its real-model estimate (about 12.1 GB with
  the pool warm) adds the pool once to a recorded 1,007-token peak of 6.145 GB
  that had no pool in it, so the real-model headroom is only about 0.6 GB
  without the ring. The synthetic headroom of 4.5 GB or more in 12.4 is for
  MoE layers alone and must not be read as the real figure.
