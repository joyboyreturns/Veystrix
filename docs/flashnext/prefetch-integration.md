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
