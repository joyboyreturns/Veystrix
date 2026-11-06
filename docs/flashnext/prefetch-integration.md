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
weights with the real shapes (hidden 2048, intermediate 800, Q4/G32, 512
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
