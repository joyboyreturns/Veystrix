"""Stage 1 of the in-pool prefill read path (opt-in, off by default).

Today a prefill MoE layer reads its routed experts into freshly allocated NumPy
arrays through the page cache, one buffered ``pread`` and copy per row. Stage 1
changes only where those bytes land and how they are read:

* The destination is one half of a part-major ring leased from the 6 GB expert
  pool (``pool_ring.PoolRing``), so no per-layer read buffer is allocated.
* Rows are read with ``preadv`` on ``F_NOCACHE`` descriptors, straight into the
  ring, so the page cache is not filled.
* The nine part arrays handed to ``mx.gather_qmm`` are zero-copy contiguous
  slices of the pool's Metal-wrapped buffer.

It reads exactly the experts the router selected, in first-seen order, so the
``gather_qmm`` operands and the ``local`` index map are identical to the
current path. There is no prediction and no read concurrent with GPU work: a
layer's reads still start only after the previous layer's compute is done, which
is why one ring half is enough.

Ownership and safety
    ``session`` leases the ring when a prefill starts and returns the slots when
    it ends, on every exit path. The half used by a layer stays COMPUTING until
    the next layer's reads begin (the router sync that precedes them proves the
    previous layer's GPU work finished) or the prefill ends. Any failure,
    capacity miss, busy half or working-set pressure makes ``stage`` return
    None and the caller uses the existing read path for that layer.

Memory
    The ring adds nothing beyond the pool. ``acquire`` refuses unless the pool,
    MLX's active memory and a transient estimate fit under Metal's recommended
    working set with a margin, and ``stage`` re-checks the live active memory
    every layer and disables the ring for the rest of the prefill if it does
    not.

Everything is off unless ``FLASHNEXT_PREFILL_RING_EXPERTS`` is a positive
number and the expert pool is enabled.
"""
from __future__ import annotations

import contextlib
import os
import time
from collections import namedtuple
from concurrent.futures import wait

from .pool_ring import (
    HazardError, PoolRing, RingError, fits_working_set, plan_ring,
)

FLAG_EXPERTS = "FLASHNEXT_PREFILL_RING_EXPERTS"
FLAG_MIN_TOKENS = "FLASHNEXT_PREFILL_RING_MIN_TOKENS"
FLAG_MARGIN_GB = "FLASHNEXT_PREFILL_RING_MARGIN_GB"
FLAG_TRANSIENT_GB = "FLASHNEXT_PREFILL_RING_TRANSIENT_GB"

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
KINDS = ("weight", "scales", "biases")
CLOSE_TIMEOUT = 30.0

PartPlan = namedtuple("PartPlan", "name dtype tail row_bytes")

# Why the last acquire declined, for diagnostics and tests.
LAST_REASON = [None]


class RingUnavailable(RuntimeError):
    """The checkpoint layout cannot use the ring."""


def _number(name: str, default: str) -> float:
    raw = os.environ.get(name, default) or default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


def configured_experts() -> int:
    """Experts per ring half requested by the environment (0 disables)."""
    return max(0, int(_number(FLAG_EXPERTS, "0")))


def enabled() -> bool:
    return configured_experts() > 0


def min_tokens() -> int:
    return max(0, int(_number(FLAG_MIN_TOKENS, "512")))


def margin_bytes() -> int:
    return int(max(0.0, _number(FLAG_MARGIN_GB, "0.5")) * 1e9)


def transient_bytes() -> int:
    return int(max(0.0, _number(FLAG_TRANSIENT_GB, "2.0")) * 1e9)


def plan_parts(store, switch_prefix: str):
    """The nine parts of one layer's experts, in ring order."""
    plans = []
    for projection in PROJECTIONS:
        for kind in KINDS:
            ref = store.refs.get(f"{switch_prefix}.{projection}.{kind}")
            if ref is None:
                raise RingUnavailable(f"missing tensor {projection}.{kind}")
            if ref.dtype not in ("U32", "BF16"):
                raise RingUnavailable(f"unsupported dtype {ref.dtype}")
            plans.append(PartPlan(
                f"{projection}.{kind}", ref.dtype, tuple(ref.shape[1:]), ref.row_bytes,
            ))
    return tuple(plans)


def fence() -> None:
    """Wait until every GPU job queued before this call has finished.

    ``mx.synchronize`` is avoided on purpose: it keeps the GIL while it waits,
    and MLX's completion handler can need the GIL to free a numpy-backed
    operand, which deadlocks. ``mx.eval`` releases the GIL, and a job queued
    behind earlier work on the same stream cannot finish before it.
    """
    import mlx.core as mx

    mx.eval(mx.zeros((1,)) + 1)


def metal_working_set_limit():
    """Metal's recommended working set in bytes, or None when unknown."""
    import mlx.core as mx

    reader = getattr(mx, "device_info", None)
    if reader is None:
        reader = getattr(getattr(mx, "metal", None), "device_info", None)
    if reader is None:
        return None
    try:
        return int(reader().get("max_recommended_working_set_size", 0)) or None
    except Exception:
        return None


def _read_rows(lease, view, fd, base, file_start, row_bytes, rows) -> None:
    """Read rows of one part into their ring slots. Runs on a worker thread."""
    with lease.writing():
        for slot, expert in rows:
            dest = view[base + slot * row_bytes: base + (slot + 1) * row_bytes]
            try:
                got = os.preadv(fd, [dest], file_start + expert * row_bytes)
            finally:
                dest.release()
            if got != row_bytes:
                raise OSError(f"short expert read: {got} of {row_bytes} bytes")


class PrefillRing:
    """Reads one layer's routed experts into the ring and hands back views."""

    def __init__(
        self, pool, ring, plans, submit, resolve, keepwarm=None,
        active_memory=None, limit_bytes=None, margin=0, pool_bytes=0,
        synchronize=None,
    ):
        self.pool = pool
        self.ring = ring
        self.plans = tuple(plans)
        self._submit = submit
        self._resolve = resolve
        self._keepwarm = keepwarm
        self._active_memory = active_memory
        self._limit = limit_bytes
        self._margin = int(margin)
        self._pool_bytes = int(pool_bytes)
        self._synchronize = synchronize
        self._pool_view = pool._view
        self._buf32 = None
        self._buf16 = None
        self.pending = None
        self.disabled = None
        self.last_error = None
        self._stats = {
            "staged_layers": 0, "staged_experts": 0, "staged_bytes": 0,
            "wait_seconds": 0.0, "fallback_capacity": 0, "fallback_busy": 0,
            "fallback_error": 0, "fallback_disabled": 0, "fallback_memory": 0,
        }

    def stats(self) -> dict:
        snapshot = dict(self._stats)
        snapshot["disabled"] = self.disabled
        snapshot["half_experts"] = self.ring.half_experts
        snapshot["ring_bytes"] = self.ring.ring_bytes
        snapshot["last_error"] = self.last_error
        return snapshot

    # -- one layer ----------------------------------------------------------

    def stage(self, store, switch_prefix: str, wanted):
        """Zero-copy ``[(weight, scales, biases)] x 3`` for ``wanted``, or None.

        None always means "use the existing read path for this layer".
        """
        self._retire()
        if self.disabled:
            self._stats["fallback_disabled"] += 1
            return None
        count = len(wanted)
        if count == 0 or count > self.ring.half_experts:
            self._stats["fallback_capacity"] += 1
            return None
        if self._over_limit():
            self.disabled = "working set"
            self._stats["fallback_memory"] += 1
            return None
        lease = self.ring.begin_fill()
        if lease is None:
            self._stats["fallback_busy"] += 1
            return None
        futures = []
        began = time.perf_counter()
        try:
            futures = self._queue_reads(lease, store, switch_prefix, wanted)
            if self._keepwarm is not None:
                self._keepwarm(futures)
            for future in futures:
                self._resolve(future, None)
            lease.mark_filled(timeout=CLOSE_TIMEOUT)
            lease.begin_compute()
            weights = self._views(lease, count)
        except BaseException as error:
            if futures:
                wait(futures)
            lease.release()
            if isinstance(error, Exception):
                self._stats["fallback_error"] += 1
                self.last_error = repr(error)
                return None
            raise
        self._stats["wait_seconds"] += time.perf_counter() - began
        self._stats["staged_layers"] += 1
        self._stats["staged_experts"] += count
        self._stats["staged_bytes"] += count * self.ring.expert_bytes
        self.pending = lease
        return weights

    def _over_limit(self) -> bool:
        if self._limit is None or self._active_memory is None:
            return False
        return (
            self._active_memory() + self._pool_bytes + self._margin > self._limit
        )

    def _queue_reads(self, lease, store, switch_prefix, wanted):
        rows = list(enumerate(wanted))                  # (ring row, expert)
        chunk = max(1, int(getattr(store, "_pread_chunk", 2)))
        futures = []
        for plan in self.plans:
            ref = store.refs[f"{switch_prefix}.{plan.name}"]
            fd = self.pool._fd(ref.shard)               # F_NOCACHE, read-ahead off
            base, _ = lease.row_range(plan.name, 0, 1)
            for start in range(0, len(rows), chunk):
                futures.append(self._submit(
                    _read_rows, lease, self._pool_view, fd, base, ref.start,
                    plan.row_bytes, rows[start:start + chunk],
                ))
        return futures

    def _views(self, lease, count: int):
        import mlx.core as mx

        if self._buf32 is None:
            self._buf32 = self.pool.buffer_mx
            self._buf16 = self._buf32.view(mx.bfloat16)
        arrays = {}
        for plan in self.plans:
            offset, _ = lease.row_range(plan.name, 0, 1)
            elements = count
            for dimension in plan.tail:
                elements *= dimension
            if plan.dtype == "U32":
                source, first = self._buf32, offset // 4
            else:
                source, first = self._buf16, offset // 2
            arrays[plan.name] = source[first:first + elements].reshape(
                (count, *plan.tail)
            )
        return [
            tuple(arrays[f"{projection}.{kind}"] for kind in KINDS)
            for projection in PROJECTIONS
        ]

    # -- lifetime -----------------------------------------------------------

    def _retire(self) -> None:
        """Release the previous layer's half. Call only after its GPU work
        is known to have finished."""
        lease, self.pending = self.pending, None
        if lease is None:
            return
        try:
            lease.end_compute()
        except HazardError:
            pass
        lease.release()

    def finish(self) -> None:
        self._retire()

    def abort(self) -> None:
        """Failure path: wait for the GPU, then release."""
        if self._synchronize is not None:
            try:
                self._synchronize()
            except Exception:
                pass
        self._retire()

    def close(self) -> bool:
        return self.ring.close(timeout=CLOSE_TIMEOUT)


def acquire(language, tokens: int):
    """Lease the ring for one prefill, or None (the reason is in LAST_REASON)."""
    LAST_REASON[0] = None
    requested = configured_experts()
    if requested <= 0:
        LAST_REASON[0] = "disabled"
        return None
    if tokens < min_tokens():
        LAST_REASON[0] = "short prompt"
        return None
    import mlx.core as mx

    from . import expert_cache, expert_pool

    if not expert_pool.enabled():
        LAST_REASON[0] = "expert pool is off"
        return None
    layers = []
    body = getattr(language, "model", language)
    for layer in getattr(body, "layers", ()):
        glu = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if isinstance(glu, expert_cache.StreamingSwitchGLU):
            layers.append(glu)
    if not layers:
        LAST_REASON[0] = "no streamed layers"
        return None
    first = layers[0]
    store = first.gate_proj.cache.store

    def prefix_of(glu):
        return glu.gate_proj.cache.prefix.rsplit(".", 1)[0]

    try:
        plans = plan_parts(store, prefix_of(first))
        for glu in layers[1:]:
            if plan_parts(store, prefix_of(glu)) != plans:
                raise RingUnavailable("layers differ in expert layout")
    except RingUnavailable as error:
        LAST_REASON[0] = str(error)
        return None
    pool = expert_pool.for_store(store, first.gate_proj.group_size)
    if pool is None:
        LAST_REASON[0] = "no pool"
        return None
    expert_bytes = sum(plan.row_bytes for plan in plans)
    half_experts = plan_ring(
        pool.slots, pool.stride, expert_bytes,
        min(requested, first.gate_proj.num_experts),
        max_pool_fraction=0.5, halves=1,
    )
    if half_experts <= 0:
        LAST_REASON[0] = "pool too small"
        return None
    limit = metal_working_set_limit()
    pool_bytes = len(pool._mm)
    if limit is None:
        LAST_REASON[0] = "working-set limit unknown"
        return None
    active = mx.get_active_memory()
    if not fits_working_set(
        pool_bytes, active + transient_bytes(), limit, margin_bytes()
    ):
        LAST_REASON[0] = (
            f"working set: pool {pool_bytes / 1e9:.2f} GB + active "
            f"{active / 1e9:.2f} GB + transient {transient_bytes() / 1e9:.2f} GB "
            f"+ margin {margin_bytes() / 1e9:.2f} GB exceeds limit "
            f"{limit / 1e9:.2f} GB"
        )
        return None
    try:
        ring = PoolRing(
            pool, [(plan.name, plan.row_bytes) for plan in plans], half_experts,
            header=expert_pool.HEADER, halves=1,
        )
    except RingError as error:
        LAST_REASON[0] = f"lease refused: {error}"
        return None
    keepwarm = expert_cache._keep_gpu_warm_until if expert_cache._KEEPWARM[0] else None
    state = PrefillRing(
        pool, ring, plans, expert_cache._submit_read, expert_cache._resolve_future,
        keepwarm=keepwarm, active_memory=mx.get_active_memory,
        limit_bytes=limit, margin=margin_bytes(), pool_bytes=pool_bytes,
        synchronize=fence,
    )
    expert_cache.set_prefill_ring(state)
    return state


def release(state, failed: bool = False) -> None:
    """Return the ring. Safe on every exit path, including cancellation."""
    if state is None:
        return
    from . import expert_cache

    expert_cache.set_prefill_ring(None)
    try:
        if failed:
            state.abort()
        else:
            state.finish()
    finally:
        state.close()


@contextlib.contextmanager
def session(language, tokens: int):
    """Hold the ring for one prefill. Yields the ring or None."""
    state = acquire(language, tokens) if enabled() else None
    failed = False
    try:
        yield state
    except BaseException:
        failed = True
        raise
    finally:
        release(state, failed)
