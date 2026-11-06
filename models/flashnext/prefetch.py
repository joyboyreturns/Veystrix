"""Bounded asynchronous expert prefetch for prefill (opt-in, not wired in).

Prefill alternates strictly between waiting for one layer's expert reads and
running that layer's GPU work, so the SSD and the GPU never work together
(`docs/flashnext/prefetch-design.md`). This module is the scheduler half of an
inter-layer overlap: while layer L computes, the experts layer L+1 is expected
to route to are read on a small thread pool into staging buffers. Nothing here
changes what the model computes. An expert that was not staged is read on
demand afterwards, exactly as today, so a wrong prediction costs bytes and
never correctness.

The module imports neither MLX nor NumPy. The reader, the allocator and the
executor are injected, which keeps the scheduler testable without a model and
lets the runtime supply its own I/O path later.

Memory rules, for a 16 GB machine that already runs with about 100 MB free:

* A hard byte budget covers every staging buffer from allocation until it is
  actually freed. A request that does not fit is rejected, never queued, and
  never blocks the caller.
* The default allocator uses anonymous ``mmap``. Pages are committed only as
  reads land, and ``close()`` returns them immediately.
* Cancelled and released tickets keep their bytes reserved until every worker
  that could still write into the buffer has finished, so the accounted peak
  is the real peak.
* ``safe_budget`` clamps a configured budget to the memory that is actually
  free at the moment of use.

Disabled by default: ``FLASHNEXT_PREFILL_PREFETCH_GB`` unset or 0 means
``enabled()`` is False and nothing constructs a scheduler.
"""
from __future__ import annotations

import mmap
import os
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, wait

SIZE_FLAG = "FLASHNEXT_PREFILL_PREFETCH_GB"

Part = namedtuple("Part", "name row_bytes")


def configured_bytes() -> int:
    """Budget requested by the environment, in bytes (0 disables)."""
    raw = os.environ.get(SIZE_FLAG, "0") or "0"
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{SIZE_FLAG} must be a number of GB") from error
    return max(0, int(value * 1024 ** 3))


def enabled() -> bool:
    return configured_bytes() > 0


def safe_budget(configured: int, free_bytes: int, floor_bytes: int) -> int:
    """Clamp a budget so at least ``floor_bytes`` of memory stays free."""
    return max(0, min(int(configured), int(free_bytes) - int(floor_bytes)))


class PrefetchLayout:
    """The per-expert parts a ticket stages, in file order."""

    def __init__(self, parts):
        self.parts = tuple(Part(name, int(size)) for name, size in parts)
        if not self.parts or any(part.row_bytes <= 0 for part in self.parts):
            raise ValueError("a layout needs parts with positive row sizes")
        self.expert_bytes = sum(part.row_bytes for part in self.parts)


def count_routes(routed, n_experts: int):
    """Per-expert token counts from a flat list of routed expert ids."""
    counts = [0] * n_experts
    for expert in routed:
        if 0 <= expert < n_experts:
            counts[expert] += 1
    return counts


def select_experts(
    counts,
    n_experts: int,
    tokens: int,
    max_experts: int,
    dense_min_tokens: int = 2048,
    min_count: int = 1,
):
    """Choose which experts to stage for one layer, ascending by id.

    Above ``dense_min_tokens`` a prompt touches most experts, so every expert
    is staged and no prediction is needed. Below it the experts with at least
    ``min_count`` predicted tokens are staged, the most-used first when the
    cap bites. ``counts`` may be None when nothing is predicted. The result
    never exceeds ``max_experts``, which the caller derives from the budget.
    """
    cap = max(0, min(int(max_experts), int(n_experts)))
    if cap == 0:
        return []
    if tokens >= dense_min_tokens:
        if cap == n_experts:
            return list(range(n_experts))
        if counts is None:
            return list(range(cap))
        ranked = sorted(range(n_experts), key=lambda e: (-counts[e], e))
        return sorted(ranked[:cap])
    if counts is None:
        return []
    chosen = [e for e in range(n_experts) if counts[e] >= min_count]
    if len(chosen) > cap:
        chosen.sort(key=lambda e: (-counts[e], e))
        chosen = chosen[:cap]
    return sorted(chosen)


def mmap_allocator(nbytes: int):
    """Anonymous, lazily committed, page-aligned buffer."""
    return mmap.mmap(-1, nbytes)


def mmap_free(buffer) -> bool:
    """Close a buffer; False while an exported view still pins it."""
    try:
        buffer.close()
    except BufferError:
        return False
    return True


class _Ticket:
    """One layer's staging buffers and the reads filling them."""

    def __init__(self, layer, experts, layout, chunk, nbytes):
        self.layer = layer
        self.experts = experts
        self.index = {expert: row for row, expert in enumerate(experts)}
        self.layout = layout
        self.chunk = chunk
        self.nbytes = nbytes
        self.buffers = []
        self.futures = []          # futures[part][chunk]
        self.errors = []
        self.cancelled = threading.Event()
        self.pending = 0
        self.claims = 0
        self.retired = False
        self.freed = False
        self.lock = threading.Lock()

    def chunk_of(self, row: int) -> int:
        return row // self.chunk


class Claim:
    """What a consumer may use from a ticket, plus what it must still read.

    ``staged`` maps expert id to its row in every part buffer. ``missing``
    lists the experts to read on demand, in the order they were asked for.
    The buffers stay valid until ``release``.
    """

    def __init__(self, scheduler, ticket, staged, missing):
        self._scheduler = scheduler
        self._ticket = ticket
        self.staged = staged
        self.missing = missing
        self._views = []
        self._released = False

    def rows(self, part_name: str):
        """View of one staged part: ``len(ticket.experts)`` consecutive rows."""
        if self._released or self._ticket is None:
            raise ValueError("claim is released")
        for index, part in enumerate(self._ticket.layout.parts):
            if part.name == part_name:
                view = memoryview(self._ticket.buffers[index])
                part_view = view[: len(self._ticket.experts) * part.row_bytes]
                self._views.extend((view, part_view))
                return part_view
        raise KeyError(part_name)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for view in reversed(self._views):
            try:
                view.release()
            except (ValueError, BufferError):
                pass
        self._views.clear()
        if self._ticket is not None:
            self._scheduler._release_claim(self._ticket)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()


class ExpertPrefetcher:
    """Bounded, cancellable, asynchronous expert staging.

    ``reader(part_name, experts, out)`` fills ``out`` (a writable memoryview
    of ``len(experts)`` consecutive rows) with those experts' rows of one
    part. It runs on worker threads and must be safe to call concurrently on
    disjoint slices, like ``SafeTensorStore.rows_into``.
    """

    def __init__(
        self,
        layout: PrefetchLayout,
        budget_bytes: int,
        reader,
        executor=None,
        workers: int = 4,
        chunk: int = 2,
        max_inflight: int = 1,
        allocator=mmap_allocator,
        free=mmap_free,
    ):
        if chunk < 1 or max_inflight < 1:
            raise ValueError("chunk and max_inflight must be at least 1")
        self.layout = layout
        self.budget = max(0, int(budget_bytes))
        self._reader = reader
        self._chunk = int(chunk)
        self._max_inflight = int(max_inflight)
        self._allocator = allocator
        self._free = free
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="flashnext-prefetch"
        )
        self._lock = threading.Lock()
        self._unclaimed = {}       # layer -> ticket, not yet claimed
        self._live = []            # every ticket whose bytes are reserved
        self._reserved = 0
        self._peak = 0
        self._closed = False
        self._stats = {
            "submitted": 0, "rejected": {}, "experts_requested": 0,
            "experts_staged": 0, "experts_missing": 0, "experts_wasted": 0,
            "bytes_read": 0, "read_errors": 0, "cancelled_tickets": 0,
            "claims": 0, "claim_timeouts": 0,
        }

    # -- accounting ---------------------------------------------------------

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved

    def max_experts(self) -> int:
        """Largest ticket the whole budget could hold."""
        return self.budget // self.layout.expert_bytes

    def stats(self) -> dict:
        with self._lock:
            self._reap_locked()
            snapshot = dict(self._stats)
            snapshot["rejected"] = dict(self._stats["rejected"])
            snapshot["reserved_bytes"] = self._reserved
            snapshot["peak_reserved_bytes"] = self._peak
            snapshot["budget_bytes"] = self.budget
            snapshot["live_tickets"] = len(self._live)
            return snapshot

    def _reject(self, reason: str):
        rejected = self._stats["rejected"]
        rejected[reason] = rejected.get(reason, 0) + 1
        return None

    # -- submission ---------------------------------------------------------

    def submit(self, layer: int, experts):
        """Start staging ``experts`` of ``layer``; None when refused.

        Refusal is normal and silent: the caller reads on demand instead.
        Reasons are counted in ``stats()['rejected']``.
        """
        experts = sorted(set(int(e) for e in experts))
        nbytes = len(experts) * self.layout.expert_bytes
        with self._lock:
            self._reap_locked()
            if self._closed:
                return self._reject("closed")
            if not experts:
                return self._reject("empty")
            if self.budget <= 0 or nbytes > self.budget:
                return self._reject("budget")
            if layer in self._unclaimed:
                return self._reject("duplicate")
            if len(self._unclaimed) >= self._max_inflight:
                return self._reject("inflight")
            if self._reserved + nbytes > self.budget:
                return self._reject("budget")
            self._reserved += nbytes
            self._peak = max(self._peak, self._reserved)
            ticket = _Ticket(layer, experts, self.layout, self._chunk, nbytes)
            self._unclaimed[layer] = ticket
            self._live.append(ticket)
            self._stats["submitted"] += 1
            self._stats["experts_requested"] += len(experts)
        try:
            for part in self.layout.parts:
                ticket.buffers.append(
                    self._allocator(len(experts) * part.row_bytes)
                )
        except BaseException:
            with self._lock:
                self._unclaimed.pop(layer, None)
                ticket.retired = True
                self._stats["submitted"] -= 1
                self._stats["experts_requested"] -= len(experts)
                self._reject("alloc")
                self._free_locked(ticket)
            return None
        self._launch(ticket)
        return ticket

    def _launch(self, ticket) -> None:
        chunk = ticket.chunk
        total = len(ticket.experts)
        count = -(-total // chunk)
        ticket.pending = len(self.layout.parts) * count
        for index, part in enumerate(self.layout.parts):
            row = []
            for start in range(0, total, chunk):
                stop = min(start + chunk, total)
                try:
                    future = self._executor.submit(
                        self._read_chunk, ticket, index, part, start, stop
                    )
                except BaseException:
                    # A refused submit still owes its pending slot; the failed
                    # future's callback pays it, and the ticket stops reading.
                    ticket.cancelled.set()
                    future = _failed_future()
                future.add_done_callback(lambda _f, t=ticket: self._finish(t))
                row.append(future)
            ticket.futures.append(row)

    def _read_chunk(self, ticket, index, part, start, stop) -> None:
        if ticket.cancelled.is_set():
            return
        view = memoryview(ticket.buffers[index])
        try:
            out = view[start * part.row_bytes: stop * part.row_bytes]
            try:
                self._reader(part.name, ticket.experts[start:stop], out)
            finally:
                out.release()
        except BaseException as error:
            with ticket.lock:
                ticket.errors.append((part.name, start, stop, error))
            with self._lock:
                self._stats["read_errors"] += 1
            raise
        finally:
            view.release()
        with self._lock:
            self._stats["bytes_read"] += (stop - start) * part.row_bytes

    def _finish(self, ticket) -> None:
        with self._lock:
            ticket.pending -= 1
            if ticket.pending <= 0:
                self._free_locked(ticket)

    # -- consumption --------------------------------------------------------

    def claim(self, layer: int, needed, timeout=None) -> Claim:
        """Take what is staged for ``layer`` and list the rest as missing.

        Waits only for the read chunks that cover ``needed`` experts, and
        cancels chunks nobody needs that have not started. A chunk that
        failed, was cancelled or missed ``timeout`` counts as missing, so
        this never raises for an I/O problem.
        """
        needed = list(dict.fromkeys(int(e) for e in needed))
        with self._lock:
            ticket = self._unclaimed.pop(layer, None)
            self._stats["claims"] += 1
            if ticket is None or ticket.retired:
                self._stats["experts_missing"] += len(needed)
                return Claim(self, None, {}, needed)
            ticket.claims += 1
        wanted_chunks = {
            ticket.chunk_of(ticket.index[e]) for e in needed if e in ticket.index
        }
        for row in ticket.futures:
            for number, future in enumerate(row):
                if number not in wanted_chunks:
                    future.cancel()
        waiting = [
            row[number] for row in ticket.futures for number in wanted_chunks
        ]
        if waiting:
            _done, not_done = wait(waiting, timeout=timeout)
            if not_done:
                with self._lock:
                    self._stats["claim_timeouts"] += 1
                for future in not_done:
                    future.cancel()
        staged, missing = {}, []
        for expert in needed:
            row = ticket.index.get(expert)
            if row is not None and self._chunk_ok(ticket, ticket.chunk_of(row)):
                staged[expert] = row
            else:
                missing.append(expert)
        with self._lock:
            self._stats["experts_staged"] += len(staged)
            self._stats["experts_missing"] += len(missing)
            self._stats["experts_wasted"] += len(ticket.experts) - len(staged)
        return Claim(self, ticket, staged, missing)

    @staticmethod
    def _chunk_ok(ticket, number: int) -> bool:
        for row in ticket.futures:
            future = row[number]
            if (
                not future.done() or future.cancelled()
                or future.exception() is not None
            ):
                return False
        return True

    def _release_claim(self, ticket) -> None:
        with self._lock:
            ticket.claims -= 1
            ticket.retired = True
            self._free_locked(ticket)

    # -- cancellation and shutdown ------------------------------------------

    def cancel(self, layer=None) -> int:
        """Cancel unclaimed tickets (all, or one layer). Returns the count.

        Reads that have not started are dropped. Reads in flight finish into
        their buffer, and the bytes stay reserved until they do.
        """
        with self._lock:
            if layer is None:
                tickets = list(self._unclaimed.values())
                self._unclaimed.clear()
            else:
                found = self._unclaimed.pop(layer, None)
                tickets = [found] if found is not None else []
            for ticket in tickets:
                ticket.cancelled.set()
                ticket.retired = True
                self._stats["cancelled_tickets"] += 1
        for ticket in tickets:
            for row in ticket.futures:
                for future in row:
                    future.cancel()
        with self._lock:
            for ticket in tickets:
                self._free_locked(ticket)
        return len(tickets)

    def drain(self, timeout=None) -> bool:
        """Wait until every worker is idle and every buffer is returned."""
        with self._lock:
            tickets = list(self._live)
        waiting = [
            future for ticket in tickets
            for row in ticket.futures for future in row
        ]
        if waiting:
            wait(waiting, timeout=timeout)
        with self._lock:
            self._reap_locked()
            return not self._live

    def close(self, timeout=None) -> bool:
        """Cancel everything, wait, and stop the owned executor."""
        with self._lock:
            self._closed = True
        self.cancel()
        clean = self.drain(timeout)
        if self._owns_executor:
            self._executor.shutdown(wait=False)
        return clean

    # -- buffer lifetime ----------------------------------------------------

    def _free_locked(self, ticket) -> None:
        """Return a ticket's bytes once nothing can touch its buffers."""
        if ticket.freed or not ticket.retired:
            return
        if ticket.pending > 0 or ticket.claims > 0:
            return
        for number, buffer in enumerate(ticket.buffers):
            if buffer is None:
                continue
            if not self._free(buffer):
                return          # still pinned by a view; retried by _reap
            ticket.buffers[number] = None
        ticket.buffers = []
        ticket.freed = True
        self._reserved -= ticket.nbytes
        if ticket in self._live:
            self._live.remove(ticket)

    def _reap_locked(self) -> None:
        for ticket in list(self._live):
            self._free_locked(ticket)


def _failed_future():
    from concurrent.futures import Future

    future = Future()
    future.set_exception(RuntimeError("prefetch executor refused the read"))
    return future
