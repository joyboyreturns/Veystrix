"""Expert prefetch scheduler: selection, misses, async reads, cancel, memory.

Pure Python. No MLX, NumPy, checkpoint or drive: the reader is a mock.
"""
from __future__ import annotations

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from models.flashnext import prefetch
from models.flashnext.prefetch import (
    ExpertPrefetcher,
    PrefetchLayout,
    count_routes,
    safe_budget,
    select_experts,
)

LAYOUT = PrefetchLayout([("w", 8), ("s", 4)])        # 12 bytes per expert
WAIT = 5.0


def pattern(part: str, expert: int, size: int) -> bytes:
    return bytes((ord(part[0]) + expert + i) % 251 for i in range(size))


def plain_allocator(nbytes):
    return bytearray(nbytes)


def plain_free(_buffer):
    return True


def filling_reader(calls=None):
    sizes = {part.name: part.row_bytes for part in LAYOUT.parts}

    def read(part, experts, out):
        if calls is not None:
            calls.append((part, tuple(experts)))
        size = sizes[part]
        for position, expert in enumerate(experts):
            out[position * size:(position + 1) * size] = pattern(part, expert, size)

    return read


def make(budget=12 * 16, reader=None, **options):
    options.setdefault("executor", ThreadPoolExecutor(max_workers=2))
    options.setdefault("allocator", plain_allocator)
    options.setdefault("free", plain_free)
    return ExpertPrefetcher(LAYOUT, budget, reader or filling_reader(), **options)


class SelectionTests(unittest.TestCase):
    def test_dense_mode_takes_every_expert_without_a_prediction(self):
        self.assertEqual(
            select_experts(None, 8, tokens=4096, max_experts=8), list(range(8))
        )

    def test_dense_mode_under_a_cap_keeps_the_most_used(self):
        counts = [0, 9, 1, 7, 0, 8, 2, 0]
        self.assertEqual(
            select_experts(counts, 8, tokens=4096, max_experts=3), [1, 3, 5]
        )

    def test_dense_mode_under_a_cap_without_counts_is_deterministic(self):
        self.assertEqual(
            select_experts(None, 8, tokens=4096, max_experts=3), [0, 1, 2]
        )

    def test_sparse_mode_selects_predicted_experts_ascending(self):
        counts = [0, 3, 0, 1, 0, 2, 0, 0]
        self.assertEqual(
            select_experts(counts, 8, tokens=100, max_experts=8), [1, 3, 5]
        )

    def test_sparse_mode_cap_drops_the_least_used_and_ties_by_id(self):
        counts = [5, 5, 5, 1, 9, 0, 0, 0]
        self.assertEqual(
            select_experts(counts, 8, tokens=100, max_experts=3), [0, 1, 4]
        )

    def test_sparse_mode_min_count_and_missing_prediction(self):
        counts = [1, 2, 3, 0]
        self.assertEqual(
            select_experts(counts, 4, tokens=10, max_experts=4, min_count=2), [1, 2]
        )
        self.assertEqual(select_experts(None, 4, tokens=10, max_experts=4), [])

    def test_zero_cap_selects_nothing(self):
        self.assertEqual(select_experts([1] * 4, 4, tokens=9999, max_experts=0), [])

    def test_count_routes_ignores_out_of_range_ids(self):
        self.assertEqual(count_routes([0, 2, 2, 7, -1, 3], 4), [1, 0, 2, 1])


class BudgetHelperTests(unittest.TestCase):
    def test_safe_budget_leaves_the_floor_free(self):
        self.assertEqual(safe_budget(6_000, free_bytes=5_000, floor_bytes=1_500), 3_500)
        self.assertEqual(safe_budget(1_000, free_bytes=9_000, floor_bytes=1_500), 1_000)
        self.assertEqual(safe_budget(6_000, free_bytes=100, floor_bytes=1_500), 0)

    def test_disabled_unless_the_flag_is_set(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(prefetch.SIZE_FLAG, None)
            self.assertFalse(prefetch.enabled())
            self.assertEqual(prefetch.configured_bytes(), 0)
        with mock.patch.dict(os.environ, {prefetch.SIZE_FLAG: "0"}):
            self.assertFalse(prefetch.enabled())
        with mock.patch.dict(os.environ, {prefetch.SIZE_FLAG: "1.5"}):
            self.assertTrue(prefetch.enabled())
            self.assertEqual(prefetch.configured_bytes(), int(1.5 * 1024 ** 3))
        with mock.patch.dict(os.environ, {prefetch.SIZE_FLAG: "lots"}):
            with self.assertRaises(ValueError):
                prefetch.configured_bytes()

    def test_layout_rejects_empty_or_zero_parts(self):
        with self.assertRaises(ValueError):
            PrefetchLayout([])
        with self.assertRaises(ValueError):
            PrefetchLayout([("w", 0)])


class StagingTests(unittest.TestCase):
    def test_staged_rows_hold_each_experts_own_bytes(self):
        scheduler = make(chunk=2)
        try:
            self.assertIsNotNone(scheduler.submit(3, [9, 2, 5, 2]))
            with scheduler.claim(3, [5, 2, 9]) as claim:
                self.assertEqual(claim.missing, [])
                self.assertEqual(claim.staged, {2: 0, 5: 1, 9: 2})
                for part in LAYOUT.parts:
                    rows = claim.rows(part.name)
                    for expert, row in claim.staged.items():
                        got = bytes(rows[row * part.row_bytes:(row + 1) * part.row_bytes])
                        self.assertEqual(got, pattern(part.name, expert, part.row_bytes))
                    del rows
        finally:
            scheduler.close()

    def test_unpredicted_experts_are_misses_in_request_order(self):
        scheduler = make()
        try:
            scheduler.submit(1, [1, 2, 3])
            with scheduler.claim(1, [7, 2, 7, 1, 9]) as claim:
                self.assertEqual(sorted(claim.staged), [1, 2])
                self.assertEqual(claim.missing, [7, 9])
        finally:
            scheduler.close()

    def test_claim_without_a_ticket_is_all_misses(self):
        scheduler = make()
        try:
            with scheduler.claim(4, [3, 1, 3]) as claim:
                self.assertEqual(claim.staged, {})
                self.assertEqual(claim.missing, [3, 1])
                with self.assertRaises(ValueError):
                    claim.rows("w")
            self.assertEqual(scheduler.stats()["experts_missing"], 2)
        finally:
            scheduler.close()

    def test_unneeded_experts_are_counted_as_waste(self):
        scheduler = make()
        try:
            scheduler.submit(1, [0, 1, 2, 3])
            with scheduler.claim(1, [0, 1]):
                pass
            stats = scheduler.stats()
            self.assertEqual(stats["experts_staged"], 2)
            self.assertEqual(stats["experts_wasted"], 2)
        finally:
            scheduler.close()

    def test_failed_read_becomes_a_miss_not_an_error(self):
        def reader(part, experts, out):
            if 4 in experts:
                raise OSError("short pread")
            filling_reader()(part, experts, out)

        scheduler = make(reader=reader, chunk=1)
        try:
            scheduler.submit(2, [1, 4, 6])
            with scheduler.claim(2, [1, 4, 6]) as claim:
                self.assertEqual(sorted(claim.staged), [1, 6])
                self.assertEqual(claim.missing, [4])
            self.assertGreaterEqual(scheduler.stats()["read_errors"], 1)
        finally:
            scheduler.close()

    def test_default_mmap_allocator_round_trip_and_release(self):
        scheduler = ExpertPrefetcher(
            LAYOUT, 12 * 8, filling_reader(),
            executor=ThreadPoolExecutor(max_workers=2),
        )
        try:
            scheduler.submit(0, [3, 4])
            with scheduler.claim(0, [3, 4]) as claim:
                rows = claim.rows("w")
                self.assertEqual(bytes(rows[8:16]), pattern("w", 4, 8))
                del rows
            self.assertTrue(scheduler.drain(WAIT))
            self.assertEqual(scheduler.reserved_bytes, 0)
        finally:
            scheduler.close()


class AsyncTests(unittest.TestCase):
    def test_submit_returns_before_the_reads_finish(self):
        release = threading.Event()
        started = threading.Event()

        def reader(part, experts, out):
            started.set()
            release.wait(WAIT)

        scheduler = make(reader=reader)
        try:
            ticket = scheduler.submit(1, [0, 1])
            self.assertIsNotNone(ticket)
            self.assertTrue(started.wait(WAIT))
            self.assertFalse(all(f.done() for row in ticket.futures for f in row))
            release.set()
            self.assertTrue(scheduler.claim(1, [0, 1]).release() is None)
        finally:
            release.set()
            scheduler.close()

    def test_claim_waits_only_for_the_chunks_it_needs(self):
        gate = threading.Event()

        def reader(part, experts, out):
            if 5 in experts:
                gate.wait(WAIT)
            else:
                filling_reader()(part, experts, out)

        scheduler = make(reader=reader, chunk=1, executor=ThreadPoolExecutor(4))
        try:
            scheduler.submit(1, [0, 1, 5])
            with scheduler.claim(1, [0, 1], timeout=WAIT) as claim:
                self.assertEqual(sorted(claim.staged), [0, 1])
                self.assertEqual(claim.missing, [])
                self.assertGreater(scheduler.reserved_bytes, 0)
        finally:
            gate.set()
            scheduler.close()

    def test_a_late_chunk_times_out_into_a_miss(self):
        gate = threading.Event()

        def reader(part, experts, out):
            gate.wait(WAIT)

        scheduler = make(reader=reader, chunk=1)
        try:
            scheduler.submit(1, [2])
            claim = scheduler.claim(1, [2], timeout=0.05)
            self.assertEqual(claim.staged, {})
            self.assertEqual(claim.missing, [2])
            self.assertEqual(scheduler.stats()["claim_timeouts"], 1)
            claim.release()
        finally:
            gate.set()
            scheduler.close()

    def test_runs_with_an_inline_executor(self):
        class Inline:
            def submit(self, function, *args):
                from concurrent.futures import Future

                future = Future()
                try:
                    future.set_result(function(*args))
                except BaseException as error:
                    future.set_exception(error)
                return future

        scheduler = make(executor=Inline())
        scheduler.submit(1, [1, 2])
        with scheduler.claim(1, [1, 2]) as claim:
            self.assertEqual(sorted(claim.staged), [1, 2])
        self.assertTrue(scheduler.drain(WAIT))
        self.assertEqual(scheduler.reserved_bytes, 0)

    def test_executor_that_refuses_work_degrades_to_misses(self):
        class Refusing:
            def submit(self, *args):
                raise RuntimeError("shut down")

        scheduler = make(executor=Refusing())
        scheduler.submit(1, [1, 2])
        with scheduler.claim(1, [1, 2]) as claim:
            self.assertEqual(claim.staged, {})
            self.assertEqual(claim.missing, [1, 2])
        self.assertTrue(scheduler.drain(WAIT))
        self.assertEqual(scheduler.reserved_bytes, 0)


class MemoryLimitTests(unittest.TestCase):
    def test_request_above_budget_is_rejected_without_allocating(self):
        allocated = []

        def allocator(nbytes):
            allocated.append(nbytes)
            return bytearray(nbytes)

        scheduler = make(budget=12 * 3, allocator=allocator)
        try:
            self.assertIsNone(scheduler.submit(1, [0, 1, 2, 3]))
            self.assertEqual(allocated, [])
            self.assertEqual(scheduler.stats()["rejected"], {"budget": 1})
            self.assertEqual(scheduler.max_experts(), 3)
        finally:
            scheduler.close()

    def test_second_ticket_must_fit_beside_the_first(self):
        release = threading.Event()

        def reader(part, experts, out):
            release.wait(WAIT)

        scheduler = make(budget=12 * 4, reader=reader, max_inflight=2)
        try:
            self.assertIsNotNone(scheduler.submit(1, [0, 1, 2]))
            self.assertIsNone(scheduler.submit(2, [3, 4]))
            self.assertIsNotNone(scheduler.submit(2, [3]))
            stats = scheduler.stats()
            self.assertEqual(stats["rejected"], {"budget": 1})
            self.assertLessEqual(stats["peak_reserved_bytes"], 12 * 4)
        finally:
            release.set()
            scheduler.close()

    def test_budget_is_returned_after_claim_release(self):
        scheduler = make(budget=12 * 2)
        try:
            scheduler.submit(1, [0, 1])
            claim = scheduler.claim(1, [0, 1])
            self.assertEqual(scheduler.reserved_bytes, 24)
            self.assertIsNone(scheduler.submit(2, [5]))      # still held
            claim.release()
            self.assertTrue(scheduler.drain(WAIT))
            self.assertEqual(scheduler.reserved_bytes, 0)
            self.assertIsNotNone(scheduler.submit(2, [5]))
        finally:
            scheduler.close()

    def test_zero_budget_never_stages(self):
        scheduler = make(budget=0)
        try:
            self.assertIsNone(scheduler.submit(1, [0]))
            self.assertEqual(scheduler.stats()["rejected"], {"budget": 1})
        finally:
            scheduler.close()

    def test_lookahead_depth_duplicates_empty_and_closed(self):
        release = threading.Event()
        scheduler = make(reader=lambda *a: release.wait(WAIT), max_inflight=1)
        try:
            self.assertIsNotNone(scheduler.submit(1, [0]))
            self.assertIsNone(scheduler.submit(1, [1]))       # same layer
            self.assertIsNone(scheduler.submit(2, [1]))       # depth limit
            self.assertIsNone(scheduler.submit(3, []))
            release.set()
            scheduler.close()
            self.assertIsNone(scheduler.submit(4, [1]))
            self.assertEqual(
                scheduler.stats()["rejected"],
                {"duplicate": 1, "inflight": 1, "empty": 1, "closed": 1},
            )
        finally:
            release.set()
            scheduler.close()

    def test_allocation_failure_is_a_rejection_that_returns_the_bytes(self):
        def allocator(nbytes):
            raise MemoryError

        scheduler = make(allocator=allocator)
        try:
            self.assertIsNone(scheduler.submit(1, [0, 1]))
            stats = scheduler.stats()
            self.assertEqual(stats["rejected"], {"alloc": 1})
            self.assertEqual(stats["reserved_bytes"], 0)
            self.assertEqual(stats["submitted"], 0)
        finally:
            scheduler.close()

    def test_a_pinned_buffer_keeps_its_bytes_reserved_until_it_frees(self):
        pinned = [True]

        def free(_buffer):
            return not pinned[0]

        scheduler = make(free=free)
        try:
            scheduler.submit(1, [0, 1])
            scheduler.claim(1, [0, 1]).release()
            scheduler.drain(WAIT)
            self.assertEqual(scheduler.reserved_bytes, 24)
            pinned[0] = False
            self.assertEqual(scheduler.stats()["reserved_bytes"], 0)
        finally:
            scheduler.close()

    def test_reserved_bytes_never_exceed_the_budget_under_load(self):
        budget = 12 * 6
        observed = []
        scheduler = None

        def reader(part, experts, out):
            observed.append(scheduler.reserved_bytes)
            filling_reader()(part, experts, out)

        scheduler = make(budget=budget, reader=reader, max_inflight=2)
        try:
            for layer in range(60):
                scheduler.submit(layer, [layer % 7, (layer + 1) % 7, (layer + 2) % 7])
                if layer >= 1:
                    with scheduler.claim(layer - 1, [(layer - 1) % 7]):
                        pass
            scheduler.cancel()
            self.assertTrue(scheduler.drain(WAIT))
            stats = scheduler.stats()
            self.assertLessEqual(stats["peak_reserved_bytes"], budget)
            self.assertTrue(observed)
            self.assertLessEqual(max(observed), budget)
            self.assertEqual(stats["reserved_bytes"], 0)
        finally:
            scheduler.close()


class CancellationTests(unittest.TestCase):
    def test_cancel_drops_queued_reads_and_waits_for_running_ones(self):
        started = threading.Event()
        release = threading.Event()
        calls = []
        layout = PrefetchLayout([("w", 4)])

        def reader(part, experts, out):
            calls.append(tuple(experts))
            started.set()
            release.wait(WAIT)

        scheduler = ExpertPrefetcher(
            layout, 4 * 8, reader, executor=ThreadPoolExecutor(max_workers=1),
            chunk=1, allocator=plain_allocator, free=plain_free,
        )
        try:
            self.assertIsNotNone(scheduler.submit(1, [0, 1, 2, 3]))
            self.assertTrue(started.wait(WAIT))
            self.assertEqual(scheduler.cancel(1), 1)
            # One read is still writing: its bytes stay accounted.
            self.assertEqual(scheduler.reserved_bytes, 16)
            self.assertIsNone(scheduler.submit(2, [0, 1, 2, 3, 4, 5]))
            release.set()
            self.assertTrue(scheduler.drain(WAIT))
            self.assertEqual(scheduler.reserved_bytes, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(scheduler.stats()["cancelled_tickets"], 1)
        finally:
            release.set()
            scheduler.close()

    def test_claim_after_cancel_reads_everything_on_demand(self):
        scheduler = make()
        try:
            scheduler.submit(1, [0, 1])
            scheduler.cancel()
            with scheduler.claim(1, [0, 1]) as claim:
                self.assertEqual(claim.staged, {})
                self.assertEqual(claim.missing, [0, 1])
        finally:
            scheduler.close()

    def test_cancel_one_layer_leaves_the_others(self):
        release = threading.Event()
        scheduler = make(reader=lambda *a: release.wait(WAIT), max_inflight=2)
        try:
            scheduler.submit(1, [0])
            scheduler.submit(2, [1])
            self.assertEqual(scheduler.cancel(1), 1)
            self.assertEqual(scheduler.cancel(1), 0)
            release.set()
            with scheduler.claim(2, [1]) as claim:
                self.assertEqual(sorted(claim.staged), [1])
        finally:
            release.set()
            scheduler.close()

    def test_cancel_does_not_touch_a_claim_in_use(self):
        scheduler = make()
        try:
            scheduler.submit(1, [0, 1])
            claim = scheduler.claim(1, [0, 1])
            self.assertEqual(scheduler.cancel(), 0)
            rows = claim.rows("w")
            self.assertEqual(bytes(rows[0:8]), pattern("w", 0, 8))
            del rows
            claim.release()
        finally:
            scheduler.close()

    def test_close_is_clean_and_idempotent(self):
        release = threading.Event()
        scheduler = ExpertPrefetcher(
            LAYOUT, 12 * 8, lambda *a: release.wait(0.05),
            allocator=plain_allocator, free=plain_free,
        )
        scheduler.submit(1, [0, 1])
        self.assertTrue(scheduler.close(WAIT))
        self.assertTrue(scheduler.close(WAIT))
        self.assertEqual(scheduler.reserved_bytes, 0)


if __name__ == "__main__":
    unittest.main()
