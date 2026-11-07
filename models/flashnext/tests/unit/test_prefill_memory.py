"""Opt-in MLX allocator cache cap during prefill, and how prefill_target wires
it with the ring. Real MLX for the limit itself, mocks for the wiring. No
model, no checkpoint, no benchmark."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from models.flashnext import prefill, prefill_memory, prefill_ring

MIB = 1024 * 1024


def churn(count=6, size=16_000_000):
    """Allocate and free buffers so MLX's cache fills."""
    for _ in range(count):
        array = mx.zeros((size,))
        mx.eval(array)
        del array
    mx.eval(mx.zeros(1))


def current_limit():
    """Read the cache limit (MLX only returns it when it is set)."""
    value = mx.set_cache_limit(10**12)
    mx.set_cache_limit(value)
    return value


class FlagTests(unittest.TestCase):
    def test_off_by_default_and_parsed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLASHNEXT_PREFILL_CACHE_LIMIT_MB", None)
            self.assertEqual(prefill_memory.cache_limit_mb(), 0)
            self.assertFalse(prefill_memory.enabled())
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_CACHE_LIMIT_MB": "512"}):
            self.assertEqual(prefill_memory.cache_limit_mb(), 512)
            self.assertTrue(prefill_memory.enabled())

    def test_bad_values_raise(self):
        for value in ("lots", "-1"):
            with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_CACHE_LIMIT_MB": value}):
                with self.assertRaises(ValueError):
                    prefill_memory.cache_limit_mb()


class CacheCapTests(unittest.TestCase):
    def setUp(self):
        self.original = current_limit()
        self.addCleanup(mx.set_cache_limit, self.original)
        mx.clear_cache()

    def test_off_does_not_touch_the_limit(self):
        with mock.patch.object(mx, "set_cache_limit") as setter:
            with prefill_memory.cache_cap(0) as previous:
                self.assertIsNone(previous)
            setter.assert_not_called()

    def test_sets_the_limit_inside_and_restores_it_after(self):
        with prefill_memory.cache_cap(64) as previous:
            self.assertEqual(previous, self.original)
            self.assertEqual(current_limit(), 64 * MIB)
        self.assertEqual(current_limit(), self.original)

    def test_restores_the_limit_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with prefill_memory.cache_cap(64):
                raise RuntimeError("fail")
        self.assertEqual(current_limit(), self.original)

    def test_restores_the_limit_after_a_keyboard_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            with prefill_memory.cache_cap(64):
                raise KeyboardInterrupt()
        self.assertEqual(current_limit(), self.original)

    def test_reads_the_environment_when_no_size_is_given(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_CACHE_LIMIT_MB": "32"}):
            with prefill_memory.cache_cap():
                self.assertEqual(current_limit(), 32 * MIB)
        self.assertEqual(current_limit(), self.original)

    def test_an_oversized_cache_is_trimmed_on_entry(self):
        # MLX enforces the limit at the next allocation, so entering the cap
        # clears at once; otherwise the first layers would run uncapped.
        churn()
        before = mx.get_cache_memory()
        if before <= MIB:
            self.skipTest("this MLX build did not keep freed buffers")
        with prefill_memory.cache_cap(1):
            self.assertLessEqual(mx.get_cache_memory(), MIB)

    def test_a_capped_cache_stays_no_larger_than_an_uncapped_one(self):
        churn()
        uncapped = mx.get_cache_memory()
        mx.clear_cache()
        with prefill_memory.cache_cap(1):
            churn()
            capped = mx.get_cache_memory()
        self.assertLessEqual(capped, uncapped)

    def test_values_computed_inside_the_cap_are_unchanged(self):
        mx.random.seed(4)
        a = mx.random.normal((256, 256))
        want = a @ a
        mx.eval(want)
        with prefill_memory.cache_cap(1):
            got = a @ a
            mx.eval(got)
        self.assertTrue(bool(mx.array_equal(got, want)))


class Recorder:
    """A context manager that logs entry and exit into a shared list."""

    def __init__(self, log, name, value=None):
        self.log, self.name, self.value = log, name, value

    def __enter__(self):
        self.log.append(f"enter {self.name}")
        return self.value

    def __exit__(self, *args):
        self.log.append(f"exit {self.name}")
        return False


class PrefillWiringTests(unittest.TestCase):
    ids = mx.zeros((1, 4), dtype=mx.uint32)
    language = SimpleNamespace()

    def call(self):
        return prefill.prefill_target(self.language, self.ids, [])

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {})
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("FLASHNEXT_PREFILL_CACHE_LIMIT_MB", "FLASHNEXT_PREFILL_RING_EXPERTS",
                     "FLASHNEXT_PREFILL_RING_SCRATCH"):
            os.environ.pop(name, None)
        self.original = current_limit()
        self.addCleanup(mx.set_cache_limit, self.original)

    def test_both_off_calls_the_existing_function_and_nothing_else(self):
        with mock.patch.object(prefill, "_prefill_target", return_value="done") as inner, \
                mock.patch.object(prefill_memory, "cache_cap") as cap, \
                mock.patch.object(prefill_ring, "session") as session:
            self.assertEqual(self.call(), "done")
        inner.assert_called_once()
        cap.assert_not_called()
        session.assert_not_called()
        self.assertEqual(current_limit(), self.original)

    def test_cap_only_caps_the_cache_for_the_call_and_skips_the_ring(self):
        os.environ["FLASHNEXT_PREFILL_CACHE_LIMIT_MB"] = "64"
        seen = []

        def inner(*args):
            seen.append(current_limit())
            return (None, None, 1)

        with mock.patch.object(prefill, "_prefill_target", side_effect=inner), \
                mock.patch.object(prefill_ring, "session") as session:
            self.call()
        self.assertEqual(seen, [64 * MIB])
        session.assert_not_called()
        self.assertEqual(current_limit(), self.original)

    def test_cap_and_ring_nest_the_ring_inside_the_cap(self):
        os.environ["FLASHNEXT_PREFILL_CACHE_LIMIT_MB"] = "64"
        os.environ["FLASHNEXT_PREFILL_RING_EXPERTS"] = "8"
        log = []

        def inner(*args):
            log.append("prefill")
            return (None, None, mx.array([5], dtype=mx.uint32))

        with mock.patch.object(prefill, "_prefill_target", side_effect=inner), \
                mock.patch.object(prefill_memory, "cache_cap",
                                  return_value=Recorder(log, "cap")), \
                mock.patch.object(prefill_ring, "session",
                                  return_value=Recorder(log, "ring", object())), \
                mock.patch.object(prefill_ring, "fence",
                                  side_effect=lambda: log.append("fence")):
            self.call()
        self.assertEqual(
            log,
            ["enter cap", "enter ring", "prefill", "fence", "exit ring", "exit cap"],
        )

    def test_a_failing_prefill_still_restores_the_cache_limit(self):
        os.environ["FLASHNEXT_PREFILL_CACHE_LIMIT_MB"] = "64"
        with mock.patch.object(prefill, "_prefill_target", side_effect=ValueError("x")):
            with self.assertRaises(ValueError):
                self.call()
        self.assertEqual(current_limit(), self.original)

    def test_ring_only_leaves_the_cache_limit_alone(self):
        os.environ["FLASHNEXT_PREFILL_RING_EXPERTS"] = "8"
        seen = []

        def inner(*args):
            seen.append(current_limit())
            return (None, None, 1)

        with mock.patch.object(prefill, "_prefill_target", side_effect=inner), \
                mock.patch.object(prefill_ring, "session",
                                  return_value=Recorder([], "ring", None)):
            self.call()
        self.assertEqual(seen, [self.original])


if __name__ == "__main__":
    unittest.main()
