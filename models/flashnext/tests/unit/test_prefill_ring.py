"""Stage 1 prefill ring: exact tensors, zero-copy views, ownership, fallback.

A small synthetic checkpoint with the real expert row sizes (hidden 2560,
intermediate 640, Q4/G32) stands in for the model. Two real
``StreamingSwitchGLU`` layers run through the real ``SafeTensorStore`` and a real
``ExpertPool``; nothing is downloaded and no model weights are needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from models.flashnext import expert_cache, expert_pool, prefill, prefill_ring
from models.flashnext.pool_ring import FREE, HazardError
from models.flashnext.store import SafeTensorStore

EXPERTS = 12
HIDDEN = 2560
INTER = 640
SHAPES = {
    "gate_proj": (INTER, HIDDEN // 8, HIDDEN // 32),
    "up_proj": (INTER, HIDDEN // 8, HIDDEN // 32),
    "down_proj": (HIDDEN, INTER // 8, INTER // 32),
}
LAYERS = (1, 2)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ENV = {
    "FLASHNEXT_EXPERT_POOL_GB": "0.15",
    "FLASHNEXT_PREFILL_RING_EXPERTS": "8",
    "FLASHNEXT_PREFILL_RING_MIN_TOKENS": "0",
    "FLASHNEXT_PREFILL_RING_MARGIN_GB": "0.5",
    "FLASHNEXT_PREFILL_RING_TRANSIENT_GB": "0.1",
}


def silu_glu(up, gate):
    return nn.silu(gate) * up


def prefix_of(layer):
    return f"language_model.model.layers.{layer}.mlp.switch_mlp"


def build_checkpoint(directory):
    mx.random.seed(11)
    tensors = {}
    for layer in LAYERS:
        for projection, (rows, packed, groups) in SHAPES.items():
            base = f"{prefix_of(layer)}.{projection}"
            tensors[f"{base}.weight"] = mx.random.randint(
                0, 2**31 - 1, (EXPERTS, rows, packed), dtype=mx.uint32
            )
            tensors[f"{base}.scales"] = (
                mx.random.uniform(0.001, 0.01, (EXPERTS, rows, groups))
            ).astype(mx.bfloat16)
            tensors[f"{base}.biases"] = (
                mx.random.uniform(-0.01, 0.01, (EXPERTS, rows, groups))
            ).astype(mx.bfloat16)
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(os.path.join(directory, shard), tensors)
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
        json.dump({"weight_map": {key: shard for key in tensors}}, out)


class RingCase(unittest.TestCase):
    """Builds the store, pool and two streamed layers once per test."""

    pool_gb = "0.15"

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        build_checkpoint(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        patcher = mock.patch.dict(
            os.environ, {**ENV, "FLASHNEXT_EXPERT_POOL_GB": self.pool_gb})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = SafeTensorStore(self.directory.name)
        self.addCleanup(self.store.close)
        self.layers = [
            expert_cache.StreamingSwitchGLU(
                self.store, prefix_of(layer), 32, 4, "affine", silu_glu,
                layer_id=layer,
            )
            for layer in LAYERS
        ]
        self.language = SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(switch_mlp=glu))
            for glu in self.layers
        ]))
        self.pool = expert_pool.for_store(self.store, 32)
        self.addCleanup(self.pool.close)
        self.addCleanup(expert_cache.set_prefill_ring, None)
        self.free_before = sorted(self.pool._free)

    def inputs(self, tokens=16, width=4, seed=3, experts=8):
        """Routes over the first ``experts`` experts (the ring half holds 8)."""
        mx.random.seed(seed)
        x = mx.random.normal((1, tokens, HIDDEN)).astype(mx.bfloat16)
        indices = mx.random.randint(0, experts, (1, tokens, width)).astype(mx.uint32)
        return x, indices

    def run_layers(self, x, indices):
        outputs = []
        for glu in self.layers:
            y = glu(x, indices)
            mx.eval(y)            # the router sync of the next layer does this
            outputs.append(y)
        return outputs

    def assertRestored(self):
        self.assertEqual(sorted(self.pool._free), self.free_before)
        self.assertIsNone(expert_cache._PREFILL_RING[0])


class TensorCorrectnessTests(RingCase):
    def test_ring_output_is_bitwise_equal_to_existing_path(self):
        x, indices = self.inputs()
        reference = self.run_layers(x, indices)
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring, prefill_ring.LAST_REASON[0])
            self.assertIs(expert_cache._PREFILL_RING[0], ring)
            produced = self.run_layers(x, indices)
            stats = ring.stats()
        for got, want in zip(produced, reference):
            self.assertTrue(bool(mx.array_equal(got, want)))
        self.assertEqual(stats["staged_layers"], len(LAYERS))
        self.assertEqual(stats["fallback_error"], 0)
        self.assertRestored()

    def test_staged_views_hold_the_checkpoint_rows_in_first_seen_order(self):
        routed = [7, 2, 7, 11, 2, 5]
        wanted = list(dict.fromkeys(routed))
        self.assertEqual(wanted, [7, 2, 11, 5])
        with prefill_ring.session(self.language, 1024) as ring:
            prefix = prefix_of(1)
            weights = ring.stage(self.store, prefix, wanted)
            self.assertIsNotNone(weights)
            for projection, parts in zip(PROJECTIONS, weights):
                for kind, array in zip(prefill_ring.KINDS, parts):
                    name = f"{prefix}.{projection}.{kind}"
                    expected = self.store.rows_np(name, wanted)
                    got = np.array(array.view(mx.uint16) if kind != "weight" else array)
                    self.assertEqual(got.shape, expected.shape, name)
                    self.assertEqual(got.tobytes(), expected.tobytes(), name)
                    self.assertEqual(
                        array.dtype, mx.uint32 if kind == "weight" else mx.bfloat16
                    )

    def test_views_are_zero_copy(self):
        with prefill_ring.session(self.language, 1024) as ring:
            mx.eval(mx.zeros(1))
            before = mx.get_active_memory()
            weights = ring.stage(self.store, prefix_of(1), list(range(8)))
            mx.eval([array for parts in weights for array in parts])
            grown = mx.get_active_memory() - before
        # Eight experts are ~24.6 MB; a copy of any projection is >= 6 MB.
        self.assertLess(grown, 1_000_000)

    def test_only_the_leased_range_is_written(self):
        header = expert_pool.HEADER
        with prefill_ring.session(self.language, 1024) as ring:
            first = ring.ring.first_slot
            below = header + first * self.pool.stride
            digest = hashlib.sha256(self.pool.array[:below]).digest()
            self.assertIsNotNone(ring.stage(self.store, prefix_of(1), list(range(8))))
            self.assertEqual(
                hashlib.sha256(self.pool.array[:below]).digest(), digest
            )
            self.assertGreaterEqual(first, 16)

    def test_decode_sized_calls_never_touch_the_ring(self):
        x, indices = self.inputs(tokens=4)
        with prefill_ring.session(self.language, 1024) as ring:
            y = self.layers[0](x, indices)
            mx.eval(y)
            self.assertEqual(ring.stats()["staged_layers"], 0)
        self.assertRestored()


class HighOffsetTests(RingCase):
    """A 4.6 GB pool puts the ring above 4.29 GB, where a bf16 element index
    passes 2**31. Only the touched ring pages become resident."""

    pool_gb = "4.6"

    def test_views_beyond_the_32_bit_element_range_hold_the_checkpoint_rows(self):
        wanted = [9, 3, 11, 0, 6]
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring, prefill_ring.LAST_REASON[0])
            top = (expert_pool.HEADER + ring.ring.first_slot * self.pool.stride)
            self.assertGreater(top // 2, 2**31 - 1)   # old bf16 index overflowed
            weights = ring.stage(self.store, prefix_of(1), wanted)
            self.assertIsNotNone(weights, ring.stats()["last_error"])
            for projection, parts in zip(PROJECTIONS, weights):
                for kind, array in zip(prefill_ring.KINDS, parts):
                    name = f"{prefix_of(1)}.{projection}.{kind}"
                    expected = self.store.rows_np(name, wanted)
                    got = np.array(array.view(mx.uint16) if kind != "weight" else array)
                    self.assertEqual(got.tobytes(), expected.tobytes(), name)

    def test_output_is_bitwise_equal_with_the_ring_high_in_a_big_pool(self):
        x, indices = self.inputs()
        reference = self.run_layers(x, indices)
        with prefill_ring.session(self.language, 1024) as ring:
            produced = self.run_layers(x, indices)
            self.assertEqual(ring.stats()["staged_layers"], len(LAYERS))
        for got, want in zip(produced, reference):
            self.assertTrue(bool(mx.array_equal(got, want)))


class FallbackTests(RingCase):
    def reference_and_ring(self, x, indices, ring):
        reference = self.run_layers(x, indices)
        produced = self.run_layers(x, indices)
        for got, want in zip(produced, reference):
            self.assertTrue(bool(mx.array_equal(got, want)))
        return ring.stats()

    def test_more_experts_than_a_half_falls_back(self):
        x, indices = self.inputs(tokens=64, width=8, seed=9, experts=EXPERTS)
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "2"}):
            with prefill_ring.session(self.language, 1024) as ring:
                self.assertEqual(ring.ring.half_experts, 2)
                stats = self.reference_and_ring(x, indices, ring)
        self.assertEqual(stats["staged_layers"], 0)
        self.assertGreaterEqual(stats["fallback_capacity"], 1)
        self.assertRestored()

    def test_read_error_releases_the_half_and_falls_back(self):
        x, indices = self.inputs()
        with prefill_ring.session(self.language, 1024) as ring:
            with mock.patch.object(prefill_ring, "_read_rows", side_effect=OSError("boom")):
                got = self.layers[0](x, indices)
                mx.eval(got)
            self.assertEqual(ring.stats()["fallback_error"], 1)
            self.assertIn("boom", ring.stats()["last_error"])
            self.assertEqual([h.state for h in ring.ring._halves], [FREE])
            # The ring keeps working for the next layer.
            again = self.layers[1](x, indices)
            mx.eval(again)
            self.assertEqual(ring.stats()["staged_layers"], 1)
        want = self.layers[0](x, indices)
        mx.eval(want)
        self.assertTrue(bool(mx.array_equal(got, want)))
        self.assertRestored()

    def test_short_read_is_an_error_not_silent_garbage(self):
        with prefill_ring.session(self.language, 1024) as ring:
            with mock.patch.object(prefill_ring.os, "preadv", return_value=7):
                self.assertIsNone(ring.stage(self.store, prefix_of(1), [0, 1]))
            self.assertIn("short expert read", ring.stats()["last_error"])
            self.assertEqual([h.state for h in ring.ring._halves], [FREE])

    def test_busy_half_falls_back(self):
        with prefill_ring.session(self.language, 1024) as ring:
            held = ring.ring.begin_fill()
            self.assertIsNotNone(held)
            self.assertIsNone(ring.stage(self.store, prefix_of(1), [0, 1]))
            self.assertEqual(ring.stats()["fallback_busy"], 1)
            held.release()
            self.assertIsNotNone(ring.stage(self.store, prefix_of(1), [0, 1]))

    def test_memory_pressure_disables_the_ring_for_the_rest_of_the_prefill(self):
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring.stage(self.store, prefix_of(1), [0, 1]))
            ring._active_memory = lambda: ring._limit      # no room left
            self.assertIsNone(ring.stage(self.store, prefix_of(2), [0, 1]))
            self.assertEqual(ring.disabled, "working set")
            ring._active_memory = lambda: 0                # stays off anyway
            self.assertIsNone(ring.stage(self.store, prefix_of(2), [0, 1]))
            self.assertEqual(ring.stats()["fallback_disabled"], 1)
        self.assertRestored()

    def test_empty_request_falls_back(self):
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNone(ring.stage(self.store, prefix_of(1), []))


class AcquireTests(RingCase):
    def reason(self):
        return prefill_ring.LAST_REASON[0]

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "0"}):
            self.assertFalse(prefill_ring.enabled())
            with prefill_ring.session(self.language, 1024) as ring:
                self.assertIsNone(ring)
                self.assertIsNone(expert_cache._PREFILL_RING[0])
        os.environ.pop("FLASHNEXT_PREFILL_RING_EXPERTS")
        self.assertFalse(prefill_ring.enabled())

    def test_short_prompts_do_not_use_the_ring(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_MIN_TOKENS": "512"}):
            self.assertIsNone(prefill_ring.acquire(self.language, 100))
            self.assertEqual(self.reason(), "short prompt")
            state = prefill_ring.acquire(self.language, 512)
            self.assertIsNotNone(state)
            prefill_ring.release(state)
        self.assertRestored()

    def test_requires_the_expert_pool(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_EXPERT_POOL_GB": "0"}):
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
            self.assertEqual(self.reason(), "expert pool is off")

    def test_refuses_when_the_working_set_does_not_fit(self):
        # Active memory is patched: reading the real value twice races with
        # MLX freeing earlier tests' arrays asynchronously. It already
        # contains the pool, which must count once.
        active = len(self.pool._mm) + 100_000_000
        limit = active + 400_000_000
        with mock.patch.object(mx, "get_active_memory", return_value=active):
            with mock.patch.object(
                prefill_ring, "metal_working_set_limit", return_value=limit
            ):
                # 0.1 GB transient + 0.5 GB margin does not fit in 0.4 GB.
                self.assertIsNone(prefill_ring.acquire(self.language, 1024))
                self.assertTrue(
                    self.reason().startswith("working set: MLX active"), self.reason())
                self.assertIn("exceeds limit", self.reason())
            with mock.patch.object(
                prefill_ring, "metal_working_set_limit",
                return_value=limit + 400_000_000,
            ):
                state = prefill_ring.acquire(self.language, 1024)
                self.assertIsNotNone(state)
                prefill_ring.release(state)
        self.assertRestored()

    def test_the_pool_is_counted_once(self):
        pool_bytes = len(self.pool._mm)
        self.assertEqual(prefill_ring.working_set_bytes(pool_bytes + 5, pool_bytes),
                         pool_bytes + 5)
        self.assertEqual(prefill_ring.working_set_bytes(7, pool_bytes), 7 + pool_bytes)
        # The fact the formula rests on, checked against MLX itself: every live
        # wrapped pool is inside active memory. A lower bound, not a before and
        # after difference, because MLX frees earlier tests' arrays
        # asynchronously and would make a difference unreliable.
        stride = self.pool.stride
        second = expert_pool.ExpertPool(
            self.store, expert_pool.HEADER + 20 * stride, 32)
        try:
            live = len(self.pool._mm) + len(second._mm)
            self.assertGreaterEqual(mx.get_active_memory(), live)
        finally:
            second.close()

    def test_a_pool_sized_to_fit_is_accepted_once_counted_once(self):
        # Room for the pool once plus transient (0.1 GB) and margin (0.5 GB),
        # but not for the pool counted twice.
        pool_bytes = len(self.pool._mm)
        active = pool_bytes + 100_000_000
        limit = active + 100_000_000 + 500_000_000 + 50_000_000
        self.assertLess(limit - active, pool_bytes + 100_000_000 + 500_000_000)
        with mock.patch.object(mx, "get_active_memory", return_value=active):
            with mock.patch.object(
                prefill_ring, "metal_working_set_limit", return_value=limit
            ):
                state = prefill_ring.acquire(self.language, 1024)
                self.assertIsNotNone(state, self.reason())
                prefill_ring.release(state)

    def test_refuses_when_the_limit_is_unknown(self):
        with mock.patch.object(prefill_ring, "metal_working_set_limit", return_value=None):
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
            self.assertEqual(self.reason(), "working-set limit unknown")

    def test_refuses_a_layout_it_cannot_view(self):
        ref = self.store.refs[f"{prefix_of(1)}.gate_proj.weight"]
        original = ref.dtype
        ref.dtype = "F32"
        try:
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
            self.assertIn("unsupported dtype", self.reason())
        finally:
            ref.dtype = original

    def test_no_streamed_layers(self):
        empty = SimpleNamespace(model=SimpleNamespace(layers=[]))
        self.assertIsNone(prefill_ring.acquire(empty, 1024))
        self.assertEqual(self.reason(), "no streamed layers")

    def test_real_device_reports_a_working_set_limit(self):
        limit = prefill_ring.metal_working_set_limit()
        self.assertTrue(limit is None or limit > 1_000_000_000)

    def test_bad_flag_values_raise(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "lots"}):
            with self.assertRaises(ValueError):
                prefill_ring.configured_experts()


class OwnershipTests(RingCase):
    def test_session_returns_every_slot_on_normal_exit(self):
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertLess(len(self.pool._free), len(self.free_before))
            ring.stage(self.store, prefix_of(1), [0, 1, 2])
        self.assertRestored()

    def test_exception_in_the_prefill_still_returns_the_ring(self):
        class Boom(Exception):
            pass

        with self.assertRaises(Boom):
            with prefill_ring.session(self.language, 1024) as ring:
                ring.stage(self.store, prefix_of(1), [0, 1, 2])
                raise Boom()
        self.assertRestored()

    def test_exception_waits_for_the_gpu_before_release(self):
        calls = []
        with self.assertRaises(RuntimeError):
            with prefill_ring.session(self.language, 1024) as ring:
                ring._synchronize = lambda: calls.append("sync")
                ring.stage(self.store, prefix_of(1), [0, 1, 2])
                raise RuntimeError("fail")
        self.assertEqual(calls, ["sync"])
        self.assertRestored()

    def test_cancellation_during_a_read_releases_everything(self):
        with self.assertRaises(KeyboardInterrupt):
            with prefill_ring.session(self.language, 1024) as ring:
                ring._resolve = mock.Mock(side_effect=KeyboardInterrupt())
                ring.stage(self.store, prefix_of(1), [0, 1, 2, 3])
        self.assertRestored()

    def test_pool_is_usable_for_decode_after_the_session(self):
        with prefill_ring.session(self.language, 1024) as ring:
            ring.stage(self.store, prefix_of(1), [0, 1, 2])
        slots, misses = self.pool.assign(1, [0, 1, 2, 3])
        self.assertEqual(len(misses), 4)
        self.assertEqual(len(set(slots.values())), 4)

    def test_staged_slots_are_evicted_from_the_decode_cache(self):
        slots, _ = self.pool.assign(1, list(range(10)))
        top = max(slots.values())
        with prefill_ring.session(self.language, 1024) as ring:
            leased_first = ring.ring.first_slot
            for key, slot in self.pool._slot_of.items():
                self.assertLess(slot, leased_first, key)
        self.assertTrue(top >= 0)

    def test_second_session_after_the_first_works(self):
        for _ in range(2):
            with prefill_ring.session(self.language, 1024) as ring:
                self.assertIsNotNone(ring.stage(self.store, prefix_of(1), [4, 5]))
            self.assertRestored()

    def test_released_half_cannot_be_written(self):
        with prefill_ring.session(self.language, 1024) as ring:
            lease = ring.ring.begin_fill()
            lease.release()
            with self.assertRaises(HazardError):
                with lease.writing():
                    pass


class PrefillHookTests(RingCase):
    def test_flag_off_calls_the_existing_function_untouched(self):
        os.environ["FLASHNEXT_PREFILL_RING_EXPERTS"] = "0"
        with mock.patch.object(prefill, "_prefill_target", return_value="done") as inner:
            with mock.patch.object(prefill_ring, "session") as session:
                result = prefill.prefill_target(
                    self.language, mx.zeros((1, 4), dtype=mx.uint32), [], True, False, None
                )
        self.assertEqual(result, "done")
        inner.assert_called_once()
        session.assert_not_called()

    def test_flag_on_wraps_the_prefill_and_flushes_before_release(self):
        order = []

        class FakeSession:
            def __enter__(self_inner):
                order.append("enter")
                return object()

            def __exit__(self_inner, *args):
                order.append("exit")
                return False

        def inner(*args):
            order.append("prefill")
            return (None, None, mx.array([5], dtype=mx.uint32))

        with mock.patch.object(prefill, "_prefill_target", side_effect=inner):
            with mock.patch.object(prefill_ring, "session", return_value=FakeSession()):
                with mock.patch.object(
                    prefill_ring, "fence", side_effect=lambda: order.append("sync")
                ):
                    prefill.prefill_target(
                        self.language, mx.zeros((1, 4), dtype=mx.uint32), []
                    )
        self.assertEqual(order, ["enter", "prefill", "sync", "exit"])

    def test_flag_on_without_a_ring_does_not_flush(self):
        class NoRing:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *args):
                return False

        with mock.patch.object(prefill, "_prefill_target", return_value=(None, None, 1)):
            with mock.patch.object(prefill_ring, "session", return_value=NoRing()):
                with mock.patch.object(prefill_ring, "fence") as sync:
                    prefill.prefill_target(
                        self.language, mx.zeros((1, 4), dtype=mx.uint32), []
                    )
        sync.assert_not_called()

    def test_prompt_failure_inside_the_hook_still_releases_the_ring(self):
        with mock.patch.object(prefill, "_prefill_target", side_effect=ValueError("x")):
            with self.assertRaises(ValueError):
                prefill.prefill_target(
                    self.language, mx.zeros((1, 2000), dtype=mx.uint32), []
                )
        self.assertRestored()


if __name__ == "__main__":
    unittest.main()
