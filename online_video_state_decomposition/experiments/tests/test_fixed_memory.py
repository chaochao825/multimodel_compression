from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from probes.benchmark_fixed_memory import (
    AdaptiveSlotMemory,
    InstantPlusLongTermMemory,
    OjaSubspaceMemory,
    RecentWindowMemory,
    ReservoirMemory,
    normalize_rows,
    pool_regions,
    reconstruction_metrics,
    state_accounting,
)


class FixedMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(123)

    def test_region_pooling_handles_nondivisible_grid(self) -> None:
        frame = np.arange(5 * 7 * 3, dtype=np.float64).reshape(5, 7, 3)
        pooled = pool_regions(frame, rows=2, cols=3)
        self.assertEqual(pooled.shape, (6, 3))
        self.assertTrue(np.all(np.isfinite(pooled)))

    def test_recent_window_retains_only_latest_vectors(self) -> None:
        memory = RecentWindowMemory(capacity=2, hidden_dim=3)
        values = np.eye(3)
        memory.update(values)
        reconstructed = memory.reconstruct(values)
        metrics = reconstruction_metrics(
            values,
            reconstructed,
            coverage_threshold=0.99,
        )
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)

    def test_reservoir_is_bounded_and_deterministic(self) -> None:
        values = normalize_rows(self.rng.normal(size=(100, 8)))
        left = ReservoirMemory(capacity=7, hidden_dim=8, seed=9)
        right = ReservoirMemory(capacity=7, hidden_dim=8, seed=9)
        left.update(values)
        right.update(values)
        queries = normalize_rows(self.rng.normal(size=(4, 8)))
        self.assertTrue(
            np.allclose(
                left.reconstruct(queries),
                right.reconstruct(queries),
            )
        )

    def test_adaptive_slots_reconstruct_cluster_centers(self) -> None:
        centers = normalize_rows(self.rng.normal(size=(3, 16)))
        values = np.concatenate(
            [
                normalize_rows(
                    center + 0.02 * self.rng.normal(size=(20, 16))
                )
                for center in centers
            ]
        )
        memory = AdaptiveSlotMemory(
            capacity=3,
            hidden_dim=16,
            min_lr=0.05,
            replace_similarity=0.75,
        )
        memory.update(values)
        metrics = reconstruction_metrics(
            centers,
            memory.reconstruct(centers),
            coverage_threshold=0.90,
        )
        self.assertGreater(metrics["mean_cosine"], 0.95)

    def test_oja_subspace_learns_low_rank_stream(self) -> None:
        basis = np.linalg.qr(self.rng.normal(size=(24, 3)))[0]
        memory = OjaSubspaceMemory(
            capacity=3,
            hidden_dim=24,
            seed=11,
            learning_rate=1.0,
        )
        for _ in range(100):
            coefficients = self.rng.normal(size=(32, 3))
            memory.update(normalize_rows(coefficients @ basis.T))
        queries = normalize_rows(self.rng.normal(size=(64, 3)) @ basis.T)
        metrics = reconstruction_metrics(
            queries,
            memory.reconstruct(queries),
            coverage_threshold=0.90,
        )
        self.assertGreater(metrics["mean_cosine"], 0.90)

    def test_instant_plus_long_term_preserves_current_and_prior(self) -> None:
        long_term = AdaptiveSlotMemory(
            capacity=2,
            hidden_dim=4,
            min_lr=0.05,
            replace_similarity=0.75,
        )
        memory = InstantPlusLongTermMemory(
            instant_capacity=2,
            hidden_dim=4,
            long_term=long_term,
        )
        first = np.eye(4)[:2]
        second = np.eye(4)[2:]
        memory.update(first)
        self.assertTrue(np.allclose(memory.reconstruct(first), first))
        memory.update(second)
        self.assertTrue(np.allclose(memory.reconstruct(second), second))
        self.assertTrue(np.allclose(memory.reconstruct(first), first))

    def test_hybrid_accounting_keeps_total_payload_fixed(self) -> None:
        accounting = state_accounting(
            "instant_oja",
            capacity=32,
            hidden_dim=1024,
            storage_bits=16,
            instant_capacity=16,
        )
        self.assertEqual(accounting["payload_bytes"], 32 * 1024 * 2)

    def test_multi_frame_instant_cache_only_consolidates_evictions(self) -> None:
        long_term = AdaptiveSlotMemory(
            capacity=2,
            hidden_dim=6,
            min_lr=0.05,
            replace_similarity=0.75,
        )
        memory = InstantPlusLongTermMemory(
            instant_capacity=4,
            hidden_dim=6,
            long_term=long_term,
        )
        frames = [np.eye(6)[index : index + 2] for index in (0, 2, 4)]
        memory.update(frames[0])
        memory.update(frames[1])
        self.assertTrue(np.allclose(memory.reconstruct(frames[0]), frames[0]))
        self.assertTrue(np.allclose(memory.reconstruct(frames[1]), frames[1]))
        memory.update(frames[2])
        self.assertTrue(np.allclose(memory.reconstruct(frames[0]), frames[0]))
        self.assertTrue(np.allclose(memory.reconstruct(frames[1]), frames[1]))
        self.assertTrue(np.allclose(memory.reconstruct(frames[2]), frames[2]))


if __name__ == "__main__":
    unittest.main()
