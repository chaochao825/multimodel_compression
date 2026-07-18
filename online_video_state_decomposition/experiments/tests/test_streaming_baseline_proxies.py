from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

from streaming_baseline_proxies import PROXY_METHODS, run_proxy  # noqa: E402


class StreamingBaselineProxyTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260719)
        self.vectors = rng.normal(size=(32, 24))
        self.query = rng.normal(size=24)
        self.kwargs = {
            "evidence_budget": 8,
            "pool_capacity": 16,
            "recent_anchors": 3,
            "storage_bits": 16,
        }

    def test_all_proxies_are_deterministic_finite_and_nonempty(self) -> None:
        for method in PROXY_METHODS:
            with self.subTest(method=method):
                first = run_proxy(
                    method, self.vectors, self.query, **self.kwargs
                )
                second = run_proxy(
                    method, self.vectors, self.query, **self.kwargs
                )
                self.assertGreater(len(first.evidence_vectors), 0)
                self.assertTrue(np.all(np.isfinite(first.evidence_vectors)))
                np.testing.assert_allclose(
                    first.evidence_vectors, second.evidence_vectors
                )
                self.assertEqual(first.evidence_indices, second.evidence_indices)
                self.assertGreater(first.accounting.total_retained_bytes, 0)

    def test_query_free_writes_do_not_depend_on_question(self) -> None:
        other_query = -self.query
        for method in (
            "causalmem_feature_proxy",
            "stc_feature_proxy",
            "statekv_feature_proxy",
        ):
            with self.subTest(method=method):
                first = run_proxy(
                    method, self.vectors, self.query, **self.kwargs
                )
                second = run_proxy(
                    method, self.vectors, other_query, **self.kwargs
                )
                self.assertFalse(first.query_conditioned)
                self.assertEqual(first.evidence_indices, second.evidence_indices)

    def test_causalmem_retention_and_basis_are_bounded(self) -> None:
        result = run_proxy(
            "causalmem_feature_proxy",
            self.vectors,
            self.query,
            **self.kwargs,
        )
        self.assertLessEqual(len(result.evidence_vectors), 8)
        self.assertLessEqual(int(result.diagnostics["basis_rank"]), 8)
        self.assertTrue(result.accounting.active_state_bounded)
        self.assertTrue(result.accounting.total_state_bounded)

    def test_streamingtom_has_bounded_active_and_growing_int4_archive(self) -> None:
        short = run_proxy(
            "streamingtom_feature_proxy",
            self.vectors[:16],
            self.query,
            **self.kwargs,
        )
        long = run_proxy(
            "streamingtom_feature_proxy",
            self.vectors,
            self.query,
            **self.kwargs,
        )
        fp16_archive = len(self.vectors) // 2 * self.vectors.shape[1] * 2
        self.assertTrue(long.accounting.active_state_bounded)
        self.assertFalse(long.accounting.total_state_bounded)
        self.assertGreater(
            long.accounting.archive_bytes, short.accounting.archive_bytes
        )
        self.assertLess(long.accounting.archive_bytes, fp16_archive)
        self.assertLessEqual(len(long.evidence_vectors), 8)

    def test_stc_terminal_feature_adaptation_is_bounded(self) -> None:
        result = run_proxy(
            "stc_feature_proxy", self.vectors, self.query, **self.kwargs
        )
        self.assertLessEqual(len(result.evidence_vectors), 8)
        self.assertTrue(result.accounting.active_state_bounded)
        self.assertTrue(result.accounting.total_state_bounded)
        self.assertFalse(
            bool(result.diagnostics["spatial_token_structure_represented"])
        )

    def test_selectstream_keeps_bounded_graph_and_exact_recent(self) -> None:
        result = run_proxy(
            "selectstream_feature_proxy",
            self.vectors,
            self.query,
            **self.kwargs,
        )
        self.assertLessEqual(int(result.diagnostics["memory_nodes"]), 16)
        self.assertEqual(int(result.diagnostics["recent_direct_vectors"]), 3)
        self.assertLessEqual(len(result.evidence_vectors), 8)
        self.assertTrue(result.accounting.total_state_bounded)

    def test_oasis_bounds_roots_but_accounts_for_growing_archive(self) -> None:
        short = run_proxy(
            "oasis_feature_proxy",
            self.vectors[:16],
            self.query,
            **self.kwargs,
        )
        long = run_proxy(
            "oasis_feature_proxy",
            self.vectors,
            self.query,
            **self.kwargs,
        )
        self.assertLessEqual(int(long.diagnostics["event_roots"]), 13)
        self.assertTrue(long.accounting.active_state_bounded)
        self.assertFalse(long.accounting.total_state_bounded)
        self.assertGreater(
            long.accounting.archive_bytes, short.accounting.archive_bytes
        )
        self.assertLessEqual(len(long.evidence_vectors), 8)

    def test_statekv_cstate_is_bounded_but_dstate_grows(self) -> None:
        short = run_proxy(
            "statekv_feature_proxy",
            self.vectors[:16],
            self.query,
            **self.kwargs,
        )
        long = run_proxy(
            "statekv_feature_proxy",
            self.vectors,
            self.query,
            **self.kwargs,
        )
        self.assertLessEqual(int(long.diagnostics["cstate_vectors"]), 16)
        self.assertTrue(long.accounting.active_state_bounded)
        self.assertFalse(long.accounting.total_state_bounded)
        self.assertGreater(
            long.accounting.detailed_decode_bytes,
            short.accounting.detailed_decode_bytes,
        )
        self.assertEqual(len(long.evidence_vectors), len(self.vectors))


if __name__ == "__main__":
    unittest.main()
