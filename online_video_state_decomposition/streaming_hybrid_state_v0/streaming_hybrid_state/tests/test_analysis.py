from __future__ import annotations

import unittest

from streaming_hybrid_state.analyze import (
    combined_analysis,
    predictor_analysis,
)
from streaming_hybrid_state.core import DecisionTree
from streaming_hybrid_state.evaluate import TreeController


class StreamingHybridAnalysisTest(unittest.TestCase):
    def test_tree_description_counts_topology_bits(self) -> None:
        tree = DecisionTree(
            feature=0,
            threshold=0.5,
            left=DecisionTree(label=0),
            right=DecisionTree(label=1),
        )
        controller = TreeController(tree)
        self.assertEqual(controller.static_bits, 26)

    def test_memory_candidate_is_not_compute_candidate(self) -> None:
        combined = [
            {
                "layer": "22",
                "budget_bps": "2.0",
                "controller": "always_int4_refresh",
                "predictor": "previous",
                "innovation_codec": "pq",
                "innovation_fraction": "1.0",
                "mean_cosine": "0.96",
                "p05_cosine": "0.90",
                "effective_bps": "4.016",
                "reuse_rate": "0.0",
                "predict_rate": "0.0",
                "innovation_rate": "0.0",
                "refresh_rate": "1.0",
                "action_cost_proxy": "1.0",
            },
            {
                "layer": "22",
                "budget_bps": "2.0",
                "controller": "decision_tree",
                "predictor": "previous",
                "innovation_codec": "pq",
                "innovation_fraction": "1.0",
                "mean_cosine": "0.955",
                "p05_cosine": "0.90",
                "effective_bps": "2.1",
                "reuse_rate": "0.2",
                "predict_rate": "0.0",
                "innovation_rate": "0.6",
                "refresh_rate": "0.2",
                "action_cost_proxy": "0.35",
            },
        ]
        vq = [
            {
                "layer": "22",
                "split": "test",
                "method": "raw_pq",
                "codec": "pq",
                "selected_fraction": "1.0",
                "effective_bps": "2.0",
                "mean_cosine": "0.94",
            }
        ]
        rows, memory, compute = combined_analysis(combined, vq, [22])
        learned = next(
            row for row in rows if row["controller"] == "decision_tree"
        )
        self.assertTrue(learned["memory_representation_candidate"])
        self.assertFalse(learned["conditional_compute_candidate"])
        self.assertAlmostEqual(learned["encoder_required_rate"], 0.8)
        self.assertEqual(memory["verdict"], "Positive")
        self.assertEqual(compute["verdict"], "Negative")

    def test_fourier_loses_when_test_nmse_is_higher(self) -> None:
        rows = []
        for split in ("val", "test"):
            rows.extend(
                [
                    {
                        "layer": "22",
                        "split": split,
                        "predictor": "previous",
                        "nmse": "0.3",
                        "mean_cosine": "0.8",
                        "ops_per_scalar_proxy": "0",
                        "raw_temporal_spectral_entropy": "0.4",
                        "residual_temporal_spectral_entropy": "0.6",
                        "spectral_entropy_reduction": "-0.5",
                    },
                    {
                        "layer": "22",
                        "split": split,
                        "predictor": "fourier_h4_k1",
                        "nmse": "1.0",
                        "mean_cosine": "0.6",
                        "ops_per_scalar_proxy": "32",
                        "raw_temporal_spectral_entropy": "0.4",
                        "residual_temporal_spectral_entropy": "0.5",
                        "spectral_entropy_reduction": "-0.25",
                    },
                ]
            )
        comparison, verdict = predictor_analysis(rows, [22])
        self.assertFalse(comparison[0]["fourier_win"])
        self.assertEqual(verdict["verdict"], "Negative")


if __name__ == "__main__":
    unittest.main()
