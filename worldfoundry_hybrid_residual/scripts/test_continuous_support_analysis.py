#!/usr/bin/env python3

import unittest

from analyze_continuous_support_relaxation import build_tradeoff_rows, summarize_tradeoff


class ContinuousSupportAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "sample_id": "sample",
                "cell": "layer14_step09_middle",
                "head": "2",
                "tile_index": "1",
                "density": "0.25",
                "discrete_recomputed_output_relative_l2": "0.05",
                "fractional_output_relative_l2": "0.002",
                "weight_sum": "512",
                "weight_dynamic_scalars": "2048",
                "weighted_topk_1x_output_relative_l2": "0.30",
                "weighted_topk_1p25x_output_relative_l2": "0.20",
                "weighted_topk_1p5x_output_relative_l2": "0.10",
                "weighted_topk_1p75x_output_relative_l2": "0.05",
                "weighted_topk_2x_output_relative_l2": "0.02",
                "weighted_topk_2p25x_output_relative_l2": "0.015",
                "weighted_topk_2p5x_output_relative_l2": "0.012",
                "weighted_topk_2p75x_output_relative_l2": "0.009",
                "weighted_topk_3x_output_relative_l2": "0.007",
                "weighted_topk_3p25x_output_relative_l2": "0.006",
                "weighted_topk_3p5x_output_relative_l2": "0.004",
                "weighted_topk_3p75x_output_relative_l2": "0.003",
                "weighted_topk_4x_output_relative_l2": "0.002",
                "refit_weighted_topk_1x_output_relative_l2": "0.05",
                "refit_weighted_topk_1p25x_output_relative_l2": "0.04",
                "refit_weighted_topk_1p5x_output_relative_l2": "0.03",
                "refit_weighted_topk_1p75x_output_relative_l2": "0.025",
                "refit_weighted_topk_2x_output_relative_l2": "0.018",
                "refit_weighted_topk_2p25x_output_relative_l2": "0.014",
                "refit_weighted_topk_2p5x_output_relative_l2": "0.011",
                "refit_weighted_topk_2p75x_output_relative_l2": "0.008",
                "refit_weighted_topk_3x_output_relative_l2": "0.006",
                "refit_weighted_topk_3p25x_output_relative_l2": "0.005",
                "refit_weighted_topk_3p5x_output_relative_l2": "0.004",
                "refit_weighted_topk_3p75x_output_relative_l2": "0.003",
                "refit_weighted_topk_4x_output_relative_l2": "0.002",
            }
        ]

    def test_tradeoff_preserves_density_and_scalar_cost(self) -> None:
        rows = build_tradeoff_rows(self.records)
        topk3 = next(row for row in rows if row["mode"] == "refit weighted top-k 3x")
        self.assertAlmostEqual(float(topk3["tile_density"]), 0.75)
        self.assertEqual(int(topk3["dynamic_tile_scalars"]), 1536)

    def test_summary_finds_first_passing_executable_density(self) -> None:
        summary = summarize_tradeoff(build_tradeoff_rows(self.records))
        self.assertAlmostEqual(
            float(summary["minimum_weighted_topk_density_passing_1pct"]), 0.6875
        )
        self.assertAlmostEqual(
            float(summary["corresponding_ideal_tile_arithmetic_speedup_upper_bound"]),
            1.0 / 0.6875,
        )
        self.assertEqual(
            summary["verdict"],
            "STOP_POSTHOC_WEIGHTED_SUPPORT_NO_ARITHMETIC_INTERSECTION",
        )
        self.assertFalse(summary["quality_speed_joint_gate_pass"])

    def test_payload_is_clamped_at_candidate_tile_count(self) -> None:
        record = dict(self.records[0])
        record["density"] = "0.5"
        record["weight_sum"] = "1024"
        rows = build_tradeoff_rows([record])
        full = next(row for row in rows if row["mode"] == "refit weighted top-k 4x")
        self.assertEqual(int(full["dynamic_tile_scalars"]), 2048)


if __name__ == "__main__":
    unittest.main()
