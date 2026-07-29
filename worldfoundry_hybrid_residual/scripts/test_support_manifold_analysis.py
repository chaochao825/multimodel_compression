#!/usr/bin/env python3

import unittest

from analyze_support_manifold_oracle import (
    family_choices,
    layer14_frontier,
    rank_budget_frontier,
)


class SupportManifoldAnalysisTests(unittest.TestCase):
    def test_frontier_uses_worst_layer14_cell(self) -> None:
        rows = []
        for cell, aggregate, worst in (
            ("layer14_step00", 0.004, 0.009),
            ("layer14_step09", 0.006, 0.015),
        ):
            rows.append(
                {
                    "cell": cell,
                    "family": "fixed64",
                    "density_target": "0.25",
                    "adaptive_rank16_output_relative_l2": str(aggregate),
                    "adaptive_rank16_worst_record_relative_l2": str(worst),
                    "adaptive_rank16_p95_record_relative_l2": str(worst),
                    "rank_aware_relative_improvement_mean": "0.1",
                    "rank_required_for_record_gate_max": "20",
                    "execution_density_mean": "0.25",
                    "kernel_tile_multiplier_vs_fixed64": "1.0",
                    "support_pregate_pass": "False",
                }
            )
        result = layer14_frontier(rows)[0]
        self.assertEqual(result["cells"], 2)
        self.assertAlmostEqual(float(result["max_worst_record_output_relative_l2"]), 0.015)
        self.assertFalse(result["all_cells_pass"])

    def test_family_choice_counts_only_oracle_rows(self) -> None:
        rows = [
            {
                "cell": "layer14_step00",
                "density_target": "0.25",
                "family": "support_family_oracle",
                "chosen_family": "shifted32",
            },
            {
                "cell": "layer14_step00",
                "density_target": "0.25",
                "family": "support_family_oracle",
                "chosen_family": "fixed64",
            },
            {
                "cell": "layer14_step00",
                "density_target": "0.25",
                "family": "fixed64",
                "chosen_family": "fixed64",
            },
        ]
        result = family_choices(rows)
        self.assertEqual(sum(int(row["records"]) for row in result), 2)
        self.assertAlmostEqual(sum(float(row["fraction"]) for row in result), 1.0)

    def test_rank_budget_reconstructs_residual_from_captured_energy(self) -> None:
        row = {
            "cell": "layer14_step00",
            "family": "fixed64",
            "density_target": "0.25",
            "reference_sq": "4.0",
            "critical_residual_sq": "1.0",
            "defect_energy_rank4": "0.75",
            "defect_energy_rank8": "0.8",
            "defect_energy_rank16": "0.9",
            "defect_energy_rank32": "0.96",
        }
        result = rank_budget_frontier([row])
        rank16 = next(item for item in result if int(item["rank"]) == 16)
        self.assertAlmostEqual(float(rank16["aggregate_output_relative_l2"]), (0.1 / 4.0) ** 0.5)


if __name__ == "__main__":
    unittest.main()
