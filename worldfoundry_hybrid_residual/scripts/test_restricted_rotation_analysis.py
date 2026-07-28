#!/usr/bin/env python3
"""Regression tests for restricted-rotation plotting edge shapes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analyze_restricted_rotation_oracle import head_fallback_frontier, plot_rotation_curves


class RestrictedRotationAnalysisTests(unittest.TestCase):
    def test_two_splits_one_density_uses_column_axes(self) -> None:
        rows = []
        for split in ("seed_holdout", "prompt_holdout"):
            rows.append(
                {
                    "split": split,
                    "cell": "capacity",
                    "density": "0.25",
                    "family": "butterfly",
                    "generators": "8",
                    "worst_record_output_relative_l2": "0.012",
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plot_rotation_curves(rows, output)
            self.assertGreater((output / "restricted_rotation_error_by_M.png").stat().st_size, 0)
            self.assertGreater((output / "restricted_rotation_error_by_M.pdf").stat().st_size, 0)

    def test_universal_head_fallback_uses_union_across_splits(self) -> None:
        rows = []
        for split in ("seed_holdout", "prompt_holdout"):
            for head, error in ((0, 0.004), (1, 0.02 if split == "seed_holdout" else 0.005)):
                rows.append(
                    {
                        "split": split,
                        "cell": "capacity",
                        "method": "contribution_norm",
                        "density": "0.25",
                        "family": "butterfly",
                        "generators": "8",
                        "head": str(head),
                        "output_relative_l2": str(error),
                        "residual_sq": str(error * error),
                        "reference_sq": "1.0",
                        "dynamic_scalars": "512",
                        "extra_work_ratio": "0.001",
                    }
                )
        manifest = {
            "protocol": {
                "gates": {
                    "oracle_worst_record_output_relative_l2": 0.01,
                    "oracle_aggregate_output_relative_l2": 0.005,
                    "max_dynamic_scalars": 512,
                    "whole_attention_measured_speedup_for_future_kernel": 1.5,
                }
            }
        }
        result = head_fallback_frontier(rows, manifest)[0]
        self.assertEqual(result["fallback_heads"], "1")
        self.assertTrue(result["quality_gate_after_oracle_head_fallback"])
        self.assertTrue(result["speed_upper_bound_gate"])
        self.assertTrue(result["oracle_hybrid_gate"])


if __name__ == "__main__":
    unittest.main()
