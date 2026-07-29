#!/usr/bin/env python3
"""Unit tests for same-test content-tail error accounting."""

from __future__ import annotations

import unittest

from analyze_content_generated_tail import (
    add_squared_error_accounting,
    aggregate_record_subset,
)


def record(variant: str, route: str, sample: str, residual: float) -> dict[str, str]:
    return {
        "model_variant": variant,
        "route": route,
        "sample_id": sample,
        "reference_sq": "100.0",
        "content_residual_sq": str(residual * 4),
        "post_adaptive_rank16_residual_sq": str(residual * 2),
        "post_adaptive_rank16_per_tile_residual_sq": str(residual),
        "post_adaptive_rank16_output_relative_l2": str((residual * 2 / 100) ** 0.5),
        "post_adaptive_rank16_worst_tile_relative_l2": str((residual / 100) ** 0.5),
    }


class SameTestAccountingTest(unittest.TestCase):
    def test_subset_aggregation_uses_only_requested_samples(self) -> None:
        data = [
            record("capacity", "oracle", "test", 1.0),
            record("capacity", "oracle", "other", 100.0),
        ]
        result = aggregate_record_subset(
            data,
            variant="capacity",
            route="oracle",
            sample_ids={"test"},
            stage="capacity_same_test",
            evidence="unit_test",
        )
        self.assertEqual(result["records"], 1)
        self.assertAlmostEqual(result["per_tile_rank16_output_relative_l2"], 0.1)

    def test_squared_error_increments_sum_to_proxy(self) -> None:
        stages = [
            {"per_tile_rank16_output_relative_l2": 0.10},
            {"per_tile_rank16_output_relative_l2": 0.11},
            {"per_tile_rank16_output_relative_l2": 0.20},
        ]
        output = add_squared_error_accounting(stages)
        fractions = [row["incremental_squared_error_fraction_of_proxy"] for row in output]
        self.assertAlmostEqual(sum(fractions), 1.0)
        self.assertAlmostEqual(fractions[0], 0.25)

    def test_missing_sample_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_record_subset(
                [record("capacity", "oracle", "test", 1.0)],
                variant="capacity",
                route="oracle",
                sample_ids={"test", "missing"},
                stage="capacity_same_test",
                evidence="unit_test",
            )


if __name__ == "__main__":
    unittest.main()
