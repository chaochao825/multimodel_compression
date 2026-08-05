import unittest

from probe_ar_video_residual_width_write import decide, required_capture_keys


def protocol():
    return {
        "scope": {
            "prompt_ids": ["p0", "p1"],
            "cells": [
                {"layer": 1, "current_start_frame": 3, "denoising_call_index": 2}
            ],
            "event_tile_fractions": [0.05],
            "low_rank_residual_rank": 16,
        },
        "evaluation": {
            "aggregate_oracle_gate": 0.005,
            "worst_oracle_gate": 0.01,
            "minimum_relative_improvement_over_strongest_baseline": 0.2,
            "minimum_arithmetic_reduction": 1.5,
        },
    }


def summary(selector, aggregate, worst, reduction=1.6):
    return {
        "scope": "held_out",
        "selector": selector,
        "event_fraction": 0.05,
        "correction": "adaptive_rank_oracle",
        "rank": 16,
        "aggregate_relative_av_l2": aggregate,
        "worst_head_relative_av_l2": worst,
        "minimum_arithmetic_reduction": reduction,
    }


class ResidualWidthProbeTest(unittest.TestCase):
    def test_required_keys_are_cartesian_product(self) -> None:
        self.assertEqual(
            required_capture_keys(protocol()),
            {("p0", 1, 3, 2), ("p1", 1, 3, 2)},
        )

    def test_strict_pass_opens_predictor_only(self) -> None:
        rows = [
            summary("kv_deviation", 0.02, 0.03),
            summary("dense_attention_mass_oracle", 0.018, 0.03),
            summary("value_leverage_oracle", 0.017, 0.03),
            summary("residual_width_singleton_oracle", 0.004, 0.009),
        ]
        result = decide(protocol(), rows, True)
        self.assertEqual(result["classification"], "representation_pass")
        self.assertFalse(result["deployable"])

    def test_improvement_without_quality_is_mechanism_only(self) -> None:
        rows = [
            summary("kv_deviation", 0.02, 0.04),
            summary("dense_attention_mass_oracle", 0.02, 0.04),
            summary("value_leverage_oracle", 0.02, 0.04),
            summary("residual_width_singleton_oracle", 0.015, 0.03),
        ]
        result = decide(protocol(), rows, True)
        self.assertEqual(result["classification"], "mechanism_signal_only")

    def test_no_improvement_stops_direction(self) -> None:
        rows = [
            summary("kv_deviation", 0.02, 0.04),
            summary("dense_attention_mass_oracle", 0.018, 0.04),
            summary("value_leverage_oracle", 0.019, 0.04),
            summary("residual_width_singleton_oracle", 0.018, 0.04),
        ]
        result = decide(protocol(), rows, True)
        self.assertEqual(result["classification"], "null")


if __name__ == "__main__":
    unittest.main()
