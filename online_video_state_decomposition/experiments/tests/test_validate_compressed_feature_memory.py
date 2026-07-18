from __future__ import annotations

import unittest

from experiments.probes.validate_compressed_feature_memory import (
    expected_feature_payload_bytes,
    selection_split_audit,
)


class CompressedFeatureMemoryValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "calibration": {"task": [0, 1]},
            "evaluation": {"task": [2, 3]},
            "excluded_prior_formal": {"task": [4, 5]},
            "reserve": {"task": [6, 7, 8, 9]},
        }

    def test_confirmation_selection_is_a_reserve_subset(self) -> None:
        checks, intersections = selection_split_audit(
            {"task_0006", "task_0008"},
            self.manifest,
            expected_samples=2,
        )

        self.assertTrue(all(checks.values()))
        self.assertEqual(intersections["selection_outside_reserve"], 0)

    def test_split_leakage_is_rejected(self) -> None:
        checks, intersections = selection_split_audit(
            {"task_0001", "task_0006"},
            self.manifest,
            expected_samples=2,
        )

        self.assertFalse(checks["selection_manifest_is_subset_of_reserve"])
        self.assertFalse(checks["selection_disjoint_from_calibration"])
        self.assertEqual(intersections["selection_calibration"], 1)
        self.assertEqual(intersections["selection_outside_reserve"], 1)

    def test_routed_payload_formula_matches_uint8_slots_and_mask(self) -> None:
        common = {
            "source_frames": 16,
            "source_tokens": 64,
            "hidden_size": 4096,
            "rank": 256,
            "dense_feature_bytes": 8_388_608,
        }
        routed = expected_feature_payload_bytes(
            "pca_r256_route_grid2_s4",
            **common,
        )
        fixed = expected_feature_payload_bytes("pca_r256_s4", **common)
        grid = expected_feature_payload_bytes("pca_r256_grid2x2", **common)
        self.assertEqual(routed, 1_048_656)
        self.assertEqual(fixed, 1_048_704)
        self.assertEqual(grid, 1_048_576)
        self.assertLess(grid, routed)
        self.assertLess(routed, fixed)


if __name__ == "__main__":
    unittest.main()
