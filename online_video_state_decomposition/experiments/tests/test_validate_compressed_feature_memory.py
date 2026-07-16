from __future__ import annotations

import unittest

from experiments.probes.validate_compressed_feature_memory import (
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


if __name__ == "__main__":
    unittest.main()
