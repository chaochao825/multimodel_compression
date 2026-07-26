from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_attention_head_stability import classify_head, pearson, summarize_pair


def feature(entropy: float, geometry: float, top64: float, participation: float) -> dict[str, float]:
    return {
        "actual_normalized_entropy_mean": entropy,
        "geometry_mass_mean": geometry,
        "actual_top64_mass_mean": top64,
        "actual_participation_support_fraction_mean": participation,
    }


class AttentionHeadStabilityTests(unittest.TestCase):
    def test_rule_based_classes_cover_three_regimes(self) -> None:
        self.assertEqual(classify_head(feature(0.4, 0.9, 0.8, 0.01)), "localized")
        self.assertEqual(classify_head(feature(0.7, 0.6, 0.4, 0.10)), "transitional")
        self.assertEqual(classify_head(feature(0.9, 0.2, 0.1, 0.60)), "diffuse")

    def test_identical_feature_vectors_have_unit_correlation(self) -> None:
        self.assertEqual(pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_stable_head_classes_pass_pilot_gate(self) -> None:
        left = {
            0: feature(0.4, 0.9, 0.8, 0.01),
            1: feature(0.7, 0.6, 0.4, 0.10),
            2: feature(0.9, 0.2, 0.1, 0.60),
        }
        right = {
            0: feature(0.42, 0.88, 0.78, 0.02),
            1: feature(0.69, 0.62, 0.42, 0.12),
            2: feature(0.91, 0.18, 0.08, 0.58),
        }
        pair, rows = summarize_pair(
            "a",
            left,
            "b",
            right,
            entropy_correlation_gate=0.90,
            geometry_correlation_gate=0.90,
            class_agreement_gate=0.75,
            localized_jaccard_gate=0.50,
        )
        self.assertTrue(pair["router_class_pilot_go"])
        self.assertEqual(pair["class_agreement"], 1.0)
        self.assertEqual(len(rows), 3)

    def test_head_id_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "head IDs"):
            summarize_pair(
                "a",
                {0: feature(0.4, 0.9, 0.8, 0.01)},
                "b",
                {1: feature(0.4, 0.9, 0.8, 0.01)},
                entropy_correlation_gate=0.90,
                geometry_correlation_gate=0.90,
                class_agreement_gate=0.75,
                localized_jaccard_gate=0.50,
            )


if __name__ == "__main__":
    unittest.main()
