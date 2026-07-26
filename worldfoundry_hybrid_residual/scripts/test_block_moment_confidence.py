from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_block_moment_confidence import summarize_acceptance


class BlockMomentConfidenceTests(unittest.TestCase):
    def test_fallback_work_and_quality_only_use_accepted_records(self) -> None:
        rows = [
            {
                "router_selected_mass_proxy": "0.99",
                "residual_sq": "1.0",
                "reference_sq": "10000.0",
                "output_relative_l2": "0.01",
                "attention_work_ratio": "0.25",
                "head_role": "localized",
            },
            {
                "router_selected_mass_proxy": "0.50",
                "residual_sq": "900.0",
                "reference_sq": "10000.0",
                "output_relative_l2": "0.30",
                "attention_work_ratio": "0.25",
                "head_role": "diffuse",
            },
        ]
        summary = summarize_acceptance(rows, 0.9)
        self.assertEqual(summary["accepted_records"], 1)
        self.assertAlmostEqual(float(summary["aggregate_output_relative_l2"]), 0.01)
        self.assertAlmostEqual(
            float(summary["fallback_adjusted_arithmetic_speedup"]), 1.6
        )


if __name__ == "__main__":
    unittest.main()
