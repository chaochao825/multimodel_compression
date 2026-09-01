from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_wan_rcm_vae_schedule as evaluator  # noqa: E402


class VaeScheduleEvaluatorTest(unittest.TestCase):
    def test_downloaded_exp053_closes_as_exactness_null(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "wan_rcm_vae_schedule_exp053_20260901"
        )
        result = evaluator.evaluate(root)
        self.assertEqual(result["outcome"], "exactness-null")
        self.assertEqual(result["selected_chunk_size"], 4)
        self.assertTrue(result["guards"]["f17_selected_candidate_exact"])
        self.assertFalse(result["guards"]["f81_all_exact"])
        self.assertFalse(result["guards"]["component_speed_pass"])
        self.assertFalse(result["guards"]["projected_speed_pass"])
        self.assertEqual(len(result["f81_prompts"]), 4)


if __name__ == "__main__":
    unittest.main()
