from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_wan_rcm_vae_cudagraph as evaluator  # noqa: E402


class RecordedResultTest(unittest.TestCase):
    def test_recorded_exp055_closes_as_speed_boundary(self) -> None:
        result_root = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "wan_rcm_vae_cudagraph_exp055_20260902"
        )
        result = evaluator.evaluate(result_root)
        self.assertEqual(result["outcome"], "speed-boundary")
        self.assertTrue(result["guards"]["f17_exact_and_stale_free"])
        self.assertTrue(result["guards"]["f81_component_exact"])
        self.assertTrue(result["guards"]["f81_request_exact"])
        self.assertTrue(result["guards"]["component_speed_ok"])
        self.assertTrue(result["guards"]["projected_speed_ok"])
        self.assertFalse(result["guards"]["measured_speed_ok"])


if __name__ == "__main__":
    unittest.main()
