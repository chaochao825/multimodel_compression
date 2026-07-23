from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_tri_mode_oracle import route_decision  # noqa: E402


class RouteDecisionTest(unittest.TestCase):
    def test_stop_condition_overrides_further_schedule_search(self) -> None:
        stop, adaptive, next_gate = route_decision(1.19, 1.18)
        self.assertTrue(stop)
        self.assertFalse(adaptive)
        self.assertTrue(next_gate.startswith("stop schedule-combination search"))

    def test_large_oracle_gap_selects_adaptive_controller(self) -> None:
        stop, adaptive, next_gate = route_decision(1.4, 1.08)
        self.assertFalse(stop)
        self.assertTrue(adaptive)
        self.assertIn("sample-adaptive controller", next_gate)

    def test_viable_universal_route_proceeds_to_rollout(self) -> None:
        stop, adaptive, next_gate = route_decision(1.3, 1.2)
        self.assertFalse(stop)
        self.assertFalse(adaptive)
        self.assertEqual(next_gate, "verify beam-search combinations with real rollouts")


if __name__ == "__main__":
    unittest.main()
