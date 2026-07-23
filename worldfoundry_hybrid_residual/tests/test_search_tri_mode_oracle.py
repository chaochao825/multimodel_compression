from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from search_tri_mode_oracle import Cell, beam_search, build_cells  # noqa: E402


class BeamSearchTest(unittest.TestCase):
    def test_cost_is_scaled_by_actual_action_calls(self) -> None:
        rows = [
            {
                "action": "C",
                "step_start": "8",
                "step_end": "8",
                "block_start": "0",
                "block_end": "2",
                "quality_loss_worst": "0.001",
                "requested_cache_mean": "6",
                "executed_cache_mean": "6",
            }
        ]
        cells = build_cells(
            rows,
            {"C": {"fractional_saving": 0.57, "action_calls": 1140.0}},
        )
        self.assertEqual(len(cells), 1)
        self.assertAlmostEqual(cells[0].saving, 0.003)

    def test_actions_at_same_cell_are_mutually_exclusive(self) -> None:
        cells = [
            Cell("Q", 0, 3, 0, 5, 0.001, 0.02),
            Cell("C", 0, 3, 0, 5, 0.002, 0.04),
            Cell("C", 4, 7, 0, 5, 0.003, 0.03),
        ]
        states = beam_search(
            cells,
            quality_loss_budget=0.01,
            beam_width=16,
            max_cells=3,
        )
        for state in states:
            positions = [cells[index].position for index in state.selected]
            self.assertEqual(len(positions), len(set(positions)))
        best = states[0]
        self.assertEqual(best.selected, (1, 2))
        self.assertAlmostEqual(best.saving, 0.07)

    def test_quality_budget_is_enforced(self) -> None:
        cells = [
            Cell("C", 0, 3, 0, 5, 0.015, 0.08),
            Cell("C", 4, 7, 0, 5, 0.010, 0.07),
        ]
        states = beam_search(
            cells,
            quality_loss_budget=0.02,
            beam_width=8,
            max_cells=2,
        )
        self.assertTrue(all(state.quality_loss <= 0.02 for state in states))
        self.assertNotIn((0, 1), [state.selected for state in states])

    def test_cfg_branches_can_be_selected_independently(self) -> None:
        cells = [
            Cell("C", 12, 12, 6, 6, 0.001, 0.01, branches=(0,)),
            Cell("C", 12, 12, 6, 6, 0.001, 0.01, branches=(1,)),
        ]
        states = beam_search(
            cells,
            quality_loss_budget=0.01,
            beam_width=8,
            max_cells=2,
        )
        self.assertIn((0, 1), [state.selected for state in states])

    def test_partially_overlapping_ranges_are_mutually_exclusive(self) -> None:
        cells = [
            Cell("Q", 0, 3, 0, 5, 0.001, 0.02),
            Cell("C", 3, 4, 5, 6, 0.001, 0.04),
            Cell("C", 4, 5, 7, 8, 0.001, 0.03),
        ]
        states = beam_search(
            cells,
            quality_loss_budget=0.01,
            beam_width=16,
            max_cells=3,
        )
        self.assertNotIn((0, 1), [state.selected for state in states])
        self.assertIn((1, 2), [state.selected for state in states])


if __name__ == "__main__":
    unittest.main()
