from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from trajectory_budget_runtime import (  # noqa: E402
    TriModeBlockController,
    TriModeSchedule,
)


class FakeLinearController:
    def __init__(self, block_count: int) -> None:
        self.block_indices = tuple(range(block_count))
        self.modes = {index: "dense" for index in self.block_indices}
        self.history: list[tuple[int, str]] = []

    def set_mode(self, mode: str) -> None:
        for block in self.block_indices:
            self.modes[block] = mode

    def set_block_mode(self, block_index: int, mode: str) -> None:
        self.modes[block_index] = mode
        self.history.append((block_index, mode))

    def reset_runtime_stats(self) -> None:
        self.history.clear()


class AddBlock(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.value


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([AddBlock(1.0), AddBlock(2.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TriModeScheduleTest(unittest.TestCase):
    def test_conflicting_actions_are_rejected(self) -> None:
        payload = {
            "name": "conflict",
            "overrides": [
                {
                    "action": "Q",
                    "steps": [0],
                    "blocks": [0],
                    "branches": [0],
                },
                {
                    "action": "C",
                    "steps": [0],
                    "blocks": [0],
                    "branches": [0],
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "conflicting actions"):
            TriModeSchedule.from_dict(payload, sampling_steps=2, block_count=2)

    def test_action_counts_include_cfg_branches(self) -> None:
        schedule = TriModeSchedule.from_dict(
            {
                "name": "one-cell",
                "overrides": [
                    {
                        "action": "Q",
                        "step_start": 0,
                        "step_end": 1,
                        "block_start": 0,
                        "block_end": 0,
                    }
                ],
            },
            sampling_steps=2,
            block_count=2,
        )
        self.assertEqual(schedule.action_counts(), {"C": 0, "D": 4, "Q": 4})


class TriModeControllerTest(unittest.TestCase):
    def test_cache_fallback_is_branch_specific_and_actions_are_exclusive(self) -> None:
        model = ToyModel()
        linears = FakeLinearController(block_count=2)
        schedule = TriModeSchedule.from_dict(
            {
                "name": "cache-then-quant",
                "overrides": [
                    {
                        "action": "C",
                        "step_start": 0,
                        "step_end": 1,
                        "blocks": [0],
                    },
                    {
                        "action": "Q",
                        "steps": [1],
                        "blocks": [1],
                        "branches": [1],
                    },
                ],
            },
            sampling_steps=2,
            block_count=2,
        )
        controller = TriModeBlockController(model, linears, sampling_steps=2)
        try:
            controller.begin(schedule)
            outputs = [model(torch.zeros(2)) for _ in range(4)]
            for output in outputs:
                torch.testing.assert_close(output, torch.full((2,), 3.0))
            stats = controller.stats()
            self.assertEqual(stats["tri_mode_requested_cache"], 4)
            self.assertEqual(stats["tri_mode_requested_quant"], 1)
            self.assertEqual(stats["tri_mode_requested_dense"], 3)
            self.assertEqual(stats["tri_mode_executed_cache"], 2)
            self.assertEqual(stats["tri_mode_executed_quant"], 1)
            self.assertEqual(stats["tri_mode_executed_dense"], 5)
            self.assertEqual(stats["tri_mode_cache_fallbacks"], 2)
            self.assertEqual(stats["tri_mode_cache_refreshes"], 2)
            self.assertEqual(stats["tri_mode_max_cache_age"], 1)
            self.assertEqual(controller.call_index, 4)
            controller.assert_complete()
            quant_events = [
                event for event in controller.events if event.executed_action == "Q"
            ]
            self.assertEqual(len(quant_events), 1)
            self.assertEqual((quant_events[0].step, quant_events[0].branch), (1, 1))
        finally:
            controller.restore()

    def test_more_than_two_cfg_calls_per_step_is_rejected(self) -> None:
        model = ToyModel()
        linears = FakeLinearController(block_count=2)
        schedule = TriModeSchedule.from_dict(
            {"name": "dense"}, sampling_steps=1, block_count=2
        )
        controller = TriModeBlockController(model, linears, sampling_steps=1)
        try:
            controller.begin(schedule)
            model(torch.zeros(1))
            model(torch.zeros(1))
            controller.assert_complete()
            with self.assertRaisesRegex(RuntimeError, "more than the expected"):
                model(torch.zeros(1))
        finally:
            controller.restore()


if __name__ == "__main__":
    unittest.main()
