from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from activation_defect_runtime import ActivationDefectController  # noqa: E402


class FakeLinearController:
    def __init__(self) -> None:
        self.block_indices = (0,)
        self.mode = "dense"

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def set_block_mode(self, block_index: int, mode: str) -> None:
        self.mode = mode


class ModeBlock(torch.nn.Module):
    def __init__(self, controller: FakeLinearController) -> None:
        super().__init__()
        self.controller = controller

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 if self.controller.mode == "dense" else 0.9
        return x + scale * (x + 1.0)


class ToyModel(torch.nn.Module):
    def __init__(self, controller: FakeLinearController) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([ModeBlock(controller)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x)


class ActivationDefectControllerTest(unittest.TestCase):
    def test_counterfactuals_do_not_change_dense_trajectory(self) -> None:
        linears = FakeLinearController()
        model = ToyModel(linears)
        controller = ActivationDefectController(
            model,
            linears,
            sampling_steps=2,
            steps=(1,),
            blocks=(0,),
            branches=(0, 1),
            forecast_scales=(0.5,),
            sample_rows=4,
        )
        try:
            controller.begin_run("toy")
            inputs = [torch.full((3, 2), float(index)) for index in range(4)]
            outputs = [model(value) for value in inputs]
            expected = [value + value + 1.0 for value in inputs]
            for actual, reference in zip(outputs, expected, strict=True):
                torch.testing.assert_close(actual, reference)
            controller.assert_complete()
            kinds = [record["kind"] for record in controller.records]
            self.assertEqual(kinds.count("Q"), 2)
            self.assertEqual(kinds.count("C"), 2)
            self.assertEqual(kinds.count("C_FORECAST"), 0)
            self.assertTrue(all(record["sample"].device.type == "cpu" for record in controller.records))
        finally:
            controller.restore()


if __name__ == "__main__":
    unittest.main()
