from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from probes.metrics import residual_concentration, singular_value_metrics
from probes.transport import (
    apply_global_bccb,
    apply_shift_basis,
    best_integer_shift,
    fit_global_bccb,
    fit_shift_basis,
    local_offsets,
    shift_grid,
    warp_grid_bilinear,
)


class ProbeMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(123)
        self.source = self.rng.normal(size=(8, 10, 6))

    def test_integer_shift_recovery(self) -> None:
        target = shift_grid(self.source, 1, -1, cyclic=True)
        dy, dx, error = best_integer_shift(
            self.source,
            target,
            max_shift=2,
            cyclic=True,
        )
        self.assertEqual((dy, dx), (1, -1))
        self.assertLess(error, 1e-10)

    def test_backward_flow_warp_recovers_integer_translation(self) -> None:
        target = shift_grid(self.source, 1, -2, cyclic=False)
        backward_flow = np.zeros((*self.source.shape[:2], 2))
        backward_flow[..., 0] = 2.0
        backward_flow[..., 1] = -1.0
        prediction = warp_grid_bilinear(self.source, backward_flow)
        self.assertTrue(np.allclose(prediction, target))

    def test_local_bccb_recovers_shift(self) -> None:
        target = shift_grid(self.source, -1, 1, cyclic=True)
        prediction, weights = fit_shift_basis(
            self.source,
            target,
            local_offsets(1),
            cyclic=True,
            ridge=1e-10,
        )
        self.assertLess(np.linalg.norm(target - prediction), 1e-7)
        self.assertEqual(weights.shape, (9,))
        replay = apply_shift_basis(
            self.source,
            local_offsets(1),
            weights,
            cyclic=True,
        )
        self.assertLess(np.linalg.norm(target - replay), 1e-7)

    def test_global_bccb_recovers_shift(self) -> None:
        target = shift_grid(self.source, 2, -3, cyclic=True)
        prediction, kernel = fit_global_bccb(
            self.source,
            target,
            ridge=1e-12,
        )
        self.assertLess(np.linalg.norm(target - prediction), 1e-7)
        self.assertEqual(kernel.shape, self.source.shape[:2])
        replay = apply_global_bccb(self.source, kernel)
        self.assertLess(np.linalg.norm(target - replay), 1e-7)

    def test_rank_metrics(self) -> None:
        left = self.rng.normal(size=(40, 3))
        right = self.rng.normal(size=(3, 20))
        metrics = singular_value_metrics(left @ right, [1, 2, 3, 4])
        self.assertGreater(metrics["energy_at_rank"]["3"], 0.999999)
        self.assertLess(metrics["stable_rank"], 3.01)

    def test_event_concentration(self) -> None:
        residual = np.zeros((8, 8, 4), dtype=np.float64)
        residual[2:4, 4:6] = 10.0
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:4, 4:6] = True
        metrics = residual_concentration(
            residual,
            block_size=2,
            fractions=[0.1],
            event_mask=mask,
        )
        top = metrics["top"]["0.1000"]
        self.assertGreater(top["energy_ratio"], 0.999999)
        self.assertGreater(top["event_recall"], 0.999999)


if __name__ == "__main__":
    unittest.main()
