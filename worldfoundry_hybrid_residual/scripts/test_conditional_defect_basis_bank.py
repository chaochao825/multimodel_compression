#!/usr/bin/env python3
"""Unit tests for position-bucketed defect basis banks."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("basis bank tests require torch") from error

from probe_conditional_defect_basis_bank import evaluate_bank, tile_group


class ConditionalDefectBasisBankTests(unittest.TestCase):
    def test_tile_groups_cover_all_tiles_contiguously(self) -> None:
        groups = [tile_group(index, 16, 4) for index in range(16)]
        self.assertEqual(groups, [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)

    def test_position_banks_repair_rotating_subspaces(self) -> None:
        generator = torch.Generator().manual_seed(23)
        left0 = torch.randn(16, 2, generator=generator)
        left1 = torch.randn(16, 2, generator=generator)
        basis, _ = torch.linalg.qr(torch.randn(8, 4, generator=generator))
        calibration = (left0 @ basis[:, :2].T, left1 @ basis[:, 2:].T)
        test = tuple(value.clone() for value in calibration)
        references = tuple(torch.ones_like(value) for value in test)
        one_bank = evaluate_bank(calibration, test, references, bank_count=1, rank=2)
        two_banks = evaluate_bank(calibration, test, references, bank_count=2, rank=2)
        self.assertGreater(float(one_bank["relative_l2"]), 1e-3)
        self.assertLess(float(two_banks["relative_l2"]), 1e-5)


if __name__ == "__main__":
    unittest.main()
