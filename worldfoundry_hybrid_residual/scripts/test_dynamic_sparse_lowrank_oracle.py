#!/usr/bin/env python3
"""Unit tests for the output-aware dynamic sparse plus low-rank oracle."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("dynamic sparse oracle tests require torch") from error

from probe_dynamic_sparse_lowrank_oracle import (
    aligned_query_tile_starts,
    block_output_contributions,
    dense_output_greedy_order,
    mask_jaccard,
    renorm_output_greedy_order,
    residual_after_basis,
    right_singular_basis,
    subspace_overlap,
)


class DynamicSparseLowRankOracleTests(unittest.TestCase):
    def test_query_tiles_are_aligned_unique_and_in_range(self) -> None:
        starts = aligned_query_tile_starts(tokens=32760, tile_size=64, tile_count=8)
        self.assertEqual(len(starts), len(set(starts)))
        self.assertTrue(all(start % 64 == 0 for start in starts))
        self.assertLessEqual(starts[-1] + 64, 32760)

    def test_block_contributions_sum_to_dense_attention_output(self) -> None:
        generator = torch.Generator().manual_seed(7)
        logits = torch.randn(5, 11, generator=generator)
        attention = torch.softmax(logits, dim=1)
        value = torch.randn(11, 4, generator=generator)
        contributions, mass, padding = block_output_contributions(attention, value, 4)
        self.assertEqual(padding, 1)
        self.assertTrue(torch.allclose(contributions.sum(dim=0), attention @ value, atol=1e-6))
        self.assertTrue(torch.allclose(mass.sum(dim=0), torch.ones(5), atol=1e-6))

    def test_dense_greedy_first_step_minimizes_output_error(self) -> None:
        contributions = torch.tensor([[[10.0]], [[-9.0]], [[1.0]]])
        order = dense_output_greedy_order(contributions, budget=1)
        self.assertEqual(int(order[0]), 2)

    def test_renormalized_greedy_targets_normalized_output(self) -> None:
        contributions = torch.tensor([[[1.0]], [[0.0]], [[9.0]]])
        mass = torch.tensor([[0.1], [0.8], [0.1]])
        order = renorm_output_greedy_order(contributions, mass, budget=1)
        # Block 2 has the largest raw contribution, but block 0 alone
        # reproduces the dense output after selected-key renormalization.
        self.assertEqual(int(order[0]), 0)

    def test_right_basis_exactly_repairs_low_rank_defect(self) -> None:
        generator = torch.Generator().manual_seed(11)
        left = torch.randn(32, 3, generator=generator)
        right, _ = torch.linalg.qr(torch.randn(16, 3, generator=generator))
        defect = left @ right.T
        basis, energy = right_singular_basis(defect, rank=3)
        residual = residual_after_basis(defect, basis)
        self.assertGreater(energy, 0.99999)
        self.assertLess(float(residual.norm() / defect.norm()), 1e-5)
        self.assertGreater(subspace_overlap(right, basis), 0.99999)

    def test_mask_jaccard_averages_query_tiles(self) -> None:
        left = torch.tensor([[True, True, False], [True, False, False]])
        right = torch.tensor([[True, False, True], [True, False, False]])
        self.assertAlmostEqual(mask_jaccard(left, right), (1 / 3 + 1) / 2)


if __name__ == "__main__":
    unittest.main()
