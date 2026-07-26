#!/usr/bin/env python3
"""Unit tests for alternating low-rank-tail-aware sparse routing."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("tail-aware sparse tests require torch") from error

from probe_tail_aware_sparse_router import TileData, residual_for_selections, tail_aware_route


class TailAwareSparseRouterTests(unittest.TestCase):
    def test_residual_uses_selected_key_renormalization(self) -> None:
        contributions = torch.tensor([[[0.1]], [[0.0]], [[0.9]]])
        mass = torch.tensor([[0.1], [0.8], [0.1]])
        tile = TileData(contributions, mass, contributions.sum(dim=0))
        residual = residual_for_selections((tile,), (torch.tensor([0]),))
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual)))

    def test_accepted_alternation_never_worsens_initial_objective(self) -> None:
        generator = torch.Generator().manual_seed(19)
        tiles = []
        for _ in range(2):
            attention = torch.softmax(torch.randn(8, 16, generator=generator), dim=1)
            value = torch.randn(16, 6, generator=generator)
            blocks = attention.reshape(8, 4, 4)
            values = value.reshape(4, 4, 6)
            contributions = torch.einsum("qbk,bkd->bqd", blocks, values)
            mass = blocks.sum(dim=2).T
            tiles.append(TileData(contributions, mass, attention @ value))
        result = tail_aware_route(tuple(tiles), budget=2, rank=2, iterations=2)
        self.assertLessEqual(result.final_relative_l2, result.initial_relative_l2 + 1e-9)
        self.assertEqual(len(result.selections), 2)


if __name__ == "__main__":
    unittest.main()
