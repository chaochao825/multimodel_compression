from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_block_moment_marginal import (
    combine_exact_and_moments,
    padded_groups,
)


class BlockMomentMarginalTests(unittest.TestCase):
    def test_singleton_groups_reproduce_dense_attention(self) -> None:
        torch.manual_seed(3)
        queries = torch.randn(5, 4)
        keys = torch.randn(8, 4)
        values = torch.randn(8, 4)
        moments = padded_groups(keys, values, 1)
        selected = torch.zeros(2, dtype=torch.bool)
        actual, _, _, _ = combine_exact_and_moments(
            queries,
            keys,
            values,
            selected,
            4,
            moments,
            1,
            0.5,
            "diag_gaussian",
        )
        expected = torch.softmax(queries @ keys.T * 0.5, dim=1) @ values
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_constant_blocks_are_exact_with_centroids(self) -> None:
        queries = torch.tensor([[0.5, -1.0], [1.0, 0.25]])
        keys = torch.tensor([[1.0, 2.0]] * 4 + [[-2.0, 0.5]] * 4)
        values = torch.tensor([[3.0, -1.0]] * 4 + [[0.5, 2.0]] * 4)
        moments = padded_groups(keys, values, 4)
        selected = torch.zeros(2, dtype=torch.bool)
        actual, _, _, _ = combine_exact_and_moments(
            queries,
            keys,
            values,
            selected,
            4,
            moments,
            4,
            2**-0.5,
            "centroid",
        )
        expected = torch.softmax(queries @ keys.T * (2**-0.5), dim=1) @ values
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_all_selected_blocks_reproduce_dense_attention(self) -> None:
        torch.manual_seed(5)
        queries = torch.randn(3, 4)
        keys = torch.randn(7, 4)
        values = torch.randn(7, 4)
        moments = padded_groups(keys, values, 2)
        selected = torch.ones(2, dtype=torch.bool)
        actual, _, selected_keys, tail_groups = combine_exact_and_moments(
            queries,
            keys,
            values,
            selected,
            4,
            moments,
            2,
            0.5,
            "centroid",
        )
        expected = torch.softmax(queries @ keys.T * 0.5, dim=1) @ values
        self.assertEqual(selected_keys, 7)
        self.assertEqual(tail_groups, 0)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
