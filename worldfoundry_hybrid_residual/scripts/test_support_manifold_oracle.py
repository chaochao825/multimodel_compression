#!/usr/bin/env python3

import unittest

import torch

from support_manifold_oracle_core import (
    GroupProblem,
    adaptive_tail,
    assemble_defect,
    atom_statistics,
    contiguous_partition,
    initial_selections,
    motion_path_selection,
    optimize_support,
    support_cost,
    thw_partition,
)


class SupportManifoldCoreTests(unittest.TestCase):
    def test_shifted_partition_covers_tokens_once(self) -> None:
        partition = contiguous_partition(19, 8, offset=4)
        covered = partition.indices[partition.valid]
        self.assertEqual(covered.numel(), 19)
        self.assertTrue(torch.equal(covered.sort().values, torch.arange(19)))
        self.assertEqual(partition.execution_width, 8)

    def test_thw_partition_preserves_flattened_grid(self) -> None:
        partition = thw_partition((2, 3, 5), tile_height=2, tile_width=2)
        covered = partition.indices[partition.valid]
        self.assertTrue(torch.equal(covered.sort().values, torch.arange(30)))
        self.assertEqual(tuple(partition.coordinates.shape), (12, 3))

    def test_atom_contributions_reconstruct_dense_output(self) -> None:
        torch.manual_seed(1)
        attention = torch.softmax(torch.randn(5, 19), dim=1)
        value = torch.randn(19, 7)
        partition = contiguous_partition(19, 8, offset=4)
        stats = atom_statistics(attention, value, partition)
        self.assertTrue(
            torch.allclose(stats.contributions.sum(dim=0), attention @ value, atol=1e-6)
        )
        self.assertTrue(torch.allclose(stats.mass.sum(dim=0), torch.ones(5), atol=1e-6))

    def test_rank_aware_search_never_regresses_best_initializer(self) -> None:
        torch.manual_seed(2)
        attention = torch.softmax(torch.randn(8, 24), dim=1)
        value = torch.randn(24, 10)
        reference = attention @ value
        stats = atom_statistics(attention, value, contiguous_partition(24, 4))
        group = GroupProblem(reference, stats, budget=3)
        groups = (group,)
        initial_energy = []
        for method in ("contribution_norm", "mass"):
            defect = assemble_defect(groups, initial_selections(groups, method))
            _, residual, _ = adaptive_tail(defect, rank=2)
            initial_energy.append(float(residual.square().sum()))
        result = optimize_support(groups, rank=2, alternations=2, swap_steps=2, shortlist=3)
        self.assertLessEqual(result.residual_sq, min(initial_energy) + 1e-6)
        self.assertAlmostEqual(result.initial_residual_sq, min(initial_energy), places=5)

    def test_motion_path_and_cost_obey_budget(self) -> None:
        torch.manual_seed(3)
        attention = torch.softmax(torch.randn(8, 32), dim=1)
        value = torch.randn(32, 6)
        partition = thw_partition((2, 4, 4), tile_height=2, tile_width=2)
        stats = atom_statistics(attention, value, partition)
        group = GroupProblem(attention @ value, stats, budget=2)
        selected = motion_path_selection(group, temporal=2, tile_rows=2, tile_columns=2)
        self.assertEqual(selected.numel(), 2)
        cost = support_cost((group,), (selected,), key_tokens=32)
        self.assertEqual(cost["kernel_tiles"], 2)
        self.assertAlmostEqual(float(cost["execution_density"]), 0.25)


if __name__ == "__main__":
    unittest.main()
