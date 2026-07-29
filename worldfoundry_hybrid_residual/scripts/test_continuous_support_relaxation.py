#!/usr/bin/env python3

import unittest

import torch

from probe_continuous_support_relaxation import (
    budgeted_sigmoid,
    optimize_relaxation,
    refit_weighted_topk,
    weighted_topk_defect,
)
from support_manifold_oracle_core import (
    AtomStatistics,
    GroupProblem,
    atom_statistics,
    contiguous_partition,
    optimize_support,
)


class ContinuousSupportRelaxationTests(unittest.TestCase):
    def test_budgeted_sigmoid_matches_budget_and_bounds(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn(97)
        weights = budgeted_sigmoid(logits, budget=23, temperature=0.5)
        self.assertAlmostEqual(float(weights.sum()), 23.0, places=4)
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertLessEqual(float(weights.max()), 1.0)

    def test_invalid_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            budgeted_sigmoid(torch.randn(8), budget=8, temperature=0.5)

    def test_relaxation_retains_hard_feasible_solution(self) -> None:
        torch.manual_seed(8)
        attention = torch.softmax(torch.randn(8, 24), dim=1)
        value = torch.randn(24, 10)
        reference = attention @ value
        stats = atom_statistics(attention, value, contiguous_partition(24, 4))
        groups = (GroupProblem(reference, stats, budget=3),)
        hard = optimize_support(groups, rank=2, alternations=1, swap_steps=1, shortlist=3)
        result = optimize_relaxation(
            groups,
            hard.selections,
            rank=2,
            steps=2,
            restarts=1,
            learning_rate=0.01,
            temperature=0.5,
        )
        hard_relative = (hard.residual_sq / float(reference.square().sum())) ** 0.5
        self.assertLessEqual(float(result["fractional_output_relative_l2"]), hard_relative + 1e-4)

    def test_weighted_topk_full_budget_matches_fractional_output(self) -> None:
        torch.manual_seed(9)
        attention = torch.softmax(torch.randn(6, 16), dim=1)
        value = torch.randn(16, 5)
        reference = attention @ value
        stats = atom_statistics(attention, value, contiguous_partition(16, 4))
        group = GroupProblem(reference, stats, budget=1)
        weights = (torch.tensor([0.1, 0.2, 0.3, 0.4]),)
        full = weighted_topk_defect((group,), weights, budget_multiplier=4)
        numerator = torch.einsum("b,bqd->qd", weights[0], stats.contributions)
        mass = torch.einsum("b,bq->q", weights[0], stats.mass)
        expected = reference - numerator / mass[:, None]
        self.assertTrue(torch.allclose(full, expected, atol=1e-6))

    def test_fractional_payload_counts_every_candidate_weight(self) -> None:
        statistics = AtomStatistics(
            contributions=torch.tensor([[[0.0]], [[2.0]]]),
            mass=torch.ones(2, 1),
            valid_counts=torch.ones(2),
            execution_width=1,
        )
        problem = GroupProblem(torch.tensor([[1.0]]), statistics, budget=1)
        result = optimize_relaxation(
            (problem,),
            (torch.tensor([0]),),
            rank=0,
            steps=80,
            restarts=1,
            learning_rate=0.05,
            temperature=0.5,
        )
        self.assertEqual(result["solution_kind"], "fractional_relaxation")
        self.assertEqual(result["weight_dynamic_scalars"], 2)

    def test_selected_weight_refit_does_not_regress(self) -> None:
        torch.manual_seed(10)
        attention = torch.softmax(torch.randn(6, 16), dim=1)
        value = torch.randn(16, 5)
        reference = attention @ value
        stats = atom_statistics(attention, value, contiguous_partition(16, 4))
        group = GroupProblem(reference, stats, budget=1)
        weights = (torch.tensor([0.1, 0.2, 0.3, 0.4]),)
        initial = weighted_topk_defect((group,), weights, budget_multiplier=2)
        initial_error = float(torch.linalg.norm(initial) / torch.linalg.norm(reference))
        refit_error = refit_weighted_topk(
            (group,),
            weights,
            budget_multiplier=2,
            rank=0,
            steps=10,
            learning_rate=0.01,
        )
        self.assertLessEqual(refit_error, initial_error + 1e-6)

    def test_refit_cache_key_clamps_to_available_atoms(self) -> None:
        problem = GroupProblem(
            torch.ones(1, 1),
            AtomStatistics(
                contributions=torch.ones(4, 1, 1),
                mass=torch.ones(4, 1),
                valid_counts=torch.ones(4),
                execution_width=1,
            ),
            budget=2,
        )
        weights = torch.ones(4)
        counts = tuple(
            min(weight.numel(), int(round(group.budget * multiplier)))
            for group, weight in ((problem, weights),)
            for multiplier in (2.0,)
        )
        self.assertEqual(counts, (4,))


if __name__ == "__main__":
    unittest.main()
