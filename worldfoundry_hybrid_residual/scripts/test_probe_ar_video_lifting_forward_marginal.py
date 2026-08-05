from __future__ import annotations

import unittest

from probe_ar_video_lifting_forward_marginal import forward_marginal_search


class ForwardMarginalSearchTest(unittest.TestCase):
    def test_joint_marginal_corrects_redundant_singletons(self) -> None:
        def evaluate(indices: list[int]) -> float:
            selected = set(indices)
            score = 10.0
            if 0 in selected:
                score -= 6.0
            if 1 in selected:
                score -= 5.0
            if 2 in selected:
                score -= 4.0
            if {0, 1}.issubset(selected):
                score += 6.0
            return score

        selected, singleton_scores, trajectory = forward_marginal_search(
            budget=2,
            candidate_count=3,
            evaluate=evaluate,
            denominator_sq=10.0,
        )

        self.assertEqual(sorted(range(3), key=singleton_scores.__getitem__)[:2], [0, 1])
        self.assertEqual(selected, [0, 2])
        self.assertEqual([row["selected_index"] for row in trajectory], [0, 2])

    def test_budget_must_fit_candidate_set(self) -> None:
        with self.assertRaises(ValueError):
            forward_marginal_search(
                budget=3,
                candidate_count=2,
                evaluate=lambda _: 0.0,
                denominator_sq=1.0,
            )


if __name__ == "__main__":
    unittest.main()
