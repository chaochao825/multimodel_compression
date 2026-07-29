#!/usr/bin/env python3

import unittest

import torch

from content_generated_tail_core import (
    PositiveLinearTail,
    adaptive_rank_residual,
    contiguous_layout,
    group_corrections,
    layout_tokens,
    layout_tokens_padded,
    output_for_group_selection,
    semantic_layout,
    trajectory_width_selection,
)


class ContentGeneratedTailTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.heads = 2
        self.tokens = 19
        self.channels = 8
        self.rank = 4
        self.q = torch.randn(self.heads, 5, self.channels)
        self.k = torch.randn(self.heads, self.tokens, self.channels)
        self.v = torch.randn(self.heads, self.tokens, self.channels)
        self.scale = self.channels**-0.5
        self.model = PositiveLinearTail(
            self.heads,
            self.channels,
            self.rank,
            torch.ones(self.heads),
            torch.ones(self.heads),
            seed=11,
        )

    def test_all_exact_tokens_recover_dense_attention(self) -> None:
        selected = torch.arange(self.tokens).repeat(self.heads, 1)
        output, denominator = self.model(self.q, self.k, self.v, selected, self.scale)
        dense = torch.softmax(
            torch.einsum("hqd,hnd->hqn", self.q, self.k) * self.scale, dim=2
        ) @ self.v
        self.assertTrue(torch.all(denominator > 0))
        torch.testing.assert_close(output, dense, atol=2e-5, rtol=2e-5)

    def test_group_corrections_match_module_forward(self) -> None:
        layout = contiguous_layout(self.tokens, 4, self.q.device)
        groups = torch.tensor([0, 2, 4])
        tokens, valid = layout_tokens_padded(layout, groups)
        selected = tokens.repeat(self.heads, 1)
        selected_valid = valid.repeat(self.heads, 1)
        module_output, _ = self.model(
            self.q,
            self.k,
            self.v,
            selected,
            self.scale,
            selected_valid,
        )
        for head in range(self.heads):
            state = group_corrections(
                self.model,
                head,
                self.q[head],
                self.k[head],
                self.v[head],
                layout,
                self.scale,
            )
            grouped_output = output_for_group_selection(*state, groups)
            torch.testing.assert_close(
                grouped_output, module_output[head], atol=2e-5, rtol=2e-5
            )

    def test_semantic_layout_is_a_permutation_with_padding(self) -> None:
        layout = semantic_layout(
            self.q[0], self.k[0], self.v[0], 4, "value_aware"
        )
        tokens = layout.indices[layout.valid]
        self.assertEqual(tokens.numel(), self.tokens)
        self.assertEqual(tokens.unique().numel(), self.tokens)

    def test_trajectory_width_selection_respects_budget(self) -> None:
        layout = contiguous_layout(self.tokens, 4, self.q.device)
        state = group_corrections(
            self.model,
            0,
            self.q[0],
            self.k[0],
            self.v[0],
            layout,
            self.scale,
        )
        reference = torch.softmax(self.q[0] @ self.k[0].T * self.scale, dim=1) @ self.v[0]
        selected = trajectory_width_selection(reference, *state, budget=3, add_chunk=1)
        self.assertEqual(selected.numel(), 3)
        self.assertEqual(selected.unique().numel(), 3)

        selected_rank_aware = trajectory_width_selection(
            reference, *state, budget=3, add_chunk=1, post_rank=2
        )
        self.assertEqual(selected_rank_aware.numel(), 3)
        self.assertEqual(selected_rank_aware.unique().numel(), 3)

    def test_adaptive_rank_residual_reaches_zero_at_full_rank(self) -> None:
        defect = torch.randn(5, 8)
        residual = adaptive_rank_residual(defect, 5)
        self.assertLess(float(residual.norm()), 1e-5)


if __name__ == "__main__":
    unittest.main()
