import math
import unittest

import torch

from ar_video_residual_memory_core import (
    adaptive_rank_projection,
    arithmetic_reduction,
    attention_from_representatives,
    build_representatives,
    dense_attention,
    fit_output_basis,
    make_recency_plan,
    phase_align_keys_for_temporal_summaries,
    project_onto_output_basis,
    relative_l2_by_head,
    select_residual_event_tiles,
)


class TemporalPlanTest(unittest.TestCase):
    def test_recency_plan_is_disjoint_complete_and_bounded(self) -> None:
        plan = make_recency_plan(15, sink_frames=3, recent_frames=3, max_summary_groups=3)
        self.assertEqual(plan.exact_frames, (0, 1, 2, 12, 13, 14))
        self.assertEqual(plan.summary_groups, ((3, 4, 5, 6, 7, 8), (9, 10), (11,)))
        self.assertEqual(plan.covered_frames, tuple(range(15)))

    def test_no_middle_frames_needs_no_summary(self) -> None:
        plan = make_recency_plan(6, sink_frames=3, recent_frames=3, max_summary_groups=4)
        self.assertEqual(plan.exact_frames, tuple(range(6)))
        self.assertEqual(plan.summary_groups, ())


class SharedNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_single_frame_groups_are_exact(self) -> None:
        frames, spatial, heads, dim = 5, 4, 2, 3
        key = torch.randn(frames, spatial, heads, dim)
        value = torch.randn(frames, spatial, heads, dim)
        query = torch.randn(6, heads, dim)
        plan = make_recency_plan(frames, 1, 1, max_summary_groups=3)
        reps = build_representatives(key, value, plan)
        dense = dense_attention(query, key.flatten(0, 1), value.flatten(0, 1))
        compressed = attention_from_representatives(query, reps)
        torch.testing.assert_close(compressed, dense, atol=2e-6, rtol=2e-6)

    def test_phase_aligned_summary_removes_known_temporal_rope(self) -> None:
        frames, height, width, heads, dim = 4, 2, 2, 2, 6
        positions = torch.arange(8).float()[:, None]
        base_angles = torch.tensor([[0.17, 0.29, 0.41]])
        rope_freqs = torch.polar(torch.ones(8, 3), positions * base_angles)
        canonical = torch.view_as_complex(
            torch.randn(height * width, heads, dim).reshape(
                height * width, heads, dim // 2, 2
            )
        )
        key_complex = []
        for frame in range(frames):
            spatial_multipliers = []
            for row in range(height):
                for column in range(width):
                    spatial_multipliers.append(
                        torch.stack(
                            [rope_freqs[frame, 0], rope_freqs[row, 1], rope_freqs[column, 2]]
                        )
                    )
            multiplier = torch.stack(spatial_multipliers)
            key_complex.append(canonical * multiplier[:, None, :])
        key = torch.view_as_real(torch.stack(key_complex)).flatten(-2)
        plan = make_recency_plan(frames, sink_frames=1, recent_frames=1, max_summary_groups=1)
        aligned = phase_align_keys_for_temporal_summaries(
            key,
            plan,
            absolute_frame_ids=[0, 1, 2, 3],
            height=height,
            width=width,
            rope_freqs=rope_freqs,
        )

        aligned_complex = torch.view_as_complex(
            aligned.reshape(frames, height * width, heads, dim // 2, 2)
        )
        torch.testing.assert_close(aligned_complex[1], aligned_complex[2])
        value = torch.randn_like(key)
        reps = build_representatives(key, value, plan, summary_key=aligned)
        torch.testing.assert_close(reps.key[-height * width :], aligned[1])

    def test_identical_keys_use_log_multiplicity_exactly(self) -> None:
        frames, spatial, heads, dim = 6, 3, 2, 4
        spatial_key = torch.randn(1, spatial, heads, dim)
        key = spatial_key.expand(frames, -1, -1, -1).clone()
        value = torch.randn(frames, spatial, heads, dim)
        query = torch.randn(5, heads, dim)
        plan = make_recency_plan(frames, 0, 0, max_summary_groups=1)
        reps = build_representatives(key, value, plan)
        self.assertTrue(torch.allclose(reps.log_multiplicity, torch.full((spatial,), math.log(frames))))
        dense = dense_attention(query, key.flatten(0, 1), value.flatten(0, 1))
        compressed = attention_from_representatives(query, reps)
        torch.testing.assert_close(compressed, dense, atol=2e-6, rtol=2e-6)

    def test_event_tokens_are_removed_from_summary_counts(self) -> None:
        frames, spatial, heads, dim = 4, 4, 1, 2
        key = torch.randn(frames, spatial, heads, dim)
        value = torch.randn(frames, spatial, heads, dim)
        plan = make_recency_plan(frames, 0, 0, max_summary_groups=1)
        mask = torch.zeros(frames, spatial, dtype=torch.bool)
        mask[0, :2] = True
        reps = build_representatives(key, value, plan, mask)
        self.assertEqual(reps.exact_token_count, 2)
        self.assertEqual(reps.summary_token_count, spatial)
        sorted_logs = torch.sort(reps.log_multiplicity).values
        expected = torch.tensor([0.0, 0.0, math.log(3), math.log(3), math.log(4), math.log(4)])
        torch.testing.assert_close(sorted_logs, expected)


class ResidualTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)

    def test_event_selection_uses_bounded_regular_tiles(self) -> None:
        key = torch.randn(8, 16, 2, 4)
        value = torch.randn(8, 16, 2, 4)
        plan = make_recency_plan(8, 1, 1, max_summary_groups=2)
        mask = select_residual_event_tiles(key, value, plan, tile_size=4, tile_fraction=0.25)
        self.assertFalse(bool(mask[0].any()))
        self.assertFalse(bool(mask[-1].any()))
        self.assertEqual(int(mask.sum().item()) % 4, 0)

    def test_adaptive_rank_is_monotone_and_full_rank_is_exact(self) -> None:
        target = torch.randn(12, 2, 6)
        estimate = torch.randn_like(target)
        defect = target - estimate
        errors = []
        for rank in (0, 1, 3, 6):
            corrected = estimate + adaptive_rank_projection(defect, rank)
            errors.append(float(relative_l2_by_head(target, corrected).mean()))
        self.assertTrue(all(a >= b - 1e-7 for a, b in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 2e-6)

    def test_frozen_basis_reconstructs_shared_output_subspace(self) -> None:
        heads, value_dim, rank = 2, 7, 3
        true_basis = torch.linalg.qr(torch.randn(heads, value_dim, rank))[0]
        defects = []
        for _ in range(3):
            coefficients = torch.randn(10, heads, rank)
            defects.append(torch.einsum("qhr,hdr->qhd", coefficients, true_basis))
        fitted = fit_output_basis(defects[:2], rank)
        projected = project_onto_output_basis(defects[2], fitted)
        torch.testing.assert_close(projected, defects[2], atol=2e-5, rtol=2e-5)

    def test_arithmetic_reduction(self) -> None:
        self.assertEqual(arithmetic_reduction(1500, 750), 2.0)


if __name__ == "__main__":
    unittest.main()
