import math
import unittest

import torch

from ar_video_causal_bccb_core import (
    build_spatial_kernel_bank,
    displacement_indices,
    exact_sink_recent_frames,
    fft_arithmetic_reduction,
    logits_from_spatial_kernel,
    make_captured_query_layout,
    pool_kernel_bank_by_relative_frame,
    project_logits_to_spatial_kernel,
    structured_attention_from_kernel_bank,
)
from ar_video_residual_memory_core import dense_attention


class LayoutAndDisplacementTest(unittest.TestCase):
    def test_layout_matches_capture_sampling(self) -> None:
        layout = make_captured_query_layout(24, 16, 4, 2)
        self.assertEqual(layout.query_frames, 3)
        self.assertEqual(layout.positions_per_frame.tolist(), [0, 1, 2, 3, 12, 13, 14, 15])
        self.assertEqual(layout.tile_ids_per_frame.tolist(), [0] * 4 + [1] * 4)

    def test_periodic_and_nonperiodic_displacements_differ_at_boundary(self) -> None:
        query = torch.tensor([0], dtype=torch.long)
        periodic, periodic_bins = displacement_indices(query, 2, 3, True)
        toeplitz, toeplitz_bins = displacement_indices(query, 2, 3, False)
        self.assertEqual(periodic_bins, 6)
        self.assertEqual(toeplitz_bins, 15)
        self.assertEqual(periodic.shape, toeplitz.shape)
        self.assertNotEqual(periodic[0, -1].item(), toeplitz[0, -1].item())


class KernelProjectionTest(unittest.TestCase):
    def test_periodic_projection_recovers_shift_invariant_logits(self) -> None:
        width = 8
        positions = torch.arange(width, dtype=torch.long)
        phase = 2.0 * math.pi * positions.float() / width
        features = torch.stack([phase.cos(), phase.sin()], dim=-1)[:, None, :]
        groups = torch.zeros(width, dtype=torch.long)
        kernel = project_logits_to_spatial_kernel(
            features, features, positions, groups, 1, width, True
        )
        reconstructed = logits_from_spatial_kernel(
            kernel, positions, groups, 1, width, True
        )
        expected = torch.einsum("qhd,khd->hqk", features, features) / math.sqrt(2.0)
        torch.testing.assert_close(reconstructed, expected, atol=1e-6, rtol=1e-6)

    def test_relative_frame_pooling_only_shares_equal_offsets(self) -> None:
        bank = torch.arange(2 * 1 * 3 * 1 * 1, dtype=torch.float32).reshape(2, 1, 3, 1, 1)
        pooled = pool_kernel_bank_by_relative_frame(bank, [10, 11], [8, 9, 10])
        self.assertEqual(pooled.shape, bank.shape)
        self.assertEqual(pooled[0, 0, 1].item(), pooled[1, 0, 2].item())
        self.assertNotEqual(pooled[0, 0, 0].item(), pooled[0, 0, 1].item())


class StructuredAttentionTest(unittest.TestCase):
    def test_all_exact_frames_match_dense_attention(self) -> None:
        torch.manual_seed(7)
        height, width, frames, heads, dim = 2, 3, 2, 2, 4
        spatial = height * width
        query = torch.randn(spatial, heads, dim)
        key = torch.randn(frames * spatial, heads, dim)
        value = torch.randn_like(key)
        layout = make_captured_query_layout(spatial, spatial, spatial, 1)
        bank = build_spatial_kernel_bank(
            query, key, layout, frames, height, width, True, "global"
        )
        estimate = structured_attention_from_kernel_bank(
            query,
            key,
            value,
            bank,
            layout,
            frames,
            height,
            width,
            True,
            exact_frame_indices=range(frames),
        )
        expected = dense_attention(query, key, value)
        torch.testing.assert_close(estimate, expected, atol=2e-5, rtol=2e-5)

    def test_event_mask_overrides_only_selected_tokens(self) -> None:
        torch.manual_seed(11)
        height, width, frames, heads, dim = 1, 4, 2, 1, 2
        spatial = height * width
        query = torch.randn(spatial, heads, dim)
        key = torch.randn(frames * spatial, heads, dim)
        value = torch.randn_like(key)
        layout = make_captured_query_layout(spatial, spatial, spatial, 1)
        bank = torch.zeros((1, 1, frames, heads, spatial))
        mask = torch.zeros((frames, spatial), dtype=torch.bool)
        mask[1, 2] = True
        without_event = structured_attention_from_kernel_bank(
            query, key, value, bank, layout, frames, height, width, True
        )
        with_event = structured_attention_from_kernel_bank(
            query, key, value, bank, layout, frames, height, width, True, event_mask=mask
        )
        self.assertFalse(torch.equal(without_event, with_event))

    def test_cost_and_exact_frame_helpers(self) -> None:
        self.assertEqual(exact_sink_recent_frames(12, 3, 3), (0, 1, 2, 9, 10, 11))
        reduction = fft_arithmetic_reduction(3, 12, 30, 52, 4, 6, 0.05, True)
        self.assertGreater(reduction, 1.0)


if __name__ == "__main__":
    unittest.main()
