from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from feature_memory_codec import (
        LowRankFeatureCodec,
        _budgeted_residual_scores,
        encode_budgeted_feature_memory,
        encode_feature_memory,
        encode_pooled_sparse_feature_memory,
        encode_routed_spatial_sparse_feature_memory,
        encode_spatial_grid_feature_memory,
        fit_pca_codec,
        load_codec,
        load_routed_feature_memory,
        reconstruct_feature_memory,
        relative_reconstruction_error,
        sample_feature_tokens,
        save_codec,
        save_routed_feature_memory,
    )
    from mvbench_llava_compressed_feature_memory import (
        parse_adaptive_modes,
        parse_optional_csv_list,
        validate_probe_config,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed locally")
class FeatureMemoryCodecTest(unittest.TestCase):
    def test_optional_probe_lists_accept_empty_values(self) -> None:
        self.assertEqual(parse_optional_csv_list(""), [])
        self.assertEqual(parse_adaptive_modes("  "), [])

    def test_routed_probe_config_rejects_invalid_layouts_and_ratios(self) -> None:
        config = {
            "pool_grid": 8,
            "routed_grid_error_ratio": 1.0,
            "spatial_residual_grids": [2],
            "routed_residual_grids": [2],
        }
        validate_probe_config(config)
        for ratio in (float("nan"), float("inf"), 0.0, -1.0):
            invalid = {**config, "routed_grid_error_ratio": ratio}
            with self.assertRaises(ValueError):
                validate_probe_config(invalid)
        with self.assertRaises(ValueError):
            validate_probe_config(
                {**config, "spatial_residual_grids": [3]}
            )
        with self.assertRaises(ValueError):
            validate_probe_config(
                {
                    **config,
                    "pool_grid": 17,
                    "spatial_residual_grids": [1],
                    "routed_residual_grids": [1],
                }
            )

    def test_identity_basis_reconstructs_exactly(self) -> None:
        features = torch.arange(
            2 * 3 * 4,
            dtype=torch.float32,
        ).reshape(2, 3, 4)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(4),
            basis=torch.eye(4),
        )
        state = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=0,
            storage_dtype=torch.float32,
        )
        reconstructed = reconstruct_feature_memory(
            state,
            codec,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(reconstructed, features)

    def test_sparse_residual_restores_highest_error_token(self) -> None:
        features = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            ]
        )
        codec = LowRankFeatureCodec(
            mean=torch.zeros(3),
            basis=torch.tensor([[1.0], [0.0], [0.0]]),
        )
        low_rank = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=0,
            storage_dtype=torch.float32,
        )
        sparse = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=1,
            storage_dtype=torch.float32,
        )
        self.assertEqual(int(sparse.residual_indices[0, 0]), 1)
        low_rank_error = relative_reconstruction_error(
            features,
            reconstruct_feature_memory(
                low_rank,
                codec,
                output_dtype=torch.float32,
            ),
        )
        sparse_reconstruction = reconstruct_feature_memory(
            sparse,
            codec,
            output_dtype=torch.float32,
        )
        sparse_error = relative_reconstruction_error(
            features,
            sparse_reconstruction,
        )
        self.assertLess(sparse_error, low_rank_error)
        torch.testing.assert_close(
            sparse_reconstruction[0, 1],
            features[0, 1],
        )

    def test_global_budget_concentrates_on_high_error_frame(self) -> None:
        features = torch.tensor(
            [
                [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
                [[0.0, 8.0], [0.0, 7.0], [0.0, 0.5]],
            ]
        )
        codec = LowRankFeatureCodec(
            mean=torch.zeros(2),
            basis=torch.tensor([[1.0], [0.0]]),
        )
        state = encode_budgeted_feature_memory(
            features,
            codec,
            residual_token_budget=2,
            allocation="global_energy",
            storage_dtype=torch.float32,
        )
        self.assertEqual(state.residual_frame_counts().tolist(), [0, 2])
        reconstruction = reconstruct_feature_memory(
            state,
            codec,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(reconstruction[1, :2], features[1, :2])

    def test_balanced_global_budget_retains_each_frame(self) -> None:
        features = torch.tensor(
            [
                [[0.0, 1.0], [0.0, 2.0]],
                [[0.0, 8.0], [0.0, 7.0]],
                [[0.0, 0.5], [0.0, 0.25]],
            ]
        )
        codec = LowRankFeatureCodec(
            mean=torch.zeros(2),
            basis=torch.tensor([[1.0], [0.0]]),
        )
        state = encode_budgeted_feature_memory(
            features,
            codec,
            residual_token_budget=3,
            allocation="global_energy",
            minimum_per_frame=1,
            storage_dtype=torch.float32,
        )
        self.assertEqual(state.residual_frame_counts().tolist(), [1, 1, 1])

    def test_budgeted_selected_frames_preserve_requested_order(self) -> None:
        features = torch.zeros(3, 2, 3)
        features[0, 0, 2] = 3.0
        features[1, 1, 2] = 4.0
        features[2, 0, 2] = 5.0
        codec = LowRankFeatureCodec(
            mean=torch.zeros(3),
            basis=torch.eye(3)[:, :2],
        )
        state = encode_budgeted_feature_memory(
            features,
            codec,
            residual_token_budget=3,
            allocation="global_energy",
            minimum_per_frame=1,
            storage_dtype=torch.float32,
        )
        selected = reconstruct_feature_memory(
            state,
            codec,
            frame_positions=[2, 0],
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(selected, features[[2, 0]])

    def test_global_budget_matches_fixed_quota_payload_bytes(self) -> None:
        features = torch.randn(2, 4, 6)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(6, dtype=torch.float16),
            basis=torch.eye(6, dtype=torch.float16)[:, :2],
        )
        fixed = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=2,
        )
        global_state = encode_budgeted_feature_memory(
            features,
            codec,
            residual_token_budget=4,
            allocation="global_energy",
        )
        self.assertEqual(
            global_state.stream_state_bytes,
            fixed.stream_state_bytes,
        )

    def test_temporal_novelty_prefers_changed_frame(self) -> None:
        features = torch.zeros(2, 2, 3)
        features[0, 0, 2] = 3.0
        features[1, 1, 2] = 2.5
        features[1, 1, 0] = 20.0
        codec = LowRankFeatureCodec(
            mean=torch.zeros(3),
            basis=torch.eye(3)[:, :1],
        )
        global_state = encode_budgeted_feature_memory(
            features,
            codec,
            residual_token_budget=1,
            allocation="global_energy",
            storage_dtype=torch.float32,
        )
        temporal_state = encode_budgeted_feature_memory(
            features,
            codec,
            residual_token_budget=1,
            allocation="temporal_novelty",
            temporal_novelty_weight=2.0,
            storage_dtype=torch.float32,
        )
        self.assertEqual(global_state.residual_frame_counts().tolist(), [1, 0])
        self.assertEqual(
            temporal_state.residual_frame_counts().tolist(),
            [0, 1],
        )

    def test_temporal_novelty_scores_are_prefix_invariant(self) -> None:
        generator = torch.Generator().manual_seed(23)
        prefix_features = torch.randn(3, 5, 4, generator=generator)
        prefix_residual = torch.randn(3, 5, 4, generator=generator)
        future_features = torch.randn(2, 5, 4, generator=generator) * 20.0
        future_residual = torch.randn(2, 5, 4, generator=generator)
        prefix_scores = _budgeted_residual_scores(
            prefix_features,
            prefix_residual,
            allocation="temporal_novelty",
            temporal_novelty_weight=1.5,
        )
        extended_scores = _budgeted_residual_scores(
            torch.cat([prefix_features, future_features]),
            torch.cat([prefix_residual, future_residual]),
            allocation="temporal_novelty",
            temporal_novelty_weight=1.5,
        )
        torch.testing.assert_close(extended_scores[:3], prefix_scores)

    def test_pooled_residual_restores_diffuse_frame_shift(self) -> None:
        features = torch.zeros(2, 4, 3)
        features[0, :, 2] = 2.0
        features[1, :, 2] = -1.5
        codec = LowRankFeatureCodec(
            mean=torch.zeros(3),
            basis=torch.eye(3)[:, :2],
        )
        state = encode_pooled_sparse_feature_memory(
            features,
            codec,
            residual_vectors_per_frame=1,
            storage_dtype=torch.float32,
        )
        reconstructed = reconstruct_feature_memory(
            state,
            codec,
            frame_positions=[1, 0],
            output_dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(reconstructed, features[[1, 0]], atol=1e-6)
        )

    def test_pooled_sparse_uses_no_more_bytes_than_fixed_quota(self) -> None:
        features = torch.randn(3, 6, 8)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(8, dtype=torch.float16),
            basis=torch.eye(8, dtype=torch.float16)[:, :2],
        )
        fixed = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=4,
        )
        pooled_sparse = encode_pooled_sparse_feature_memory(
            features,
            codec,
            residual_vectors_per_frame=4,
        )
        self.assertEqual(
            pooled_sparse.residual_value_bytes,
            fixed.residual_value_bytes,
        )
        self.assertLess(
            pooled_sparse.stream_state_bytes,
            fixed.stream_state_bytes,
        )

    def test_spatial_grid_restores_block_constant_residual(self) -> None:
        features = torch.zeros(2, 16, 3)
        spatial = torch.tensor(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )
        features[0, :, 2] = spatial.reshape(-1)
        features[1, :, 2] = -spatial.reshape(-1)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(3),
            basis=torch.eye(3)[:, :2],
        )
        state = encode_spatial_grid_feature_memory(
            features,
            codec,
            residual_grid_size=2,
            storage_dtype=torch.float32,
        )
        reconstructed = reconstruct_feature_memory(
            state,
            codec,
            frame_positions=[1, 0],
            output_dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(reconstructed, features[[1, 0]], atol=1e-6)
        )

    def test_spatial_grid_matches_fixed_value_budget_without_indices(self) -> None:
        features = torch.randn(3, 16, 8)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(8, dtype=torch.float16),
            basis=torch.eye(8, dtype=torch.float16)[:, :2],
        )
        fixed = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=4,
        )
        spatial = encode_spatial_grid_feature_memory(
            features,
            codec,
            residual_grid_size=2,
        )
        self.assertEqual(spatial.residual_value_bytes, fixed.residual_value_bytes)
        self.assertEqual(spatial.residual_index_bytes, 0)
        self.assertLess(spatial.stream_state_bytes, fixed.stream_state_bytes)

    def test_routed_residual_selects_grid_and_sparse_per_frame(self) -> None:
        features = torch.zeros(2, 16, 3)
        block_constant = torch.tensor(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )
        features[0, :, 2] = block_constant.reshape(-1)
        features[1, [0, 5, 10, 15], 2] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        codec = LowRankFeatureCodec(
            mean=torch.zeros(3),
            basis=torch.eye(3)[:, :2],
        )
        state = encode_routed_spatial_sparse_feature_memory(
            features,
            codec,
            residual_grid_size=2,
            storage_dtype=torch.float32,
        )
        self.assertEqual(state.grid_mode.tolist(), [True, False])
        reconstructed = reconstruct_feature_memory(
            state,
            codec,
            frame_positions=[1, 0],
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(reconstructed, features[[1, 0]])

    def test_routed_residual_matches_fixed_value_budget(self) -> None:
        features = torch.randn(3, 16, 8)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(8, dtype=torch.float16),
            basis=torch.eye(8, dtype=torch.float16)[:, :2],
        )
        fixed = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=4,
        )
        routed = encode_routed_spatial_sparse_feature_memory(
            features,
            codec,
            residual_grid_size=2,
        )
        self.assertEqual(routed.residual_value_bytes, fixed.residual_value_bytes)
        self.assertLess(routed.residual_index_bytes, fixed.residual_index_bytes)
        self.assertLess(routed.stream_state_bytes, fixed.stream_state_bytes)

    def test_routed_fp16_chooses_lowest_stored_candidate_error(self) -> None:
        generator = torch.Generator().manual_seed(31)
        features = torch.randn(5, 16, 8, generator=generator)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(8, dtype=torch.float16),
            basis=torch.eye(8, dtype=torch.float16)[:, :2],
        )
        fixed = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=4,
        )
        grid = encode_spatial_grid_feature_memory(
            features,
            codec,
            residual_grid_size=2,
        )
        routed = encode_routed_spatial_sparse_feature_memory(
            features,
            codec,
            residual_grid_size=2,
        )
        fixed_reconstruction = reconstruct_feature_memory(
            fixed,
            codec,
            output_dtype=torch.float16,
        )
        grid_reconstruction = reconstruct_feature_memory(
            grid,
            codec,
            output_dtype=torch.float16,
        )
        routed_reconstruction = reconstruct_feature_memory(
            routed,
            codec,
            output_dtype=torch.float16,
        )

        def frame_error(reconstruction: torch.Tensor) -> torch.Tensor:
            return torch.sum(
                (features.float() - reconstruction.float()).square(),
                dim=(1, 2),
            )

        expected = torch.minimum(
            frame_error(fixed_reconstruction),
            frame_error(grid_reconstruction),
        )
        torch.testing.assert_close(frame_error(routed_reconstruction), expected)

    def test_routed_state_is_prefix_invariant(self) -> None:
        generator = torch.Generator().manual_seed(37)
        prefix = torch.randn(3, 16, 8, generator=generator)
        future = torch.randn(2, 16, 8, generator=generator) * 10.0
        codec = LowRankFeatureCodec(
            mean=torch.zeros(8, dtype=torch.float16),
            basis=torch.eye(8, dtype=torch.float16)[:, :2],
        )
        prefix_state = encode_routed_spatial_sparse_feature_memory(
            prefix,
            codec,
            residual_grid_size=2,
        )
        extended_state = encode_routed_spatial_sparse_feature_memory(
            torch.cat([prefix, future]),
            codec,
            residual_grid_size=2,
        )
        torch.testing.assert_close(extended_state.latents[:3], prefix_state.latents)
        torch.testing.assert_close(
            extended_state.grid_mode[:3], prefix_state.grid_mode
        )
        torch.testing.assert_close(
            extended_state.residual_indices[:3], prefix_state.residual_indices
        )
        torch.testing.assert_close(
            extended_state.residual_values[:3], prefix_state.residual_values
        )

    def test_routed_state_archive_round_trip_preserves_payload(self) -> None:
        features = torch.randn(3, 16, 8)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(8, dtype=torch.float16),
            basis=torch.eye(8, dtype=torch.float16)[:, :2],
        )
        state = encode_routed_spatial_sparse_feature_memory(
            features,
            codec,
            residual_grid_size=2,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "routed_state.pt"
            save_routed_feature_memory(state, path)
            archive_bytes = path.stat().st_size
            loaded = load_routed_feature_memory(path)
        self.assertGreater(archive_bytes, state.stream_state_bytes)
        self.assertEqual(loaded.stream_state_bytes, state.stream_state_bytes)
        torch.testing.assert_close(loaded.latents, state.latents)
        torch.testing.assert_close(loaded.grid_mode, state.grid_mode)
        torch.testing.assert_close(loaded.residual_indices, state.residual_indices)
        torch.testing.assert_close(loaded.residual_values, state.residual_values)

    def test_selected_frame_reconstruction_preserves_order(self) -> None:
        features = torch.randn(4, 2, 3)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(3),
            basis=torch.eye(3),
        )
        state = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=0,
            storage_dtype=torch.float32,
        )
        selected = reconstruct_feature_memory(
            state,
            codec,
            frame_positions=[3, 1],
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(selected, features[[3, 1]])

    def test_stream_bytes_separate_latent_and_sparse_payloads(self) -> None:
        features = torch.randn(2, 4, 6)
        codec = LowRankFeatureCodec(
            mean=torch.zeros(6, dtype=torch.float16),
            basis=torch.eye(6, dtype=torch.float16)[:, :2],
        )
        state = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=1,
        )
        self.assertEqual(state.latent_bytes, 2 * 4 * 2 * 2)
        self.assertEqual(state.residual_value_bytes, 2 * 1 * 6 * 2)
        self.assertEqual(state.residual_index_bytes, 2 * 1 * 2)
        self.assertEqual(
            state.stream_state_bytes,
            state.latent_bytes
            + state.residual_value_bytes
            + state.residual_index_bytes,
        )
        self.assertEqual(codec.parameter_bytes, (6 + 6 * 2) * 2)

    def test_token_sampling_is_bounded_and_deterministic(self) -> None:
        features = torch.arange(
            3 * 4 * 2,
            dtype=torch.float32,
        ).reshape(3, 4, 2)
        first, first_positions = sample_feature_tokens(
            features,
            count=5,
            seed=17,
        )
        second, second_positions = sample_feature_tokens(
            features,
            count=5,
            seed=17,
        )
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first_positions, second_positions)
        self.assertEqual(len(first), 5)
        self.assertEqual(
            first_positions.tolist(),
            sorted(first_positions.tolist()),
        )

    def test_fitted_pca_recovers_synthetic_low_rank_features(self) -> None:
        generator = torch.Generator().manual_seed(9)
        left = torch.randn(64, 2, generator=generator)
        right = torch.randn(2, 12, generator=generator)
        features = (left @ right).reshape(4, 16, 12)
        codec, metadata = fit_pca_codec(
            features,
            rank=2,
            storage_dtype=torch.float32,
        )
        state = encode_feature_memory(
            features,
            codec,
            residual_tokens_per_frame=0,
            storage_dtype=torch.float32,
        )
        reconstruction = reconstruct_feature_memory(
            state,
            codec,
            output_dtype=torch.float32,
        )
        self.assertLess(
            relative_reconstruction_error(features, reconstruction),
            1e-4,
        )
        self.assertGreater(
            float(metadata["explained_energy_ratio"]),
            0.9999,
        )

    def test_codec_round_trip_preserves_metadata(self) -> None:
        codec = LowRankFeatureCodec(
            mean=torch.arange(4, dtype=torch.float16),
            basis=torch.eye(4, dtype=torch.float16)[:, :2],
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "codec.pt"
            save_codec(codec, path, metadata={"split": "calibration"})
            loaded, metadata = load_codec(path)
        torch.testing.assert_close(loaded.mean, codec.mean)
        torch.testing.assert_close(loaded.basis, codec.basis)
        self.assertEqual(metadata["split"], "calibration")


if __name__ == "__main__":
    unittest.main()
