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
        encode_feature_memory,
        fit_pca_codec,
        load_codec,
        reconstruct_feature_memory,
        relative_reconstruction_error,
        sample_feature_tokens,
        save_codec,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed locally")
class FeatureMemoryCodecTest(unittest.TestCase):
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
