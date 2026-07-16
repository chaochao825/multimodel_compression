from __future__ import annotations

import math
import unittest

import numpy as np

from streaming_hybrid_state.core import (
    ClipSequence,
    Codebook,
    fit_codebook,
    fit_decision_tree,
    pq_quantize,
    predict_next,
    prepare_hidden_sequence,
    rgb_summary_features,
    run_residual_codec,
    scalar_quantize,
    split_runs,
)
from streaming_hybrid_state.evaluate import temporal_spectral_entropy


class StreamingHybridCoreTest(unittest.TestCase):
    def test_prepare_hidden_sequence_pools_and_normalizes(self) -> None:
        values = np.arange(3 * 4 * 4 * 8, dtype=np.float32).reshape(
            3,
            4,
            4,
            8,
        )
        prepared = prepare_hidden_sequence(
            values,
            pool_rows=2,
            pool_cols=2,
        )
        self.assertEqual(prepared.shape, (3, 4, 8))
        norms = np.linalg.norm(prepared, axis=-1)
        self.assertTrue(np.all((np.isclose(norms, 1.0)) | (norms < 1e-6)))

    def test_stratified_split_is_disjoint_and_uses_three_one_two(self) -> None:
        clips = []
        for category in ("a", "b"):
            for index in range(6):
                clips.append(
                    ClipSequence(
                        run=f"{category}_{index}",
                        category=category,
                        layer=22,
                        values=np.zeros((2, 1, 2), dtype=np.float32),
                        frames_rgb=np.zeros((2, 8, 8, 3), dtype=np.uint8),
                    )
                )
        splits = split_runs(clips)
        self.assertEqual(
            {name: len(runs) for name, runs in splits.items()},
            {"train": 6, "val": 2, "test": 4},
        )
        self.assertFalse(splits["train"] & splits["val"])
        self.assertFalse(splits["train"] & splits["test"])
        self.assertFalse(splits["val"] & splits["test"])

    def test_split_rejects_empty_validation_configuration(self) -> None:
        clips = [
            ClipSequence(
                run=f"a_{index}",
                category="a",
                layer=22,
                values=np.zeros((2, 1, 2), dtype=np.float32),
                frames_rgb=np.zeros((2, 8, 8, 3), dtype=np.uint8),
            )
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "at least four"):
            split_runs(clips)

    def test_linear_predictor_is_causal_extrapolation(self) -> None:
        history = [
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[2.0, 4.0]], dtype=np.float32),
        ]
        prediction = predict_next(history, "linear")
        np.testing.assert_allclose(prediction, [[3.0, 6.0]])

    def test_temporal_spectral_entropy_distinguishes_constant_and_noise(
        self,
    ) -> None:
        constant = np.ones((16, 2, 4), dtype=np.float32)
        rng = np.random.default_rng(7)
        noise = rng.normal(size=(16, 2, 4)).astype(np.float32)
        constant_entropy = temporal_spectral_entropy([constant])
        noise_entropy = temporal_spectral_entropy([noise])
        self.assertLess(constant_entropy, 1e-6)
        self.assertGreater(noise_entropy, constant_entropy + 0.2)

    def test_scalar_quantization_counts_scale_metadata(self) -> None:
        values = np.linspace(-1.0, 1.0, 2 * 3 * 8, dtype=np.float32).reshape(
            2,
            3,
            8,
        )
        payload = scalar_quantize(values, 4)
        self.assertEqual(payload.payload_bits, values.size * 4)
        self.assertEqual(payload.metadata_bits, 2 * 3 * 16)
        self.assertEqual(payload.reconstruction.shape, values.shape)

    def test_codebook_and_tree_thresholds_use_fp16_precision(self) -> None:
        rng = np.random.default_rng(19)
        samples = rng.normal(size=(64, 2)).astype(np.float32)
        codebook = fit_codebook(
            samples,
            index_bits=2,
            group_dim=2,
            seed=5,
            iterations=2,
        )
        np.testing.assert_array_equal(
            codebook.values,
            codebook.values.astype(np.float16).astype(np.float32),
        )

        features = np.linspace(0.0, 1.0, 32, dtype=np.float64)[:, None]
        labels = (features[:, 0] > 0.37).astype(np.int64)
        tree = fit_decision_tree(
            features,
            labels,
            max_depth=2,
            min_samples=2,
        )
        self.assertFalse(tree.is_leaf)
        self.assertEqual(
            tree.threshold,
            float(np.float16(tree.threshold)),
        )

    def test_pq_payload_and_bitmap_accounting(self) -> None:
        codebook = Codebook(
            values=np.asarray(
                [
                    [-1.0, -1.0],
                    [-1.0, 1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            index_bits=2,
            group_dim=2,
        )
        values = np.asarray(
            [[[1.0, 1.0, -1.0, -1.0]]],
            dtype=np.float32,
        )
        full = pq_quantize(values, codebook)
        sparse = pq_quantize(values, codebook, selected_fraction=0.5)
        self.assertEqual(full.payload_bits, 4)
        self.assertEqual(full.metadata_bits, 0)
        self.assertEqual(sparse.payload_bits, 2)
        self.assertEqual(sparse.metadata_bits, 2)

    def test_residual_codec_is_open_loop_and_shape_preserving(self) -> None:
        rng = np.random.default_rng(11)
        sequence = rng.normal(size=(5, 2, 8)).astype(np.float32)
        residuals = np.diff(sequence, axis=0).reshape(-1, 2)
        codebook = fit_codebook(
            residuals,
            index_bits=2,
            group_dim=2,
            seed=3,
            max_samples=100,
            iterations=3,
        )
        reconstructed, accounting = run_residual_codec(
            sequence,
            predictor_name="previous",
            codebook=codebook,
            selected_fraction=1.0,
        )
        self.assertEqual(reconstructed.shape, sequence.shape)
        self.assertTrue(np.isfinite(reconstructed).all())
        self.assertGreater(accounting["payload_bits"], 0)
        self.assertTrue(math.isclose(accounting["innovation_rate"], 0.8))
        self.assertTrue(math.isclose(accounting["refresh_rate"], 0.2))

    def test_rgb_features_do_not_depend_on_future_frames(self) -> None:
        rng = np.random.default_rng(13)
        frames = rng.integers(0, 256, size=(5, 32, 32, 3), dtype=np.uint8)
        changed = frames.copy()
        changed[4] = 255 - changed[4]
        original_features = rgb_summary_features(frames)
        changed_features = rgb_summary_features(changed)
        np.testing.assert_allclose(
            original_features[:4],
            changed_features[:4],
        )
        np.testing.assert_allclose(original_features[0], 0.0)


if __name__ == "__main__":
    unittest.main()
