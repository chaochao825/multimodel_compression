from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_nystrom_sparse_tail import (
    arithmetic_work_ratio,
    landmark_partition_output,
    mass_mixture_output,
    nystrom_components,
    nystrom_factors,
    segment_landmarks,
    validate_capture_payload,
)


class NystromSparseTailTests(unittest.TestCase):
    def test_capture_metadata_mismatch_is_rejected(self) -> None:
        tensor = torch.zeros(1, 6, 2, 4, dtype=torch.bfloat16)
        row = {
            "sample_id": "sample",
            "prompt_index": 0,
            "seed": 7,
            "sampling_step": 9,
            "timestep": "859.0",
            "branch": "cond",
            "layer": 14,
        }
        metadata = {
            **row,
            "timestep": 859.0,
            "attention_kind": "self",
            "q_shape": list(tensor.shape),
            "k_shape": list(tensor.shape),
            "v_shape": list(tensor.shape),
            "token_count": 6,
            "grid_size": [1, 2, 3],
            "dtype": "torch.bfloat16",
        }
        payload = {
            "q": tensor,
            "k": tensor,
            "v": tensor,
            "softmax_scale": 0.5,
            "metadata": metadata,
        }
        validate_capture_payload(payload, Path("capture.pt"), row)
        metadata["layer"] = 15
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            validate_capture_payload(payload, Path("capture.pt"), row)

    def test_segment_landmarks_are_group_means(self) -> None:
        tensor = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        actual = segment_landmarks(tensor, 3)
        expected = torch.stack(
            [tensor[0:2].mean(0), tensor[2:4].mean(0), tensor[4:6].mean(0)]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_full_landmarks_recover_dense_attention(self) -> None:
        torch.manual_seed(11)
        queries = torch.randn(5, 4) * 0.25
        keys = torch.randn(5, 4) * 0.25
        scale = 0.5
        landmarks, inverse, right, _ = nystrom_factors(
            queries, keys, 5, "uniform", scale, 1e-7
        )
        _, signed, nonnegative, _ = nystrom_components(
            queries, landmarks, inverse, right, scale
        )
        expected = torch.softmax(queries @ keys.T * scale, dim=1)
        self.assertTrue(torch.allclose(signed, expected, atol=2e-4, rtol=2e-4))
        self.assertTrue(torch.allclose(nonnegative, expected, atol=2e-4, rtol=2e-4))

    def test_all_selected_hybrid_is_exact(self) -> None:
        torch.manual_seed(13)
        queries = torch.randn(3, 4)
        keys = torch.randn(7, 4)
        values = torch.randn(7, 4)
        scores = queries @ keys.T * 0.5
        dense = torch.softmax(scores, dim=1)
        noisy_tail = torch.softmax(torch.randn_like(scores), dim=1) @ values
        selected = torch.ones(7, dtype=torch.bool)
        actual = mass_mixture_output(
            scores,
            noisy_tail,
            values,
            selected,
            torch.ones(3),
        )
        expected = dense @ values
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_all_selected_landmark_partition_is_exact(self) -> None:
        torch.manual_seed(17)
        queries = torch.randn(3, 4)
        keys = torch.randn(7, 4)
        values = torch.randn(7, 4)
        scores = queries @ keys.T * 0.5
        dense = torch.softmax(scores, dim=1)
        left = torch.softmax(torch.randn(3, 2), dim=1)
        right = torch.softmax(torch.randn(2, 7), dim=1)
        selected = torch.ones(7, dtype=torch.bool)
        actual = landmark_partition_output(
            scores,
            left,
            right,
            right @ values,
            values,
            selected,
            torch.ones(3),
        )
        self.assertTrue(
            torch.allclose(actual, dense @ values, atol=1e-6, rtol=1e-6)
        )

    def test_partition_overhead_is_charged(self) -> None:
        mixture, _ = arithmetic_work_ratio(
            "proxy_mass_nystrom_mixture", 64, 32760, 0.125, 64, 64, 128
        )
        partition, parts = arithmetic_work_ratio(
            "proxy_mass_landmark_partition", 64, 32760, 0.125, 64, 64, 128
        )
        self.assertGreater(partition, mixture)
        self.assertGreater(float(parts["partition_work_ratio"]), 0.0)


if __name__ == "__main__":
    unittest.main()
