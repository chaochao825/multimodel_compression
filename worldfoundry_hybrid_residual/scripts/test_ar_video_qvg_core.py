from __future__ import annotations

import unittest

import torch

from ar_video_qvg_core import (
    compression_ratio,
    gather_frames_for_qvg,
    logical_rtn_bytes,
    quantized_frame_indices,
    scatter_qvg_frames,
    tensor_tree_nbytes,
    transform_key_rope,
)


class QVGCoreTest(unittest.TestCase):
    def test_rope_round_trip(self) -> None:
        torch.manual_seed(0)
        key = torch.randn(3, 6, 2, 12)
        angles = torch.randn(16, 6)
        freqs = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)
        roped = transform_key_rope(key, [0, 4, 7], 2, 3, freqs, inverse=False)
        restored = transform_key_rope(roped, [0, 4, 7], 2, 3, freqs, inverse=True)
        self.assertLess(float((restored.float() - key).abs().max()), 2e-6)

    def test_frame_policy(self) -> None:
        self.assertEqual(quantized_frame_indices(12, "none", 3, 3), tuple(range(12)))
        self.assertEqual(quantized_frame_indices(12, "sink_recent", 3, 3), tuple(range(3, 9)))

    def test_gather_scatter_preserves_exact_frames(self) -> None:
        source = torch.arange(4 * 3 * 2 * 2).reshape(4, 3, 2, 2).float()
        indices = (1, 2)
        gathered = gather_frames_for_qvg(source, indices)
        reconstructed = gathered + 100
        result = scatter_qvg_frames(source, reconstructed, indices)
        self.assertTrue(torch.equal(result[0], source[0]))
        self.assertTrue(torch.equal(result[3], source[3]))
        self.assertTrue(torch.equal(result[1:3], source[1:3] + 100))

    def test_storage_accounting(self) -> None:
        state = {
            "a": torch.zeros(7, dtype=torch.uint8),
            "b": [torch.zeros(3, dtype=torch.bfloat16), None],
        }
        self.assertEqual(tensor_tree_nbytes(state), 13)
        tensor = torch.zeros(1, 2, 8, 64, dtype=torch.bfloat16)
        self.assertEqual(logical_rtn_bytes(tensor, 2, 64), 272)
        self.assertEqual(compression_ratio([tensor], 512), 4.0)


if __name__ == "__main__":
    unittest.main()
