from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_joint_quant_lr_shaping import (
    alternating_candidate,
    apply_lr,
    fit_activation_defect_lr,
    select_block_sparse,
    subspace_overlap,
    symmetric_channel_quantize,
)


class JointQuantLRShapingTests(unittest.TestCase):
    def test_channel_quantization_is_bounded(self) -> None:
        weight = torch.tensor([[1.0, -0.5], [2.0, -1.0]])
        quantized = symmetric_channel_quantize(weight, bits=4, clip=0.75)
        self.assertEqual(quantized.shape, weight.shape)
        self.assertLessEqual(float(quantized[0].abs().max()), 0.75 + 1e-6)
        self.assertLessEqual(float(quantized[1].abs().max()), 1.5 + 1e-6)

    def test_activation_defect_fit_recovers_low_rank_mapping(self) -> None:
        generator = torch.Generator().manual_seed(7)
        inputs = torch.randn(32, 12, generator=generator)
        left = torch.randn(12, 2, generator=generator)
        right = torch.randn(9, 2, generator=generator)
        defect = (inputs @ left) @ right.T
        factors = fit_activation_defect_lr(inputs, defect, rank=2, ridge=1e-7, seed=8)
        estimate = apply_lr(inputs, factors["input_factor"], factors["output_basis"])
        error = float((estimate - defect).norm() / defect.norm())
        self.assertLess(error, 1e-3)

    def test_sparse_selector_obeys_block_budget(self) -> None:
        error = torch.zeros(8, 8)
        error[:4, :4] = 10.0
        error[4:, 4:] = 1.0
        inputs = torch.ones(16, 8)
        sparse, blocks, values = select_block_sparse(
            error, inputs, ratio=0.25, block_out=4, block_in=4
        )
        self.assertEqual(blocks, 1)
        self.assertEqual(values, 16)
        self.assertTrue(torch.equal(sparse[:4, :4], error[:4, :4]))
        self.assertEqual(float(sparse[4:, 4:].abs().sum()), 0.0)

    def test_subspace_overlap(self) -> None:
        basis = torch.eye(5)[:, :2]
        self.assertAlmostEqual(subspace_overlap(basis, basis), 1.0)
        other = torch.eye(5)[:, 2:4]
        self.assertAlmostEqual(subspace_overlap(basis, other), 0.0)

    def test_alternating_residual_shaping_runs(self) -> None:
        generator = torch.Generator().manual_seed(11)
        inputs = torch.randn(24, 10, generator=generator)
        weight = torch.randn(8, 10, generator=generator)
        result = alternating_candidate(
            inputs,
            inputs,
            weight,
            None,
            None,
            bits=4,
            clip=0.8,
            rank=2,
            ridge=1e-5,
            seed=12,
            iterations=2,
            sparse_ratio=0.25,
            block_out=4,
            block_in=5,
        )
        self.assertEqual(result["main_weight"].shape, weight.shape)
        self.assertEqual(result["factors"]["input_factor"].shape, (10, 2))
        self.assertEqual(result["factors"]["output_basis"].shape, (8, 2))
        self.assertEqual(result["alternating_iterations"], 2)

    def test_shaping_strength_relaxes_quantized_residual(self) -> None:
        generator = torch.Generator().manual_seed(13)
        inputs = torch.randn(24, 10, generator=generator)
        weight = torch.randn(8, 10, generator=generator)
        post_hoc = alternating_candidate(
            inputs,
            inputs,
            weight,
            None,
            None,
            bits=4,
            clip=0.8,
            rank=2,
            ridge=1e-5,
            seed=14,
            iterations=2,
            shaping_strength=0.0,
        )
        shaped = alternating_candidate(
            inputs,
            inputs,
            weight,
            None,
            None,
            bits=4,
            clip=0.8,
            rank=2,
            ridge=1e-5,
            seed=14,
            iterations=2,
            shaping_strength=1.0,
        )
        self.assertEqual(post_hoc["shaping_strength"], 0.0)
        self.assertEqual(shaped["shaping_strength"], 1.0)
        self.assertFalse(
            torch.equal(post_hoc["quantized_weight"], shaped["quantized_weight"])
        )

        with self.assertRaisesRegex(ValueError, "shaping strength"):
            alternating_candidate(
                inputs,
                inputs,
                weight,
                None,
                None,
                bits=4,
                clip=0.8,
                rank=2,
                ridge=1e-5,
                seed=14,
                iterations=1,
                shaping_strength=1.1,
            )


if __name__ == "__main__":
    unittest.main()
