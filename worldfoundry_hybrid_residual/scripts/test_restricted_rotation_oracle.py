#!/usr/bin/env python3
"""High-value numerical invariants for the restricted-rotation oracle."""

from __future__ import annotations

import math
import unittest

import torch

from restricted_rotation_oracle_core import (
    apply_butterfly,
    apply_dcd,
    apply_orthogonal_bcm,
    batch_residual_squares,
    greedy_givens_prefixes,
    householder_prefixes,
    orthonormalize,
    rotation_cost,
    subspace_overlap,
)


class RestrictedRotationCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)

    def test_householder_rank_reflections_recover_target_subspace(self) -> None:
        source = orthonormalize(torch.randn(32, 4))
        target = orthonormalize(torch.randn(32, 4))
        prefixes, vectors = householder_prefixes(source, target, (1, 2, 4))
        self.assertEqual(len(vectors), 4)
        self.assertGreater(
            float(subspace_overlap(prefixes[4], orthonormalize(target.double()))),
            1.0 - 1e-10,
        )

    def test_householder_endpoint_survives_high_defect_amplification(self) -> None:
        source = orthonormalize(torch.randn(128, 16))
        target = orthonormalize(torch.randn(128, 16))
        coefficients = 1000.0 * torch.randn(64, 16)
        orthogonal_noise = torch.randn(64, 128)
        orthogonal_noise = orthogonal_noise - (orthogonal_noise @ target) @ target.T
        defect = coefficients @ target.T + 1e-3 * orthogonal_noise
        prefixes, _ = householder_prefixes(source, target, (16,))
        target_audit = orthonormalize(target.double())
        adaptive = batch_residual_squares(defect.double()[None], target_audit[None])[0]
        mapped = batch_residual_squares(defect.double()[None], prefixes[16][None])[0]
        self.assertLessEqual(float(mapped), float(adaptive) + 1e-6 * float(defect.square().sum()))

    def test_givens_prefixes_do_not_worsen_true_defect_objective(self) -> None:
        source = orthonormalize(torch.randn(16, 3))
        target = orthonormalize(torch.randn(16, 3))
        defect = torch.randn(12, 3) @ target.T
        prefixes, parameters = greedy_givens_prefixes(source, defect, (1, 2, 4), grid_points=17)
        baseline = float(batch_residual_squares(defect[None], source[None])[0])
        losses = [float(batch_residual_squares(defect[None], prefixes[count][None])[0]) for count in (1, 2, 4)]
        self.assertLessEqual(losses[0], baseline + 1e-5)
        self.assertLessEqual(losses[1], losses[0] + 1e-5)
        self.assertLessEqual(losses[2], losses[1] + 1e-5)
        self.assertEqual(len(parameters), 4)

    def test_orthogonal_bcm_preserves_basis_gram(self) -> None:
        source = orthonormalize(torch.randn(2, 32, 4))
        phases = torch.randn(2, 3, 4, 3)
        transformed = apply_orthogonal_bcm(source, phases, block_size=8)
        expected = torch.eye(4).expand(2, -1, -1)
        torch.testing.assert_close(transformed.transpose(1, 2) @ transformed, expected, atol=2e-5, rtol=2e-5)

    def test_butterfly_preserves_basis_gram(self) -> None:
        source = orthonormalize(torch.randn(2, 32, 4))
        angles = torch.randn(2, 7, 16)
        transformed = apply_butterfly(source, angles)
        expected = torch.eye(4).expand(2, -1, -1)
        torch.testing.assert_close(transformed.transpose(1, 2) @ transformed, expected, atol=2e-5, rtol=2e-5)

    def test_dcd_returns_orthonormal_basis(self) -> None:
        source = orthonormalize(torch.randn(2, 16, 3))
        left = torch.randn(2, 2, 16)
        right = torch.randn(2, 2, 16)
        phases = torch.randn(2, 2, 7)
        transformed = apply_dcd(source, left, right, phases, max_log_scale=0.25)
        expected = torch.eye(3).expand(2, -1, -1)
        torch.testing.assert_close(transformed.transpose(1, 2) @ transformed, expected, atol=2e-5, rtol=2e-5)

    def test_common_rotation_preserves_subspace_overlap(self) -> None:
        left = orthonormalize(torch.randn(32, 5))
        right = orthonormalize(torch.randn(32, 5))
        rotation = orthonormalize(torch.randn(32, 32))
        before = subspace_overlap(left, right)
        after = subspace_overlap(rotation @ left, rotation @ right)
        self.assertAlmostEqual(float(before), float(after), places=5)

    def test_cost_model_separates_generator_count_from_payload(self) -> None:
        common = {
            "query_tokens": 64,
            "key_tokens": 32760,
            "channels": 128,
            "rank": 16,
            "block_size": 16,
        }
        givens = rotation_cost("givens", 16, **common)
        householder = rotation_cost("householder", 16, **common)
        self.assertEqual(givens.dynamic_scalars, 48)
        self.assertEqual(householder.dynamic_scalars, 2048)
        self.assertLess(givens.work_ratio, 0.05)
        self.assertLess(householder.work_ratio, 0.05)
        self.assertTrue(math.isfinite(householder.work_ratio))


if __name__ == "__main__":
    unittest.main()
