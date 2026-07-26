#!/usr/bin/env python3
"""Unit tests for leakage-safe cross-sample geometry defect basis transfer."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("geometry basis transfer tests require torch") from error

from probe_geometry_basis_transfer import (
    basis_transfer_metrics,
    flatten_sample_ids,
    select_ridge_candidate,
    validate_split_ids,
)


class GeometryBasisTransferTests(unittest.TestCase):
    def test_sample_ids_accept_repeated_and_comma_separated_values(self) -> None:
        self.assertEqual(flatten_sample_ids(["s0,s1", "s2"]), ("s0", "s1", "s2"))

    def test_transferable_subspace_has_high_energy_overlap_and_low_error(self) -> None:
        generator = torch.Generator().manual_seed(7)
        basis, _ = torch.linalg.qr(torch.randn(16, 3, generator=generator))
        calibration = torch.randn(48, 3, generator=generator) @ basis.T
        heldout = torch.randn(32, 3, generator=generator) @ basis.T
        metrics = basis_transfer_metrics(calibration, heldout, rank=3)
        projection = metrics["coefficient_oracle_projection"]
        self.assertIsInstance(projection, torch.Tensor)
        self.assertGreater(float(metrics["heldout_self_oracle_rank_energy"]), 0.99999)
        self.assertGreater(float(metrics["frozen_calibration_basis_energy"]), 0.99999)
        self.assertGreater(float(metrics["subspace_overlap"]), 0.99999)
        self.assertLess(float((heldout - projection).norm() / heldout.norm()), 1e-5)

    def test_nontransferable_orthogonal_subspaces_are_rejected(self) -> None:
        generator = torch.Generator().manual_seed(11)
        orthogonal, _ = torch.linalg.qr(torch.randn(16, 6, generator=generator))
        calibration = torch.randn(48, 3, generator=generator) @ orthogonal[:, :3].T
        heldout = torch.randn(32, 3, generator=generator) @ orthogonal[:, 3:6].T
        metrics = basis_transfer_metrics(calibration, heldout, rank=3)
        projection = metrics["coefficient_oracle_projection"]
        self.assertIsInstance(projection, torch.Tensor)
        self.assertGreater(float(metrics["heldout_self_oracle_rank_energy"]), 0.99999)
        self.assertLess(float(metrics["frozen_calibration_basis_energy"]), 1e-10)
        self.assertLess(float(metrics["subspace_overlap"]), 1e-10)
        self.assertLess(float(projection.norm() / heldout.norm()), 1e-5)

    def test_split_contract_rejects_sample_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "split leakage"):
            validate_split_ids(["cal"], ["validation", "shared"], ["shared", "test"])

    def test_ridge_selection_rejects_test_rows(self) -> None:
        leaked = [
            {
                "split": "test",
                "layer": 0,
                "sampling_step": 0,
                "branch": "cond",
                "mask": "s3_tfull",
                "head": 0,
                "rank": 8,
                "ridge_feature": "q",
                "ridge_lambda": 0.01,
                "ridge_corrected_output_relative_l2": 0.01,
            }
        ]
        with self.assertRaisesRegex(ValueError, "validation rows only"):
            select_ridge_candidate(leaked)

    def test_ridge_selection_uses_validation_mean_only(self) -> None:
        common = {
            "split": "validation",
            "layer": 0,
            "sampling_step": 9,
            "branch": "uncond",
            "mask": "s3_temporal_pm2",
            "head": 2,
            "rank": 8,
        }
        rows = [
            {**common, "sample_id": "v0", "ridge_feature": "q", "ridge_lambda": 0.01,
             "ridge_corrected_output_relative_l2": 0.03},
            {**common, "sample_id": "v1", "ridge_feature": "q", "ridge_lambda": 0.01,
             "ridge_corrected_output_relative_l2": 0.01},
            {**common, "sample_id": "v0", "ridge_feature": "concat", "ridge_lambda": 0.1,
             "ridge_corrected_output_relative_l2": 0.015},
            {**common, "sample_id": "v1", "ridge_feature": "concat", "ridge_lambda": 0.1,
             "ridge_corrected_output_relative_l2": 0.015},
        ]
        key = ((0, 9, "uncond"), "s3_temporal_pm2", 2, 8)
        selected = select_ridge_candidate(rows)[key]
        self.assertEqual(selected["ridge_feature"], "concat")
        self.assertEqual(selected["ridge_lambda"], 0.1)
        self.assertEqual(selected["validation_samples"], 2)


if __name__ == "__main__":
    unittest.main()
