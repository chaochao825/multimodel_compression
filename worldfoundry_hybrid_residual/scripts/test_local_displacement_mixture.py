from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_local_displacement_mixture import (
    fit_basis,
    fit_ridge_gate,
    interior_query_indices,
    local_key_indices,
    project_rows,
)


class LocalDisplacementMixtureTests(unittest.TestCase):
    def test_interior_offsets_are_valid_and_ordered(self) -> None:
        shape = (5, 7, 9)
        radius = (1, 2, 2)
        queries = interior_query_indices(shape, radius, 7)
        keys = local_key_indices(queries, shape, radius)
        self.assertEqual(keys.shape, (7, 75))
        self.assertGreaterEqual(int(keys.min()), 0)
        self.assertLess(int(keys.max()), 5 * 7 * 9)

    def test_rank_two_mixture_reconstructs_generated_rows(self) -> None:
        torch.manual_seed(7)
        mean = torch.rand(9)
        raw_basis = torch.randn(9, 2)
        basis, _ = torch.linalg.qr(raw_basis)
        coefficients = torch.randn(20, 2)
        rows = mean + coefficients @ basis.T
        fitted_mean, fitted_basis, energy = fit_basis(rows, 2)
        reconstructed = project_rows(rows, fitted_mean, fitted_basis)
        self.assertGreater(energy, 0.999999)
        self.assertTrue(torch.allclose(rows, reconstructed, atol=1e-5, rtol=1e-5))

    def test_rank_zero_ridge_gate_has_empty_output(self) -> None:
        features = torch.randn(12, 5)
        targets = torch.empty(12, 0)
        gate = fit_ridge_gate(features, targets, 1.0)
        self.assertEqual(gate.predict(features).shape, (12, 0))


if __name__ == "__main__":
    unittest.main()
