#!/usr/bin/env python3
"""Synthetic correctness tests for EXP-046 capacity and Gate semantics."""

from __future__ import annotations

import unittest

import torch

from analyze_wan_rank_state_capacity import (
    aggregate_cells,
    build_gate,
    validate_rows,
)
from rank_state_capacity_core import (
    estimated_wan_block_macs,
    randomized_rank_state_spectrum,
    state_capacity_rows,
)


def synthetic_config() -> dict[str, object]:
    return {
        "blocks": list(range(20, 30)),
        "target_steps": [4, 6],
        "branches": [0, 1],
        "horizons": [1, 2, 3],
        "ranks": [8, 16, 32, 64, 96],
        "sample_plan": [
            {"prompt_index": index, "seed": 100 + index, "split": "selection"}
            for index in range(4)
        ],
        "gate": {
            "decision_rank": 64,
            "diagnostic_rank": 96,
            "maximum_aggregate_output_relative_l2": 0.005,
            "maximum_worst_output_relative_l2": 0.01,
            "minimum_passing_layers_per_step_horizon": 6,
            "maximum_render_to_exact_macs": 0.1,
            "required_selection_identities": 4,
        },
    }


class RankStateCoreTests(unittest.TestCase):
    def test_randomized_svd_recovers_exact_low_rank_matrix(self) -> None:
        generator = torch.Generator().manual_seed(7)
        left = torch.randn(96, 4, generator=generator)
        right = torch.randn(4, 40, generator=generator)
        matrix = left @ right
        spectrum = randomized_rank_state_spectrum(
            matrix,
            max_rank=8,
            oversample=4,
            power_iterations=2,
            seed=19,
        )
        self.assertLess(spectrum.error_sq(4) / spectrum.total_energy, 1e-6)

    def test_randomized_svd_is_deterministic_and_monotonic(self) -> None:
        matrix = torch.randn(80, 32, generator=torch.Generator().manual_seed(11))
        first = randomized_rank_state_spectrum(
            matrix, max_rank=16, oversample=8, power_iterations=2, seed=23
        )
        second = randomized_rank_state_spectrum(
            matrix, max_rank=16, oversample=8, power_iterations=2, seed=23
        )
        torch.testing.assert_close(first.singular_values, second.singular_values)
        rows = state_capacity_rows(
            first,
            ranks=(4, 8, 16),
            residual_target_sq=first.total_energy,
            output_target_sq=2 * first.total_energy,
            estimated_exact_block_macs=1_000_000,
        )
        errors = [float(row["error_sq"]) for row in rows]
        self.assertGreaterEqual(errors[0], errors[1])
        self.assertGreaterEqual(errors[1], errors[2])

    def test_randomized_tail_energy_tracks_exact_svd(self) -> None:
        generator = torch.Generator().manual_seed(13)
        left, _ = torch.linalg.qr(torch.randn(160, 48, generator=generator))
        right, _ = torch.linalg.qr(torch.randn(72, 48, generator=generator))
        singular = torch.logspace(0, -2, 48)
        matrix = (left * singular) @ right.T
        spectrum = randomized_rank_state_spectrum(
            matrix,
            max_rank=24,
            oversample=16,
            power_iterations=2,
            seed=29,
        )
        exact = torch.linalg.svdvals(matrix)
        exact_error = float(exact[16:].double().square().sum())
        observed_error = spectrum.error_sq(16)
        self.assertLess(abs(observed_error - exact_error) / exact_error, 0.01)

    def test_cost_model_counts_attention_projection_and_ffn(self) -> None:
        observed = estimated_wan_block_macs(tokens=10, hidden_size=8, ffn_size=16)
        expected = 4 * 10 * 8 * 8 + 2 * 10 * 8 * 16 + 2 * 10 * 10 * 8
        self.assertEqual(observed, expected)


class RankStateGateTests(unittest.TestCase):
    def make_rows(self) -> list[dict[str, object]]:
        config = synthetic_config()
        rows: list[dict[str, object]] = []
        for item in config["sample_plan"]:
            sample_id = f"p{int(item['prompt_index']):02d}_seed{int(item['seed'])}"
            for block in config["blocks"]:
                for step in config["target_steps"]:
                    for branch in config["branches"]:
                        for horizon in config["horizons"]:
                            previous = 0.20
                            for rank in (0, *config["ranks"]):
                                if rank == 0:
                                    relative = previous
                                elif rank < 64:
                                    relative = 0.015 * (64 / rank) ** 0.5
                                elif block < 26:
                                    relative = 0.004 if rank == 64 else 0.003
                                else:
                                    relative = 0.02 if rank == 64 else 0.003
                                previous = relative
                                rows.append(
                                    {
                                        "sample_id": sample_id,
                                        "split": "selection",
                                        "block": block,
                                        "target_step": step,
                                        "branch": branch,
                                        "horizon": horizon,
                                        "rank": rank,
                                        "error_sq": relative**2,
                                        "residual_target_sq": 1.0,
                                        "output_target_sq": 1.0,
                                        "output_relative_l2": relative,
                                        "target_visible": True,
                                        "defect_remaining_energy": relative**2 / 0.04,
                                        "render_to_exact_macs": 0.0 if rank == 0 else 0.01,
                                        "base_plus_render_to_exact_macs": 0.02,
                                    }
                                )
        return rows

    def test_gate_requires_six_layers_for_every_step_and_horizon(self) -> None:
        config = synthetic_config()
        rows = self.make_rows()
        samples, complete = validate_rows(rows, config, "selection")
        self.assertEqual(len(samples), 4)
        self.assertTrue(complete)
        cells = aggregate_cells(rows, config)
        coverage, decision = build_gate(cells, config, complete)
        self.assertEqual(decision, "PASS")
        rank64 = [row for row in coverage if int(row["rank"]) == 64]
        self.assertTrue(all(int(row["passing_layers"]) == 6 for row in rank64))

    def test_rank96_is_boundary_not_a_rank64_rescue(self) -> None:
        config = synthetic_config()
        rows = self.make_rows()
        for row in rows:
            if int(row["rank"]) == 64 and int(row["block"]) == 25:
                row["error_sq"] = 0.02**2
                row["output_relative_l2"] = 0.02
                row["defect_remaining_energy"] = 0.01
        _, complete = validate_rows(rows, config, "selection")
        cells = aggregate_cells(rows, config)
        _, decision = build_gate(cells, config, complete)
        self.assertEqual(decision, "BOUNDARY")

    def test_missing_metric_cell_is_rejected(self) -> None:
        config = synthetic_config()
        rows = self.make_rows()
        with self.assertRaises(ValueError):
            validate_rows(rows[:-1], config, "selection")


if __name__ == "__main__":
    unittest.main()
