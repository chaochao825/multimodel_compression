from __future__ import annotations

import math

from analyze_wan_rcm_state_closure import energy_summary, gate_decision, output_summary


def test_output_summary_pools_energy_instead_of_averaging_ratios() -> None:
    rows = [
        {"error_sq": 1.0, "output_target_sq": 1.0, "output_relative_l2": 1.0},
        {"error_sq": 1.0, "output_target_sq": 9.0, "output_relative_l2": 1.0 / 3.0},
    ]
    summary = output_summary(rows)
    assert math.isclose(summary["aggregate"], math.sqrt(0.2))
    assert summary["worst"] == 1.0


def test_energy_summary_separates_subspace_capture_from_output_amplification() -> None:
    rows = [
        {
            "error_sq": 1.0,
            "target_sq": 4.0,
            "output_target_sq": 1.0,
            "output_relative_l2": 1.0,
        }
    ]
    summary = energy_summary(rows)
    assert summary["captured_residual_energy"] == 0.75
    assert summary["residual_relative_l2"] == 0.5
    assert summary["output_relative_l2"] == 1.0
    assert summary["residual_to_output_scale"] == 2.0


def test_gate_reports_directional_only_when_relative_effect_is_broad_but_absolute_gate_fails() -> None:
    blocks = [
        {
            "block": block,
            "capacity_pass": block < 24,
            "h1_pass": block < 24,
            "weight_pass": block < 27,
        }
        for block in range(20, 30)
    ]
    pooled = {
        "h2": {"aggregate": 0.01, "worst": 0.02},
        "h3": {"aggregate": 0.01, "worst": 0.02},
        "two_lag_advantage": 0.05,
        "shared_basis_penalty": 0.05,
    }
    thresholds = {
        "minimum_passing_layers": 6,
        "maximum_open_loop_aggregate": 0.02,
        "maximum_open_loop_worst": 0.04,
        "maximum_history_advantage": 1.10,
        "maximum_shared_basis_penalty": 1.25,
    }
    decision = gate_decision(blocks, pooled, thresholds)
    assert decision["outcome"] == "directional-only"
    assert decision["weight_pass_layers"] == list(range(20, 27))
