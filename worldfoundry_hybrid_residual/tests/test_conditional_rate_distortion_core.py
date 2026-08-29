from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from conditional_rate_distortion_core import (  # noqa: E402
    ModuleTrajectory,
    apply_diagonal_field,
    error_terms,
    fit_diagonal_field,
    oracle_gap_recovery,
    zero_cost_speedup_ceiling,
)


def make_trajectory(sample_id: str, scale: float = 1.0) -> ModuleTrajectory:
    generator = torch.Generator().manual_seed(17)
    interface = torch.randn(6, 12, 8, generator=generator) * scale
    target = torch.zeros_like(interface)
    target[0] = 0.3 * interface[0]
    target[1] = 0.7 * target[0] + 0.4 * (interface[1] - interface[0])
    channel_a = torch.linspace(0.55, 0.85, 8)
    channel_b = torch.linspace(-0.15, 0.05, 8)
    channel_c = torch.linspace(0.20, 0.45, 8)
    for step in range(2, 6):
        target[step] = (
            channel_a * target[step - 1]
            + channel_b * target[step - 2]
            + channel_c * (interface[step] - interface[step - 1])
        )
    block_output = interface + target
    return ModuleTrajectory(
        sample_id=sample_id,
        target_name="whole_block",
        block=0,
        branch=0,
        interface=interface,
        target=target,
        block_output=block_output,
    )


def test_diagonal_field_recovers_known_causal_dynamics() -> None:
    calibration = make_trajectory("calibration")
    heldout = make_trajectory("heldout", scale=1.3)
    field = fit_diagonal_field([calibration], target_step=4, ridge=1e-8)
    prediction = apply_diagonal_field(field, heldout, target_step=4)
    metrics = error_terms(prediction, heldout, target_step=4)
    assert metrics["target_relative_l2"] < 1e-5
    assert metrics["block_output_relative_l2"] < 1e-5


def test_oracle_gap_recovery_uses_linear_risk_difference() -> None:
    assert oracle_gap_recovery(0.10, 0.04, 0.02) == 0.75
    assert oracle_gap_recovery(0.02, 0.01, 0.02) == float("-inf")


def test_zero_cost_speedup_ceiling_matches_amdahl() -> None:
    expected = 1.0 / (1.0 - 0.5 * 0.4)
    assert zero_cost_speedup_ceiling(0.5, 0.4) == expected
