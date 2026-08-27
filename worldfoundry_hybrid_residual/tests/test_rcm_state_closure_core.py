from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from rcm_state_closure_core import (  # noqa: E402
    ClosureTrajectory,
    capacity_error_terms,
    fit_stagewise_transitions,
    orthogonal_error_terms,
    project_trajectory,
    rollout_coordinates,
)


def make_trajectory(offset: float) -> ClosureTrajectory:
    generator = torch.Generator().manual_seed(31)
    block_input = torch.randn(4, 12, 6, generator=generator)
    basis = torch.eye(6)[:, :2]
    coordinates = torch.zeros(4, 12, 2)
    coordinates[0] = torch.randn(12, 2, generator=generator) + offset
    projected_input = block_input @ basis
    for stage in range(1, 4):
        drift = projected_input[stage] - projected_input[stage - 1]
        coordinates[stage] = (0.65 + 0.05 * stage) * coordinates[stage - 1]
        coordinates[stage] += (0.2 - 0.02 * stage) * drift
    residual = coordinates @ basis.transpose(0, 1)
    return ClosureTrajectory(block_input=block_input, residual=residual)


def test_stagewise_drift_rollout_recovers_low_rate_dynamics() -> None:
    basis = torch.eye(6)[:, :2]
    calibration = [make_trajectory(0.0), make_trajectory(0.3)]
    transitions = fit_stagewise_transitions(
        calibration,
        basis,
        ridge=1e-8,
        device=torch.device("cpu"),
    )
    held_out = make_trajectory(-0.2)
    input_coordinates, residual_coordinates = project_trajectory(
        held_out, basis, device=torch.device("cpu")
    )
    for horizon in (1, 2, 3):
        prediction = rollout_coordinates(
            input_coordinates,
            residual_coordinates,
            transitions,
            method="drift",
            target_stage=3,
            horizon=horizon,
        )
        torch.testing.assert_close(prediction, residual_coordinates[3], atol=2e-5, rtol=2e-5)


def test_orthogonal_error_identity_matches_dense_reconstruction() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(9, 6, generator=generator)
    basis, _ = torch.linalg.qr(torch.randn(6, 3, generator=generator))
    target_coordinates = target @ basis
    predicted_coordinates = target_coordinates + 0.1
    error_sq, target_sq = orthogonal_error_terms(
        target, target_coordinates, predicted_coordinates
    )
    dense_prediction = predicted_coordinates @ basis.transpose(0, 1)
    torch.testing.assert_close(
        torch.tensor(error_sq, dtype=torch.float64),
        (target - dense_prediction).double().square().sum(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(target_sq, dtype=torch.float64), target.double().square().sum()
    )
    capacity_sq, _ = capacity_error_terms(target, target_coordinates)
    dense_capacity = target_coordinates @ basis.transpose(0, 1)
    torch.testing.assert_close(
        torch.tensor(capacity_sq, dtype=torch.float64),
        (target - dense_capacity).double().square().sum(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_two_lag_rejects_open_loop_without_prior_exact_state() -> None:
    basis = torch.eye(6)[:, :2]
    trajectory = make_trajectory(0.0)
    transitions = fit_stagewise_transitions(
        [trajectory], basis, ridge=1e-6, device=torch.device("cpu")
    )
    input_coordinates, residual_coordinates = project_trajectory(
        trajectory, basis, device=torch.device("cpu")
    )
    try:
        rollout_coordinates(
            input_coordinates,
            residual_coordinates,
            transitions,
            method="ar2_drift",
            target_stage=3,
            horizon=2,
        )
    except ValueError as error:
        assert "one-step" in str(error)
    else:
        raise AssertionError("two-lag rollout must not silently consume unavailable states")
