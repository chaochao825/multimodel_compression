#!/usr/bin/env python3
"""Low-rate state fitting and rollout primitives for EXP-048."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ClosureTrajectory:
    """One model/trajectory/layer sequence over four denoising stages."""

    block_input: torch.Tensor
    residual: torch.Tensor

    def validate(self) -> None:
        if self.block_input.ndim != 3:
            raise ValueError("block_input must have shape [steps, tokens, channels]")
        if self.residual.shape != self.block_input.shape:
            raise ValueError("residual and block_input must have identical shapes")
        if self.step_count < 2:
            raise ValueError("closure trajectories require at least two stages")
        if not torch.isfinite(self.block_input).all():
            raise ValueError("block_input contains non-finite values")
        if not torch.isfinite(self.residual).all():
            raise ValueError("residual contains non-finite values")

    @property
    def step_count(self) -> int:
        return int(self.block_input.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.block_input.shape[1])

    @property
    def channel_count(self) -> int:
        return int(self.block_input.shape[2])


def fit_basis(
    matrices: Iterable[torch.Tensor],
    *,
    rank: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit an uncentered calibration-only channel basis."""

    rows = [matrix.reshape(-1, matrix.shape[-1]) for matrix in matrices]
    if not rows:
        raise ValueError("fit_basis requires calibration matrices")
    channels = int(rows[0].shape[1])
    if any(int(row.shape[1]) != channels for row in rows):
        raise ValueError("all basis matrices must share the channel width")
    if rank <= 0 or rank > channels:
        raise ValueError("rank must lie in [1, channels]")
    data = torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)
    q = min(rank + oversampling, min(data.shape))
    if q < rank:
        raise ValueError("calibration matrix is too small for the requested rank")
    devices = [] if device.type == "cpu" else [device]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        _, singular_values, vectors = torch.pca_lowrank(
            data,
            q=q,
            center=False,
            niter=power_iterations,
        )
    basis = vectors[:, :rank].contiguous()
    total_energy = float(data.double().square().sum())
    captured_energy = float(singular_values[:rank].double().square().sum())
    diagnostics = {
        "rows": float(data.shape[0]),
        "channels": float(channels),
        "captured_energy_ratio": captured_energy / max(total_energy, 1e-30),
        "orthogonality_error": float(
            (
                basis.transpose(0, 1) @ basis
                - torch.eye(rank, device=device, dtype=basis.dtype)
            )
            .abs()
            .max()
        ),
    }
    return basis.cpu(), diagnostics


def project_trajectory(
    trajectory: ClosureTrajectory,
    basis: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    trajectory.validate()
    basis_device = basis.to(device=device, dtype=torch.float32)
    block_input = trajectory.block_input.to(device=device, dtype=torch.float32)
    residual = trajectory.residual.to(device=device, dtype=torch.float32)
    return (
        (block_input @ basis_device).cpu(),
        (residual @ basis_device).cpu(),
    )


def _fit_coordinate_regression(
    features: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Fit one small independent regression per state coordinate."""

    if features.ndim != 3 or target.ndim != 2:
        raise ValueError("features/target must have shapes [rows, terms, rank]/[rows, rank]")
    if features.shape[0] != target.shape[0] or features.shape[2] != target.shape[1]:
        raise ValueError("coordinate regression dimensions do not align")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    x = features.to(dtype=torch.float64)
    y = target.to(dtype=torch.float64)
    gram = torch.einsum("nkr,nlr->rkl", x, x)
    rhs = torch.einsum("nkr,nr->rk", x, y)
    diagonal_mean = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype).unsqueeze(0)
    regularized = gram + ridge * diagonal_mean[:, None, None] * identity
    coefficients = torch.linalg.solve(regularized, rhs.unsqueeze(-1)).squeeze(-1)
    return coefficients.transpose(0, 1).to(dtype=torch.float32).contiguous()


def fit_stagewise_transitions(
    trajectories: Iterable[ClosureTrajectory],
    basis: torch.Tensor,
    *,
    ridge: float,
    device: torch.device,
) -> dict[str, dict[int, torch.Tensor]]:
    """Fit stagewise first-order, innovation, and two-lag state dynamics."""

    projected = [project_trajectory(item, basis, device=device) for item in trajectories]
    if not projected:
        raise ValueError("transition fitting requires calibration trajectories")
    step_count = int(projected[0][0].shape[0])
    if any(int(item[0].shape[0]) != step_count for item in projected):
        raise ValueError("all calibration trajectories must share the stage count")
    output: dict[str, dict[int, torch.Tensor]] = {
        "ar1": {},
        "drift": {},
        "ar2_drift": {},
    }
    for stage in range(1, step_count):
        previous = torch.cat([item[1][stage - 1] for item in projected], dim=0)
        drift = torch.cat(
            [item[0][stage] - item[0][stage - 1] for item in projected], dim=0
        )
        target = torch.cat([item[1][stage] for item in projected], dim=0)
        output["ar1"][stage] = _fit_coordinate_regression(
            previous.unsqueeze(1), target, ridge
        )
        output["drift"][stage] = _fit_coordinate_regression(
            torch.stack((previous, drift), dim=1), target, ridge
        )
        if stage >= 2:
            lag2 = torch.cat([item[1][stage - 2] for item in projected], dim=0)
            output["ar2_drift"][stage] = _fit_coordinate_regression(
                torch.stack((previous, lag2, drift), dim=1), target, ridge
            )
    return output


def rollout_coordinates(
    input_coordinates: torch.Tensor,
    residual_coordinates: torch.Tensor,
    transitions: dict[str, dict[int, torch.Tensor]],
    *,
    method: str,
    target_stage: int,
    horizon: int,
) -> torch.Tensor:
    """Roll a compressed state from the last exact anchor to the target stage."""

    if input_coordinates.shape != residual_coordinates.shape:
        raise ValueError("input and residual state coordinates must align")
    if target_stage <= 0 or target_stage >= residual_coordinates.shape[0]:
        raise ValueError("target_stage lies outside the state trajectory")
    if horizon <= 0 or horizon > target_stage:
        raise ValueError("horizon must lie in [1, target_stage]")
    anchor = target_stage - horizon
    state = residual_coordinates[anchor]
    if method == "reuse":
        return state
    if method == "ar2_drift" and horizon != 1:
        raise ValueError("two-lag state is evaluated only for one-step prediction")
    previous_state = None if anchor == 0 else residual_coordinates[anchor - 1]
    for stage in range(anchor + 1, target_stage + 1):
        drift = input_coordinates[stage] - input_coordinates[stage - 1]
        coefficients = transitions[method][stage].to(
            device=state.device, dtype=state.dtype
        )
        if method == "ar1":
            next_state = coefficients[0] * state
        elif method == "drift":
            next_state = coefficients[0] * state + coefficients[1] * drift
        elif method == "ar2_drift":
            if previous_state is None:
                raise ValueError("two-lag prediction requires an earlier exact state")
            next_state = (
                coefficients[0] * state
                + coefficients[1] * previous_state
                + coefficients[2] * drift
            )
        else:
            raise ValueError(f"unsupported closure method: {method}")
        previous_state, state = state, next_state
    return state


def orthogonal_error_terms(
    target: torch.Tensor,
    target_coordinates: torch.Tensor,
    predicted_coordinates: torch.Tensor,
) -> tuple[float, float]:
    """Return prediction SSE and target energy without materializing a renderer."""

    if target_coordinates.shape != predicted_coordinates.shape:
        raise ValueError("target and predicted coordinates must align")
    target_energy = float(target.double().square().sum())
    coordinate_energy = float(target_coordinates.double().square().sum())
    orthogonal_energy = max(target_energy - coordinate_energy, 0.0)
    coordinate_error = float(
        (target_coordinates.double() - predicted_coordinates.double()).square().sum()
    )
    return orthogonal_energy + coordinate_error, target_energy


def capacity_error_terms(
    target: torch.Tensor, target_coordinates: torch.Tensor
) -> tuple[float, float]:
    target_energy = float(target.double().square().sum())
    coordinate_energy = float(target_coordinates.double().square().sum())
    return max(target_energy - coordinate_energy, 0.0), target_energy
