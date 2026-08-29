#!/usr/bin/env python3
"""Shared numerical core for the EXP-049 conditional-interface screen."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModuleTrajectory:
    """One identity, target, block, and CFG branch over denoising steps."""

    sample_id: str
    target_name: str
    block: int
    branch: int
    interface: torch.Tensor
    target: torch.Tensor
    block_output: torch.Tensor

    def validate(self) -> None:
        if self.interface.ndim != 3:
            raise ValueError("interface must have shape [steps, rows, channels]")
        if self.target.shape != self.interface.shape:
            raise ValueError("target and interface must have identical shapes")
        if self.block_output.shape != self.interface.shape:
            raise ValueError("block_output and interface must have identical shapes")
        if self.interface.shape[0] < 3:
            raise ValueError("a conditional trajectory requires at least three steps")
        if self.target_name not in {"self_attn", "ffn", "whole_block"}:
            raise ValueError(f"unsupported target: {self.target_name}")
        for name, tensor in (
            ("interface", self.interface),
            ("target", self.target),
            ("block_output", self.block_output),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")

    @property
    def channel_count(self) -> int:
        return int(self.target.shape[-1])


@dataclass(frozen=True)
class DiagonalField:
    """Three per-channel coefficients for two histories and current drift."""

    coefficients: torch.Tensor

    def validate(self) -> None:
        if self.coefficients.ndim != 2 or self.coefficients.shape[1] != 3:
            raise ValueError("diagonal coefficients must have shape [channels, 3]")
        if not torch.isfinite(self.coefficients).all():
            raise ValueError("diagonal coefficients contain non-finite values")


@dataclass(frozen=True)
class ScalarAR2:
    lag1: float
    lag2: float


def conditional_features(
    trajectory: ModuleTrajectory, target_step: int
) -> torch.Tensor:
    """Return `[rows, channels, 3]` causal features for one target step."""

    trajectory.validate()
    if target_step < 2 or target_step >= trajectory.target.shape[0]:
        raise ValueError("target_step must retain two exact historical states")
    drift = trajectory.interface[target_step] - trajectory.interface[target_step - 1]
    return torch.stack(
        (
            trajectory.target[target_step - 1],
            trajectory.target[target_step - 2],
            drift,
        ),
        dim=-1,
    )


def _ridge_system(
    gram: torch.Tensor, ridge: float
) -> torch.Tensor:
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    diagonal_mean = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    scale = ridge * diagonal_mean.clamp_min(torch.finfo(gram.dtype).eps)
    identity = torch.eye(
        gram.shape[-1], device=gram.device, dtype=gram.dtype
    )
    return gram + scale[..., None, None] * identity


def fit_diagonal_field(
    trajectories: list[ModuleTrajectory], target_step: int, ridge: float
) -> DiagonalField:
    """Fit a calibration-only channel field across identities and token rows."""

    if not trajectories:
        raise ValueError("cannot fit a diagonal field without trajectories")
    channel_count = trajectories[0].channel_count
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for trajectory in trajectories:
        trajectory.validate()
        if trajectory.channel_count != channel_count:
            raise ValueError("all calibration trajectories must share channels")
        features.append(conditional_features(trajectory, target_step))
        targets.append(trajectory.target[target_step])
    x = torch.cat(features, dim=0).double()
    y = torch.cat(targets, dim=0).double()
    gram = torch.einsum("nci,ncj->cij", x, x)
    rhs = torch.einsum("nci,nc->ci", x, y)
    coefficients = torch.linalg.solve(
        _ridge_system(gram, ridge), rhs.unsqueeze(-1)
    ).squeeze(-1)
    field = DiagonalField(coefficients=coefficients.float())
    field.validate()
    return field


def apply_diagonal_field(
    field: DiagonalField,
    trajectory: ModuleTrajectory,
    target_step: int,
) -> torch.Tensor:
    field.validate()
    features = conditional_features(trajectory, target_step)
    if features.shape[1] != field.coefficients.shape[0]:
        raise ValueError("field and trajectory channel counts differ")
    return torch.einsum(
        "nci,ci->nc", features, field.coefficients.to(features.device)
    )


def fit_scalar_ar2(
    trajectories: list[ModuleTrajectory], target_step: int, ridge: float
) -> ScalarAR2:
    if not trajectories:
        raise ValueError("cannot fit AR(2) without trajectories")
    x_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    for trajectory in trajectories:
        trajectory.validate()
        x_rows.append(
            torch.stack(
                (
                    trajectory.target[target_step - 1],
                    trajectory.target[target_step - 2],
                ),
                dim=-1,
            ).reshape(-1, 2)
        )
        y_rows.append(trajectory.target[target_step].reshape(-1))
    x = torch.cat(x_rows, dim=0).double()
    y = torch.cat(y_rows, dim=0).double()
    gram = x.T @ x
    rhs = x.T @ y
    coefficients = torch.linalg.solve(_ridge_system(gram, ridge), rhs)
    return ScalarAR2(lag1=float(coefficients[0]), lag2=float(coefficients[1]))


def apply_scalar_ar2(
    params: ScalarAR2, trajectory: ModuleTrajectory, target_step: int
) -> torch.Tensor:
    return (
        params.lag1 * trajectory.target[target_step - 1]
        + params.lag2 * trajectory.target[target_step - 2]
    )


def error_terms(
    prediction: torch.Tensor, trajectory: ModuleTrajectory, target_step: int
) -> dict[str, float]:
    """Return target-relative and induced additive block-output error terms."""

    target = trajectory.target[target_step]
    difference = prediction.double() - target.double()
    error_sq = float(difference.square().sum())
    target_sq = float(target.double().square().sum())
    block_output_sq = float(
        trajectory.block_output[target_step].double().square().sum()
    )
    return {
        "error_sq": error_sq,
        "target_sq": target_sq,
        "block_output_sq": block_output_sq,
        "target_relative_l2": (error_sq / max(target_sq, 1e-30)) ** 0.5,
        "block_output_relative_l2": (
            error_sq / max(block_output_sq, 1e-30)
        )
        ** 0.5,
    }


def oracle_gap_recovery(
    ar2_risk: float, deployable_risk: float, target_visible_risk: float
) -> float:
    denominator = ar2_risk - target_visible_risk
    if denominator <= 0:
        return float("-inf")
    return (ar2_risk - deployable_risk) / denominator


def zero_cost_speedup_ceiling(runtime_share: float, selected_fraction: float) -> float:
    """Optimistic Amdahl ceiling before candidate renderer cost is known."""

    if not 0 <= runtime_share <= 1:
        raise ValueError("runtime_share must lie in [0, 1]")
    if not 0 <= selected_fraction <= 1:
        raise ValueError("selected_fraction must lie in [0, 1]")
    return 1.0 / max(1.0 - runtime_share * selected_fraction, 1e-30)
