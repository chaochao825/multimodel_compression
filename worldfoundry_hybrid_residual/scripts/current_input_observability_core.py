#!/usr/bin/env python3
"""Causal predictors and metrics for the EXP-045 observability Gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


Shift3D = tuple[int, int, int]


@dataclass(frozen=True)
class CellTrajectory:
    """One identity, layer, and CFG branch over sampler steps."""

    block_input: torch.Tensor
    residual: torch.Tensor
    adaln: torch.Tensor
    qk_sketch: torch.Tensor
    thw: tuple[int, int, int]

    def validate(self) -> None:
        if self.block_input.ndim != 3:
            raise ValueError("block_input must have shape [steps, tokens, channels]")
        if self.residual.shape != self.block_input.shape:
            raise ValueError("residual and block_input must have the same shape")
        if self.adaln.ndim != 3 or self.adaln.shape[0] != self.step_count:
            raise ValueError("adaln must have shape [steps, 6, channels]")
        if self.adaln.shape[1:] != (6, self.channel_count):
            raise ValueError("adaln must contain all six channel modulation fields")
        if self.qk_sketch.ndim != 3:
            raise ValueError("qk_sketch must have shape [steps, tokens, features]")
        if self.qk_sketch.shape[:2] != self.block_input.shape[:2]:
            raise ValueError("qk_sketch step and token axes must match block_input")
        if self.token_count != self.thw[0] * self.thw[1] * self.thw[2]:
            raise ValueError("THW geometry does not match the trajectory token count")
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


@dataclass(frozen=True)
class ChannelParams:
    residual_scale: torch.Tensor
    input_scale: torch.Tensor


@dataclass(frozen=True)
class ScalarAR2Params:
    lag1: float
    lag2: float


@dataclass
class ScalarAR2Statistics:
    gram11: float = 0.0
    gram12: float = 0.0
    gram22: float = 0.0
    rhs1: float = 0.0
    rhs2: float = 0.0
    observations: int = 0

    def update(
        self,
        target: torch.Tensor,
        lag1: torch.Tensor,
        lag2: torch.Tensor,
    ) -> None:
        self.gram11 += float(lag1.double().square().sum())
        self.gram12 += float((lag1.double() * lag2.double()).sum())
        self.gram22 += float(lag2.double().square().sum())
        self.rhs1 += float((lag1.double() * target.double()).sum())
        self.rhs2 += float((lag2.double() * target.double()).sum())
        self.observations += int(target.numel())

    def fit(self, ridge: float) -> ScalarAR2Params:
        if self.observations <= 0:
            raise ValueError("cannot fit scalar AR(2) without calibration observations")
        scale = ridge * max(0.5 * (self.gram11 + self.gram22), 1e-30)
        gram11 = self.gram11 + scale
        gram22 = self.gram22 + scale
        determinant = max(gram11 * gram22 - self.gram12**2, 1e-30)
        return ScalarAR2Params(
            lag1=(self.rhs1 * gram22 - self.rhs2 * self.gram12) / determinant,
            lag2=(self.rhs2 * gram11 - self.rhs1 * self.gram12) / determinant,
        )


@dataclass(frozen=True)
class BroydenParams:
    input_scale: torch.Tensor
    secant_inputs: torch.Tensor
    secant_defects: torch.Tensor
    regularization: float


@dataclass(frozen=True)
class TransportParams:
    base: ChannelParams
    shifts: tuple[Shift3D, ...]
    coefficients: torch.Tensor


@dataclass(frozen=True)
class DPLRParams:
    input_scale: torch.Tensor
    output_basis: torch.Tensor
    input_basis: torch.Tensor
    singular_scale: torch.Tensor


@dataclass(frozen=True)
class PredictionResult:
    prediction: torch.Tensor
    effective_secants: int
    shifts: tuple[Shift3D, ...]


def shift_bank_75() -> tuple[Shift3D, ...]:
    """Return the preregistered 3x5x5 local spacetime shift bank."""

    return tuple(
        (dt, dy, dx)
        for dt in (-1, 0, 1)
        for dy in (-2, -1, 0, 1, 2)
        for dx in (-2, -1, 0, 1, 2)
    )


def nonperiodic_shift(
    tensor: torch.Tensor,
    thw: tuple[int, int, int],
    shift: Shift3D,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift token rows on THW with zeros outside the physical boundary."""

    if tensor.ndim != 2:
        raise ValueError("nonperiodic_shift expects [tokens, features]")
    t_size, h_size, w_size = thw
    if tensor.shape[0] != t_size * h_size * w_size:
        raise ValueError("THW geometry does not match token count")
    dt, dy, dx = shift
    source = tensor.reshape(t_size, h_size, w_size, tensor.shape[-1])
    output = torch.zeros_like(source)
    mask = torch.zeros(
        (t_size, h_size, w_size, 1), device=tensor.device, dtype=tensor.dtype
    )

    target_t = slice(max(dt, 0), min(t_size + dt, t_size))
    target_y = slice(max(dy, 0), min(h_size + dy, h_size))
    target_x = slice(max(dx, 0), min(w_size + dx, w_size))
    source_t = slice(max(-dt, 0), min(t_size - dt, t_size))
    source_y = slice(max(-dy, 0), min(h_size - dy, h_size))
    source_x = slice(max(-dx, 0), min(w_size - dx, w_size))
    output[target_t, target_y, target_x] = source[source_t, source_y, source_x]
    mask[target_t, target_y, target_x] = 1
    return output.reshape_as(tensor), mask.reshape(tensor.shape[0], 1)


def relative_l2_terms(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float]:
    error_sq = float((prediction.double() - target.double()).square().sum())
    target_sq = float(target.double().square().sum())
    relative = (error_sq / max(target_sq, 1e-30)) ** 0.5
    return relative, error_sq, target_sq


def oracle_recovery_fraction(
    ar2_risk: float, method_risk: float, oracle_risk: float
) -> float:
    denominator = ar2_risk - oracle_risk
    if denominator <= 0:
        return float("-inf")
    return (ar2_risk - method_risk) / denominator


def _regularizer(energy: torch.Tensor, ridge: float) -> torch.Tensor:
    return ridge * energy.clamp_min(torch.finfo(energy.dtype).eps)


def fit_channel_predictor(
    block_input: torch.Tensor,
    residual: torch.Tensor,
    anchor: int,
    ridge: float,
) -> ChannelParams:
    """Fit per-channel `a*r_prev + b*delta_h` from exact history only."""

    if anchor < 1:
        raise ValueError("channel predictor requires one exact transition")
    previous = residual[:anchor]
    drift = block_input[1 : anchor + 1] - block_input[:anchor]
    target = residual[1 : anchor + 1]
    reduce_dims = (0, 1)
    a11 = previous.square().sum(dim=reduce_dims)
    a22 = drift.square().sum(dim=reduce_dims)
    a12 = (previous * drift).sum(dim=reduce_dims)
    b1 = (previous * target).sum(dim=reduce_dims)
    b2 = (drift * target).sum(dim=reduce_dims)
    scale = _regularizer(0.5 * (a11 + a22), ridge)
    a11 = a11 + scale
    a22 = a22 + scale
    determinant = (a11 * a22 - a12.square()).clamp_min(
        torch.finfo(a11.dtype).eps
    )
    residual_scale = (b1 * a22 - b2 * a12) / determinant
    input_scale = (b2 * a11 - b1 * a12) / determinant
    return ChannelParams(residual_scale=residual_scale, input_scale=input_scale)


def fit_online_scalar_ar2(
    residual: torch.Tensor, anchor: int, ridge: float
) -> ScalarAR2Params:
    if anchor < 2:
        raise ValueError("online scalar AR(2) requires two exact transitions")
    statistics = ScalarAR2Statistics()
    for step in range(2, anchor + 1):
        statistics.update(residual[step], residual[step - 1], residual[step - 2])
    return statistics.fit(ridge)


def fit_broyden(
    block_input: torch.Tensor,
    residual: torch.Tensor,
    anchor: int,
    max_secants: int,
    ridge: float,
) -> BroydenParams:
    """Fit a diagonal base plus a low-rank multisecant correction."""

    if anchor < 1 or max_secants <= 0:
        raise ValueError("Broyden requires exact history and positive max_secants")
    input_steps = block_input[1 : anchor + 1] - block_input[:anchor]
    residual_steps = residual[1 : anchor + 1] - residual[:anchor]
    numerator = (input_steps * residual_steps).sum(dim=(0, 1))
    denominator = input_steps.square().sum(dim=(0, 1))
    input_scale = numerator / (
        denominator + _regularizer(denominator, ridge)
    ).clamp_min(torch.finfo(input_steps.dtype).eps)
    count = min(anchor, max_secants)
    secant_inputs = input_steps[-count:].reshape(count, -1).transpose(0, 1)
    base_outputs = input_steps[-count:] * input_scale
    secant_defects = (residual_steps[-count:] - base_outputs).reshape(
        count, -1
    ).transpose(0, 1)
    return BroydenParams(
        input_scale=input_scale,
        secant_inputs=secant_inputs,
        secant_defects=secant_defects,
        regularization=ridge,
    )


def apply_broyden(params: BroydenParams, input_drift: torch.Tensor) -> torch.Tensor:
    base = params.input_scale * input_drift
    flat = input_drift.reshape(-1)
    gram = params.secant_inputs.transpose(0, 1) @ params.secant_inputs
    rhs = params.secant_inputs.transpose(0, 1) @ flat
    mean_diagonal = gram.diagonal().mean().clamp_min(
        torch.finfo(gram.dtype).eps
    )
    regularized = gram + torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    ) * (params.regularization * mean_diagonal)
    coordinates = torch.linalg.solve(regularized, rhs)
    correction = (params.secant_defects @ coordinates).reshape_as(input_drift)
    return base + correction


def _transport_feature(
    residual: torch.Tensor,
    thw: tuple[int, int, int],
    shift: Shift3D,
) -> torch.Tensor:
    shifted, _ = nonperiodic_shift(residual, thw, shift)
    # The boundary drop is part of a nonperiodic transport operator. Masking it
    # would silently turn the edge into an identity map and bias the predictor.
    return shifted - residual


def select_observable_shifts(
    current_feature: torch.Tensor,
    previous_feature: torch.Tensor,
    thw: tuple[int, int, int],
    count: int,
    candidates: Iterable[Shift3D],
) -> tuple[Shift3D, ...]:
    """Select shifts by current/previous feature agreement without target residual."""

    if count <= 0:
        raise ValueError("transport expert count must be positive")
    scores: list[tuple[float, Shift3D]] = []
    current = current_feature.float()
    previous = previous_feature.float()
    for shift in candidates:
        if shift == (0, 0, 0):
            continue
        shifted, mask = nonperiodic_shift(previous, thw, shift)
        weighted_current = current * mask
        numerator = (weighted_current * shifted).double().sum().abs()
        denominator = (
            weighted_current.double().square().sum().sqrt()
            * shifted.double().square().sum().sqrt()
        ).clamp_min(1e-30)
        scores.append((float(numerator / denominator), shift))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return tuple(shift for _, shift in scores[:count])


def select_historical_shifts(
    block_input: torch.Tensor,
    residual: torch.Tensor,
    anchor: int,
    thw: tuple[int, int, int],
    count: int,
    candidates: Iterable[Shift3D],
    ridge: float,
    score_channels: int = 16,
) -> tuple[Shift3D, ...]:
    """Select shifts using only exact historical transition residuals."""

    if score_channels <= 0:
        raise ValueError("score_channels must be positive")
    channel_count = residual.shape[-1]
    indices = torch.linspace(
        0,
        channel_count - 1,
        min(score_channels, channel_count),
        device=residual.device,
        dtype=torch.float64,
    ).round().long()
    scored_input = block_input.index_select(-1, indices)
    scored_residual = residual.index_select(-1, indices)
    base = fit_channel_predictor(scored_input, scored_residual, anchor, ridge)
    drifts = scored_input[1 : anchor + 1] - scored_input[:anchor]
    targets = scored_residual[1 : anchor + 1] - (
        base.residual_scale * scored_residual[:anchor] + base.input_scale * drifts
    )
    scores: list[tuple[float, Shift3D]] = []
    for shift in candidates:
        if shift == (0, 0, 0):
            continue
        features = torch.stack(
            [
                _transport_feature(scored_residual[index], thw, shift)
                for index in range(anchor)
            ]
        )
        numerator = (features.double() * targets.double()).sum()
        denominator = features.double().square().sum().clamp_min(1e-30)
        scores.append((float(numerator.square() / denominator), shift))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return tuple(shift for _, shift in scores[:count])


def fit_transport(
    block_input: torch.Tensor,
    residual: torch.Tensor,
    anchor: int,
    thw: tuple[int, int, int],
    shifts: tuple[Shift3D, ...],
    ridge: float,
) -> TransportParams:
    if not shifts:
        raise ValueError("fit_transport requires at least one shift")
    drifts = block_input[1 : anchor + 1] - block_input[:anchor]
    transport_features = torch.stack(
        [
            torch.stack(
                [
                    _transport_feature(residual[index], thw, shift)
                    for shift in shifts
                ],
                dim=1,
            )
            for index in range(anchor)
        ],
        dim=0,
    )
    features = torch.cat(
        (
            residual[:anchor].unsqueeze(2),
            drifts.unsqueeze(2),
            transport_features,
        ),
        dim=2,
    )
    target = residual[1 : anchor + 1]
    # [step, token, expert, channel] -> one small solve per channel.
    gram = torch.einsum("stmc,stnc->cmn", features, features)
    rhs = torch.einsum("stmc,stc->cm", features, target)
    mean_diagonal = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(
        torch.finfo(gram.dtype).eps
    )
    identity = torch.eye(
        2 + len(shifts), device=gram.device, dtype=gram.dtype
    ).unsqueeze(0)
    regularized = gram + ridge * mean_diagonal[:, None, None] * identity
    coefficients = torch.linalg.solve(regularized, rhs.unsqueeze(-1)).squeeze(-1)
    base = ChannelParams(
        residual_scale=coefficients[:, 0], input_scale=coefficients[:, 1]
    )
    return TransportParams(
        base=base, shifts=shifts, coefficients=coefficients[:, 2:]
    )


def apply_transport(
    params: TransportParams,
    previous_residual: torch.Tensor,
    input_drift: torch.Tensor,
    thw: tuple[int, int, int],
) -> torch.Tensor:
    prediction = (
        params.base.residual_scale * previous_residual
        + params.base.input_scale * input_drift
    )
    for index, shift in enumerate(params.shifts):
        feature = _transport_feature(previous_residual, thw, shift)
        prediction = prediction + feature * params.coefficients[:, index]
    return prediction


def fit_dplr(
    block_input: torch.Tensor,
    residual: torch.Tensor,
    anchor: int,
    rank: int,
    ridge: float,
) -> DPLRParams:
    """Fit an activation-conditioned diagonal-plus-low-rank secant map."""

    if rank <= 0:
        raise ValueError("DPLR rank must be positive")
    input_steps = block_input[1 : anchor + 1] - block_input[:anchor]
    residual_steps = residual[1 : anchor + 1] - residual[:anchor]
    numerator = (input_steps * residual_steps).sum(dim=(0, 1))
    denominator = input_steps.square().sum(dim=(0, 1))
    input_scale = numerator / (
        denominator + _regularizer(denominator, ridge)
    ).clamp_min(torch.finfo(input_steps.dtype).eps)
    x = input_steps.reshape(-1, input_steps.shape[-1])
    y = (residual_steps - input_steps * input_scale).reshape(-1, x.shape[-1])
    cross = y.transpose(0, 1) @ x
    sketch_width = min(rank + 4, cross.shape[1])
    generator = torch.Generator(device=cross.device).manual_seed(2026082045)
    omega = torch.randn(
        cross.shape[1],
        sketch_width,
        device=cross.device,
        dtype=cross.dtype,
        generator=generator,
    )
    range_basis, _ = torch.linalg.qr(cross @ omega, mode="reduced")
    range_basis, _ = torch.linalg.qr(
        cross @ (cross.T @ range_basis), mode="reduced"
    )
    reduced = range_basis.T @ cross
    reduced_output, singular_values, input_basis_h = torch.linalg.svd(
        reduced, full_matrices=False
    )
    output_basis = range_basis @ reduced_output
    effective_rank = min(rank, output_basis.shape[1])
    output_basis = output_basis[:, :effective_rank]
    input_basis = input_basis_h[:effective_rank].transpose(0, 1)
    projected = x @ input_basis
    variance = projected.square().sum(dim=0)
    singular_scale = singular_values[:effective_rank] / (
        variance + _regularizer(variance, ridge)
    ).clamp_min(torch.finfo(variance.dtype).eps)
    return DPLRParams(
        input_scale=input_scale,
        output_basis=output_basis,
        input_basis=input_basis,
        singular_scale=singular_scale,
    )


def apply_dplr(params: DPLRParams, input_drift: torch.Tensor) -> torch.Tensor:
    coordinates = input_drift @ params.input_basis
    correction = (coordinates * params.singular_scale) @ params.output_basis.T
    return params.input_scale * input_drift + correction


def _fit_method(
    method: str,
    trajectory: CellTrajectory,
    anchor: int,
    target_step: int,
    ridge: float,
) -> object:
    h = trajectory.block_input
    r = trajectory.residual
    if method in {"ar2", "taylor1"}:
        return None
    if method == "online_ar2":
        return fit_online_scalar_ar2(r, anchor, ridge)
    if method == "diagonal":
        return fit_channel_predictor(h, r, anchor, ridge)
    if method.startswith("broyden") and "transport" not in method:
        count = int(method.removeprefix("broyden"))
        return fit_broyden(h, r, anchor, count, ridge)
    if method.startswith("dplr"):
        rank = int(method.removeprefix("dplr"))
        return fit_dplr(h, r, anchor, rank, ridge)
    if method.startswith("transport"):
        prefix, selection = method.split("_")
        count = int(prefix.removeprefix("transport"))
        if selection == "qk":
            shifts = select_observable_shifts(
                trajectory.qk_sketch[target_step],
                trajectory.qk_sketch[target_step - 1],
                trajectory.thw,
                count,
                shift_bank_75(),
            )
        elif selection == "history":
            shifts = select_historical_shifts(
                h, r, anchor, trajectory.thw, count, shift_bank_75(), ridge
            )
        else:
            raise ValueError(f"unsupported transport selector: {selection}")
        return fit_transport(h, r, anchor, trajectory.thw, shifts, ridge)
    raise ValueError(f"unsupported causal method: {method}")


def rollout_predict(
    trajectory: CellTrajectory,
    *,
    method: str,
    target_step: int,
    horizon: int,
    ridge: float = 1e-4,
    calibrated_ar2: dict[int, ScalarAR2Params] | None = None,
) -> PredictionResult:
    """Predict without reading residuals after the last exact anchor."""

    trajectory.validate()
    if horizon not in (1, 2, 3):
        raise ValueError("registered open-loop horizon must be 1, 2, or 3")
    anchor = target_step - horizon
    if target_step >= trajectory.step_count or anchor < 1:
        raise ValueError("target/horizon lacks the two residual states required by AR2")
    params = _fit_method(method, trajectory, anchor, target_step, ridge)
    states: dict[int, torch.Tensor] = {
        index: trajectory.residual[index] for index in range(anchor + 1)
    }
    effective_secants = 0
    shifts: tuple[Shift3D, ...] = ()
    if isinstance(params, BroydenParams):
        effective_secants = int(params.secant_inputs.shape[1])
    if isinstance(params, TransportParams):
        shifts = params.shifts
    for step in range(anchor + 1, target_step + 1):
        previous = states[step - 1]
        input_drift = trajectory.block_input[step] - trajectory.block_input[step - 1]
        if method == "ar2":
            if calibrated_ar2 is None or step not in calibrated_ar2:
                raise ValueError(f"calibrated AR(2) parameters missing for step {step}")
            coefficients = calibrated_ar2[step]
            prediction = (
                coefficients.lag1 * previous
                + coefficients.lag2 * states[step - 2]
            )
        elif method == "taylor1":
            prediction = 2 * previous - states[step - 2]
        elif isinstance(params, ScalarAR2Params):
            prediction = params.lag1 * previous + params.lag2 * states[step - 2]
        elif isinstance(params, ChannelParams):
            prediction = (
                params.residual_scale * previous + params.input_scale * input_drift
            )
        elif isinstance(params, BroydenParams):
            prediction = previous + apply_broyden(params, input_drift)
        elif isinstance(params, TransportParams):
            prediction = apply_transport(
                params, previous, input_drift, trajectory.thw
            )
        elif isinstance(params, DPLRParams):
            prediction = previous + apply_dplr(params, input_drift)
        else:
            raise TypeError("fitted predictor has an unsupported type")
        states[step] = prediction
    return PredictionResult(
        prediction=states[target_step],
        effective_secants=effective_secants,
        shifts=shifts,
    )


def target_visible_transport_oracle(
    trajectory: CellTrajectory,
    *,
    target_step: int,
    ridge: float = 1e-6,
) -> PredictionResult:
    """Compute the preregistered target-visible 75-shift token-LS ceiling."""

    trajectory.validate()
    if target_step < 2 or target_step >= trajectory.step_count:
        raise ValueError("oracle requires two prior exact residual states")
    target = trajectory.residual[target_step]
    previous = trajectory.residual[target_step - 1]
    previous2 = trajectory.residual[target_step - 2]
    best_error = torch.full(
        (trajectory.token_count,),
        float("inf"),
        device=target.device,
        dtype=torch.float64,
    )
    best_prediction = torch.zeros_like(target)
    for shift in shift_bank_75():
        x1, mask1 = nonperiodic_shift(previous, trajectory.thw, shift)
        x2, mask2 = nonperiodic_shift(previous2, trajectory.thw, shift)
        valid = mask1 * mask2
        a11 = x1.double().square().sum(dim=1)
        a22 = x2.double().square().sum(dim=1)
        a12 = (x1.double() * x2.double()).sum(dim=1)
        b1 = (x1.double() * target.double()).sum(dim=1)
        b2 = (x2.double() * target.double()).sum(dim=1)
        scale = ridge * (0.5 * (a11 + a22)).clamp_min(1e-30)
        determinant = ((a11 + scale) * (a22 + scale) - a12.square()).clamp_min(
            1e-30
        )
        coefficient1 = (b1 * (a22 + scale) - b2 * a12) / determinant
        coefficient2 = (b2 * (a11 + scale) - b1 * a12) / determinant
        candidate = (
            coefficient1[:, None].to(target.dtype) * x1
            + coefficient2[:, None].to(target.dtype) * x2
        ) * valid
        error = (candidate.double() - target.double()).square().sum(dim=1)
        error = torch.where(valid[:, 0].bool(), error, best_error)
        improved = error < best_error
        best_prediction[improved] = candidate[improved]
        best_error = torch.minimum(best_error, error)
    return PredictionResult(
        prediction=best_prediction,
        effective_secants=0,
        shifts=shift_bank_75(),
    )
