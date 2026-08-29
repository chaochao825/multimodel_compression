from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureStatistics:
    rows: int
    feature_sum: torch.Tensor
    gram: torch.Tensor


def feature_statistics(features: torch.Tensor) -> FeatureStatistics:
    if features.ndim < 2:
        raise ValueError("features must have at least two dimensions")
    matrix = features.reshape(-1, features.shape[-1]).float()
    return FeatureStatistics(
        rows=matrix.shape[0],
        feature_sum=matrix.sum(dim=0),
        gram=matrix.transpose(0, 1) @ matrix,
    )


def merge_statistics(statistics: Iterable[FeatureStatistics]) -> FeatureStatistics:
    values = list(statistics)
    if not values:
        raise ValueError("at least one statistics object is required")
    dimension = values[0].feature_sum.numel()
    if any(value.feature_sum.numel() != dimension for value in values):
        raise ValueError("feature dimensions differ")
    return FeatureStatistics(
        rows=sum(value.rows for value in values),
        feature_sum=torch.stack([value.feature_sum for value in values]).sum(dim=0),
        gram=torch.stack([value.gram for value in values]).sum(dim=0),
    )


def centered_covariance(statistics: FeatureStatistics) -> torch.Tensor:
    if statistics.rows < 2:
        raise ValueError("centered covariance requires at least two rows")
    mean_outer = torch.outer(statistics.feature_sum, statistics.feature_sum)
    covariance = (
        statistics.gram - mean_outer / statistics.rows
    ) / (statistics.rows - 1)
    return 0.5 * (covariance + covariance.transpose(0, 1))


def descending_eigenspace(
    matrix: torch.Tensor,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not 0 < rank < matrix.shape[0]:
        raise ValueError("rank must be between zero and the matrix dimension")
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    order = torch.arange(
        eigenvalues.numel() - 1,
        -1,
        -1,
        device=eigenvalues.device,
    )
    eigenvalues = eigenvalues.index_select(0, order)
    eigenvectors = eigenvectors.index_select(1, order)
    return eigenvalues, eigenvectors[:, :rank]


def subspace_squared_cosines(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("subspace bases must have the same two-dimensional shape")
    return torch.linalg.svdvals(left.transpose(0, 1) @ right).square()


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(subspace_squared_cosines(left, right).mean().item())


def eigengap_summary(
    eigenvalues: torch.Tensor,
    *,
    rank: int,
    radius: int = 16,
) -> dict[str, object]:
    if not 0 < rank < eigenvalues.numel():
        raise ValueError("rank is outside the eigenspectrum")
    start = max(0, rank - radius)
    stop = min(eigenvalues.numel(), rank + radius + 1)
    gap = eigenvalues[rank - 1] - eigenvalues[rank]
    scale = eigenvalues[rank - 1].abs().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    return {
        "rank": rank,
        "window_start_one_based": start + 1,
        "window_stop_one_based": stop,
        "window": [float(value) for value in eigenvalues[start:stop].tolist()],
        "absolute_gap": float(gap.item()),
        "relative_gap": float((gap / scale).item()),
    }


def bootstrap_subspace_stability(
    per_video: list[FeatureStatistics],
    reference_basis: torch.Tensor,
    *,
    sample_size: int,
    replicates: int,
    seed: int,
) -> list[dict[str, float | int]]:
    if not 0 < sample_size <= len(per_video):
        raise ValueError("sample size exceeds the video pool")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    output = []
    for replicate in range(replicates):
        indices = rng.choice(len(per_video), size=sample_size, replace=True)
        covariance = centered_covariance(
            merge_statistics([per_video[int(index)] for index in indices])
        )
        eigenvalues, basis = descending_eigenspace(
            covariance,
            rank=reference_basis.shape[1],
        )
        squared_cosines = subspace_squared_cosines(reference_basis, basis)
        output.append(
            {
                "replicate": replicate,
                "overlap": float(squared_cosines.mean().item()),
                "minimum_squared_cosine": float(squared_cosines.min().item()),
                "rank_eigenvalue": float(
                    eigenvalues[reference_basis.shape[1] - 1].item()
                ),
            }
        )
    return output


def channel_reader_risk(
    gradients: torch.Tensor,
    margins: torch.Tensor,
    *,
    feature_norm_squared: float,
    margin_floor: float,
) -> torch.Tensor:
    if gradients.ndim != 3:
        raise ValueError("gradients must have shape [competitors, tokens, channels]")
    if margins.shape != (gradients.shape[0],):
        raise ValueError("one margin is required per competitor")
    if margin_floor <= 0.0:
        raise ValueError("margin floor must be positive")
    risk = torch.zeros(
        (gradients.shape[-1], gradients.shape[-1]),
        device=gradients.device,
        dtype=torch.float32,
    )
    for gradient, margin in zip(gradients.float(), margins.float(), strict=True):
        denominator = margin.abs().clamp_min(margin_floor).square()
        risk.add_(
            gradient.transpose(0, 1) @ gradient,
            alpha=feature_norm_squared / float(denominator.item()),
        )
    return 0.5 * (risk + risk.transpose(0, 1))


def commutator_ratio(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("commutator operands must be equal square matrices")
    numerator = torch.linalg.vector_norm(left @ right - right @ left)
    denominator = (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    ).clamp_min(torch.finfo(left.dtype).eps)
    return float((numerator / denominator).item())


def tail_energy_fraction(eigenvalues: torch.Tensor, *, rank: int) -> float:
    if not 0 <= rank < eigenvalues.numel():
        raise ValueError("rank is outside the eigenspectrum")
    total = eigenvalues.clamp_min(0).sum().clamp_min(
        torch.finfo(eigenvalues.dtype).eps
    )
    return float((eigenvalues[rank:].clamp_min(0).sum() / total).item())


def local_linearity_summary(
    exact_shifts: torch.Tensor,
    linear_shifts: torch.Tensor,
    baseline_margins: torch.Tensor,
) -> dict[str, float]:
    if exact_shifts.shape != linear_shifts.shape or exact_shifts.shape != baseline_margins.shape:
        raise ValueError("linearity tensors must have equal shapes")
    exact = exact_shifts.float().flatten()
    linear = linear_shifts.float().flatten()
    margins = baseline_margins.float().flatten()
    centered_exact = exact - exact.mean()
    centered_linear = linear - linear.mean()
    correlation = torch.dot(centered_exact, centered_linear) / (
        torch.linalg.vector_norm(centered_exact)
        * torch.linalg.vector_norm(centered_linear)
    ).clamp_min(torch.finfo(torch.float32).eps)
    relative_error = torch.linalg.vector_norm(exact - linear) / torch.linalg.vector_norm(
        exact
    ).clamp_min(torch.finfo(torch.float32).eps)
    exact_flip = margins + exact <= 0
    linear_flip = margins + linear <= 0
    return {
        "pearson": float(correlation.item()),
        "relative_l2": float(relative_error.item()),
        "adverse_sign_agreement": float(((exact < 0) == (linear < 0)).float().mean().item()),
        "flip_agreement": float((exact_flip == linear_flip).float().mean().item()),
    }
