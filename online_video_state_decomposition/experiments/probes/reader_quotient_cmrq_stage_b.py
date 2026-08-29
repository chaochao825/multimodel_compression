from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class DomainMoments:
    mean: torch.Tensor
    covariance: torch.Tensor


def equally_weighted_moments(
    domains: Iterable[DomainMoments],
) -> DomainMoments:
    values = list(domains)
    if not values:
        raise ValueError("at least one domain is required")
    dimension = values[0].mean.numel()
    if any(value.mean.shape != (dimension,) for value in values):
        raise ValueError("domain means have incompatible shapes")
    if any(value.covariance.shape != (dimension, dimension) for value in values):
        raise ValueError("domain covariances have incompatible shapes")
    mean = torch.stack([value.mean for value in values]).mean(dim=0)
    covariance = torch.zeros_like(values[0].covariance)
    for value in values:
        offset = value.mean - mean
        covariance.add_(value.covariance + torch.outer(offset, offset))
    covariance.div_(len(values))
    return DomainMoments(
        mean=mean,
        covariance=0.5 * (covariance + covariance.transpose(0, 1)),
    )


def projected_top_atoms(
    matrix: torch.Tensor,
    bulk_basis: torch.Tensor,
    *,
    atom_count: int,
) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if bulk_basis.ndim != 2 or bulk_basis.shape[0] != matrix.shape[0]:
        raise ValueError("bulk basis and matrix dimensions differ")
    if not 0 < atom_count <= matrix.shape[0] - bulk_basis.shape[1]:
        raise ValueError("atom count exceeds the orthogonal complement")
    matrix_bulk = matrix @ bulk_basis
    projected = (
        matrix
        - bulk_basis @ matrix_bulk.transpose(0, 1)
        - matrix_bulk @ bulk_basis.transpose(0, 1)
        + bulk_basis
        @ (bulk_basis.transpose(0, 1) @ matrix_bulk)
        @ bulk_basis.transpose(0, 1)
    )
    projected = 0.5 * (projected + projected.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(projected)
    order = torch.arange(
        eigenvalues.numel() - 1,
        eigenvalues.numel() - atom_count - 1,
        -1,
        device=eigenvalues.device,
    )
    atoms = eigenvectors.index_select(1, order)
    atoms = atoms - bulk_basis @ (bulk_basis.transpose(0, 1) @ atoms)
    return torch.linalg.qr(atoms, mode="reduced").Q


def random_complement_atoms(
    bulk_basis: torch.Tensor,
    *,
    atom_count: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    atoms = torch.randn(
        bulk_basis.shape[0],
        atom_count,
        generator=generator,
        dtype=torch.float32,
    ).to(device=bulk_basis.device, dtype=bulk_basis.dtype)
    atoms = atoms - bulk_basis @ (bulk_basis.transpose(0, 1) @ atoms)
    return torch.linalg.qr(atoms, mode="reduced").Q


def fixed_rank_hybrid(
    feature_basis: torch.Tensor,
    atom_source: torch.Tensor,
    *,
    rank: int,
    atom_count: int,
) -> torch.Tensor:
    if feature_basis.ndim != 2 or atom_source.ndim != 2:
        raise ValueError("basis tensors must be two-dimensional")
    if feature_basis.shape[0] != atom_source.shape[0]:
        raise ValueError("basis tensors have different channel dimensions")
    if not 0 < atom_count < rank <= feature_basis.shape[1]:
        raise ValueError("invalid rank or atom count")
    if atom_source.shape[1] < atom_count:
        raise ValueError("atom source is too small")
    bulk = feature_basis[:, : rank - atom_count]
    atoms = atom_source[:, :atom_count]
    atoms = atoms - bulk @ (bulk.transpose(0, 1) @ atoms)
    atoms = torch.linalg.qr(atoms, mode="reduced").Q
    return torch.cat([bulk, atoms], dim=1)


def boundary_mixed_basis(
    feature_basis: torch.Tensor,
    atom_source: torch.Tensor,
    feature_covariance: torch.Tensor,
    risk_matrix: torch.Tensor,
    *,
    rank: int,
    atom_count: int,
    risk_weight: float,
) -> torch.Tensor:
    if risk_weight <= 0.0:
        raise ValueError("risk weight must be positive")
    if feature_covariance.shape != risk_matrix.shape:
        raise ValueError("feature and risk matrices must have equal shapes")
    bulk = feature_basis[:, : rank - atom_count]
    feature_tail = feature_basis[:, rank - atom_count : rank]
    atoms = atom_source[:, :atom_count]
    atoms = atoms - bulk @ (bulk.transpose(0, 1) @ atoms)
    union = torch.linalg.qr(torch.cat([feature_tail, atoms], dim=1), mode="reduced").Q
    feature_scale = torch.trace(feature_covariance).clamp_min(
        torch.finfo(feature_covariance.dtype).eps
    )
    risk_scale = torch.trace(risk_matrix).clamp_min(
        torch.finfo(risk_matrix.dtype).eps
    )
    score = union.transpose(0, 1) @ (
        feature_covariance / feature_scale + risk_weight * risk_matrix / risk_scale
    ) @ union
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (score + score.transpose(0, 1)))
    order = torch.arange(
        eigenvalues.numel() - 1,
        eigenvalues.numel() - atom_count - 1,
        -1,
        device=eigenvalues.device,
    )
    boundary = union @ eigenvectors.index_select(1, order)
    boundary = boundary - bulk @ (bulk.transpose(0, 1) @ boundary)
    boundary = torch.linalg.qr(boundary, mode="reduced").Q
    return torch.cat([bulk, boundary], dim=1)


def trace_capture(matrix: torch.Tensor, basis: torch.Tensor) -> float:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if basis.ndim != 2 or basis.shape[0] != matrix.shape[0]:
        raise ValueError("basis and matrix dimensions differ")
    denominator = torch.trace(matrix).clamp_min(torch.finfo(matrix.dtype).eps)
    numerator = torch.trace(basis.transpose(0, 1) @ matrix @ basis)
    return float((numerator / denominator).item())


def orthogonality_error(basis: torch.Tensor) -> float:
    identity = torch.eye(
        basis.shape[1],
        device=basis.device,
        dtype=basis.dtype,
    )
    return float(
        torch.linalg.matrix_norm(basis.transpose(0, 1) @ basis - identity).item()
    )


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize_exact_rows(
    rows: list[dict[str, object]],
    *,
    margin_floor: float,
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("exact summary requires at least one row")
    if margin_floor <= 0.0:
        raise ValueError("margin floor must be positive")
    feature_l2 = [float(row["feature_relative_l2"]) for row in rows]
    candidate_kl = [float(row["candidate_kl"]) for row in rows]
    adverse_ratio = [float(row["maximum_normalized_adverse_shift"]) for row in rows]
    prediction_match = [int(row["prediction_match"]) for row in rows]
    harmful = [int(row["harmful"]) for row in rows]
    beneficial = [int(row["beneficial"]) for row in rows]
    outside_near_tie = [
        row for row in rows if float(row["minimum_margin"]) > margin_floor
    ]
    if not outside_near_tie:
        raise ValueError("all exact-reader samples are inside the near-tie set")
    return {
        "sample_count": len(rows),
        "feature_relative_l2_mean": float(np.mean(feature_l2)),
        "feature_relative_l2_p95": quantile(feature_l2, 0.95),
        "candidate_kl_mean": float(np.mean(candidate_kl)),
        "candidate_kl_p95": quantile(candidate_kl, 0.95),
        "agreement": float(np.mean(prediction_match)),
        "mismatch_count": len(rows) - sum(prediction_match),
        "near_tie_count": len(rows) - len(outside_near_tie),
        "agreement_outside_near_tie": float(
            np.mean([int(row["prediction_match"]) for row in outside_near_tie])
        ),
        "mismatch_outside_near_tie": sum(
            1 for row in outside_near_tie if int(row["prediction_match"]) == 0
        ),
        "harmful_count": sum(harmful),
        "beneficial_count": sum(beneficial),
        "normalized_adverse_mean": float(np.mean(adverse_ratio)),
        "normalized_adverse_p95": quantile(adverse_ratio, 0.95),
        "normalized_adverse_max": max(adverse_ratio),
    }


def summarize_progressive_fallback(
    rows: list[dict[str, object]],
    *,
    margin_threshold: float,
    compressed_state_bytes: int,
    dense_state_bytes: int,
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("progressive summary requires at least one row")
    if margin_threshold < 0.0:
        raise ValueError("margin threshold must be non-negative")
    if compressed_state_bytes <= 0 or dense_state_bytes <= compressed_state_bytes:
        raise ValueError("state byte counts are invalid")
    fallback = [
        float(row["approximate_top1_margin"]) <= margin_threshold for row in rows
    ]
    retained = [row for row, exact in zip(rows, fallback, strict=True) if not exact]
    fallback_count = sum(fallback)
    fallback_rate = fallback_count / len(rows)
    effective_kl = [
        0.0 if exact else float(row["candidate_kl"])
        for row, exact in zip(rows, fallback, strict=True)
    ]
    effective_l2 = [
        0.0 if exact else float(row["feature_relative_l2"])
        for row, exact in zip(rows, fallback, strict=True)
    ]
    conservative_bytes = (
        compressed_state_bytes + fallback_rate * dense_state_bytes
    )
    ideal_preroute_bytes = (
        (1.0 - fallback_rate) * compressed_state_bytes
        + fallback_rate * dense_state_bytes
    )
    return {
        "sample_count": len(rows),
        "margin_threshold": margin_threshold,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "remaining_mismatch_count": sum(
            1 - int(row["prediction_match"]) for row in retained
        ),
        "remaining_harmful_count": sum(int(row["harmful"]) for row in retained),
        "effective_candidate_kl_mean": float(np.mean(effective_kl)),
        "effective_candidate_kl_p95": quantile(effective_kl, 0.95),
        "effective_feature_l2_mean": float(np.mean(effective_l2)),
        "conservative_transfer_ratio": dense_state_bytes / conservative_bytes,
        "ideal_preroute_transfer_ratio": dense_state_bytes / ideal_preroute_bytes,
    }
