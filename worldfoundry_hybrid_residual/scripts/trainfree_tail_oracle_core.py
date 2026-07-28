#!/usr/bin/env python3
"""Numerical kernels for train-free sparse-critical Attention tail oracles."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def token_coordinates(
    grid_size: tuple[int, int, int] | list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(grid_size) != 3 or any(int(size) <= 0 for size in grid_size):
        raise ValueError(f"invalid T/H/W grid: {grid_size}")
    axes = [
        torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)
        for size in grid_size
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 3)


def oracle_mass_block_selection(
    probabilities: torch.Tensor,
    block_size: int,
    density: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a tile-shared dense-mass oracle block mask and token mask."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [queries, keys]")
    if block_size <= 0 or not 0.0 < density < 1.0:
        raise ValueError("block_size must be positive and density must lie in (0, 1)")
    tokens = probabilities.shape[1]
    blocks = math.ceil(tokens / block_size)
    padded = blocks * block_size
    if padded != tokens:
        probabilities = torch.nn.functional.pad(probabilities, (0, padded - tokens))
    block_mass = probabilities.reshape(probabilities.shape[0], blocks, block_size).sum(2)
    selected_count = max(1, min(blocks - 1, int(round(blocks * density))))
    selected_indices = block_mass.mean(0).topk(selected_count).indices
    selected_blocks = torch.zeros(blocks, dtype=torch.bool, device=probabilities.device)
    selected_blocks[selected_indices] = True
    token_blocks = torch.arange(tokens, device=probabilities.device) // block_size
    return selected_blocks, selected_blocks.index_select(0, token_blocks)


def _tail_numerator(
    weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if values.ndim == 2:
        return weights @ values
    if values.ndim == 3:
        if values.shape[:2] != weights.shape:
            raise ValueError("query-dependent tail values do not match tail weights")
        return torch.einsum("qg,qgd->qd", weights, values)
    raise ValueError("tail values must have shape [groups, d] or [queries, groups, d]")


def combine_shared_logits(
    scores: torch.Tensor,
    values: torch.Tensor,
    selected_keys: torch.Tensor,
    tail_logits: torch.Tensor,
    tail_values: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine exact and approximate terms before one shared normalization."""

    if scores.ndim != 2 or values.ndim != 2:
        raise ValueError("scores and values must be rank-2")
    if selected_keys.shape != (scores.shape[1],):
        raise ValueError("selected key mask has the wrong shape")
    if tail_logits.ndim != 2 or tail_logits.shape[0] != scores.shape[0]:
        raise ValueError("tail logits must have shape [queries, groups]")
    if tail_logits.shape[1] == 0:
        raise ValueError("tail approximation must contain at least one group")

    exact_logits = scores[:, selected_keys]
    maxima = tail_logits.max(1).values
    if exact_logits.shape[1]:
        maxima = torch.maximum(maxima, exact_logits.max(1).values)
    tail_weights = torch.exp(tail_logits - maxima[:, None])
    numerator = _tail_numerator(tail_weights, tail_values)
    denominator = tail_weights.sum(1)
    if exact_logits.shape[1]:
        exact_weights = torch.exp(exact_logits - maxima[:, None])
        numerator = numerator + exact_weights @ values[selected_keys]
        denominator = denominator + exact_weights.sum(1)
    output = numerator / denominator.clamp_min(1e-30)[:, None]
    return output, {
        "shared_denominator_min": float(denominator.min()),
        "shared_denominator_nonpositive_fraction": float((denominator <= 0).float().mean()),
    }


def combine_shared_signed_weights(
    scores: torch.Tensor,
    values: torch.Tensor,
    selected_keys: torch.Tensor,
    tail_weights: torch.Tensor,
    tail_values: torch.Tensor,
    shift: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine exact weights and signed polynomial tail weights once."""

    if shift.shape != (scores.shape[0],):
        raise ValueError("shift must have one value per query")
    exact_logits = scores[:, selected_keys]
    numerator = _tail_numerator(tail_weights, tail_values)
    denominator = tail_weights.sum(1)
    if exact_logits.shape[1]:
        exact_weights = torch.exp(exact_logits - shift[:, None])
        numerator = numerator + exact_weights @ values[selected_keys]
        denominator = denominator + exact_weights.sum(1)
    safe = torch.where(
        denominator.abs() >= 1e-30,
        denominator,
        torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
        * 1e-30,
    )
    return numerator / safe[:, None], {
        "shared_denominator_min": float(denominator.min()),
        "shared_denominator_nonpositive_fraction": float((denominator <= 0).float().mean()),
    }


def polynomial_tail_output(
    scores: torch.Tensor,
    values: torch.Tensor,
    selected_keys: torch.Tensor,
    order: int,
    center_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not 1 <= order <= 8:
        raise ValueError("polynomial order must lie in [1, 8]")
    tail_scores = scores[:, ~selected_keys]
    tail_values = values[~selected_keys]
    if center_mode == "mean":
        center = tail_scores.mean(1)
    elif center_mode == "midrange":
        center = 0.5 * (tail_scores.min(1).values + tail_scores.max(1).values)
    else:
        raise ValueError(f"unsupported polynomial center: {center_mode}")
    delta = tail_scores - center[:, None]
    polynomial = torch.ones_like(delta)
    term = torch.ones_like(delta)
    for degree in range(1, order + 1):
        term = term * delta / degree
        polynomial = polynomial + term
    shift = scores.max(1).values
    tail_weights = torch.exp(center - shift)[:, None] * polynomial
    output, diagnostics = combine_shared_signed_weights(
        scores,
        values,
        selected_keys,
        tail_weights,
        tail_values,
        shift,
    )
    diagnostics.update(
        {
            "negative_tail_weight_fraction": float((tail_weights < 0).float().mean()),
            "tail_score_range_mean": float(
                (tail_scores.max(1).values - tail_scores.min(1).values).mean()
            ),
            "tail_score_std_mean": float(tail_scores.std(1, unbiased=False).mean()),
            "tail_max_abs_centered_score": float(delta.abs().max()),
        }
    )
    return output, diagnostics


def standardize_feature_family(tensor: torch.Tensor) -> torch.Tensor:
    centered = tensor - tensor.mean(0, keepdim=True)
    scale = centered.square().mean(0, keepdim=True).sqrt().clamp_min(1e-5)
    return centered / scale / math.sqrt(tensor.shape[1])


def build_coreset_features(
    keys: torch.Tensor,
    values: torch.Tensor,
    coordinates: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    key_features = standardize_feature_family(keys)
    if variant == "k_only":
        return key_features
    value_features = standardize_feature_family(values)
    if variant == "joint_kv":
        return torch.cat((key_features, value_features), dim=1)
    if variant == "value_aware_kv_thw":
        position = coordinates / math.sqrt(coordinates.shape[1])
        return torch.cat((key_features, value_features, 0.5 * position), dim=1)
    raise ValueError(f"unsupported coreset feature variant: {variant}")


def output_leverage_importance(
    tail_probabilities: torch.Tensor,
    tail_values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    value_sq = tail_values.square().sum(1)[None]
    reference_sq = reference.square().sum(1)[:, None]
    cross = reference @ tail_values.T
    difference_sq = (value_sq + reference_sq - 2.0 * cross).clamp_min(0)
    importance = (tail_probabilities.square() * difference_sq).mean(0).sqrt()
    return importance.clamp_min(torch.finfo(importance.dtype).eps)


def nearest_assignments(
    features: torch.Tensor,
    centers: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    center_sq = centers.square().sum(1)
    assignments = []
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start : start + chunk_size]
        distance = (
            chunk.square().sum(1, keepdim=True)
            + center_sq[None]
            - 2.0 * chunk @ centers.T
        )
        assignments.append(distance.argmin(1))
    return torch.cat(assignments)


def weighted_kmeans_assignments(
    features: torch.Tensor,
    weights: torch.Tensor,
    clusters: int,
    *,
    iterations: int,
    fit_tokens: int,
    seed: int,
) -> torch.Tensor:
    if features.ndim != 2 or weights.shape != (features.shape[0],):
        raise ValueError("features/weights have incompatible shapes")
    if not 0 < clusters <= features.shape[0]:
        raise ValueError("invalid cluster count")
    if iterations <= 0 or fit_tokens < clusters:
        raise ValueError("iterations must be positive and fit_tokens >= clusters")
    weights = weights.float().clamp_min(0)
    weights = weights / weights.sum().clamp_min(1e-30)
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    fit_count = min(int(fit_tokens), features.shape[0])
    fit_indices = torch.multinomial(
        weights,
        fit_count,
        replacement=False,
        generator=generator,
    )
    fit_features = features.index_select(0, fit_indices)
    fit_weights = weights.index_select(0, fit_indices)
    init_probabilities = fit_weights / fit_weights.sum().clamp_min(1e-30)
    center_indices = torch.multinomial(
        init_probabilities,
        clusters,
        replacement=False,
        generator=generator,
    )
    centers = fit_features.index_select(0, center_indices).clone()
    for _ in range(iterations):
        assignment = nearest_assignments(fit_features, centers)
        sums = torch.zeros_like(centers)
        totals = torch.zeros(clusters, dtype=fit_weights.dtype, device=features.device)
        sums.index_add_(0, assignment, fit_features * fit_weights[:, None])
        totals.index_add_(0, assignment, fit_weights)
        nonempty = totals > 0
        centers[nonempty] = sums[nonempty] / totals[nonempty, None]
    return nearest_assignments(features, centers)


def cluster_centroids(
    keys: torch.Tensor,
    values: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.bincount(assignments, minlength=clusters).to(keys.dtype)
    key_sums = keys.new_zeros((clusters, keys.shape[1]))
    value_sums = values.new_zeros((clusters, values.shape[1]))
    key_sums.index_add_(0, assignments, keys)
    value_sums.index_add_(0, assignments, values)
    active = counts > 0
    return (
        counts[active],
        key_sums[active] / counts[active, None],
        value_sums[active] / counts[active, None],
    )


def coreset_tail_output(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    coordinates: torch.Tensor,
    scores: torch.Tensor,
    probabilities: torch.Tensor,
    reference: torch.Tensor,
    selected_keys: torch.Tensor,
    *,
    clusters: int,
    variant: str,
    scale: float,
    iterations: int,
    fit_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    tail_keys = keys[~selected_keys]
    tail_values = values[~selected_keys]
    tail_coordinates = coordinates[~selected_keys]
    tail_probabilities = probabilities[:, ~selected_keys]
    features = build_coreset_features(tail_keys, tail_values, tail_coordinates, variant)
    if variant == "value_aware_kv_thw":
        weights = output_leverage_importance(tail_probabilities, tail_values, reference)
    else:
        weights = torch.ones(tail_keys.shape[0], device=keys.device, dtype=keys.dtype)
    assignments = weighted_kmeans_assignments(
        features,
        weights,
        clusters,
        iterations=iterations,
        fit_tokens=fit_tokens,
        seed=seed,
    )
    counts, key_means, value_means = cluster_centroids(
        tail_keys, tail_values, assignments, clusters
    )
    tail_logits = queries @ key_means.T * scale + counts.log()[None]
    output, diagnostics = combine_shared_logits(
        scores, values, selected_keys, tail_logits, value_means
    )
    diagnostics.update(
        {
            "active_tail_groups": float(counts.numel()),
            "empty_cluster_fraction": float(1.0 - counts.numel() / clusters),
        }
    )
    return output, diagnostics


def prepare_group_moments(
    keys: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
    components: int,
    max_rank: int,
) -> dict[str, torch.Tensor | int]:
    if block_size <= 0 or components <= 0 or block_size % components:
        raise ValueError("components must divide block_size")
    group_size = block_size // components
    tokens, key_dimension = keys.shape
    if values.shape[0] != tokens:
        raise ValueError("keys and values must have the same token count")
    value_dimension = values.shape[1]
    groups = math.ceil(tokens / group_size)
    padded_tokens = groups * group_size
    padding = padded_tokens - tokens
    if padding:
        keys = torch.cat((keys, keys.new_zeros((padding, key_dimension))))
        values = torch.cat((values, values.new_zeros((padding, value_dimension))))
    valid = torch.arange(padded_tokens, device=keys.device) < tokens
    valid = valid.reshape(groups, group_size)
    counts = valid.sum(1).to(keys.dtype)
    mask = valid[:, :, None].to(keys.dtype)
    grouped_keys = keys.reshape(groups, group_size, key_dimension)
    grouped_values = values.reshape(groups, group_size, value_dimension)
    key_mean = (grouped_keys * mask).sum(1) / counts[:, None]
    value_mean = (grouped_values * mask).sum(1) / counts[:, None]
    centered_keys = (grouped_keys - key_mean[:, None]) * mask
    centered_values = (grouped_values - value_mean[:, None]) * mask
    key_variance = centered_keys.square().sum(1) / counts[:, None]
    diagonal_cross = (
        (centered_values * centered_keys).sum(1) / counts[:, None]
        if value_dimension == key_dimension
        else values.new_zeros((groups, 0))
    )

    available_rank = min(max_rank, group_size, key_dimension)
    if available_rank > 0:
        u, singular, vh = torch.linalg.svd(centered_keys, full_matrices=False)
        u = u[:, :, :available_rank]
        singular = singular[:, :available_rank]
        vh = vh[:, :available_rank]
        root_count = counts.sqrt()[:, None, None]
        d_key = vh.transpose(1, 2) * singular[:, None, :] / root_count
        d_value = (
            torch.einsum("gsd,gsr->gdr", centered_values, u) / root_count
        )
    else:
        d_key = keys.new_zeros((groups, key_dimension, 0))
        d_value = values.new_zeros((groups, value_dimension, 0))
    return {
        "group_size": group_size,
        "components": components,
        "counts": counts,
        "key_mean": key_mean,
        "value_mean": value_mean,
        "key_variance": key_variance,
        "diagonal_cross_covariance": diagonal_cross,
        "d_key": d_key,
        "d_value": d_value,
        "parent_blocks": torch.arange(groups, device=keys.device) // components,
    }


def covariance_tail_output(
    queries: torch.Tensor,
    scores: torch.Tensor,
    values: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_keys: torch.Tensor,
    moments: dict[str, torch.Tensor | int],
    *,
    variant: str,
    rank: int,
    scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    parent = moments["parent_blocks"]
    if not isinstance(parent, torch.Tensor):
        raise TypeError("invalid parent block tensor")
    active = ~selected_blocks.index_select(0, parent)
    counts = moments["counts"][active]
    key_mean = moments["key_mean"][active]
    value_mean = moments["value_mean"][active]
    q_scaled = queries * scale
    tail_logits = q_scaled @ key_mean.T + counts.log()[None]
    tail_values: torch.Tensor = value_mean
    if variant == "diag_gaussian":
        variance = moments["key_variance"][active]
        cross = moments["diagonal_cross_covariance"][active]
        if cross.shape[1] != q_scaled.shape[1]:
            raise ValueError(
                "diagonal Gaussian requires equal query/key and value dimensions"
            )
        tail_logits = tail_logits + 0.5 * (q_scaled.square() @ variance.T)
        tail_values = value_mean[None] + q_scaled[:, None] * cross[None]
    elif variant == "lowrank_gaussian":
        d_key = moments["d_key"][active, :, :rank]
        d_value = moments["d_value"][active, :, :rank]
        projection = torch.einsum("qd,gdr->qgr", q_scaled, d_key)
        tail_logits = tail_logits + 0.5 * projection.square().sum(2)
        tail_values = value_mean[None] + torch.einsum(
            "qgr,gdr->qgd", projection, d_value
        )
    elif variant != "centroid":
        raise ValueError(f"unsupported covariance variant: {variant}")
    output, diagnostics = combine_shared_logits(
        scores, values, selected_keys, tail_logits, tail_values
    )
    diagnostics["active_tail_groups"] = float(active.sum())
    return output, diagnostics


def covariance_query_work_ratio(
    selected_keys: int,
    tokens: int,
    active_groups: int,
    variant: str,
    rank: int,
) -> float:
    if variant == "centroid":
        group_multiplier = 1.0
    elif variant == "diag_gaussian":
        group_multiplier = 2.0
    elif variant == "lowrank_gaussian":
        group_multiplier = 1.0 + rank
    else:
        raise ValueError(f"unsupported covariance variant: {variant}")
    return (selected_keys + group_multiplier * active_groups) / tokens


def lowrank_covariance_products(
    moments: dict[str, torch.Tensor | int], rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reconstructed K covariance and V/K cross covariance for tests."""

    d_key = moments["d_key"][:, :, :rank]
    d_value = moments["d_value"][:, :, :rank]
    return (
        torch.einsum("gdr,ger->gde", d_key, d_key),
        torch.einsum("gdr,ger->gde", d_value, d_key),
    )


def finite_diagnostics(diagnostics: dict[str, Any]) -> None:
    for name, value in diagnostics.items():
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            raise ValueError(f"non-finite diagnostic {name}={value}")
