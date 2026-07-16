from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def relative_fro_error(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    return float(
        np.linalg.norm(target64 - prediction64)
        / (np.linalg.norm(target64) + 1e-12)
    )


def singular_value_metrics(
    matrix: np.ndarray,
    ranks: Iterable[int],
) -> dict[str, object]:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {x.shape}")
    singular_values = np.linalg.svd(x, compute_uv=False)
    energy = singular_values * singular_values
    total = float(energy.sum())
    if total <= 1e-24:
        probabilities = np.zeros_like(energy)
        effective_rank = 0.0
        stable_rank = 0.0
    else:
        probabilities = energy / total
        nonzero = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
        stable_rank = float(total / (energy[0] + 1e-24))
    cumulative = np.cumsum(energy) / (total + 1e-24)
    energy_at_rank = {
        str(int(rank)): float(cumulative[min(int(rank), len(cumulative)) - 1])
        if rank > 0 and len(cumulative)
        else 0.0
        for rank in ranks
    }
    return {
        "shape": [int(v) for v in x.shape],
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "energy_at_rank": energy_at_rank,
        "singular_values": singular_values.tolist(),
    }


def block_energy_map(residual: np.ndarray, block_size: int) -> np.ndarray:
    x = np.asarray(residual, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected [H,W,D] residual, got {x.shape}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    height, width, _hidden = x.shape
    block_height = math.ceil(height / block_size)
    block_width = math.ceil(width / block_size)
    energy = np.zeros((block_height, block_width), dtype=np.float64)
    for by in range(block_height):
        for bx in range(block_width):
            patch = x[
                by * block_size : min((by + 1) * block_size, height),
                bx * block_size : min((bx + 1) * block_size, width),
            ]
            energy[by, bx] = float(np.sum(patch * patch))
    return energy


def block_reduce_mask(mask: np.ndarray, block_size: int) -> np.ndarray:
    x = np.asarray(mask, dtype=bool)
    if x.ndim != 2:
        raise ValueError(f"expected [H,W] mask, got {x.shape}")
    height, width = x.shape
    block_height = math.ceil(height / block_size)
    block_width = math.ceil(width / block_size)
    out = np.zeros((block_height, block_width), dtype=bool)
    for by in range(block_height):
        for bx in range(block_width):
            out[by, bx] = bool(
                np.any(
                    x[
                        by * block_size : min((by + 1) * block_size, height),
                        bx * block_size : min((bx + 1) * block_size, width),
                    ]
                )
            )
    return out


def gini_coefficient(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or float(x.sum()) <= 1e-24:
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    index = np.arange(1, x.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * x) / np.sum(x) - (x.size + 1)) / x.size)


def residual_concentration(
    residual: np.ndarray,
    block_size: int,
    fractions: Iterable[float],
    event_mask: np.ndarray | None = None,
) -> dict[str, object]:
    energy_map = block_energy_map(residual, block_size)
    flat = energy_map.reshape(-1)
    order = np.argsort(flat)[::-1]
    total = float(flat.sum())
    event_blocks = (
        block_reduce_mask(event_mask, block_size).reshape(-1)
        if event_mask is not None
        else None
    )
    rows: dict[str, dict[str, float | int]] = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"top fraction must be in (0,1], got {fraction}")
        count = max(1, int(math.ceil(flat.size * fraction)))
        selected = order[:count]
        captured = float(flat[selected].sum() / (total + 1e-24))
        recall = 0.0
        event_count = 0
        if event_blocks is not None:
            event_count = int(event_blocks.sum())
            if event_count:
                recall = float(event_blocks[selected].sum() / event_count)
        rows[f"{fraction:.4f}"] = {
            "selected_blocks": count,
            "energy_ratio": captured,
            "event_recall": recall,
            "event_blocks": event_count,
        }
    return {
        "block_shape": [int(v) for v in energy_map.shape],
        "gini": gini_coefficient(flat),
        "top": rows,
    }
