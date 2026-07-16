from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .metrics import relative_fro_error


def shift_grid(
    source: np.ndarray,
    dy: int,
    dx: int,
    *,
    cyclic: bool,
) -> np.ndarray:
    x = np.asarray(source)
    if x.ndim != 3:
        raise ValueError(f"expected [H,W,D], got {x.shape}")
    if cyclic:
        return np.roll(x, shift=(dy, dx), axis=(0, 1))

    height, width, _hidden = x.shape
    out = np.zeros_like(x)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = x[
            src_y0:src_y1,
            src_x0:src_x1,
        ]
    return out


def warp_grid_bilinear(
    source: np.ndarray,
    backward_flow: np.ndarray,
) -> np.ndarray:
    x = np.asarray(source, dtype=np.float64)
    flow = np.asarray(backward_flow, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected source [H,W,D], got {x.shape}")
    if flow.shape != (*x.shape[:2], 2):
        raise ValueError(
            f"expected backward flow [H,W,2], got {flow.shape}"
        )
    height, width, _hidden = x.shape
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    sample_x = grid_x + flow[..., 0]
    sample_y = grid_y + flow[..., 1]
    valid = (
        (sample_x >= 0.0)
        & (sample_x <= width - 1)
        & (sample_y >= 0.0)
        & (sample_y <= height - 1)
    )
    floor_x = np.floor(sample_x)
    floor_y = np.floor(sample_y)
    x0 = np.clip(floor_x.astype(np.int64), 0, width - 1)
    y0 = np.clip(floor_y.astype(np.int64), 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    weight_x = sample_x - floor_x
    weight_y = sample_y - floor_y
    top = (
        (1.0 - weight_x)[..., None] * x[y0, x0]
        + weight_x[..., None] * x[y0, x1]
    )
    bottom = (
        (1.0 - weight_x)[..., None] * x[y1, x0]
        + weight_x[..., None] * x[y1, x1]
    )
    prediction = (
        (1.0 - weight_y)[..., None] * top
        + weight_y[..., None] * bottom
    )
    prediction[~valid] = 0.0
    return prediction


def local_offsets(radius: int) -> list[tuple[int, int]]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]


def best_integer_shift(
    source: np.ndarray,
    target: np.ndarray,
    max_shift: int,
    *,
    cyclic: bool = True,
) -> tuple[int, int, float]:
    best = (0, 0, float("inf"))
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            prediction = shift_grid(source, dy, dx, cyclic=cyclic)
            error = relative_fro_error(target, prediction)
            if error < best[2]:
                best = (dy, dx, error)
    return best


def fit_shift_basis(
    source: np.ndarray,
    target: np.ndarray,
    offsets: Iterable[tuple[int, int]],
    *,
    cyclic: bool,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    offset_list = list(offsets)
    if not offset_list:
        raise ValueError("offsets must not be empty")
    basis = np.stack(
        [
            shift_grid(source, dy, dx, cyclic=cyclic).reshape(-1)
            for dy, dx in offset_list
        ],
        axis=1,
    ).astype(np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    gram = basis.T @ basis
    gram.flat[:: gram.shape[0] + 1] += ridge
    weights = np.linalg.solve(gram, basis.T @ y)
    prediction = (basis @ weights).reshape(target.shape)
    return prediction, weights


def apply_shift_basis(
    source: np.ndarray,
    offsets: Iterable[tuple[int, int]],
    weights: np.ndarray,
    *,
    cyclic: bool,
) -> np.ndarray:
    offset_list = list(offsets)
    coefficient = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(offset_list) != coefficient.size:
        raise ValueError(
            f"offset and weight counts differ: {len(offset_list)} vs {coefficient.size}"
        )
    prediction = np.zeros_like(source, dtype=np.float64)
    for (dy, dx), weight in zip(offset_list, coefficient, strict=True):
        prediction += weight * shift_grid(source, dy, dx, cyclic=cyclic)
    return prediction


def fit_global_bccb(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(f"source and target must share [H,W,D], got {x.shape} and {y.shape}")
    xf = np.fft.fft2(x, axes=(0, 1))
    yf = np.fft.fft2(y, axes=(0, 1))
    numerator = np.sum(np.conj(xf) * yf, axis=2)
    denominator = np.sum(np.abs(xf) ** 2, axis=2) + ridge
    kernel_frequency = numerator / denominator
    prediction_frequency = kernel_frequency[:, :, None] * xf
    prediction = np.fft.ifft2(prediction_frequency, axes=(0, 1)).real
    kernel = np.fft.ifft2(kernel_frequency).real
    return prediction, kernel


def apply_global_bccb(source: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    x = np.asarray(source, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if x.ndim != 3 or k.shape != x.shape[:2]:
        raise ValueError(
            f"expected source [H,W,D] and kernel [H,W], got {x.shape} and {k.shape}"
        )
    prediction_frequency = np.fft.fft2(k)[:, :, None] * np.fft.fft2(
        x,
        axes=(0, 1),
    )
    return np.fft.ifft2(prediction_frequency, axes=(0, 1)).real


def fit_low_rank_token_map(
    source: np.ndarray,
    target: np.ndarray,
    *,
    rank: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source, dtype=np.float64).reshape(-1, source.shape[-1])
    y = np.asarray(target, dtype=np.float64).reshape(-1, target.shape[-1])
    if x.shape != y.shape:
        raise ValueError(f"source and target token matrices differ: {x.shape} vs {y.shape}")
    token_count = x.shape[0]
    covariance = x @ x.T
    covariance.flat[:: token_count + 1] += ridge
    dense_map = (y @ x.T) @ np.linalg.inv(covariance)
    u, singular_values, vh = np.linalg.svd(dense_map, full_matrices=False)
    kept = min(rank, singular_values.size)
    low_rank_map = (u[:, :kept] * singular_values[:kept]) @ vh[:kept]
    prediction = (low_rank_map @ x).reshape(target.shape)
    return prediction, low_rank_map
