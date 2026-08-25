from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def horizon_shift(values: np.ndarray, offset: int) -> np.ndarray:
    if offset < 0 or offset >= values.shape[-2]:
        raise ValueError("offset must be in [0, horizon)")
    shifted = np.zeros_like(values)
    if offset == 0:
        shifted[...] = values
    else:
        shifted[..., : values.shape[-2] - offset, :] = values[..., offset:, :]
    return shifted


def overlap_mask(horizon: int, offset: int) -> np.ndarray:
    if offset < 0 or offset >= horizon:
        raise ValueError("offset must be in [0, horizon)")
    mask = np.zeros(horizon, dtype=bool)
    mask[: horizon - offset] = True
    return mask


def reuse_with_exact_tail(
    previous: np.ndarray,
    current: np.ndarray,
    offset: int,
    aligned: bool,
) -> np.ndarray:
    if previous.shape != current.shape:
        raise ValueError("previous and current must have identical shapes")
    mask = overlap_mask(current.shape[-2], offset)
    output = current.copy()
    source = horizon_shift(previous, offset) if aligned else previous
    output[..., mask, :] = source[..., mask, :]
    return output


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / (np.linalg.norm(target) + EPS))


def row_relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction_rows = prediction.reshape(len(prediction), -1)
    target_rows = target.reshape(len(target), -1)
    return np.linalg.norm(prediction_rows - target_rows, axis=1) / (
        np.linalg.norm(target_rows, axis=1) + EPS
    )


def captured_energy(target: np.ndarray, approximation: np.ndarray) -> float:
    denominator = np.linalg.norm(target) ** 2
    residual = np.linalg.norm(target - approximation) ** 2
    return float(1.0 - residual / (denominator + EPS))


def _temporal_design(
    source: np.ndarray,
    radius: int,
    circular: bool,
) -> np.ndarray:
    if source.ndim != 3:
        raise ValueError("source must have shape [samples, horizon, channels]")
    sample_count, horizon, channels = source.shape
    positions = np.arange(horizon)
    parts = []
    for offset in range(-radius, radius + 1):
        indices = positions + offset
        if circular:
            parts.append(source[:, indices % horizon])
        else:
            shifted = np.zeros((sample_count, horizon, channels), dtype=source.dtype)
            valid = (indices >= 0) & (indices < horizon)
            shifted[:, valid] = source[:, indices[valid]]
            parts.append(shifted)
    return np.stack(parts, axis=-1)


class DepthwiseTemporalRegressor:
    def __init__(self, radius: int, circular: bool, alpha: float):
        self.radius = radius
        self.circular = circular
        self.alpha = alpha

    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray,
        valid_positions: np.ndarray,
    ) -> "DepthwiseTemporalRegressor":
        if source.shape != target.shape:
            raise ValueError("source and target must have identical shapes")
        design = _temporal_design(source, self.radius, self.circular)
        x = design[:, valid_positions].reshape(-1, source.shape[-1], design.shape[-1])
        y = target[:, valid_positions].reshape(-1, source.shape[-1])
        coefficients = []
        for channel in range(source.shape[-1]):
            channel_x = np.concatenate(
                [x[:, channel].astype(np.float64), np.ones((len(x), 1))], axis=1
            )
            channel_y = y[:, channel].astype(np.float64)
            gram = channel_x.T @ channel_x
            gram[:-1, :-1] += self.alpha * np.eye(gram.shape[0] - 1)
            coefficients.append(np.linalg.solve(gram, channel_x.T @ channel_y))
        fitted = np.stack(coefficients)
        self.weight = fitted[:, :-1].astype(source.dtype)
        self.bias = fitted[:, -1].astype(source.dtype)
        return self

    def predict(self, source: np.ndarray) -> np.ndarray:
        design = _temporal_design(source, self.radius, self.circular)
        return np.einsum("nhdk,dk->nhd", design, self.weight) + self.bias

    @property
    def parameter_count(self) -> int:
        return int(self.weight.size + self.bias.size)

    def macs_per_sample(self, horizon: int) -> int:
        return int(horizon * self.weight.size)


class RidgeMap:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeMap":
        x = features.astype(np.float64)
        y = targets.astype(np.float64)
        self.x_mean = x.mean(axis=0)
        self.y_mean = y.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.x_scale = scale
        centered_x = (x - self.x_mean) / self.x_scale
        centered_y = y - self.y_mean
        if centered_x.shape[1] <= centered_x.shape[0]:
            gram = centered_x.T @ centered_x
            gram += self.alpha * np.eye(gram.shape[0])
            self.weight = np.linalg.solve(gram, centered_x.T @ centered_y)
        else:
            gram = centered_x @ centered_x.T
            gram += self.alpha * np.eye(gram.shape[0])
            self.weight = centered_x.T @ np.linalg.solve(gram, centered_y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (features.astype(np.float64) - self.x_mean) / self.x_scale
        return (x @ self.weight + self.y_mean).astype(features.dtype)

    @property
    def parameter_count(self) -> int:
        return int(
            self.weight.size
            + self.x_mean.size
            + self.x_scale.size
            + self.y_mean.size
        )

    @property
    def macs_per_sample(self) -> int:
        return int(self.weight.size)


@dataclass
class FrozenBasis:
    mean: np.ndarray
    basis: np.ndarray
    sample_shape: tuple[int, ...]

    @classmethod
    def fit(cls, values: np.ndarray, rank: int) -> "FrozenBasis":
        flat = values.reshape(len(values), -1).astype(np.float64)
        mean = flat.mean(axis=0)
        centered = flat - mean
        gram = centered @ centered.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        rows = []
        for index in order:
            if len(rows) == rank:
                break
            if eigenvalues[index] <= EPS:
                continue
            row = eigenvectors[:, index] @ centered
            row /= np.sqrt(eigenvalues[index])
            rows.append(row)
        if not rows:
            raise ValueError("calibration values have no nonzero principal direction")
        basis = np.stack(rows)
        return cls(
            mean=mean.astype(values.dtype),
            basis=basis.astype(values.dtype),
            sample_shape=values.shape[1:],
        )

    def coefficients(self, values: np.ndarray) -> np.ndarray:
        flat = values.reshape(len(values), -1)
        return (flat - self.mean) @ self.basis.T

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        flat = self.mean + coefficients @ self.basis
        return flat.reshape((len(coefficients),) + self.sample_shape)

    def project(self, values: np.ndarray) -> np.ndarray:
        return self.reconstruct(self.coefficients(values))


def coefficient_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.linalg.norm(prediction - target) ** 2
    centered = target - target.mean(axis=0)
    denominator = np.linalg.norm(centered) ** 2
    return float(1.0 - residual / (denominator + EPS))


def flatten_feature_groups(*groups: np.ndarray) -> np.ndarray:
    if not groups:
        raise ValueError("at least one feature group is required")
    sample_count = len(groups[0])
    if any(len(group) != sample_count for group in groups):
        raise ValueError("all feature groups must have the same sample count")
    return np.concatenate(
        [group.reshape(sample_count, -1) for group in groups], axis=1
    )


def oracle_gap_recovery(baseline: float, candidate: float, oracle: float) -> float:
    denominator = baseline - oracle
    if denominator <= EPS:
        raise ValueError("oracle must improve on the baseline")
    return float((baseline - candidate) / denominator)


def transfer_basis_coefficients(
    coefficients: np.ndarray,
    source: FrozenBasis,
    target: FrozenBasis,
) -> np.ndarray:
    if source.sample_shape != target.sample_shape:
        raise ValueError("source and target bases must describe the same sample shape")
    if coefficients.shape[1] != source.basis.shape[0]:
        raise ValueError("coefficient width does not match the source basis rank")
    rotation = source.basis @ target.basis.T
    offset = (source.mean - target.mean) @ target.basis.T
    return coefficients @ rotation + offset
