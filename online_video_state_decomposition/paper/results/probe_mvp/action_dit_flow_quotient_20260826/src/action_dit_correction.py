from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from structured_models import ReducedRankRegressor, RidgeRegressor


EPS = 1e-12


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / (np.linalg.norm(target) + EPS))


def row_relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = prediction.reshape(len(prediction), -1)
    ref = target.reshape(len(target), -1)
    return np.linalg.norm(pred - ref, axis=1) / np.maximum(
        np.linalg.norm(ref, axis=1), EPS
    )


def bucket_ids(step_indices: np.ndarray, step_count: int, bucket_count: int) -> np.ndarray:
    if step_count <= 0 or bucket_count <= 0:
        raise ValueError("step_count and bucket_count must be positive")
    if np.any(step_indices < 0) or np.any(step_indices >= step_count):
        raise ValueError("step index outside the registered sampling schedule")
    return np.minimum(
        step_indices.astype(np.int64) * bucket_count // step_count,
        bucket_count - 1,
    )


def observed_step_count(step_indices: np.ndarray) -> int:
    unique = np.unique(step_indices.astype(np.int64))
    if len(unique) == 0 or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("observed sampling steps must be contiguous and start at zero")
    return int(len(unique))


def fake_quantize_per_output_channel(weight, bits: int):
    import torch

    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = (1 << (bits - 1)) - 1
    source = weight.detach().float()
    scale = source.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.clamp(torch.round(source / scale), -qmax, qmax)
    return (quantized * scale).to(dtype=weight.dtype)


def quantize_action_dit_ffn(model, bits: int) -> dict[str, int]:
    import torch

    selected = []
    for name, module in model.named_modules():
        is_condition_mlp = name in {"encoder.0", "encoder.2"}
        is_decoder_ffn = name.endswith(".linear1") or name.endswith(".linear2")
        if isinstance(module, torch.nn.Linear) and (is_condition_mlp or is_decoder_ffn):
            module.weight.data.copy_(fake_quantize_per_output_channel(module.weight, bits))
            selected.append((name, module.weight.numel()))
    if not selected:
        raise RuntimeError("no registered action-DiT FFN weights were found")
    return {
        "selected_weight_count": int(sum(count for _, count in selected)),
        "selected_module_count": len(selected),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def _check_shapes(
    noisy_action: np.ndarray,
    quantized_output: np.ndarray,
    condition: np.ndarray,
    defect: np.ndarray,
    buckets: np.ndarray,
) -> None:
    if noisy_action.shape != quantized_output.shape or defect.shape != noisy_action.shape:
        raise ValueError("noisy action, quantized output, and defect must share shape")
    if noisy_action.ndim != 3:
        raise ValueError("action tensors must have shape [sample, horizon, action_dim]")
    if condition.ndim != 2 or len(condition) != len(noisy_action):
        raise ValueError("condition must have shape [sample, condition_dim]")
    if buckets.shape != (len(noisy_action),):
        raise ValueError("buckets must have shape [sample]")


class BucketMeanRegressor:
    def fit(self, noisy_action, quantized_output, condition, defect, buckets):
        _check_shapes(noisy_action, quantized_output, condition, defect, buckets)
        self.means = {}
        for bucket in np.unique(buckets):
            self.means[int(bucket)] = defect[buckets == bucket].mean(axis=0)
        return self

    def predict(self, noisy_action, quantized_output, condition, buckets):
        output = np.empty_like(noisy_action)
        for bucket, mean in self.means.items():
            output[buckets == bucket] = mean
        return output

    @property
    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.means.values()))

    @property
    def macs_per_sample(self) -> int:
        return 0


class BucketChannelAffineRegressor:
    """OHB-like action-channel scale and bias shared over the horizon."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, noisy_action, quantized_output, condition, defect, buckets):
        _check_shapes(noisy_action, quantized_output, condition, defect, buckets)
        full_output = quantized_output + defect
        self.models = {}
        action_dim = quantized_output.shape[-1]
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            bucket_models = []
            for channel in range(action_dim):
                x = quantized_output[mask, :, channel].reshape(-1, 1)
                y = full_output[mask, :, channel].reshape(-1, 1)
                bucket_models.append(RidgeRegressor(self.alpha).fit(x, y))
            self.models[int(bucket)] = bucket_models
        self.horizon = quantized_output.shape[1]
        self.action_dim = action_dim
        return self

    def predict(self, noisy_action, quantized_output, condition, buckets):
        correction = np.empty_like(quantized_output)
        for bucket, models in self.models.items():
            mask = buckets == bucket
            for channel, model in enumerate(models):
                values = quantized_output[mask, :, channel].reshape(-1, 1)
                full = model.predict(values).reshape(mask.sum(), self.horizon)
                correction[mask, :, channel] = (
                    full - quantized_output[mask, :, channel]
                )
        return correction

    @property
    def parameter_count(self) -> int:
        return int(
            sum(
                model.parameter_count
                for models in self.models.values()
                for model in models
            )
        )

    @property
    def macs_per_sample(self) -> int:
        return self.horizon * self.action_dim


def _temporal_design(
    noisy_action: np.ndarray,
    quantized_output: np.ndarray,
    radius: int,
    circular: bool,
) -> np.ndarray:
    source = np.concatenate([noisy_action, quantized_output], axis=2)
    sample_count, horizon, width = source.shape
    parts = []
    positions = np.arange(horizon)
    for offset in range(-radius, radius + 1):
        indices = positions + offset
        if circular:
            parts.append(source[:, indices % horizon])
        else:
            valid = (indices >= 0) & (indices < horizon)
            shifted = np.zeros((sample_count, horizon, width), dtype=source.dtype)
            shifted[:, valid] = source[:, indices[valid]]
            parts.append(shifted)
    return np.concatenate(parts, axis=2)


class TemporalKernelRegressor:
    def __init__(self, radius: int, circular: bool, alpha: float):
        self.radius = radius
        self.circular = circular
        self.alpha = alpha

    def fit(self, noisy_action, quantized_output, condition, defect, buckets):
        _check_shapes(noisy_action, quantized_output, condition, defect, buckets)
        design = _temporal_design(
            noisy_action, quantized_output, self.radius, self.circular
        )
        self.models = {}
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            x = design[mask].reshape(-1, design.shape[-1])
            y = defect[mask].reshape(-1, defect.shape[-1])
            self.models[int(bucket)] = RidgeRegressor(self.alpha).fit(x, y)
        self.horizon = noisy_action.shape[1]
        self.action_dim = noisy_action.shape[2]
        self.design_width = design.shape[2]
        return self

    def predict(self, noisy_action, quantized_output, condition, buckets):
        design = _temporal_design(
            noisy_action, quantized_output, self.radius, self.circular
        )
        output = np.empty_like(noisy_action)
        for bucket, model in self.models.items():
            mask = buckets == bucket
            prediction = model.predict(
                design[mask].reshape(-1, self.design_width)
            )
            output[mask] = prediction.reshape(mask.sum(), self.horizon, self.action_dim)
        return output

    @property
    def parameter_count(self) -> int:
        return int(sum(model.parameter_count for model in self.models.values()))

    @property
    def macs_per_sample(self) -> int:
        return self.horizon * self.design_width * self.action_dim


def _global_features(
    noisy_action: np.ndarray,
    quantized_output: np.ndarray,
    condition: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            noisy_action.reshape(len(noisy_action), -1),
            quantized_output.reshape(len(quantized_output), -1),
            condition,
        ],
        axis=1,
    )


class BucketReducedRankRegressor:
    def __init__(self, rank: int, alpha: float):
        self.rank = rank
        self.alpha = alpha

    def fit(self, noisy_action, quantized_output, condition, defect, buckets):
        _check_shapes(noisy_action, quantized_output, condition, defect, buckets)
        features = _global_features(noisy_action, quantized_output, condition)
        targets = defect.reshape(len(defect), -1)
        self.models = {}
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            self.models[int(bucket)] = ReducedRankRegressor(
                self.rank, self.alpha
            ).fit(features[mask], targets[mask])
        self.horizon = noisy_action.shape[1]
        self.action_dim = noisy_action.shape[2]
        self.feature_width = features.shape[1]
        return self

    def predict(self, noisy_action, quantized_output, condition, buckets):
        features = _global_features(noisy_action, quantized_output, condition)
        output = np.empty_like(noisy_action)
        for bucket, model in self.models.items():
            mask = buckets == bucket
            output[mask] = model.predict(features[mask]).reshape(
                mask.sum(), self.horizon, self.action_dim
            )
        return output

    @property
    def parameter_count(self) -> int:
        return int(sum(model.parameter_count for model in self.models.values()))

    @property
    def macs_per_sample(self) -> int:
        output_width = self.horizon * self.action_dim
        return self.feature_width * self.rank + self.rank * output_width


class BucketDenseRegressor:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, noisy_action, quantized_output, condition, defect, buckets):
        _check_shapes(noisy_action, quantized_output, condition, defect, buckets)
        features = _global_features(noisy_action, quantized_output, condition)
        targets = defect.reshape(len(defect), -1)
        self.models = {}
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            self.models[int(bucket)] = RidgeRegressor(self.alpha).fit(
                features[mask], targets[mask]
            )
        self.horizon = noisy_action.shape[1]
        self.action_dim = noisy_action.shape[2]
        self.feature_width = features.shape[1]
        return self

    def predict(self, noisy_action, quantized_output, condition, buckets):
        features = _global_features(noisy_action, quantized_output, condition)
        output = np.empty_like(noisy_action)
        for bucket, model in self.models.items():
            mask = buckets == bucket
            output[mask] = model.predict(features[mask]).reshape(
                mask.sum(), self.horizon, self.action_dim
            )
        return output

    @property
    def parameter_count(self) -> int:
        return int(sum(model.parameter_count for model in self.models.values()))

    @property
    def macs_per_sample(self) -> int:
        return self.feature_width * self.horizon * self.action_dim


class TemporalLowRankRegressor:
    def __init__(self, radius: int, rank: int, alpha: float):
        self.temporal = TemporalKernelRegressor(radius, False, alpha)
        self.low_rank = BucketReducedRankRegressor(rank, alpha)

    def fit(self, noisy_action, quantized_output, condition, defect, buckets):
        self.temporal.fit(noisy_action, quantized_output, condition, defect, buckets)
        first = self.temporal.predict(noisy_action, quantized_output, condition, buckets)
        self.low_rank.fit(
            noisy_action,
            quantized_output,
            condition,
            defect - first,
            buckets,
        )
        return self

    def predict(self, noisy_action, quantized_output, condition, buckets):
        return self.temporal.predict(
            noisy_action, quantized_output, condition, buckets
        ) + self.low_rank.predict(noisy_action, quantized_output, condition, buckets)

    @property
    def parameter_count(self) -> int:
        return self.temporal.parameter_count + self.low_rank.parameter_count

    @property
    def macs_per_sample(self) -> int:
        return self.temporal.macs_per_sample + self.low_rank.macs_per_sample


@dataclass(frozen=True)
class BasisProjection:
    rank: int
    residual_relative_l2: float
    captured_energy: float


def frozen_basis_projection(
    calibration_defect: np.ndarray,
    calibration_buckets: np.ndarray,
    evaluation_defect: np.ndarray,
    evaluation_buckets: np.ndarray,
    ranks: tuple[int, ...],
) -> list[BasisProjection]:
    flat_calibration = calibration_defect.reshape(len(calibration_defect), -1)
    flat_evaluation = evaluation_defect.reshape(len(evaluation_defect), -1)
    residuals = {rank: np.empty_like(flat_evaluation) for rank in ranks}
    for bucket in np.unique(calibration_buckets):
        cal_mask = calibration_buckets == bucket
        eval_mask = evaluation_buckets == bucket
        mean = flat_calibration[cal_mask].mean(axis=0)
        centered = flat_calibration[cal_mask] - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        eval_centered = flat_evaluation[eval_mask] - mean
        for rank in ranks:
            basis = vt[:rank]
            projection = eval_centered @ basis.T @ basis
            residuals[rank][eval_mask] = eval_centered - projection
    denominator = np.linalg.norm(flat_evaluation)
    output = []
    for rank in ranks:
        residual_norm = np.linalg.norm(residuals[rank])
        output.append(
            BasisProjection(
                rank=rank,
                residual_relative_l2=float(residual_norm / (denominator + EPS)),
                captured_energy=float(1.0 - residual_norm**2 / (denominator**2 + EPS)),
            )
        )
    return output
