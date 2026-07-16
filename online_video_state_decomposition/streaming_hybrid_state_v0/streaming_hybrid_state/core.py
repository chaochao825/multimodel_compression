from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PREDICTOR_SPECS: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("previous", {}),
    ("ema_025", {"alpha": 0.25}),
    ("ema_050", {"alpha": 0.50}),
    ("ema_075", {"alpha": 0.75}),
    ("linear", {}),
    ("fourier_h4_k1", {"history": 4, "harmonics": 1}),
    ("fourier_h8_k2", {"history": 8, "harmonics": 2}),
)

PQ_SPECS: tuple[tuple[str, int, int], ...] = (
    ("pq_0p5b_k16_d8", 4, 8),
    ("pq_1b_k16_d4", 4, 4),
    ("pq_1p5b_k64_d4", 6, 4),
    ("pq_2b_k256_d4", 8, 4),
    ("pq_4b_k256_d2", 8, 2),
)


@dataclass(frozen=True)
class ClipSequence:
    run: str
    category: str
    layer: int
    values: np.ndarray
    frames_rgb: np.ndarray


@dataclass(frozen=True)
class Codebook:
    values: np.ndarray
    index_bits: int
    group_dim: int

    @property
    def k(self) -> int:
        return int(self.values.shape[0])

    @property
    def static_bits(self) -> int:
        return int(self.values.size * 16)


@dataclass(frozen=True)
class QuantizedPayload:
    reconstruction: np.ndarray
    payload_bits: int
    metadata_bits: int
    indices: np.ndarray | None = None
    selected_fraction: float = 1.0


def normalize_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(
        x,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(x),
    )


def pool_grid(
    frame: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"frame must be [H,W,D], got {x.shape}")
    height, width, hidden_dim = x.shape
    if not 1 <= rows <= height or not 1 <= cols <= width:
        raise ValueError(f"invalid pool grid {rows}x{cols} for {x.shape}")
    pooled: list[np.ndarray] = []
    for row_ids in np.array_split(np.arange(height), rows):
        for col_ids in np.array_split(np.arange(width), cols):
            region = x[np.ix_(row_ids, col_ids)]
            pooled.append(region.reshape(-1, hidden_dim).mean(axis=0))
    return np.stack(pooled)


def prepare_hidden_sequence(
    sequence: np.ndarray,
    *,
    pool_rows: int,
    pool_cols: int,
    center: bool = True,
) -> np.ndarray:
    x = np.asarray(sequence)
    if x.ndim != 4:
        raise ValueError(f"sequence must be [T,H,W,D], got {x.shape}")
    frames = []
    for frame in x:
        pooled = pool_grid(frame, rows=pool_rows, cols=pool_cols)
        if center:
            pooled = pooled - pooled.mean(axis=0, keepdims=True)
        frames.append(normalize_rows(pooled))
    return np.stack(frames).astype(np.float32)


def discover_clips(
    root: Path,
    *,
    layers: Sequence[int],
    pool_rows: int,
    pool_cols: int,
) -> list[ClipSequence]:
    clips: list[ClipSequence] = []
    for npz_path in sorted(root.glob("*/hidden.npz")):
        metadata_path = npz_path.with_name("metadata.json")
        category = "unlabeled"
        if metadata_path.exists():
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            category = str(metadata.get("category") or category)
        with np.load(npz_path) as archive:
            frames_rgb = np.asarray(archive["frames_rgb"], dtype=np.uint8)
            for layer in layers:
                key = f"hidden_layer_{layer}"
                if key not in archive.files:
                    continue
                clips.append(
                    ClipSequence(
                        run=npz_path.parent.name,
                        category=category,
                        layer=int(layer),
                        values=prepare_hidden_sequence(
                            archive[key],
                            pool_rows=pool_rows,
                            pool_cols=pool_cols,
                        ),
                        frames_rgb=frames_rgb.copy(),
                    )
                )
    if not clips:
        raise FileNotFoundError(f"no hidden.npz inputs found under {root}")
    return clips


def split_runs(
    clips: Sequence[ClipSequence],
) -> dict[str, set[str]]:
    by_category: dict[str, list[str]] = {}
    for clip in clips:
        by_category.setdefault(clip.category, [])
        if clip.run not in by_category[clip.category]:
            by_category[clip.category].append(clip.run)
    output = {"train": set(), "val": set(), "test": set()}
    for category, runs in sorted(by_category.items()):
        ordered = sorted(runs)
        if len(ordered) < 4:
            raise ValueError(
                f"category {category!r} needs at least four runs, got {len(ordered)}"
            )
        train_end = len(ordered) - 3
        val_end = len(ordered) - 2
        output["train"].update(ordered[:train_end])
        output["val"].update(ordered[train_end:val_end])
        output["test"].update(ordered[val_end:])
    if output["train"] & output["val"]:
        raise AssertionError("train/val split overlap")
    if output["train"] & output["test"]:
        raise AssertionError("train/test split overlap")
    if output["val"] & output["test"]:
        raise AssertionError("val/test split overlap")
    return output


def predictor_name_and_params(
    name: str,
) -> tuple[str, dict[str, float | int]]:
    lookup = dict(PREDICTOR_SPECS)
    if name not in lookup:
        raise ValueError(f"unknown predictor: {name}")
    if name.startswith("ema_"):
        return "ema", dict(lookup[name])
    if name.startswith("fourier_"):
        return "fourier", dict(lookup[name])
    return name, dict(lookup[name])


def _fourier_design(
    length: int,
    harmonics: int,
    *,
    sample_at: float | None = None,
) -> np.ndarray:
    if length < 2:
        raise ValueError("Fourier history must contain at least two frames")
    if harmonics < 1:
        raise ValueError("harmonics must be positive")
    time = (
        np.arange(length, dtype=np.float64)
        if sample_at is None
        else np.asarray([sample_at], dtype=np.float64)
    )
    center = (length - 1) / 2.0
    columns = [
        np.ones_like(time),
        (time - center) / max(float(length), 1.0),
    ]
    for harmonic in range(1, harmonics + 1):
        phase = 2.0 * math.pi * harmonic * time / float(length)
        columns.extend((np.cos(phase), np.sin(phase)))
    return np.stack(columns, axis=1)


def predict_next(
    history: Sequence[np.ndarray],
    method: str,
    params: dict[str, float | int] | None = None,
) -> np.ndarray:
    if not history:
        raise ValueError("history cannot be empty")
    cfg = params or {}
    if method == "previous":
        return np.asarray(history[-1], dtype=np.float32).copy()
    if method == "linear":
        if len(history) < 2:
            return np.asarray(history[-1], dtype=np.float32).copy()
        return (
            2.0 * np.asarray(history[-1], dtype=np.float32)
            - np.asarray(history[-2], dtype=np.float32)
        )
    if method == "ema":
        alpha = float(cfg.get("alpha", 0.5))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0,1]")
        state = np.asarray(history[0], dtype=np.float32).copy()
        for frame in history[1:]:
            state = alpha * np.asarray(frame, dtype=np.float32) + (
                1.0 - alpha
            ) * state
        return state
    if method == "fourier":
        requested = int(cfg.get("history", 8))
        harmonics = int(cfg.get("harmonics", 2))
        selected = list(history[-requested:])
        if len(selected) < max(4, 2 * harmonics + 2):
            return predict_next(history, "linear")
        length = len(selected)
        design = _fourier_design(length, harmonics)
        next_design = _fourier_design(
            length,
            harmonics,
            sample_at=float(length),
        )
        flat = np.stack(selected).reshape(length, -1).astype(np.float64)
        ridge = 1e-3 * np.eye(design.shape[1], dtype=np.float64)
        ridge[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + ridge,
            design.T @ flat,
        )
        prediction = (next_design @ coefficients).reshape(selected[-1].shape)
        return prediction.astype(np.float32)
    raise ValueError(f"unsupported predictor: {method}")


def predictor_cost_per_scalar(
    method: str,
    params: dict[str, float | int] | None = None,
) -> float:
    cfg = params or {}
    if method == "previous":
        return 0.0
    if method == "ema":
        return 2.0
    if method == "linear":
        return 2.0
    if method == "fourier":
        length = int(cfg.get("history", 8))
        basis = 2 + 2 * int(cfg.get("harmonics", 2))
        return float(2 * length * basis)
    raise ValueError(f"unsupported predictor: {method}")


def reconstruction_metrics(
    target: np.ndarray,
    reconstruction: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(target, dtype=np.float64)
    y = np.asarray(reconstruction, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    error = x - y
    denominator = max(float(np.mean(x * x)), 1e-12)
    target_flat = x.reshape(-1, x.shape[-1])
    recon_flat = y.reshape(-1, y.shape[-1])
    target_norm = np.linalg.norm(target_flat, axis=1)
    recon_norm = np.linalg.norm(recon_flat, axis=1)
    cosine = np.sum(target_flat * recon_flat, axis=1) / np.maximum(
        target_norm * recon_norm,
        1e-12,
    )
    return {
        "mse": float(np.mean(error * error)),
        "nmse": float(np.mean(error * error) / denominator),
        "mean_cosine": float(np.mean(cosine)),
        "p05_cosine": float(np.quantile(cosine, 0.05)),
    }


def residual_concentration(
    residual: np.ndarray,
    fraction: float = 0.10,
) -> float:
    x = np.asarray(residual, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("residual must contain token vectors")
    energy = np.sum(x * x, axis=-1).reshape(-1)
    if not energy.size or float(energy.sum()) <= 0.0:
        return 1.0
    keep = max(1, int(math.ceil(fraction * energy.size)))
    selected = np.partition(energy, energy.size - keep)[-keep:]
    return float(selected.sum() / energy.sum())


def empirical_entropy(indices: np.ndarray, k: int) -> float:
    x = np.asarray(indices, dtype=np.int64).reshape(-1)
    if not x.size:
        return 0.0
    counts = np.bincount(x, minlength=k).astype(np.float64)
    probabilities = counts[counts > 0.0] / float(x.size)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def scalar_quantize(values: np.ndarray, bits: int) -> QuantizedPayload:
    if bits < 2:
        raise ValueError("scalar quantization requires at least two bits")
    x = np.asarray(values, dtype=np.float32)
    qmax = (1 << (bits - 1)) - 1
    scales = np.max(np.abs(x), axis=-1, keepdims=True) / max(qmax, 1)
    scales = np.maximum(scales, 1e-8)
    scales = scales.astype(np.float16).astype(np.float32)
    quantized = np.rint(x / scales).clip(-qmax, qmax)
    reconstruction = (quantized * scales).astype(np.float32)
    vector_count = int(np.prod(x.shape[:-1]))
    return QuantizedPayload(
        reconstruction=reconstruction,
        payload_bits=int(x.size * bits),
        metadata_bits=vector_count * 16,
    )


def _sample_rows(
    samples: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x
    selected = np.sort(
        rng.choice(x.shape[0], size=max_samples, replace=False)
    )
    return x[selected]


def assign_codebook(
    samples: np.ndarray,
    codebook: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=np.float32)
    c = np.asarray(codebook, dtype=np.float32)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError("samples/codebook must be compatible rank-2 arrays")
    indices = np.empty(x.shape[0], dtype=np.int32)
    distances = np.empty(x.shape[0], dtype=np.float32)
    code_norm = np.sum(c * c, axis=1)
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        batch = x[start:stop]
        squared = (
            np.sum(batch * batch, axis=1, keepdims=True)
            + code_norm[None]
            - 2.0 * (batch @ c.T)
        )
        batch_indices = np.argmin(squared, axis=1)
        indices[start:stop] = batch_indices
        distances[start:stop] = squared[
            np.arange(stop - start), batch_indices
        ]
    return indices, np.maximum(distances, 0.0)


def fit_codebook(
    samples: np.ndarray,
    *,
    index_bits: int,
    group_dim: int,
    seed: int,
    max_samples: int = 12_000,
    iterations: int = 8,
) -> Codebook:
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != group_dim:
        raise ValueError(
            f"samples must be [N,{group_dim}], got {x.shape}"
        )
    if index_bits <= 0:
        raise ValueError("index_bits must be positive")
    k = 1 << index_bits
    rng = np.random.default_rng(seed)
    train = _sample_rows(x, max_samples=max_samples, rng=rng)
    if train.shape[0] < k:
        repeats = int(math.ceil(k / max(train.shape[0], 1)))
        train = np.tile(train, (repeats, 1))
    selected = rng.choice(train.shape[0], size=k, replace=False)
    centroids = train[selected].copy()
    for _ in range(max(1, iterations)):
        indices, distances = assign_codebook(train, centroids)
        counts = np.bincount(indices, minlength=k)
        updated = np.zeros_like(centroids)
        for column in range(group_dim):
            updated[:, column] = np.bincount(
                indices,
                weights=train[:, column],
                minlength=k,
            )
        alive = counts > 0
        updated[alive] /= counts[alive, None]
        dead = np.flatnonzero(~alive)
        if dead.size:
            hardest = np.argsort(-distances, kind="stable")
            updated[dead] = train[hardest[: dead.size]]
        movement = float(np.mean((updated - centroids) ** 2))
        centroids = updated
        if movement < 1e-8:
            break
    return Codebook(
        values=centroids.astype(np.float16).astype(np.float32),
        index_bits=index_bits,
        group_dim=group_dim,
    )


def split_groups(values: np.ndarray, group_dim: int) -> tuple[np.ndarray, int]:
    x = np.asarray(values, dtype=np.float32)
    hidden_dim = x.shape[-1]
    padded_dim = int(math.ceil(hidden_dim / group_dim) * group_dim)
    if padded_dim != hidden_dim:
        padding = [(0, 0)] * x.ndim
        padding[-1] = (0, padded_dim - hidden_dim)
        x = np.pad(x, padding)
    return x.reshape(-1, group_dim), hidden_dim


def join_groups(
    groups: np.ndarray,
    *,
    output_shape: Sequence[int],
    hidden_dim: int,
) -> np.ndarray:
    padded_dim = int(math.ceil(hidden_dim / groups.shape[-1]) * groups.shape[-1])
    reconstructed = groups.reshape(*output_shape[:-1], padded_dim)
    return reconstructed[..., :hidden_dim].astype(np.float32)


def collect_group_samples(
    sequences: Iterable[np.ndarray],
    *,
    group_dim: int,
) -> np.ndarray:
    groups = []
    for sequence in sequences:
        split, _ = split_groups(sequence, group_dim)
        groups.append(split)
    if not groups:
        raise ValueError("cannot collect groups from zero sequences")
    return np.concatenate(groups, axis=0)


def pq_quantize(
    values: np.ndarray,
    codebook: Codebook,
    *,
    selected_fraction: float = 1.0,
) -> QuantizedPayload:
    if not 0.0 < selected_fraction <= 1.0:
        raise ValueError("selected_fraction must be in (0,1]")
    x = np.asarray(values, dtype=np.float32)
    groups, hidden_dim = split_groups(x, codebook.group_dim)
    indices, _ = assign_codebook(groups, codebook.values)
    reconstructed_groups = codebook.values[indices].copy()
    group_count = groups.shape[0]
    metadata_bits = 0
    if selected_fraction < 1.0:
        groups_per_vector = int(
            math.ceil(hidden_dim / codebook.group_dim)
        )
        reshaped = groups.reshape(-1, groups_per_vector, codebook.group_dim)
        energies = np.sum(reshaped * reshaped, axis=-1)
        keep = max(1, int(math.ceil(selected_fraction * groups_per_vector)))
        selected = np.zeros_like(energies, dtype=bool)
        top = np.argpartition(
            energies,
            groups_per_vector - keep,
            axis=1,
        )[:, -keep:]
        selected[
            np.arange(selected.shape[0])[:, None],
            top,
        ] = True
        selected_flat = selected.reshape(-1)
        reconstructed_groups[~selected_flat] = 0.0
        payload_bits = int(selected_flat.sum()) * codebook.index_bits
        metadata_bits = int(selected_flat.size)
        selected_fraction = float(selected_flat.mean())
        reported_indices = indices[selected_flat]
    else:
        payload_bits = group_count * codebook.index_bits
        reported_indices = indices
    reconstruction = join_groups(
        reconstructed_groups,
        output_shape=x.shape,
        hidden_dim=hidden_dim,
    )
    return QuantizedPayload(
        reconstruction=reconstruction,
        payload_bits=payload_bits,
        metadata_bits=metadata_bits,
        indices=reported_indices,
        selected_fraction=selected_fraction,
    )


def predictor_residuals(
    sequence: np.ndarray,
    *,
    predictor_name: str,
) -> np.ndarray:
    method, params = predictor_name_and_params(predictor_name)
    x = np.asarray(sequence, dtype=np.float32)
    residuals = []
    history: list[np.ndarray] = [x[0]]
    for frame in x[1:]:
        prediction = predict_next(history, method, params)
        residuals.append(frame - prediction)
        history.append(frame)
    return np.stack(residuals)


def run_residual_codec(
    sequence: np.ndarray,
    *,
    predictor_name: str,
    codebook: Codebook,
    selected_fraction: float,
    full_refresh_bits: int = 4,
    actions: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"sequence must be [T,N,D], got {x.shape}")
    method, params = predictor_name_and_params(predictor_name)
    first = scalar_quantize(x[0], full_refresh_bits)
    decoded = [first.reconstruction]
    payload_bits = first.payload_bits
    metadata_bits = first.metadata_bits
    all_indices = []
    action_counts = np.zeros(4, dtype=np.int64)
    action_counts[3] += 1
    for index in range(1, x.shape[0]):
        prediction = predict_next(decoded, method, params)
        action = 2 if actions is None else int(actions[index])
        if action == 0:
            current = decoded[-1].copy()
        elif action == 1:
            current = prediction
        elif action == 2:
            residual = x[index] - prediction
            quantized = pq_quantize(
                residual,
                codebook,
                selected_fraction=selected_fraction,
            )
            current = prediction + quantized.reconstruction
            payload_bits += quantized.payload_bits
            metadata_bits += quantized.metadata_bits
            if quantized.indices is not None:
                all_indices.append(quantized.indices)
        elif action == 3:
            quantized = scalar_quantize(x[index], full_refresh_bits)
            current = quantized.reconstruction
            payload_bits += quantized.payload_bits
            metadata_bits += quantized.metadata_bits
        else:
            raise ValueError(f"invalid action {action}")
        action_counts[action] += 1
        decoded.append(np.asarray(current, dtype=np.float32))
    reconstructed = np.stack(decoded)
    scalar_count = int(x.size)
    indices = (
        np.concatenate(all_indices)
        if all_indices
        else np.empty(0, dtype=np.int32)
    )
    accounting = {
        "payload_bits": float(payload_bits),
        "metadata_bits": float(metadata_bits),
        "payload_bps": float((payload_bits + metadata_bits) / scalar_count),
        "index_entropy": empirical_entropy(indices, codebook.k),
        "reuse_rate": float(action_counts[0] / x.shape[0]),
        "predict_rate": float(action_counts[1] / x.shape[0]),
        "innovation_rate": float(action_counts[2] / x.shape[0]),
        "refresh_rate": float(action_counts[3] / x.shape[0]),
    }
    return reconstructed, accounting


def rgb_summary_features(frames_rgb: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames_rgb, dtype=np.float32) / 255.0
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames_rgb must be [T,H,W,3], got {frames.shape}")
    gray = (
        0.299 * frames[..., 0]
        + 0.587 * frames[..., 1]
        + 0.114 * frames[..., 2]
    )
    gray = gray[:, ::8, ::8]
    features = np.zeros((frames.shape[0], 7), dtype=np.float32)
    for index in range(1, frames.shape[0]):
        diff = gray[index] - gray[index - 1]
        spectrum = np.abs(np.fft.rfft2(diff)) ** 2
        height, width = spectrum.shape
        low = spectrum[: max(1, height // 4), : max(1, width // 4)].sum()
        total = max(float(spectrum.sum()), 1e-12)
        hist_a, _ = np.histogram(gray[index - 1], bins=16, range=(0, 1))
        hist_b, _ = np.histogram(gray[index], bins=16, range=(0, 1))
        hist_a = hist_a / max(hist_a.sum(), 1)
        hist_b = hist_b / max(hist_b.sum(), 1)
        grad_y = np.diff(diff, axis=0)
        grad_x = np.diff(diff, axis=1)
        features[index] = (
            float(np.mean(np.abs(diff))),
            float(np.mean(diff * diff)),
            float(np.max(np.abs(diff))),
            float(low / total),
            float(1.0 - low / total),
            float(np.abs(hist_a - hist_b).sum()),
            float(
                0.5
                * (
                    np.mean(np.abs(grad_y))
                    + np.mean(np.abs(grad_x))
                )
            ),
        )
    return features


def frame_cosine(target: np.ndarray, reconstruction: np.ndarray) -> float:
    metrics = reconstruction_metrics(target, reconstruction)
    return float(metrics["mean_cosine"])


class DecisionTree:
    def __init__(
        self,
        *,
        feature: int | None = None,
        threshold: float = 0.0,
        left: DecisionTree | None = None,
        right: DecisionTree | None = None,
        label: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = float(threshold)
        self.left = left
        self.right = right
        self.label = label

    @property
    def is_leaf(self) -> bool:
        return self.label is not None

    def predict_one(self, row: np.ndarray) -> int:
        if self.is_leaf:
            assert self.label is not None
            return int(self.label)
        assert self.feature is not None
        child = self.left if row[self.feature] <= self.threshold else self.right
        assert child is not None
        return child.predict_one(row)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.predict_one(row) for row in np.asarray(features)],
            dtype=np.int64,
        )

    def node_count(self) -> int:
        if self.is_leaf:
            return 1
        assert self.left is not None and self.right is not None
        return 1 + self.left.node_count() + self.right.node_count()

    def depth(self) -> int:
        if self.is_leaf:
            return 0
        assert self.left is not None and self.right is not None
        return 1 + max(self.left.depth(), self.right.depth())


def _gini(labels: np.ndarray, class_count: int) -> float:
    if not labels.size:
        return 0.0
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    probabilities = counts / float(labels.size)
    return float(1.0 - np.sum(probabilities * probabilities))


def fit_decision_tree(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_depth: int = 3,
    min_samples: int = 8,
    class_count: int = 4,
) -> DecisionTree:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape[0] != y.shape[0] or x.ndim != 2:
        raise ValueError("invalid decision-tree training arrays")

    def build(indices: np.ndarray, depth: int) -> DecisionTree:
        local_y = y[indices]
        majority = int(
            np.argmax(np.bincount(local_y, minlength=class_count))
        )
        if (
            depth >= max_depth
            or indices.size < 2 * min_samples
            or np.all(local_y == local_y[0])
        ):
            return DecisionTree(label=majority)
        parent_impurity = _gini(local_y, class_count)
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature in range(x.shape[1]):
            values = x[indices, feature]
            quantiles = np.unique(
                np.quantile(values, np.linspace(0.1, 0.9, 9))
            )
            for threshold in quantiles:
                left_mask = values <= threshold
                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]
                if (
                    left_indices.size < min_samples
                    or right_indices.size < min_samples
                ):
                    continue
                weighted = (
                    left_indices.size * _gini(y[left_indices], class_count)
                    + right_indices.size * _gini(y[right_indices], class_count)
                ) / indices.size
                gain = parent_impurity - weighted
                if best is None or gain > best[0]:
                    best = (
                        gain,
                        feature,
                        float(threshold),
                        left_indices,
                        right_indices,
                    )
        if best is None or best[0] <= 1e-9:
            return DecisionTree(label=majority)
        _, feature, threshold, left_indices, right_indices = best
        return DecisionTree(
            feature=feature,
            threshold=float(np.float16(threshold)),
            left=build(left_indices, depth + 1),
            right=build(right_indices, depth + 1),
        )

    return build(np.arange(x.shape[0]), 0)


def fit_monotonic_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    candidates = np.unique(np.quantile(x, np.linspace(0.05, 0.95, 19)))
    if candidates.size < 3:
        candidates = np.linspace(float(x.min()), float(x.max()), 4)[1:]
    best_accuracy = -1.0
    best_thresholds = np.asarray(candidates[:3], dtype=np.float64)
    best_mapping = np.arange(4, dtype=np.int64)
    for left_index in range(candidates.size):
        for middle_index in range(left_index + 1, candidates.size):
            for right_index in range(middle_index + 1, candidates.size):
                thresholds = np.asarray(
                    [
                        candidates[left_index],
                        candidates[middle_index],
                        candidates[right_index],
                    ]
                )
                bins = np.digitize(x, thresholds)
                mapping = np.zeros(4, dtype=np.int64)
                for bin_id in range(4):
                    local = y[bins == bin_id]
                    mapping[bin_id] = (
                        int(np.argmax(np.bincount(local, minlength=4)))
                        if local.size
                        else bin_id
                    )
                predictions = mapping[bins]
                accuracy = float(np.mean(predictions == y))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_thresholds = thresholds
                    best_mapping = mapping
    return best_thresholds, best_mapping


def predict_monotonic_thresholds(
    scores: np.ndarray,
    thresholds: np.ndarray,
    mapping: np.ndarray,
) -> np.ndarray:
    bins = np.digitize(np.asarray(scores).reshape(-1), thresholds)
    return np.asarray(mapping, dtype=np.int64)[bins]


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    class_count: int = 4,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    recalls = []
    for class_id in range(class_count):
        mask = y == class_id
        if np.any(mask):
            recalls.append(float(np.mean(p[mask] == class_id)))
    return {
        "accuracy": float(np.mean(p == y)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
    }
