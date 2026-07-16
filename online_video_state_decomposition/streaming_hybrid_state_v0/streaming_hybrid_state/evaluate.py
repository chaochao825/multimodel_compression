from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from streaming_hybrid_state.core import (
    PQ_SPECS,
    PREDICTOR_SPECS,
    ClipSequence,
    Codebook,
    DecisionTree,
    classification_metrics,
    collect_group_samples,
    discover_clips,
    empirical_entropy,
    fit_codebook,
    fit_decision_tree,
    fit_monotonic_thresholds,
    frame_cosine,
    pq_quantize,
    predict_monotonic_thresholds,
    predict_next,
    predictor_cost_per_scalar,
    predictor_name_and_params,
    predictor_residuals,
    reconstruction_metrics,
    residual_concentration,
    rgb_summary_features,
    run_residual_codec,
    scalar_quantize,
    split_groups,
    split_runs,
)


EVENT_FRACTIONS = (0.25, 0.50, 1.00)
SCALAR_BITS = (2, 4, 8)
CONTROLLER_BUDGETS = (0.50, 1.00, 1.58, 2.00, 4.00)


@dataclass(frozen=True)
class ControllerExamples:
    features: np.ndarray
    labels: np.ndarray
    runs: tuple[str, ...]
    frames: np.ndarray


class Controller:
    name: str
    static_bits: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ThresholdController(Controller):
    def __init__(self, thresholds: np.ndarray, mapping: np.ndarray) -> None:
        self.name = "threshold"
        self.thresholds = (
            np.asarray(thresholds, dtype=np.float16).astype(np.float64)
        )
        self.mapping = np.asarray(mapping, dtype=np.int64)
        self.static_bits = int(self.thresholds.size * 16 + self.mapping.size * 2)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return predict_monotonic_thresholds(
            np.asarray(features)[:, 0],
            self.thresholds,
            self.mapping,
        )


class TreeController(Controller):
    def __init__(self, tree: DecisionTree) -> None:
        self.name = "decision_tree"
        self.tree = tree
        internal = max(0, (tree.node_count() - 1) // 2)
        leaves = internal + 1
        topology_bits = tree.node_count()
        self.static_bits = int(
            internal * (3 + 16) + leaves * 2 + topology_bits
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.tree.predict(features)


class TorchController(Controller):
    def __init__(
        self,
        *,
        name: str,
        predict_fn: Any,
        static_bits: int,
        extra: dict[str, Any],
    ) -> None:
        self.name = name
        self._predict_fn = predict_fn
        self.static_bits = int(static_bits)
        self.extra = dict(extra)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict_fn(features), dtype=np.int64)


def parse_int_list(value: str) -> list[int]:
    output = sorted({int(item) for item in value.split(",") if item.strip()})
    if not output:
        raise ValueError("integer list cannot be empty")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate causal predictors, multi-bit VQ, discrete controllers, "
            "and their combination on a shared hidden-state corpus."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="15,22")
    parser.add_argument("--pool-rows", type=int, default=4)
    parser.add_argument("--pool-cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--quality-cosine", type=float, default=0.95)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--no-torch", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "maximum clips per category for a smoke run; 0 keeps all clips "
            "and any positive value must be at least 4"
        ),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def codebook_hash(codebook: Codebook) -> str:
    digest = hashlib.sha256()
    digest.update(codebook.values.astype("<f4", copy=False).tobytes())
    digest.update(str(codebook.index_bits).encode("ascii"))
    digest.update(str(codebook.group_dim).encode("ascii"))
    return digest.hexdigest()


def selected_clips(
    clips: Sequence[ClipSequence],
    *,
    layer: int,
    runs: set[str],
) -> list[ClipSequence]:
    return [
        clip for clip in clips if clip.layer == layer and clip.run in runs
    ]


def concatenate_targets(
    clips: Sequence[ClipSequence],
    reconstructions: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.concatenate([clip.values for clip in clips], axis=0)
    recon = np.concatenate(list(reconstructions), axis=0)
    return targets, recon


def temporal_spectral_entropy(residuals: Sequence[np.ndarray]) -> float:
    entropies = []
    for residual in residuals:
        x = np.asarray(residual, dtype=np.float64)
        if x.shape[0] < 2:
            continue
        spectrum = np.abs(np.fft.rfft(x, axis=0)) ** 2
        probabilities = spectrum / np.maximum(
            spectrum.sum(axis=0, keepdims=True),
            1e-12,
        )
        entropy = -np.sum(
            probabilities * np.log2(np.maximum(probabilities, 1e-12)),
            axis=0,
        )
        normalizer = math.log2(max(spectrum.shape[0], 2))
        entropies.append(float(np.mean(entropy / normalizer)))
    return float(np.mean(entropies)) if entropies else 0.0


def evaluate_predictors(
    clips: Sequence[ClipSequence],
    splits: dict[str, set[str]],
    *,
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows: list[dict[str, Any]] = []
    best: dict[int, str] = {}
    for layer in layers:
        for predictor_name, _ in PREDICTOR_SPECS:
            method, params = predictor_name_and_params(predictor_name)
            for split_name in ("val", "test"):
                local_clips = selected_clips(
                    clips,
                    layer=layer,
                    runs=splits[split_name],
                )
                targets = []
                predictions = []
                residual_sequences = []
                raw_sequences = []
                evaluated_frames = 0
                for clip in local_clips:
                    start = min(8, clip.values.shape[0] - 1)
                    clip_residuals = []
                    raw_sequences.append(clip.values[start:])
                    for frame_index in range(start, clip.values.shape[0]):
                        prediction = predict_next(
                            list(clip.values[:frame_index]),
                            method,
                            params,
                        )
                        target = clip.values[frame_index]
                        targets.append(target)
                        predictions.append(prediction)
                        clip_residuals.append(target - prediction)
                        evaluated_frames += 1
                    if clip_residuals:
                        residual_sequences.append(np.stack(clip_residuals))
                target_array = np.stack(targets)
                prediction_array = np.stack(predictions)
                metrics = reconstruction_metrics(
                    target_array,
                    prediction_array,
                )
                raw_entropy = temporal_spectral_entropy(raw_sequences)
                residual_entropy = temporal_spectral_entropy(
                    residual_sequences
                )
                rows.append(
                    {
                        "layer": layer,
                        "split": split_name,
                        "predictor": predictor_name,
                        "clips": len(local_clips),
                        "frames": evaluated_frames,
                        **metrics,
                        "residual_top10_energy": residual_concentration(
                            target_array - prediction_array
                        ),
                        "raw_temporal_spectral_entropy": raw_entropy,
                        "residual_temporal_spectral_entropy": residual_entropy,
                        "spectral_entropy_reduction": (
                            1.0 - residual_entropy / raw_entropy
                            if raw_entropy > 0.0
                            else 0.0
                        ),
                        "ops_per_scalar_proxy": predictor_cost_per_scalar(
                            method,
                            params,
                        ),
                    }
                )
        candidates = [
            row
            for row in rows
            if int(row["layer"]) == layer and row["split"] == "val"
        ]
        selected = min(
            candidates,
            key=lambda row: (
                float(row["nmse"]),
                float(row["ops_per_scalar_proxy"]),
                str(row["predictor"]),
            ),
        )
        best[layer] = str(selected["predictor"])
    for row in rows:
        row["selected_on_val"] = (
            str(row["predictor"]) == best[int(row["layer"])]
        )
    return rows, best


def raw_index_entropy(
    arrays: Iterable[np.ndarray],
    codebook: Codebook,
    *,
    selected_fraction: float = 1.0,
) -> float:
    indices = []
    for array in arrays:
        payload = pq_quantize(
            array,
            codebook,
            selected_fraction=selected_fraction,
        )
        if payload.indices is not None:
            indices.append(payload.indices)
    merged = (
        np.concatenate(indices)
        if indices
        else np.empty(0, dtype=np.int32)
    )
    return empirical_entropy(merged, codebook.k)


def evaluate_raw_pq(
    clips: Sequence[ClipSequence],
    codebook: Codebook,
) -> dict[str, float]:
    reconstructions = []
    payload_bits = 0
    metadata_bits = 0
    indices = []
    for clip in clips:
        payload = pq_quantize(clip.values, codebook)
        reconstructions.append(payload.reconstruction)
        payload_bits += payload.payload_bits
        metadata_bits += payload.metadata_bits
        if payload.indices is not None:
            indices.append(payload.indices)
    target, reconstruction = concatenate_targets(clips, reconstructions)
    metrics = reconstruction_metrics(target, reconstruction)
    scalar_count = int(target.size)
    merged = np.concatenate(indices)
    return {
        **metrics,
        "payload_bits": float(payload_bits),
        "metadata_bits": float(metadata_bits),
        "payload_bps": float((payload_bits + metadata_bits) / scalar_count),
        "effective_bps": float(
            (payload_bits + metadata_bits + codebook.static_bits)
            / scalar_count
        ),
        "index_entropy": empirical_entropy(merged, codebook.k),
        "index_entropy_bps": float(
            empirical_entropy(merged, codebook.k) / codebook.group_dim
        ),
        "residual_top10_energy": residual_concentration(
            target - reconstruction
        ),
    }


def evaluate_residual_pq(
    clips: Sequence[ClipSequence],
    *,
    predictor_name: str,
    codebook: Codebook,
    selected_fraction: float,
) -> dict[str, float]:
    reconstructions = []
    payload_bits = 0.0
    metadata_bits = 0.0
    residual_arrays = []
    for clip in clips:
        reconstruction, accounting = run_residual_codec(
            clip.values,
            predictor_name=predictor_name,
            codebook=codebook,
            selected_fraction=selected_fraction,
        )
        reconstructions.append(reconstruction)
        payload_bits += accounting["payload_bits"]
        metadata_bits += accounting["metadata_bits"]
        residual_arrays.append(
            predictor_residuals(
                clip.values,
                predictor_name=predictor_name,
            )
        )
    target, reconstruction = concatenate_targets(clips, reconstructions)
    metrics = reconstruction_metrics(target, reconstruction)
    scalar_count = int(target.size)
    index_entropy = raw_index_entropy(
        residual_arrays,
        codebook,
        selected_fraction=selected_fraction,
    )
    return {
        **metrics,
        "payload_bits": payload_bits,
        "metadata_bits": metadata_bits,
        "payload_bps": float((payload_bits + metadata_bits) / scalar_count),
        "effective_bps": float(
            (payload_bits + metadata_bits + codebook.static_bits)
            / scalar_count
        ),
        "index_entropy": index_entropy,
        "index_entropy_bps": float(
            (
                index_entropy * selected_fraction
                + (1.0 if selected_fraction < 1.0 else 0.0)
            )
            / codebook.group_dim
        ),
        "residual_top10_energy": residual_concentration(
            target - reconstruction
        ),
    }


def evaluate_scalar_quantization(
    clips: Sequence[ClipSequence],
    bits: int,
) -> dict[str, float]:
    reconstructions = []
    payload_bits = 0
    metadata_bits = 0
    for clip in clips:
        payload = scalar_quantize(clip.values, bits)
        reconstructions.append(payload.reconstruction)
        payload_bits += payload.payload_bits
        metadata_bits += payload.metadata_bits
    target, reconstruction = concatenate_targets(clips, reconstructions)
    metrics = reconstruction_metrics(target, reconstruction)
    scalar_count = int(target.size)
    return {
        **metrics,
        "payload_bits": float(payload_bits),
        "metadata_bits": float(metadata_bits),
        "payload_bps": float((payload_bits + metadata_bits) / scalar_count),
        "effective_bps": float((payload_bits + metadata_bits) / scalar_count),
        "index_entropy": math.nan,
        "index_entropy_bps": math.nan,
        "residual_top10_energy": residual_concentration(
            target - reconstruction
        ),
    }


def evaluate_vq(
    clips: Sequence[ClipSequence],
    splits: dict[str, set[str]],
    *,
    layers: Sequence[int],
    best_predictors: dict[int, str],
    seed: int,
    max_train_samples: int,
    kmeans_iterations: int,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, str], Codebook],
]:
    rows: list[dict[str, Any]] = []
    residual_codebooks: dict[tuple[int, str], Codebook] = {}
    for layer in layers:
        train_clips = selected_clips(
            clips,
            layer=layer,
            runs=splits["train"],
        )
        predictor_name = best_predictors[layer]
        train_raw = [clip.values for clip in train_clips]
        train_residual = [
            predictor_residuals(
                clip.values,
                predictor_name=predictor_name,
            )
            for clip in train_clips
        ]
        for spec_index, (spec_name, index_bits, group_dim) in enumerate(
            PQ_SPECS
        ):
            raw_codebook = fit_codebook(
                collect_group_samples(
                    train_raw,
                    group_dim=group_dim,
                ),
                index_bits=index_bits,
                group_dim=group_dim,
                seed=seed + 1009 * layer + 17 * spec_index,
                max_samples=max_train_samples,
                iterations=kmeans_iterations,
            )
            residual_codebook = fit_codebook(
                collect_group_samples(
                    train_residual,
                    group_dim=group_dim,
                ),
                index_bits=index_bits,
                group_dim=group_dim,
                seed=seed + 1009 * layer + 17 * spec_index + 1,
                max_samples=max_train_samples,
                iterations=kmeans_iterations,
            )
            residual_codebooks[(layer, spec_name)] = residual_codebook
            for split_name in ("val", "test"):
                local_clips = selected_clips(
                    clips,
                    layer=layer,
                    runs=splits[split_name],
                )
                raw_metrics = evaluate_raw_pq(local_clips, raw_codebook)
                rows.append(
                    {
                        "layer": layer,
                        "split": split_name,
                        "method": "raw_pq",
                        "codec": spec_name,
                        "predictor": "",
                        "selected_fraction": 1.0,
                        "index_bits": index_bits,
                        "group_dim": group_dim,
                        "nominal_bps": index_bits / group_dim,
                        "codebook_static_bits": raw_codebook.static_bits,
                        "codebook_sha256": codebook_hash(raw_codebook),
                        **raw_metrics,
                    }
                )
                for selected_fraction in EVENT_FRACTIONS:
                    residual_metrics = evaluate_residual_pq(
                        local_clips,
                        predictor_name=predictor_name,
                        codebook=residual_codebook,
                        selected_fraction=selected_fraction,
                    )
                    rows.append(
                        {
                            "layer": layer,
                            "split": split_name,
                            "method": "residual_pq",
                            "codec": spec_name,
                            "predictor": predictor_name,
                            "selected_fraction": selected_fraction,
                            "index_bits": index_bits,
                            "group_dim": group_dim,
                            "nominal_bps": (
                                (
                                    selected_fraction * index_bits
                                    + (
                                        1.0
                                        if selected_fraction < 1.0
                                        else 0.0
                                    )
                                )
                                / group_dim
                            ),
                            "codebook_static_bits": (
                                residual_codebook.static_bits
                            ),
                            "codebook_sha256": codebook_hash(
                                residual_codebook
                            ),
                            **residual_metrics,
                        }
                    )
        for bits in SCALAR_BITS:
            for split_name in ("val", "test"):
                local_clips = selected_clips(
                    clips,
                    layer=layer,
                    runs=splits[split_name],
                )
                metrics = evaluate_scalar_quantization(local_clips, bits)
                rows.append(
                    {
                        "layer": layer,
                        "split": split_name,
                        "method": "scalar_quant",
                        "codec": f"int{bits}",
                        "predictor": "",
                        "selected_fraction": 1.0,
                        "index_bits": bits,
                        "group_dim": 1,
                        "nominal_bps": float(bits),
                        "codebook_static_bits": 0,
                        "codebook_sha256": "",
                        **metrics,
                    }
                )
    return rows, residual_codebooks


def choose_innovation_codec(
    vq_rows: Sequence[dict[str, Any]],
    *,
    layer: int,
    budget: float,
) -> dict[str, Any]:
    candidates = [
        row
        for row in vq_rows
        if int(row["layer"]) == layer
        and row["split"] == "val"
        and row["method"] == "residual_pq"
        and float(row["nominal_bps"]) <= budget + 1e-9
    ]
    if not candidates:
        raise RuntimeError(
            f"no residual PQ candidate under {budget} bps for layer {layer}"
        )
    return max(
        candidates,
        key=lambda row: (
            float(row["mean_cosine"]),
            -float(row["payload_bps"]),
            -float(row["effective_bps"]),
        ),
    )


def controller_examples(
    clips: Sequence[ClipSequence],
    *,
    predictor_name: str,
    codebook: Codebook,
    selected_fraction: float,
    quality_cosine: float,
) -> ControllerExamples:
    features = []
    labels = []
    runs = []
    frames = []
    method, params = predictor_name_and_params(predictor_name)
    for clip in clips:
        rgb_features = rgb_summary_features(clip.frames_rgb)
        for frame_index in range(1, clip.values.shape[0]):
            target = clip.values[frame_index]
            prediction = predict_next(
                list(clip.values[:frame_index]),
                method,
                params,
            )
            residual_payload = pq_quantize(
                target - prediction,
                codebook,
                selected_fraction=selected_fraction,
            )
            full_payload = scalar_quantize(target, 4)
            candidates = (
                clip.values[frame_index - 1],
                prediction,
                prediction + residual_payload.reconstruction,
                full_payload.reconstruction,
            )
            label = 3
            for action, candidate in enumerate(candidates):
                if frame_cosine(target, candidate) >= quality_cosine:
                    label = action
                    break
            features.append(rgb_features[frame_index])
            labels.append(label)
            runs.append(clip.run)
            frames.append(frame_index)
    return ControllerExamples(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        runs=tuple(runs),
        frames=np.asarray(frames, dtype=np.int64),
    )


def train_threshold_controller(
    train: ControllerExamples,
) -> ThresholdController:
    thresholds, mapping = fit_monotonic_thresholds(
        train.features[:, 0],
        train.labels,
    )
    return ThresholdController(thresholds, mapping)


def train_tree_controller(
    train: ControllerExamples,
    val: ControllerExamples,
) -> TreeController:
    candidates = []
    for depth in (2, 3, 4):
        tree = fit_decision_tree(
            train.features,
            train.labels,
            max_depth=depth,
            min_samples=8,
        )
        metrics = classification_metrics(
            val.labels,
            tree.predict(val.features),
        )
        candidates.append((metrics["balanced_accuracy"], -tree.node_count(), tree))
    _, _, selected = max(candidates, key=lambda item: (item[0], item[1]))
    return TreeController(selected)


def train_torch_controllers(
    train: ControllerExamples,
    val: ControllerExamples,
    *,
    seed: int,
) -> tuple[TorchController, TorchController, dict[str, float]]:
    import torch
    from torch import nn

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    x_train = torch.as_tensor(train.features, dtype=torch.float32)
    y_train = torch.as_tensor(train.labels, dtype=torch.long)
    x_val = torch.as_tensor(val.features, dtype=torch.float32)
    y_val = torch.as_tensor(val.labels, dtype=torch.long)
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-6)

    class TinyMLP(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 24),
                nn.ReLU(),
                nn.Linear(24, 4),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.network((values - mean) / std)

    def train_mlp(local_seed: int) -> tuple[nn.Module, float]:
        torch.manual_seed(local_seed)
        model = TinyMLP(x_train.shape[1])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.02,
            weight_decay=1e-3,
        )
        best_state = None
        best_accuracy = -1.0
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x_train), y_train)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy = float(
                    (model(x_val).argmax(dim=1) == y_val).float().mean()
                )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
        assert best_state is not None
        model.load_state_dict(best_state)
        return model.eval(), best_accuracy

    mlp_candidates = [train_mlp(seed + offset) for offset in range(3)]
    mlp, _ = max(mlp_candidates, key=lambda item: item[1])
    mlp_params = sum(parameter.numel() for parameter in mlp.parameters())
    normalizer_params = int(mean.numel() + std.numel())

    def mlp_predict(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(values, dtype=torch.float32)
            return mlp(tensor).argmax(dim=1).cpu().numpy()

    mlp_controller = TorchController(
        name="mlp",
        predict_fn=mlp_predict,
        static_bits=(mlp_params + normalizer_params) * 32,
        extra={
            "parameter_count": mlp_params,
            "normalizer_parameter_count": normalizer_params,
        },
    )

    thresholds = (
        torch.quantile(
            x_train,
            torch.tensor([0.25, 0.50, 0.75]),
            dim=0,
        )
        .T.contiguous()
        .to(torch.float16)
        .to(torch.float32)
    )

    def encode_bits(values: torch.Tensor) -> torch.Tensor:
        return (
            values[:, :, None] > thresholds[None]
        ).reshape(values.shape[0], -1).to(torch.float32)

    truth = torch.tensor(
        [
            [(gate >> bit) & 1 for bit in range(4)]
            for gate in range(16)
        ],
        dtype=torch.float32,
    )

    class LogicLayer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            width: int,
            *,
            generator: torch.Generator,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.width = width
            self.register_buffer(
                "left",
                torch.randint(
                    0,
                    input_dim,
                    (width,),
                    generator=generator,
                ),
            )
            self.register_buffer(
                "right",
                torch.randint(
                    0,
                    input_dim,
                    (width,),
                    generator=generator,
                ),
            )
            self.logits = nn.Parameter(torch.zeros(width, 16))
            nn.init.normal_(self.logits, std=0.05)

        def forward(
            self,
            values: torch.Tensor,
            *,
            hard: bool,
            straight_through: bool,
            temperature: float,
        ) -> torch.Tensor:
            left = values[:, self.left]
            right = values[:, self.right]
            basis = torch.stack(
                (
                    (1.0 - left) * (1.0 - right),
                    (1.0 - left) * right,
                    left * (1.0 - right),
                    left * right,
                ),
                dim=-1,
            )
            gate_outputs = torch.einsum("bwi,gi->bwg", basis, truth)
            probabilities = torch.softmax(self.logits / temperature, dim=-1)
            soft = torch.sum(
                gate_outputs * probabilities[None],
                dim=-1,
            )
            if not hard:
                return soft
            gate_ids = self.logits.argmax(dim=-1)
            hard_output = gate_outputs.gather(
                2,
                gate_ids[None, :, None].expand(values.shape[0], -1, 1),
            )[:, :, 0]
            if straight_through:
                return hard_output + soft - soft.detach()
            return hard_output

    class DLGN(nn.Module):
        def __init__(self, input_dim: int, width: int, local_seed: int) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(local_seed)
            self.width = width
            self.layers = nn.ModuleList(
                [
                    LogicLayer(input_dim, width, generator=generator),
                    LogicLayer(width, width, generator=generator),
                ]
            )

        def forward(
            self,
            values: torch.Tensor,
            *,
            hard: bool,
            straight_through: bool = False,
            temperature: float = 1.0,
        ) -> torch.Tensor:
            current = encode_bits(values)
            for layer in self.layers:
                current = layer(
                    current,
                    hard=hard,
                    straight_through=straight_through,
                    temperature=temperature,
                )
            return current.reshape(current.shape[0], 4, -1).mean(dim=2)

    def train_dlgn(local_seed: int) -> tuple[DLGN, float, float]:
        torch.manual_seed(local_seed)
        model = DLGN(encode_bits(x_train[:1]).shape[1], 64, local_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        best_state = None
        best_hard = -1.0
        best_soft = -1.0
        for epoch in range(500):
            temperature = max(0.15, 1.5 * (0.992 ** epoch))
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                x_train,
                hard=True,
                straight_through=True,
                temperature=temperature,
            )
            loss = nn.functional.cross_entropy(logits, y_train)
            entropy = 0.0
            for layer in model.layers:
                probabilities = torch.softmax(
                    layer.logits / temperature,
                    dim=-1,
                )
                entropy = entropy + (
                    -probabilities
                    * torch.log(probabilities.clamp_min(1e-8))
                ).sum(dim=-1).mean()
            total_loss = loss + 1e-4 * entropy
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                hard_accuracy = float(
                    (
                        model(x_val, hard=True).argmax(dim=1) == y_val
                    )
                    .float()
                    .mean()
                )
                soft_accuracy = float(
                    (
                        model(x_val, hard=False).argmax(dim=1) == y_val
                    )
                    .float()
                    .mean()
                )
            if (hard_accuracy, soft_accuracy) > (best_hard, best_soft):
                best_hard = hard_accuracy
                best_soft = soft_accuracy
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
        assert best_state is not None
        model.load_state_dict(best_state)
        return model.eval(), best_hard, best_soft

    dlgn_candidates = [
        train_dlgn(seed + 100 + offset) for offset in range(3)
    ]
    dlgn, _, _ = max(
        dlgn_candidates,
        key=lambda item: (item[1], item[2]),
    )
    with torch.no_grad():
        val_hard = dlgn(x_val, hard=True).argmax(dim=1)
        val_soft = dlgn(x_val, hard=False).argmax(dim=1)
        soft_hard_agreement = float(
            (val_hard == val_soft).float().mean()
        )
        val_hard_accuracy = float((val_hard == y_val).float().mean())
        val_soft_accuracy = float((val_soft == y_val).float().mean())

    def dlgn_predict(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(values, dtype=torch.float32)
            return dlgn(tensor, hard=True).argmax(dim=1).cpu().numpy()

    input_bits = int(thresholds.numel())
    width = dlgn.width
    wiring_bits = (
        width * 2 * math.ceil(math.log2(max(input_bits, 2)))
        + width * 2 * math.ceil(math.log2(width))
    )
    operation_bits = width * len(dlgn.layers) * 4
    threshold_bits = int(thresholds.numel() * 16)
    dlgn_controller = TorchController(
        name="dlgn",
        predict_fn=dlgn_predict,
        static_bits=wiring_bits + operation_bits + threshold_bits,
        extra={
            "gate_count": width * len(dlgn.layers),
            "depth": len(dlgn.layers),
            "input_bits": input_bits,
            "val_soft_accuracy": val_soft_accuracy,
            "val_hard_accuracy": val_hard_accuracy,
            "val_discretization_gap": (
                val_soft_accuracy - val_hard_accuracy
            ),
            "val_soft_hard_agreement": soft_hard_agreement,
        },
    )
    dlgn_metrics = {
        "val_soft_accuracy": val_soft_accuracy,
        "val_hard_accuracy": val_hard_accuracy,
        "val_discretization_gap": val_soft_accuracy - val_hard_accuracy,
        "val_soft_hard_agreement": soft_hard_agreement,
    }
    return mlp_controller, dlgn_controller, dlgn_metrics


def evaluate_controller_policy(
    clips: Sequence[ClipSequence],
    *,
    controller: Controller,
    predictor_name: str,
    codebook: Codebook,
    selected_fraction: float,
) -> dict[str, float]:
    targets = []
    reconstructions = []
    payload_bits = 0.0
    metadata_bits = 0.0
    action_counts = np.zeros(4, dtype=np.int64)
    for clip in clips:
        features = rgb_summary_features(clip.frames_rgb)
        actions = np.full(clip.values.shape[0], 3, dtype=np.int64)
        actions[1:] = controller.predict(features[1:])
        reconstruction, accounting = run_residual_codec(
            clip.values,
            predictor_name=predictor_name,
            codebook=codebook,
            selected_fraction=selected_fraction,
            actions=actions,
        )
        targets.append(clip.values)
        reconstructions.append(reconstruction)
        payload_bits += accounting["payload_bits"]
        metadata_bits += accounting["metadata_bits"]
        action_counts += np.bincount(actions, minlength=4)
    target = np.concatenate(targets, axis=0)
    reconstruction = np.concatenate(reconstructions, axis=0)
    metrics = reconstruction_metrics(target, reconstruction)
    frame_count = int(action_counts.sum())
    scalar_count = int(target.size)
    action_metadata_bits = 2 * frame_count
    effective_bits = (
        payload_bits
        + metadata_bits
        + action_metadata_bits
        + codebook.static_bits
        + controller.static_bits
    )
    action_cost_proxy = (
        0.0 * action_counts[0]
        + 0.05 * action_counts[1]
        + 0.25 * action_counts[2]
        + 1.0 * action_counts[3]
    ) / max(frame_count, 1)
    encoder_required_rate = float(
        (action_counts[2] + action_counts[3]) / max(frame_count, 1)
    )
    encoder_skip_rate = float(
        (action_counts[0] + action_counts[1]) / max(frame_count, 1)
    )
    return {
        **metrics,
        "evaluated_scalars": float(scalar_count),
        "payload_bps": float(
            (payload_bits + metadata_bits + action_metadata_bits)
            / scalar_count
        ),
        "effective_bps": float(effective_bits / scalar_count),
        "controller_static_bits": float(controller.static_bits),
        "codebook_static_bits": float(codebook.static_bits),
        "reuse_rate": float(action_counts[0] / frame_count),
        "predict_rate": float(action_counts[1] / frame_count),
        "innovation_rate": float(action_counts[2] / frame_count),
        "refresh_rate": float(action_counts[3] / frame_count),
        "encoder_required_rate": encoder_required_rate,
        "encoder_skip_rate": encoder_skip_rate,
        "state_update_cost_proxy": float(action_cost_proxy),
        "action_cost_proxy": float(action_cost_proxy),
    }


def evaluate_controllers(
    clips: Sequence[ClipSequence],
    splits: dict[str, set[str]],
    *,
    layers: Sequence[int],
    best_predictors: dict[int, str],
    vq_rows: Sequence[dict[str, Any]],
    residual_codebooks: dict[tuple[int, str], Codebook],
    quality_cosine: float,
    seed: int,
    use_torch: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    controller_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for layer in layers:
        for budget in CONTROLLER_BUDGETS:
            selected_codec = choose_innovation_codec(
                vq_rows,
                layer=layer,
                budget=budget,
            )
            codec_name = str(selected_codec["codec"])
            selected_fraction = float(selected_codec["selected_fraction"])
            codebook = residual_codebooks[(layer, codec_name)]
            predictor_name = best_predictors[layer]
            examples: dict[str, ControllerExamples] = {}
            local_clips: dict[str, list[ClipSequence]] = {}
            for split_name in ("train", "val", "test"):
                local_clips[split_name] = selected_clips(
                    clips,
                    layer=layer,
                    runs=splits[split_name],
                )
                examples[split_name] = controller_examples(
                    local_clips[split_name],
                    predictor_name=predictor_name,
                    codebook=codebook,
                    selected_fraction=selected_fraction,
                    quality_cosine=quality_cosine,
                )
            controllers: list[Controller] = [
                train_threshold_controller(examples["train"]),
                train_tree_controller(examples["train"], examples["val"]),
            ]
            dlgn_extra: dict[str, float] = {}
            if use_torch:
                mlp, dlgn, dlgn_extra = train_torch_controllers(
                    examples["train"],
                    examples["val"],
                    seed=seed + layer + int(round(100 * budget)),
                )
                controllers.extend((mlp, dlgn))
            for controller in controllers:
                predictions = controller.predict(examples["test"].features)
                classification = classification_metrics(
                    examples["test"].labels,
                    predictions,
                )
                row: dict[str, Any] = {
                    "layer": layer,
                    "budget_bps": budget,
                    "controller": controller.name,
                    "predictor": predictor_name,
                    "innovation_codec": codec_name,
                    "innovation_fraction": selected_fraction,
                    "quality_cosine": quality_cosine,
                    "test_samples": examples["test"].labels.size,
                    **classification,
                    "controller_static_bits": controller.static_bits,
                }
                if isinstance(controller, TreeController):
                    row["tree_nodes"] = controller.tree.node_count()
                    row["tree_depth"] = controller.tree.depth()
                if isinstance(controller, TorchController):
                    row.update(controller.extra)
                if controller.name == "dlgn":
                    row.update(dlgn_extra)
                controller_rows.append(row)
                combined = evaluate_controller_policy(
                    local_clips["test"],
                    controller=controller,
                    predictor_name=predictor_name,
                    codebook=codebook,
                    selected_fraction=selected_fraction,
                )
                combined_rows.append(
                    {
                        "layer": layer,
                        "budget_bps": budget,
                        "controller": controller.name,
                        "predictor": predictor_name,
                        "innovation_codec": codec_name,
                        "innovation_fraction": selected_fraction,
                        "quality_cosine": quality_cosine,
                        **combined,
                    }
                )
            for baseline_name, fixed_action in (
                ("always_reuse", 0),
                ("always_predict", 1),
                ("always_innovation", 2),
                ("always_int4_refresh", 3),
            ):
                class FixedController(Controller):
                    def __init__(self) -> None:
                        self.name = baseline_name
                        self.static_bits = 0

                    def predict(self, features: np.ndarray) -> np.ndarray:
                        return np.full(
                            np.asarray(features).shape[0],
                            fixed_action,
                            dtype=np.int64,
                        )

                combined = evaluate_controller_policy(
                    local_clips["test"],
                    controller=FixedController(),
                    predictor_name=predictor_name,
                    codebook=codebook,
                    selected_fraction=selected_fraction,
                )
                combined_rows.append(
                    {
                        "layer": layer,
                        "budget_bps": budget,
                        "controller": baseline_name,
                        "predictor": predictor_name,
                        "innovation_codec": codec_name,
                        "innovation_fraction": selected_fraction,
                        "quality_cosine": quality_cosine,
                        **combined,
                    }
                )
    return controller_rows, combined_rows


def split_manifest(
    clips: Sequence[ClipSequence],
    splits: dict[str, set[str]],
) -> list[dict[str, Any]]:
    run_to_category = {clip.run: clip.category for clip in clips}
    rows = []
    for split_name, runs in splits.items():
        for run in sorted(runs):
            rows.append(
                {
                    "run": run,
                    "category": run_to_category[run],
                    "split": split_name,
                }
            )
    return rows


def build_markdown_summary(
    *,
    predictor_rows: Sequence[dict[str, Any]],
    vq_rows: Sequence[dict[str, Any]],
    controller_rows: Sequence[dict[str, Any]],
    combined_rows: Sequence[dict[str, Any]],
    best_predictors: dict[int, str],
    use_torch: bool,
) -> str:
    lines = [
        "# Streaming Hybrid State V0 Result Summary",
        "",
        "This is a representation-level causal probe. It does not claim "
        "end-to-end Video-LLM accuracy, PPA, or encoder speedup.",
        "",
        "## Selected Predictors",
        "",
        "| Layer | Validation-selected predictor | Test NMSE | Test cosine |",
        "|---:|---|---:|---:|",
    ]
    for layer, predictor in sorted(best_predictors.items()):
        row = next(
            item
            for item in predictor_rows
            if int(item["layer"]) == layer
            and item["split"] == "test"
            and item["predictor"] == predictor
        )
        lines.append(
            f"| {layer} | {predictor} | {float(row['nmse']):.6f} | "
            f"{float(row['mean_cosine']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Best VQ Points",
            "",
            "| Layer | Budget | Method | Codec | Stream bps | Effective bps | Cosine |",
            "|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for layer in sorted(best_predictors):
        for budget in CONTROLLER_BUDGETS:
            selected = choose_innovation_codec(
                vq_rows,
                layer=layer,
                budget=budget,
            )
            test = next(
                item
                for item in vq_rows
                if int(item["layer"]) == layer
                and item["split"] == "test"
                and item["method"] == selected["method"]
                and item["codec"] == selected["codec"]
                and float(item["selected_fraction"])
                == float(selected["selected_fraction"])
            )
            lines.append(
                f"| {layer} | {budget:.2f} | {test['method']} | "
                f"{test['codec']}@{float(test['selected_fraction']):.2f} | "
                f"{float(test['payload_bps']):.4f} | "
                f"{float(test['effective_bps']):.4f} | "
                f"{float(test['mean_cosine']):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Controller And Combined Policy",
            "",
            f"Torch controllers enabled: `{use_torch}`.",
            "",
            "| Layer | Budget | Controller | Action accuracy | Combined cosine | "
            "Refresh rate | Effective bps |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    lookup = {
        (
            int(row["layer"]),
            float(row["budget_bps"]),
            str(row["controller"]),
        ): row
        for row in combined_rows
    }
    for row in controller_rows:
        key = (
            int(row["layer"]),
            float(row["budget_bps"]),
            str(row["controller"]),
        )
        combined = lookup[key]
        lines.append(
            f"| {key[0]} | {key[1]:.2f} | {key[2]} | "
            f"{float(row['accuracy']):.4f} | "
            f"{float(combined['mean_cosine']):.6f} | "
            f"{float(combined['refresh_rate']):.4f} | "
            f"{float(combined['effective_bps']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Predictor, VQ, and controller rows use disjoint clip-level splits.",
            "- Codebook, bitmap, scale, action, and controller metadata are counted.",
            "- `payload_bps` excludes one-time static codebook/controller bits; "
            "`effective_bps` amortizes them over the test corpus.",
            "- DLGN policy results use the final hard gate network. Soft accuracy "
            "and discretization gap are reported separately.",
            "- The RGB controller is a pre-encoder proxy, but innovation coding "
            "still requires current hidden features. `encoder_required_rate` "
            "therefore counts both innovation and refresh actions. No ViT skip "
            "or measured-compute claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not 0.0 < args.quality_cosine <= 1.0:
        raise ValueError("quality cosine must be in (0,1]")
    layers = parse_int_list(args.layers)
    started = time.time()
    clips = discover_clips(
        args.root,
        layers=layers,
        pool_rows=args.pool_rows,
        pool_cols=args.pool_cols,
    )
    if args.limit > 0:
        if args.limit < 4:
            raise ValueError("--limit must be 0 or at least 4")
        by_category: dict[str, list[str]] = {}
        for clip in clips:
            by_category.setdefault(clip.category, [])
            if clip.run not in by_category[clip.category]:
                by_category[clip.category].append(clip.run)
        allowed_runs = {
            run
            for runs in by_category.values()
            for run in sorted(runs)[: args.limit]
        }
        clips = [clip for clip in clips if clip.run in allowed_runs]
    splits = split_runs(clips)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "split_manifest.csv", split_manifest(clips, splits))

    predictor_rows, best_predictors = evaluate_predictors(
        clips,
        splits,
        layers=layers,
    )
    write_csv(args.out_dir / "predictor_results.csv", predictor_rows)

    vq_rows, residual_codebooks = evaluate_vq(
        clips,
        splits,
        layers=layers,
        best_predictors=best_predictors,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        kmeans_iterations=args.kmeans_iterations,
    )
    write_csv(args.out_dir / "vq_results.csv", vq_rows)

    use_torch = not args.no_torch
    if use_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            use_torch = False
    controller_rows, combined_rows = evaluate_controllers(
        clips,
        splits,
        layers=layers,
        best_predictors=best_predictors,
        vq_rows=vq_rows,
        residual_codebooks=residual_codebooks,
        quality_cosine=args.quality_cosine,
        seed=args.seed,
        use_torch=use_torch,
    )
    write_csv(args.out_dir / "controller_results.csv", controller_rows)
    write_csv(args.out_dir / "combined_results.csv", combined_rows)

    summary = {
        "scope": (
            "Representation-level causal comparison on pooled frozen visual "
            "hidden states. No end-to-end Video-LLM, PPA, or encoder-speed "
            "claim is made."
        ),
        "root": str(args.root.resolve()),
        "layers": layers,
        "pool_grid": [args.pool_rows, args.pool_cols],
        "runs": {
            name: sorted(values) for name, values in splits.items()
        },
        "best_predictors": {
            str(layer): predictor
            for layer, predictor in best_predictors.items()
        },
        "quality_cosine": args.quality_cosine,
        "torch_controllers": use_torch,
        "predictor_rows": len(predictor_rows),
        "vq_rows": len(vq_rows),
        "controller_rows": len(controller_rows),
        "combined_rows": len(combined_rows),
        "elapsed_seconds": time.time() - started,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    if use_torch:
        import torch

        summary["versions"]["torch"] = torch.__version__
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "RESULT_SUMMARY.md").write_text(
        build_markdown_summary(
            predictor_rows=predictor_rows,
            vq_rows=vq_rows,
            controller_rows=controller_rows,
            combined_rows=combined_rows,
            best_predictors=best_predictors,
            use_torch=use_torch,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
