from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = (
    "recent_window",
    "uniform_reservoir",
    "adaptive_slots",
    "oja_subspace",
    "instant_adaptive",
    "instant_oja",
)


def parse_int_list(value: str) -> list[int]:
    output = sorted({int(item) for item in value.split(",") if item.strip()})
    if not output or output[0] < 0:
        raise ValueError("integer lists must be non-empty and non-negative")
    return output


def parse_str_list(value: str) -> list[str]:
    output = [item.strip() for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("string lists must be non-empty")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory whose immediate children contain hidden.npz files.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--capacities", default="16,32,64")
    parser.add_argument("--delays", default="0,1,2,4,8")
    parser.add_argument(
        "--modes",
        default="raw_unit,frame_centered_unit",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--pool-rows", type=int, default=4)
    parser.add_argument("--pool-cols", type=int, default=4)
    parser.add_argument("--instant-frames", type=int, default=1)
    parser.add_argument("--storage-bits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oja-lr", type=float, default=0.5)
    parser.add_argument("--slot-min-lr", type=float, default=0.05)
    parser.add_argument(
        "--slot-replace-similarity",
        type=float,
        default=0.75,
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    parser.add_argument("--plot-capacity", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(values),
    )


def pool_regions(
    frame: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"frame must have [H,W,D], got {values.shape}")
    height, width, hidden_dim = values.shape
    if not 1 <= rows <= height or not 1 <= cols <= width:
        raise ValueError(
            f"pool grid {rows}x{cols} is invalid for {height}x{width}"
        )
    pooled = []
    for row_indices in np.array_split(np.arange(height), rows):
        for column_indices in np.array_split(np.arange(width), cols):
            region = values[np.ix_(row_indices, column_indices)]
            pooled.append(region.reshape(-1, hidden_dim).mean(axis=0))
    return np.stack(pooled)


def prepare_sequence(
    sequence: np.ndarray,
    *,
    rows: int,
    cols: int,
    mode: str,
) -> list[np.ndarray]:
    values = np.asarray(sequence)
    if values.ndim != 4:
        raise ValueError(f"sequence must have [T,H,W,D], got {values.shape}")
    output = []
    for frame in values:
        pooled = pool_regions(frame, rows=rows, cols=cols)
        if mode == "raw_unit":
            transformed = pooled
        elif mode == "frame_centered_unit":
            transformed = pooled - pooled.mean(axis=0, keepdims=True)
        else:
            raise ValueError(f"unsupported representation mode: {mode}")
        output.append(normalize_rows(transformed))
    return output


def nearest_reconstruction(
    queries: np.ndarray,
    memory: np.ndarray,
) -> np.ndarray:
    if memory.size == 0:
        return np.zeros_like(queries)
    similarities = queries @ memory.T
    return memory[np.argmax(similarities, axis=1)]


class RecentWindowMemory:
    def __init__(self, capacity: int, hidden_dim: int) -> None:
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self._values = np.empty((0, hidden_dim), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        self._values = np.concatenate((self._values, values), axis=0)[
            -self.capacity :
        ]

    def reconstruct(self, queries: np.ndarray) -> np.ndarray:
        return nearest_reconstruction(queries, self._values)


class ReservoirMemory:
    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        *,
        seed: int,
    ) -> None:
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self._values = np.empty((0, hidden_dim), dtype=np.float64)
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        for value in values:
            self._seen += 1
            if self._values.shape[0] < self.capacity:
                self._values = np.concatenate(
                    (self._values, value[None]),
                    axis=0,
                )
                continue
            replacement = int(self._rng.integers(0, self._seen))
            if replacement < self.capacity:
                self._values[replacement] = value

    def reconstruct(self, queries: np.ndarray) -> np.ndarray:
        return nearest_reconstruction(queries, self._values)


class AdaptiveSlotMemory:
    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        *,
        min_lr: float,
        replace_similarity: float,
    ) -> None:
        if not 0 < min_lr <= 1:
            raise ValueError("min_lr must be in (0,1]")
        if not -1 <= replace_similarity <= 1:
            raise ValueError("replace_similarity must be in [-1,1]")
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.min_lr = min_lr
        self.replace_similarity = replace_similarity
        self._values = np.empty((0, hidden_dim), dtype=np.float64)
        self._counts = np.empty((0,), dtype=np.int64)

    def _redundant_slot(self) -> int:
        similarities = self._values @ self._values.T
        np.fill_diagonal(similarities, -np.inf)
        left, right = np.unravel_index(
            int(np.argmax(similarities)),
            similarities.shape,
        )
        return (
            int(left)
            if self._counts[left] <= self._counts[right]
            else int(right)
        )

    def update(self, values: np.ndarray) -> None:
        for value in values:
            if self._values.shape[0] < self.capacity:
                self._values = np.concatenate(
                    (self._values, value[None]),
                    axis=0,
                )
                self._counts = np.concatenate(
                    (self._counts, np.ones(1, dtype=np.int64))
                )
                continue
            similarities = self._values @ value
            index = int(np.argmax(similarities))
            if float(similarities[index]) < self.replace_similarity:
                index = self._redundant_slot()
                self._values[index] = value
                self._counts[index] = 1
                continue
            self._counts[index] += 1
            learning_rate = max(
                self.min_lr,
                1.0 / np.sqrt(float(self._counts[index])),
            )
            updated = (
                (1.0 - learning_rate) * self._values[index]
                + learning_rate * value
            )
            norm = float(np.linalg.norm(updated))
            self._values[index] = (
                updated / norm if norm > 1e-12 else updated
            )

    def reconstruct(self, queries: np.ndarray) -> np.ndarray:
        return nearest_reconstruction(queries, self._values)


class OjaSubspaceMemory:
    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        *,
        seed: int,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.capacity = min(capacity, hidden_dim)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self._step = 0
        self._rng = np.random.default_rng(seed)
        self._basis: np.ndarray | None = None

    def _initialize(self, values: np.ndarray) -> None:
        _, _, right = np.linalg.svd(
            values,
            full_matrices=False,
        )
        retained = min(self.capacity, right.shape[0])
        basis = right[:retained].T
        if retained < self.capacity:
            random = self._rng.normal(
                size=(self.hidden_dim, self.capacity - retained)
            )
            if retained:
                random -= basis @ (basis.T @ random)
            random = np.linalg.qr(random, mode="reduced")[0]
            basis = np.concatenate((basis, random), axis=1)
        self._basis = np.linalg.qr(basis, mode="reduced")[0][
            :, : self.capacity
        ]

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        if self._basis is None:
            self._initialize(values)
            self._step = 1
            return
        self._step += 1
        projected = values @ self._basis
        covariance_action = values.T @ projected / values.shape[0]
        in_span = self._basis @ (
            projected.T @ projected / values.shape[0]
        )
        rate = self.learning_rate / np.sqrt(float(self._step))
        candidate = self._basis + rate * (covariance_action - in_span)
        self._basis = np.linalg.qr(candidate, mode="reduced")[0]

    def reconstruct(self, queries: np.ndarray) -> np.ndarray:
        if self._basis is None:
            return np.zeros_like(queries)
        return (queries @ self._basis) @ self._basis.T


class InstantPlusLongTermMemory:
    def __init__(
        self,
        *,
        instant_capacity: int,
        hidden_dim: int,
        long_term: object | None,
    ) -> None:
        self.instant_capacity = instant_capacity
        self.hidden_dim = hidden_dim
        self.long_term = long_term
        self._instant = np.empty((0, hidden_dim), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        combined = np.concatenate(
            (self._instant, np.asarray(values)),
            axis=0,
        )
        overflow = max(0, combined.shape[0] - self.instant_capacity)
        if self.long_term is not None and overflow:
            self.long_term.update(combined[:overflow])
        self._instant = combined[-self.instant_capacity :].copy()

    def reconstruct(self, queries: np.ndarray) -> np.ndarray:
        instant = nearest_reconstruction(queries, self._instant)
        if self.long_term is None:
            return instant
        long_term = self.long_term.reconstruct(queries)
        instant_norm = np.maximum(
            np.linalg.norm(instant, axis=1),
            1e-12,
        )
        long_norm = np.maximum(
            np.linalg.norm(long_term, axis=1),
            1e-12,
        )
        query_norm = np.maximum(
            np.linalg.norm(queries, axis=1),
            1e-12,
        )
        instant_cosine = np.sum(queries * instant, axis=1) / (
            query_norm * instant_norm
        )
        long_cosine = np.sum(queries * long_term, axis=1) / (
            query_norm * long_norm
        )
        use_long = long_cosine > instant_cosine
        output = instant.copy()
        output[use_long] = long_term[use_long]
        return output


def make_memory(
    method: str,
    *,
    capacity: int,
    hidden_dim: int,
    seed: int,
    oja_lr: float,
    slot_min_lr: float,
    slot_replace_similarity: float,
    instant_capacity: int,
) -> object:
    if method == "recent_window":
        return RecentWindowMemory(capacity, hidden_dim)
    if method == "uniform_reservoir":
        return ReservoirMemory(capacity, hidden_dim, seed=seed)
    if method == "adaptive_slots":
        return AdaptiveSlotMemory(
            capacity,
            hidden_dim,
            min_lr=slot_min_lr,
            replace_similarity=slot_replace_similarity,
        )
    if method == "oja_subspace":
        return OjaSubspaceMemory(
            capacity,
            hidden_dim,
            seed=seed,
            learning_rate=oja_lr,
        )
    long_capacity = capacity - instant_capacity
    if method == "instant_adaptive":
        long_term = (
            AdaptiveSlotMemory(
                long_capacity,
                hidden_dim,
                min_lr=slot_min_lr,
                replace_similarity=slot_replace_similarity,
            )
            if long_capacity > 0
            else None
        )
        return InstantPlusLongTermMemory(
            instant_capacity=instant_capacity,
            hidden_dim=hidden_dim,
            long_term=long_term,
        )
    if method == "instant_oja":
        long_term = (
            OjaSubspaceMemory(
                long_capacity,
                hidden_dim,
                seed=seed,
                learning_rate=oja_lr,
            )
            if long_capacity > 0
            else None
        )
        return InstantPlusLongTermMemory(
            instant_capacity=instant_capacity,
            hidden_dim=hidden_dim,
            long_term=long_term,
        )
    raise ValueError(f"unknown method: {method}")


def reconstruction_metrics(
    queries: np.ndarray,
    reconstruction: np.ndarray,
    *,
    coverage_threshold: float,
) -> dict[str, float]:
    query_norms = np.linalg.norm(queries, axis=1)
    reconstructed_norms = np.linalg.norm(reconstruction, axis=1)
    denominator = np.maximum(query_norms * reconstructed_norms, 1e-12)
    cosine = np.sum(queries * reconstruction, axis=1) / denominator
    relative_l2 = np.linalg.norm(
        queries - reconstruction,
        axis=1,
    ) / np.maximum(query_norms, 1e-12)
    return {
        "mean_cosine": float(np.mean(cosine)),
        "median_cosine": float(np.median(cosine)),
        "mean_relative_l2": float(np.mean(relative_l2)),
        "coverage": float(np.mean(cosine >= coverage_threshold)),
    }


def state_accounting(
    method: str,
    *,
    capacity: int,
    hidden_dim: int,
    storage_bits: int,
    instant_capacity: int,
) -> dict[str, int]:
    payload_bytes = capacity * hidden_dim * storage_bits // 8
    if method == "recent_window":
        metadata_bytes = 16
        read_flops = 2 * capacity * hidden_dim
        update_flops = 0
    elif method == "uniform_reservoir":
        metadata_bytes = 16
        read_flops = 2 * capacity * hidden_dim
        update_flops = 0
    elif method == "adaptive_slots":
        metadata_bytes = 8 * capacity + 8
        read_flops = 2 * capacity * hidden_dim
        update_flops = 2 * capacity * hidden_dim + 4 * hidden_dim
    elif method == "oja_subspace":
        metadata_bytes = 8
        read_flops = 4 * capacity * hidden_dim
        update_flops = 6 * capacity * hidden_dim
    elif method == "instant_adaptive":
        long_capacity = capacity - instant_capacity
        metadata_bytes = 8 * long_capacity + 24
        read_flops = 2 * capacity * hidden_dim
        update_flops = (
            2 * long_capacity * hidden_dim + 4 * hidden_dim
            if long_capacity > 0
            else 0
        )
    elif method == "instant_oja":
        long_capacity = capacity - instant_capacity
        metadata_bytes = 16
        read_flops = (
            2 * instant_capacity * hidden_dim
            + 4 * long_capacity * hidden_dim
        )
        update_flops = 6 * long_capacity * hidden_dim
    else:
        raise ValueError(f"unknown method: {method}")
    return {
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "total_state_bytes": payload_bytes + metadata_bytes,
        "read_flops_per_query_token": read_flops,
        "update_flops_per_input_token": update_flops,
    }


def evaluate_sequence(
    sequence: list[np.ndarray],
    *,
    run: str,
    category: str,
    layer: int,
    mode: str,
    capacity: int,
    delays: list[int],
    storage_bits: int,
    seed: int,
    oja_lr: float,
    slot_min_lr: float,
    slot_replace_similarity: float,
    coverage_threshold: float,
    methods: list[str],
    instant_frames: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not sequence:
        return [], []
    hidden_dim = sequence[0].shape[1]
    tokens_per_frame = sequence[0].shape[0]
    rows: list[dict[str, object]] = []
    accounting_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        instant_capacity = min(
            tokens_per_frame * instant_frames,
            capacity,
        )
        memory = make_memory(
            method,
            capacity=capacity,
            hidden_dim=hidden_dim,
            seed=seed + 1009 * method_index,
            oja_lr=oja_lr,
            slot_min_lr=slot_min_lr,
            slot_replace_similarity=slot_replace_similarity,
            instant_capacity=instant_capacity,
        )
        accounting_rows.append(
            {
                "run": run,
                "category": category,
                "layer": layer,
                "mode": mode,
                "method": method,
                "capacity": capacity,
                "hidden_dim": hidden_dim,
                "storage_bits": storage_bits,
                "instant_capacity": (
                    instant_capacity
                    if method.startswith("instant_")
                    else 0
                ),
                **state_accounting(
                    method,
                    capacity=capacity,
                    hidden_dim=hidden_dim,
                    storage_bits=storage_bits,
                    instant_capacity=instant_capacity,
                ),
            }
        )
        for frame_index, current in enumerate(sequence):
            memory.update(current)
            for delay in delays:
                if frame_index < delay:
                    continue
                queries = sequence[frame_index - delay]
                reconstruction = memory.reconstruct(queries)
                rows.append(
                    {
                        "run": run,
                        "category": category,
                        "layer": layer,
                        "mode": mode,
                        "method": method,
                        "capacity": capacity,
                        "delay": delay,
                        "frame": frame_index,
                        "query_frame": frame_index - delay,
                        "query_tokens": queries.shape[0],
                        **reconstruction_metrics(
                            queries,
                            reconstruction,
                            coverage_threshold=coverage_threshold,
                        ),
                    }
                )
    return rows, accounting_rows


def discover_inputs(root: Path, limit: int) -> list[Path]:
    inputs = sorted(root.resolve().glob("*/hidden.npz"))
    if limit > 0:
        inputs = inputs[:limit]
    if not inputs:
        raise FileNotFoundError(f"no immediate-child hidden.npz under {root}")
    return inputs


def aggregate_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    metrics = (
        "mean_cosine",
        "median_cosine",
        "mean_relative_l2",
        "coverage",
    )
    keys = ("category", "layer", "mode", "method", "capacity", "delay")
    expanded = list(rows)
    expanded.extend({**row, "category": "__all__"} for row in rows)
    grouped: dict[
        tuple[object, ...],
        dict[str, object],
    ] = defaultdict(
        lambda: {
            "runs": set(),
            **{metric: [] for metric in metrics},
        }
    )
    for row in expanded:
        key = tuple(row[name] for name in keys)
        grouped[key]["runs"].add(str(row["run"]))
        for metric in metrics:
            grouped[key][metric].append(float(row[metric]))
    output: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(x) for x in item)):
        record = {name: value for name, value in zip(keys, key, strict=True)}
        values = grouped[key]
        record["run_count"] = len(values["runs"])
        record["observation_count"] = len(values["mean_cosine"])
        for metric in metrics:
            samples = np.asarray(values[metric], dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(samples))
            record[f"{metric}_std"] = float(np.std(samples))
        output.append(record)

    recent_lookup = {
        (
            row["category"],
            row["layer"],
            row["mode"],
            row["capacity"],
            row["delay"],
        ): float(row["mean_cosine_mean"])
        for row in output
        if row["method"] == "recent_window"
    }
    for row in output:
        baseline = recent_lookup[
            (
                row["category"],
                row["layer"],
                row["mode"],
                row["capacity"],
                row["delay"],
            )
        ]
        row["cosine_gain_vs_recent"] = (
            float(row["mean_cosine_mean"]) - baseline
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run_checkpoint(
    run_dir: Path,
    *,
    fingerprint: str,
    input_path: Path,
    input_index: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    summary_path = run_dir / "run_summary.json"
    metrics_path = run_dir / "metrics.json"
    accounting_path = run_dir / "accounting.json"
    if not (
        summary_path.exists()
        and metrics_path.exists()
        and accounting_path.exists()
    ):
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stat = input_path.stat()
        if (
            summary.get("configuration_fingerprint") != fingerprint
            or int(summary.get("input_index", -1)) != input_index
            or int(summary.get("input_size", -1)) != stat.st_size
            or int(summary.get("input_mtime_ns", -1)) != stat.st_mtime_ns
        ):
            return None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        accounting = json.loads(
            accounting_path.read_text(encoding="utf-8")
        )
        if not isinstance(metrics, list) or not isinstance(accounting, list):
            return None
        return metrics, accounting
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_run_checkpoint(
    run_dir: Path,
    *,
    fingerprint: str,
    input_path: Path,
    input_index: int,
    run: str,
    category: str,
    layers: list[int],
    metrics: list[dict[str, object]],
    accounting: list[dict[str, object]],
    seconds: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, separators=(",", ":")),
        encoding="utf-8",
    )
    (run_dir / "accounting.json").write_text(
        json.dumps(accounting, separators=(",", ":")),
        encoding="utf-8",
    )
    stat = input_path.stat()
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "run": run,
                "category": category,
                "layers": layers,
                "input": str(input_path),
                "input_index": input_index,
                "input_size": stat.st_size,
                "input_mtime_ns": stat.st_mtime_ns,
                "configuration_fingerprint": fingerprint,
                "metric_rows": len(metrics),
                "accounting_rows": len(accounting),
                "seconds": seconds,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def plot_retention_curves(
    rows: list[dict[str, object]],
    path: Path,
    *,
    capacity: int,
    mode: str,
) -> bool:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["category"] == "__all__"
        and int(row["capacity"]) == capacity
        and row["mode"] == mode
    ]
    if not selected:
        return False
    layers = sorted({int(row["layer"]) for row in selected})
    fig, axes = plt.subplots(
        1,
        len(layers),
        figsize=(4.2 * len(layers), 3.9),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    colors = {
        "recent_window": "#6c757d",
        "uniform_reservoir": "#457b9d",
        "adaptive_slots": "#e76f51",
        "oja_subspace": "#2a9d8f",
        "instant_adaptive": "#f4a261",
        "instant_oja": "#6a4c93",
    }
    present_methods = [
        method
        for method in METHODS
        if any(row["method"] == method for row in selected)
    ]
    for axis, layer in zip(axes, layers, strict=True):
        for method in present_methods:
            method_rows = [
                row
                for row in selected
                if int(row["layer"]) == layer and row["method"] == method
            ]
            method_rows.sort(key=lambda row: int(row["delay"]))
            if not method_rows:
                continue
            delays = np.asarray(
                [int(row["delay"]) for row in method_rows],
                dtype=np.int64,
            )
            means = np.asarray(
                [float(row["mean_cosine_mean"]) for row in method_rows]
            )
            stds = np.asarray(
                [float(row["mean_cosine_std"]) for row in method_rows]
            )
            axis.plot(
                delays,
                means,
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=method.replace("_", " "),
            )
            axis.fill_between(
                delays,
                np.clip(means - stds, -1, 1),
                np.clip(means + stds, -1, 1),
                color=colors[method],
                alpha=0.10,
                linewidth=0,
            )
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel("Query delay (hidden frames)")
        axis.set_ylim(-0.05, 1.03)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean reconstruction cosine")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=len(present_methods),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"Matched-payload retention, capacity={capacity}, mode={mode}"
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def plot_category_gain(
    rows: list[dict[str, object]],
    path: Path,
    *,
    capacity: int,
    mode: str,
) -> bool:
    import matplotlib.pyplot as plt

    candidate_methods = [
        method
        for method in (
            "adaptive_slots",
            "oja_subspace",
            "instant_adaptive",
            "instant_oja",
        )
        if any(row["method"] == method for row in rows)
    ]
    selected = [
        row
        for row in rows
        if row["category"] != "__all__"
        and int(row["capacity"]) == capacity
        and row["mode"] == mode
        and row["method"] in candidate_methods
    ]
    if not selected:
        return False
    max_delay = max(int(row["delay"]) for row in selected)
    selected = [row for row in selected if int(row["delay"]) == max_delay]
    categories = sorted({str(row["category"]) for row in selected})
    layers = sorted({int(row["layer"]) for row in selected})
    columns = min(2, len(candidate_methods))
    rows_count = int(np.ceil(len(candidate_methods) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(5.8 * columns, 4.6 * rows_count),
        sharey=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, method in zip(axes, candidate_methods, strict=False):
        matrix = np.full((len(categories), len(layers)), np.nan)
        for category_index, category in enumerate(categories):
            for layer_index, layer in enumerate(layers):
                match = next(
                    (
                        row
                        for row in selected
                        if row["method"] == method
                        and row["category"] == category
                        and int(row["layer"]) == layer
                    ),
                    None,
                )
                if match is not None:
                    matrix[category_index, layer_index] = float(
                        match["cosine_gain_vs_recent"]
                    )
        image = axis.imshow(
            matrix,
            cmap="RdYlGn",
            vmin=-0.20,
            vmax=0.20,
            aspect="auto",
        )
        axis.set_xticks(range(len(layers)), [f"L{layer}" for layer in layers])
        axis.set_yticks(range(len(categories)), categories)
        axis.set_title(method.replace("_", " "))
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes[len(candidate_methods) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Long-delay cosine gain versus recent window "
        f"(delay={max_delay}, capacity={capacity})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def plot_budget_frontier(
    rows: list[dict[str, object]],
    accounting: list[dict[str, object]],
    path: Path,
    *,
    mode: str,
) -> bool:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["category"] == "__all__" and row["mode"] == mode
    ]
    if not selected:
        return False
    max_delay = max(int(row["delay"]) for row in selected)
    selected = [row for row in selected if int(row["delay"]) == max_delay]
    byte_lookup = {
        (
            int(row["layer"]),
            str(row["mode"]),
            str(row["method"]),
            int(row["capacity"]),
        ): int(row["total_state_bytes"])
        for row in accounting
    }
    layers = sorted({int(row["layer"]) for row in selected})
    colors = {
        "recent_window": "#6c757d",
        "uniform_reservoir": "#457b9d",
        "adaptive_slots": "#e76f51",
        "oja_subspace": "#2a9d8f",
        "instant_adaptive": "#f4a261",
        "instant_oja": "#6a4c93",
    }
    present_methods = [
        method
        for method in METHODS
        if any(row["method"] == method for row in selected)
    ]
    fig, axes = plt.subplots(
        1,
        len(layers),
        figsize=(4.2 * len(layers), 3.9),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, layer in zip(axes, layers, strict=True):
        for method in present_methods:
            method_rows = [
                row
                for row in selected
                if int(row["layer"]) == layer and row["method"] == method
            ]
            method_rows.sort(key=lambda row: int(row["capacity"]))
            x = [
                byte_lookup[
                    (
                        layer,
                        mode,
                        method,
                        int(row["capacity"]),
                    )
                ]
                / 1024
                for row in method_rows
            ]
            y = [float(row["mean_cosine_mean"]) for row in method_rows]
            axis.plot(
                x,
                y,
                marker="o",
                color=colors[method],
                label=method.replace("_", " "),
            )
        axis.set_xscale("log", base=2)
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel("Retained state (KiB)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(f"Mean cosine at delay {max_delay}")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=len(present_methods),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("Matched-byte memory frontier")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    if args.pool_rows <= 0 or args.pool_cols <= 0:
        raise ValueError("pool dimensions must be positive")
    if args.instant_frames <= 0:
        raise ValueError("instant_frames must be positive")
    if args.storage_bits <= 0 or args.storage_bits % 8 != 0:
        raise ValueError("storage_bits must be a positive multiple of 8")
    if not 0 < args.coverage_threshold <= 1:
        raise ValueError("coverage_threshold must be in (0,1]")
    capacities = parse_int_list(args.capacities)
    if capacities[0] == 0:
        raise ValueError("capacities must be positive")
    delays = parse_int_list(args.delays)
    modes = parse_str_list(args.modes)
    methods = list(dict.fromkeys(parse_str_list(args.methods)))
    unknown_methods = set(methods) - set(METHODS)
    if unknown_methods:
        raise ValueError(f"unknown methods: {sorted(unknown_methods)}")
    if "recent_window" not in methods:
        raise ValueError("methods must include recent_window as the baseline")
    requested_layers = (
        parse_int_list(args.layers) if args.layers.strip() else None
    )
    configuration = {
        "requested_layers": requested_layers,
        "capacities": capacities,
        "delays": delays,
        "modes": modes,
        "pool_rows": args.pool_rows,
        "pool_cols": args.pool_cols,
        "instant_frames": args.instant_frames,
        "storage_bits": args.storage_bits,
        "seed": args.seed,
        "oja_lr": args.oja_lr,
        "slot_min_lr": args.slot_min_lr,
        "slot_replace_similarity": args.slot_replace_similarity,
        "coverage_threshold": args.coverage_threshold,
        "methods": methods,
    }
    fingerprint = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    inputs = discover_inputs(args.root, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.out_dir / "per_run"
    metric_rows: list[dict[str, object]] = []
    accounting_rows: list[dict[str, object]] = []
    input_rows = []
    for input_index, path in enumerate(inputs):
        started = time.monotonic()
        metadata_path = path.with_name("metadata.json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        run = path.parent.name
        category = str(metadata.get("category") or "unlabeled")
        metadata_layers = [
            int(layer) for layer in metadata.get("layers", [])
        ]
        if metadata_layers:
            available_layers = sorted(metadata_layers)
        else:
            with np.load(path) as archive:
                available_layers = sorted(
                    int(key.rsplit("_", 1)[-1])
                    for key in archive.files
                    if key.startswith("hidden_layer_")
                )
        layers = (
            [layer for layer in requested_layers if layer in available_layers]
            if requested_layers is not None
            else available_layers
        )
        if not layers:
            raise ValueError(f"no requested layers found in {path}")
        input_rows.append(
            {
                "run": run,
                "category": category,
                "npz": str(path),
                "metadata": (
                    str(metadata_path) if metadata_path.exists() else None
                ),
                "layers": layers,
                "frame_indices": metadata.get("frame_indices"),
            }
        )
        checkpoint_dir = checkpoint_root / run
        checkpoint = (
            None
            if args.overwrite
            else load_run_checkpoint(
                checkpoint_dir,
                fingerprint=fingerprint,
                input_path=path,
                input_index=input_index,
            )
        )
        if checkpoint is not None:
            run_metrics, run_accounting = checkpoint
            metric_rows.extend(run_metrics)
            accounting_rows.extend(run_accounting)
            print(
                json.dumps(
                    {
                        "position": input_index + 1,
                        "total": len(inputs),
                        "run": run,
                        "category": category,
                        "status": "resumed_checkpoint",
                        "metric_rows": len(run_metrics),
                    }
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "position": input_index + 1,
                    "total": len(inputs),
                    "run": run,
                    "category": category,
                    "status": "started",
                }
            ),
            flush=True,
        )
        run_metrics: list[dict[str, object]] = []
        run_accounting: list[dict[str, object]] = []
        with np.load(path) as archive:
            for layer in layers:
                raw_sequence = archive[f"hidden_layer_{layer}"]
                for mode_index, mode in enumerate(modes):
                    sequence = prepare_sequence(
                        raw_sequence,
                        rows=args.pool_rows,
                        cols=args.pool_cols,
                        mode=mode,
                    )
                    for capacity in capacities:
                        rows, state_rows = evaluate_sequence(
                            sequence,
                            run=run,
                            category=category,
                            layer=layer,
                            mode=mode,
                            capacity=capacity,
                            delays=delays,
                            storage_bits=args.storage_bits,
                            seed=(
                                args.seed
                                + 100003 * input_index
                                + 1009 * layer
                                + 101 * mode_index
                                + capacity
                            ),
                            oja_lr=args.oja_lr,
                            slot_min_lr=args.slot_min_lr,
                            slot_replace_similarity=(
                                args.slot_replace_similarity
                            ),
                            coverage_threshold=args.coverage_threshold,
                            methods=methods,
                            instant_frames=args.instant_frames,
                        )
                        run_metrics.extend(rows)
                        run_accounting.extend(state_rows)
        seconds = time.monotonic() - started
        write_run_checkpoint(
            checkpoint_dir,
            fingerprint=fingerprint,
            input_path=path,
            input_index=input_index,
            run=run,
            category=category,
            layers=layers,
            metrics=run_metrics,
            accounting=run_accounting,
            seconds=seconds,
        )
        metric_rows.extend(run_metrics)
        accounting_rows.extend(run_accounting)
        print(
            json.dumps(
                {
                    "position": input_index + 1,
                    "total": len(inputs),
                    "run": run,
                    "category": category,
                    "status": "completed",
                    "seconds": round(seconds, 2),
                    "metric_rows": len(run_metrics),
                }
            ),
            flush=True,
        )
    aggregate = aggregate_rows(metric_rows)
    unique_accounting = []
    seen_accounting = set()
    for row in accounting_rows:
        key = (
            row["layer"],
            row["mode"],
            row["method"],
            row["capacity"],
            row["hidden_dim"],
            row["storage_bits"],
            row["instant_capacity"],
        )
        if key in seen_accounting:
            continue
        seen_accounting.add(key)
        unique_accounting.append(
            {
                key_name: row[key_name]
                for key_name in (
                    "layer",
                    "mode",
                    "method",
                    "capacity",
                    "hidden_dim",
                    "storage_bits",
                    "instant_capacity",
                    "payload_bytes",
                    "metadata_bytes",
                    "total_state_bytes",
                    "read_flops_per_query_token",
                    "update_flops_per_input_token",
                )
            }
        )

    write_csv(args.out_dir / "memory_query_metrics.csv", metric_rows)
    write_csv(args.out_dir / "memory_summary.csv", aggregate)
    write_csv(
        args.out_dir / "memory_state_accounting.csv",
        unique_accounting,
    )
    figures = []
    if not args.skip_plots:
        retention_name = "memory_retention_curves.png"
        if plot_retention_curves(
            aggregate,
            args.out_dir / retention_name,
            capacity=args.plot_capacity,
            mode="frame_centered_unit",
        ):
            figures.extend(
                [retention_name, "memory_retention_curves.pdf"]
            )
        category_name = "memory_category_gain.png"
        if plot_category_gain(
            aggregate,
            args.out_dir / category_name,
            capacity=args.plot_capacity,
            mode="frame_centered_unit",
        ):
            figures.extend([category_name, "memory_category_gain.pdf"])
        frontier_name = "memory_budget_frontier.png"
        if plot_budget_frontier(
            aggregate,
            unique_accounting,
            args.out_dir / frontier_name,
            mode="frame_centered_unit",
        ):
            figures.extend([frontier_name, "memory_budget_frontier.pdf"])

    summary = {
        "inputs": input_rows,
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "methods": methods,
        "metric_rows": len(metric_rows),
        "aggregate_rows": len(aggregate),
        "figures": figures,
        "scope": (
            "Representation-level matched-state diagnostic. A query is a "
            "held-out delayed region vector; each bounded memory returns an "
            "approximation. This does not substitute for downstream online "
            "QA or language-conditioned retrieval."
        ),
    }
    (args.out_dir / "memory_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "inputs": len(inputs),
                "metric_rows": len(metric_rows),
                "aggregate_rows": len(aggregate),
                "figures": figures,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
