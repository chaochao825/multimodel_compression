from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", type=Path)
    source.add_argument(
        "--root",
        type=Path,
        help="Directory whose immediate children contain hidden.npz files.",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path)
    return parser.parse_args()


def discover_inputs(npz: Path | None, root: Path | None) -> list[Path]:
    if npz is not None:
        return [npz.resolve()]
    assert root is not None
    inputs = sorted(path.resolve() for path in root.glob("*/hidden.npz"))
    if not inputs:
        raise FileNotFoundError(f"no immediate-child hidden.npz under {root}")
    return inputs


def mean_pairwise_cosine(tokens: np.ndarray) -> float:
    norms = np.linalg.norm(tokens, axis=1)
    valid = norms > 1e-12
    unit = tokens[valid] / norms[valid, None]
    count = unit.shape[0]
    if count < 2:
        return float("nan")
    summed = unit.sum(axis=0)
    return float((summed @ summed - count) / (count * (count - 1)))


def mean_row_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    valid = denominator > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(
        np.mean(
            np.sum(left[valid] * right[valid], axis=1)
            / denominator[valid]
        )
    )


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float((left @ right) / denominator)


def centered_spectrum(centered: np.ndarray) -> tuple[float, float]:
    singular_values = np.linalg.svd(
        centered,
        compute_uv=False,
        full_matrices=False,
    )
    energy = singular_values**2
    total = float(energy.sum())
    if total <= 1e-24:
        return 0.0, 0.0
    probability = energy / total
    nonzero = probability[probability > 0]
    effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    return float(probability[0]), effective_rank


def frame_rows(
    sequence: np.ndarray,
    *,
    run: str,
    layer: int,
) -> list[dict[str, object]]:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            f"hidden layer {layer} must have [T,H,W,D], got {values.shape}"
        )
    output: list[dict[str, object]] = []
    previous_tokens: np.ndarray | None = None
    previous_mean: np.ndarray | None = None
    for frame_index, frame in enumerate(values):
        tokens = frame.reshape(-1, frame.shape[-1])
        token_norms = np.linalg.norm(tokens, axis=1)
        mean_vector = tokens.mean(axis=0)
        centered = tokens - mean_vector
        raw_energy = float(np.mean(np.sum(tokens**2, axis=1)))
        centered_energy = float(np.mean(np.sum(centered**2, axis=1)))
        top1_energy, effective_rank = centered_spectrum(centered)
        row: dict[str, object] = {
            "run": run,
            "layer": layer,
            "frame": frame_index,
            "tokens": tokens.shape[0],
            "hidden_dim": tokens.shape[1],
            "token_norm_mean": float(np.mean(token_norms)),
            "token_norm_std": float(np.std(token_norms)),
            "token_norm_cv": float(
                np.std(token_norms) / max(np.mean(token_norms), 1e-12)
            ),
            "mean_direction_energy_fraction": float(
                np.clip(1.0 - centered_energy / max(raw_energy, 1e-24), 0, 1)
            ),
            "spatial_centered_energy_fraction": float(
                centered_energy / max(raw_energy, 1e-24)
            ),
            "centered_top1_energy_fraction": top1_energy,
            "centered_effective_rank": effective_rank,
            "within_frame_pairwise_cosine": mean_pairwise_cosine(tokens),
            "same_position_temporal_cosine": None,
            "frame_mean_temporal_cosine": None,
            "relative_temporal_change": None,
        }
        if previous_tokens is not None and previous_mean is not None:
            row["same_position_temporal_cosine"] = mean_row_cosine(
                previous_tokens,
                tokens,
            )
            row["frame_mean_temporal_cosine"] = vector_cosine(
                previous_mean,
                mean_vector,
            )
            row["relative_temporal_change"] = float(
                np.linalg.norm(tokens - previous_tokens)
                / max(np.linalg.norm(previous_tokens), 1e-12)
            )
        output.append(row)
        previous_tokens = tokens
        previous_mean = mean_vector
    return output


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "token_norm_mean",
        "token_norm_cv",
        "mean_direction_energy_fraction",
        "spatial_centered_energy_fraction",
        "centered_top1_energy_fraction",
        "centered_effective_rank",
        "within_frame_pairwise_cosine",
        "same_position_temporal_cosine",
        "frame_mean_temporal_cosine",
        "relative_temporal_change",
    ]
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in metrics}
    )
    for row in rows:
        key = (str(row["run"]), int(row["layer"]))
        for metric in metrics:
            value = row.get(metric)
            if value is not None and np.isfinite(float(value)):
                grouped[key][metric].append(float(value))
    output = []
    for (run, layer), values in sorted(grouped.items()):
        record: dict[str, object] = {"run": run, "layer": layer}
        for metric in metrics:
            samples = values[metric]
            record[f"{metric}_mean"] = (
                float(np.mean(samples)) if samples else None
            )
            record[f"{metric}_std"] = (
                float(np.std(samples)) if samples else None
            )
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    runs = sorted({str(row["run"]) for row in rows})
    colors = ["#264653", "#2a9d8f", "#e76f51", "#8d5a97"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    panels = [
        (
            "centered_top1_energy_fraction_mean",
            "Centered top-1 energy",
            (0.0, 1.03),
            False,
        ),
        (
            "centered_effective_rank_mean",
            "Centered effective rank",
            None,
            True,
        ),
        (
            "within_frame_pairwise_cosine_mean",
            "Mean token-pair cosine",
            (0.0, 1.03),
            False,
        ),
    ]
    for axis, (metric, ylabel, ylim, log_scale) in zip(
        axes,
        panels,
        strict=True,
    ):
        for color, run in zip(colors, runs, strict=False):
            selected = [row for row in rows if row["run"] == run]
            selected.sort(key=lambda row: int(row["layer"]))
            axis.plot(
                [int(row["layer"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                color=color,
                label=run,
            )
        if log_scale:
            axis.set_yscale("log")
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.set_xlabel("Visual block")
        axis.set_ylabel(ylabel)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=len(runs),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    inputs = discover_inputs(args.npz, args.root)
    rows: list[dict[str, object]] = []
    input_metadata = []
    for path in inputs:
        run = path.parent.name
        metadata_path = path.with_name("metadata.json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        input_metadata.append(
            {
                "run": run,
                "npz": str(path),
                "metadata": str(metadata_path) if metadata_path.exists() else None,
                "model_dir": metadata.get("model_dir"),
                "video": metadata.get("video"),
                "frame_indices": metadata.get("frame_indices"),
            }
        )
        with np.load(path) as archive:
            layer_keys = sorted(
                (
                    key
                    for key in archive.files
                    if key.startswith("hidden_layer_")
                ),
                key=lambda key: int(key.rsplit("_", 1)[-1]),
            )
            for key in layer_keys:
                layer = int(key.rsplit("_", 1)[-1])
                rows.extend(
                    frame_rows(
                        archive[key],
                        run=run,
                        layer=layer,
                    )
                )
    summary = aggregate_rows(rows)
    write_csv(args.out_csv, rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "inputs": input_metadata,
                "summary": summary,
                "scope": (
                    "Diagnostics separate common mean direction, centered "
                    "spatial rank, token similarity, and temporal stability."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if args.out_plot is not None:
        plot_summary(args.out_plot, summary)
    print(
        json.dumps(
            {
                "inputs": len(inputs),
                "frame_rows": len(rows),
                "summary_rows": len(summary),
                "out_csv": str(args.out_csv.resolve()),
                "out_json": str(args.out_json.resolve()),
                "out_plot": (
                    str(args.out_plot.resolve())
                    if args.out_plot is not None
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
