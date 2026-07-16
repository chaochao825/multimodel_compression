from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory whose immediate children contain analysis/analysis_summary.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="qwen")
    parser.add_argument("--display-name", default="Qwen3-VL")
    parser.add_argument("--analysis-name", default="analysis")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_runs(
    root: Path,
    analysis_name: str,
) -> list[tuple[str, Path]]:
    runs = []
    for child in sorted(root.iterdir()):
        summary = child / analysis_name / "analysis_summary.json"
        if summary.exists():
            runs.append((child.name, summary))
    if not runs:
        raise FileNotFoundError(
            f"no {analysis_name}/analysis_summary.json files found under {root}"
        )
    return runs


def collect(root: Path, analysis_name: str) -> dict[str, object]:
    rank_rows: list[dict[str, object]] = []
    transport_rows: list[dict[str, object]] = []
    subspace_raw: list[dict[str, object]] = []
    residual_raw: list[dict[str, object]] = []
    run_metadata: list[dict[str, object]] = []

    for run_name, summary_path in discover_runs(root, analysis_name):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata_path = summary_path.parent.parent / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        category = (
            summary.get("category")
            or metadata.get("category")
            or "unlabeled"
        )
        run_metadata.append(
            {
                "run": run_name,
                "category": category,
                "video": summary.get("video"),
                "model_dir": summary.get("model_dir"),
                "frame_indices": summary.get("frame_indices"),
                "grid_thw": summary.get("grid_thw"),
            }
        )
        for layer, layer_data in summary["layers"].items():
            for alignment in (
                "unaligned_temporal",
                "pixel_aligned_temporal",
                "hidden_aligned_temporal",
                "unaligned_temporal_change",
                "pixel_aligned_temporal_change",
                "hidden_aligned_temporal_change",
                "history_feature_subspace",
                "history_feature_subspace_centered",
                "history_feature_subspace_token_normalized",
                "frame_spatial_raw",
                "frame_spatial_centered",
                "frame_spatial_token_normalized",
            ):
                metrics = layer_data["rank_summary"][alignment]
                for rank, energy in metrics["energy_at_rank"].items():
                    rank_rows.append(
                        {
                            "run": run_name,
                            "category": category,
                            "video": summary.get("video"),
                            "layer": int(layer),
                            "alignment": alignment,
                            "rank": int(rank),
                            "energy_ratio": float(energy),
                            "effective_rank": float(metrics["effective_rank"]),
                            "stable_rank": float(metrics["stable_rank"]),
                        }
                    )
            for method, metrics in layer_data["method_summary"].items():
                transport_rows.append(
                    {
                        "run": run_name,
                        "category": category,
                        "video": summary.get("video"),
                        "layer": int(layer),
                        "method": method,
                        "mean_error": metrics["mean_error"],
                        "stable_error": metrics["stable_error"],
                        "event_like_error": metrics["event_like_error"],
                        "mean_centered_error": metrics[
                            "mean_centered_error"
                        ],
                        "stable_centered_error": metrics[
                            "stable_centered_error"
                        ],
                        "event_like_centered_error": metrics[
                            "event_like_centered_error"
                        ],
                    }
                )

        for row in read_csv(
            summary_path.parent / "causal_subspace_metrics.csv"
        ):
            subspace_raw.append(
                {
                    "run": run_name,
                    "category": category,
                    "layer": int(row["layer"]),
                    "frame": int(row["frame"]),
                    "rank": int(row["rank"]),
                    "mode": row["mode"],
                    "relative_projection_error": float(
                        row["relative_projection_error"]
                    ),
                    "raw_reconstruction_error": float(
                        row["raw_reconstruction_error"]
                    ),
                }
            )
        for row in read_csv(summary_path.parent / "residual_metrics.csv"):
            residual_raw.append(
                {
                    "run": run_name,
                    "category": category,
                    "layer": int(row["layer"]),
                    "frame": int(row["frame"]),
                    "method": row["method"],
                    "fraction": float(row["fraction"]),
                    "energy_ratio": float(row["energy_ratio"]),
                    "pixel_change_recall": float(row["pixel_change_recall"]),
                    "gini": float(row["gini"]),
                }
            )

    subspace_rows = aggregate_rows(
        subspace_raw,
        keys=("layer", "rank", "mode"),
        values=(
            "relative_projection_error",
            "raw_reconstruction_error",
        ),
    )
    residual_rows = aggregate_rows(
        residual_raw,
        keys=("layer", "method", "fraction"),
        values=("energy_ratio", "pixel_change_recall", "gini"),
    )
    rank_aggregate = aggregate_rows(
        rank_rows,
        keys=("layer", "alignment", "rank"),
        values=("energy_ratio", "effective_rank", "stable_rank"),
    )
    transport_aggregate = aggregate_rows(
        transport_rows,
        keys=("layer", "method"),
        values=(
            "mean_error",
            "stable_error",
            "event_like_error",
            "mean_centered_error",
            "stable_centered_error",
            "event_like_centered_error",
        ),
    )
    rank_by_category = aggregate_rows(
        rank_rows,
        keys=("category", "layer", "alignment", "rank"),
        values=("energy_ratio", "effective_rank", "stable_rank"),
    )
    transport_by_category = aggregate_rows(
        transport_rows,
        keys=("category", "layer", "method"),
        values=(
            "mean_error",
            "stable_error",
            "event_like_error",
            "mean_centered_error",
            "stable_centered_error",
            "event_like_centered_error",
        ),
    )
    subspace_by_category = aggregate_rows(
        subspace_raw,
        keys=("category", "layer", "rank", "mode"),
        values=(
            "relative_projection_error",
            "raw_reconstruction_error",
        ),
    )
    residual_by_category = aggregate_rows(
        residual_raw,
        keys=("category", "layer", "method", "fraction"),
        values=("energy_ratio", "pixel_change_recall", "gini"),
    )
    return {
        "run_metadata": run_metadata,
        "rank_raw": rank_rows,
        "rank_aggregate": rank_aggregate,
        "transport_raw": transport_rows,
        "transport_aggregate": transport_aggregate,
        "subspace_raw": subspace_raw,
        "subspace_aggregate": subspace_rows,
        "residual_raw": residual_raw,
        "residual_aggregate": residual_rows,
        "rank_by_category": rank_by_category,
        "transport_by_category": transport_by_category,
        "subspace_by_category": subspace_by_category,
        "residual_by_category": residual_by_category,
    }


def aggregate_rows(
    rows: list[dict[str, object]],
    *,
    keys: tuple[str, ...],
    values: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(
        lambda: {value: [] for value in values}
    )
    for row in rows:
        key = tuple(row[name] for name in keys)
        for value in values:
            raw = row.get(value)
            if raw is not None:
                grouped[key][value].append(float(raw))
    output = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        record = {name: value for name, value in zip(keys, key, strict=True)}
        record["count"] = max(
            (len(grouped[key][value]) for value in values),
            default=0,
        )
        for value in values:
            samples = grouped[key][value]
            record[f"{value}_mean"] = (
                float(np.mean(samples)) if samples else None
            )
            record[f"{value}_std"] = (
                float(np.std(samples)) if samples else None
            )
        output.append(record)
    return output


def plot_temporal_spectrum(
    rows: list[dict[str, object]],
    path: Path,
    *,
    display_name: str,
) -> None:
    layers = sorted({int(row["layer"]) for row in rows})
    alignments = [
        "unaligned_temporal_change",
        "pixel_aligned_temporal_change",
        "hidden_aligned_temporal_change",
    ]
    colors = {
        "unaligned_temporal_change": "#6c757d",
        "pixel_aligned_temporal_change": "#2a9d8f",
        "hidden_aligned_temporal_change": "#e76f51",
    }
    labels = {
        "unaligned_temporal_change": "unaligned change",
        "pixel_aligned_temporal_change": "pixel-aligned change",
        "hidden_aligned_temporal_change": "hidden-aligned change",
    }
    fig, axes = plt.subplots(
        1,
        len(layers),
        figsize=(4.2 * len(layers), 3.8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, layer in zip(axes, layers, strict=True):
        for alignment in alignments:
            selected = [
                row
                for row in rows
                if int(row["layer"]) == layer
                and row["alignment"] == alignment
            ]
            selected.sort(key=lambda row: int(row["rank"]))
            axis.plot(
                [int(row["rank"]) for row in selected],
                [float(row["energy_ratio_mean"]) for row in selected],
                marker="o",
                linewidth=1.8,
                color=colors[alignment],
                label=labels[alignment],
            )
        axis.axhline(0.70, color="#d1495b", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_ylim(0, 1.03)
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel("Rank")
    axes[0].set_ylabel("Explained temporal energy")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        f"{display_name} temporal spectrum across analyzed segments"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_state_spectrum(
    rows: list[dict[str, object]],
    path: Path,
    *,
    display_name: str,
) -> None:
    layers = sorted({int(row["layer"]) for row in rows})
    alignments = [
        "frame_spatial_centered",
        "history_feature_subspace_centered",
        "history_feature_subspace_token_normalized",
    ]
    colors = {
        "frame_spatial_centered": "#264653",
        "history_feature_subspace_centered": "#2a9d8f",
        "history_feature_subspace_token_normalized": "#e76f51",
    }
    labels = {
        "frame_spatial_centered": "current-frame spatial",
        "history_feature_subspace_centered": "history feature, centered",
        "history_feature_subspace_token_normalized": (
            "history feature, token-normalized"
        ),
    }
    fig, axes = plt.subplots(
        1,
        len(layers),
        figsize=(4.2 * len(layers), 3.9),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, layer in zip(axes, layers, strict=True):
        for alignment in alignments:
            selected = [
                row
                for row in rows
                if int(row["layer"]) == layer
                and row["alignment"] == alignment
            ]
            selected.sort(key=lambda row: int(row["rank"]))
            ranks = np.asarray(
                [int(row["rank"]) for row in selected],
                dtype=np.int64,
            )
            means = np.asarray(
                [float(row["energy_ratio_mean"]) for row in selected],
                dtype=np.float64,
            )
            stds = np.asarray(
                [float(row["energy_ratio_std"]) for row in selected],
                dtype=np.float64,
            )
            axis.plot(
                ranks,
                means,
                marker="o",
                linewidth=1.8,
                color=colors[alignment],
                label=labels[alignment],
            )
            axis.fill_between(
                ranks,
                np.clip(means - stds, 0.0, 1.0),
                np.clip(means + stds, 0.0, 1.0),
                color=colors[alignment],
                alpha=0.12,
                linewidth=0,
            )
        axis.axhline(
            0.70,
            color="#d1495b",
            linestyle="--",
            linewidth=1,
            label="70% preregistered gate" if layer == layers[-1] else None,
        )
        axis.set_xscale("log", base=2)
        axis.set_ylim(0, 1.03)
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel("Rank")
    axes[0].set_ylabel("Explained feature energy")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        f"{display_name} persistent-state spectrum across analyzed segments"
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_transport(rows: list[dict[str, object]], path: Path) -> None:
    methods = [
        "identity",
        "pixel_shift_zero",
        "optical_flow_warp",
        "hidden_shift_cyclic",
        "local_bttb_pair",
        "local_bccb_pair",
        "local_bttb_causal",
        "local_bccb_causal",
        "global_bccb_causal",
    ]
    layers = sorted({int(row["layer"]) for row in rows})
    fig, axes = plt.subplots(
        1,
        len(layers),
        figsize=(4.4 * len(layers), 4.2),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, layer in zip(axes, layers, strict=True):
        lookup = {
            str(row["method"]): row
            for row in rows
            if int(row["layer"]) == layer
        }
        present = [method for method in methods if method in lookup]
        stable = [
            float(lookup[method]["stable_centered_error_mean"])
            if lookup[method]["stable_centered_error_mean"] is not None
            else np.nan
            for method in present
        ]
        event_like = [
            float(lookup[method]["event_like_centered_error_mean"])
            if lookup[method]["event_like_centered_error_mean"] is not None
            else np.nan
            for method in present
        ]
        x = np.arange(len(present))
        axis.bar(
            x - 0.19,
            stable,
            0.38,
            color="#287271",
            label="stable half",
        )
        axis.bar(
            x + 0.19,
            event_like,
            0.38,
            color="#e07a5f",
            label="top-change quartile",
        )
        axis.set_xticks(x, present, rotation=55, ha="right", fontsize=8)
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel("Predictor")
    axes[0].set_ylabel("Relative activation error")
    axes[-1].legend(frameon=False)
    fig.suptitle("Pair-oracle and causal transport diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_subspace(rows: list[dict[str, object]], path: Path) -> None:
    rows = [row for row in rows if row["mode"] == "history_centered"]
    layers = sorted({int(row["layer"]) for row in rows})
    fig, axis = plt.subplots(figsize=(6.8, 4.5))
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#8d5a97"]
    for color, layer in zip(palette, layers, strict=False):
        selected = [row for row in rows if int(row["layer"]) == layer]
        selected.sort(key=lambda row: int(row["rank"]))
        axis.plot(
            [int(row["rank"]) for row in selected],
            [
                float(row["relative_projection_error_mean"])
                for row in selected
            ],
            marker="o",
            color=color,
            label=f"Layer {layer}",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Causal feature-subspace rank")
    axis.set_ylabel("Centered projection error")
    axis.set_title("History-only centered subspace preservation")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residual(rows: list[dict[str, object]], path: Path) -> None:
    selected = [
        row for row in rows if abs(float(row["fraction"]) - 0.10) < 1e-9
    ]
    methods = sorted({str(row["method"]) for row in selected})
    layers = sorted({int(row["layer"]) for row in selected})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for layer in layers:
        energy = []
        recall = []
        for method in methods:
            match = next(
                row
                for row in selected
                if int(row["layer"]) == layer and row["method"] == method
            )
            energy.append(float(match["energy_ratio_mean"]))
            recall.append(float(match["pixel_change_recall_mean"]))
        axes[0].plot(methods, energy, marker="o", label=f"L{layer}")
        axes[1].plot(methods, recall, marker="s", label=f"L{layer}")
    for axis, title, ylabel in (
        (axes[0], "Top-10% residual energy", "Captured energy"),
        (axes[1], "Pixel-change proxy recall", "Recall"),
    ):
        axis.set_ylim(0, 1.03)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.legend(frameon=False)
    fig.suptitle("Residual block concentration")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_category_probe_summary(
    rank_rows: list[dict[str, object]],
    subspace_rows: list[dict[str, object]],
    transport_rows: list[dict[str, object]],
    residual_rows: list[dict[str, object]],
    path: Path,
) -> bool:
    categories = sorted(
        {
            str(row["category"])
            for row in rank_rows
            if row["category"] != "unlabeled"
        }
    )
    if not categories:
        return False
    layers = sorted({int(row["layer"]) for row in rank_rows})
    category_counts = {
        category: max(
            int(row["count"])
            for row in rank_rows
            if row["category"] == category
            and row["alignment"]
            == "history_feature_subspace_token_normalized"
            and int(row["rank"]) == 32
        )
        for category in categories
    }
    category_labels = [
        f"{category} (n={category_counts[category]})"
        for category in categories
    ]

    def matrix_from_rows(
        rows: list[dict[str, object]],
        *,
        selector: Callable[[dict[str, object]], bool],
        value: str,
    ) -> np.ndarray:
        matrix = np.full((len(categories), len(layers)), np.nan)
        for category_index, category in enumerate(categories):
            for layer_index, layer in enumerate(layers):
                match = next(
                    (
                        row
                        for row in rows
                        if row["category"] == category
                        and int(row["layer"]) == layer
                        and selector(row)
                    ),
                    None,
                )
                if match is not None and match.get(value) is not None:
                    matrix[category_index, layer_index] = float(
                        match[value]
                    )
        return matrix

    rank_matrix = matrix_from_rows(
        rank_rows,
        selector=lambda row: (
            row["alignment"]
            == "history_feature_subspace_token_normalized"
            and int(row["rank"]) == 32
        ),
        value="energy_ratio_mean",
    )
    subspace_matrix = matrix_from_rows(
        subspace_rows,
        selector=lambda row: (
            row["mode"] == "history_centered"
            and int(row["rank"]) == 32
        ),
        value="relative_projection_error_mean",
    )
    residual_matrix = matrix_from_rows(
        residual_rows,
        selector=lambda row: (
            row["method"] == "identity"
            and abs(float(row["fraction"]) - 0.10) < 1e-9
        ),
        value="energy_ratio_mean",
    )

    def improvement_matrix(
        method: str,
        *,
        reference: str = "identity",
        minimum_reference_error: float = 0.05,
    ) -> np.ndarray:
        matrix = np.full((len(categories), len(layers)), np.nan)
        for category_index, category in enumerate(categories):
            for layer_index, layer in enumerate(layers):
                selected = [
                    row
                    for row in transport_rows
                    if row["category"] == category
                    and int(row["layer"]) == layer
                ]
                lookup = {str(row["method"]): row for row in selected}
                if reference not in lookup or method not in lookup:
                    continue
                baseline = lookup[reference][
                    "stable_centered_error_mean"
                ]
                candidate = lookup[method][
                    "stable_centered_error_mean"
                ]
                if baseline is None or candidate is None:
                    continue
                baseline_value = float(baseline)
                if baseline_value < minimum_reference_error:
                    continue
                matrix[category_index, layer_index] = (
                    baseline_value - float(candidate)
                ) / max(baseline_value, 1e-12)
        return matrix

    flow_matrix = improvement_matrix("optical_flow_warp")
    bttb_matrix = improvement_matrix("local_bttb_causal")
    bccb_increment_matrix = improvement_matrix(
        "local_bccb_causal",
        reference="local_bttb_causal",
    )

    def draw_heatmap(
        axis: object,
        matrix: np.ndarray,
        *,
        title: str,
        vmin: float,
        vmax: float,
        cmap_name: str,
    ) -> None:
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e9ecef")
        image = axis.imshow(
            matrix,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
        )
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(len(layers)), [f"L{layer}" for layer in layers])
        axis.set_yticks(range(len(categories)), category_labels)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    if value < vmin:
                        label = f"<{vmin:.2f}"
                    elif value > vmax:
                        label = f">{vmax:.2f}"
                    else:
                        label = f"{value:.2f}"
                    axis.text(
                        column_index,
                        row_index,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=(
                            "white"
                            if abs(value) > 0.65 * max(abs(vmin), abs(vmax))
                            else "#212529"
                        ),
                    )
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    draw_heatmap(
        axes[0, 0],
        rank_matrix,
        title="Rank-32 token-normalized energy (gate 0.70)",
        vmin=0.0,
        vmax=1.0,
        cmap_name="viridis",
    )
    draw_heatmap(
        axes[0, 1],
        subspace_matrix,
        title="Causal rank-32 projection error",
        vmin=0.0,
        vmax=1.0,
        cmap_name="magma_r",
    )
    draw_heatmap(
        axes[0, 2],
        residual_matrix,
        title="Top-10% residual energy (gate 0.70)",
        vmin=0.0,
        vmax=1.0,
        cmap_name="plasma",
    )
    for axis, matrix, title, reference_label in (
        (
            axes[1, 0],
            flow_matrix,
            "Optical-flow improvement vs identity",
            "gate 0.10",
        ),
        (
            axes[1, 1],
            bttb_matrix,
            "Local BTTB improvement vs identity",
            "gate 0.10",
        ),
        (
            axes[1, 2],
            bccb_increment_matrix,
            "Local BCCB incremental gain vs BTTB",
            "zero-gain reference",
        ),
    ):
        draw_heatmap(
            axis,
            matrix,
            title=f"{title} ({reference_label})",
            vmin=-0.10,
            vmax=0.20,
            cmap_name="RdYlGn",
        )
    fig.suptitle(
        "Category-stratified probe summary; transport cells are N/A when "
        "the reference error is below 0.05",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip().replace(" ", "_")
    if not prefix:
        raise ValueError("prefix must not be empty")
    collected = collect(args.root.resolve(), args.analysis_name)
    write_csv(args.out_dir / "rank_summary.csv", collected["rank_aggregate"])
    write_csv(
        args.out_dir / "transport_summary.csv",
        collected["transport_aggregate"],
    )
    write_csv(
        args.out_dir / "causal_subspace_summary.csv",
        collected["subspace_aggregate"],
    )
    write_csv(
        args.out_dir / "residual_summary.csv",
        collected["residual_aggregate"],
    )
    write_csv(
        args.out_dir / "rank_summary_by_category.csv",
        collected["rank_by_category"],
    )
    write_csv(
        args.out_dir / "transport_summary_by_category.csv",
        collected["transport_by_category"],
    )
    write_csv(
        args.out_dir / "causal_subspace_summary_by_category.csv",
        collected["subspace_by_category"],
    )
    write_csv(
        args.out_dir / "residual_summary_by_category.csv",
        collected["residual_by_category"],
    )
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(
            {
                "runs": collected["run_metadata"],
                "display_name": args.display_name,
                "prefix": prefix,
                "analysis_name": args.analysis_name,
                "rank_rows": len(collected["rank_raw"]),
                "transport_rows": len(collected["transport_raw"]),
                "subspace_rows": len(collected["subspace_raw"]),
                "residual_rows": len(collected["residual_raw"]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_temporal_spectrum(
        collected["rank_aggregate"],
        args.out_dir / f"{prefix}_temporal_spectrum.png",
        display_name=args.display_name,
    )
    plot_state_spectrum(
        collected["rank_aggregate"],
        args.out_dir / f"{prefix}_state_spectrum.png",
        display_name=args.display_name,
    )
    plot_transport(
        collected["transport_aggregate"],
        args.out_dir / f"{prefix}_transport_error.png",
    )
    plot_subspace(
        collected["subspace_aggregate"],
        args.out_dir / f"{prefix}_causal_subspace.png",
    )
    plot_residual(
        collected["residual_aggregate"],
        args.out_dir / f"{prefix}_residual_concentration.png",
    )
    figures = [
        f"{prefix}_temporal_spectrum.png",
        f"{prefix}_state_spectrum.png",
        f"{prefix}_state_spectrum.pdf",
        f"{prefix}_transport_error.png",
        f"{prefix}_causal_subspace.png",
        f"{prefix}_residual_concentration.png",
    ]
    category_figure = f"{prefix}_category_probe_summary.png"
    if plot_category_probe_summary(
        collected["rank_by_category"],
        collected["subspace_by_category"],
        collected["transport_by_category"],
        collected["residual_by_category"],
        args.out_dir / category_figure,
    ):
        figures.extend(
            [
                category_figure,
                f"{prefix}_category_probe_summary.pdf",
            ]
        )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "runs": len(collected["run_metadata"]),
                "figures": figures,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
