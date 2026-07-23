#!/usr/bin/env python3
"""Generate publication-ready plots for the World Foundry H200 probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "h200_live"
OUT = DATA / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "attention": "#D55E00",
    "ffn": "#0072B2",
    "middle": "#009E73",
    "periodic": "#CC79A7",
    "eager": "#999999",
    "triton": "#E69F00",
}

MULTI_METHODS = (
    ("dense_cache007", "Dense + cache .07"),
    ("dense_cache008", "Dense + cache .08"),
    ("fp8", "Full FP8"),
    ("fp8_middle1", "FP8 middle-1"),
    ("hybrid_middle1", "Hybrid middle-1"),
    ("hybrid_middle1_cache008", "Hybrid + cache .08"),
)

MULTI_COLORS = (
    "#777777",
    "#56B4E9",
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#CC79A7",
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def read_summary(run: str) -> pd.DataFrame:
    path = DATA / run / "analysis" / "method_summary.csv"
    frame = pd.read_csv(path)
    frame["run"] = run
    return frame


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUT / f"{name}.pdf")
    figure.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(figure)


def bootstrap_interval(
    values: np.ndarray,
    *,
    geometric: bool = False,
    seed: int,
    samples: int = 20_000,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return np.nan, np.nan, np.nan
    transformed = np.log(clean) if geometric else clean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(transformed), size=(samples, len(transformed)))
    draws = transformed[indices].mean(axis=1)
    estimate = transformed.mean()
    if geometric:
        estimate = np.exp(estimate)
        draws = np.exp(draws)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(estimate), float(low), float(high)


def plot_method_tradeoff() -> None:
    attention = read_summary("worldfoundry_hybrid_f17_pilot_v1")
    ffn = read_summary("worldfoundry_ffn_hybrid_f17_pilot_v2")
    rows: list[dict[str, object]] = [
        {
            "family": "Dense reference",
            "method": "dense",
            "speedup": 1.0,
            "ssim": 1.0,
            "psnr_db": float("inf"),
        }
    ]
    for family, frame in (("Attention q/o", attention), ("FFN", ffn)):
        for item in frame.to_dict("records"):
            rows.append(
                {
                    "family": family,
                    "method": item["method"],
                    "speedup": item["paired_speedup_geomean"],
                    "ssim": item["frame_ssim_mean"],
                    "psnr_db": item["pixel_psnr_db_mean"],
                }
            )
    write_csv(OUT / "method_quality_speed_tradeoff.csv", rows)

    figure, axis = plt.subplots(figsize=(7.3, 4.7))
    family_style = {
        "Dense reference": ("black", "*"),
        "Attention q/o": (COLORS["attention"], "s"),
        "FFN": (COLORS["ffn"], "o"),
    }
    labels = {
        ("Attention q/o", "dense_cache008"): ("q/o cache", (7, 9)),
        ("Attention q/o", "fp8"): ("q/o FP8", (4, -13)),
        ("Attention q/o", "hybrid"): ("q/o hybrid", (4, 7)),
        ("Attention q/o", "hybrid_cache008"): ("q/o hybrid+cache", (4, -13)),
        ("FFN", "dense_cache008"): ("FFN cache", (-65, 9)),
        ("FFN", "fp8"): ("FFN FP8", (4, 7)),
        ("FFN", "hybrid"): ("FFN hybrid", (4, 7)),
        ("FFN", "hybrid_refresh4"): ("FFN refresh-4", (4, 7)),
        ("FFN", "hybrid_refresh4_cache008"): ("FFN refresh-4+cache", (4, 7)),
    }
    for family, (color, marker) in family_style.items():
        subset = [row for row in rows if row["family"] == family]
        axis.scatter(
            [row["speedup"] for row in subset],
            [row["ssim"] for row in subset],
            color=color,
            marker=marker,
            s=60,
            label=family,
            zorder=3,
        )
        for row in subset:
            label_spec = labels.get((family, str(row["method"])))
            if label_spec is not None:
                label, offset = label_spec
                axis.annotate(
                    label,
                    (float(row["speedup"]), float(row["ssim"])),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7.2,
                )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    axis.axhline(0.95, color="#666666", linestyle=":", linewidth=0.8)
    axis.set_xlabel("Paired end-to-end speedup (x)")
    axis.set_ylabel("Frame SSIM versus dense")
    axis.set_xlim(0.915, 1.045)
    axis.set_ylim(0.62, 1.015)
    axis.legend(loc="upper left")
    save_figure(figure, "method_quality_speed_tradeoff")


def plot_fusion_ablation() -> None:
    eager = read_summary("worldfoundry_ffn_hybrid_f17_pilot_v1")
    fused = read_summary("worldfoundry_ffn_hybrid_f17_pilot_v2")
    methods = ["fp8", "hybrid", "hybrid_refresh4", "hybrid_refresh4_cache008"]
    labels = ["FP8", "Hybrid", "Refresh-4", "Refresh-4 + cache"]
    rows: list[dict[str, object]] = []
    for implementation, frame in (("Eager cast", eager), ("Triton fused", fused)):
        indexed = frame.set_index("method")
        for method, label in zip(methods, labels):
            rows.append(
                {
                    "implementation": implementation,
                    "method": method,
                    "label": label,
                    "speedup": indexed.loc[method, "paired_speedup_geomean"],
                    "ssim": indexed.loc[method, "frame_ssim_mean"],
                }
            )
    write_csv(OUT / "triton_fusion_ablation.csv", rows)

    frame = pd.DataFrame(rows)
    positions = list(range(len(methods)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for offset, implementation, color in (
        (-width / 2, "Eager cast", COLORS["eager"]),
        (width / 2, "Triton fused", COLORS["triton"]),
    ):
        values = frame[frame["implementation"] == implementation]["speedup"]
        axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=implementation,
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Paired end-to-end speedup (x)")
    axis.set_ylim(0.80, 1.06)
    axis.legend(loc="upper left")
    save_figure(figure, "triton_fusion_ablation")


def plot_schedule_frontier() -> None:
    run = "worldfoundry_ffn_hybrid_f17_schedule_screen_v2"
    summary = read_summary(run)
    generation = pd.read_csv(DATA / run / "data" / "generation_runs.csv")
    merged = summary.merge(
        generation[
            [
                "method",
                "precision_approximate_fraction",
                "precision_approximate_model_forwards",
                "precision_dense_model_forwards",
            ]
        ],
        on="method",
        how="left",
    )
    merged.to_csv(OUT / "schedule_quality_frontier.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=True)
    groups = {
        "Contiguous middle": merged[merged["method"].str.startswith("hybrid_middle")],
        "Periodic dense": merged[merged["method"].str.startswith("hybrid_refresh")],
    }
    for label, frame in groups.items():
        frame = frame.sort_values("precision_approximate_fraction")
        color = COLORS["middle"] if label.startswith("Contiguous") else COLORS["periodic"]
        marker = "o" if label.startswith("Contiguous") else "s"
        axes[0].plot(
            frame["precision_approximate_fraction"] * 100,
            frame["frame_ssim_mean"],
            color=color,
            marker=marker,
            label=label,
        )
        axes[1].plot(
            frame["precision_approximate_fraction"] * 100,
            frame["paired_speedup_geomean"],
            color=color,
            marker=marker,
            label=label,
        )
    fp8_one = merged[merged["method"] == "fp8_middle1"]
    axes[0].scatter(
        fp8_one["precision_approximate_fraction"] * 100,
        fp8_one["frame_ssim_mean"],
        color=COLORS["ffn"],
        marker="^",
        s=55,
        label="FP8 middle-1",
        zorder=3,
    )
    axes[1].scatter(
        fp8_one["precision_approximate_fraction"] * 100,
        fp8_one["paired_speedup_geomean"],
        color=COLORS["ffn"],
        marker="^",
        s=55,
        label="FP8 middle-1",
        zorder=3,
    )
    axes[0].axhline(0.95, color="#666666", linestyle=":", linewidth=0.8)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Frame SSIM versus dense")
    axes[1].set_ylabel("Paired end-to-end speedup (x)")
    for axis in axes:
        axis.set_xlabel("Approximate model forwards (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    save_figure(figure, "schedule_quality_frontier")


def plot_multisample_validation() -> None:
    run = "worldfoundry_ffn_hybrid_f17_multiseed_v2"
    paired = pd.read_csv(DATA / run / "analysis" / "paired_metrics.csv")
    paired = paired[paired["status"] == "ok"].copy()
    summary_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    for method_index, (method, label) in enumerate(MULTI_METHODS):
        frame = paired[paired["method"] == method].copy()
        if frame.empty:
            raise RuntimeError(f"missing multi-sample method: {method}")
        frame["label"] = label
        frame["method_index"] = method_index
        selected_frames.append(frame)
        ssim = bootstrap_interval(
            frame["frame_ssim_mean"].to_numpy(), seed=20260723 + method_index
        )
        speed = bootstrap_interval(
            frame["paired_speedup"].to_numpy(),
            geometric=True,
            seed=20260823 + method_index,
        )
        summary_rows.append(
            {
                "method": method,
                "label": label,
                "pairs": len(frame),
                "ssim_mean": ssim[0],
                "ssim_ci_low": ssim[1],
                "ssim_ci_high": ssim[2],
                "ssim_min": frame["frame_ssim_mean"].min(),
                "speedup_geomean": speed[0],
                "speedup_ci_low": speed[1],
                "speedup_ci_high": speed[2],
                "speedup_min": frame["paired_speedup"].min(),
                "pixel_psnr_db_mean": frame["pixel_psnr_db"].replace(
                    [np.inf, -np.inf], np.nan
                ).mean(),
                "cache_fraction_mean": frame[
                    "cached_model_forward_fraction"
                ].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "multisample_bootstrap_summary.csv", index=False)
    selected = pd.concat(selected_frames, ignore_index=True)
    selected.to_csv(OUT / "multisample_paired_points.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    positions = np.arange(len(MULTI_METHODS))
    jitter = np.linspace(-0.13, 0.13, 8)
    for method_index, ((method, _), color) in enumerate(
        zip(MULTI_METHODS, MULTI_COLORS)
    ):
        frame = selected[selected["method"] == method].sort_values(
            ["prompt_index", "seed"]
        )
        offset = jitter[: len(frame)]
        axes[0].scatter(
            method_index + offset,
            frame["frame_ssim_mean"],
            color=color,
            alpha=0.55,
            s=24,
            zorder=2,
        )
        axes[1].scatter(
            method_index + offset,
            frame["paired_speedup"],
            color=color,
            alpha=0.55,
            s=24,
            zorder=2,
        )
        row = summary.iloc[method_index]
        axes[0].errorbar(
            method_index,
            row["ssim_mean"],
            yerr=[
                [row["ssim_mean"] - row["ssim_ci_low"]],
                [row["ssim_ci_high"] - row["ssim_mean"]],
            ],
            fmt="o",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=3,
            zorder=3,
        )
        axes[1].errorbar(
            method_index,
            row["speedup_geomean"],
            yerr=[
                [row["speedup_geomean"] - row["speedup_ci_low"]],
                [row["speedup_ci_high"] - row["speedup_geomean"]],
            ],
            fmt="o",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=3,
            zorder=3,
        )
    labels = [label for _, label in MULTI_METHODS]
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=24, ha="right")
    axes[0].axhline(0.95, color="#666666", linestyle=":", linewidth=0.8)
    axes[0].axhline(0.98, color="#666666", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Frame SSIM versus dense")
    axes[0].set_ylim(0.60, 1.015)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Paired end-to-end speedup (x)")
    axes[1].set_ylim(0.96, 1.055)
    save_figure(figure, "multisample_quality_speed")


def plot_prompt_sensitivity() -> None:
    run = "worldfoundry_ffn_hybrid_f17_multiseed_v2"
    paired = pd.read_csv(DATA / run / "analysis" / "paired_metrics.csv")
    methods = (
        ("dense_cache008", "Dense cache .08"),
        ("fp8", "Full FP8"),
        ("fp8_middle1", "FP8 middle-1"),
        ("hybrid_middle1", "Hybrid middle-1"),
        ("hybrid_middle1_cache008", "Hybrid + cache .08"),
    )
    prompt_labels = ("Panda", "Dancer", "Car", "Lighthouse")
    grouped = (
        paired[paired["method"].isin(method for method, _ in methods)]
        .groupby(["method", "prompt_index"], as_index=False)["frame_ssim_mean"]
        .mean()
    )
    grouped["method_label"] = grouped["method"].map(dict(methods))
    grouped["prompt_label"] = grouped["prompt_index"].map(
        dict(enumerate(prompt_labels))
    )
    grouped.to_csv(OUT / "multisample_prompt_sensitivity.csv", index=False)
    matrix = np.vstack(
        [
            grouped[grouped["method"] == method]
            .set_index("prompt_index")
            .reindex(range(len(prompt_labels)))["frame_ssim_mean"]
            .to_numpy()
            for method, _ in methods
        ]
    )

    figure, axis = plt.subplots(figsize=(6.8, 4.1))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.64, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value < 0.80 else "black"
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    axis.set_xticks(range(len(prompt_labels)), prompt_labels)
    axis.set_yticks(range(len(methods)), [label for _, label in methods])
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Mean SSIM over two seeds")
    save_figure(figure, "multisample_prompt_sensitivity")


def plot_rank16_revision_tradeoff() -> None:
    rank8_run = "worldfoundry_ffn_hybrid_f17_multiseed_v2"
    rank16_run = "worldfoundry_ffn_rank16_f17_multiseed_v1"
    rank8 = pd.read_csv(DATA / rank8_run / "analysis" / "paired_metrics.csv")
    rank16 = pd.read_csv(DATA / rank16_run / "analysis" / "paired_metrics.csv")
    specifications = (
        (rank16, "dense_cache008", "Dense + cache .08", "#56B4E9"),
        (rank16, "fp8_middle1", "FP8 middle-1", "#0072B2"),
        (rank8, "hybrid_middle1", "Rank-8 + sparse", "#009E73"),
        (rank16, "hybrid_middle1", "Rank-16", "#D55E00"),
        (rank8, "hybrid_middle1_cache008", "Rank-8 + sparse + cache", "#CC79A7"),
        (rank16, "hybrid_middle1_cache008", "Rank-16 + cache", "#E69F00"),
    )
    rows: list[dict[str, object]] = []
    for index, (source, method, label, color) in enumerate(specifications):
        frame = source[source["method"] == method]
        ssim = bootstrap_interval(
            frame["frame_ssim_mean"].to_numpy(), seed=20260923 + index
        )
        speed = bootstrap_interval(
            frame["paired_speedup"].to_numpy(),
            geometric=True,
            seed=20261023 + index,
        )
        rows.append(
            {
                "method": method,
                "label": label,
                "source": "rank8_sparse" if source is rank8 else "rank16",
                "color": color,
                "ssim_mean": ssim[0],
                "ssim_ci_low": ssim[1],
                "ssim_ci_high": ssim[2],
                "speedup_geomean": speed[0],
                "speedup_ci_low": speed[1],
                "speedup_ci_high": speed[2],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "rank16_revision_tradeoff.csv", index=False)

    offsets = {
        "Dense + cache .08": (-8, 8),
        "FP8 middle-1": (20, 17),
        "Rank-8 + sparse": (5, -15),
        "Rank-16": (-18, 18),
        "Rank-8 + sparse + cache": (5, -12),
        "Rank-16 + cache": (5, 8),
    }
    figure, axis = plt.subplots(figsize=(7.4, 4.7))
    for row in rows:
        axis.errorbar(
            row["speedup_geomean"],
            row["ssim_mean"],
            xerr=[
                [row["speedup_geomean"] - row["speedup_ci_low"]],
                [row["speedup_ci_high"] - row["speedup_geomean"]],
            ],
            yerr=[
                [row["ssim_mean"] - row["ssim_ci_low"]],
                [row["ssim_ci_high"] - row["ssim_mean"]],
            ],
            fmt="o",
            color=row["color"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=3,
            zorder=3,
        )
        axis.annotate(
            row["label"],
            (row["speedup_geomean"], row["ssim_mean"]),
            xytext=offsets[row["label"]],
            textcoords="offset points",
            fontsize=8,
            ha="right" if offsets[row["label"]][0] < 0 else "left",
        )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    axis.axhline(0.95, color="#666666", linestyle=":", linewidth=0.8)
    axis.set_xlabel("Paired end-to-end speedup (x)")
    axis.set_ylabel("Frame SSIM versus dense")
    axis.set_xlim(0.975, 1.082)
    axis.set_ylim(0.925, 0.972)
    save_figure(figure, "rank16_revision_tradeoff")


def plot_residual_component_ablation() -> None:
    component_runs = (
        ("sparse", "Sparse only"),
        ("lr8", "Rank-8 only"),
        ("lr8_sparse", "Rank-8 + sparse"),
        ("lr12sparse", "Rank-12 + sparse"),
        ("lr16", "Rank-16 only"),
    )
    baseline_run = "worldfoundry_ffn_f17_components_lr8_v1"
    baseline_pairs = pd.read_csv(DATA / baseline_run / "analysis" / "paired_metrics.csv")
    baseline_pairs = baseline_pairs[baseline_pairs["method"] == "fp8_middle1"]
    detail_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = [
        {
            "variant": "fp8_middle1",
            "label": "FP8 middle-1",
            "ssim_mean": baseline_pairs["frame_ssim_mean"].mean(),
            "ssim_min": baseline_pairs["frame_ssim_mean"].min(),
            "speedup_geomean": float(
                np.exp(np.log(baseline_pairs["paired_speedup"]).mean())
            ),
            "low_rank_values": 0,
            "sparse_values": 0,
        }
    ]
    baseline_detail = baseline_pairs.copy()
    baseline_detail["variant"] = "fp8_middle1"
    baseline_detail["label"] = "FP8 middle-1"
    detail_frames.append(baseline_detail)

    for variant, label in component_runs:
        if variant == "lr8_sparse":
            run = "worldfoundry_ffn_hybrid_f17_multiseed_v2"
            pairs = pd.read_csv(DATA / run / "analysis" / "paired_metrics.csv")
            pairs = pairs[
                (pairs["method"] == "hybrid_middle1")
                & (pairs["prompt_index"].isin((0, 1)))
                & (pairs["seed"] == 20260723)
            ]
            manifest_path = DATA / run / "generation_manifest.json"
        else:
            run = f"worldfoundry_ffn_f17_components_{variant}_v1"
            pairs = pd.read_csv(DATA / run / "analysis" / "paired_metrics.csv")
            pairs = pairs[pairs["method"] == "hybrid_middle1"]
            manifest_path = DATA / run / "data" / "generation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        replacement = manifest["replacement"]
        rows.append(
            {
                "variant": variant,
                "label": label,
                "ssim_mean": pairs["frame_ssim_mean"].mean(),
                "ssim_min": pairs["frame_ssim_mean"].min(),
                "speedup_geomean": float(
                    np.exp(np.log(pairs["paired_speedup"]).mean())
                ),
                "low_rank_values": replacement["low_rank_values"],
                "sparse_values": replacement["sparse_values"],
            }
        )
        detail = pairs.copy()
        detail["variant"] = variant
        detail["label"] = label
        detail_frames.append(detail)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "residual_component_ablation.csv", index=False)
    detail = pd.concat(detail_frames, ignore_index=True)
    detail.to_csv(OUT / "residual_component_ablation_points.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    positions = np.arange(len(summary))
    colors = ("#0072B2", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00")
    for position, (row, color) in enumerate(zip(summary.to_dict("records"), colors)):
        points = detail[detail["variant"] == row["variant"]]["frame_ssim_mean"]
        axes[0].scatter(
            np.full(len(points), position) + np.linspace(-0.06, 0.06, len(points)),
            points,
            color=color,
            alpha=0.65,
            s=35,
        )
        axes[0].scatter(
            position,
            row["ssim_mean"],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=55,
            zorder=3,
        )
    axes[0].axhline(
        summary.iloc[0]["ssim_mean"], color="#555555", linestyle=":", linewidth=0.8
    )
    axes[0].set_ylabel("Mean frame SSIM versus dense")
    axes[0].set_ylim(0.885, 0.975)

    low_rank = summary["low_rank_values"] / 1e6
    sparse = summary["sparse_values"] / 1e6
    axes[1].bar(positions, low_rank, color="#0072B2", label="Low-rank values")
    axes[1].bar(
        positions,
        sparse,
        bottom=low_rank,
        color="#E69F00",
        label="Static row-block values",
    )
    axes[1].set_ylabel("BF16 residual values (million)")
    axes[1].legend(loc="upper left", fontsize=8)
    labels = summary["label"].tolist()
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=24, ha="right")
    save_figure(figure, "residual_component_ablation")


def main() -> None:
    configure_style()
    plot_method_tradeoff()
    plot_fusion_ablation()
    plot_schedule_frontier()
    plot_multisample_validation()
    plot_prompt_sensitivity()
    plot_rank16_revision_tradeoff()
    plot_residual_component_ablation()
    expected = [
        OUT / f"{name}.{extension}"
        for name in (
            "method_quality_speed_tradeoff",
            "triton_fusion_ablation",
            "schedule_quality_frontier",
            "multisample_quality_speed",
            "multisample_prompt_sensitivity",
            "rank16_revision_tradeoff",
            "residual_component_ablation",
        )
        for extension in ("pdf", "png")
    ]
    missing = [str(path) for path in expected if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError(f"missing or empty figure outputs: {missing}")
    print(f"PLOT_OK outputs={len(expected)} directory={OUT}")


if __name__ == "__main__":
    main()
