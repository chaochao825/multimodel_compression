#!/usr/bin/env python3
"""Generate follow-up plots from the existing World Foundry and H200 records."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "followup_v1"
FIGURES.mkdir(parents=True, exist_ok=True)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_plot(name: str, frame: pd.DataFrame, fig: plt.Figure) -> None:
    frame.to_csv(FIGURES / f"{name}.csv", index=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def activation_plots() -> None:
    path = RESULTS / "activation_aware_v2" / "activation_aware_metrics.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    summary = (
        data.groupby("method", as_index=False)
        .agg(
            params=("params", "mean"),
            budget_ratio=("budget_ratio", "mean"),
            holdout_activation_rel_l2=("holdout_activation_rel_l2", "mean"),
            holdout_activation_cosine=("holdout_activation_cosine", "mean"),
            holdout_weight_rel_fro=("holdout_weight_rel_fro", "mean"),
        )
        .sort_values("holdout_activation_rel_l2")
    )
    summary.to_csv(RESULTS / "activation_aware_summary_v2.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    families = {
        "ternary_main": ("Ternary", "#4c78a8"),
        "fro_lowrank": ("Frobenius LR", "#f58518"),
        "fro_bcm": ("Frobenius BCM", "#54a24b"),
        "activation_lowrank": ("Activation LR", "#e45756"),
        "activation_bcm": ("Activation BCM", "#b279a2"),
        "activation_lr": ("LR + row sparse", "#ff9da6"),
    }
    for _, row in summary.iterrows():
        method = str(row["method"])
        family = next((value for key, value in families.items() if method.startswith(key)), ("Other", "#777777"))
        ax.scatter(row["budget_ratio"], row["holdout_activation_rel_l2"], s=55, color=family[1], label=family[0])
        # Label only the budget-matched headline methods; the full CSV keeps
        # every point and avoids unreadable collisions in the dense cluster.
        headline = {
            "ternary_main",
            "fro_lowrank_r16",
            "fro_bcm_plus_lr_b128_r10",
            "activation_lowrank_r16",
            "activation_bcm_plus_lr_b128_r10",
            "activation_lr_r8_row_sparse_b16",
        }
        if method in headline:
            label = method.replace("_", " ")
            offsets = {
                "activation_lowrank_r16": (4, 12),
                "activation_lr_r8_row_sparse_b16": (4, -13),
                "activation_bcm_plus_lr_b128_r10": (4, 10),
            }
            ax.annotate(label, (row["budget_ratio"], row["holdout_activation_rel_l2"]), xytext=offsets.get(method, (4, 3)), textcoords="offset points", fontsize=7)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, loc="upper right")
    ax.set_xlabel("Parameter budget ratio")
    ax.set_ylabel("Held-out relative activation L2")
    ax.set_title("Activation-aware residual probe, Wan blocks")
    save_plot("activation_frontier_v2", summary, fig)

    keep = [
        "ternary_main",
        "fro_lowrank_r16",
        "fro_bcm_plus_lr_b128_r10",
        "activation_lowrank_r16",
        "activation_bcm_plus_lr_b128_r10",
        "activation_lr_r8_row_sparse_b16",
    ]
    bars = summary[summary.method.isin(keep)].copy()
    order = [item for item in keep if item in set(bars.method)]
    bars["method"] = pd.Categorical(bars["method"], categories=order, ordered=True)
    bars = bars.sort_values("method")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(np.arange(len(bars)), bars["holdout_activation_rel_l2"], color="#2f6690")
    ax.set_xticks(np.arange(len(bars)))
    ax.set_xticklabels([re.sub("_", "\\n", value.replace("activation_", "act_").replace("fro_", "fro_")) for value in bars.method.astype(str)], fontsize=7)
    ax.set_ylabel("Held-out relative activation L2")
    ax.set_title("Matched-budget residual comparison")
    save_plot("activation_matched_budget_v2", bars, fig)

    structure = pd.read_csv(RESULTS / "activation_aware_v2" / "activation_aware_structure.csv")
    structure_summary = structure.groupby("block_size", as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(structure_summary))
    width = 0.25
    ax.bar(x - width, structure_summary["fro_bcm_capture"], width, label="Frobenius capture")
    ax.bar(x, structure_summary["activation_bcm_fit_capture"], width, label="Activation fit capture")
    ax.bar(x + width, structure_summary["activation_bcm_holdout_capture"], width, label="Activation holdout capture")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"b={int(v)}" for v in structure_summary.block_size])
    ax.set_ylabel("Residual energy captured")
    ax.set_title("BCM capture: fit versus held-out activation")
    ax.legend(fontsize=8)
    save_plot("bcm_capture_fit_holdout_v2", structure_summary, fig)


def h200_plots() -> None:
    path = RESULTS / "h200_bcm_cached_v1" / "h200_benchmark.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    keep = [
        "bf16_dense",
        "fp8_dynamic_main",
        "fp8_static_input_main",
        "fp8_dynamic_svd_r16",
        "fp8_static_input_svd_r16",
        "fp8_dynamic_cached_bcm_b128_svd_r10",
        "fp8_static_input_cached_bcm_b128_svd_r10",
        "fp8_dynamic_svd_r8_rowsparse_b16",
        "fp8_static_input_svd_r8_rowsparse_b16",
    ]
    selected = data[data.method.isin(keep)].copy()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for method, group in selected.groupby("method", sort=False):
        group = group.sort_values("rows")
        scope = group.measurement_scope.iloc[0]
        ax.plot(group.rows, group.latency_ms_median, marker="o", label=f"{method} ({scope})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Activation rows")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("H200 NVL latency: dense, FP8, and residual branches")
    ax.legend(fontsize=6, ncol=2)
    save_plot("h200_latency_tradeoff_v1", selected, fig)

    paired_rows = []
    for rows in sorted(data.rows.unique()):
        subset = data[data.rows == rows].set_index("method")
        for scope, on_demand, cached in [
            ("measured", "fp8_dynamic_bcm_b128_svd_r10", "fp8_dynamic_cached_bcm_b128_svd_r10"),
            ("lower_bound", "fp8_static_input_bcm_b128_svd_r10", "fp8_static_input_cached_bcm_b128_svd_r10"),
        ]:
            if on_demand in subset.index and cached in subset.index:
                old = float(subset.loc[on_demand, "latency_ms_median"])
                new = float(subset.loc[cached, "latency_ms_median"])
                paired_rows.append({"rows": rows, "scope": scope, "on_demand_ms": old, "cached_ms": new, "cached_speedup": old / new})
    paired = pd.DataFrame(paired_rows)
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for scope, group in paired.groupby("scope"):
        ax.plot(group.rows, group.cached_speedup, marker="o", label=scope)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Activation rows")
    ax.set_ylabel("On-demand BCM / cached BCM latency")
    ax.set_title("Caching generator FFT is not the main H200 bottleneck")
    ax.legend()
    save_plot("h200_cached_fft_effect_v1", paired, fig)


def worldfoundry_plots() -> None:
    generation = ROOT / "remote_snapshot" / "wan_generation_f81_hybrid_20step_v1" / "generation_runs.csv"
    if generation.exists():
        data = pd.read_csv(generation)
        grouped = data.groupby("method", as_index=False)["seconds_including_text_and_vae"].median()
        grouped = grouped.sort_values("seconds_including_text_and_vae")
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.barh(grouped.method, grouped.seconds_including_text_and_vae, color="#4c78a8")
        ax.set_xlabel("End-to-end seconds, one 20-step sample")
        ax.set_title("Existing World Foundry Wan H200 generation records")
        save_plot("worldfoundry_pipeline_latency_existing", grouped, fig)

    nfe = RESULTS / "nfe_sweep_f17_quality_v1" / "paired_video_metrics.csv"
    if nfe.exists():
        data = pd.read_csv(nfe)
        data["steps"] = data["video"].str.extract(r"step(\d+)").astype(float)
        data = data[np.isfinite(data["steps"])]
        data = data.sort_values("steps")
        plot_data = data[["video", "steps", "frame_ssim_mean", "frame_psnr_mean_db", "pixel_psnr_db"]].copy()
        fig, ax1 = plt.subplots(figsize=(7.4, 4.6))
        ax1.plot(plot_data.steps, plot_data.frame_ssim_mean, marker="o", color="#54a24b", label="Frame SSIM")
        ax1.set_xlabel("Sampling steps")
        ax1.set_ylabel("Frame SSIM vs step-20 reference", color="#54a24b")
        ax1.set_ylim(0, 1.02)
        ax2 = ax1.twinx()
        ax2.plot(plot_data.steps, plot_data.frame_psnr_mean_db, marker="s", color="#f58518", label="Frame PSNR")
        ax2.set_ylabel("Frame PSNR (dB)", color="#f58518")
        ax1.set_title("Existing NFE sweep: quality falls before cache/precision gains")
        save_plot("nfe_quality_existing", plot_data, fig)

    teacache = ROOT / "remote_snapshot" / "pilot8_teacache_f81_quality_v1" / "paired_video_summary.csv"
    if teacache.exists():
        data = pd.read_csv(teacache)
        data.to_csv(RESULTS / "teacache_quality_existing.csv", index=False)


def main() -> None:
    style()
    activation_plots()
    h200_plots()
    worldfoundry_plots()
    print(f"wrote plots to {FIGURES}")


if __name__ == "__main__":
    main()
