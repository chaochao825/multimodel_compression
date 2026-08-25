from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "action_dit_transport_cache_20260826"
FIGURE_ROOT = PROJECT_ROOT / "figures"

COLORS = {
    "raw_reuse": "#7A7A7A",
    "shift_reuse": "#0072B2",
    "shift_toeplitz_r2": "#009E73",
    "shift_rank8_feature": "#E69F00",
    "shift_rank8_prev_flow": "#D55E00",
    "shift_rank8_oracle": "#CC79A7",
}

LABELS = {
    "raw_reuse": "Raw reuse",
    "shift_reuse": "Horizon shift",
    "shift_toeplitz_r2": "Shift + Toeplitz",
    "shift_rank8_feature": "Shift + feature LR",
    "shift_rank8_prev_flow": "Shift + state LR",
    "shift_rank8_oracle": "Shift + oracle LR",
}


def load_results():
    summary_rows = []
    method_rows = []
    geometry_frames = []
    suffix_frames = []
    for run_dir in sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        train_id = int(run_dir.name.split("_")[0].replace("train", ""))
        offset = int(summary["control_offset"])
        mechanism = summary["mechanism"]
        summary_rows.append(
            {
                "run": run_dir.name,
                "checkpoint": train_id,
                "offset": offset,
                **mechanism,
            }
        )
        for method, metrics in summary["exact_suffix"].items():
            method_rows.append(
                {
                    "run": run_dir.name,
                    "checkpoint": train_id,
                    "offset": offset,
                    "method": method,
                    **metrics,
                }
            )
        geometry = pd.read_csv(run_dir / "geometry_metrics.csv")
        geometry["run"] = run_dir.name
        geometry["checkpoint"] = train_id
        geometry["offset"] = offset
        geometry_frames.append(geometry)
        suffix = pd.read_csv(run_dir / "exact_suffix_metrics.csv")
        suffix["run"] = run_dir.name
        suffix["checkpoint"] = train_id
        suffix["offset"] = offset
        suffix_frames.append(suffix)
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(method_rows),
        pd.concat(geometry_frames, ignore_index=True),
        pd.concat(suffix_frames, ignore_index=True),
    )


def save_bound_data(summary, methods, geometry, suffix):
    summary.to_csv(FIGURE_ROOT / "action_dit_transport_cache_summary.csv", index=False)
    methods.to_csv(FIGURE_ROOT / "action_dit_transport_cache_method_risk.csv", index=False)
    geometry_aggregate = (
        geometry.groupby(["offset", "noise_mode", "method"], as_index=False)[
            "activation_relative_l2"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    geometry_aggregate.to_csv(
        FIGURE_ROOT / "action_dit_transport_cache_noise_control.csv", index=False
    )
    shift = suffix[suffix["method"] == "shift_reuse"][
        ["checkpoint", "offset", "layer", "flow_point", "velocity_relative_l2"]
    ].rename(columns={"velocity_relative_l2": "shift_risk"})
    toeplitz = suffix[suffix["method"] == "shift_toeplitz_r2"][
        ["checkpoint", "offset", "layer", "flow_point", "velocity_relative_l2"]
    ].rename(columns={"velocity_relative_l2": "toeplitz_risk"})
    layer_step = shift.merge(
        toeplitz, on=["checkpoint", "offset", "layer", "flow_point"]
    )
    layer_step["toeplitz_improvement"] = (
        1.0 - layer_step["toeplitz_risk"] / layer_step["shift_risk"]
    )
    layer_step.to_csv(
        FIGURE_ROOT / "action_dit_transport_cache_layer_step.csv", index=False
    )
    return geometry_aggregate, layer_step


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def plot_summary(summary, methods, geometry_aggregate):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    selected_methods = [
        "raw_reuse",
        "shift_reuse",
        "shift_toeplitz_r2",
        "shift_rank8_feature",
        "shift_rank8_prev_flow",
        "shift_rank8_oracle",
    ]
    m8 = methods[(methods["offset"] == 8) & methods["method"].isin(selected_methods)]
    x = np.arange(3)
    width = 0.125
    for index, method in enumerate(selected_methods):
        values = (
            m8[m8["method"] == method]
            .sort_values("checkpoint")["velocity_relative_l2"]
            .to_numpy()
        )
        axes[0, 0].bar(
            x + (index - 2.5) * width,
            100 * values,
            width,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0, 0].set_xticks(x, ["ckpt 0", "ckpt 1", "ckpt 2"])
    axes[0, 0].set_ylabel("Exact-suffix velocity error (%)")
    axes[0, 0].legend(ncol=2, frameon=False, loc="upper right")
    axes[0, 0].text(-0.14, 1.04, "a", transform=axes[0, 0].transAxes, weight="bold")
    style_axis(axes[0, 0])

    noise_methods = ["raw_reuse", "shift_reuse", "shift_toeplitz_r2"]
    noise = geometry_aggregate[
        (geometry_aggregate["offset"] == 8)
        & geometry_aggregate["method"].isin(noise_methods)
    ]
    x = np.arange(len(noise_methods))
    for index, mode in enumerate(["aligned", "independent"]):
        values = []
        errors = []
        for method in noise_methods:
            row = noise[(noise["noise_mode"] == mode) & (noise["method"] == method)]
            values.append(float(row["mean"].iloc[0]))
            errors.append(float(row["std"].iloc[0]))
        axes[0, 1].bar(
            x + (index - 0.5) * 0.34,
            np.asarray(values),
            0.34,
            yerr=np.asarray(errors),
            capsize=2,
            color=["#0072B2", "#999999"][index],
            label=["Aligned latent", "Independent latent"][index],
        )
    axes[0, 1].set_xticks(x, [LABELS[method] for method in noise_methods], rotation=12)
    axes[0, 1].set_ylabel("FFN activation relative L2")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].text(-0.14, 1.04, "b", transform=axes[0, 1].transAxes, weight="bold")
    style_axis(axes[0, 1])

    m8_summary = summary[summary["offset"] == 8].sort_values("checkpoint")
    metrics = [
        "shift_risk_improvement_vs_raw",
        "toeplitz_risk_improvement_vs_shift",
        "rank8_heldout_energy_mean",
        "feature_coefficient_r2_mean",
        "previous_flow_coefficient_r2_mean",
    ]
    labels = ["Shift/raw", "Toeplitz/shift", "Rank-8 energy", "Feature R2", "State R2"]
    x = np.arange(len(metrics))
    for index, row in m8_summary.reset_index(drop=True).iterrows():
        values = [float(row[metric]) for metric in metrics]
        axes[1, 0].plot(
            x,
            values,
            marker=["o", "s", "^"][index],
            linewidth=1.5,
            label=f"ckpt {int(row['checkpoint'])}",
        )
    axes[1, 0].axhline(0.7, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 0].set_xticks(x, labels, rotation=18)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Fraction / R2")
    axes[1, 0].legend(frameon=False, ncol=3, loc="lower right")
    axes[1, 0].text(-0.14, 1.04, "c", transform=axes[1, 0].transAxes, weight="bold")
    style_axis(axes[1, 0])

    compute = summary.groupby("offset", as_index=False)[
        ["overlap_fraction", "control_tick_only_denoiser_speed_ceiling"]
    ].mean()
    x = np.arange(len(compute))
    axes[1, 1].bar(
        x - 0.18,
        compute["overlap_fraction"],
        0.36,
        color="#56B4E9",
        label="Reusable token fraction",
    )
    axes[1, 1].set_ylabel("Reusable token fraction")
    axes[1, 1].set_xticks(x, [f"m={value}" for value in compute["offset"]])
    second = axes[1, 1].twinx()
    second.plot(
        x,
        compute["control_tick_only_denoiser_speed_ceiling"],
        color="#D55E00",
        marker="D",
        linewidth=1.8,
        label="Denoiser ceiling",
    )
    second.axhline(1.2, color="#D55E00", linestyle="--", linewidth=0.8, alpha=0.7)
    second.set_ylabel("Ideal denoiser speedup (x)")
    handles1, labels1 = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axes[1, 1].legend(handles1 + handles2, labels1 + labels2, frameon=False)
    axes[1, 1].text(-0.14, 1.04, "d", transform=axes[1, 1].transAxes, weight="bold")
    style_axis(axes[1, 1])
    second.spines["top"].set_visible(False)

    for suffix in ["png", "pdf", "svg"]:
        figure.savefig(
            FIGURE_ROOT / f"action_dit_transport_cache_summary.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_heatmap(layer_step):
    data = layer_step[layer_step["offset"] == 8]
    matrix = data.groupby(["layer", "flow_point"])["toeplitz_improvement"].mean().unstack()
    figure, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.7)
    axis.set_xlabel("Flow point (early to late)")
    axis.set_ylabel("Decoder layer")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Toeplitz risk reduction vs shifted reuse")
    for suffix in ["png", "pdf", "svg"]:
        figure.savefig(
            FIGURE_ROOT / f"action_dit_transport_cache_layer_step.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main():
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    summary, methods, geometry, suffix = load_results()
    geometry_aggregate, layer_step = save_bound_data(
        summary, methods, geometry, suffix
    )
    plot_summary(summary, methods, geometry_aggregate)
    plot_heatmap(layer_step)


if __name__ == "__main__":
    main()
