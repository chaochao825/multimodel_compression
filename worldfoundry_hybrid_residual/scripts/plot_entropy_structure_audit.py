#!/usr/bin/env python3
"""Build the function-aware entropy audit dashboard from formal probe CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd


COLORS = {
    "ink": "#20322F",
    "teal": "#087E78",
    "blue": "#2F6690",
    "orange": "#D97834",
    "red": "#B43A4B",
    "gold": "#D6A62E",
    "gray": "#8B918E",
    "light_gray": "#CBD0CC",
    "paper": "#F5F0E7",
}

SIGNAL_LABELS = {
    "q": "Q",
    "k": "K",
    "v": "V",
    "ffn_input": "FFN input",
    "ffn_hidden_post_gelu": "Post-GELU",
    "ffn_output": "FFN output",
}

VARIANT_LABELS = {
    "q_lowpass": "Q low-pass",
    "k_lowpass": "K low-pass",
    "qk_lowpass": "Q+K low-pass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFEFB")
    ax.grid(axis="y", color="#D9DDD8", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA19D")
    ax.spines["bottom"].set_color("#9AA19D")
    ax.tick_params(colors=COLORS["ink"], labelsize=8)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=COLORS["teal"],
        va="top",
    )
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=COLORS["ink"], pad=9)


def save_panel_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    frame.to_csv(output_dir / name, index=False, lineterminator="\n")


def load_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "token": pd.read_csv(
            raw_dir / "token_thw_spectrum_f17_f81_v1" / "token_thw_spectrum.csv"
        ),
        "attention": pd.read_csv(
            raw_dir / "spectral_qk_router_f17_f81_v1" / "spectral_qk_attention.csv"
        ),
        "router": pd.read_csv(
            raw_dir / "spectral_qk_router_f17_f81_v1" / "spectral_qk_router.csv"
        ),
        "mp": pd.read_csv(
            raw_dir / "weight_mp_outliers_wan_ffn6_v1" / "weight_mp_summary.csv"
        ),
        "mixed": pd.read_csv(
            raw_dir / "weight_split_ffn_activation_v1" / "weight_split_activation_metrics.csv"
        ),
        "quant": pd.read_csv(
            raw_dir
            / "ffn_activation_structure_f17_v1"
            / "analysis"
            / "activation_quantization_summary.csv"
        ),
        "latency": pd.read_csv(
            raw_dir / "h200_thw_router_f17_f81_v1" / "h200_thw_router_benchmark.csv"
        ),
        "ffn_spectrum": pd.read_csv(
            raw_dir
            / "ffn_activation_structure_f17_v1"
            / "data"
            / "ffn_token_spectrum.csv"
        ),
        "distribution": pd.read_csv(
            raw_dir
            / "ffn_activation_structure_f17_v1"
            / "analysis"
            / "activation_distribution.csv"
        ),
    }


def panel_a(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = data[
        (data["centering"] == "centered")
        & np.isclose(data["requested_density"], 0.125)
        & data["control"].isin(["original", "token_shuffled"])
    ].copy()
    frame["resolution"] = frame["case"].str.extract(r"(f\d+)", expand=False).str.upper()
    frame = (
        frame.groupby(["resolution", "signal", "control"], as_index=False)["lowpass_retained_energy"]
        .mean()
    )
    order = [(res, signal) for res in ["F17", "F81"] for signal in ["q", "k", "v"]]
    x = np.arange(len(order))
    width = 0.36
    for offset, control, color, label in [
        (-width / 2, "original", COLORS["teal"], "Original THW"),
        (width / 2, "token_shuffled", COLORS["light_gray"], "Token shuffled"),
    ]:
        values = []
        for resolution, signal in order:
            row = frame[
                (frame["resolution"] == resolution)
                & (frame["signal"] == signal)
                & (frame["control"] == control)
            ]
            values.append(float(row["lowpass_retained_energy"].iloc[0]))
        ax.bar(x + offset, values, width, color=color, label=label, edgecolor="none")
    ax.axhline(0.125, color=COLORS["orange"], linestyle="--", linewidth=1.1, label="Random expectation")
    ax.set_xticks(x, [f"{res}\n{SIGNAL_LABELS[sig]}" for res, sig in order])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Energy retained", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
    panel_label(ax, "A", "True THW low-pass energy at 12.5% density")
    save_panel_table(frame, output_dir, "panel_a_thw_energy.csv")
    return frame


def panel_b(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = data[data["case"].str.startswith("f81")].copy()
    frame = (
        frame.groupby(["variant", "requested_density"], as_index=False)["output_relative_l2"]
        .mean()
    )
    palette = {
        "q_lowpass": COLORS["teal"],
        "k_lowpass": COLORS["orange"],
        "qk_lowpass": COLORS["red"],
    }
    for variant in ["q_lowpass", "k_lowpass", "qk_lowpass"]:
        part = frame[frame["variant"] == variant].sort_values("requested_density")
        ax.plot(
            part["requested_density"] * 100,
            part["output_relative_l2"] * 100,
            marker="o",
            linewidth=2,
            markersize=4,
            color=palette[variant],
            label=VARIANT_LABELS[variant],
        )
    ax.set_xlabel("Retained frequency density", fontsize=8)
    ax.set_ylabel("Attention output relative L2", fontsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "B", "F81 energy does not imply softmax fidelity")
    save_panel_table(frame, output_dir, "panel_b_f81_attention_error.csv")
    return frame


def panel_c(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = data[
        data["case"].str.startswith("f81") & (data["router_topk"] == 128)
    ].copy()
    frame = (
        frame.groupby(["variant", "requested_density"], as_index=False)["exact_topk_recall"]
        .mean()
    )
    palette = {
        "q_lowpass": COLORS["teal"],
        "k_lowpass": COLORS["orange"],
        "qk_lowpass": COLORS["red"],
    }
    for variant in ["q_lowpass", "k_lowpass", "qk_lowpass"]:
        part = frame[frame["variant"] == variant].sort_values("requested_density")
        ax.plot(
            part["requested_density"] * 100,
            part["exact_topk_recall"] * 100,
            marker="s",
            linewidth=2,
            markersize=4,
            color=palette[variant],
            label=VARIANT_LABELS[variant],
        )
    ax.set_xlabel("Retained frequency density", fontsize=8)
    ax.set_ylabel("Exact top-128 recall", fontsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    panel_label(ax, "C", "Q spectrum is useful only as a coarse router")
    save_panel_table(frame, output_dir, "panel_c_f81_router_recall.csv")
    return frame


def panel_d(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    variants = ["original_centered", "entry_shuffled", "matched_gaussian", "biwhitened"]
    labels = ["Original", "Entry\nshuffle", "Gaussian", "Biwhitened"]
    colors = [COLORS["teal"], COLORS["light_gray"], COLORS["light_gray"], COLORS["blue"]]
    frame = data[data["variant"].isin(variants)].copy()
    summary = (
        frame.groupby("variant", as_index=False)["above_mp_upper_5pct"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    x = np.arange(len(variants))
    means = [float(summary.loc[summary.variant == item, "mean"].iloc[0]) for item in variants]
    ax.bar(x, means, color=colors, width=0.65, edgecolor="none", alpha=0.9)
    offsets = np.linspace(-0.13, 0.13, frame["weight_key"].nunique())
    for idx, variant in enumerate(variants):
        values = frame[frame["variant"] == variant]["above_mp_upper_5pct"].to_numpy()
        ax.scatter(
            idx + offsets[: len(values)],
            values,
            s=18,
            color=COLORS["ink"],
            alpha=0.72,
            zorder=3,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Eigenvalues above MP edge", fontsize=8)
    ax.text(
        0.03,
        0.95,
        "6 Wan FFN matrices",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        color=COLORS["ink"],
    )
    panel_label(ax, "D", "FFN weights reject a random-matrix null")
    save_panel_table(frame, output_dir, "panel_d_mp_outliers.csv")
    return summary


def panel_e(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = (
        data.groupby(["method", "rank", "stored_bits_ratio_vs_bf16"], as_index=False)[
            "activation_output_relative_l2"
        ]
        .mean()
        .sort_values(["method", "rank"])
    )
    methods = {
        "spectral_fp16_plus_int4": ("Spectral tail + INT4", COLORS["teal"], "o"),
        "random_fp16_plus_int4": ("Random tail + INT4", COLORS["gray"], "o"),
    }
    for method, (label, color, marker) in methods.items():
        part = frame[frame["method"] == method].sort_values("rank")
        ax.plot(
            part["stored_bits_ratio_vs_bf16"] * 100,
            part["activation_output_relative_l2"] * 100,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
            label=label,
        )
        for row in part.itertuples():
            ax.annotate(
                f"r{int(row.rank)}",
                (row.stored_bits_ratio_vs_bf16 * 100, row.activation_output_relative_l2 * 100),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
                color=color,
            )
    points = {
        "int4_groupwise": ("INT4 group", COLORS["orange"], "D"),
        "fp8_tensor": ("FP8 tensor", COLORS["blue"], "s"),
    }
    for method, (label, color, marker) in points.items():
        row = frame[frame["method"] == method].iloc[0]
        ax.scatter(
            row["stored_bits_ratio_vs_bf16"] * 100,
            row["activation_output_relative_l2"] * 100,
            color=color,
            marker=marker,
            s=50,
            label=label,
            zorder=4,
        )
    ax.set_xlabel("Stored weight bits vs BF16", fontsize=8)
    ax.set_ylabel("Held-out FFN output relative L2", fontsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    panel_label(ax, "E", "Spectral outliers help, but do not rescue INT4")
    save_panel_table(frame, output_dir, "panel_e_weight_split_activation_error.csv")
    return frame


def panel_f(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    schemes = ["tensor_global", "tensor_bucket", "tensor_step", "token_group128_dynamic"]
    frame = data[
        data["scheme"].isin(schemes) & np.isclose(data["scale_margin"], 1.0)
    ].copy()
    frame = (
        frame.groupby(["dtype", "scheme"], as_index=False)["mean_relative_l2"]
        .mean()
    )
    labels = ["Global", "5-step\nbucket", "Per-step", "Token\ngroup-128"]
    palette = {"fp8_e4m3": COLORS["blue"], "int8": COLORS["teal"], "int4": COLORS["red"]}
    dtype_labels = {"fp8_e4m3": "FP8 E4M3", "int8": "INT8", "int4": "INT4"}
    x = np.arange(len(schemes))
    for dtype in ["fp8_e4m3", "int8", "int4"]:
        values = [
            float(frame[(frame.dtype == dtype) & (frame.scheme == scheme)]["mean_relative_l2"].iloc[0])
            for scheme in schemes
        ]
        ax.plot(
            x,
            np.asarray(values) * 100,
            marker="o",
            linewidth=2,
            color=palette[dtype],
            label=dtype_labels[dtype],
        )
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out activation relative L2", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "F", "Timestep scales help INT8, not FP8 precision")
    save_panel_table(frame, output_dir, "panel_f_activation_quantization.csv")
    return frame


def panel_g(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    operations = [
        "q_thw_lowpass_roundtrip_fp32",
        "qk_thw_lowpass_roundtrip_fp32",
        "q_spatial_pool2_bf16",
        "qk_spatial_pool2_bf16",
    ]
    labels = ["Q FFT", "Q+K FFT", "Q pool2", "Q+K pool2"]
    frame = data[data["operation"].isin(operations)].copy()
    frame["resolution"] = frame["case"].str.extract(r"(f\d+)", expand=False).str.upper()
    x = np.arange(len(operations))
    width = 0.34
    for offset, resolution, color in [
        (-width / 2, "F17", COLORS["orange"]),
        (width / 2, "F81", COLORS["teal"]),
    ]:
        values = [
            float(
                frame[(frame.resolution == resolution) & (frame.operation == operation)][
                    "latency_ratio_vs_fa3"
                ].iloc[0]
            )
            for operation in operations
        ]
        ax.bar(x + offset, values, width, color=color, label=resolution, edgecolor="none")
    ax.axhline(1.0, color=COLORS["red"], linewidth=1, linestyle="--", label="FA3 attention")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 2.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Latency / FA3 BF16 attention", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}x"))
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    panel_label(ax, "G", "H200 cost gate favors pooling over online FFT")
    save_panel_table(frame, output_dir, "panel_g_h200_router_cost.csv")
    return frame


def panel_h(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = data[
        np.isclose(data["requested_density"], 0.125)
        & data["control"].isin(["original", "token_shuffled"])
    ].copy()
    summary = (
        frame.groupby(["signal", "control"], as_index=False)["lowpass_retained_energy"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    signals = ["ffn_input", "ffn_hidden_post_gelu", "ffn_output"]
    x = np.arange(len(signals))
    width = 0.36
    for offset, control, color, label in [
        (-width / 2, "original", COLORS["teal"], "Original THW"),
        (width / 2, "token_shuffled", COLORS["light_gray"], "Token shuffled"),
    ]:
        means = []
        lows = []
        highs = []
        for signal in signals:
            row = summary[(summary.signal == signal) & (summary.control == control)].iloc[0]
            means.append(row["mean"])
            lows.append(row["mean"] - row["min"])
            highs.append(row["max"] - row["mean"])
        ax.bar(x + offset, means, width, color=color, edgecolor="none", label=label)
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack([lows, highs]),
            fmt="none",
            ecolor=COLORS["ink"],
            elinewidth=0.8,
            capsize=2,
        )
    ax.axhline(0.125, color=COLORS["orange"], linestyle="--", linewidth=1)
    ax.set_xticks(x, [SIGNAL_LABELS[item] for item in signals])
    ax.set_ylabel("Energy retained", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_ylim(0, 0.86)
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "H", "FFN activations have nonstationary THW structure")
    save_panel_table(summary, output_dir, "panel_h_ffn_activation_thw.csv")
    return summary


def panel_i(ax: plt.Axes, data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = data[data["run_id"] == "p00_seed20260723"].copy()
    summary = (
        frame.groupby(["block", "signal"], as_index=False)["abs_max"]
        .agg(lambda values: values.max() / values.min())
        .rename(columns={"abs_max": "max_abs_scale_drift"})
    )
    blocks = [0, 12, 24, 29]
    signals = ["ffn_input", "ffn_hidden_post_gelu", "ffn_output"]
    matrix = np.asarray(
        [
            [
                float(
                    summary[(summary.block == block) & (summary.signal == signal)][
                        "max_abs_scale_drift"
                    ].iloc[0]
                )
                for signal in signals
            ]
            for block in blocks
        ]
    )
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=1.0, vmax=4.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > 2.55 else COLORS["ink"]
            ax.text(col, row, f"{matrix[row, col]:.2f}x", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(np.arange(len(signals)), [SIGNAL_LABELS[item] for item in signals], rotation=12)
    ax.set_yticks(np.arange(len(blocks)), [f"Block {block}" for block in blocks])
    ax.grid(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("max / min", fontsize=7)
    panel_label(ax, "I", "One fixed FFN scale is not trajectory-stable")
    save_panel_table(summary, output_dir, "panel_i_scale_drift.csv")
    return summary


def write_evidence_summary(output_dir: Path, panels: dict[str, pd.DataFrame]) -> None:
    a = panels["a"]
    b = panels["b"]
    c = panels["c"]
    e = panels["e"]
    g = panels["g"]

    def value(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
        return float(frame.loc[mask, column].iloc[0])

    rows = [
        {
            "metric": "f81_q_centered_thw_energy_density_12p5",
            "value": value(a, (a.resolution == "F81") & (a.signal == "q") & (a.control == "original"), "lowpass_retained_energy"),
            "unit": "ratio",
        },
        {
            "metric": "f81_k_centered_thw_energy_density_12p5",
            "value": value(a, (a.resolution == "F81") & (a.signal == "k") & (a.control == "original"), "lowpass_retained_energy"),
            "unit": "ratio",
        },
        {
            "metric": "f81_v_centered_thw_energy_density_12p5",
            "value": value(a, (a.resolution == "F81") & (a.signal == "v") & (a.control == "original"), "lowpass_retained_energy"),
            "unit": "ratio",
        },
        {
            "metric": "f81_q_lowpass_attention_output_error_density_12p5",
            "value": value(b, (b.variant == "q_lowpass") & np.isclose(b.requested_density, 0.125), "output_relative_l2"),
            "unit": "ratio",
        },
        {
            "metric": "f81_k_lowpass_attention_output_error_density_12p5",
            "value": value(b, (b.variant == "k_lowpass") & np.isclose(b.requested_density, 0.125), "output_relative_l2"),
            "unit": "ratio",
        },
        {
            "metric": "f81_q_lowpass_top128_recall_density_12p5",
            "value": value(c, (c.variant == "q_lowpass") & np.isclose(c.requested_density, 0.125), "exact_topk_recall"),
            "unit": "ratio",
        },
        {
            "metric": "int4_groupwise_ffn_output_error",
            "value": value(e, e.method == "int4_groupwise", "activation_output_relative_l2"),
            "unit": "ratio",
        },
        {
            "metric": "spectral_r16_plus_int4_ffn_output_error",
            "value": value(
                e,
                (e["method"] == "spectral_fp16_plus_int4") & (e["rank"] == 16),
                "activation_output_relative_l2",
            ),
            "unit": "ratio",
        },
        {
            "metric": "f81_q_fft_latency_vs_fa3",
            "value": value(g, (g.resolution == "F81") & (g.operation == "q_thw_lowpass_roundtrip_fp32"), "latency_ratio_vs_fa3"),
            "unit": "ratio",
        },
        {
            "metric": "f81_q_pool2_latency_vs_fa3",
            "value": value(g, (g.resolution == "F81") & (g.operation == "q_spatial_pool2_bf16"), "latency_ratio_vs_fa3"),
            "unit": "ratio",
        },
    ]
    pd.DataFrame(rows).to_csv(output_dir / "evidence_summary.csv", index=False, lineterminator="\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.raw_dir)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(17.2, 13.4), constrained_layout=False)
    fig.patch.set_facecolor(COLORS["paper"])
    for ax in axes.flat:
        style_axis(ax)

    panels = {
        "a": panel_a(axes[0, 0], data["token"], args.output_dir),
        "b": panel_b(axes[0, 1], data["attention"], args.output_dir),
        "c": panel_c(axes[0, 2], data["router"], args.output_dir),
        "d": panel_d(axes[1, 0], data["mp"], args.output_dir),
        "e": panel_e(axes[1, 1], data["mixed"], args.output_dir),
        "f": panel_f(axes[1, 2], data["quant"], args.output_dir),
        "g": panel_g(axes[2, 0], data["latency"], args.output_dir),
        "h": panel_h(axes[2, 1], data["ffn_spectrum"], args.output_dir),
        "i": panel_i(axes[2, 2], data["distribution"], args.output_dir),
    }
    write_evidence_summary(args.output_dir, panels)

    fig.subplots_adjust(left=0.065, right=0.975, top=0.965, bottom=0.07, wspace=0.28, hspace=0.39)
    fig.savefig(args.output_dir / "entropy_structure_audit_dashboard.png", dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(args.output_dir / "entropy_structure_audit_dashboard.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
