#!/usr/bin/env python3
"""Plot Hessian error Gram, matched-rate gains, and Taylor validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "remote_logs" / "hessian_orthogonal_compression_20260712.json"
DEFAULT_OUTPUT_DIR = ROOT / "figures"
METHODS = ("structured", "pruning", "quantization")
LABELS = ("Structured", "Pruning", "Quantization")
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
MAGENTA = "#CC79A7"
GRAY = "#666666"


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "mathtext.fontset": "stix",
        }
    )


def gram_matrix(summary: Sequence[Mapping[str, object]], metric: str, floor: float | None) -> np.ndarray:
    matrix = np.eye(len(METHODS), dtype=np.float64)
    index = {method: idx for idx, method in enumerate(METHODS)}
    for row in summary:
        if row["metric"] != metric:
            continue
        current_floor = row["probability_floor"]
        if floor is None:
            if current_floor is not None:
                continue
        elif current_floor is None or not np.isclose(float(current_floor), floor):
            continue
        left = index[str(row["left"])]
        right = index[str(row["right"])]
        matrix[left, right] = matrix[right, left] = float(row["mean_hessian_cosine"])
    return matrix


def draw_heatmap(ax: plt.Axes, matrix: np.ndarray, panel: str, subtitle: str) -> None:
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=24, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_title(f"({panel}) {subtitle}", loc="left", fontweight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if abs(matrix[row, col]) > 0.62 else "black"
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=color)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    return image


def matched_summary(
    payload: Mapping[str, object], metric: str, comparison_scope: str
) -> list[Mapping[str, object]]:
    rows = [
        row
        for row in payload["summary"]["matched_rate"]
        if row["selection_metric"] == metric
        and row["comparison_scope"] == comparison_scope
    ]
    return sorted(rows, key=lambda row: float(row["payload_cap_fraction"]))


def plot(payload: Mapping[str, object], output_dir: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6))

    frobenius = gram_matrix(
        payload["summary"]["error_gram"], "frobenius_hessian", None
    )
    local_kl = gram_matrix(
        payload["summary"]["error_gram"], "local_kl_hessian", 1e-8
    )
    draw_heatmap(axes[0, 0], frobenius, "a", "Compression-error cosine, Frobenius metric")
    draw_heatmap(axes[0, 1], local_kl, "b", "Compression-error cosine, local KL Hessian")

    variants = payload["summary"]["compression_variants"]
    variant_order = (
        ("naive_max", "Max scale", BLUE),
        ("one_scale_max", "1-scale repair", ORANGE),
        ("naive_bounded_cross_null", "Bounded cross-null", GREEN),
        ("naive_loss_optimal", "Loss-optimal scale", MAGENTA),
        ("obs_max", "Full OBS", GRAY),
    )
    x = np.arange(len(variant_order))
    rho = np.asarray(
        [max(float(variants[key]["mean_abs_hessian_correlation"]), 1e-17) for key, _, _ in variant_order]
    )
    bars = axes[0, 2].bar(x, rho, color=[color for _, _, color in variant_order], width=0.72)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xticks(x, [label for _, label, _ in variant_order], rotation=25, ha="right")
    axes[0, 2].set_ylabel(r"Mean $|\rho_H|$ (log)")
    axes[0, 2].set_title("(c) Orthogonality requires the right repair subspace", loc="left", fontweight="bold")
    axes[0, 2].set_ylim(1e-17, max(rho) * 20.0)
    axes[0, 2].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, rho):
        axes[0, 2].text(
            bar.get_x() + bar.get_width() / 2,
            value * (1.8 if value > 1e-12 else 12.0),
            f"{value:.1e}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=35,
        )

    hessian = matched_summary(
        payload, "mean_damped_hessian_quadratic", "taylor_comfort_only"
    )
    caps = 100.0 * np.asarray([row["payload_cap_fraction"] for row in hessian], dtype=np.float64)
    single = np.asarray([row["mean_single_loss"] for row in hessian], dtype=np.float64)
    combo = np.asarray([row["mean_combo_loss"] for row in hessian], dtype=np.float64)
    actual_payload = 100.0 * np.asarray(
        [row["mean_matched_payload_fraction"] for row in hessian], dtype=np.float64
    )
    axes[1, 0].plot(actual_payload[1:], single[1:], "o-", color=BLUE, linewidth=2, label="Best single")
    axes[1, 0].plot(actual_payload[1:], combo[1:], "s-", color=ORANGE, linewidth=2, label="Best combination")
    axes[1, 0].plot(
        actual_payload[0], single[0], marker="o", markerfacecolor="none",
        markeredgecolor=BLUE, markeredgewidth=1.6, linestyle="none"
    )
    axes[1, 0].plot(
        actual_payload[0], combo[0], marker="s", markerfacecolor="none",
        markeredgecolor=ORANGE, markeredgewidth=1.6, linestyle="none"
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Rate-matched budget / dense FP16 (%)")
    axes[1, 0].set_ylabel("Mean damped-Hessian loss (log)")
    axes[1, 0].set_title("(d) <=1% rate match, both endpoints Taylor-valid", loc="left", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].text(
        0.98,
        0.58,
        "Taylor-valid: n=6 at 20%; n=8 otherwise",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="bottom",
        color=GRAY,
        fontsize=8,
    )

    # Both ratios use the exact same Hessian-selected codec pair.  This avoids
    # comparing independently optimized Hessian and KL envelopes.
    hessian_ratio = 1.0 - np.asarray(
        [row["mean_same_selection_relative_hessian_gain"] for row in hessian]
    )
    exact_ratio = 1.0 - np.asarray(
        [row["mean_same_selection_relative_endpoint_kl_gain"] for row in hessian]
    )
    axes[1, 1].plot(caps[1:], hessian_ratio[1:], "o-", color=GREEN, linewidth=2, label="Damped Hessian")
    axes[1, 1].plot(caps[1:], exact_ratio[1:], "s--", color=MAGENTA, linewidth=2, label="Exact endpoint KL")
    axes[1, 1].plot(
        caps[0], hessian_ratio[0], marker="o", markerfacecolor="none",
        markeredgecolor=GREEN, markeredgewidth=1.6, linestyle="none"
    )
    axes[1, 1].plot(
        caps[0], exact_ratio[0], marker="s", markerfacecolor="none",
        markeredgecolor=MAGENTA, markeredgewidth=1.6, linestyle="none"
    )
    axes[1, 1].axhline(1.0, color=GRAY, linestyle=":", linewidth=1.4)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Payload cap / dense FP16 (%)")
    axes[1, 1].set_ylabel("Combination loss / best-single loss (log)")
    axes[1, 1].set_title("(e) Same codecs: Hessian prediction vs endpoint KL", loc="left", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].text(20.5, 1.2, "single wins", color=GRAY, fontsize=8)
    axes[1, 1].text(20.5, 0.42, "combination wins", color=GRAY, fontsize=8)

    all_rows = payload["rows"]
    for category, marker, color, label in (
        ("single_quantization", "o", BLUE, "Quantization only"),
        ("single_pruning", "^", GREEN, "Pruning only"),
        ("combined_prune_quant", "s", ORANGE, "Prune + quantize"),
    ):
        selected = [row for row in all_rows if row["category"] == category]
        quadratic = np.asarray([row["mean_fisher_quadratic"] for row in selected])
        actual = np.asarray([row["mean_actual_kl"] for row in selected])
        axes[1, 2].scatter(
            quadratic,
            actual,
            marker=marker,
            color=color,
            s=14,
            alpha=0.35,
            edgecolors="none",
            label=label,
        )
    positive = [
        float(row[key])
        for row in all_rows
        for key in ("mean_fisher_quadratic", "mean_actual_kl")
        if float(row[key]) > 0.0
    ]
    lower = max(min(positive), 1e-10)
    upper = max(positive)
    line = np.geomspace(lower, upper, 200)
    axes[1, 2].fill_between(line, 0.8 * line, 1.25 * line, color=GRAY, alpha=0.12)
    axes[1, 2].plot(line, line, color=GRAY, linestyle=":", linewidth=1.4)
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("Local Fisher quadratic prediction (log)")
    axes[1, 2].set_ylabel("Exact endpoint KL (log)")
    axes[1, 2].set_title("(f) Hessian claims require endpoint Taylor validity", loc="left", fontweight="bold")
    axes[1, 2].legend(loc="upper left")
    axes[1, 2].grid(alpha=0.2)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.98, wspace=0.30, hspace=0.38)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output_dir / f"fig25_hessian_orthogonal_compression.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    plot(payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
