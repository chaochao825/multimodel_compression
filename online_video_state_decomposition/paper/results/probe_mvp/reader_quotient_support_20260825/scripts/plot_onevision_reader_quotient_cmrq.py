from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = (
    ROOT
    / "analysis"
    / "onevision_reader_quotient_stage_a_20260830"
    / "cmrq_analysis"
)
FIGURES = ROOT / "figures"
COLORS = {
    "pooled3_pca_r456": "#0072B2",
    "vsi_pca_train96_r456": "#009E73",
    "cmrq_risk_atoms32_r456": "#D55E00",
    "cmrq_mix_g32_w0p3_r456": "#CC79A7",
    "feature_only_null_atoms32_r456": "#56B4E9",
    "random_null_atoms32_r456": "#999999",
    "permuted_risk_null_atoms32_r456": "#E69F00",
    "permuted_mix_g32_w0p3_r456": "#F0E442",
}
LABELS = {
    "pooled3_pca_r456": "Pooled PCA",
    "vsi_pca_train96_r456": "VSI PCA",
    "cmrq_risk_atoms32_r456": "Risk-32",
    "cmrq_mix_g32_w0p3_r456": "CMRQ mix",
    "feature_only_null_atoms32_r456": "Feature null",
    "random_null_atoms32_r456": "Random null",
    "permuted_risk_null_atoms32_r456": "Permuted risk",
    "permuted_mix_g32_w0p3_r456": "Permuted mix",
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.6)


def panel_label(axis: plt.Axes, value: str) -> None:
    axis.text(-0.14, 1.04, value, transform=axis.transAxes, fontweight="bold")


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def plot_feature_risk(axis: plt.Axes) -> None:
    rows = read_rows("feature_risk_tradeoff.csv")
    labels = {
        "pooled3_pca_r456": "PCA",
        "risk_only_r456": "Risk only",
        "cmrq_risk_atoms16_r456": "+16",
        "cmrq_risk_atoms32_r456": "+32",
        "cmrq_risk_atoms64_r456": "+64",
        "cmrq_risk_atoms96_r456": "+96",
    }
    for row in rows:
        method = row["method"]
        feature = 100.0 * float(row["pooled_feature_capture"])
        risk = 100.0 * float(row["reader_risk_capture"])
        color = "#D55E00" if method.startswith("cmrq_") else "#0072B2"
        if method == "risk_only_r456":
            color = "#999999"
        axis.scatter(feature, risk, s=48, color=color, edgecolor="white", zorder=3)
        axis.annotate(
            labels[method],
            (feature, risk),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Pooled feature energy captured (%)")
    axis.set_ylabel("Reader-risk trace captured (%)")
    axis.set_xlim(24, 99)
    axis.set_ylim(20, 94)
    style_axis(axis)
    panel_label(axis, "a")


def plot_risk_overlap(axis: plt.Axes) -> None:
    rows = read_rows("reader_risk_stability.csv")
    labels = ("fold0", "fold1", "fold2")
    matrix = np.eye(3)
    index = {label: position for position, label in enumerate(labels)}
    for row in rows:
        left = index[row["left"]]
        right = index[row["right"]]
        matrix[left, right] = matrix[right, left] = float(row["overlap_r32"])
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="cividis")
    axis.set_xticks(range(3), labels)
    axis.set_yticks(range(3), labels)
    axis.set_xlabel("Risk-fit split")
    axis.set_ylabel("Risk-fit split")
    for row in range(3):
        for column in range(3):
            color = "white" if matrix[row, column] < 0.55 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Top-32 overlap $\|U_i^\top U_j\|_F^2/32$")
    panel_label(axis, "b")


def plot_crossfit_pareto(axis: plt.Axes) -> None:
    rows = read_rows("cmrq_crossfit_methods.csv")
    for row in rows:
        method = row["method"]
        mean = 1_000.0 * float(row["candidate_kl_mean"])
        p95 = 1_000.0 * float(row["candidate_kl_p95"])
        harmful = int(row["harmful_count"])
        marker = "X" if harmful else "o"
        axis.scatter(
            mean,
            p95,
            s=58,
            marker=marker,
            color=COLORS[method],
            edgecolor="black" if marker == "X" else "white",
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            LABELS[method],
            (mean, p95),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    axis.set_xlabel(r"Mean candidate KL ($\times 10^{-3}$)")
    axis.set_ylabel(r"P95 candidate KL ($\times 10^{-3}$)")
    axis.text(
        0.98,
        0.04,
        "X: at least one harmful flip",
        transform=axis.transAxes,
        ha="right",
        fontsize=8,
    )
    style_axis(axis)
    panel_label(axis, "c")


def plot_progressive(axis: plt.Axes) -> None:
    rows = read_rows("progressive_fallback_curve.csv")
    methods = (
        "pooled3_pca_r456",
        "vsi_pca_train96_r456",
        "cmrq_risk_atoms32_r456",
        "cmrq_mix_g32_w0p3_r456",
    )
    for method in methods:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["fallback_rate"]),
        )
        fallback = [100.0 * float(row["fallback_rate"]) for row in selected]
        ratio = [float(row["conservative_transfer_ratio"]) for row in selected]
        axis.plot(
            fallback,
            ratio,
            marker="o",
            linewidth=1.7,
            color=COLORS[method],
            label=LABELS[method],
        )
        first = selected[0]
        axis.annotate(
            f"M{first['remaining_mismatch_count']}/H{first['remaining_harmful_count']}",
            (fallback[0], ratio[0]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7.5,
            color=COLORS[method],
        )
    axis.set_xlabel("Exact fallback rate (%)")
    axis.set_ylabel("Conservative state-transfer ratio")
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    axis.text(
        0.02,
        0.04,
        "M/H: remaining mismatch/harmful at margin = 0",
        transform=axis.transAxes,
        fontsize=7.7,
    )
    style_axis(axis)
    panel_label(axis, "d")


def main() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.7))
    plot_feature_risk(axes[0, 0])
    plot_risk_overlap(axes[0, 1])
    plot_crossfit_pareto(axes[1, 0])
    plot_progressive(axes[1, 1])
    figure.tight_layout(pad=1.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "onevision_reader_quotient_cmrq_stage_b"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = stem.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    normalize_svg(svg_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
