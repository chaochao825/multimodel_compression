from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "onevision_reader_quotient_stage_a_20260830"
FIGURES = ROOT / "figures"
COLORS = {
    "source": "#0072B2",
    "target": "#D55E00",
    "vsi": "#009E73",
    "source-target": "#CC79A7",
    "source-vsi": "#56B4E9",
    "target-vsi": "#E69F00",
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped_mean_range(
    rows: list[dict[str, str]],
    *,
    key_fields: tuple[str, ...],
    value_field: str,
) -> dict[tuple[str, ...], tuple[float, float, float]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(
            float(row[value_field])
        )
    return {
        key: (float(np.mean(values)), min(values), max(values))
        for key, values in grouped.items()
    }


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    domain_rows = read_rows("domain_spectra.csv")
    bootstrap_rows = read_rows("bootstrap_stability.csv")
    cross_rows = read_rows("cross_domain_overlap.csv")
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))

    axis = axes[0, 0]
    domains = [row["domain"] for row in domain_rows]
    explained = [100.0 * (1.0 - float(row["tail_energy_fraction"])) for row in domain_rows]
    bars = axis.bar(
        np.arange(len(domains)),
        explained,
        color=[COLORS[domain] for domain in domains],
        width=0.66,
    )
    axis.set_xticks(np.arange(len(domains)), [value.capitalize() for value in domains])
    axis.set_ylabel("Rank-456 explained energy (%)")
    axis.set_ylim(95.5, 98.2)
    for bar, row in zip(bars, domain_rows, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"gap {100 * float(row['relative_eigengap']):.3f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    style_axis(axis)
    axis.text(-0.13, 1.04, "a", transform=axis.transAxes, fontweight="bold")

    overlap = grouped_mean_range(
        bootstrap_rows,
        key_fields=("domain", "sample_size"),
        value_field="overlap",
    )
    axis = axes[0, 1]
    sample_sizes = (20, 60, 120)
    for domain in domains:
        values = [overlap[(domain, str(size))] for size in sample_sizes]
        means = np.asarray([value[0] for value in values])
        lower = means - np.asarray([value[1] for value in values])
        upper = np.asarray([value[2] for value in values]) - means
        axis.errorbar(
            sample_sizes,
            means,
            yerr=np.vstack([lower, upper]),
            color=COLORS[domain],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=domain.capitalize(),
        )
    axis.set_xlabel("Videos per covariance estimate")
    axis.set_ylabel("Overlap with 120-video basis")
    axis.set_xticks(sample_sizes)
    axis.set_ylim(0.85, 0.995)
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)
    axis.text(-0.13, 1.04, "b", transform=axis.transAxes, fontweight="bold")

    minimum = grouped_mean_range(
        bootstrap_rows,
        key_fields=("domain", "sample_size"),
        value_field="minimum_squared_cosine",
    )
    axis = axes[1, 0]
    for domain in domains:
        values = [minimum[(domain, str(size))][0] for size in sample_sizes]
        axis.plot(
            sample_sizes,
            values,
            color=COLORS[domain],
            marker="o",
            linewidth=1.8,
            label=domain.capitalize(),
        )
    axis.set_yscale("log")
    axis.set_xlabel("Videos per covariance estimate")
    axis.set_ylabel(r"Mean weakest principal $\cos^2$")
    axis.set_xticks(sample_sizes)
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)
    axis.text(-0.13, 1.04, "c", transform=axis.transAxes, fontweight="bold")

    finite_cross = [row for row in cross_rows if row["sample_size"] != "full"]
    cross_overlap = grouped_mean_range(
        finite_cross,
        key_fields=("left", "right", "sample_size"),
        value_field="overlap",
    )
    full_cross = {
        (row["left"], row["right"]): float(row["overlap"])
        for row in cross_rows
        if row["sample_size"] == "full"
    }
    axis = axes[1, 1]
    for left, right in full_cross:
        label = f"{left.capitalize()}-{right.capitalize()}"
        color = COLORS[f"{left}-{right}"]
        means = [
            cross_overlap[(left, right, str(size))][0] for size in sample_sizes
        ]
        axis.plot(
            sample_sizes,
            means,
            color=color,
            marker="o",
            linewidth=1.8,
            label=label,
        )
        axis.axhline(
            full_cross[(left, right)],
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
        )
    axis.set_xlabel("Videos per domain")
    axis.set_ylabel("Cross-domain rank-456 overlap")
    axis.set_xticks(sample_sizes)
    axis.set_ylim(0.75, 0.88)
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)
    axis.text(-0.13, 1.04, "d", transform=axis.transAxes, fontweight="bold")

    figure.tight_layout(pad=1.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "onevision_reader_quotient_stage_a_spectrum"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = stem.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    normalize_svg(svg_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
