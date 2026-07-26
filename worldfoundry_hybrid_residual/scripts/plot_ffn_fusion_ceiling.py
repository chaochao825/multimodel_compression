#!/usr/bin/env python3
"""Plot local and end-to-end Wan FFN fusion ceilings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("fusion ceiling CSV is empty")
    return rows


def grouped_bars(
    axis: plt.Axes,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    *,
    ylabel: str,
) -> None:
    positions = list(range(len(labels)))
    width = 0.76 / len(series)
    start = -(len(series) - 1) * width / 2
    for index, (name, values, color) in enumerate(series):
        offsets = [position + start + index * width for position in positions]
        bars = axis.bar(offsets, values, width=width, label=name, color=color)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    axis.set_xticks(positions, labels)
    axis.set_ylabel(ylabel)
    axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle=":")
    axis.grid(axis="y", alpha=0.2)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["case"] for row in rows]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    grouped_bars(
        axes[0],
        labels,
        [
            (
                "remove hidden traffic",
                [float(row["ideal_intermediate_traffic_local_speedup"]) for row in rows],
                "#4477AA",
            ),
            (
                "remove full epilogue",
                [float(row["ideal_epilogue_local_speedup"]) for row in rows],
                "#228833",
            ),
            (
                "standalone Triton",
                [float(row["standalone_triton_projected_local_speedup"]) for row in rows],
                "#CC6677",
            ),
        ],
        ylabel="Complete-FFN speedup (x)",
    )
    axes[0].set_title("(a) Measured local ceilings")
    axes[0].legend(fontsize=7, frameon=False)

    grouped_bars(
        axes[1],
        labels,
        [
            (
                "remove hidden traffic",
                [float(row["ideal_intermediate_traffic_e2e_speedup"]) for row in rows],
                "#4477AA",
            ),
            (
                "remove full epilogue",
                [float(row["ideal_epilogue_e2e_speedup"]) for row in rows],
                "#228833",
            ),
            (
                "remove entire FFN",
                [float(row["remove_entire_ffn_e2e_ceiling"]) for row in rows],
                "#AA3377",
            ),
            (
                "remove all elementwise",
                [float(row["remove_all_elementwise_e2e_ceiling"]) for row in rows],
                "#EE7733",
            ),
        ],
        ylabel="Estimated end-to-end speedup (x)",
    )
    axes[1].set_title("(b) Amdahl ceilings")
    axes[1].legend(fontsize=6.7, frameon=False)

    share_series = [
        (
            "self-attention",
            [100.0 * float(row["profile_self_attention_share"]) for row in rows],
            "#0077BB",
        ),
        (
            "elementwise/memory",
            [100.0 * float(row["profile_elementwise_share"]) for row in rows],
            "#EE7733",
        ),
        (
            "linear GEMM",
            [100.0 * float(row["profile_linear_share"]) for row in rows],
            "#009988",
        ),
        (
            "estimated FFN subset",
            [100.0 * float(row["estimated_ffn_share"]) for row in rows],
            "#CC3311",
        ),
    ]
    positions = list(range(len(labels)))
    width = 0.76 / len(share_series)
    start = -(len(share_series) - 1) * width / 2
    for index, (name, values, color) in enumerate(share_series):
        axes[2].bar(
            [position + start + index * width for position in positions],
            values,
            width=width,
            label=name,
            color=color,
        )
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylabel("Incremental denoise runtime share (%)")
    axes[2].set_title("(c) Where the runtime remains")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(fontsize=6.7, frameon=False)

    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"ffn_fusion_ceiling.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
