#!/usr/bin/env python3
"""Plot the EXP-055 component gain and complete-request boundary."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


COLORS = {
    "eager": "#526777",
    "graph": "#D36B3F",
    "denoiser": "#4C78A8",
    "vae": "#F58518",
    "transfer": "#54A24B",
    "serialization": "#E45756",
    "text": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def main() -> None:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    component = read_csv(evaluation_dir / "f81_component_rows.csv")
    request = read_csv(evaluation_dir / "f81_request_rows.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), constrained_layout=True)

    labels = [
        f"P{int(row['prompt_index'])}/R{int(row['round'])}" for row in component
    ]
    x = list(range(len(labels)))
    width = 0.38
    axes[0].bar(
        [value - width / 2 for value in x],
        [float(row["eager_seconds"]) for row in component],
        width,
        color=COLORS["eager"],
        label="Eager",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        [float(row["graph_seconds"]) for row in component],
        width,
        color=COLORS["graph"],
        label="CUDA Graph",
    )
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylabel("Complete VAE latency (s)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(
        0.98,
        0.96,
        f"median {median(component, 'eager_seconds') / median(component, 'graph_seconds'):.3f}x",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        weight="bold",
    )

    request_x = list(range(len(request)))
    request_labels = [f"P{int(row['prompt_index'])}" for row in request]
    axes[1].bar(
        [value - width / 2 for value in request_x],
        [float(row["eager_request_seconds"]) for row in request],
        width,
        color=COLORS["eager"],
        label="Paired eager",
    )
    axes[1].bar(
        [value + width / 2 for value in request_x],
        [float(row["graph_request_seconds"]) for row in request],
        width,
        color=COLORS["graph"],
        label="CUDA Graph",
    )
    axes[1].axhline(9.637995031895116, color="#333333", ls="--", lw=1.1, label="Incumbent")
    axes[1].axhline(9.1790428875, color="#B22222", ls=":", lw=1.4, label="1.05x gate")
    axes[1].set_xticks(request_x, request_labels)
    axes[1].set_ylabel("Resident request latency (s)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    axes[1].text(
        0.98,
        0.05,
        "absolute 1.033x",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        weight="bold",
    )

    stacks = (
        ("text", "Text"),
        ("denoiser", "Denoiser"),
        ("vae", "VAE"),
        ("transfer", "D2H"),
        ("serialization", "Serialize"),
    )
    bottoms = [0.0, 0.0]
    for key, label in stacks:
        values = [
            median(request, f"eager_{key}_seconds"),
            median(request, f"graph_{key}_seconds"),
        ]
        axes[2].bar(
            [0, 1],
            values,
            bottom=bottoms,
            width=0.58,
            color=COLORS[key],
            label=label,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axes[2].set_xticks([0, 1], ["Paired eager", "CUDA Graph"])
    axes[2].set_ylabel("Median component time (s)")
    axes[2].legend(frameon=False, fontsize=8, ncol=2, loc="upper center")

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.12, 1.03, label, transform=axis.transAxes, weight="bold", fontsize=12)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)

    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"wan_rcm_vae_cudagraph_exp055.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
