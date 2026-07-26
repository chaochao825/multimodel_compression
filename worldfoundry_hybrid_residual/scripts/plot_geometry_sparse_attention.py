#!/usr/bin/env python3
"""Plot geometry-sparse attention density, tail, and dense-fallback trade-offs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = read_rows(input_dir / "geometry_attention_fallback_curve.csv")
    heads = read_rows(input_dir / "geometry_attention_heads.csv")

    cases = sorted({row["case"] for row in curve})
    for case in cases:
        figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
        selected = [row for row in curve if row["case"] == case]
        groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in selected:
            groups[(row["mask"], int(row["tail_rank"]))].append(row)
        for (mask, rank), rows in sorted(groups.items()):
            rows.sort(key=lambda row: int(row["dense_fallback_heads"]))
            if rank in (0, 8, 16):
                axes[0].plot(
                    [float(row["effective_attention_density"]) for row in rows],
                    [float(row["output_relative_l2"]) for row in rows],
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=f"{mask}, r={rank}",
                )
        axes[0].axhline(0.02, color="#B23A48", linestyle="--", linewidth=1)
        axes[0].axhline(0.05, color="#D98E04", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Effective attention density")
        axes[0].set_ylabel("Pre-output-projection relative L2")
        axes[0].set_title(f"{case}: sparse + tail + dense fallback")
        axes[0].set_yscale("log")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=6, ncol=2)

        head_selected = [row for row in heads if row["case"] == case]
        head_groups: dict[str, list[float]] = defaultdict(list)
        density: dict[str, float] = {}
        for row in head_selected:
            head_groups[row["mask"]].append(float(row["exact_attention_mass_mean"]))
            density[row["mask"]] = float(row["execution_density"])
        names = sorted(head_groups, key=density.get)
        axes[1].plot(
            [density[name] for name in names],
            [sum(head_groups[name]) / len(head_groups[name]) for name in names],
            color="#0B6E75",
            marker="s",
            linewidth=2,
        )
        for name in names:
            axes[1].annotate(
                name.split("_", 1)[0],
                (density[name], sum(head_groups[name]) / len(head_groups[name])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1].set_xlabel("64x64 tile execution density")
        axes[1].set_ylabel("Exact attention mass retained")
        axes[1].set_title("Content-independent mask coverage")
        axes[1].grid(alpha=0.25)
        figure.savefig(output_dir / f"geometry_sparse_pareto_{case}.png", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
