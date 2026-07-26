#!/usr/bin/env python3
"""Visualize calibration-frozen geometry sparse attention on held-out cells."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.cells_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["mask"])].append(row)

    figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    markers = {"validation": "o", "test": "^"}
    for (split, mask), selected in sorted(grouped.items()):
        axis.scatter(
            [float(row["effective_execution_density"]) for row in selected],
            [float(row["output_relative_l2_energy_proxy"]) for row in selected],
            s=42,
            alpha=0.75,
            marker=markers.get(split, "s"),
            label=f"{split}: {mask}",
        )
    axis.axhline(0.02, color="#B23A48", linestyle="--", linewidth=1.4, label="2% error gate")
    axis.axvline(0.125, color="#D98E04", linestyle="--", linewidth=1.4, label="12.5% density gate")
    axis.set_xlabel("Effective 64x64 tile execution density")
    axis.set_ylabel("Evaluation multi-head relative L2 energy proxy")
    axis.set_title("Calibration-frozen F81 geometry policy: validation vs test")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(args.out_dir / "geometry_generalization_holdout.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
