#!/usr/bin/env python3
"""Plot held-out geometry defect basis transfer and deployability gates."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-go-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "split",
        "mask",
        "rank",
        "coefficient_oracle_error_max",
        "frozen_basis_energy_p05",
        "subspace_overlap_p05",
        "ridge_error_max",
    }
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    return rows


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in read_rows(args.stop_go_csv) if row["split"] == "test"]
    if not rows:
        raise ValueError("stop/go CSV contains no test rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["mask"]].append(row)
    colors = ["#0b6e75", "#d46b28", "#4467a8", "#9a4d71", "#5a7f36"]
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    for color, (mask, selected) in zip(colors, sorted(grouped.items())):
        ordered = sorted(selected, key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in ordered]
        energy = [100.0 * float(row["frozen_basis_energy_p05"]) for row in ordered]
        overlap = [100.0 * float(row["subspace_overlap_p05"]) for row in ordered]
        coefficient_error = [
            100.0 * float(row["coefficient_oracle_error_max"]) for row in ordered
        ]
        ridge_error = [
            100.0 * float(row["ridge_error_max"])
            if row["ridge_error_max"] not in ("", "nan")
            else float("nan")
            for row in ordered
        ]
        axes[0].plot(ranks, energy, marker="o", color=color, label=mask)
        axes[1].plot(ranks, overlap, marker="o", color=color, label=mask)
        axes[2].plot(
            ranks,
            coefficient_error,
            marker="o",
            color=color,
            linestyle="--",
            label=f"{mask}: coeff oracle",
        )
        axes[2].plot(
            ranks,
            ridge_error,
            marker="s",
            color=color,
            linestyle="-",
            label=f"{mask}: ridge",
        )

    axes[0].axhline(80.0, color="#8b1e1e", linestyle=":", linewidth=1.3)
    axes[1].axhline(50.0, color="#8b1e1e", linestyle=":", linewidth=1.3)
    axes[2].axhline(2.0, color="#8b1e1e", linestyle=":", linewidth=1.3)
    axes[2].axhline(5.0, color="#b7791f", linestyle=":", linewidth=1.3)
    axes[0].set_title("(a) Frozen-basis energy, test p05")
    axes[1].set_title("(b) Subspace overlap, test p05")
    axes[2].set_title("(c) Corrected output error, test max")
    axes[0].set_ylabel("Energy retained (%)")
    axes[1].set_ylabel("Normalized overlap (%)")
    axes[2].set_ylabel("Relative L2 error (%)")
    for axis in axes:
        axis.set_xlabel("Defect rank")
        axis.set_xticks(sorted({int(row["rank"]) for row in rows}))
        axis.grid(alpha=0.2)
    axes[0].set_ylim(bottom=0.0, top=100.0)
    axes[1].set_ylim(bottom=0.0, top=100.0)
    axes[2].set_ylim(bottom=0.0)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[2].legend(fontsize=7.3, loc="best")
    figure.suptitle(
        "F81 geometry attention: cross-sample defect basis transfer",
        fontsize=15,
        y=1.02,
    )
    figure.text(
        0.5,
        -0.02,
        "Coefficient oracle uses held-out dense defects; ridge uses calibration weights and validation-only selection.",
        ha="center",
        fontsize=8.5,
        color="#4a4a4a",
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"geometry_basis_transfer_decision.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
