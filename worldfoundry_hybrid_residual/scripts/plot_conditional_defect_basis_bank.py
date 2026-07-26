#!/usr/bin/env python3
"""Visualize the cross-seed position-bucketed defect basis-bank probe."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise SystemExit("matplotlib is required in the experiment environment") from error
    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["bank_count"]))
    banks = [int(row["bank_count"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    axes[0].plot(
        banks,
        [100 * float(row["aggregate_relative_l2"]) for row in rows],
        marker="o",
        color="#0B6E75",
        label="Aggregate",
    )
    axes[0].plot(
        banks,
        [100 * float(row["head_error_max"]) for row in rows],
        marker="s",
        color="#B23A48",
        label="Worst head",
    )
    axes[0].axhline(2, color="#333333", linestyle=":", label="2% target")
    axes[0].set_ylabel("Output relative L2 (%)")
    axes[0].set_title("Position banks do not close transfer gap")
    axes[0].set_yscale("log")

    axes[1].plot(
        banks,
        [float(row["basis_overlap_mean"]) for row in rows],
        marker="o",
        color="#D97941",
        label="Cross-seed basis overlap",
    )
    axes[1].plot(
        banks,
        [float(row["captured_energy_mean"]) for row in rows],
        marker="s",
        color="#4F6D2F",
        label="Captured test energy",
    )
    axes[1].set_ylim(0.6, 1.0)
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Smaller calibration buckets overfit")

    axes[2].bar(
        [str(bank) for bank in banks],
        [float(row["basis_fp16_mib"]) for row in rows],
        color="#557A95",
    )
    axes[2].set_ylabel("Basis storage (MiB, FP16)")
    axes[2].set_title("Storage grows without quality payoff")
    for axis in axes:
        axis.set_xlabel("Position basis-bank count")
        axis.grid(True, alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Wan F81 frozen-mask rank-16 basis-bank audit", fontsize=14, fontweight="bold")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "conditional_defect_basis_bank.png", dpi=180)
    fig.savefig(args.output_dir / "conditional_defect_basis_bank.pdf")
    print(f"[plot] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
