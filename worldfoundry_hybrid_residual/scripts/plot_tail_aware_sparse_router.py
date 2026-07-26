#!/usr/bin/env python3
"""Compare fixed-objective sparse routers with tail-aware alternation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--tail-aware-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise SystemExit("matplotlib is required in the experiment environment") from error
    baseline = [
        row
        for row in read_rows(args.baseline_summary)
        if int(row["rank"]) == args.rank and row["normalization"] == "renormalized"
    ]
    tail = [row for row in read_rows(args.tail_aware_summary) if int(row["rank"]) == args.rank]
    comparison = []
    for tail_row in sorted(tail, key=lambda row: float(row["density"])):
        density = float(tail_row["density"])
        candidates = [row for row in baseline if float(row["density"]) == density]
        best_dynamic = min(candidates, key=lambda row: float(row["test_oracle_self_relative_l2"]))
        best_static = min(
            candidates,
            key=lambda row: float(row["test_frozen_calibration_basis_relative_l2"]),
        )
        initial = float(tail_row["test_dynamic_initial_relative_l2"])
        final = float(tail_row["test_dynamic_final_relative_l2"])
        comparison.append(
            {
                "density": density,
                "rank": args.rank,
                "baseline_dynamic_method": best_dynamic["method"],
                "baseline_dynamic_relative_l2": float(best_dynamic["test_oracle_self_relative_l2"]),
                "tail_aware_initial_relative_l2": initial,
                "tail_aware_final_relative_l2": final,
                "tail_aware_relative_improvement": (initial - final) / max(initial, 1e-30),
                "baseline_static_method": best_static["method"],
                "baseline_static_relative_l2": float(best_static["test_frozen_calibration_basis_relative_l2"]),
                "tail_aware_static_relative_l2": float(tail_row["test_frozen_calibration_basis_relative_l2"]),
                "mask_jaccard": float(tail_row["mask_jaccard_mean"]),
                "basis_overlap": float(tail_row["basis_overlap_mean"]),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "tail_aware_router_comparison.csv", comparison)
    density = [100 * float(row["density"]) for row in comparison]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].plot(
        density,
        [100 * float(row["baseline_dynamic_relative_l2"]) for row in comparison],
        marker="o",
        label="Best fixed-objective router",
        color="#0B6E75",
    )
    axes[0].plot(
        density,
        [100 * float(row["tail_aware_final_relative_l2"]) for row in comparison],
        marker="s",
        label="Tail-aware alternating",
        color="#D97941",
    )
    axes[0].axhline(2, color="#333333", linestyle=":", label="2% target")
    axes[0].set_title("Dynamic representation witness")
    axes[0].set_ylabel("Output relative L2 (%)")
    axes[0].set_yscale("log")

    axes[1].bar(
        [str(value) for value in density],
        [100 * float(row["tail_aware_relative_improvement"]) for row in comparison],
        color="#4F6D2F",
    )
    axes[1].set_title("Alternation gain is marginal")
    axes[1].set_ylabel("Relative improvement (%)")

    axes[2].plot(
        density,
        [float(row["mask_jaccard"]) for row in comparison],
        marker="o",
        label="Mask Jaccard",
        color="#557A95",
    )
    axes[2].plot(
        density,
        [float(row["basis_overlap"]) for row in comparison],
        marker="s",
        label="Basis overlap",
        color="#B23A48",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Cross-seed transfer structure")
    axes[2].set_ylabel("Overlap")
    for axis in axes:
        axis.set_xlabel("Selected key-block density (%)")
        axis.grid(True, alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Tail-aware sparse routing audit, rank-{args.rank}", fontsize=14, fontweight="bold")
    fig.savefig(args.output_dir / "tail_aware_sparse_router.png", dpi=180)
    fig.savefig(args.output_dir / "tail_aware_sparse_router.pdf")
    print(f"[plot] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
