#!/usr/bin/env python3
"""Summarize the F81 geometry pilot into a compact stop/go decision."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independence-summary", type=Path, required=True)
    parser.add_argument("--generalization-cells", type=Path, required=True)
    parser.add_argument("--basis-stop-go", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    independence = json.loads(args.independence_summary.read_text(encoding="utf-8"))
    cells = read_csv(args.generalization_cells)
    basis = read_csv(args.basis_stop_go)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = [row for row in cells if row["split"] == "validation"]
    sparse_heads: dict[str, list[int]] = defaultdict(list)
    for row in validation:
        sparse_heads[row["mask"]].append(int(row["sparse_heads"]))
    static_rows = [
        {
            "mask": mask,
            "validation_cells": len(values),
            "min_sparse_heads": min(values),
            "max_sparse_heads": max(values),
            "all_validation_cells_have_sparse_head": min(values) > 0,
        }
        for mask, values in sorted(sparse_heads.items())
    ]
    test_rank16 = [
        row for row in basis if row["split"] == "test" and int(row["rank"]) == 16
    ]
    if not test_rank16:
        raise ValueError("basis stop/go CSV contains no test rank-16 rows")
    basis_rows = [
        {
            "mask": row["mask"],
            "rank": 16,
            "coefficient_oracle_error_max": float(row["coefficient_oracle_error_max"]),
            "ridge_error_max": float(row["ridge_error_max"]),
            "frozen_basis_energy_p05": float(row["frozen_basis_energy_p05"]),
            "subspace_overlap_p05": float(row["subspace_overlap_p05"]),
        }
        for row in sorted(test_rank16, key=lambda item: item["mask"])
    ]
    cell = independence["cells"][0]
    decision = {
        "nominal_replays": int(cell["nominal_replays"]),
        "unique_qkv_content_groups": int(cell["unique_qkv_content_groups"]),
        "nominal_test_samples": int(cell["nominal_test_samples"]),
        "unique_test_qkv_content_groups": int(cell["unique_test_qkv_content_groups"]),
        "cross_split_content_independent": bool(
            independence["cross_split_content_independent"]
        ),
        "static_geometry_validation_go": any(
            row["all_validation_cells_have_sparse_head"] for row in static_rows
        ),
        "rank16_coefficient_oracle_strict_go": any(
            row["coefficient_oracle_error_max"] <= 0.02 for row in basis_rows
        ),
        "rank16_coefficient_oracle_relaxed_go": any(
            row["coefficient_oracle_error_max"] <= 0.05 for row in basis_rows
        ),
        "rank16_ridge_relaxed_go": any(
            row["ridge_error_max"] <= 0.05 for row in basis_rows
        ),
        "best_rank16_coefficient_oracle": min(
            basis_rows, key=lambda row: row["coefficient_oracle_error_max"]
        ),
        "best_rank16_ridge": min(basis_rows, key=lambda row: row["ridge_error_max"]),
        "scope": (
            "Layer-0 step-0 QKV has only two independent seed-defined content groups; "
            "prompt and CFG-branch transfer are not observable in this capture."
        ),
    }
    decision["overall_fixed_geometry_frozen_basis_go"] = bool(
        decision["cross_split_content_independent"]
        and decision["static_geometry_validation_go"]
        and decision["rank16_coefficient_oracle_strict_go"]
        and decision["rank16_ridge_relaxed_go"]
    )
    payload = {"decision": decision, "static_geometry": static_rows, "basis_rank16": basis_rows}
    (args.output_dir / "geometry_pilot_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "geometry_pilot_decision.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mask",
                "validation_sparse_heads_min",
                "coefficient_oracle_error_max",
                "ridge_error_max",
                "frozen_basis_energy_p05",
                "subspace_overlap_p05",
            ],
        )
        writer.writeheader()
        by_mask = {row["mask"]: row for row in basis_rows}
        for row in static_rows:
            transfer = by_mask.get(row["mask"], {})
            writer.writerow(
                {
                    "mask": row["mask"],
                    "validation_sparse_heads_min": row["min_sparse_heads"],
                    "coefficient_oracle_error_max": transfer.get(
                        "coefficient_oracle_error_max", ""
                    ),
                    "ridge_error_max": transfer.get("ridge_error_max", ""),
                    "frozen_basis_energy_p05": transfer.get(
                        "frozen_basis_energy_p05", ""
                    ),
                    "subspace_overlap_p05": transfer.get("subspace_overlap_p05", ""),
                }
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.5))
    evidence_labels = ["Nominal\nreplays", "Unique QKV\ngroups", "Nominal test\nsamples", "Unique test\ngroups"]
    evidence_values = [
        decision["nominal_replays"],
        decision["unique_qkv_content_groups"],
        decision["nominal_test_samples"],
        decision["unique_test_qkv_content_groups"],
    ]
    axes[0].bar(evidence_labels, evidence_values, color=["#6d8ea0", "#b33a3a", "#6d8ea0", "#b33a3a"])
    axes[0].set_title("(a) Independent evidence audit")
    axes[0].set_ylabel("Count")

    masks = [row["mask"] for row in static_rows]
    counts = [row["min_sparse_heads"] for row in static_rows]
    axes[1].barh(masks, counts, color="#b33a3a")
    axes[1].axvline(1, color="#b7791f", linestyle=":", linewidth=1.2)
    axes[1].set_xlim(0, 12)
    axes[1].set_title("(b) Calibration-frozen sparse heads")
    axes[1].set_xlabel("Minimum over validation cells (of 12)")

    basis_masks = [row["mask"] for row in basis_rows]
    oracle = [100.0 * row["coefficient_oracle_error_max"] for row in basis_rows]
    ridge = [100.0 * row["ridge_error_max"] for row in basis_rows]
    positions = list(range(len(basis_masks)))
    width = 0.36
    axes[2].bar(
        [position - width / 2 for position in positions],
        oracle,
        width=width,
        label="held-out coefficient oracle",
        color="#176d75",
    )
    axes[2].bar(
        [position + width / 2 for position in positions],
        ridge,
        width=width,
        label="validation-selected ridge",
        color="#d47a2a",
    )
    axes[2].axhline(2.0, color="#8b1e1e", linestyle=":", linewidth=1.2)
    axes[2].axhline(5.0, color="#b7791f", linestyle=":", linewidth=1.2)
    axes[2].set_xticks(positions, basis_masks, rotation=15, ha="right")
    axes[2].set_ylabel("Test max relative L2 error (%)")
    axes[2].set_title("(c) Frozen rank-16 correction")
    axes[2].legend(fontsize=7.5)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("F81 layer-0/step-0 geometry pilot: STOP fixed sparse + frozen basis", y=1.02)
    figure.text(
        0.5,
        -0.03,
        "Prompt/CFG independence is untested here; the independent seed transfer already fails the 2% and 5% gates.",
        ha="center",
        fontsize=8.5,
        color="#4a4a4a",
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"geometry_pilot_decision.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
