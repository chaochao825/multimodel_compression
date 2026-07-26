#!/usr/bin/env python3
"""Freeze geometry sparse heads on one sample and evaluate another sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-sample-id", required=True)
    parser.add_argument("--validation-sample-id", required=True)
    parser.add_argument("--test-sample-id", action="append", default=[])
    parser.add_argument("--expected-heads", type=int, default=12)
    parser.add_argument("--error-target", type=float, default=0.02)
    parser.add_argument("--max-dense-heads", type=int, default=3)
    parser.add_argument("--max-execution-density", type=float, default=0.125)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cell_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["layer"],
        row["sampling_step"],
        row["timestep"],
        row["branch"],
        row["mask"],
        row["head"],
    )


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["layer"],
        row["sampling_step"],
        row["timestep"],
        row["branch"],
        row["mask"],
    )


def index_unique_rows(
    rows: list[dict[str, str]], sample_id: str
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    duplicates: list[tuple[str, ...]] = []
    for row in rows:
        key = cell_key(row)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if duplicates:
        raise RuntimeError(
            f"sample {sample_id} has duplicate policy keys: {duplicates[:3]}"
        )
    return indexed


def validate_head_groups(
    rows: list[dict[str, str]], sample_id: str, expected_heads: int
) -> None:
    grouped: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for row in rows:
        grouped[group_key(row)].add(int(row["head"]))
    invalid = [
        (key, sorted(heads))
        for key, heads in grouped.items()
        if len(heads) != expected_heads
    ]
    if invalid:
        raise RuntimeError(
            f"sample {sample_id} has incomplete head groups: {invalid[:3]}"
        )


def mask_summary(
    rows: list[dict[str, object]], split: str, sample_id: str
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for mask in sorted({str(row["mask"]) for row in rows}):
        selected = [row for row in rows if row["mask"] == mask]
        summaries.append(
            {
                "split": split,
                "sample_id": sample_id,
                "mask": mask,
                "cells": len(selected),
                "gate_pass_rate": sum(bool(row["gate_passed"]) for row in selected)
                / len(selected),
                "max_output_relative_l2": max(
                    float(row["output_relative_l2_energy_proxy"])
                    for row in selected
                ),
                "mean_effective_execution_density": sum(
                    float(row["effective_execution_density"]) for row in selected
                )
                / len(selected),
                "max_dense_heads": max(int(row["dense_heads"]) for row in selected),
                "all_cells_passed": all(bool(row["gate_passed"]) for row in selected),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.heads_csv)
    requested_samples = [
        args.calibration_sample_id,
        args.validation_sample_id,
        *args.test_sample_id,
    ]
    if len(set(requested_samples)) != len(requested_samples):
        raise ValueError("calibration, validation, and test sample IDs must be disjoint")
    rows_by_sample = {
        sample_id: [row for row in rows if row["sample_id"] == sample_id]
        for sample_id in requested_samples
    }
    missing_samples = [
        sample_id for sample_id, sample_rows in rows_by_sample.items() if not sample_rows
    ]
    if missing_samples:
        raise RuntimeError(f"missing requested samples: {missing_samples}")

    calibration = rows_by_sample[args.calibration_sample_id]
    validate_head_groups(calibration, args.calibration_sample_id, args.expected_heads)
    calibration_index = index_unique_rows(calibration, args.calibration_sample_id)
    policy = {
        key: row["static_sparse_head_gate"].lower() == "true"
        for key, row in calibration_index.items()
    }
    policy_rows: list[dict[str, object]] = []
    for key, row in sorted(calibration_index.items()):
        sparse = policy[key]
        policy_rows.append(
            {
                **{name: row[name] for name in (
                    "sample_id", "layer", "sampling_step", "timestep", "branch", "mask", "head"
                )},
                "action": "sparse" if sparse else "dense",
                "calibration_output_relative_l2": float(row["output_relative_l2"]),
                "calibration_query_nrmse_p95": float(row["query_output_nrmse_p95"]),
                "calibration_query_nrmse_p99": float(row["query_output_nrmse_p99"]),
                "calibration_query_cosine_p05": float(row["query_cosine_p05"]),
                "calibration_lse_error_p95": float(row["absolute_lse_error_p95"]),
            }
        )

    def evaluate_sample(split: str, sample_id: str) -> list[dict[str, object]]:
        sample_rows = rows_by_sample[sample_id]
        validate_head_groups(sample_rows, sample_id, args.expected_heads)
        sample_index = index_unique_rows(sample_rows, sample_id)
        calibration_keys = set(calibration_index)
        sample_keys = set(sample_index)
        if sample_keys != calibration_keys:
            missing = sorted(calibration_keys - sample_keys)
            extra = sorted(sample_keys - calibration_keys)
            raise RuntimeError(
                f"sample {sample_id} policy coverage mismatch: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in sample_rows:
            grouped[group_key(row)].append(row)

        evaluated: list[dict[str, object]] = []
        for key, head_rows in sorted(grouped.items()):
            head_rows = sorted(head_rows, key=lambda row: int(row["head"]))
            sparse_rows = [row for row in head_rows if policy[cell_key(row)]]
            reference_energy = sum(
                float(row["reference_squared_norm"]) for row in head_rows
            )
            defect_energy = sum(
                float(row["defect_squared_norm"]) for row in sparse_rows
            )
            sparse_heads = len(sparse_rows)
            dense_heads = len(head_rows) - sparse_heads
            sparse_density = float(head_rows[0]["execution_density"])
            effective_density = (
                dense_heads + sparse_heads * sparse_density
            ) / len(head_rows)
            output_error = math.sqrt(defect_energy / max(reference_energy, 1e-30))
            passed = (
                output_error <= args.error_target
                and dense_heads <= args.max_dense_heads
                and effective_density <= args.max_execution_density
            )
            evaluated.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "layer": key[0],
                    "sampling_step": key[1],
                    "timestep": key[2],
                    "branch": key[3],
                    "mask": key[4],
                    "heads": len(head_rows),
                    "sparse_heads": sparse_heads,
                    "dense_heads": dense_heads,
                    "sparse_execution_density": sparse_density,
                    "effective_execution_density": effective_density,
                    "output_relative_l2_energy_proxy": output_error,
                    "gate_passed": passed,
                }
            )
        return evaluated

    cell_rows = evaluate_sample("validation", args.validation_sample_id)
    for sample_id in args.test_sample_id:
        cell_rows.extend(evaluate_sample("test", sample_id))

    mask_rows: list[dict[str, object]] = []
    for split, sample_id in [
        ("validation", args.validation_sample_id),
        *(("test", sample_id) for sample_id in args.test_sample_id),
    ]:
        selected = [
            row
            for row in cell_rows
            if row["split"] == split and row["sample_id"] == sample_id
        ]
        mask_rows.extend(mask_summary(selected, split, sample_id))

    validation_go_masks = [
        str(row["mask"])
        for row in mask_rows
        if row["split"] == "validation" and bool(row["all_cells_passed"])
    ]
    test_selected_results = [
        row
        for row in mask_rows
        if row["split"] == "test" and row["mask"] in validation_go_masks
    ]

    write_rows(args.out_dir / "geometry_static_head_policy.csv", policy_rows)
    write_rows(args.out_dir / "geometry_generalization_cells.csv", cell_rows)
    write_rows(args.out_dir / "geometry_mask_split_summary.csv", mask_rows)
    summary = {
        "calibration_sample_id": args.calibration_sample_id,
        "validation_sample_id": args.validation_sample_id,
        "test_sample_ids": args.test_sample_id,
        "policy_heads": len(policy_rows),
        "evaluation_cells": len(cell_rows),
        "mask_split_results": mask_rows,
        "validation_go_masks": validation_go_masks,
        "test_results_for_validation_selected_masks": test_selected_results,
        "independent_test_gate_passed": bool(validation_go_masks)
        and bool(test_selected_results)
        and all(bool(row["all_cells_passed"]) for row in test_selected_results),
        "selection_source": "validation sample only; test samples never select masks",
        "scope": "pre-output-projection energy proxy with a calibration-frozen per-cell head policy",
        "warning": "this is not end-to-end quality or fused-kernel latency evidence",
    }
    (args.out_dir / "geometry_generalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
