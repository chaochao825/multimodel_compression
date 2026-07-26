#!/usr/bin/env python3
"""Calibrate a moment-mass fallback gate without reading test errors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--error-target", type=float, default=0.02)
    parser.add_argument("--aggregate-target", type=float, default=0.01)
    parser.add_argument("--minimum-validation-coverage", type=float, default=0.05)
    parser.add_argument("--attention-speed-target", type=float, default=1.5)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configuration(row: dict[str, str]) -> tuple[str, str, str]:
    return row["density"], row["tail_group_size"], row["method"]


def summarize_acceptance(
    rows: list[dict[str, str]], threshold: float
) -> dict[str, object]:
    accepted = [
        row
        for row in rows
        if float(row["router_selected_mass_proxy"]) >= threshold
    ]
    coverage = len(accepted) / max(len(rows), 1)
    if accepted:
        residual_sq = sum(float(row["residual_sq"]) for row in accepted)
        reference_sq = sum(float(row["reference_sq"]) for row in accepted)
        aggregate = math.sqrt(residual_sq / max(reference_sq, 1e-30))
        maximum = max(float(row["output_relative_l2"]) for row in accepted)
        p95 = sorted(float(row["output_relative_l2"]) for row in accepted)[
            round(0.95 * (len(accepted) - 1))
        ]
        approximate_work = sum(
            float(row["attention_work_ratio"]) for row in accepted
        ) / len(accepted)
    else:
        aggregate = maximum = p95 = 0.0
        approximate_work = 1.0
    effective_work = (1.0 - coverage) + coverage * approximate_work
    return {
        "threshold": threshold,
        "records": len(rows),
        "accepted_records": len(accepted),
        "coverage": coverage,
        "aggregate_output_relative_l2": aggregate,
        "record_error_p95": p95,
        "record_error_max": maximum,
        "approximate_attention_work_ratio": approximate_work,
        "fallback_adjusted_attention_work_ratio": effective_work,
        "fallback_adjusted_arithmetic_speedup": 1.0 / effective_work,
        "accepted_head_roles": dict(Counter(row["head_role"] for row in accepted)),
    }


def threshold_candidates(rows: list[dict[str, str]]) -> list[float]:
    values = sorted({float(row["router_selected_mass_proxy"]) for row in rows})
    return [0.0, *values, 1.0 + 1e-9]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in read_rows(args.heads_csv)
        if row["router"] == "moment" and row["method"] == "centroid"
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[configuration(row)].append(row)

    calibrated = []
    for config, group in grouped.items():
        validation = [row for row in group if row["split"] == "validation"]
        test = [row for row in group if row["split"] == "test"]
        feasible = []
        for threshold in threshold_candidates(validation):
            summary = summarize_acceptance(validation, threshold)
            if (
                float(summary["coverage"]) >= args.minimum_validation_coverage
                and float(summary["record_error_max"]) <= args.error_target
                and float(summary["aggregate_output_relative_l2"])
                <= args.aggregate_target
            ):
                feasible.append(summary)
        if not feasible:
            continue
        selected = max(
            feasible,
            key=lambda summary: (
                float(summary["fallback_adjusted_arithmetic_speedup"]),
                float(summary["coverage"]),
            ),
        )
        test_summary = summarize_acceptance(test, float(selected["threshold"]))
        calibrated.append(
            {
                "density": config[0],
                "tail_group_size": config[1],
                "method": config[2],
                "validation": selected,
                "test": test_summary,
            }
        )

    if calibrated:
        chosen = max(
            calibrated,
            key=lambda record: float(
                record["validation"]["fallback_adjusted_arithmetic_speedup"]
            ),
        )
        test = chosen["test"]
        quality_go = (
            float(test["record_error_max"]) <= args.error_target
            and float(test["aggregate_output_relative_l2"])
            <= args.aggregate_target
        )
        speed_go = (
            float(test["fallback_adjusted_arithmetic_speedup"])
            >= args.attention_speed_target
        )
        if quality_go and speed_go:
            verdict = "GO_NUMERICAL_AND_ARITHMETIC_GATES"
        elif quality_go:
            verdict = "QUALITY_GO_SYSTEM_SPEED_NO_GO"
        else:
            verdict = "NO_GO_VALIDATION_GATE_DID_NOT_TRANSFER"
    else:
        chosen = None
        verdict = "NO_GO_NO_VALIDATION_FEASIBLE_GATE"

    decision = {
        "verdict": verdict,
        "chosen_without_test_selection": chosen,
        "all_validation_feasible_configurations": calibrated,
        "gates": {
            "aggregate_output_relative_l2": args.aggregate_target,
            "every_record_output_relative_l2": args.error_target,
            "minimum_validation_coverage": args.minimum_validation_coverage,
            "fallback_adjusted_arithmetic_attention_speedup": args.attention_speed_target,
        },
        "methodology": (
            "configuration and threshold maximize validation arithmetic speed under quality "
            "constraints; test errors are read only after the choice is frozen"
        ),
    }
    (args.output_dir / "confidence_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for record in calibrated:
        for split in ("validation", "test"):
            flat_rows.append(
                {
                    "density": record["density"],
                    "tail_group_size": record["tail_group_size"],
                    "method": record["method"],
                    "split": split,
                    **record[split],
                    "accepted_head_roles": json.dumps(
                        record[split]["accepted_head_roles"], sort_keys=True
                    ),
                }
            )
    if flat_rows:
        fields = list(flat_rows[0])
        with (args.output_dir / "confidence_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(flat_rows)
    print(f"[confidence] verdict={verdict}")


if __name__ == "__main__":
    main()
