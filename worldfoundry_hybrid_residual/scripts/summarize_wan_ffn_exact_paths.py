#!/usr/bin/env python3
"""Summarize Wan FFN exact-path timing, correctness, and amortization gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_MIN_MEDIAN_SPEEDUP = 1.10
DEFAULT_MIN_P95_SPEEDUP = 1.00
DEFAULT_MIN_AMORTIZED_SPEEDUP = 1.00
DEFAULT_MAX_INCREMENTAL_MEMORY_GIB = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-median-speedup", type=float, default=DEFAULT_MIN_MEDIAN_SPEEDUP
    )
    parser.add_argument(
        "--min-p95-speedup", type=float, default=DEFAULT_MIN_P95_SPEEDUP
    )
    parser.add_argument(
        "--min-amortized-speedup", type=float, default=DEFAULT_MIN_AMORTIZED_SPEEDUP
    )
    parser.add_argument(
        "--max-incremental-memory-gib",
        type=float,
        default=DEFAULT_MAX_INCREMENTAL_MEMORY_GIB,
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    try:
        return float(raw) if raw != "" else default
    except (TypeError, ValueError):
        return default


def as_bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {"1", "true", "yes"}


def harmonic_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return math.nan
    return len(values) / sum(1.0 / value for value in values)


def evaluate_rows(
    rows: list[dict[str, str]],
    *,
    min_median_speedup: float,
    min_p95_speedup: float,
    min_amortized_speedup: float,
    max_incremental_memory_gib: float,
) -> list[dict[str, object]]:
    eager: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        if row.get("path") == "eager" and row.get("status") == "ok":
            eager[(row["case"], int(row["layer"]))] = row

    evaluated: list[dict[str, object]] = []
    for row in rows:
        key = (row.get("case", ""), int(row.get("layer", -1)))
        baseline = eager.get(key)
        output: dict[str, object] = dict(row)
        if row.get("status") != "ok" or baseline is None:
            output.update(
                {
                    "median_speedup": math.nan,
                    "p95_speedup": math.nan,
                    "amortized_speedup": math.nan,
                    "correctness_gate": False,
                    "performance_gate": False,
                    "amortization_gate": False,
                    "memory_gate": False,
                    "decision": "NO-GO",
                    "decision_reason": "execution_error_or_missing_eager_baseline",
                }
            )
            evaluated.append(output)
            continue

        median = as_float(row, "latency_ms_median")
        p95 = as_float(row, "latency_ms_p95")
        amortized = as_float(row, "amortized_latency_ms")
        eager_median = as_float(baseline, "latency_ms_median")
        eager_p95 = as_float(baseline, "latency_ms_p95")
        median_speedup = eager_median / median
        p95_speedup = eager_p95 / p95
        amortized_speedup = eager_median / amortized
        setup_ms = as_float(row, "setup_ms", 0.0)
        steady_saving_ms = eager_median - median
        break_even_calls = (
            setup_ms / steady_saving_ms
            if setup_ms > 0.0 and steady_saving_ms > 0.0
            else 0.0
            if setup_ms == 0.0
            else math.inf
        )
        memory_gib = as_float(row, "incremental_peak_allocated_bytes", 0.0) / 2**30
        is_baseline = row.get("path") == "eager"
        correctness_gate = as_bool(row, "bitwise_equal")
        performance_gate = median_speedup >= min_median_speedup and p95_speedup >= min_p95_speedup
        amortization_gate = amortized_speedup >= min_amortized_speedup
        memory_gate = memory_gib <= max_incremental_memory_gib
        decision = (
            "REFERENCE"
            if is_baseline
            else "GO"
            if correctness_gate and performance_gate and amortization_gate and memory_gate
            else "NO-GO"
        )
        failed = []
        if not correctness_gate:
            failed.append("not_bitwise_exact")
        if not performance_gate:
            failed.append("steady_state_speed")
        if not amortization_gate:
            failed.append("compile_or_capture_amortization")
        if not memory_gate:
            failed.append("incremental_memory")
        output.update(
            {
                "median_speedup": median_speedup,
                "p95_speedup": p95_speedup,
                "amortized_speedup": amortized_speedup,
                "break_even_calls_vs_eager": break_even_calls,
                "incremental_peak_allocated_gib": memory_gib,
                "correctness_gate": correctness_gate,
                "performance_gate": performance_gate,
                "amortization_gate": amortization_gate,
                "memory_gate": memory_gate,
                "decision": decision,
                "decision_reason": "reference" if is_baseline else ";".join(failed) or "all_gates_passed",
            }
        )
        evaluated.append(output)
    return evaluated


def aggregate_paths(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("path") != "eager":
            groups[str(row.get("path"))].append(row)
    aggregates: list[dict[str, object]] = []
    for path, group in sorted(groups.items()):
        valid = [row for row in group if row.get("status") == "ok"]
        decisions = [str(row.get("decision")) for row in group]
        speedups = [float(row["median_speedup"]) for row in valid]
        p95_speedups = [float(row["p95_speedup"]) for row in valid]
        amortized = [float(row["amortized_speedup"]) for row in valid]
        aggregates.append(
            {
                "path": path,
                "cells_expected": len(group),
                "cells_ok": len(valid),
                "cells_go": decisions.count("GO"),
                "all_cells_go": bool(group) and all(decision == "GO" for decision in decisions),
                "median_speedup_hmean": harmonic_mean(speedups),
                "median_speedup_min": min(speedups, default=math.nan),
                "p95_speedup_min": min(p95_speedups, default=math.nan),
                "amortized_speedup_min": min(amortized, default=math.nan),
                "bitwise_exact_all": bool(valid)
                and len(valid) == len(group)
                and all(bool(row.get("correctness_gate")) for row in valid),
                "decision": "GO" if group and all(decision == "GO" for decision in decisions) else "NO-GO",
            }
        )
    return aggregates


def main() -> None:
    args = parse_args()
    if min(
        args.min_median_speedup,
        args.min_p95_speedup,
        args.min_amortized_speedup,
        args.max_incremental_memory_gib,
    ) <= 0.0:
        raise ValueError("all thresholds must be positive")
    rows = read_csv(args.input)
    if not rows:
        raise ValueError("benchmark CSV is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluated = evaluate_rows(
        rows,
        min_median_speedup=args.min_median_speedup,
        min_p95_speedup=args.min_p95_speedup,
        min_amortized_speedup=args.min_amortized_speedup,
        max_incremental_memory_gib=args.max_incremental_memory_gib,
    )
    aggregates = aggregate_paths(evaluated)
    write_csv(args.output_dir / "wan_ffn_exact_summary.csv", evaluated)
    write_csv(args.output_dir / "wan_ffn_exact_decisions.csv", aggregates)
    payload = {
        "scope": "complete Wan checkpoint FFN exact-path gate",
        "thresholds": {
            "bitwise_equal_required": True,
            "min_median_speedup": args.min_median_speedup,
            "min_p95_speedup": args.min_p95_speedup,
            "min_amortized_speedup": args.min_amortized_speedup,
            "max_incremental_memory_gib": args.max_incremental_memory_gib,
        },
        "paths": aggregates,
        "interpretation": (
            "GO requires every F17/F81 representative-layer cell to be bitwise "
            "equal to eager and to pass steady-state, p95, amortization, and memory "
            "gates. A compile result does not by itself prove GEMM-epilogue fusion."
        ),
    }
    (args.output_dir / "wan_ffn_exact_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
