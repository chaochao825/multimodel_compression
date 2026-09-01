#!/usr/bin/env python3
"""Evaluate the frozen EXP-052 exact resident-runtime Gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiment_artifacts import atomic_write_json, require_fresh_output_dir


METHODS = ("teacher20", "native4", "rcm4")
EXPECTED_FORWARD_CALLS = {"teacher20": 40, "native4": 8, "rcm4": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speedup-threshold", type=float, default=2.5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_success(path: Path) -> None:
    payload = load_json(path)
    if payload["status"] != "complete":
        raise RuntimeError(f"incomplete EXP-052 artifact: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(input_root: Path, speedup_threshold: float) -> dict[str, Any]:
    text_root = input_root / "text-screen" / "attempt02"
    require_success(text_root / "SUCCESS.json")
    text = load_json(text_root / "manifest.json")["result"]

    f17: dict[str, dict[str, Any]] = {}
    f81: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        f17_root = input_root / "f17-exact" / method
        f81_root = input_root / "f81-resident" / method
        require_success(f17_root / "SUCCESS.json")
        require_success(f81_root / "SUCCESS.json")
        f17[method] = load_json(f17_root / "manifest.json")["result"]
        f81[method] = load_json(f81_root / "manifest.json")["result"]

    text_exact = bool(text["exact"])
    text_screen_pass = bool(text["advance"])
    f17_exact = all(bool(f17[method]["video_equal"]) for method in METHODS)
    f17_calls = all(bool(f17[method]["network_calls_equal"]) for method in METHODS)
    positive_cache_guard = all(
        int(f81[method]["text_policy"]["positive_cache_hits"]) == 0
        and int(f81[method]["text_policy"]["positive_model_calls"]) == 4
        for method in METHODS
    )
    request_completeness = all(
        len(f81[method]["rows"]) == 4
        and all(row["status"] == "ok" for row in f81[method]["rows"])
        for method in METHODS
    )
    forward_call_guard = all(
        all(
            int(row["network_forward_calls"]) == EXPECTED_FORWARD_CALLS[method]
            for row in f81[method]["rows"]
        )
        for method in METHODS
    )
    memory_guard = all(
        float(row["peak_reserved_mib"]) * (1024.0**2)
        < float(load_json(input_root / "f81-resident" / method / "manifest.json")["environment"]["gpu_total_memory_bytes"])
        for method in METHODS
        for row in f81[method]["rows"]
    )

    method_rows = []
    request_rows = []
    for method in METHODS:
        summary = f81[method]["summary"]
        method_rows.append(
            {
                "method": method,
                "median_request_seconds": float(summary["median_request_seconds"]),
                "median_text_seconds": float(summary["median_text_seconds"]),
                "median_denoiser_seconds": float(summary["median_denoiser_seconds"]),
                "median_vae_seconds": float(summary["median_vae_seconds"]),
                "median_cpu_transfer_seconds": float(
                    summary["median_cpu_transfer_seconds"]
                ),
                "median_serialization_seconds": float(
                    summary["median_serialization_seconds"]
                ),
                "median_peak_reserved_mib": float(
                    summary["median_peak_reserved_mib"]
                ),
            }
        )
        for row in f81[method]["rows"]:
            request_rows.append(
                {
                    "method": method,
                    "prompt_index": int(row["prompt_index"]),
                    "request_seconds": float(row["request_seconds"]),
                    "text_seconds": float(row["text_seconds"]),
                    "denoiser_seconds": float(row["denoiser_seconds"]),
                    "vae_seconds": float(row["vae_seconds"]),
                    "cpu_transfer_seconds": float(row["cpu_transfer_seconds"]),
                    "serialization_seconds": float(row["serialization_seconds"]),
                    "network_forward_calls": int(row["network_forward_calls"]),
                }
            )

    method_map = {row["method"]: row for row in method_rows}
    warm_speedup = (
        float(method_map["teacher20"]["median_request_seconds"])
        / float(method_map["rcm4"]["median_request_seconds"])
    )
    denoiser_speedup = (
        float(method_map["teacher20"]["median_denoiser_seconds"])
        / float(method_map["rcm4"]["median_denoiser_seconds"])
    )
    exact_guards = (
        text_exact
        and text_screen_pass
        and f17_exact
        and f17_calls
        and positive_cache_guard
        and request_completeness
        and forward_call_guard
        and memory_guard
    )
    if exact_guards and warm_speedup >= speedup_threshold:
        outcome = "pass"
    elif exact_guards:
        outcome = "speed-boundary"
    else:
        outcome = "invalid"

    return {
        "experiment_id": "EXP-052",
        "gate_id": "G-031",
        "outcome": outcome,
        "text_screen": {
            "exact": text_exact,
            "advance": text_screen_pass,
            "minimum_median_saving_seconds": float(
                text["minimum_median_saving_seconds"]
            ),
            "required_saving_seconds": float(text["required_saving_seconds"]),
        },
        "guards": {
            "f17_video_exact": f17_exact,
            "f17_network_calls_equal": f17_calls,
            "positive_cache_guard": positive_cache_guard,
            "request_completeness": request_completeness,
            "forward_call_guard": forward_call_guard,
            "memory_guard": memory_guard,
            "all_exact_guards": exact_guards,
        },
        "primary": {
            "resident_warm_speedup": warm_speedup,
            "required_speedup": speedup_threshold,
            "denoiser_speedup": denoiser_speedup,
        },
        "method_rows": method_rows,
        "request_rows": request_rows,
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    require_fresh_output_dir(output_dir)
    summary = evaluate(input_root, args.speedup_threshold)
    atomic_write_json(output_dir / "gate_summary.json", summary)
    write_csv(output_dir / "method_summary.csv", summary["method_rows"])
    write_csv(output_dir / "request_rows.csv", summary["request_rows"])
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {"status": "complete", "outcome": summary["outcome"]},
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
