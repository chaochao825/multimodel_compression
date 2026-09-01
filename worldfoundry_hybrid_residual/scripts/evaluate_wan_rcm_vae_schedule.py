#!/usr/bin/env python3
"""Evaluate the frozen EXP-053 exact VAE scheduling Gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiment_artifacts import atomic_write_json, require_fresh_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_manifest(path: Path, stage: str) -> dict[str, Any]:
    success_path = path.parent / "SUCCESS.json"
    if not success_path.is_file():
        raise FileNotFoundError(f"missing stage success marker: {success_path}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if success["status"] != "complete" or success["stage"] != stage:
        raise ValueError(f"invalid success marker for {stage}: {success}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["experiment_id"] != "EXP-053" or manifest["gate_id"] != "G-032":
        raise ValueError(f"identity mismatch in {path}")
    if manifest["stage"] != stage:
        raise ValueError(f"stage mismatch in {path}")
    return manifest


def evaluate(result_root: Path) -> dict[str, Any]:
    f17 = read_manifest(result_root / "f17-screen" / "manifest.json", "f17-screen")
    f81 = read_manifest(
        result_root / "f81-component" / "manifest.json", "f81-component"
    )

    selected_chunk = int(f17["result"]["selected_chunk_size"])
    if selected_chunk != int(f81["result"]["selected_chunk_size"]):
        raise ValueError("F81 did not use the frozen F17-selected chunk")
    prompt_rows = f81["result"]["rows"]
    prompt_indices = tuple(int(row["prompt_index"]) for row in prompt_rows)
    if prompt_indices != (0, 1, 2, 3):
        raise ValueError(f"unexpected F81 prompt coverage: {prompt_indices}")
    if any(int(row["latent"]["network_forward_calls"]) != 4 for row in prompt_rows):
        raise ValueError("rCM network call count changed")

    f17_rows = []
    for row in f17["result"]["benchmark"]["candidates"]:
        f17_rows.append(
            {
                "chunk_size": int(row["chunk_size"]),
                "bitwise_equal": bool(row["exactness"]["bitwise_equal"]),
                "max_abs": float(row["exactness"]["max_abs"]),
                "relative_l2": float(row["exactness"]["relative_l2"]),
                "median_seconds": float(row["median_seconds"]),
                "speedup": float(row["speedup"]),
                "selected": int(row["chunk_size"]) == selected_chunk,
            }
        )

    f81_rows = []
    for row in prompt_rows:
        candidate = row["benchmark"]["candidates"][0]
        f81_rows.append(
            {
                "prompt_index": int(row["prompt_index"]),
                "chunk_size": int(candidate["chunk_size"]),
                "bitwise_equal": bool(candidate["exactness"]["bitwise_equal"]),
                "max_abs": float(candidate["exactness"]["max_abs"]),
                "relative_l2": float(candidate["exactness"]["relative_l2"]),
                "baseline_median_seconds": float(
                    row["benchmark"]["baseline_median_seconds"]
                ),
                "candidate_median_seconds": float(candidate["median_seconds"]),
                "speedup": float(candidate["speedup"]),
            }
        )

    summary = f81["result"]["summary"]
    exact = bool(summary["bitwise_exact"])
    component_speed_pass = float(summary["vae_speedup"]) >= float(
        summary["min_vae_speedup"]
    )
    projected_speed_pass = float(
        summary["projected_incremental_warm_speedup"]
    ) >= float(summary["min_projected_warm_speedup"])
    if not exact:
        outcome = "exactness-null"
    elif not component_speed_pass or not projected_speed_pass:
        outcome = "speed-boundary"
    else:
        outcome = "component-pass-awaiting-endpoint"

    return {
        "experiment_id": "EXP-053",
        "gate_id": "G-032",
        "outcome": outcome,
        "selected_chunk_size": selected_chunk,
        "f17_candidates": f17_rows,
        "f81_prompts": f81_rows,
        "primary": {
            "bitwise_exact": exact,
            "vae_speedup": float(summary["vae_speedup"]),
            "projected_warm_seconds": float(summary["projected_warm_seconds"]),
            "projected_incremental_warm_speedup": float(
                summary["projected_incremental_warm_speedup"]
            ),
        },
        "guards": {
            "f17_selected_candidate_exact": any(
                row["selected"] and row["bitwise_equal"] for row in f17_rows
            ),
            "f81_all_exact": exact,
            "component_speed_pass": component_speed_pass,
            "projected_speed_pass": projected_speed_pass,
            "network_calls_equal": True,
            "endpoint_not_authorized": outcome != "component-pass-awaiting-endpoint",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    output_dir = args.output_dir.resolve()
    require_fresh_output_dir(output_dir)
    result = evaluate(result_root)
    write_csv(output_dir / "f17_candidates.csv", result["f17_candidates"])
    write_csv(output_dir / "f81_prompt_rows.csv", result["f81_prompts"])
    atomic_write_json(output_dir / "gate_summary.json", result)
    atomic_write_json(output_dir / "SUCCESS.json", {"status": "complete"})
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
