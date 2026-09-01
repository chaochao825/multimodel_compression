#!/usr/bin/env python3
"""Evaluate the frozen EXP-055 exact VAE CUDA Graph Gate."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
    if manifest["experiment_id"] != "EXP-055" or manifest["gate_id"] != "G-034":
        raise ValueError(f"identity mismatch in {path}")
    if manifest["stage"] != stage:
        raise ValueError(f"stage mismatch in {path}")
    return manifest


def comparison_passes(row: dict[str, Any]) -> bool:
    comparison = row["comparison"]
    return bool(row["repeat_equal"]) and all(
        bool(comparison[field])
        for field in (
            "bitwise_equal",
            "finite",
            "dtype_equal",
            "device_equal",
            "cpu_raw_equal",
            "cpu_decoded_equal",
        )
    )


def evaluate(result_root: Path) -> dict[str, Any]:
    f17 = read_manifest(result_root / "f17-screen" / "manifest.json", "f17-screen")
    component = read_manifest(
        result_root / "f81-component" / "manifest.json", "f81-component"
    )
    request = read_manifest(
        result_root / "f81-request" / "manifest.json", "f81-request"
    )

    source_commits = {
        f17["source_commit"], component["source_commit"], request["source_commit"]
    }
    if source_commits != {"ed3cb14dd936f92cdc9f9381af7369991509b41f"}:
        raise ValueError(f"source identity drift: {source_commits}")
    checkpoint_identities = {
        f17["result"]["checkpoint_identity"],
        component["result"]["checkpoint_identity"],
        request["result"]["checkpoint_identity"],
    }
    if checkpoint_identities != {
        "3baa20e8e64c7f1ee6e4a377f5a04b8e4d193e0a1a1241814879a004fd77370a"
    }:
        raise ValueError(f"checkpoint identity drift: {checkpoint_identities}")

    f17_rows = []
    expected_f17_order = (0, 1, 2, 1, 0)
    observed_f17_order = tuple(
        int(row["latent_index"]) for row in f17["result"]["replay_rows"]
    )
    if observed_f17_order != expected_f17_order:
        raise ValueError(f"F17 replay order drift: {observed_f17_order}")
    for row in f17["result"]["replay_rows"]:
        f17_rows.append(
            {
                "position": int(row["position"]),
                "latent_index": int(row["latent_index"]),
                "seed": int(row["seed"]),
                "seconds": float(row["seconds"]),
                "peak_reserved_mib": float(row["peak_reserved_mib"]),
                "exact": comparison_passes(row),
            }
        )

    component_rows = []
    observed_component_order = tuple(
        int(row["prompt_index"]) for row in component["result"]["rows"]
    )
    if observed_component_order != (0, 1, 2, 3, 3, 2, 1, 0):
        raise ValueError(f"F81 component replay order drift: {observed_component_order}")
    for row in component["result"]["rows"]:
        component_rows.append(
            {
                "round": int(row["round"]),
                "position": int(row["position"]),
                "prompt_index": int(row["prompt_index"]),
                "eager_seconds": float(row["eager_seconds"]),
                "graph_seconds": float(row["graph_seconds"]),
                "speedup": float(row["speedup"]),
                "eager_peak_reserved_mib": float(row["eager_peak_reserved_mib"]),
                "graph_peak_reserved_mib": float(row["graph_peak_reserved_mib"]),
                "exact": comparison_passes(row),
            }
        )

    request_rows = []
    observed_request_order = tuple(
        int(row["prompt_index"]) for row in request["result"]["rows"]
    )
    if observed_request_order != (0, 1, 2, 3):
        raise ValueError(f"F81 request order drift: {observed_request_order}")
    for row in request["result"]["rows"]:
        eager = row["eager"]
        graph = row["graph"]
        request_rows.append(
            {
                "prompt_index": int(row["prompt_index"]),
                "seed": int(row["seed"]),
                "cpu_video_equal": bool(row["cpu_video_equal"]),
                "network_calls_equal": bool(row["network_calls_equal"]),
                "eager_text_seconds": float(eager["text_seconds"]),
                "eager_denoiser_seconds": float(eager["denoiser_seconds"]),
                "eager_vae_seconds": float(eager["vae_seconds"]),
                "eager_transfer_seconds": float(eager["cpu_transfer_seconds"]),
                "eager_serialization_seconds": float(eager["serialization_seconds"]),
                "eager_request_seconds": float(eager["request_seconds"]),
                "graph_text_seconds": float(graph["text_seconds"]),
                "graph_denoiser_seconds": float(graph["denoiser_seconds"]),
                "graph_vae_seconds": float(graph["vae_seconds"]),
                "graph_transfer_seconds": float(graph["cpu_transfer_seconds"]),
                "graph_serialization_seconds": float(graph["serialization_seconds"]),
                "graph_request_seconds": float(graph["request_seconds"]),
                "graph_peak_reserved_mib": float(graph["peak_reserved_mib"]),
                "paired_speedup": float(eager["request_seconds"])
                / float(graph["request_seconds"]),
            }
        )

    component_summary = component["result"]["summary"]
    request_summary = request["result"]["summary"]
    f17_exact = all(row["exact"] for row in f17_rows) and bool(
        f17["result"]["stale_state_free"]
    )
    component_exact = all(row["exact"] for row in component_rows)
    request_exact = all(
        row["cpu_video_equal"] and row["network_calls_equal"]
        for row in request_rows
    )
    memory_ok = float(request_summary["peak_reserved_mib"]) <= 59948.0
    component_speed_ok = float(component_summary["vae_speedup"]) >= 1.12
    projected_speed_ok = (
        float(component_summary["projected_request_speedup"]) >= 1.05
    )
    measured_speed_ok = (
        float(request_summary["median_graph_request_seconds"]) <= 9.1790428875
    )

    if not f17_exact or not component_exact or not request_exact:
        outcome = "exactness-null"
    elif not memory_ok:
        outcome = "invalid-memory"
    elif not component_speed_ok or not projected_speed_ok:
        outcome = "performance-null"
    elif not measured_speed_ok:
        outcome = "speed-boundary"
    else:
        outcome = "pass"

    return {
        "experiment_id": "EXP-055",
        "gate_id": "G-034",
        "outcome": outcome,
        "f17_replays": f17_rows,
        "component_rows": component_rows,
        "request_rows": request_rows,
        "primary": {
            "f17_bitwise_exact": f17_exact,
            "f81_component_bitwise_exact": component_exact,
            "f81_request_bitwise_exact": request_exact,
            "vae_speedup": float(component_summary["vae_speedup"]),
            "projected_request_speedup": float(
                component_summary["projected_request_speedup"]
            ),
            "measured_request_seconds": float(
                request_summary["median_graph_request_seconds"]
            ),
            "measured_request_speedup_over_incumbent": float(
                request_summary["request_speedup_over_incumbent"]
            ),
            "paired_request_speedup": float(
                statistics.median(row["paired_speedup"] for row in request_rows)
            ),
            "peak_reserved_mib": float(request_summary["peak_reserved_mib"]),
        },
        "guards": {
            "f17_exact_and_stale_free": f17_exact,
            "f81_component_exact": component_exact,
            "f81_request_exact": request_exact,
            "network_calls_equal": all(
                row["network_calls_equal"] for row in request_rows
            ),
            "memory_ok": memory_ok,
            "component_speed_ok": component_speed_ok,
            "projected_speed_ok": projected_speed_ok,
            "measured_speed_ok": measured_speed_ok,
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
    output_dir = args.output_dir.resolve()
    require_fresh_output_dir(output_dir)
    result = evaluate(args.result_root.resolve())
    write_csv(output_dir / "f17_replays.csv", result["f17_replays"])
    write_csv(output_dir / "f81_component_rows.csv", result["component_rows"])
    write_csv(output_dir / "f81_request_rows.csv", result["request_rows"])
    atomic_write_json(output_dir / "gate_summary.json", result)
    atomic_write_json(output_dir / "SUCCESS.json", {"status": "complete"})
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
