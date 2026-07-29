#!/usr/bin/env python3
"""Merge disjoint support-manifold sample shards with coverage validation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch

from experiment_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    object_sha256,
    require_fresh_output_dir,
)
from probe_support_manifold_oracle import aggregate_records, build_decision, load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_shard(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    success = json.loads((path / "SUCCESS.json").read_text(encoding="utf-8"))
    for name, expected in success["artifact_sha256"].items():
        actual = file_sha256(path / name)
        if actual != expected:
            raise ValueError(f"shard artifact hash mismatch: {path}/{name}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return success, manifest


def record_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        row["sample_id"],
        row["cell"],
        int(row["head"]),
        int(row["tile_index"]),
        row["family"],
        float(row["density_target"]),
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    protocol_path = args.protocol_config.resolve()
    protocol = load_protocol(protocol_path)
    require_fresh_output_dir(args.output_dir)
    records: list[dict[str, str]] = []
    shard_sources = []
    observed_samples: set[str] = set()
    seen: set[tuple[object, ...]] = set()
    capture_fingerprints: dict[str, dict[str, object]] = {}
    for raw_path in args.input_dir:
        path = raw_path.resolve()
        success, manifest = verify_shard(path)
        if manifest["protocol_object_sha256"] != object_sha256(protocol):
            raise ValueError(f"shard protocol mismatch: {path}")
        shard_rows = read_csv(path / "support_records.csv")
        for row in shard_rows:
            key = record_key(row)
            if key in seen:
                raise ValueError(f"duplicate merged record: {key}")
            seen.add(key)
            records.append(row)
            observed_samples.add(row["sample_id"])
        for fingerprint in manifest["capture_fingerprints"]:
            capture_path = str(fingerprint["path"])
            previous = capture_fingerprints.get(capture_path)
            if previous is not None and previous != fingerprint:
                raise ValueError(f"capture fingerprint disagreement: {capture_path}")
            capture_fingerprints[capture_path] = fingerprint
        shard_sources.append(
            {
                "path": str(path),
                "success_sha256": file_sha256(path / "SUCCESS.json"),
                "run_id": success["run_id"],
                "sample_shard": manifest["semantics"]["sample_shard"],
                "records": len(shard_rows),
            }
        )
    expected_samples = set(map(str, protocol["scope"]["sample_ids"]))
    if observed_samples != expected_samples:
        raise ValueError(
            f"merged sample coverage mismatch: missing={sorted(expected_samples-observed_samples)} "
            f"unexpected={sorted(observed_samples-expected_samples)}"
        )
    summary = aggregate_records(records, protocol)
    decision = build_decision(summary, protocol)
    atomic_write_csv(args.output_dir / "support_records.csv", records)
    atomic_write_csv(args.output_dir / "support_summary.csv", summary)
    atomic_write_json(args.output_dir / "decision.json", decision)
    manifest = {
        "schema_version": 1,
        "kind": "merged_disjoint_sample_shards",
        "elapsed_seconds": time.time() - started,
        "protocol": protocol,
        "protocol_sha256": file_sha256(protocol_path),
        "protocol_object_sha256": object_sha256(protocol),
        "shard_sources": shard_sources,
        "capture_fingerprints": list(capture_fingerprints.values()),
        "records": len(records),
        "sample_ids": sorted(observed_samples),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "semantics": {
            "sample_shards": "disjoint by registered sample_id; all layer/step cells occur in every shard",
            "merge": "raw records concatenated after exact-key duplicate and sample-coverage checks; summaries and decisions recomputed from the master protocol",
            "claim_boundary": protocol["claim_boundary"],
        },
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    artifacts = {
        name: file_sha256(args.output_dir / name)
        for name in ("support_records.csv", "support_summary.csv", "decision.json", "manifest.json")
    }
    atomic_write_json(
        args.output_dir / "SUCCESS.json",
        {"verdict": decision["verdict"], "artifact_sha256": artifacts},
    )
    print(f"[support-merge] samples={len(observed_samples)} records={len(records)}")
    print(f"[support-merge] verdict={decision['verdict']} output={args.output_dir}")


if __name__ == "__main__":
    main()
