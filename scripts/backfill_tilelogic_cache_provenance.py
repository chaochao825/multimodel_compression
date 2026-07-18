#!/usr/bin/env python3
"""Backfill immutable per-entry provenance in an existing TileLogic cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from tilespec_ex.cache import CACHE_FORMAT, file_sha256


TENSOR_FIELDS = ("thumbnail", "crops", "query", "crop_gradient")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tensor_dtypes(payload: dict[str, Any]) -> dict[str, str]:
    return {
        name: str(payload[name].dtype).removeprefix("torch.")
        for name in TENSOR_FIELDS
        if isinstance(payload.get(name), torch.Tensor)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    manifest_path = cache_dir / "cache_manifest.jsonl"
    environment_path = cache_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    source_manifest_sha256 = str(environment["manifest_sha256"])
    model_revision = str(
        environment.get("model_revision")
        or Path(str(environment["model_dir"])).resolve().name
    )
    if len(source_manifest_sha256) != 64 or not model_revision:
        raise RuntimeError("cache environment has incomplete source provenance")

    old_sha256 = _sha256(manifest_path)
    records = _read_jsonl(manifest_path)
    seen: set[tuple[str, int]] = set()
    for record in records:
        if record.get("format") != CACHE_FORMAT:
            raise RuntimeError("unsupported cache manifest format")
        key = (str(record["dataset"]), int(record["dataset_index"]))
        if key in seen:
            raise RuntimeError(f"duplicate cache key: {key}")
        seen.add(key)
        payload_path = cache_dir / str(record["cache_file"])
        if file_sha256(payload_path) != record["sha256"]:
            raise RuntimeError(f"cache payload hash mismatch: {payload_path}")
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        tensor_dtypes = _tensor_dtypes(payload)
        expected_fields = {"thumbnail", "crops", "query"}
        if bool(record["oracle"]):
            expected_fields.add("crop_gradient")
        if set(tensor_dtypes) != expected_fields:
            raise RuntimeError(f"cache tensor dtype fields differ for {key}")
        record["source_manifest_sha256"] = source_manifest_sha256
        record["model_revision"] = model_revision
        record["tensor_dtypes"] = tensor_dtypes

    payload_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    temporary = manifest_path.with_suffix(".jsonl.provenance.partial")
    temporary.write_text(payload_text, encoding="utf-8")
    temporary.replace(manifest_path)
    new_manifest_sha256 = _sha256(manifest_path)
    environment["model_revision"] = model_revision
    environment["per_entry_source_model_and_dtype_provenance"] = True
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = cache_dir / "cache_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "cache_manifest_sha256": new_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "model_revision": model_revision,
            "per_entry_tensor_dtypes": True,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    training_summary_sha256 = None
    if args.training_dir is not None:
        training_summary_path = args.training_dir.resolve() / "training_summary.json"
        training_summary = json.loads(
            training_summary_path.read_text(encoding="utf-8")
        )
        training_summary["cache_manifest_sha256"] = new_manifest_sha256
        training_summary["cache_provenance_backfill"] = {
            "format": "tilelogic_cache_provenance_backfill_v1",
            "old_manifest_sha256": old_sha256,
            "new_manifest_sha256": new_manifest_sha256,
            "payload_tensors_unchanged": True,
        }
        training_summary_path.write_text(
            json.dumps(training_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        training_summary_sha256 = _sha256(training_summary_path)
    output = {
        "format": "tilelogic_cache_provenance_backfill_v1",
        "records": len(records),
        "old_manifest_sha256": old_sha256,
        "new_manifest_sha256": new_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "model_revision": model_revision,
        "tensor_dtype_fields": list(TENSOR_FIELDS),
        "payload_tensors_unchanged": True,
        "training_summary_sha256": training_summary_sha256,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
