#!/usr/bin/env python3
"""Prove that a rate-accounting correction did not change codec semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_CHANGED_COMPONENTS = {
    "base_scalar_scales",
    "base_codebook",
    "residual_codebooks",
    "mlp_router_parameters",
    "mlp_router_normalizer",
    "logic_router",
    "router_curvature_prior",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["dataset"]), int(record["dataset_index"])


def _variant_key(variant: dict[str, Any]) -> tuple[str, float | None]:
    rate = variant.get("retention_rate")
    return str(variant["method"]), None if rate is None else float(rate)


def _non_rate_variant(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in variant.items()
        if name not in {"rate", "rate_components"}
    }


def _component_index(variant: dict[str, Any]) -> dict[tuple[str, str], int]:
    return {
        (str(item["name"]), str(item["scope"])): int(item["bits"])
        for item in variant["rate_components"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-feature-jsonl", type=Path, required=True)
    parser.add_argument("--new-feature-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_path = args.old_feature_jsonl.resolve()
    new_path = args.new_feature_jsonl.resolve()
    old_records = {_key(record): record for record in _read_jsonl(old_path)}
    new_records = {_key(record): record for record in _read_jsonl(new_path)}
    errors: list[str] = []
    changed_component_names: set[str] = set()
    changed_variants = 0
    compared_variants = 0
    if len(old_records) != 360 or set(old_records) != set(new_records):
        errors.append("old/new feature sample key sets differ from 360 paired records")
    for key in sorted(set(old_records) & set(new_records)):
        old_record = old_records[key]
        new_record = new_records[key]
        for field in old_record:
            if field in {"variants", "elapsed_seconds"}:
                continue
            if old_record[field] != new_record.get(field):
                errors.append(f"{key}: record field changed: {field}")
        old_variants = {
            _variant_key(variant): variant for variant in old_record["variants"]
        }
        new_variants = {
            _variant_key(variant): variant for variant in new_record["variants"]
        }
        if set(old_variants) != set(new_variants):
            errors.append(f"{key}: variant key set changed")
            continue
        for variant_key in sorted(old_variants, key=str):
            compared_variants += 1
            old_variant = old_variants[variant_key]
            new_variant = new_variants[variant_key]
            if _non_rate_variant(old_variant) != _non_rate_variant(new_variant):
                errors.append(f"{key}:{variant_key}: non-rate semantics changed")
            old_components = _component_index(old_variant)
            new_components = _component_index(new_variant)
            names = {
                name
                for name, scope in set(old_components) | set(new_components)
                if old_components.get((name, scope))
                != new_components.get((name, scope))
            }
            if names:
                changed_variants += 1
                changed_component_names.update(names)
            unexpected = names - ALLOWED_CHANGED_COMPONENTS
            if unexpected:
                errors.append(
                    f"{key}:{variant_key}: unexpected rate components changed: "
                    f"{sorted(unexpected)}"
                )
    if not changed_variants:
        errors.append("no rate component changed")
    output = {
        "format": "tilelogic_rate_precision_correction_validation_v1",
        "records": len(new_records),
        "compared_variants": compared_variants,
        "changed_variants": changed_variants,
        "changed_component_names": sorted(changed_component_names),
        "allowed_changed_component_names": sorted(ALLOWED_CHANGED_COMPONENTS),
        "non_rate_semantics_identical": not errors,
        "old_feature_samples_sha256": _sha256(old_path),
        "new_feature_samples_sha256": _sha256(new_path),
        "errors": errors[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
