#!/usr/bin/env python3
"""Recompute saved task scores while preserving the original JSONL in trash."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from tilespec_ex.metrics import dataset_score, normalize_answer


def rescore_record(record: dict[str, object]) -> dict[str, int]:
    variants = record["variants"]
    if not isinstance(variants, list):
        raise TypeError("variants must be a list")
    baseline = next(
        (item for item in variants if isinstance(item, dict) and item.get("method") == "none"),
        None,
    )
    if baseline is None:
        raise ValueError("record has no full-quality baseline")
    baseline_normalized = normalize_answer(baseline["prediction"])
    changed = {"scores": 0, "normalized_predictions": 0, "agreements": 0}
    dataset = str(record["dataset"])
    answers = record["answers"]
    if not isinstance(answers, list):
        raise TypeError("answers must be a list")
    for variant in variants:
        if not isinstance(variant, dict):
            raise TypeError("variant entries must be objects")
        prediction = variant["prediction"]
        score = dataset_score(dataset, prediction, answers)
        normalized = normalize_answer(prediction)
        agrees = float(normalized == baseline_normalized)
        if abs(float(variant.get("score", 0.0)) - score) > 1e-12:
            changed["scores"] += 1
        if variant.get("normalized_prediction") != normalized:
            changed["normalized_predictions"] += 1
        if abs(float(variant.get("agrees_with_full", 0.0)) - agrees) > 1e-12:
            changed["agreements"] += 1
        variant["score"] = score
        variant["normalized_prediction"] = normalized
        variant["agrees_with_full"] = agrees
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--trash-root", type=Path, required=True)
    args = parser.parse_args()
    results = args.results_dir.resolve()
    source = results / "quality_samples.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)

    temporary = results / "quality_samples.rescored.tmp"
    records = 0
    variants = 0
    changed = {"scores": 0, "normalized_predictions": 0, "agreements": 0}
    with source.open("r", encoding="utf-8") as input_handle, temporary.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_changes = rescore_record(record)
            variants += len(record["variants"])
            for key, value in record_changes.items():
                changed[key] += value
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1
    if records == 0:
        raise RuntimeError("quality result file is empty")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = args.trash_root.resolve() / f"{timestamp}-quality-rescore" / source.name
    backup.parent.mkdir(parents=True, exist_ok=False)
    shutil.move(str(source), str(backup))
    shutil.move(str(temporary), str(source))
    summary = {
        "records": records,
        "variants": variants,
        "changed_scores": changed["scores"],
        "changed_normalized_predictions": changed["normalized_predictions"],
        "changed_agreements": changed["agreements"],
        "original_staged_at": str(backup),
        "metric": "GQA exact, TextVQA VQA-style, ChartQA relaxed numeric",
    }
    (results / "rescore_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
