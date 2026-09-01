#!/usr/bin/env python3
"""Finalize EXP-054 S1 from a complete scalar-record CSV after runner repair."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import rcm_attention_atlas_core as atlas_core
import run_wan_rcm_attention_atlas as runner
from experiment_artifacts import JsonlEventLog, atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--s0-manifest", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[atlas_core.CellMetric]:
    records: list[atlas_core.CellMetric] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            record = atlas_core.CellMetric(
                identity=raw["identity"],
                split=raw["split"],
                step=int(raw["step"]),
                layer=int(raw["layer"]),
                aggregate=float(raw["aggregate"]),
                worst_head=float(raw["worst_head"]),
                worst_query_tile=float(raw["worst_query_tile"]),
            )
            if not all(
                math.isfinite(value)
                for value in (
                    record.aggregate,
                    record.worst_head,
                    record.worst_query_tile,
                )
            ):
                raise ValueError(f"non-finite metric in {record.identity}/{record.cell}")
            records.append(record)
    if len(records) != 8 * atlas_core.CELL_COUNT:
        raise ValueError(f"expected 960 complete records, got {len(records)}")
    return records


def finalize(
    config: dict[str, Any],
    s0_manifest: dict[str, Any],
    records: list[atlas_core.CellMetric],
) -> dict[str, object]:
    if s0_manifest["experiment_id"] != "EXP-054":
        raise ValueError("finalizer requires an EXP-054 S0 manifest")
    if s0_manifest["stage"] != "s0-smoke" or not s0_manifest["result"]["advance"]:
        raise ValueError("finalizer requires a passing S0 manifest")
    attention_speedup = float(
        s0_manifest["result"]["benchmark"]["attention_speedup"]
    )
    calibration_ids = runner._identity_ids(config, "calibration")
    evaluation_ids = runner._identity_ids(config, "evaluation")
    calibration_records = [
        record for record in records if record.split == "calibration"
    ]
    evaluation_records = [
        record for record in records if record.split == "evaluation"
    ]
    atlas = atlas_core.freeze_and_evaluate_atlas(
        calibration_records,
        evaluation_records,
        calibration_ids,
        evaluation_ids,
        runner._thresholds(config["thresholds"]["calibration"]),
        runner._thresholds(config["thresholds"]["evaluation"]),
        int(config["materiality"]["minimum_selected_cells"]),
    )
    projected_seconds, projected_speedup = atlas_core.projected_request(
        float(config["materiality"]["baseline_request_seconds"]),
        float(config["materiality"]["baseline_denoiser_seconds"]),
        float(config["materiality"]["historical_self_attention_share"]),
        float(atlas["coverage"]),
        attention_speedup,
    )
    atlas.update(
        {
            "attention_speedup": attention_speedup,
            "projected_request_seconds": projected_seconds,
            "projected_request_speedup": projected_speedup,
            "passes_materiality": projected_speedup
            >= float(config["materiality"]["minimum_projected_request_speedup"]),
        }
    )
    atlas["advance"] = bool(
        atlas["passes_transfer_and_count"] and atlas["passes_materiality"]
    )
    return {
        "record_count": len(records),
        "calibration_identity_count": len(calibration_ids),
        "evaluation_identity_count": len(evaluation_ids),
        "atlas": atlas,
        "advance": atlas["advance"],
    }


def main() -> None:
    args = parse_args()
    config, _base_config = runner.load_configs(args.config.resolve())
    output_dir = args.output_dir.resolve()
    records_path = args.records_csv.resolve()
    if records_path.parent != output_dir:
        raise ValueError("records CSV must belong to the S1 output directory")
    for name in ("atlas.json", "manifest.json", "SUCCESS.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite existing S1 artifact: {name}")

    s0_manifest = json.loads(args.s0_manifest.resolve().read_text(encoding="utf-8"))
    records = load_records(records_path)
    result = finalize(config, s0_manifest, records)
    log = JsonlEventLog(output_dir / "events.jsonl", "EXP-054-s1-finalize")
    log.emit(
        "repair_finalize_start",
        record_count=len(records),
        repair_index=2,
        repair="allow_zero_coverage_projection",
    )
    atomic_write_json(output_dir / "atlas.json", result["atlas"])
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "experiment_id": config["experiment_id"],
            "gate_id": config["gate_id"],
            "stage": "s1-atlas",
            "source_commit": s0_manifest["source_commit"],
            "recovered_from_complete_scalar_records": True,
            "repair_index": 2,
            "result": result,
            "environment": s0_manifest["environment"],
        },
    )
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {"status": "complete", "stage": "s1-atlas", "advance": result["advance"]},
    )
    log.emit("repair_finalize_complete", advance=result["advance"])
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
