from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import finalize_wan_rcm_attention_atlas as finalizer  # noqa: E402
import rcm_attention_atlas_core as core  # noqa: E402


class FinalizerTest(unittest.TestCase):
    def test_complete_zero_coverage_records_finalize_as_no_advance(self) -> None:
        records = [
            core.CellMetric(
                identity=identity,
                split=split,
                step=step,
                layer=layer,
                aggregate=0.02,
                worst_head=0.03,
                worst_query_tile=0.03,
            )
            for identity, split in (
                *((f"cal{index}", "calibration") for index in range(4)),
                *((f"eval{index}", "evaluation") for index in range(4)),
            )
            for step in range(core.CELL_STEPS)
            for layer in range(core.CELL_LAYERS)
        ]
        config = {
            "atlas_identities": [
                {"identity": record.identity, "split": record.split}
                for record in records[:: core.CELL_COUNT]
            ],
            "thresholds": {
                "calibration": {
                    "aggregate": 0.008,
                    "worst_head": 0.016,
                    "worst_query_tile": 0.016,
                },
                "evaluation": {
                    "aggregate": 0.01,
                    "worst_head": 0.02,
                    "worst_query_tile": 0.02,
                },
            },
            "materiality": {
                "minimum_selected_cells": 87,
                "minimum_projected_request_speedup": 1.05,
                "baseline_request_seconds": 9.637995031895116,
                "baseline_denoiser_seconds": 3.20536530460231,
                "historical_self_attention_share": 0.5388086760322437,
            },
        }
        s0 = {
            "experiment_id": "EXP-054",
            "stage": "s0-smoke",
            "result": {
                "advance": True,
                "benchmark": {"attention_speedup": 1.586376845918112},
            },
        }
        result = finalizer.finalize(config, s0, records)
        self.assertEqual(result["atlas"]["selected_cell_count"], 0)
        self.assertEqual(result["atlas"]["projected_request_speedup"], 1.0)
        self.assertFalse(result["advance"])

    def test_loader_rejects_incomplete_record_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "identity",
                        "split",
                        "step",
                        "layer",
                        "aggregate",
                        "worst_head",
                        "worst_query_tile",
                    ),
                )
                writer.writeheader()
            with self.assertRaisesRegex(ValueError, "expected 960"):
                finalizer.load_records(path)


if __name__ == "__main__":
    unittest.main()
