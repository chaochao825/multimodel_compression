from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_wan_rcm_attention_atlas as analysis  # noqa: E402


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split in ("calibration", "evaluation"):
        for identity_index in range(4):
            for step in range(4):
                for layer in range(30):
                    layer_offset = layer * 0.00005
                    records.append(
                        {
                            "identity": f"{split}-{identity_index}",
                            "split": split,
                            "step": step,
                            "layer": layer,
                            "aggregate": 0.004 + layer_offset,
                            "worst_head": 0.008 + layer_offset,
                            "worst_query_tile": 0.008 + layer_offset,
                        }
                    )
    return records


def test_cell_summary_uses_worst_identity_and_registered_thresholds() -> None:
    records = _records()
    records[0]["aggregate"] = 0.009

    cells = analysis.summarize_cells(records)
    first = cells[0]

    assert len(cells) == 120
    assert first["calibration_aggregate_max"] == 0.009
    assert first["calibration_threshold_ratio"] == 1.125
    assert first["calibration_passes"] is False
    assert first["evaluation_passes"] is True


def test_run_summary_counts_passing_cells() -> None:
    records = _records()
    cells = analysis.summarize_cells(records)

    summary = analysis.summarize_run(records, cells)

    assert summary["record_count"] == 960
    assert summary["splits"]["calibration"]["passing_cell_count"] == 120
    assert summary["splits"]["evaluation"]["passing_cell_count"] == 120
