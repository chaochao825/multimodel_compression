from __future__ import annotations

import sys
from pathlib import Path


PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from onevision_reader_quotient_equal_budget_protocol import (  # noqa: E402
    PROTOCOL_ID,
    SOURCE_TASKS,
    build_equal_budget_manifest,
)


def test_equal_budget_manifest_is_deterministic_and_preserves_source_basis() -> None:
    metadata = {}
    roots = {}
    original_ids = []
    for task in SOURCE_TASKS:
        roots[task] = f"videos/{task}"
        metadata[task] = [{"video": f"{index}.mp4"} for index in range(200)]
        original_ids.extend(f"{task}_{index:04d}" for index in (1, 3, 5, 7))
    source_summary = {
        "tasks": list(SOURCE_TASKS),
        "sample_ids": original_ids,
    }
    videomme = {
        "roles": {
            "calibration": [
                {
                    "question_id": f"q-{index}",
                    "video_id": f"v-{index}",
                    "duration": "short",
                }
                for index in range(120)
            ]
        }
    }
    first = build_equal_budget_manifest(
        mvbench_manifest={"meta": metadata, "root": roots},
        source_fit_summary=source_summary,
        videomme_manifest=videomme,
    )
    second = build_equal_budget_manifest(
        mvbench_manifest={"meta": metadata, "root": roots},
        source_fit_summary=source_summary,
        videomme_manifest=videomme,
    )
    assert first == second
    assert first["protocol_id"] == PROTOCOL_ID
    assert len(first["roles"]["source_calibration"]) == 120
    assert len(first["roles"]["target_calibration"]) == 120
    retained = {
        entry["sample_id"]
        for entry in first["roles"]["source_calibration"]
        if entry["original_source_basis_video"]
    }
    assert retained == set(original_ids)
