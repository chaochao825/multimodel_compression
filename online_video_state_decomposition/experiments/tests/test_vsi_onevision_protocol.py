from __future__ import annotations

import sys
from pathlib import Path


PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from vsi_onevision_protocol import (  # noqa: E402
    PROTOCOL_ID,
    ROLE_COUNTS,
    build_vsi_scene_split,
)


def synthetic_records() -> list[dict[str, object]]:
    records = []
    scene_index = 0
    for dataset, scene_count in (("a", 70), ("b", 81), ("c", 92)):
        for local_index in range(scene_count):
            for question_index in range(1 + local_index % 3):
                records.append(
                    {
                        "id": len(records),
                        "dataset": dataset,
                        "scene_name": f"scene-{scene_index:04d}",
                        "question_type": "direction",
                        "question": "Where?",
                        "options": ["A", "B"],
                        "ground_truth": "A",
                        "debiased": question_index % 2 == 0,
                    }
                )
            scene_index += 1
    return records


def test_vsi_split_is_scene_disjoint_balanced_and_deterministic() -> None:
    first = build_vsi_scene_split(synthetic_records())
    second = build_vsi_scene_split(synthetic_records())
    assert first == second
    assert first["protocol_id"] == PROTOCOL_ID
    assert first["role_counts"] == ROLE_COUNTS
    role_scenes = {
        role: {(entry["dataset"], entry["scene_name"]) for entry in entries}
        for role, entries in first["roles"].items()
    }
    assert len(role_scenes["calibration"]) == 120
    assert len(role_scenes["selection"]) == 60
    assert len(role_scenes["formal"]) == 63
    assert role_scenes["calibration"].isdisjoint(role_scenes["selection"])
    assert role_scenes["calibration"].isdisjoint(role_scenes["formal"])
    assert role_scenes["selection"].isdisjoint(role_scenes["formal"])
