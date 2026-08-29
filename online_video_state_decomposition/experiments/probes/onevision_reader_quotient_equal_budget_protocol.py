from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROTOCOL_ID = "onevision_reader_quotient_equal_budget_stage_a_20260830_v1"
SELECTION_SEED = 20260830
SOURCE_SAMPLES_PER_TASK = 24
SOURCE_TASKS = (
    "object_existence",
    "state_change",
    "scene_transition",
    "action_sequence",
    "moving_direction",
)


def source_sample_id(task: str, index: int) -> str:
    return f"{task}_{index:04d}"


def build_equal_budget_manifest(
    *,
    mvbench_manifest: dict[str, object],
    source_fit_summary: dict[str, object],
    videomme_manifest: dict[str, object],
    seed: int = SELECTION_SEED,
) -> dict[str, object]:
    if tuple(source_fit_summary["tasks"]) != SOURCE_TASKS:
        raise ValueError("source fit task identity mismatch")
    original_ids = {str(value) for value in source_fit_summary["sample_ids"]}
    if len(original_ids) != 20:
        raise ValueError("source fit must contain exactly 20 original videos")

    rng = np.random.default_rng(seed)
    source_entries = []
    for task in SOURCE_TASKS:
        records = mvbench_manifest["meta"][task]
        root = str(mvbench_manifest["root"][task])
        required = sorted(
            index
            for index in range(len(records))
            if source_sample_id(task, index) in original_ids
        )
        if len(required) != 4:
            raise ValueError(f"expected four original source videos for {task}")
        candidates = np.asarray(
            [index for index in range(len(records)) if index not in required],
            dtype=np.int64,
        )
        additional = sorted(
            int(value)
            for value in rng.choice(
                candidates,
                size=SOURCE_SAMPLES_PER_TASK - len(required),
                replace=False,
            )
        )
        for index in sorted(required + additional):
            record = records[index]
            source_entries.append(
                {
                    "domain": "source_mvbench",
                    "sample_id": source_sample_id(task, index),
                    "task": task,
                    "relative_video_path": str(Path(root) / str(record["video"])),
                    "original_source_basis_video": source_sample_id(task, index)
                    in original_ids,
                }
            )

    target_entries = []
    for entry in videomme_manifest["roles"]["calibration"]:
        target_entries.append(
            {
                "domain": "target_videomme",
                "sample_id": f"videomme_{entry['question_id']}",
                "duration": str(entry["duration"]),
                "relative_video_path": f"{entry['video_id']}.mp4",
            }
        )
    if len(source_entries) != 120 or len(target_entries) != 120:
        raise ValueError("equal-budget domains must each contain 120 videos")
    if len({entry["sample_id"] for entry in source_entries}) != 120:
        raise ValueError("source sample identifiers are not unique")
    if len({entry["sample_id"] for entry in target_entries}) != 120:
        raise ValueError("target sample identifiers are not unique")

    return {
        "protocol_id": PROTOCOL_ID,
        "selection_seed": seed,
        "frame_policy": "uniform16_pool_uniform8_reader",
        "role_counts": {
            "source_calibration": 120,
            "target_calibration": 120,
        },
        "roles": {
            "source_calibration": source_entries,
            "target_calibration": target_entries,
        },
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
