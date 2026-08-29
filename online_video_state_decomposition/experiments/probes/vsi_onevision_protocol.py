from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


PROTOCOL_ID = "vsi_onevision_reader_quotient_stage_a_20260830_v1"
SPLIT_SEED = 20260830
ROLE_COUNTS = {
    "calibration": 120,
    "selection": 60,
    "formal": 63,
}


def load_vsi_mcq_records(
    jsonl_path: Path,
    pruned_ids_path: Path,
) -> list[dict[str, object]]:
    pruned_ids = {
        int(value)
        for value in pruned_ids_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    records = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        options = record["options"]
        if options is None:
            continue
        if not 2 <= len(options) <= 26:
            raise ValueError(f"invalid option count for VSI question {record['id']}")
        answer = str(record["ground_truth"]).strip().upper()
        answer_index = ord(answer) - ord("A")
        if not 0 <= answer_index < len(options):
            raise ValueError(f"invalid answer for VSI question {record['id']}")
        records.append(
            {
                **record,
                "debiased": int(record["id"]) not in pruned_ids,
            }
        )
    return records


def scene_key(record: dict[str, object]) -> tuple[str, str]:
    return str(record["dataset"]), str(record["scene_name"])


def balanced_scene_order(
    scenes_by_dataset: dict[str, list[str]],
    *,
    seed: int,
) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    shuffled: dict[str, list[str]] = {}
    for dataset, scenes in sorted(scenes_by_dataset.items()):
        values = np.asarray(sorted(scenes), dtype=object)
        shuffled[dataset] = [str(value) for value in values[rng.permutation(len(values))]]

    positions = {dataset: 0 for dataset in shuffled}
    ordered: list[tuple[str, str]] = []
    while len(ordered) < sum(len(values) for values in shuffled.values()):
        available = [
            dataset
            for dataset, values in shuffled.items()
            if positions[dataset] < len(values)
        ]
        dataset = min(
            available,
            key=lambda value: (
                positions[value] / len(shuffled[value]),
                value,
            ),
        )
        position = positions[dataset]
        ordered.append((dataset, shuffled[dataset][position]))
        positions[dataset] += 1
    return ordered


def build_vsi_scene_split(
    records: list[dict[str, object]],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, object]:
    by_scene: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        by_scene.setdefault(scene_key(record), []).append(record)

    expected_scenes = sum(ROLE_COUNTS.values())
    if len(by_scene) != expected_scenes:
        raise ValueError(
            f"expected {expected_scenes} VSI MCQ scenes, found {len(by_scene)}"
        )

    scenes_by_dataset: dict[str, list[str]] = {}
    for dataset, scene_name in by_scene:
        scenes_by_dataset.setdefault(dataset, []).append(scene_name)
    ordered = balanced_scene_order(scenes_by_dataset, seed=seed)

    roles: dict[str, list[dict[str, object]]] = {}
    offset = 0
    for role, count in ROLE_COUNTS.items():
        entries = []
        for dataset, scene_name in ordered[offset : offset + count]:
            scene_records = sorted(
                by_scene[(dataset, scene_name)],
                key=lambda record: int(record["id"]),
            )
            entries.append(
                {
                    "dataset": dataset,
                    "scene_name": scene_name,
                    "sample_id": f"vsi_{dataset}_{scene_name}",
                    "relative_video_path": f"{dataset}/{scene_name}.mp4",
                    "question_ids": [int(record["id"]) for record in scene_records],
                    "debiased_question_ids": [
                        int(record["id"])
                        for record in scene_records
                        if bool(record["debiased"])
                    ],
                    "question_type_counts": dict(
                        sorted(
                            Counter(
                                str(record["question_type"])
                                for record in scene_records
                            ).items()
                        )
                    ),
                }
            )
        roles[role] = entries
        offset += count

    role_scenes = {
        role: {
            (str(entry["dataset"]), str(entry["scene_name"]))
            for entry in entries
        }
        for role, entries in roles.items()
    }
    role_names = tuple(ROLE_COUNTS)
    for left_index, left in enumerate(role_names):
        for right in role_names[left_index + 1 :]:
            if role_scenes[left] & role_scenes[right]:
                raise ValueError(f"VSI roles {left} and {right} overlap")

    return {
        "protocol_id": PROTOCOL_ID,
        "selection_seed": seed,
        "split_unit": "scene",
        "frame_policy": "uniform16_pool_uniform8_reader",
        "role_counts": ROLE_COUNTS,
        "roles": roles,
    }
