from __future__ import annotations

from collections import defaultdict

import numpy as np

from videomme_onevision_pca_protocol import (
    DURATIONS,
    HISTORICAL_VIDEO_IDS,
    SPLIT_PROTOCOL_ID as SOURCE_SPLIT_PROTOCOL_ID,
    record_is_eligible,
)


PROTOCOL_ID = "videomme_onevision_domain_residual_rank456_20260829_v1"
SELECTION_SEED = 26_082_901
CALIBRATION_PER_DURATION = 40
SELECTION_PER_DURATION = 60
FORMAL_PER_DURATION = 85
RANK = 456
SWAP_RANKS = (16, 32, 64, 96, 128)
SOURCE_CANDIDATE = "source_r456"
MEAN_CANDIDATE = "target_mean_source_r456"
TARGET_CANDIDATE = "target_pca_r456"
CANDIDATES = (
    SOURCE_CANDIDATE,
    MEAN_CANDIDATE,
    *(f"residual_swap_r{rank}" for rank in SWAP_RANKS),
    TARGET_CANDIDATE,
)


def _entry(record: dict[str, object]) -> dict[str, str]:
    return {
        "question_id": str(record["question_id"]),
        "video_id": str(record["videoID"]),
        "duration": str(record["duration"]),
        "domain": str(record["domain"]),
        "task_type": str(record["task_type"]),
    }


def build_domain_residual_manifest(
    *,
    source_split: dict[str, object],
    records: list[dict[str, object]],
    available_video_ids: set[str],
    seed: int = SELECTION_SEED,
) -> dict[str, object]:
    if source_split["protocol_id"] != SOURCE_SPLIT_PROTOCOL_ID:
        raise ValueError("source split protocol mismatch")
    source_entries = list(source_split["samples"])
    source_video_ids = {str(item["video_id"]) for item in source_entries}
    if len(source_entries) != 600 or len(source_video_ids) != 600:
        raise ValueError("source split must contain 600 unique videos")

    source_by_duration: dict[str, list[dict[str, object]]] = {
        duration: [] for duration in DURATIONS
    }
    for item in source_entries:
        source_by_duration[str(item["duration"])].append(item)
    if any(len(source_by_duration[duration]) != 200 for duration in DURATIONS):
        raise ValueError("source split must contain 200 videos per duration")

    rng = np.random.default_rng(seed)
    roles: dict[str, list[dict[str, object]]] = {
        "calibration": [],
        "selection": [],
        "formal": [],
    }
    for duration in DURATIONS:
        items = sorted(
            source_by_duration[duration],
            key=lambda item: (str(item["video_id"]), str(item["question_id"])),
        )
        order = rng.permutation(len(items))
        calibration_positions = order[:CALIBRATION_PER_DURATION]
        selection_positions = order[
            CALIBRATION_PER_DURATION : CALIBRATION_PER_DURATION
            + SELECTION_PER_DURATION
        ]
        roles["calibration"].extend(items[int(index)] for index in calibration_positions)
        roles["selection"].extend(items[int(index)] for index in selection_positions)

    untouched: dict[str, dict[str, list[dict[str, object]]]] = {
        duration: defaultdict(list) for duration in DURATIONS
    }
    for record in records:
        if not record_is_eligible(record):
            continue
        video_id = str(record["videoID"])
        duration = str(record["duration"])
        if (
            video_id in source_video_ids
            or video_id in HISTORICAL_VIDEO_IDS
            or video_id not in available_video_ids
        ):
            continue
        untouched[duration][video_id].append(record)

    for duration in DURATIONS:
        video_ids = sorted(untouched[duration])
        if len(video_ids) < FORMAL_PER_DURATION:
            raise ValueError(
                f"duration {duration} has only {len(video_ids)} untouched videos"
            )
        selected_positions = sorted(
            int(index)
            for index in rng.choice(
                len(video_ids),
                size=FORMAL_PER_DURATION,
                replace=False,
            )
        )
        for position in selected_positions:
            questions = sorted(
                untouched[duration][video_ids[position]],
                key=lambda record: str(record["question_id"]),
            )
            record = questions[int(rng.integers(0, len(questions)))]
            roles["formal"].append(_entry(record))

    expected_counts = {
        "calibration": CALIBRATION_PER_DURATION * len(DURATIONS),
        "selection": SELECTION_PER_DURATION * len(DURATIONS),
        "formal": FORMAL_PER_DURATION * len(DURATIONS),
    }
    role_video_ids: dict[str, set[str]] = {}
    for role, expected in expected_counts.items():
        entries = roles[role]
        videos = {str(item["video_id"]) for item in entries}
        questions = {str(item["question_id"]) for item in entries}
        if len(entries) != expected or len(videos) != expected or len(questions) != expected:
            raise ValueError(f"role {role} is incomplete or contains duplicates")
        role_video_ids[role] = videos
    if role_video_ids["calibration"] & role_video_ids["selection"]:
        raise ValueError("calibration and selection videos overlap")
    if role_video_ids["formal"] & source_video_ids:
        raise ValueError("formal videos overlap the observed source split")
    if role_video_ids["formal"] & HISTORICAL_VIDEO_IDS:
        raise ValueError("formal videos overlap historical diagnostics")

    return {
        "protocol_id": PROTOCOL_ID,
        "source_split_protocol_id": SOURCE_SPLIT_PROTOCOL_ID,
        "selection_seed": seed,
        "rank": RANK,
        "swap_ranks": list(SWAP_RANKS),
        "candidates": list(CANDIDATES),
        "frame_policy": "uniform16_pool_uniform8_reader",
        "role_counts": expected_counts,
        "roles": roles,
    }
