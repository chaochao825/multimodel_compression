from __future__ import annotations

import sys
from pathlib import Path


PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from videomme_onevision_domain_residual_protocol import (  # noqa: E402
    CALIBRATION_PER_DURATION,
    CANDIDATES,
    DURATIONS,
    FORMAL_PER_DURATION,
    PROTOCOL_ID,
    SELECTION_PER_DURATION,
    build_domain_residual_manifest,
)
from videomme_onevision_pca_protocol import (  # noqa: E402
    HISTORICAL_VIDEO_IDS,
    SPLIT_PROTOCOL_ID,
)


def source_split() -> dict[str, object]:
    samples = [
        {
            "question_id": f"source-{duration}-{index}",
            "video_id": f"source-{duration}-{index}",
            "duration": duration,
            "domain": "source",
            "task_type": "source",
        }
        for duration in DURATIONS
        for index in range(200)
    ]
    return {"protocol_id": SPLIT_PROTOCOL_ID, "samples": samples}


def records() -> list[dict[str, object]]:
    return [
        {
            "question_id": f"fresh-{duration}-{video_index}-{question_index}",
            "videoID": f"fresh-{duration}-{video_index}",
            "duration": duration,
            "domain": "fresh",
            "task_type": "fresh",
            "question": "Question?",
            "options": ["A. One", "B. Two", "C. Three", "D. Four"],
            "answer": "A",
        }
        for duration in DURATIONS
        for video_index in range(90)
        for question_index in range(3)
    ]


def test_manifest_is_deterministic_disjoint_and_balanced() -> None:
    rows = records()
    available = {str(row["videoID"]) for row in rows}
    first = build_domain_residual_manifest(
        source_split=source_split(),
        records=rows,
        available_video_ids=available,
    )
    second = build_domain_residual_manifest(
        source_split=source_split(),
        records=rows,
        available_video_ids=available,
    )
    assert first == second
    assert first["protocol_id"] == PROTOCOL_ID
    assert tuple(first["candidates"]) == CANDIDATES
    expected = {
        "calibration": CALIBRATION_PER_DURATION * len(DURATIONS),
        "selection": SELECTION_PER_DURATION * len(DURATIONS),
        "formal": FORMAL_PER_DURATION * len(DURATIONS),
    }
    assert first["role_counts"] == expected
    role_videos = {
        role: {entry["video_id"] for entry in entries}
        for role, entries in first["roles"].items()
    }
    assert role_videos["calibration"].isdisjoint(role_videos["selection"])
    assert role_videos["formal"].isdisjoint(role_videos["calibration"])
    assert role_videos["formal"].isdisjoint(role_videos["selection"])
    assert role_videos["formal"].isdisjoint(HISTORICAL_VIDEO_IDS)
    for role, count_per_duration in (
        ("calibration", CALIBRATION_PER_DURATION),
        ("selection", SELECTION_PER_DURATION),
        ("formal", FORMAL_PER_DURATION),
    ):
        assert {
            duration: sum(
                entry["duration"] == duration for entry in first["roles"][role]
            )
            for duration in DURATIONS
        } == {duration: count_per_duration for duration in DURATIONS}
