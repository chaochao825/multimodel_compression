from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from analyze_videomme_onevision_pca_replication import (  # noqa: E402
    classify_replication,
    validate_identity,
)
from videomme_onevision_pca_protocol import (  # noqa: E402
    DURATIONS,
    FORMAL_SAMPLES_PER_DURATION,
    FORMAL_SEED,
    HISTORICAL_VIDEO_IDS,
    SPLIT_PROTOCOL_ID,
    build_frozen_split,
)


def valid_summary() -> dict[str, float | int]:
    return {
        "samples": 600,
        "failed_samples": 0,
        "duplicate_samples": 0,
        "duplicate_videos": 0,
        "nonfinite_metrics": 0,
        "max_injection_abs": 0.0,
        "max_state_bytes": 2_860_032,
        "min_compression_ratio": 7.85,
        "reference_accuracy": 0.50,
        "candidate_accuracy": 0.50,
        "harmful_upper_95": 0.015,
        "prediction_agreement": 0.99,
        "minimum_duration_accuracy_delta": -0.02,
        "harmful_rate": 0.005,
    }


def synthetic_records(videos_per_duration: int = 205) -> list[dict[str, object]]:
    rows = []
    for duration_index, duration in enumerate(DURATIONS):
        for video_index in range(videos_per_duration):
            video_id = f"new_{duration_index}_{video_index:04d}"
            for question_index in range(3):
                rows.append(
                    {
                        "videoID": video_id,
                        "question_id": f"{duration_index}{video_index:04d}-{question_index}",
                        "duration": duration,
                        "domain": f"domain_{video_index % 3}",
                        "task_type": f"task_{question_index}",
                        "question": "Question?",
                        "options": ["A. One", "B. Two", "C. Three", "D. Four"],
                        "answer": "A",
                    }
                )
    return rows


def test_split_is_deterministic_unique_and_balanced() -> None:
    records = synthetic_records()
    available = {str(record["videoID"]) for record in records}
    first = build_frozen_split(records, available_video_ids=available)
    second = build_frozen_split(records, available_video_ids=available)
    assert first == second
    assert len(first["samples"]) == 600
    assert len({entry["video_id"] for entry in first["samples"]}) == 600
    assert len({entry["question_id"] for entry in first["samples"]}) == 600
    assert {
        duration: sum(entry["duration"] == duration for entry in first["samples"])
        for duration in DURATIONS
    } == {duration: 200 for duration in DURATIONS}


def test_split_excludes_historical_videos() -> None:
    records = synthetic_records()
    historical = next(iter(HISTORICAL_VIDEO_IDS))
    records.extend(
        {
            "videoID": historical,
            "question_id": f"historical-{index}",
            "duration": "short",
            "domain": "domain",
            "task_type": "task",
            "question": "Question?",
            "options": ["A. One", "B. Two", "C. Three", "D. Four"],
            "answer": "A",
        }
        for index in range(3)
    )
    available = {str(record["videoID"]) for record in records}
    split = build_frozen_split(records, available_video_ids=available)
    assert historical not in {entry["video_id"] for entry in split["samples"]}


def test_replication_decision_guards() -> None:
    summary = valid_summary()
    assert classify_replication(summary) == "PASS"
    assert (
        classify_replication({**summary, "prediction_agreement": 0.97})
        == "BOUNDARY"
    )
    assert (
        classify_replication({**summary, "candidate_accuracy": 0.44})
        == "ADVERSE"
    )
    assert classify_replication({**summary, "duplicate_videos": 1}) == "INVALID"


def valid_identity() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    split_rows = [
        {
            "question_id": f"{duration_index}{index:04d}-0",
            "video_id": f"video_{duration_index}_{index:04d}",
            "duration": duration,
            "domain": "domain",
            "task_type": "task",
        }
        for duration_index, duration in enumerate(DURATIONS)
        for index in range(FORMAL_SAMPLES_PER_DURATION)
    ]
    split = {
        "protocol_id": SPLIT_PROTOCOL_ID,
        "selection_seed": FORMAL_SEED,
        "samples_per_duration": FORMAL_SAMPLES_PER_DURATION,
        "historical_video_exclusions": sorted(HISTORICAL_VIDEO_IDS),
        "frame_policy": "uniform16_pool_uniform8_reader",
        "samples": split_rows,
    }
    configuration = {
        "benchmark": "Video-MME",
        "candidate": "pca_r456_s0",
        "claim_tier": "onevision_pca_r456_cross_domain_replication",
        "split_protocol_id": SPLIT_PROTOCOL_ID,
        "selection_seed": FORMAL_SEED,
        "samples_per_duration": FORMAL_SAMPLES_PER_DURATION,
        "feature_pool_frames": 16,
        "frame_budget": 8,
        "frame_policy": "uniform16_pool_uniform8_reader",
        "subtitles": "disabled",
        "duration_counts": {
            duration: FORMAL_SAMPLES_PER_DURATION for duration in DURATIONS
        },
        "sample_ids": [f"videomme_{entry['question_id']}" for entry in split_rows],
        "video_ids": [entry["video_id"] for entry in split_rows],
    }
    codec_metadata = {
        "rank": 456,
        "feature_pool_frames": 16,
        "frame_budget": 8,
    }
    return configuration, codec_metadata, split


def test_identity_validation_is_fail_closed() -> None:
    configuration, codec_metadata, split = valid_identity()
    assert len(validate_identity(configuration, codec_metadata, split)) == 600
    with pytest.raises(ValueError, match="frame_policy"):
        validate_identity(
            {**configuration, "frame_policy": "recent"},
            codec_metadata,
            split,
        )
    with pytest.raises(ValueError, match="codec field rank"):
        validate_identity(configuration, {**codec_metadata, "rank": 455}, split)
