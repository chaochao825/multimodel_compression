from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from analyze_onevision_pca_confirmation import (  # noqa: E402
    EXPECTED_EXCLUDED_BY_TASK,
    EXPECTED_TASKS,
    classify_confirmation,
    validate_configuration,
)


def valid_summary() -> dict[str, float | int]:
    return {
        "samples": 500,
        "failed_samples": 0,
        "duplicate_samples": 0,
        "nonfinite_metrics": 0,
        "max_injection_abs": 0.0,
        "max_state_bytes": 2_860_032,
        "min_compression_ratio": 7.85,
        "reference_accuracy": 0.60,
        "candidate_accuracy": 0.60,
        "harmful_upper_95": 0.015,
        "prediction_agreement": 0.99,
        "minimum_task_accuracy_delta": -0.02,
        "harmful_rate": 0.004,
    }


def test_confirmation_pass_requires_every_guard() -> None:
    summary = valid_summary()
    assert classify_confirmation(summary) == "PASS"
    assert (
        classify_confirmation({**summary, "harmful_upper_95": 0.021})
        == "BOUNDARY"
    )
    assert (
        classify_confirmation({**summary, "minimum_task_accuracy_delta": -0.06})
        == "BOUNDARY"
    )


def test_confirmation_classifies_material_harm_as_adverse() -> None:
    summary = valid_summary()
    assert (
        classify_confirmation({**summary, "candidate_accuracy": 0.54})
        == "ADVERSE"
    )
    assert (
        classify_confirmation({**summary, "prediction_agreement": 0.94})
        == "ADVERSE"
    )


def test_confirmation_rejects_incomplete_evidence() -> None:
    summary = valid_summary()
    assert classify_confirmation({**summary, "samples": 499}) == "INVALID"
    assert classify_confirmation({**summary, "nonfinite_metrics": 1}) == "INVALID"


def valid_configuration() -> tuple[dict[str, object], dict[str, object]]:
    sample_ids = [
        f"{task}_{index:04d}"
        for task in EXPECTED_TASKS
        for index in range(100)
    ]
    configuration = {
        "candidate": "pca_r456_s0",
        "claim_tier": "onevision_pca_r456_untouched_task_confirmation",
        "eligibility": "2_to_26_candidates_and_answer_in_candidates",
        "samples_per_task": 100,
        "selection_seed": 20260829,
        "sampled_frames": 32,
        "feature_pool_frames": 16,
        "frame_budget": 8,
        "tasks": list(EXPECTED_TASKS),
        "excluded_by_task": EXPECTED_EXCLUDED_BY_TASK,
        "sample_ids": sample_ids,
    }
    codec_metadata = {
        "rank": 456,
        "sampled_frames": 32,
        "feature_pool_frames": 16,
        "frame_budget": 8,
    }
    return configuration, codec_metadata


def test_confirmation_validates_frozen_identity() -> None:
    configuration, codec_metadata = valid_configuration()
    assert len(validate_configuration(configuration, codec_metadata)) == 500

    with pytest.raises(ValueError, match="selection_seed"):
        validate_configuration(
            {**configuration, "selection_seed": 20260830},
            codec_metadata,
        )
    with pytest.raises(ValueError, match="codec field rank"):
        validate_configuration(configuration, {**codec_metadata, "rank": 455})
