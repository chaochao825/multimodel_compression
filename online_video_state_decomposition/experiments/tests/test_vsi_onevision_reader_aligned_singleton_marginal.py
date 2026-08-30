from __future__ import annotations

import pytest

pytest.importorskip("torch")

from probe_vsi_onevision_reader_aligned_singleton_marginal import (
    diagnostic_decision,
    path_regressions,
    stable_benefit_order,
)


def test_stable_benefit_order_breaks_ties_by_group_index() -> None:
    assert stable_benefit_order([0.2, 0.5, 0.5, -0.1]) == [1, 2, 0, 3]


def test_path_regressions_detects_match_and_kl_reversal() -> None:
    rows = [
        {"refined_group_count": 0, "prediction_match": 0, "candidate_kl": 0.10},
        {"refined_group_count": 49, "prediction_match": 1, "candidate_kl": 0.05},
        {"refined_group_count": 98, "prediction_match": 0, "candidate_kl": 0.06},
    ]
    assert path_regressions(rows, through_group_count=98) == (1, 1)


def _summary(mean: float, p95: float, mismatch: int = 0) -> dict[str, float | int]:
    return {
        "mismatch_count": mismatch,
        "harmful_count": mismatch,
        "candidate_kl_mean": mean,
        "candidate_kl_p95": p95,
    }


def test_diagnostic_decision_accepts_strict_monotone_path() -> None:
    summaries = {
        0: _summary(0.05, 0.10, mismatch=1),
        49: _summary(0.02, 0.04, mismatch=1),
        98: _summary(0.009, 0.019),
    }
    rows = [
        {"sample_id": "a", "refined_group_count": 0, "prediction_match": 0, "candidate_kl": 0.05},
        {"sample_id": "a", "refined_group_count": 49, "prediction_match": 1, "candidate_kl": 0.02},
        {"sample_id": "a", "refined_group_count": 98, "prediction_match": 1, "candidate_kl": 0.009},
    ]
    assert diagnostic_decision(summaries, rows) == (
        "STRICT_STATIC_READER_PATH",
        98,
    )


def test_diagnostic_decision_rejects_match_regression() -> None:
    summaries = {
        0: _summary(0.05, 0.10, mismatch=1),
        49: _summary(0.03, 0.06),
        98: _summary(0.009, 0.019),
    }
    rows = [
        {"sample_id": "a", "refined_group_count": 0, "prediction_match": 0, "candidate_kl": 0.05},
        {"sample_id": "a", "refined_group_count": 49, "prediction_match": 1, "candidate_kl": 0.02},
        {"sample_id": "a", "refined_group_count": 98, "prediction_match": 0, "candidate_kl": 0.009},
    ]
    assert diagnostic_decision(summaries, rows) == ("NO_STATIC_READER_PATH", None)
