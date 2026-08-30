from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_batched_current_support_marginal import (  # noqa: E402
    classify_outcome,
    path_regressions,
    stable_top_batch,
)


def test_stable_top_batch_breaks_ties_by_group_index() -> None:
    benefits = {4: 0.5, 2: 0.5, 1: -0.1, 3: 0.7}
    assert stable_top_batch(benefits, count=3) == [3, 2, 4]


def test_path_regressions_count_match_and_kl_failures() -> None:
    rows = [
        {"selected_group_count": 0, "prediction_match": 1, "candidate_kl": 0.2},
        {"selected_group_count": 49, "prediction_match": 0, "candidate_kl": 0.1},
        {"selected_group_count": 98, "prediction_match": 1, "candidate_kl": 0.12},
    ]
    assert path_regressions(rows, through_group_count=98) == (1, 1)


def test_classify_outcome_separates_mass_gain_and_boundary() -> None:
    def metrics(mismatch: int, harmful: int, kl: float) -> dict[str, float | int]:
        return {
            "mismatch_count": mismatch,
            "harmful_count": harmful,
            "candidate_kl_mean": kl,
        }

    summaries = {
        "positioned_equal_mass": {196: metrics(1, 0, 0.02)},
        "positioned_group_mass": {196: metrics(0, 0, 0.008)},
    }
    assert (
        classify_outcome(
            summaries,
            {
                "positioned_equal_mass": None,
                "positioned_group_mass": 196,
            },
        )
        == "MASS_CURRENT_SUPPORT_HEADROOM"
    )
    assert (
        classify_outcome(
            {
                "positioned_equal_mass": {196: metrics(0, 0, 0.03)},
                "positioned_group_mass": {196: metrics(1, 0, 0.03)},
            },
            {
                "positioned_equal_mass": None,
                "positioned_group_mass": None,
            },
        )
        == "DECISION_ONLY_BOUNDARY"
    )
