from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_target_risk_budget_frontier import (  # noqa: E402
    budget_decision,
    hybrid_token_count,
    risk_mass_capture,
)


def test_hybrid_token_count_matches_registered_retention() -> None:
    assert hybrid_token_count(
        group_count=392,
        group_size=4,
        refined_group_count=0,
    ) == 392
    assert hybrid_token_count(
        group_count=392,
        group_size=4,
        refined_group_count=98,
    ) == 686
    assert hybrid_token_count(
        group_count=392,
        group_size=4,
        refined_group_count=392,
    ) == 1568


def test_risk_mass_capture_handles_empty_and_selected_support() -> None:
    risk = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert risk_mass_capture(risk, torch.empty(0, dtype=torch.long)) == 0.0
    assert risk_mass_capture(risk, torch.tensor([1, 3])) == pytest.approx(0.6)


def test_budget_decision_distinguishes_strong_weak_and_no_window() -> None:
    base = {
        "mismatch_count": 1,
        "harmful_count": 0,
        "candidate_kl_mean": 0.02,
        "candidate_kl_p95": 0.04,
    }
    strong = {
        98: base,
        160: {
            "mismatch_count": 0,
            "harmful_count": 0,
            "candidate_kl_mean": 0.009,
            "candidate_kl_p95": 0.019,
        },
    }
    assert budget_decision(strong) == ("STRONG_CAPACITY_WINDOW", 160)

    weak = {
        196: base,
        245: {
            "mismatch_count": 0,
            "harmful_count": 0,
            "candidate_kl_mean": 0.02,
            "candidate_kl_p95": 0.04,
        },
    }
    assert budget_decision(weak) == ("WEAK_CAPACITY_WINDOW", 245)
    assert budget_decision({294: base}) == ("NO_USEFUL_CAPACITY_WINDOW", None)
