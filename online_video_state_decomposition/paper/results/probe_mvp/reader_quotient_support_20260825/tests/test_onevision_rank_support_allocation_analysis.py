from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_onevision_rank_support_allocation import classify_candidate  # noqa: E402


def test_candidate_requires_every_frozen_guard() -> None:
    valid = {
        "reduction": 0.30,
        "p95_ratio": 0.8,
        "positive_tasks": 4,
        "top1_delta": 0.0,
        "absolute_kl_ratio": 0.9,
        "state_bytes": 2_867_328,
    }
    assert classify_candidate(**valid) == "GO"
    assert classify_candidate(**{**valid, "positive_tasks": 3}) == "BOUNDARY"
    assert classify_candidate(**{**valid, "top1_delta": -0.05}) == "ADVERSE"
    assert classify_candidate(**{**valid, "absolute_kl_ratio": 1.01}) == "BOUNDARY"
