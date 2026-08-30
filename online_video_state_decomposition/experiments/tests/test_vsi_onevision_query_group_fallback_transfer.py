from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_query_group_fallback_transfer import (  # noqa: E402
    apply_fallback,
    calibrate_mismatch_threshold,
)


def _row(margin: float, match: int, baseline: int, approximate: int) -> dict[str, object]:
    return {
        "compressed_top1_margin": margin,
        "prediction_match": match,
        "baseline_correct": baseline,
        "approximate_correct": approximate,
        "candidate_kl": 0.2,
    }


def test_threshold_covers_fit_mismatches_only_by_observable_margin() -> None:
    rows = [
        _row(0.2, 0, 1, 0),
        _row(0.8, 1, 1, 1),
        _row(0.1, 0, 0, 1),
    ]
    assert calibrate_mismatch_threshold(rows) == 0.2


def test_fallback_replaces_only_low_margin_outputs() -> None:
    rows = [
        _row(0.2, 0, 1, 0),
        _row(0.8, 1, 1, 1),
    ]
    delivered, summary = apply_fallback(
        rows,
        threshold=0.2,
        hybrid_token_count=40,
        full_token_count=100,
    )
    assert [row["fallback"] for row in delivered] == [1, 0]
    assert summary["delivered_agreement"] == 1.0
    assert summary["remaining_mismatch_count"] == 0
    assert summary["remaining_harmful_count"] == 0
    assert summary["effective_token_retention"] == 0.7
