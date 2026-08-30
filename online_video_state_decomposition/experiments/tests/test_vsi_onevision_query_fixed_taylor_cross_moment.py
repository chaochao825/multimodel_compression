from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_query_fixed_taylor_cross_moment import (  # noqa: E402
    classify_outcome,
    taylor_polynomial,
)


def test_taylor_polynomial_matches_registered_orders() -> None:
    values = torch.tensor([-0.5, 0.0, 0.5])

    assert torch.equal(taylor_polynomial(values, 0), torch.ones_like(values))
    assert torch.allclose(taylor_polynomial(values, 1), 1 + values)
    assert torch.allclose(
        taylor_polynomial(values, 2), 1 + values + values.square() / 2
    )
    assert torch.allclose(
        taylor_polynomial(values, 3),
        1 + values + values.square() / 2 + values.pow(3) / 6,
    )


def _summaries(passing_order: int | None) -> dict[str, dict[str, object]]:
    summaries = {}
    for order in range(4):
        passing = passing_order is not None and order >= passing_order
        summaries[f"taylor_order{order}"] = {
            "196": {
                "cell_count": 72,
                "valid_cell_count": 72,
                "invalid_cell_count": 0,
                "visual_mean": 0.009 if passing else 0.03,
                "visual_p95": 0.019 if passing else 0.05,
                "visual_worst": 0.049 if passing else 0.08,
                "full_mean": 0.004 if passing else 0.006,
                "full_p95": 0.009 if passing else 0.012,
            }
        }
    return summaries


def test_classifier_uses_lowest_passing_order() -> None:
    assert classify_outcome(_summaries(1))[0] == "TAYLOR_CROSS_MOMENT_ORDER1_PASS"
    assert classify_outcome(_summaries(2))[0] == "TAYLOR_CROSS_MOMENT_ORDER2_PASS"
    assert classify_outcome(_summaries(3))[0] == "TAYLOR_CROSS_MOMENT_ORDER3_ONLY"
    assert classify_outcome(_summaries(None))[0] == "NO_TAYLOR_CROSS_MOMENT_CAPACITY"


def test_invalid_positive_measure_state_cannot_pass() -> None:
    summaries = _summaries(1)
    summaries["taylor_order1"]["196"]["valid_cell_count"] = 71
    summaries["taylor_order1"]["196"]["invalid_cell_count"] = 1

    assert classify_outcome(summaries)[0] == "TAYLOR_CROSS_MOMENT_ORDER2_PASS"
