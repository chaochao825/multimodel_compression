from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_query_fixed_headwise_support_ceiling import (  # noqa: E402
    classify_outcome,
    headwise_exact_greedy_order,
    validate_shared_control,
)


def _state() -> SimpleNamespace:
    exact_group_z = torch.ones((2, 3))
    coarse_group_z = torch.ones((2, 3))
    exact_group_n = torch.tensor(
        [
            [[1.0], [2.0], [4.0]],
            [[4.0], [2.0], [1.0]],
        ]
    )
    coarse_group_n = torch.zeros((2, 3, 1))
    return SimpleNamespace(
        exact_group_z=exact_group_z,
        coarse_group_z=coarse_group_z,
        exact_group_n=exact_group_n,
        coarse_group_n=coarse_group_n,
        exact_visual_output=exact_group_n.sum(dim=1) / 3,
    )


def test_headwise_exact_greedy_returns_per_head_permutations() -> None:
    order = headwise_exact_greedy_order(_state())

    assert order.shape == (2, 3)
    assert torch.equal(torch.sort(order, dim=1).values, torch.tensor([[0, 1, 2]] * 2))
    assert not torch.equal(order[0], order[1])


def test_classifier_distinguishes_pass_partial_and_null() -> None:
    summaries = {"shared_exact_local": {"196": {"visual_mean": 0.04}}}
    passing = {
        "196": {
            "visual_mean": 0.009,
            "visual_p95": 0.019,
            "visual_worst": 0.049,
            "full_mean": 0.004,
            "full_p95": 0.009,
        }
    }
    assert classify_outcome(summaries, passing)[0] == "HEADWISE_SUPPORT_CAPACITY_PASS"

    partial = {
        "196": {
            "visual_mean": 0.02,
            "visual_p95": 0.03,
            "visual_worst": 0.06,
            "full_mean": 0.004,
            "full_p95": 0.009,
        }
    }
    assert classify_outcome(summaries, partial)[0] == "HEADWISE_SUPPORT_PARTIAL"

    null = {
        "196": {
            "visual_mean": 0.035,
            "visual_p95": 0.05,
            "visual_worst": 0.08,
            "full_mean": 0.006,
            "full_p95": 0.012,
        }
    }
    assert classify_outcome(summaries, null)[0] == "NO_HEADWISE_SUPPORT_CAPACITY"


def test_shared_control_validation_accepts_subset(tmp_path: Path) -> None:
    previous = tmp_path / "previous.csv"
    previous.write_text(
        "sample_id,layer_index,exact_group_count,method,visual_relative_l2,"
        "visual_worst_head_relative_l2,full_relative_l2\n"
        "sample-a,0,49,exact_local_score,0.1,0.2,0.03\n"
        "sample-b,0,49,exact_local_score,0.4,0.5,0.06\n",
        encoding="utf-8",
    )
    current = [
        {
            "sample_id": "sample-b",
            "layer_index": 0,
            "exact_group_count": 49,
            "method": "shared_exact_local",
            "visual_relative_l2": 0.4,
            "visual_worst_head_relative_l2": 0.5,
            "full_relative_l2": 0.06,
        }
    ]

    validate_shared_control(current, previous)
