from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_query_fixed_measure_remainder import (  # noqa: E402
    classify_outcome,
    exact_greedy_order,
    greedy_analytic_order,
)


def test_greedy_analytic_order_removes_every_group_once() -> None:
    priority = torch.tensor(
        [
            [3.0, 2.0, 1.0, 0.5],
            [0.1, 4.0, 1.0, 0.5],
        ]
    )

    order = greedy_analytic_order(priority)

    assert sorted(order.tolist()) == [0, 1, 2, 3]
    remaining = priority.sum(dim=1)
    certificates = [torch.linalg.vector_norm(remaining).item()]
    for group_index in order.tolist():
        remaining = remaining - priority[:, group_index]
        certificates.append(torch.linalg.vector_norm(remaining).item())
    assert all(
        current <= previous + 1e-7
        for previous, current in zip(certificates, certificates[1:])
    )


def _summary(value: float) -> dict[str, dict[str, dict[str, float]]]:
    methods = (
        "analytic_remainder",
        "attention_mass",
        "exact_local_score",
        "exact_greedy_oracle",
        "fixed_random",
    )
    budgets = (0, 49, 98, 147, 196, 392)
    return {
        method: {
            str(budget): {
                "visual_relative_l2_mean": value,
                "visual_relative_l2_p95": value,
                "visual_relative_l2_worst": value,
                "full_relative_l2_mean": value / 2,
                "full_relative_l2_p95": value / 2,
            }
            for budget in budgets
        }
        for method in methods
    }


def test_classifier_separates_certified_capacity_and_null() -> None:
    certified = _summary(0.004)
    for budget in (98, 147, 196):
        certified["attention_mass"][str(budget)]["visual_relative_l2_mean"] = 0.006
    decision, _ = classify_outcome(
        certified,
        certificate_valid_head_fraction=1.0,
        certificate_violation_count=0,
        certificate_increase_count=0,
    )
    assert decision == "QUERY_FIXED_CERTIFIED_HEADROOM"

    loose = _summary(0.004)
    loose["analytic_remainder"]["196"]["visual_relative_l2_p95"] = 0.1
    decision, _ = classify_outcome(
        loose,
        certificate_valid_head_fraction=0.5,
        certificate_violation_count=0,
        certificate_increase_count=0,
    )
    assert decision == "QUERY_FIXED_CAPACITY_BOUND_LOOSE"

    null = _summary(0.1)
    decision, _ = classify_outcome(
        null,
        certificate_valid_head_fraction=1.0,
        certificate_violation_count=0,
        certificate_increase_count=0,
    )
    assert decision == "NO_REGISTERED_QUERY_FIXED_MEASURE_PATH"


class _GreedyState:
    def __init__(self) -> None:
        self.exact_group_z = torch.ones((1, 3))
        self.coarse_group_z = torch.ones((1, 3))
        self.exact_group_n = torch.tensor([[[1.0], [2.0], [3.0]]])
        self.coarse_group_n = torch.zeros((1, 3, 1))
        self.exact_visual_output = torch.tensor([[2.0]])


def test_exact_greedy_order_recomputes_current_output() -> None:
    order = exact_greedy_order(_GreedyState())

    assert sorted(order.tolist()) == [0, 1, 2]
