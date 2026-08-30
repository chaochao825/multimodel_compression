from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_query_fixed_prototype_measure import (  # noqa: E402
    PrototypeState,
    classify_outcome,
    deterministic_kmeans,
    evaluate_prototype_state,
    select_reverse_greedy_under_budget,
    select_residual_greedy_under_budget,
    select_under_budget,
)


def test_deterministic_kmeans_separates_two_clusters() -> None:
    features = torch.tensor([[[-2.0, -2.0], [-1.9, -2.1], [2.0, 2.0], [2.1, 1.9]]])
    first = deterministic_kmeans(features, cluster_count=2, iterations=4)
    second = deterministic_kmeans(features, cluster_count=2, iterations=4)

    assert torch.equal(first, second)
    assert first[0, 0] == first[0, 1]
    assert first[0, 2] == first[0, 3]
    assert first[0, 0] != first[0, 2]


def test_budgeted_selection_never_exceeds_active_budget() -> None:
    priority = torch.tensor([[9.0, 8.0, 1.0]])
    counts = torch.tensor([[4.0, 3.0, 2.0]])
    selected, active = select_under_budget(priority, counts, active_token_budget=6)

    assert active.item() <= 6
    assert torch.equal(selected, torch.tensor([[False, True, True]]))


def test_all_exact_prototypes_reproduce_reference() -> None:
    exact_z = torch.tensor([[2.0, 1.0]])
    exact_n = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    exact_output = exact_n.sum(dim=1) / exact_z.sum(dim=1, keepdim=True)
    state = PrototypeState(
        exact_visual_output=exact_output,
        exact_full_output=exact_output,
        exact_cluster_z=exact_z,
        exact_cluster_n=exact_n,
        coarse_cluster_z=torch.ones_like(exact_z),
        coarse_cluster_n=torch.zeros_like(exact_n),
        prototype_mass_priority=torch.ones_like(exact_z),
        oracle_priority=torch.ones_like(exact_z),
        cluster_counts=torch.tensor([[2.0, 2.0]]),
        nonvisual_z=torch.zeros(1),
        nonvisual_n=torch.zeros(1, 2),
    )
    metrics = evaluate_prototype_state(
        state,
        torch.ones_like(exact_z, dtype=torch.bool),
        torch.tensor([4]),
    )

    assert metrics["visual_relative_l2"] == pytest.approx(0.0)
    assert metrics["full_relative_l2"] == pytest.approx(0.0)


def test_residual_greedy_selects_the_error_reducing_cluster() -> None:
    exact_z = torch.ones(1, 3)
    exact_n = torch.zeros(1, 3, 1)
    state = PrototypeState(
        exact_visual_output=torch.zeros(1, 1),
        exact_full_output=torch.zeros(1, 1),
        exact_cluster_z=exact_z,
        exact_cluster_n=exact_n,
        coarse_cluster_z=exact_z.clone(),
        coarse_cluster_n=torch.tensor([[[1.0], [-0.8], [0.2]]]),
        prototype_mass_priority=torch.ones_like(exact_z),
        oracle_priority=torch.tensor([[1.0, 0.8, 0.2]]),
        cluster_counts=torch.full((1, 3), 2.0),
        nonvisual_z=torch.zeros(1),
        nonvisual_n=torch.zeros(1, 1),
    )

    selected, active = select_residual_greedy_under_budget(state, active_token_budget=4)

    assert active.item() == 4
    assert torch.equal(selected, torch.tensor([[False, False, True]]))


def test_reverse_greedy_satisfies_budget_deterministically() -> None:
    exact_z = torch.ones(1, 3)
    exact_n = torch.zeros(1, 3, 1)
    state = PrototypeState(
        exact_visual_output=torch.zeros(1, 1),
        exact_full_output=torch.zeros(1, 1),
        exact_cluster_z=exact_z,
        exact_cluster_n=exact_n,
        coarse_cluster_z=exact_z.clone(),
        coarse_cluster_n=torch.tensor([[[1.0], [-0.8], [0.2]]]),
        prototype_mass_priority=torch.ones_like(exact_z),
        oracle_priority=torch.tensor([[1.0, 0.8, 0.2]]),
        cluster_counts=torch.full((1, 3), 2.0),
        nonvisual_z=torch.zeros(1),
        nonvisual_n=torch.zeros(1, 1),
    )

    first, first_active = select_reverse_greedy_under_budget(
        state, active_token_budget=4
    )
    second, second_active = select_reverse_greedy_under_budget(
        state, active_token_budget=4
    )

    assert first_active.item() <= 4
    assert torch.equal(first, second)
    assert torch.equal(first_active, second_active)


def test_classifier_requires_strict_oracle_capacity() -> None:
    row = {
        "cluster_family": "key_value",
        "prototype_count": 64,
        "selector": "oracle_local",
        "cell_count": 72,
        "active_token_count_mean": 390.0,
        "active_token_count_max": 392,
        "active_read_ratio": 4.0,
        "exact_token_fraction_mean": 0.2,
        "selected_visual_mass_mean": 0.8,
        "visual_mean": 0.004,
        "visual_p95": 0.009,
        "visual_worst": 0.019,
        "visual_worst_head": 0.03,
        "full_mean": 0.002,
        "full_p95": 0.004,
        "full_worst": 0.005,
    }
    assert classify_outcome([row])[0] == "PROTOTYPE_MIXTURE_CAPACITY_ONLY"

    row["visual_worst"] = 0.03
    assert classify_outcome([row])[0] == "NO_PROTOTYPE_MIXTURE_PATH"
