from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from probe_vsi_onevision_query_fixed_positive_gaussian_measure import (  # noqa: E402
    GaussianComponents,
)
from probe_vsi_onevision_query_fixed_progressive_exact_pages import (  # noqa: E402
    build_page_state,
    evaluate_exact_pages,
    read_costs,
)


def synthetic_components() -> GaussianComponents:
    exact_z = torch.tensor([[3.0, 1.0]])
    group_values = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    exact_n = exact_z.unsqueeze(-1) * group_values
    visual_n = exact_n.sum(dim=1)
    visual_z = exact_z.sum(dim=1)
    visual_output = visual_n / visual_z.unsqueeze(-1)
    return GaussianComponents(
        exact_visual_output=visual_output,
        exact_full_output=visual_output,
        exact_group_z=exact_z,
        exact_group_n=exact_n,
        nonvisual_z=torch.zeros(1),
        nonvisual_n=torch.zeros(1, 2),
        mean_value=group_values,
        score_center=torch.tensor([[0.0, -1.0]]),
        maximum=torch.zeros(1, 1),
        eigenvalues=torch.zeros(1, 2, 1),
        cross_value_key=torch.zeros(1, 2, 2, 1),
        query_coordinates=torch.zeros(1, 2, 1),
        group_size=2,
        replay_error=0.0,
        mean_key=torch.tensor([[[0.0, 0.0], [-1.0, 0.0]]]),
        key_min=torch.tensor([[[-0.2, -0.1], [-1.2, -0.1]]]),
        key_max=torch.tensor([[[0.5, 0.1], [-0.6, 0.1]]]),
        query_scaled=torch.tensor([[1.0, 0.0]]),
        visual_value_norm_max=torch.tensor([2.0]),
    )


def test_quest_box_is_an_upper_bound_and_full_exact_is_identity() -> None:
    components = synthetic_components()
    state = build_page_state(components)
    assert state.maximum_bound_violation <= 0

    metrics = evaluate_exact_pages(
        state,
        torch.tensor([[0, 1]]),
        nonvisual_z=components.nonvisual_z,
        nonvisual_n=components.nonvisual_n,
    )
    assert metrics["visual_relative_l2"] == pytest.approx(0.0)
    assert metrics["full_relative_l2"] == pytest.approx(0.0)
    assert metrics["certificate_coverage"] == pytest.approx(1.0)


def test_partial_exact_path_reports_covered_error() -> None:
    components = synthetic_components()
    state = build_page_state(components)
    metrics = evaluate_exact_pages(
        state,
        torch.tensor([[0]]),
        nonvisual_z=components.nonvisual_z,
        nonvisual_n=components.nonvisual_n,
    )
    assert metrics["visual_relative_l2"] > 0
    assert metrics["selected_visual_mass_mean"] == pytest.approx(0.75)
    assert metrics["certificate_coverage"] == pytest.approx(1.0)


def test_read_costs_charge_metadata_and_exact_leaves() -> None:
    dense, metadata, exact, active, leaf = read_costs(
        selector="quest_box_bound",
        group_count=32,
        group_size=49,
        head_dim=128,
        exact_group_count=8,
    )
    assert dense == 32 * 49 * 2 * 128
    assert metadata == 32 * 2 * 128
    assert exact == 8 * 49 * 2 * 128
    assert active == pytest.approx(3.69811320754717)
    assert leaf == pytest.approx(4.0)
