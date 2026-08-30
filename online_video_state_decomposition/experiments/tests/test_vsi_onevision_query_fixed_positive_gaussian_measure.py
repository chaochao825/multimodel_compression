from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_query_fixed_positive_gaussian_measure import (  # noqa: E402
    GaussianComponents,
    active_read_ratio,
    classify_outcome,
    evaluate_gaussian_support,
    hierarchical_group_offsets,
    positive_gaussian_measure,
)


def test_hierarchical_groups_partition_registered_video_grid() -> None:
    spatial = hierarchical_group_offsets(
        frame_count=8,
        token_count=196,
        topology="spatial_7x7",
        device=torch.device("cpu"),
    )
    temporal = hierarchical_group_offsets(
        frame_count=8,
        token_count=196,
        topology="temporal2_spatial_7x7",
        device=torch.device("cpu"),
    )

    assert spatial.shape == (32, 49)
    assert temporal.shape == (16, 98)
    assert torch.unique(spatial).numel() == 1568
    assert torch.unique(temporal).numel() == 1568


def test_rank_zero_is_exact_for_constant_group() -> None:
    exact_z = torch.tensor([[4.0]])
    value = torch.tensor([[[2.0, -1.0]]])
    exact_n = exact_z.unsqueeze(-1) * value
    components = GaussianComponents(
        exact_visual_output=value[:, 0],
        exact_full_output=value[:, 0],
        exact_group_z=exact_z,
        exact_group_n=exact_n,
        nonvisual_z=torch.zeros(1),
        nonvisual_n=torch.zeros(1, 2),
        mean_value=value,
        score_center=torch.zeros(1, 1),
        maximum=torch.zeros(1, 1),
        eigenvalues=torch.zeros(1, 1, 16),
        cross_value_key=torch.zeros(1, 1, 2, 16),
        query_coordinates=torch.zeros(1, 1, 16),
        group_size=4,
        replay_error=0.0,
        mean_key=torch.zeros(1, 1, 16),
        key_min=torch.zeros(1, 1, 16),
        key_max=torch.zeros(1, 1, 16),
        query_scaled=torch.zeros(1, 16),
        visual_value_norm_max=torch.linalg.vector_norm(value[:, 0], dim=-1),
        member_key=torch.zeros(1, 1, 4, 16),
        member_value=value.unsqueeze(2).expand(-1, -1, 4, -1),
    )

    state = positive_gaussian_measure(components, 0)

    compact = evaluate_gaussian_support(
        state,
        torch.empty((1, 0), dtype=torch.long),
        nonvisual_z=components.nonvisual_z,
        nonvisual_n=components.nonvisual_n,
    )
    exact = evaluate_gaussian_support(
        state,
        torch.tensor([[0]]),
        nonvisual_z=components.nonvisual_z,
        nonvisual_n=components.nonvisual_n,
    )

    assert compact["visual_relative_l2"] == pytest.approx(0.0)
    assert compact["full_relative_l2"] == pytest.approx(0.0)
    assert exact["visual_relative_l2"] == pytest.approx(0.0)
    assert exact["full_relative_l2"] == pytest.approx(0.0)
    assert bool((state.coarse_group_z >= 0).all())


def test_log_domain_stabilizer_handles_large_gaussian_variance() -> None:
    components = GaussianComponents(
        exact_visual_output=torch.ones(1, 2),
        exact_full_output=torch.ones(1, 2),
        exact_group_z=torch.ones(1, 1),
        exact_group_n=torch.ones(1, 1, 2),
        nonvisual_z=torch.zeros(1),
        nonvisual_n=torch.zeros(1, 2),
        mean_value=torch.ones(1, 1, 2),
        score_center=torch.zeros(1, 1),
        maximum=torch.zeros(1, 1),
        eigenvalues=torch.full((1, 1, 16), 100.0),
        cross_value_key=torch.zeros(1, 1, 2, 16),
        query_coordinates=torch.ones(1, 1, 16),
        group_size=49,
        replay_error=0.0,
        mean_key=torch.zeros(1, 1, 16),
        key_min=torch.zeros(1, 1, 16),
        key_max=torch.zeros(1, 1, 16),
        query_scaled=torch.ones(1, 16),
        visual_value_norm_max=torch.ones(1),
        member_key=torch.zeros(1, 1, 49, 16),
        member_value=torch.ones(1, 1, 49, 2),
    )

    state = positive_gaussian_measure(components, 16)

    assert torch.isfinite(state.coarse_group_z).all()
    assert torch.isfinite(state.coarse_group_n).all()
    assert state.coarse_group_z.item() == pytest.approx(1.0)
    assert state.measure_scale.item() < 1.0

    exact = evaluate_gaussian_support(
        state,
        torch.tensor([[0]]),
        nonvisual_z=components.nonvisual_z,
        nonvisual_n=components.nonvisual_n,
    )
    assert exact["visual_relative_l2"] == pytest.approx(0.0)
    assert exact["full_relative_l2"] == pytest.approx(0.0)


def test_active_read_ratio_charges_moments_and_exact_leaves() -> None:
    dense, moment, exact, ratio = active_read_ratio(
        group_count=32,
        group_size=49,
        head_dim=128,
        rank=4,
        exact_group_count=8,
    )

    assert dense == 32 * 49 * 256
    assert moment == 32 * (256 + 4 * 257)
    assert exact == 8 * 49 * 256
    assert ratio > 2.0


def _summary(selector: str, *, passing: bool) -> dict[str, object]:
    return {
        "topology": "spatial_7x7",
        "rank": 4,
        "selector": selector,
        "exact_fraction": 0.25,
        "exact_group_count": 8,
        "cell_count": 72,
        "active_read_ratio": 2.5,
        "visual_mean": 0.004 if passing else 0.03,
        "visual_p95": 0.009 if passing else 0.05,
        "visual_worst": 0.019 if passing else 0.08,
        "visual_worst_head": 0.03,
        "full_mean": 0.002,
        "full_p95": 0.004,
        "full_worst": 0.006,
    }


def test_classifier_separates_compact_and_oracle_capacity() -> None:
    assert classify_outcome([_summary("compact_mass", passing=True)])[0] == (
        "POSITIVE_GAUSSIAN_COMPACT_PATH"
    )
    assert classify_outcome([_summary("oracle_local", passing=True)])[0] == (
        "POSITIVE_GAUSSIAN_CAPACITY_ONLY"
    )
    assert classify_outcome([_summary("oracle_local", passing=False)])[0] == (
        "NO_POSITIVE_GAUSSIAN_MEASURE_PATH"
    )
