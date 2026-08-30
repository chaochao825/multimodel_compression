from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_query_fixed_local_secant_measure import (  # noqa: E402
    build_secant_state,
    classify_outcome,
    group_offsets,
    reconstructed_members,
    state_cost,
    symmetric_int8_dequantize,
)


def test_registered_local_groups_partition_video_grid() -> None:
    for topology in ("flat_contiguous_4", "spatial_2x2"):
        offsets = group_offsets(
            frame_count=8,
            token_count=196,
            topology=topology,
            device=torch.device("cpu"),
        )
        assert offsets.shape == (392, 4)
        assert torch.unique(offsets).numel() == 1568


def test_key_secant_recovers_shared_rank_one_key_value_group() -> None:
    coordinate = torch.tensor([-1.5, -0.5, 0.5, 1.5]).reshape(1, 1, 4, 1)
    mean_key = torch.tensor([1.0, -2.0]).reshape(1, 1, 1, 2)
    mean_value = torch.tensor([0.5, 3.0]).reshape(1, 1, 1, 2)
    key_direction = torch.tensor([2.0, 1.0]).reshape(1, 1, 1, 2)
    value_direction = torch.tensor([-1.0, 4.0]).reshape(1, 1, 1, 2)
    key = mean_key + coordinate * key_direction
    value = mean_value + coordinate * value_direction

    state = build_secant_state(key, value, method="key_secant")
    reconstructed_key, reconstructed_value = reconstructed_members(state)

    assert torch.allclose(reconstructed_key, key, atol=1e-5)
    assert torch.allclose(reconstructed_value, value, atol=1e-5)


def test_independent_secant_recovers_distinct_rank_one_coordinates() -> None:
    key_coordinate = torch.tensor([-1.5, -0.5, 0.5, 1.5]).reshape(1, 1, 4, 1)
    value_coordinate = torch.tensor([-1.0, 1.0, 1.0, -1.0]).reshape(1, 1, 4, 1)
    key = key_coordinate * torch.tensor([2.0, 1.0]).reshape(1, 1, 1, 2)
    value = value_coordinate * torch.tensor([-1.0, 4.0]).reshape(1, 1, 1, 2)

    state = build_secant_state(key, value, method="independent_secant")
    reconstructed_key, reconstructed_value = reconstructed_members(state)

    assert torch.allclose(reconstructed_key, key, atol=1e-5)
    assert torch.allclose(reconstructed_value, value, atol=1e-5)


def test_symmetric_int8_preserves_zero_and_bounds_error() -> None:
    vectors = torch.tensor([[0.0, -1.0, 0.25, 0.75]])
    dequantized = symmetric_int8_dequantize(vectors)

    assert dequantized[0, 0] == 0.0
    assert torch.max(torch.abs(dequantized - vectors)) <= 1.0 / 127.0


def test_secant_cost_meets_registered_proxy() -> None:
    cost = state_cost(
        method="independent_secant",
        quantization="symmetric_int8",
        group_count=392,
        group_size=4,
        head_dim=128,
    )

    assert cost["attention_arithmetic_ratio"] == pytest.approx(2.0)
    assert cost["state_byte_ratio"] >= 3.5


def _summary(quantization: str, *, passing: bool) -> dict[str, object]:
    return {
        "topology": "spatial_2x2",
        "method": "independent_secant",
        "quantization": quantization,
        "cell_count": 72,
        "visual_mean": 0.004 if passing else 0.03,
        "visual_p95": 0.008 if passing else 0.05,
        "visual_worst": 0.015 if passing else 0.08,
        "full_mean": 0.002,
        "full_p95": 0.004,
        "state_byte_ratio": 3.8,
        "attention_arithmetic_ratio": 2.0,
    }


def test_classifier_separates_deployable_capacity_and_null() -> None:
    assert classify_outcome([_summary("symmetric_int8", passing=True)])[0] == (
        "LOCAL_SECANT_DEPLOYABLE_CAPACITY"
    )
    assert classify_outcome([_summary("fp32", passing=True)])[0] == (
        "LOCAL_SECANT_CAPACITY_ONLY"
    )
    assert classify_outcome([_summary("fp32", passing=False)])[0] == (
        "NO_LOCAL_SECANT_PATH"
    )
