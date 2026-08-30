from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from train_vsi_onevision_additive_nz_feature_state import (  # noqa: E402
    PositiveFeatureState,
    classify,
)

from capture_vsi_onevision_additive_nz_dataset import (  # noqa: E402
    additive_nz_payload,
)
from probe_vsi_onevision_query_fixed_measure_remainder import (  # noqa: E402
    AttentionCapture,
)


def test_visual_state_is_additive_across_token_partitions() -> None:
    torch.manual_seed(3)
    model = PositiveFeatureState(head_count=2, head_dim=4, feature_width=3)
    key = torch.randn(1, 2, 6, 4)
    value = torch.randn(1, 2, 6, 4)

    full_state, full_z = model.visual_state(key, value)
    left_state, left_z = model.visual_state(key[:, :, :2], value[:, :, :2])
    right_state, right_z = model.visual_state(key[:, :, 2:], value[:, :, 2:])

    assert torch.allclose(full_state, left_state + right_state, atol=1e-6)
    assert torch.allclose(full_z, left_z + right_z, atol=1e-6)


def test_constant_values_are_exact_for_visual_output() -> None:
    torch.manual_seed(5)
    model = PositiveFeatureState(head_count=2, head_dim=4, feature_width=3)
    query = torch.randn(1, 2, 4)
    key = torch.randn(1, 2, 7, 4)
    constant = torch.tensor([1.0, -2.0, 0.5, 3.0])
    value = constant.reshape(1, 1, 1, 4).expand(1, 2, 7, 4)
    visual, _, _ = model(
        query,
        key,
        value,
        torch.zeros(1, 2),
        torch.zeros(1, 2, 4),
    )

    assert torch.allclose(visual, value[:, :, 0], atol=1e-5)


def test_capture_payload_reconstructs_visual_and_full_attention() -> None:
    torch.manual_seed(7)

    class Attention:
        scaling = 0.5

    head_count, sequence_length, head_dim = 2, 7, 4
    query = torch.randn(head_count, sequence_length, head_dim)
    key = torch.randn(head_count, sequence_length, head_dim)
    value = torch.randn(head_count, sequence_length, head_dim)
    scores = torch.matmul(query, key.transpose(-2, -1)) * Attention.scaling
    weights = torch.softmax(scores, dim=-1)
    head_output = torch.matmul(weights, value)
    unprojected = head_output.transpose(0, 1).reshape(sequence_length, -1)[-1]
    capture = AttentionCapture(
        module=Attention(),
        query=query,
        key=key,
        value=value,
        attention_mask=None,
        unprojected_output=unprojected,
    )

    payload, replay_error = additive_nz_payload(
        capture,
        visual_start=2,
        visual_token_count=3,
    )
    visual_n = payload["exact_visual_output"] * payload["exact_visual_z"].unsqueeze(-1)
    reconstructed = (visual_n + payload["nonvisual_n"]) / (
        payload["exact_visual_z"] + payload["nonvisual_z"]
    ).unsqueeze(-1)

    assert replay_error < 1e-6
    assert torch.allclose(reconstructed, payload["exact_full_output"], atol=1e-6)


def _summary(*, mean: float, p95: float, worst: float) -> dict[str, float | int]:
    return {
        "cell_count": 72,
        "visual_mean": mean,
        "visual_p95": p95,
        "visual_worst": worst,
        "full_mean": mean / 2,
        "full_p95": p95 / 2,
    }


def test_classifier_separates_strict_signal_and_null() -> None:
    baseline = _summary(mean=0.1, p95=0.2, worst=0.3)
    assert (
        classify(
            baseline=baseline,
            learned=_summary(mean=0.005, p95=0.01, worst=0.03),
            state_ratio=40.0,
        )
        == "ADDITIVE_NZ_DEV_GO"
    )
    assert (
        classify(
            baseline=baseline,
            learned=_summary(mean=0.04, p95=0.08, worst=0.2),
            state_ratio=40.0,
        )
        == "ADDITIVE_NZ_CAPACITY_SIGNAL"
    )
    assert (
        classify(
            baseline=baseline,
            learned=_summary(mean=0.08, p95=0.2, worst=0.3),
            state_ratio=40.0,
        )
        == "NO_ADDITIVE_NZ_FEATURE_STATE"
    )
