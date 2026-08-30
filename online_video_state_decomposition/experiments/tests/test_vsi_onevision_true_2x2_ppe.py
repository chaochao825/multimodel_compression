from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_true_2x2_ppe import (  # noqa: E402
    build_frequency_position_ids,
    center_ranked_member_offsets,
    classify_ppe,
    rotary_cos_sin_from_frequency_positions,
)


def test_center_ranked_offsets_use_stable_distance_order() -> None:
    features = torch.tensor([[[0.0], [3.0], [2.0], [1.0]]])
    offsets = torch.tensor([[0, 1, 2, 3]])

    ranked = center_ranked_member_offsets(features, offsets)

    assert ranked.tolist() == [[2, 3, 0, 1]]


def test_ppe_positions_split_rotary_frequency_pairs() -> None:
    base = torch.tensor([[5, 6, 7, 8]])
    offsets = torch.tensor([[0, 1, 14, 15]])

    positions = build_frequency_position_ids(
        base_position_ids=base,
        ordered_group_offsets=offsets,
        visual_start=1,
        rotary_pair_count=8,
    )

    assert positions.shape == (1, 4, 8)
    assert positions[0, 0].tolist() == [5] * 8
    assert positions[0, 1].tolist() == [1, 1, 2, 2, 15, 15, 16, 16]
    assert positions[0, 2].tolist() == [7] * 8


def test_frequency_positions_reproduce_scalar_rope() -> None:
    x = torch.zeros((1, 3, 8), dtype=torch.float32)
    inv_freq = torch.tensor([1.0, 0.1, 0.01, 0.001])
    scalar = torch.tensor([[2, 5, 7]])
    expanded = scalar.unsqueeze(-1).expand(-1, -1, inv_freq.numel())

    cos, sin = rotary_cos_sin_from_frequency_positions(
        x=x,
        inv_freq=inv_freq,
        attention_scaling=1.0,
        frequency_position_ids=expanded,
    )
    expected_freqs = (inv_freq[None, :, None] @ scalar[:, None, :].float()).transpose(
        1, 2
    )
    expected = torch.cat((expected_freqs, expected_freqs), dim=-1)

    assert torch.equal(cos, expected.cos())
    assert torch.equal(sin, expected.sin())


def test_ppe_classifier_requires_joint_quality_guards() -> None:
    strict = {
        "mismatch_reduction": 2,
        "harmful_delta": 0,
        "mean_kl_ratio": 0.8,
        "p95_kl_ratio": 0.8,
    }
    assert classify_ppe(strict) == "PPE_STRICT_HEADROOM"

    decision = {
        "mismatch_reduction": 1,
        "harmful_delta": 0,
        "mean_kl_ratio": 0.95,
        "p95_kl_ratio": 1.0,
    }
    assert classify_ppe(decision) == "PPE_DECISION_HEADROOM"

    adverse = dict(decision, harmful_delta=1)
    assert classify_ppe(adverse) == "NO_PPE_HEADROOM"
