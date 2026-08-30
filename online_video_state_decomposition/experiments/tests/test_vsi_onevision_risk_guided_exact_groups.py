from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_risk_guided_exact_groups import (  # noqa: E402
    contiguous_group_means,
    hybrid_group_tokens,
    normalized_adverse_group_risk,
)


def test_contiguous_groups_preserve_frame_order() -> None:
    features = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    groups, means = contiguous_group_means(features, group_size=2)
    assert groups.shape == (4, 2, 2)
    assert torch.equal(groups.reshape(2, 4, 2), features)
    assert torch.allclose(means[0], torch.tensor([1.0, 2.0]))


def test_hybrid_groups_expand_only_selected_groups() -> None:
    exact = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
    means = torch.tensor([[1.5], [3.5], [5.5]])
    hybrid = hybrid_group_tokens(exact, means, torch.tensor([1]))
    assert hybrid.flatten().tolist() == [1.5, 3.0, 4.0, 5.5]


def test_adverse_group_risk_is_margin_normalized() -> None:
    exact = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    means = torch.zeros((2, 1))
    gradients = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
    risk = normalized_adverse_group_risk(
        gradients,
        exact,
        means,
        torch.tensor([0.5]),
        margin_floor=0.05,
    )
    assert torch.allclose(risk, torch.tensor([6.0, 0.0]))
