from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_tiny_group_risk_controller import (  # noqa: E402
    SKETCH_WIDTH,
    controller_group_features,
    fixed_sign_projection,
    selector_summary,
    teacher_topk_labels,
)


def test_fixed_sign_projection_is_deterministic_and_normalized() -> None:
    first = fixed_sign_projection((4, SKETCH_WIDTH), seed=7)
    second = fixed_sign_projection((4, SKETCH_WIDTH), seed=7)
    assert torch.equal(first, second)
    assert set(first.unique().tolist()) == {-0.5, 0.5}


def test_controller_feature_contract_has_39_scalars() -> None:
    torch.manual_seed(4)
    exact = torch.randn(6, 4, 5)
    means = torch.randn(6, 5)
    query = torch.nn.functional.normalize(torch.randn(5), dim=0)
    hidden_projection = fixed_sign_projection((5, SKETCH_WIDTH), seed=2)
    residual_projection = fixed_sign_projection((4, 5, SKETCH_WIDTH), seed=3)
    features, diagnostics = controller_group_features(
        exact_groups=exact,
        approximate_means=means,
        query=query,
        hidden_projection=hidden_projection,
        residual_projection=residual_projection,
        groups_per_frame=3,
    )
    assert features.shape == (6, 39)
    assert diagnostics["residual_energy"].shape == (6,)
    assert diagnostics["query_score"].shape == (6,)
    assert torch.isfinite(features).all()


def test_teacher_labels_and_selector_recall_use_exact_topk() -> None:
    risk = torch.tensor([0.1, 0.9, 0.2, 0.8])
    labels = teacher_topk_labels(risk, topk=2)
    assert labels.tolist() == [0.0, 1.0, 0.0, 1.0]
    state = {"teacher_labels": labels, "teacher_risk": risk}
    perfect = selector_summary([state], [risk], topk=2)
    reversed_scores = selector_summary([state], [-risk], topk=2)
    assert perfect["mean_topk_recall"] == 1.0
    assert perfect["mean_risk_mass_capture"] == pytest.approx(1.7 / 2.0)
    assert reversed_scores["mean_topk_recall"] == 0.0
