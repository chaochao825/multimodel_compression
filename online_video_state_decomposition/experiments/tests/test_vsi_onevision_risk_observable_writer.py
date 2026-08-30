from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_risk_observable_writer import (  # noqa: E402
    JointWriterController,
    WriterDotScorer,
    gate_conditions,
    hadamard_modes,
    writer_position_features,
)


def test_hadamard_modes_preserve_all_four_residual_modes() -> None:
    torch.manual_seed(5)
    exact = torch.randn(3, 4, 7)
    approximate = torch.randn(3, 7)
    normalized, rms = hadamard_modes(exact, approximate)
    raw_modes = normalized * rms[:, :, None]
    transform = torch.tensor(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        )
    ).div(2.0)
    recovered_delta = torch.einsum("mt,gmh->gth", transform, raw_modes[:, 1:])
    assert normalized.shape == (3, 5, 7)
    assert rms.shape == (3, 5)
    assert torch.allclose(raw_modes[:, 0], approximate, atol=1e-5)
    assert torch.allclose(
        recovered_delta,
        exact - approximate[:, None, :],
        atol=1e-5,
    )


def test_writer_models_and_position_contracts() -> None:
    modes = torch.randn(2, 6, 5, 7)
    queries = torch.randn(2, 7)
    scalars = torch.randn(2, 6, 10)
    dot = WriterDotScorer(hidden=7)(modes, queries, scalars)
    joint = JointWriterController(hidden=7)(modes, queries, scalars)
    positions = writer_position_features(
        6,
        groups_per_frame=3,
        device=torch.device("cpu"),
    )
    assert dot.shape == (2, 6)
    assert joint.shape == (2, 6)
    assert positions.shape == (6, 4)
    assert torch.isfinite(dot).all()
    assert torch.isfinite(joint).all()


def test_gate_distinguishes_joint_writer_and_no_go() -> None:
    selector = {
        "fixed_controller": {"mean_topk_recall": 0.40},
        "writer_dot": {
            "mean_topk_recall": 0.46,
            "mean_risk_mass_capture": 0.52,
        },
        "joint_writer_controller": {
            "mean_topk_recall": 0.50,
            "mean_risk_mass_capture": 0.55,
        },
    }
    reader = {
        "writer_dot": {
            "agreement": 22 / 24,
            "harmful_count": 1,
            "candidate_kl_mean": 0.04,
            "candidate_accuracy": 0.5,
            "baseline_accuracy": 0.5,
        },
        "joint_writer_controller": {
            "agreement": 23 / 24,
            "harmful_count": 0,
            "candidate_kl_mean": 0.03,
            "candidate_accuracy": 0.5,
            "baseline_accuracy": 0.5,
        },
    }
    decision, conditions = gate_conditions(
        selector=selector,
        reader=reader,
        scorer_macs=1_500_000,
    )
    assert decision == "JOINT_GO"
    assert all(conditions.values())

    selector["joint_writer_controller"]["mean_topk_recall"] = 0.42
    selector["writer_dot"]["mean_topk_recall"] = 0.41
    decision, _ = gate_conditions(
        selector=selector,
        reader=reader,
        scorer_macs=1_500_000,
    )
    assert decision == "NO_GO"
