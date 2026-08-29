from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from reader_quotient_cmrq_stage_b import (  # noqa: E402
    boundary_mixed_basis,
    DomainMoments,
    equally_weighted_moments,
    fixed_rank_hybrid,
    projected_top_atoms,
    summarize_exact_rows,
    summarize_progressive_fallback,
    trace_capture,
)
from probe_vsi_onevision_cmrq_stage_b import parse_index_ranges  # noqa: E402


def test_equally_weighted_moments_include_between_domain_shift() -> None:
    left = DomainMoments(
        mean=torch.tensor([0.0, 0.0]),
        covariance=torch.diag(torch.tensor([2.0, 1.0])),
    )
    right = DomainMoments(
        mean=torch.tensor([2.0, 0.0]),
        covariance=torch.diag(torch.tensor([2.0, 1.0])),
    )
    pooled = equally_weighted_moments([left, right])
    assert torch.allclose(pooled.mean, torch.tensor([1.0, 0.0]))
    assert torch.allclose(pooled.covariance, torch.diag(torch.tensor([3.0, 1.0])))


def test_risk_atom_improves_risk_capture_at_fixed_rank() -> None:
    covariance = torch.diag(torch.tensor([9.0, 8.0, 7.0, 1.0]))
    risk = torch.diag(torch.tensor([1.0, 1.0, 1.0, 20.0]))
    feature_basis = torch.eye(4)
    bulk = feature_basis[:, :2]
    atoms = projected_top_atoms(risk, bulk, atom_count=1)
    hybrid = fixed_rank_hybrid(
        feature_basis,
        atoms,
        rank=3,
        atom_count=1,
    )
    feature_only = feature_basis[:, :3]
    assert trace_capture(risk, hybrid) > trace_capture(risk, feature_only)
    assert trace_capture(covariance, hybrid) < trace_capture(covariance, feature_only)

    mixed = boundary_mixed_basis(
        feature_basis,
        atoms,
        covariance,
        risk,
        rank=3,
        atom_count=1,
        risk_weight=0.2,
    )
    assert mixed.shape == (4, 3)
    assert trace_capture(risk, mixed) >= trace_capture(risk, feature_only)


def test_exact_summary_preserves_one_sided_events() -> None:
    rows = [
        {
            "feature_relative_l2": 0.1,
            "candidate_kl": 0.01,
            "maximum_normalized_adverse_shift": 0.4,
            "minimum_margin": 0.2,
            "prediction_match": 1,
            "harmful": 0,
            "beneficial": 0,
        },
        {
            "feature_relative_l2": 0.2,
            "candidate_kl": 0.03,
            "maximum_normalized_adverse_shift": 1.2,
            "minimum_margin": 0.0,
            "prediction_match": 0,
            "harmful": 1,
            "beneficial": 0,
        },
    ]
    summary = summarize_exact_rows(rows, margin_floor=0.05)
    assert summary["agreement"] == pytest.approx(0.5)
    assert summary["mismatch_count"] == 1
    assert summary["harmful_count"] == 1
    assert summary["near_tie_count"] == 1
    assert summary["agreement_outside_near_tie"] == pytest.approx(1.0)
    assert summary["normalized_adverse_max"] == pytest.approx(1.2)


def test_risk_fit_ranges_support_disjoint_cross_fit_blocks() -> None:
    assert parse_index_ranges("0:2,4:7") == (0, 1, 4, 5, 6)
    with pytest.raises(ValueError, match="overlap"):
        parse_index_ranges("0:3,2:4")


def test_progressive_fallback_uses_compressed_margin_and_separates_cost_models() -> None:
    rows = [
        {
            "approximate_top1_margin": 0.0,
            "candidate_kl": 0.4,
            "feature_relative_l2": 0.2,
            "prediction_match": 0,
            "harmful": 1,
        },
        {
            "approximate_top1_margin": 0.5,
            "candidate_kl": 0.2,
            "feature_relative_l2": 0.1,
            "prediction_match": 1,
            "harmful": 0,
        },
    ]
    summary = summarize_progressive_fallback(
        rows,
        margin_threshold=0.0,
        compressed_state_bytes=2,
        dense_state_bytes=10,
    )
    assert summary["fallback_count"] == 1
    assert summary["remaining_mismatch_count"] == 0
    assert summary["remaining_harmful_count"] == 0
    assert summary["effective_candidate_kl_mean"] == pytest.approx(0.1)
    assert summary["effective_feature_l2_mean"] == pytest.approx(0.05)
    assert summary["conservative_transfer_ratio"] == pytest.approx(10 / 7)
    assert summary["ideal_preroute_transfer_ratio"] == pytest.approx(10 / 6)
