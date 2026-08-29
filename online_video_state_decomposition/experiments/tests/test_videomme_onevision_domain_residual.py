from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from analyze_videomme_onevision_domain_residual_selection import (  # noqa: E402
    classify_candidate,
)
from analyze_videomme_onevision_domain_residual_flips import (  # noqa: E402
    summarize_flip_sets,
)
from videomme_onevision_domain_residual import (  # noqa: E402
    fit_residual_swap_basis,
)


def test_residual_swap_recovers_missing_target_directions() -> None:
    generator = torch.Generator().manual_seed(17)
    source_basis = torch.eye(8)[:, :4]
    coefficients = torch.randn(512, 4, generator=generator)
    target_basis = torch.eye(8)[:, (0, 1, 4, 5)]
    centered = coefficients @ target_basis.transpose(0, 1)
    basis, metadata = fit_residual_swap_basis(
        centered=centered,
        source_basis=source_basis,
        swap_rank=2,
        seed=19,
        niter=4,
    )
    source_error = torch.linalg.vector_norm(
        centered - (centered @ source_basis) @ source_basis.transpose(0, 1)
    )
    swap_error = torch.linalg.vector_norm(
        centered - (centered @ basis) @ basis.transpose(0, 1)
    )
    assert basis.shape == (8, 4)
    assert float(metadata["orthogonality_error"]) < 1e-5
    assert float(swap_error) < 1e-4 * float(source_error)


def selection_metrics(**updates: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "candidate_kl_mean": 0.006,
        "candidate_kl_p95": 0.03,
        "feature_relative_l2_mean": 0.15,
        "prediction_mismatches": 8,
        "harmful_flips": 2,
        "candidate_correct": 100,
        "max_state_bytes": 2_860_032,
        "max_injection_abs": 0.0,
        "duration_kl": {"short": 0.006, "medium": 0.006, "long": 0.006},
    }
    metrics.update(updates)
    return metrics


def test_selection_gate_distinguishes_go_capacity_and_no_go() -> None:
    source = selection_metrics(
        candidate_kl_mean=0.01,
        candidate_kl_p95=0.05,
        feature_relative_l2_mean=0.20,
        prediction_mismatches=10,
    )
    assert classify_candidate(selection_metrics(), source) == "GO"
    assert (
        classify_candidate(selection_metrics(prediction_mismatches=9), source)
        == "CAPACITY_ONLY"
    )
    assert (
        classify_candidate(selection_metrics(candidate_kl_mean=0.008), source)
        == "NO_GO"
    )


def test_flip_summary_separates_fixed_and_new_mismatches() -> None:
    rows = []
    source_outcomes = {"a": 0, "b": 0, "c": 1}
    candidate_outcomes = {"a": 1, "b": 0, "c": 0}
    for candidate, outcomes in (
        ("source", source_outcomes),
        ("candidate", candidate_outcomes),
    ):
        for sample_id, prediction_match in outcomes.items():
            rows.append(
                {
                    "candidate": candidate,
                    "sample_id": sample_id,
                    "prediction_match": prediction_match,
                    "harmful_flip": int(candidate == "candidate" and sample_id == "c"),
                    "beneficial_flip": 0,
                    "candidate_kl": 0.1 if candidate == "source" else 0.05,
                    "domain": "domain",
                    "task_type": "task",
                    "duration": "short",
                }
            )
    summary = summarize_flip_sets(
        rows,
        candidates=("source", "candidate"),
        source_candidate="source",
    )
    candidate = summary["candidate"]
    assert candidate["fixed_source_sample_ids"] == ["a"]
    assert candidate["new_mismatch_sample_ids"] == ["c"]
    assert candidate["overlap_with_source"] == 1
    assert candidate["kl_better_than_source"] == 3
