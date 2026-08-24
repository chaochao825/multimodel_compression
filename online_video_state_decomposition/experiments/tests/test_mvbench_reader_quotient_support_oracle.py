from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBES = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBES))

from mvbench_reader_quotient_support_oracle import (  # noqa: E402
    distribution_metrics,
    normalized_mixed_scores,
    sparse_reconstruction,
)


def test_sparse_reconstruction_restores_selected_tokens() -> None:
    base = torch.zeros((2, 4, 3), dtype=torch.float32)
    residual = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    scores = torch.tensor([[1.0, 4.0, 3.0, 2.0], [4.0, 1.0, 2.0, 3.0]])
    reconstruction, indices = sparse_reconstruction(base, residual, scores, 2)

    assert indices.tolist() == [[1, 2], [0, 3]]
    assert torch.equal(reconstruction[0, 1], residual[0, 1])
    assert torch.equal(reconstruction[1, 3], residual[1, 3])
    assert torch.count_nonzero(reconstruction[0, 0]) == 0


def test_mixed_scores_are_scale_invariant() -> None:
    euclidean = torch.tensor([[1.0, 2.0, 4.0]])
    fisher = torch.tensor([[4.0, 2.0, 1.0]])
    first = normalized_mixed_scores(euclidean, fisher)
    second = normalized_mixed_scores(euclidean * 7.0, fisher * 0.25)
    assert torch.allclose(first, second)


def test_distribution_metrics_are_zero_for_identical_logits() -> None:
    logits = torch.tensor([0.2, -0.3, 1.1, 0.7, -0.1])
    metrics = distribution_metrics(logits, logits.clone(), [0, 2, 3], 1)
    assert metrics["candidate_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["vocabulary_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["candidate_top1_match"] == 1
