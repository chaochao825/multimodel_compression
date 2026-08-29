from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from onevision_reader_quotient_stage_a import (  # noqa: E402
    centered_covariance,
    channel_reader_risk,
    commutator_ratio,
    descending_eigenspace,
    feature_statistics,
    local_linearity_summary,
    merge_statistics,
    subspace_overlap,
)
from analyze_onevision_reader_quotient_stage_a_spectrum import (  # noqa: E402
    bootstrap_multiplicities,
)
from probe_vsi_onevision_reader_risk_stage_a import (  # noqa: E402
    select_calibration_questions,
    strip_option_label,
)
from test_vsi_onevision_protocol import synthetic_records  # noqa: E402
from vsi_onevision_protocol import build_vsi_scene_split  # noqa: E402


def test_streaming_covariance_matches_direct_covariance() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(3, 5, 6, generator=generator)
    right = torch.randn(2, 5, 6, generator=generator)
    statistics = merge_statistics(
        [feature_statistics(left), feature_statistics(right)]
    )
    observed = centered_covariance(statistics)
    matrix = torch.cat([left.reshape(-1, 6), right.reshape(-1, 6)], dim=0)
    centered = matrix - matrix.mean(dim=0)
    expected = centered.transpose(0, 1) @ centered / (matrix.shape[0] - 1)
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-5)


def test_eigenspace_and_reader_risk_recover_known_directions() -> None:
    covariance = torch.diag(torch.tensor([9.0, 4.0, 1.0, 0.5]))
    eigenvalues, basis = descending_eigenspace(covariance, rank=2)
    assert torch.allclose(eigenvalues, torch.tensor([9.0, 4.0, 1.0, 0.5]))
    assert subspace_overlap(basis, torch.eye(4)[:, :2]) == pytest.approx(1.0)

    gradients = torch.zeros(2, 3, 4)
    gradients[0, :, 2] = 1.0
    gradients[1, :, 3] = 0.5
    risk = channel_reader_risk(
        gradients,
        torch.tensor([0.5, 2.0]),
        feature_norm_squared=4.0,
        margin_floor=0.1,
    )
    _, risk_basis = descending_eigenspace(risk, rank=2)
    assert subspace_overlap(risk_basis, torch.eye(4)[:, 2:]) == pytest.approx(1.0)
    assert commutator_ratio(covariance, risk) == pytest.approx(0.0)


def test_local_linearity_summary_detects_exact_surrogate() -> None:
    exact = torch.tensor([[-0.4, 0.2], [-0.1, 0.3]])
    margins = torch.tensor([[0.3, 0.5], [0.2, 0.1]])
    summary = local_linearity_summary(exact, exact.clone(), margins)
    assert summary["pearson"] == pytest.approx(1.0)
    assert summary["relative_l2"] == pytest.approx(0.0)
    assert summary["adverse_sign_agreement"] == pytest.approx(1.0)
    assert summary["flip_agreement"] == pytest.approx(1.0)


def test_bootstrap_multiplicities_resample_whole_videos() -> None:
    counts = bootstrap_multiplicities(
        video_count=7,
        sample_sizes=(3, 5),
        replicates=4,
        seed=11,
    )
    assert counts[3].shape == (4, 7)
    assert counts[5].shape == (4, 7)
    assert torch.equal(torch.from_numpy(counts[3].sum(axis=1)), torch.full((4,), 3))
    assert torch.equal(torch.from_numpy(counts[5].sum(axis=1)), torch.full((4,), 5))


def test_vsi_reader_question_selection_uses_calibration_debiased_only(
    tmp_path: Path,
) -> None:
    split = build_vsi_scene_split(synthetic_records())
    samples = select_calibration_questions(
        split=split,
        records=synthetic_records(),
        video_root=tmp_path,
        sample_count=12,
    )
    calibration_ids = {
        int(value)
        for scene in split["roles"]["calibration"]
        for value in scene["debiased_question_ids"]
    }
    assert len(samples) == 12
    assert all(
        int(sample.sample_id.rsplit("_", 1)[1]) in calibration_ids
        for sample in samples
    )
    assert strip_option_label("A. front-left") == "front-left"
    assert strip_option_label("B) back-right") == "back-right"
