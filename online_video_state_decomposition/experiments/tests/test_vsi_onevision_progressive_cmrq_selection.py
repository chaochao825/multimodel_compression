from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest


pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from materialize_vsi_role_videos import materialize_role  # noqa: E402
from probe_vsi_onevision_reader_risk_stage_a import (  # noqa: E402
    select_role_questions,
)
from vsi_onevision_progressive_cmrq_selection import (  # noqa: E402
    paired_bootstrap_delta,
    progressive_task_summary,
    selection_gate,
)
from vsi_onevision_protocol import PROTOCOL_ID  # noqa: E402


def _split() -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "roles": {
            "calibration": [
                {
                    "dataset": "scannet",
                    "scene_name": "cal",
                    "sample_id": "vsi_scannet_cal",
                    "relative_video_path": "scannet/cal.mp4",
                    "debiased_question_ids": [1],
                }
            ],
            "selection": [
                {
                    "dataset": "scannet",
                    "scene_name": "sel",
                    "sample_id": "vsi_scannet_sel",
                    "relative_video_path": "scannet/sel.mp4",
                    "debiased_question_ids": [2],
                }
            ],
        },
    }


def test_role_question_selection_respects_frozen_role(tmp_path: Path) -> None:
    records = [
        {
            "id": 1,
            "dataset": "scannet",
            "scene_name": "cal",
            "ground_truth": "A",
            "question": "calibration?",
            "options": ["A. yes", "B. no"],
        },
        {
            "id": 2,
            "dataset": "scannet",
            "scene_name": "sel",
            "ground_truth": "B",
            "question": "selection?",
            "options": ["A. yes", "B. no"],
        },
    ]
    selected = select_role_questions(
        split=_split(),
        records=records,
        video_root=tmp_path,
        role="selection",
        sample_count=1,
    )
    assert selected[0].sample_id == "vsi_question_2"
    assert selected[0].answer_index == 1
    assert selected[0].candidates == ("yes", "no")


def test_materialize_role_extracts_only_requested_videos(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    with zipfile.ZipFile(archive_root / "scannet.zip", "w") as archive:
        archive.writestr("scannet/cal.mp4", b"calibration")
        archive.writestr("scannet/sel.mp4", b"selection")
    out_dir = tmp_path / "videos"
    summary = materialize_role(
        split=_split(),
        archive_root=archive_root,
        out_dir=out_dir,
        role="selection",
    )
    assert summary["materialized_count"] == 1
    assert (out_dir / "scannet" / "sel.mp4").read_bytes() == b"selection"
    assert not (out_dir / "scannet" / "cal.mp4").exists()


def test_progressive_summary_uses_compressed_margin_for_task_delivery() -> None:
    rows = [
        {
            "approximate_top1_margin": 0.0,
            "candidate_kl": 0.4,
            "feature_relative_l2": 0.2,
            "prediction_match": 0,
            "baseline_correct": 1,
            "approximate_correct": 0,
            "harmful": 1,
        },
        {
            "approximate_top1_margin": 0.5,
            "candidate_kl": 0.2,
            "feature_relative_l2": 0.1,
            "prediction_match": 1,
            "baseline_correct": 1,
            "approximate_correct": 1,
            "harmful": 0,
        },
    ]
    summary = progressive_task_summary(
        rows,
        margin_threshold=0.0,
        compressed_state_bytes=2,
        dense_state_bytes=10,
    )
    assert summary["fallback_count"] == 1
    assert summary["delivered_agreement"] == pytest.approx(1.0)
    assert summary["delivered_accuracy"] == pytest.approx(1.0)
    assert summary["delivered_accuracy_delta"] == pytest.approx(0.0)


def test_selection_gate_requires_null_vsi_safety_and_cost_guards() -> None:
    method_summaries = {
        "cmrq_mix_g32_w0p3_r456": {
            "candidate_kl_mean": 0.001,
            "candidate_kl_p95": 0.002,
        },
        "permuted_mix_g32_w0p3_r456": {
            "candidate_kl_mean": 0.002,
            "candidate_kl_p95": 0.003,
        },
        "vsi_pca_cal120_r456": {
            "candidate_kl_mean": 0.0011,
            "candidate_kl_p95": 0.0021,
        },
    }
    progressive = {
        "remaining_harmful_count": 0,
        "delivered_agreement": 1.0,
        "fallback_rate": 0.1,
        "conservative_transfer_ratio": 4.2,
        "delivered_accuracy_delta": 0.0,
    }
    paired = paired_bootstrap_delta(
        [0.001, 0.0011, 0.0009],
        [0.002, 0.0021, 0.0019],
        replicates=1000,
        seed=7,
    )
    gate = selection_gate(
        method_summaries=method_summaries,
        progressive=progressive,
        paired_delta=paired,
    )
    assert gate["decision"] == "GO"
    progressive["fallback_rate"] = 0.2
    gate = selection_gate(
        method_summaries=method_summaries,
        progressive=progressive,
        paired_delta=paired,
    )
    assert gate["decision"] == "NO_GO"
