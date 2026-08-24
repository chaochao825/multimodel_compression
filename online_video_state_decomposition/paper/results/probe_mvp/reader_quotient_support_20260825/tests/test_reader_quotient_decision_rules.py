from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_reader_quotient_support_oracle import (  # noqa: E402
    classify_onevision_replication,
)


def test_onevision_replication_requires_four_positive_tasks() -> None:
    summary = {
        "aggregate_candidate_kl_reduction": 0.54,
        "candidate_kl_p95_ratio": 0.58,
    }
    assert (
        classify_onevision_replication(summary, positive_tasks=3)
        == "BOUNDARY"
    )
    assert classify_onevision_replication(summary, positive_tasks=4) == "GO"


def test_onevision_replication_tail_harm_is_adverse() -> None:
    summary = {
        "aggregate_candidate_kl_reduction": 0.54,
        "candidate_kl_p95_ratio": 1.2,
    }
    assert (
        classify_onevision_replication(summary, positive_tasks=5)
        == "ADVERSE"
    )
