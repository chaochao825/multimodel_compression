from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_measure_preserving_compaction import (  # noqa: E402
    additive_causal_mass_mask,
    compact_group_masses,
    measure_decision,
)


def test_compact_group_masses_track_merged_multiplicity() -> None:
    masses = compact_group_masses(
        group_count=3,
        group_size=4,
        selected_indices=torch.tensor([1]),
        device=torch.device("cpu"),
    )
    assert masses.tolist() == [4.0, 1.0, 1.0, 1.0, 1.0, 4.0]
    assert float(masses.sum().item()) == 12.0


def test_additive_mask_is_causal_and_adds_log_mass() -> None:
    mask = additive_causal_mass_mask(
        torch.tensor([1.0, 4.0, 2.0]),
        dtype=torch.float32,
    )[0, 0]
    assert mask[0, 0].item() == 0.0
    assert mask[1, 1].item() == pytest.approx(math.log(4.0))
    assert mask[2, 2].item() == pytest.approx(math.log(2.0))
    assert mask[2, 1].item() == pytest.approx(math.log(4.0))
    assert mask[0, 1].item() == torch.finfo(torch.float32).min


def test_measure_decision_requires_fidelity_or_decision_recovery() -> None:
    failed = {
        "mismatch_count": 1,
        "harmful_count": 0,
        "candidate_kl_mean": 0.02,
        "candidate_kl_p95": 0.04,
    }
    equal = {
        "mismatch_count": 0,
        "harmful_count": 0,
        "candidate_kl_mean": 0.02,
        "candidate_kl_p95": 0.03,
    }
    weighted = {
        "mismatch_count": 0,
        "harmful_count": 0,
        "candidate_kl_mean": 0.009,
        "candidate_kl_p95": 0.019,
    }
    summaries = {
        "positioned_equal_mass": {98: equal},
        "positioned_group_mass": {98: weighted},
    }
    assert measure_decision(summaries) == ("MASS_FIDELITY_RECOVERY", 98)

    summaries["positioned_group_mass"] = {98: {**weighted, "candidate_kl_mean": 0.015}}
    assert measure_decision(summaries) == ("MASS_DECISION_ONLY", 98)

    summaries["positioned_group_mass"] = {98: failed}
    assert measure_decision(summaries) == ("NO_MASS_RECOVERY", None)
