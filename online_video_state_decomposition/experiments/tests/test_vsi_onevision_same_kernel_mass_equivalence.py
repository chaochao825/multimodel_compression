from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_same_kernel_mass_equivalence import (  # noqa: E402
    compact_video_masses,
    deterministic_support,
    outcome,
)


def test_deterministic_support_is_unique_and_evenly_spaced() -> None:
    selected = deterministic_support(
        group_count=392,
        selected_count=196,
        device=torch.device("cpu"),
    )
    assert selected.shape == (196,)
    assert torch.equal(selected[:4], torch.tensor([0, 2, 4, 6]))
    assert int(selected[-1]) == 390
    assert torch.unique(selected).numel() == selected.numel()


def test_compact_video_masses_preserve_group_measure() -> None:
    selected = torch.tensor([1], dtype=torch.long)
    masses = compact_video_masses(
        group_count=3,
        group_size=4,
        selected_indices=selected,
        device=torch.device("cpu"),
    )
    assert torch.equal(masses, torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0, 4.0]))
    assert float(masses.sum()) == 12.0


def test_outcome_requires_both_equivalence_guards() -> None:
    assert outcome(1e-6, 1e-6) == "SAME_KERNEL_MASS_VALID"
    assert outcome(2e-5, 1e-6) == "INVALID_KERNEL_EQUIVALENCE"
    assert outcome(1e-6, 2e-5) == "INVALID_KERNEL_EQUIVALENCE"
