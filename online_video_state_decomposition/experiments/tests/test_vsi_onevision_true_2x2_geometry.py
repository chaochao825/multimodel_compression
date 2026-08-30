from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from probe_vsi_onevision_true_2x2_geometry import (  # noqa: E402
    classify_geometry_headroom,
    flat_means_and_offsets,
    spatial_2x2_means_and_offsets,
)


def test_spatial_2x2_groups_preserve_true_grid_neighbors() -> None:
    features = torch.arange(196, dtype=torch.float32).reshape(1, 196, 1)

    spatial_means, spatial_offsets = spatial_2x2_means_and_offsets(features)
    flat_means, flat_offsets = flat_means_and_offsets(features, group_size=4)

    assert spatial_means.shape == flat_means.shape == (49, 1)
    assert spatial_offsets[0].tolist() == [0, 1, 14, 15]
    assert flat_offsets[0].tolist() == [0, 1, 2, 3]
    assert spatial_means[0].item() == 7.5
    assert flat_means[0].item() == 1.5


def test_geometry_classifier_separates_strict_and_decision_headroom() -> None:
    strict = {
        "positioned_equal_mass": {
            "mismatch_reduction": 2,
            "harmful_delta": 0,
            "mean_kl_ratio": 0.8,
            "p95_kl_ratio": 0.7,
        }
    }
    assert classify_geometry_headroom(strict) == "TRUE_2X2_GEOMETRY_HEADROOM"

    decision_only = {
        "positioned_equal_mass": {
            "mismatch_reduction": 2,
            "harmful_delta": 0,
            "mean_kl_ratio": 1.0,
            "p95_kl_ratio": 1.2,
        }
    }
    assert classify_geometry_headroom(decision_only) == "TRUE_2X2_DECISION_HEADROOM"

    null = {
        "positioned_equal_mass": {
            "mismatch_reduction": 1,
            "harmful_delta": 0,
            "mean_kl_ratio": 0.5,
            "p95_kl_ratio": 0.5,
        }
    }
    assert classify_geometry_headroom(null) == "NO_TRUE_2X2_GEOMETRY_HEADROOM"
