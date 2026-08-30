from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from analyze_vsi_onevision_exact_boundary_tail import (  # noqa: E402
    support_geometry,
)


def test_support_geometry_reports_exact_mass_and_effect_overlap() -> None:
    query = torch.tensor([[1.0, 0.0]])
    key = torch.tensor([[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]])
    value = torch.tensor([[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]])
    weights = torch.softmax(torch.tensor([[4.0, 3.0, 2.0, 1.0]]), dim=1)
    output = torch.einsum("hn,hnd->hd", weights, value)

    metrics = support_geometry(
        query,
        key,
        value,
        output,
        exact_fraction=0.5,
    )

    assert torch.allclose(metrics["mass_retained"], weights[:, :2].sum(dim=1).double())
    assert 0.0 < float(metrics["tail_ess"].item()) <= 2.0
    assert 0.0 <= float(metrics["support_jaccard"].item()) <= 1.0
