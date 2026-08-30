from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
PROBE_DIR = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBE_DIR))

from train_vsi_onevision_additive_nz_feature_state import (  # noqa: E402
    PositiveFeatureState,
)
from train_vsi_onevision_exact_boundary_additive_tail import (  # noqa: E402
    hybrid_forward,
    oracle_partition,
    selector_decision,
)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    batch_size, head_count, token_count, head_dim = 2, 2, 8, 4
    query = torch.randn(batch_size, head_count, head_dim)
    key = torch.randn(batch_size, head_count, token_count, head_dim)
    value = torch.randn(batch_size, head_count, token_count, head_dim)
    scores = torch.einsum("bhd,bhnd->bhn", query, key)
    exponentials = torch.exp(scores - scores.max(dim=2, keepdim=True).values)
    visual_z = exponentials.sum(dim=2)
    visual_n = torch.einsum("bhn,bhnd->bhd", exponentials, value)
    return {
        "query_scaled": query,
        "visual_key": key,
        "visual_value": value,
        "exact_visual_z": visual_z,
        "exact_visual_output": visual_n / visual_z.unsqueeze(-1),
        "exact_full_output": visual_n / visual_z.unsqueeze(-1),
        "nonvisual_z": torch.zeros(batch_size, head_count),
        "nonvisual_n": torch.zeros(batch_size, head_count, head_dim),
    }


@pytest.mark.parametrize("selector", ["mass_topk", "effect_topk"])
def test_oracle_partition_preserves_exact_visual_measure(selector: str) -> None:
    batch = _batch()
    partition = oracle_partition(batch, selector=selector, exact_fraction=0.25)
    query = batch["query_scaled"]
    tail_scores = torch.einsum("bhd,bhnd->bhn", query, partition["tail_key"])
    all_scores = torch.einsum("bhd,bhnd->bhn", query, batch["visual_key"])
    local_exp = torch.exp(all_scores - all_scores.max(dim=2, keepdim=True).values)
    scale = batch["exact_visual_z"] / local_exp.sum(dim=2)
    tail_exp = torch.exp(tail_scores - all_scores.max(dim=2, keepdim=True).values)
    tail_exp = tail_exp * scale.unsqueeze(-1)
    tail_n = torch.einsum("bhn,bhnd->bhd", tail_exp, partition["tail_value"])
    exact_n = batch["exact_visual_output"] * batch["exact_visual_z"].unsqueeze(-1)

    assert torch.allclose(
        partition["exact_selected_z"] + tail_exp.sum(dim=2),
        batch["exact_visual_z"],
        atol=1e-5,
    )
    assert torch.allclose(partition["exact_selected_n"] + tail_n, exact_n, atol=1e-5)


def test_constant_visual_values_remain_exact_after_hybrid_read() -> None:
    batch = _batch()
    constant = torch.tensor([1.0, -2.0, 0.5, 3.0])
    batch["visual_value"] = constant.reshape(1, 1, 1, 4).expand_as(
        batch["visual_value"]
    )
    batch["exact_visual_output"] = constant.reshape(1, 1, 4).expand(2, 2, 4)
    batch["exact_full_output"] = batch["exact_visual_output"]
    model = PositiveFeatureState(head_count=2, head_dim=4, feature_width=3)
    visual, full, _, _ = hybrid_forward(
        model, batch, selector="mass_topk", exact_fraction=0.25
    )

    assert torch.allclose(visual, batch["exact_visual_output"], atol=1e-5)
    assert torch.allclose(full, batch["exact_full_output"], atol=1e-5)


def _summary(*, mean: float, p95: float, worst: float) -> dict[str, float | int]:
    return {
        "cell_count": 72,
        "visual_mean": mean,
        "visual_p95": p95,
        "visual_worst": worst,
        "full_mean": mean / 2,
        "full_p95": p95 / 2,
    }


def test_selector_decision_separates_go_signal_and_no_go() -> None:
    exact_only = _summary(mean=0.1, p95=0.2, worst=0.3)
    assert (
        selector_decision(
            exact_only=exact_only,
            learned=_summary(mean=0.004, p95=0.008, worst=0.015),
            active_state_ratio=3.0,
        )
        == "BOUNDARY_ADDITIVE_TAIL_ORACLE_GO"
    )
    assert (
        selector_decision(
            exact_only=exact_only,
            learned=_summary(mean=0.008, p95=0.015, worst=0.04),
            active_state_ratio=3.0,
        )
        == "BOUNDARY_ADDITIVE_TAIL_CAPACITY_SIGNAL"
    )
    assert (
        selector_decision(
            exact_only=exact_only,
            learned=_summary(mean=0.03, p95=0.08, worst=0.2),
            active_state_ratio=3.0,
        )
        == "NO_BOUNDARY_ADDITIVE_TAIL_PATH"
    )
