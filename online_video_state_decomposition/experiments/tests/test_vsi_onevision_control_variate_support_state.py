from __future__ import annotations

import torch

from analyze_vsi_onevision_control_variate_support_state import (
    classify,
    corrected_outputs,
    page_mask,
    select_residual_pages,
    sum_pages,
)


def test_sum_pages_preserves_trailing_dimensions() -> None:
    tensor = torch.arange(2 * 3 * 8 * 5, dtype=torch.float32).reshape(2, 3, 8, 5)
    pages = sum_pages(tensor, 4)
    assert pages.shape == (2, 3, 2, 5)
    torch.testing.assert_close(pages.sum(dim=2), tensor.sum(dim=2))


def test_full_exact_correction_recovers_exact_measure() -> None:
    approximate_z = torch.tensor([[[2.0, 1.0]]])
    exact_z = torch.tensor([[[1.0, 3.0]]])
    approximate_n = torch.tensor([[[[2.0, 0.0], [0.0, 1.0]]]])
    exact_n = torch.tensor([[[[1.0, 0.0], [0.0, 3.0]]]])
    pages = {
        "approximate_z": approximate_z,
        "exact_z": exact_z,
        "approximate_n": approximate_n,
        "exact_n": exact_n,
    }
    batch = {
        "nonvisual_z": torch.tensor([[2.0]]),
        "nonvisual_n": torch.tensor([[[2.0, 2.0]]]),
    }
    mask = torch.ones_like(exact_z, dtype=torch.bool)
    visual, full = corrected_outputs(pages, batch, mask=mask, exact_only=False)
    torch.testing.assert_close(visual, torch.tensor([[[0.25, 0.75]]]))
    torch.testing.assert_close(full, torch.tensor([[[0.5, 5.0 / 6.0]]]))


def test_residual_selector_prefers_page_that_repairs_output() -> None:
    pages = {
        "approximate_z": torch.tensor([[[1.0, 1.0, 1.0]]]),
        "exact_z": torch.tensor([[[1.0, 1.0, 1.0]]]),
        "approximate_n": torch.tensor([[[[0.0], [0.0], [0.0]]]]),
        "exact_n": torch.tensor([[[[0.0], [3.0], [0.5]]]]),
    }
    target = torch.tensor([[[1.0]]])
    selected = select_residual_pages(pages, target, selected_pages=1, round_size=1)
    expected = page_mask(torch.tensor([[[1]]]), 3)
    assert torch.equal(selected, expected)


def test_capacity_gate_requires_all_guards() -> None:
    joint = {
        "cell_count": 72,
        "visual_mean": 0.004,
        "visual_p95": 0.009,
        "visual_worst": 0.019,
    }
    comparison = {
        "relative_risk_improvement": 0.4,
        "bootstrap_lower_95": 0.3,
        "bootstrap_upper_95": 0.5,
    }
    assert classify(joint=joint, comparison=comparison, active_state_ratio=3.5) == (
        "SUPPORT_STATE_CAPACITY_GO"
    )
    assert classify(joint=joint, comparison=comparison, active_state_ratio=2.9) == (
        "NO_SUPPORT_STATE_CAPACITY"
    )
