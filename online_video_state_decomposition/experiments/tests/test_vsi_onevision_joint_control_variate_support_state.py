from __future__ import annotations

import torch

from train_vsi_onevision_joint_control_variate_support_state import (
    classify,
    corrected_visual_z,
    support_mask,
)


def test_mass_support_respects_page_budget() -> None:
    pages = {
        "exact_z": torch.tensor([[[1.0, 4.0, 3.0, 2.0]]]),
        "approximate_z": torch.ones(1, 1, 4),
        "exact_n": torch.zeros(1, 1, 4, 1),
        "approximate_n": torch.zeros(1, 1, 4, 1),
    }
    mask = support_mask(
        mode="mass",
        pages=pages,
        exact_visual_output=torch.zeros(1, 1, 1),
        exact_fraction=0.5,
        greedy_round_size=1,
    )
    assert mask.sum().item() == 2
    assert mask[0, 0, 1]
    assert mask[0, 0, 2]


def test_corrected_visual_z_uses_control_variate() -> None:
    pages = {
        "approximate_z": torch.tensor([[[1.0, 2.0]]]),
        "exact_z": torch.tensor([[[4.0, 8.0]]]),
    }
    mask = torch.tensor([[[True, False]]])
    torch.testing.assert_close(corrected_visual_z(pages, mask), torch.tensor([[6.0]]))


def test_joint_gate_requires_both_factorial_margins() -> None:
    joint = {
        "cell_count": 72,
        "visual_mean": 0.004,
        "visual_p95": 0.009,
        "visual_worst": 0.019,
    }
    strong = {"bootstrap_lower_95": 0.3}
    positive = {"bootstrap_lower_95": 0.01}
    assert (
        classify(
            joint=joint,
            versus_independent=strong,
            versus_support_only=positive,
            versus_state_only=positive,
            active_state_ratio=3.5,
        )
        == "JOINT_SUPPORT_STATE_CAPACITY_GO"
    )
    assert (
        classify(
            joint=joint,
            versus_independent=strong,
            versus_support_only={"bootstrap_lower_95": -0.01},
            versus_state_only=positive,
            active_state_ratio=3.5,
        )
        == "NO_JOINT_SUPPORT_STATE_CAPACITY"
    )
