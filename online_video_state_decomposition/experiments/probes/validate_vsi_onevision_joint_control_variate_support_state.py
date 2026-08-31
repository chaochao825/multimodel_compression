from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_vsi_onevision_control_variate_support_state import (
    contribution_pages,
    corrected_outputs,
)
from probe_vsi_onevision_query_fixed_positive_gaussian_measure import LAYERS
from train_vsi_onevision_additive_nz_feature_state import (
    DEVELOPMENT_POSITIONS,
    PositiveFeatureState,
    batch_examples,
    capture_paths,
    load_examples,
)
from train_vsi_onevision_joint_control_variate_support_state import support_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = capture_paths(args.capture_dir)
    device = torch.device(args.device)
    maximum_state_replay_error = 0.0
    maximum_exact_recovery_error = 0.0
    maximum_exact_recovery_relative_error = 0.0
    exact_page_budget_ok = True
    finite_checkpoint_tensors = True
    cells = 0

    with torch.no_grad():
        for layer_index in LAYERS:
            examples = load_examples(
                paths, positions=DEVELOPMENT_POSITIONS, layer_index=layer_index
            )
            head_count, head_dim = examples[0]["query_scaled"].shape
            checkpoint = torch.load(
                args.checkpoint_dir
                / f"layer_{layer_index:02d}_joint_support_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            for state_name in ("independent_state", "joint_state"):
                model = PositiveFeatureState(
                    head_count=head_count, head_dim=head_dim, feature_width=32
                ).to(device)
                model.load_state_dict(checkpoint[state_name])
                model.eval()
                finite_checkpoint_tensors &= all(
                    torch.isfinite(value).all().item()
                    for value in model.state_dict().values()
                )
                for example_index in range(len(examples)):
                    batch = batch_examples(
                        examples, np.asarray([example_index]), device=device
                    )
                    pages = contribution_pages(model, batch, page_size=4)
                    direct_visual, direct_full, _ = model(
                        batch["query_scaled"],
                        batch["visual_key"],
                        batch["visual_value"],
                        batch["nonvisual_z"],
                        batch["nonvisual_n"],
                    )
                    replay_visual, replay_full = corrected_outputs(
                        pages, batch, mask=None, exact_only=False
                    )
                    maximum_state_replay_error = max(
                        maximum_state_replay_error,
                        float((direct_visual - replay_visual).abs().max().item()),
                        float((direct_full - replay_full).abs().max().item()),
                    )

                    all_pages = torch.ones_like(pages["exact_z"], dtype=torch.bool)
                    exact_visual, exact_full = corrected_outputs(
                        pages, batch, mask=all_pages, exact_only=False
                    )
                    for prediction, target in (
                        (exact_visual, batch["exact_visual_output"]),
                        (exact_full, batch["exact_full_output"]),
                    ):
                        maximum_exact_recovery_error = max(
                            maximum_exact_recovery_error,
                            float((prediction - target).abs().max().item()),
                        )
                        relative_error = torch.linalg.vector_norm(
                            prediction - target
                        ) / torch.linalg.vector_norm(target).clamp_min(1e-8)
                        maximum_exact_recovery_relative_error = max(
                            maximum_exact_recovery_relative_error,
                            float(relative_error.item()),
                        )

                    for mode in ("mass", "residual"):
                        mask = support_mask(
                            mode=mode,
                            pages=pages,
                            exact_visual_output=batch["exact_visual_output"],
                            exact_fraction=0.25,
                            greedy_round_size=14,
                        )
                        exact_page_budget_ok &= torch.all(
                            mask.sum(dim=-1) == 98
                        ).item()
                    cells += 1

    passed = (
        finite_checkpoint_tensors
        and exact_page_budget_ok
        and cells == 144
        and maximum_state_replay_error <= 1e-5
        and maximum_exact_recovery_error <= 3e-5
        and maximum_exact_recovery_relative_error <= 2e-6
    )
    result = {
        "cells": cells,
        "decision": "PASS" if passed else "FAIL",
        "exact_page_budget_ok": exact_page_budget_ok,
        "finite_checkpoint_tensors": finite_checkpoint_tensors,
        "maximum_all_page_recovery_absolute_error": (
            maximum_exact_recovery_error
        ),
        "maximum_all_page_recovery_relative_error": (
            maximum_exact_recovery_relative_error
        ),
        "maximum_state_replay_absolute_error": maximum_state_replay_error,
        "scope": (
            "Read-only replay of both trained states on exposed development "
            "positions 73-96; no training, selection, or sealed-data access."
        ),
    }
    args.out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
