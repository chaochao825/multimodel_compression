from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from probe_vsi_onevision_query_fixed_positive_gaussian_measure import LAYERS
from train_vsi_onevision_additive_nz_feature_state import (
    DEVELOPMENT_POSITIONS,
    PositiveFeatureState,
    batch_examples,
    capture_paths,
    load_examples,
)
from vsi_onevision_protocol import PROTOCOL_ID


ROLE = "development_control_variate_support_state_capacity"
METHODS = (
    "exact_only_mass",
    "state_only",
    "independent_mass_correction",
    "joint_residual_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--exact-fraction", type=float, default=0.25)
    parser.add_argument("--greedy-round-size", type=int, default=14)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def sum_pages(tensor: torch.Tensor, page_size: int) -> torch.Tensor:
    token_count = tensor.shape[2]
    if token_count % page_size != 0:
        raise ValueError("visual token count must be divisible by page size")
    page_count = token_count // page_size
    trailing = tensor.shape[3:]
    return tensor.reshape(*tensor.shape[:2], page_count, page_size, *trailing).sum(
        dim=3
    )


def contribution_pages(
    model: PositiveFeatureState,
    batch: dict[str, torch.Tensor],
    *,
    page_size: int,
) -> dict[str, torch.Tensor]:
    query = batch["query_scaled"]
    key = batch["visual_key"]
    value = batch["visual_value"]

    normalized_key = key / model.key_rms[None, :, None, :]
    key_feature = model.positive(
        torch.einsum("bhnd,hdr->bhnr", normalized_key, model.key_weight)
        + model.key_bias[None, :, None, :]
    )
    normalized_query = query / model.query_rms[None, :, :]
    query_feature = model.positive(
        torch.einsum("bhd,hdr->bhr", normalized_query, model.query_weight)
        + model.query_bias[None, :, :]
    )
    approximate_token_z = (
        torch.einsum("bhr,bhnr->bhn", query_feature, key_feature)
        * torch.exp(model.log_visual_scale)[None, :, None]
    )
    approximate_token_n = approximate_token_z.unsqueeze(-1) * value

    exact_token_z = batch["exact_visual_exp"]
    visual_mass_error = (
        exact_token_z.sum(dim=-1) - batch["exact_visual_z"]
    ).abs() / batch["exact_visual_z"].abs().clamp_min(1e-8)
    if visual_mass_error.max() > 1e-6:
        raise RuntimeError("stored exact visual contributions do not sum to visual Z")
    exact_token_n = exact_token_z.unsqueeze(-1) * value

    return {
        "approximate_z": sum_pages(approximate_token_z, page_size),
        "approximate_n": sum_pages(approximate_token_n, page_size),
        "exact_z": sum_pages(exact_token_z, page_size),
        "exact_n": sum_pages(exact_token_n, page_size),
    }


def page_mask(indices: torch.Tensor, page_count: int) -> torch.Tensor:
    mask = torch.zeros(
        *indices.shape[:-1], page_count, dtype=torch.bool, device=indices.device
    )
    return mask.scatter(-1, indices, True)


def select_mass_pages(exact_z: torch.Tensor, selected_pages: int) -> torch.Tensor:
    return torch.topk(exact_z, k=selected_pages, dim=-1).indices


def select_residual_pages(
    pages: dict[str, torch.Tensor],
    exact_visual_output: torch.Tensor,
    *,
    selected_pages: int,
    round_size: int,
) -> torch.Tensor:
    approximate_n = pages["approximate_n"]
    approximate_z = pages["approximate_z"]
    delta_n = pages["exact_n"] - approximate_n
    delta_z = pages["exact_z"] - approximate_z
    current_n = approximate_n.sum(dim=2)
    current_z = approximate_z.sum(dim=2)
    selected = torch.zeros_like(approximate_z, dtype=torch.bool)
    remaining = selected_pages

    while remaining:
        batch_size = min(round_size, remaining)
        candidate_n = current_n.unsqueeze(2) + delta_n
        candidate_z = current_z.unsqueeze(2) + delta_z
        candidate_y = candidate_n / candidate_z.unsqueeze(-1).clamp_min(1e-8)
        error = (candidate_y - exact_visual_output.unsqueeze(2)).square().sum(dim=-1)
        error = error.masked_fill(selected, torch.inf)
        indices = torch.topk(error, k=batch_size, dim=-1, largest=False).indices
        chosen = page_mask(indices, approximate_z.shape[-1])
        selected |= chosen
        current_n = current_n + (delta_n * chosen.unsqueeze(-1)).sum(dim=2)
        current_z = current_z + (delta_z * chosen).sum(dim=2)
        remaining -= batch_size
    return selected


def corrected_outputs(
    pages: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    mask: torch.Tensor | None,
    exact_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if exact_only:
        if mask is None:
            raise ValueError("exact-only output requires a support mask")
        visual_n = (pages["exact_n"] * mask.unsqueeze(-1)).sum(dim=2)
        visual_z = (pages["exact_z"] * mask).sum(dim=2)
    else:
        visual_n = pages["approximate_n"].sum(dim=2)
        visual_z = pages["approximate_z"].sum(dim=2)
        if mask is not None:
            visual_n = visual_n + (
                (pages["exact_n"] - pages["approximate_n"]) * mask.unsqueeze(-1)
            ).sum(dim=2)
            visual_z = visual_z + (
                (pages["exact_z"] - pages["approximate_z"]) * mask
            ).sum(dim=2)
    visual = visual_n / visual_z.unsqueeze(-1).clamp_min(1e-8)
    full = (visual_n + batch["nonvisual_n"]) / (
        visual_z + batch["nonvisual_z"]
    ).unsqueeze(-1).clamp_min(1e-8)
    return visual, full


def relative_risk(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1e-8)


def evaluate_method(
    *,
    method: str,
    pages: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mass_mask: torch.Tensor,
    residual_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "exact_only_mass":
        return corrected_outputs(pages, batch, mask=mass_mask, exact_only=True)
    if method == "state_only":
        return corrected_outputs(pages, batch, mask=None, exact_only=False)
    if method == "independent_mass_correction":
        return corrected_outputs(pages, batch, mask=mass_mask, exact_only=False)
    if method == "joint_residual_oracle":
        return corrected_outputs(pages, batch, mask=residual_mask, exact_only=False)
    raise ValueError(f"unknown method: {method}")


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    visual = np.asarray([float(row["visual_relative_l2"]) for row in rows])
    full = np.asarray([float(row["full_relative_l2"]) for row in rows])
    risk = np.asarray([float(row["visual_risk"]) for row in rows])
    return {
        "cell_count": len(rows),
        "visual_mean": float(visual.mean()),
        "visual_p95": float(np.quantile(visual, 0.95)),
        "visual_worst": float(visual.max()),
        "full_mean": float(full.mean()),
        "full_p95": float(np.quantile(full, 0.95)),
        "visual_risk_mean": float(risk.mean()),
    }


def paired_position_bootstrap(
    rows: list[dict[str, object]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    independent = {
        (int(row["position"]), int(row["layer_index"])): float(row["visual_risk"])
        for row in rows
        if row["method"] == "independent_mass_correction"
    }
    joint = {
        (int(row["position"]), int(row["layer_index"])): float(row["visual_risk"])
        for row in rows
        if row["method"] == "joint_residual_oracle"
    }
    if independent.keys() != joint.keys():
        raise ValueError("paired support-state rows do not share identities")
    positions = sorted({position for position, _ in independent})
    layers = sorted({layer for _, layer in independent})
    independent_by_position = np.asarray(
        [
            np.mean([independent[(position, layer)] for layer in layers])
            for position in positions
        ]
    )
    joint_by_position = np.asarray(
        [
            np.mean([joint[(position, layer)] for layer in layers])
            for position in positions
        ]
    )
    point = 1.0 - joint_by_position.mean() / independent_by_position.mean()
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(positions), size=len(positions))
        samples[index] = (
            1.0
            - joint_by_position[selected].mean()
            / independent_by_position[selected].mean()
        )
    return {
        "relative_risk_improvement": float(point),
        "bootstrap_lower_95": float(np.quantile(samples, 0.025)),
        "bootstrap_upper_95": float(np.quantile(samples, 0.975)),
    }


def classify(
    *,
    joint: dict[str, float | int],
    comparison: dict[str, float],
    active_state_ratio: float,
) -> str:
    if (
        int(joint["cell_count"]) == 72
        and float(joint["visual_mean"]) <= 0.005
        and float(joint["visual_p95"]) <= 0.01
        and float(joint["visual_worst"]) <= 0.02
        and comparison["bootstrap_lower_95"] >= 0.25
        and active_state_ratio >= 3.0
    ):
        return "SUPPORT_STATE_CAPACITY_GO"
    return "NO_SUPPORT_STATE_CAPACITY"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty support-state rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if (
        args.page_size != 4
        or args.exact_fraction != 0.25
        or args.greedy_round_size != 14
        or args.bootstrap_repetitions != 10000
        or args.seed != 20260901
    ):
        raise ValueError("registered control-variate support-state identity changed")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("support-state output must be empty")
    paths = capture_paths(args.capture_dir)
    capture_summary = json.loads(
        (args.capture_dir / "summary.json").read_text(encoding="utf-8")
    )
    if capture_summary["protocol_id"] != PROTOCOL_ID:
        raise ValueError("capture protocol identity mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    maximum_state_replay_error = 0.0
    maximum_exact_recovery_error = 0.0
    maximum_exact_recovery_relative_error = 0.0

    for layer_index in LAYERS:
        examples = load_examples(
            paths, positions=DEVELOPMENT_POSITIONS, layer_index=layer_index
        )
        head_count, head_dim = examples[0]["query_scaled"].shape
        checkpoint = torch.load(
            args.checkpoint_dir / f"layer_{layer_index:02d}_feature_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint["layer_index"] != layer_index
            or checkpoint["feature_width"] != 32
        ):
            raise ValueError("additive-state checkpoint identity mismatch")
        model = PositiveFeatureState(
            head_count=head_count, head_dim=head_dim, feature_width=32
        ).to(device)
        model.load_state_dict(checkpoint["best_state"])
        model.eval()

        with torch.no_grad():
            for example_index in range(len(examples)):
                batch = batch_examples(
                    examples, np.asarray([example_index]), device=device
                )
                pages = contribution_pages(model, batch, page_size=args.page_size)
                page_count = pages["exact_z"].shape[-1]
                selected_pages = round(page_count * args.exact_fraction)
                mass_indices = select_mass_pages(pages["exact_z"], selected_pages)
                mass_mask = page_mask(mass_indices, page_count)
                residual_mask = select_residual_pages(
                    pages,
                    batch["exact_visual_output"],
                    selected_pages=selected_pages,
                    round_size=args.greedy_round_size,
                )
                if not torch.all(mass_mask.sum(dim=-1) == selected_pages):
                    raise RuntimeError("mass selector violated the exact page budget")
                if not torch.all(residual_mask.sum(dim=-1) == selected_pages):
                    raise RuntimeError(
                        "residual selector violated the exact page budget"
                    )

                direct_visual, direct_full, _ = model(
                    batch["query_scaled"],
                    batch["visual_key"],
                    batch["visual_value"],
                    batch["nonvisual_z"],
                    batch["nonvisual_n"],
                )
                state_visual, state_full = corrected_outputs(
                    pages, batch, mask=None, exact_only=False
                )
                maximum_state_replay_error = max(
                    maximum_state_replay_error,
                    float((state_visual - direct_visual).abs().max().item()),
                    float((state_full - direct_full).abs().max().item()),
                )
                all_pages = torch.ones_like(mass_mask)
                recovered_visual, recovered_full = corrected_outputs(
                    pages, batch, mask=all_pages, exact_only=False
                )
                maximum_exact_recovery_error = max(
                    maximum_exact_recovery_error,
                    float(
                        (recovered_visual - batch["exact_visual_output"])
                        .abs()
                        .max()
                        .item()
                    ),
                    float(
                        (recovered_full - batch["exact_full_output"]).abs().max().item()
                    ),
                )
                maximum_exact_recovery_relative_error = max(
                    maximum_exact_recovery_relative_error,
                    float(
                        (
                            torch.linalg.vector_norm(
                                recovered_visual - batch["exact_visual_output"]
                            )
                            / torch.linalg.vector_norm(
                                batch["exact_visual_output"]
                            ).clamp_min(1e-8)
                        ).item()
                    ),
                    float(
                        (
                            torch.linalg.vector_norm(
                                recovered_full - batch["exact_full_output"]
                            )
                            / torch.linalg.vector_norm(
                                batch["exact_full_output"]
                            ).clamp_min(1e-8)
                        ).item()
                    ),
                )
                for method in METHODS:
                    visual, full = evaluate_method(
                        method=method,
                        pages=pages,
                        batch=batch,
                        mass_mask=mass_mask,
                        residual_mask=residual_mask,
                    )
                    visual_risk = relative_risk(visual, batch["exact_visual_output"])
                    full_risk = relative_risk(full, batch["exact_full_output"])
                    rows.append(
                        {
                            "split": "development",
                            "position": example_index + 73,
                            "layer_index": layer_index,
                            "method": method,
                            "page_size": args.page_size,
                            "exact_fraction": args.exact_fraction,
                            "visual_relative_l2": float(visual_risk.sqrt().item()),
                            "visual_risk": float(visual_risk.item()),
                            "full_relative_l2": float(full_risk.sqrt().item()),
                            "full_risk": float(full_risk.item()),
                        }
                    )

    if maximum_state_replay_error > 1e-5:
        raise RuntimeError("page contributions do not replay the learned state")
    if (
        maximum_exact_recovery_error > 3e-5
        or maximum_exact_recovery_relative_error > 2e-6
    ):
        raise RuntimeError("all-page correction does not recover dense attention")

    summaries = {
        method: summarize([row for row in rows if row["method"] == method])
        for method in METHODS
    }
    comparison = paired_position_bootstrap(
        rows, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    token_count = 8 * 196
    head_dim = 128
    dense_state = 2 * token_count * head_dim
    active_state = args.exact_fraction * dense_state + 32 * (head_dim + 1)
    active_state_ratio = dense_state / active_state
    decision = classify(
        joint=summaries["joint_residual_oracle"],
        comparison=comparison,
        active_state_ratio=active_state_ratio,
    )
    write_csv(args.out_dir / "development_rows.csv", rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "methods": list(METHODS),
        "method_summaries": summaries,
        "joint_vs_independent": comparison,
        "page_size": args.page_size,
        "exact_fraction": args.exact_fraction,
        "feature_width": 32,
        "greedy_round_size": args.greedy_round_size,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "development_positions": [73, 96],
        "confirmation_positions_unread": [97, 120],
        "layers": list(LAYERS),
        "analytic_active_state_ratio": active_state_ratio,
        "maximum_state_replay_error": maximum_state_replay_error,
        "maximum_exact_recovery_error": maximum_exact_recovery_error,
        "maximum_exact_recovery_relative_error": (
            maximum_exact_recovery_relative_error
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Development-only target-visible regular-page support capacity. The joint "
            "selector reads exact residual effects and is not deployable. No task, "
            "confirmation, formal, latency, or speed claim is authorized."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
