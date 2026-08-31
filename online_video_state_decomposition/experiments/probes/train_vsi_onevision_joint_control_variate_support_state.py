from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from analyze_vsi_onevision_control_variate_support_state import (
    contribution_pages,
    corrected_outputs,
    page_mask,
    select_mass_pages,
    select_residual_pages,
)
from probe_vsi_onevision_query_fixed_positive_gaussian_measure import LAYERS
from train_vsi_onevision_additive_nz_feature_state import (
    DEVELOPMENT_POSITIONS,
    TRAIN_POSITIONS,
    PositiveFeatureState,
    batch_examples,
    capture_paths,
    load_examples,
)
from vsi_onevision_protocol import PROTOCOL_ID


ROLE = "calibration_joint_control_variate_support_state_capacity"
PRIMARY_METHODS = (
    "exact_only_mass",
    "state_only",
    "independent_mass_correction",
    "joint_residual_correction",
)
DIAGNOSTIC_METHODS = (
    "independent_residual_correction",
    "joint_mass_correction",
)
ALL_METHODS = PRIMARY_METHODS + DIAGNOSTIC_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--exact-fraction", type=float, default=0.25)
    parser.add_argument("--feature-width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--greedy-round-size", type=int, default=14)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:2")
    return parser.parse_args()


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu", copy=True)
        for key, value in model.state_dict().items()
    }


def selected_page_count(pages: dict[str, torch.Tensor], exact_fraction: float) -> int:
    return round(pages["exact_z"].shape[-1] * exact_fraction)


def support_mask(
    *,
    mode: str,
    pages: dict[str, torch.Tensor],
    exact_visual_output: torch.Tensor,
    exact_fraction: float,
    greedy_round_size: int,
) -> torch.Tensor:
    selected_pages = selected_page_count(pages, exact_fraction)
    if mode == "mass":
        indices = select_mass_pages(pages["exact_z"], selected_pages)
        return page_mask(indices, pages["exact_z"].shape[-1])
    if mode == "residual":
        with torch.no_grad():
            return select_residual_pages(
                pages,
                exact_visual_output,
                selected_pages=selected_pages,
                round_size=greedy_round_size,
            )
    raise ValueError(f"unknown support mode: {mode}")


def corrected_visual_z(
    pages: dict[str, torch.Tensor], mask: torch.Tensor
) -> torch.Tensor:
    return pages["approximate_z"].sum(dim=2) + (
        (pages["exact_z"] - pages["approximate_z"]) * mask
    ).sum(dim=2)


def relative_square_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    numerator = (prediction - target).square().sum(dim=-1)
    denominator = target.square().sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def global_relative_risk(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1e-8)


def training_objective(
    model: PositiveFeatureState,
    batch: dict[str, torch.Tensor],
    *,
    support_mode: str,
    page_size: int,
    exact_fraction: float,
    greedy_round_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pages = contribution_pages(model, batch, page_size=page_size)
    mask = support_mask(
        mode=support_mode,
        pages=pages,
        exact_visual_output=batch["exact_visual_output"],
        exact_fraction=exact_fraction,
        greedy_round_size=greedy_round_size,
    )
    visual, full = corrected_outputs(pages, batch, mask=mask, exact_only=False)
    visual_error = relative_square_error(visual, batch["exact_visual_output"])
    full_error = relative_square_error(full, batch["exact_full_output"])
    visual_z = corrected_visual_z(pages, mask)
    log_mass_error = (
        torch.log(visual_z.clamp_min(1e-8))
        - torch.log(batch["exact_visual_z"].clamp_min(1e-8))
    ).square()
    loss = visual_error.mean() + full_error.mean() + 0.05 * log_mass_error.mean()
    return loss, {
        "visual_error": visual_error,
        "full_error": full_error,
        "log_mass_error": log_mass_error,
    }


def train_arm(
    *,
    model: PositiveFeatureState,
    examples: list[dict[str, torch.Tensor]],
    batch_schedule: np.ndarray,
    support_mode: str,
    args: argparse.Namespace,
    layer_index: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history: list[dict[str, object]] = []
    for step, indices in enumerate(batch_schedule, start=1):
        model.train()
        batch = batch_examples(examples, indices, device=torch.device(args.device))
        loss, metrics = training_objective(
            model,
            batch,
            support_mode=support_mode,
            page_size=args.page_size,
            exact_fraction=args.exact_fraction,
            greedy_round_size=args.greedy_round_size,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("nonfinite support-state gradient")
        optimizer.step()
        if step == 1 or step % args.log_interval == 0:
            history.append(
                {
                    "layer_index": layer_index,
                    "support_mode": support_mode,
                    "step": step,
                    "train_loss": float(loss.item()),
                    "train_visual_error": float(
                        metrics["visual_error"].mean().sqrt().item()
                    ),
                    "train_full_error": float(
                        metrics["full_error"].mean().sqrt().item()
                    ),
                    "train_log_mass_error": float(
                        metrics["log_mass_error"].mean().sqrt().item()
                    ),
                    "gradient_norm": float(gradient_norm.item()),
                }
            )
    return cpu_state_dict(model), history


def method_model_and_support(
    method: str,
    *,
    base_model: PositiveFeatureState,
    independent_model: PositiveFeatureState,
    joint_model: PositiveFeatureState,
) -> tuple[PositiveFeatureState, str | None, bool]:
    if method == "exact_only_mass":
        return base_model, "mass", True
    if method == "state_only":
        return base_model, None, False
    if method == "independent_mass_correction":
        return independent_model, "mass", False
    if method == "joint_residual_correction":
        return joint_model, "residual", False
    if method == "independent_residual_correction":
        return independent_model, "residual", False
    if method == "joint_mass_correction":
        return joint_model, "mass", False
    raise ValueError(f"unknown method: {method}")


def evaluate(
    *,
    examples: list[dict[str, torch.Tensor]],
    layer_index: int,
    base_model: PositiveFeatureState,
    independent_model: PositiveFeatureState,
    joint_model: PositiveFeatureState,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in (base_model, independent_model, joint_model):
        model.eval()
    with torch.no_grad():
        for index in range(len(examples)):
            batch = batch_examples(
                examples, np.asarray([index]), device=torch.device(args.device)
            )
            for method in ALL_METHODS:
                model, mode, exact_only = method_model_and_support(
                    method,
                    base_model=base_model,
                    independent_model=independent_model,
                    joint_model=joint_model,
                )
                pages = contribution_pages(model, batch, page_size=args.page_size)
                mask = None
                if mode is not None:
                    mask = support_mask(
                        mode=mode,
                        pages=pages,
                        exact_visual_output=batch["exact_visual_output"],
                        exact_fraction=args.exact_fraction,
                        greedy_round_size=args.greedy_round_size,
                    )
                visual, full = corrected_outputs(
                    pages, batch, mask=mask, exact_only=exact_only
                )
                visual_risk = global_relative_risk(visual, batch["exact_visual_output"])
                full_risk = global_relative_risk(full, batch["exact_full_output"])
                rows.append(
                    {
                        "split": "development",
                        "position": index + 73,
                        "layer_index": layer_index,
                        "method": method,
                        "visual_relative_l2": float(visual_risk.sqrt().item()),
                        "visual_risk": float(visual_risk.item()),
                        "full_relative_l2": float(full_risk.sqrt().item()),
                        "full_risk": float(full_risk.item()),
                    }
                )
    return rows


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


def paired_bootstrap(
    rows: list[dict[str, object]],
    *,
    baseline_method: str,
    candidate_method: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    method_rows: dict[str, dict[tuple[int, int], float]] = {}
    for method in (baseline_method, candidate_method):
        method_rows[method] = {
            (int(row["position"]), int(row["layer_index"])): float(row["visual_risk"])
            for row in rows
            if row["method"] == method
        }
    if method_rows[baseline_method].keys() != method_rows[candidate_method].keys():
        raise ValueError("paired methods do not share identities")
    positions = sorted({key[0] for key in method_rows[baseline_method]})
    layers = sorted({key[1] for key in method_rows[baseline_method]})
    baseline = np.asarray(
        [
            np.mean(
                [method_rows[baseline_method][(position, layer)] for layer in layers]
            )
            for position in positions
        ]
    )
    candidate = np.asarray(
        [
            np.mean(
                [method_rows[candidate_method][(position, layer)] for layer in layers]
            )
            for position in positions
        ]
    )
    point = 1.0 - candidate.mean() / baseline.mean()
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(positions), size=len(positions))
        samples[index] = 1.0 - candidate[selected].mean() / baseline[selected].mean()
    return {
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "relative_risk_improvement": float(point),
        "bootstrap_lower_95": float(np.quantile(samples, 0.025)),
        "bootstrap_upper_95": float(np.quantile(samples, 0.975)),
    }


def classify(
    *,
    joint: dict[str, float | int],
    versus_independent: dict[str, float],
    versus_support_only: dict[str, float],
    versus_state_only: dict[str, float],
    active_state_ratio: float,
) -> str:
    if (
        int(joint["cell_count"]) == 72
        and float(joint["visual_mean"]) <= 0.005
        and float(joint["visual_p95"]) <= 0.01
        and float(joint["visual_worst"]) <= 0.02
        and versus_independent["bootstrap_lower_95"] >= 0.25
        and versus_support_only["bootstrap_lower_95"] > 0.0
        and versus_state_only["bootstrap_lower_95"] > 0.0
        and active_state_ratio >= 3.0
    ):
        return "JOINT_SUPPORT_STATE_CAPACITY_GO"
    return "NO_JOINT_SUPPORT_STATE_CAPACITY"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty joint support-state rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if (
        args.page_size != 4
        or args.exact_fraction != 0.25
        or args.feature_width != 32
        or args.steps != 1000
        or args.batch_size != 2
        or args.learning_rate != 3e-4
        or args.log_interval != 100
        or args.greedy_round_size != 14
        or args.bootstrap_repetitions != 10000
        or args.seed != 20260901
    ):
        raise ValueError("registered joint support-state identity changed")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("joint support-state output must be empty")
    paths = capture_paths(args.capture_dir)
    capture_summary = json.loads(
        (args.capture_dir / "summary.json").read_text(encoding="utf-8")
    )
    if capture_summary["protocol_id"] != PROTOCOL_ID:
        raise ValueError("capture protocol identity mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    batch_schedule = rng.integers(
        0, len(TRAIN_POSITIONS), size=(args.steps, args.batch_size)
    )
    all_rows: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for layer_index in LAYERS:
        train_examples = load_examples(
            paths, positions=TRAIN_POSITIONS, layer_index=layer_index
        )
        development_examples = load_examples(
            paths, positions=DEVELOPMENT_POSITIONS, layer_index=layer_index
        )
        head_count, head_dim = train_examples[0]["query_scaled"].shape
        checkpoint = torch.load(
            args.checkpoint_dir / f"layer_{layer_index:02d}_feature_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint["layer_index"] != layer_index
            or checkpoint["feature_width"] != 32
        ):
            raise ValueError("base additive-state checkpoint identity mismatch")

        def make_model() -> PositiveFeatureState:
            model = PositiveFeatureState(
                head_count=head_count,
                head_dim=head_dim,
                feature_width=args.feature_width,
            ).to(torch.device(args.device))
            model.load_state_dict(checkpoint["best_state"])
            return model

        base_model = make_model()
        independent_model = make_model()
        joint_model = make_model()
        independent_state, independent_history = train_arm(
            model=independent_model,
            examples=train_examples,
            batch_schedule=batch_schedule,
            support_mode="mass",
            args=args,
            layer_index=layer_index,
        )
        joint_state, joint_history = train_arm(
            model=joint_model,
            examples=train_examples,
            batch_schedule=batch_schedule,
            support_mode="residual",
            args=args,
            layer_index=layer_index,
        )
        history.extend(independent_history)
        history.extend(joint_history)
        all_rows.extend(
            evaluate(
                examples=development_examples,
                layer_index=layer_index,
                base_model=base_model,
                independent_model=independent_model,
                joint_model=joint_model,
                args=args,
            )
        )
        torch.save(
            {
                "layer_index": layer_index,
                "feature_width": args.feature_width,
                "independent_state": independent_state,
                "joint_state": joint_state,
            },
            args.out_dir / f"layer_{layer_index:02d}_joint_support_state.pt",
        )

    summaries = {
        method: summarize([row for row in all_rows if row["method"] == method])
        for method in ALL_METHODS
    }
    versus_independent = paired_bootstrap(
        all_rows,
        baseline_method="independent_mass_correction",
        candidate_method="joint_residual_correction",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    versus_support_only = paired_bootstrap(
        all_rows,
        baseline_method="independent_residual_correction",
        candidate_method="joint_residual_correction",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 1,
    )
    versus_state_only = paired_bootstrap(
        all_rows,
        baseline_method="joint_mass_correction",
        candidate_method="joint_residual_correction",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 2,
    )
    token_count = 8 * 196
    head_dim = 128
    dense_state = 2 * token_count * head_dim
    active_state = args.exact_fraction * dense_state + args.feature_width * (
        head_dim + 1
    )
    active_state_ratio = dense_state / active_state
    decision = classify(
        joint=summaries["joint_residual_correction"],
        versus_independent=versus_independent,
        versus_support_only=versus_support_only,
        versus_state_only=versus_state_only,
        active_state_ratio=active_state_ratio,
    )
    write_csv(args.out_dir / "development_rows.csv", all_rows)
    write_csv(args.out_dir / "training_history.csv", history)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "primary_methods": list(PRIMARY_METHODS),
        "diagnostic_methods": list(DIAGNOSTIC_METHODS),
        "method_summaries": summaries,
        "joint_vs_independent": versus_independent,
        "joint_vs_support_only_change": versus_support_only,
        "joint_vs_state_only_change": versus_state_only,
        "page_size": args.page_size,
        "exact_fraction": args.exact_fraction,
        "feature_width": args.feature_width,
        "steps_per_arm_per_layer": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "train_positions": [1, 72],
        "development_positions": [73, 96],
        "confirmation_positions_unread": [97, 120],
        "layers": list(LAYERS),
        "analytic_active_state_ratio": active_state_ratio,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Calibration-trained state with target-visible regular-page support on "
            "development. This is a function-class ceiling, not a deployable router, "
            "task, confirmation, formal, latency, or speed result."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
