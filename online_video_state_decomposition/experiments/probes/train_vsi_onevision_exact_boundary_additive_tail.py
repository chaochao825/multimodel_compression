from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from probe_vsi_onevision_query_fixed_positive_gaussian_measure import LAYERS
from train_vsi_onevision_additive_nz_feature_state import (
    DEVELOPMENT_POSITIONS,
    TRAIN_POSITIONS,
    PositiveFeatureState,
    batch_examples,
    capture_paths,
    cpu_state_dict,
    fit_normalization,
    load_examples,
    relative_square_error,
)
from vsi_onevision_protocol import PROTOCOL_ID


SELECTORS = ("mass_topk", "effect_topk")
ROLE = "calibration_development_exact_boundary_additive_tail_oracle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exact-fraction", type=float, default=0.25)
    parser.add_argument("--feature-width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def gather_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        tensor,
        dim=2,
        index=indices.unsqueeze(-1).expand(*indices.shape, tensor.shape[-1]),
    )


def oracle_partition(
    batch: dict[str, torch.Tensor],
    *,
    selector: str,
    exact_fraction: float,
) -> dict[str, torch.Tensor]:
    query = batch["query_scaled"]
    key = batch["visual_key"]
    value = batch["visual_value"]
    scores = torch.einsum("bhd,bhnd->bhn", query, key)
    local_exp = torch.exp(scores - scores.max(dim=2, keepdim=True).values)
    exp_scale = batch["exact_visual_z"] / local_exp.sum(dim=2).clamp_min(1e-8)
    exact_exp = local_exp * exp_scale.unsqueeze(-1)

    token_count = key.shape[2]
    selected_count = int(round(token_count * exact_fraction))
    if selected_count <= 0 or selected_count >= token_count:
        raise ValueError("exact boundary must retain a strict token subset")
    if selector == "mass_topk":
        priority = scores
    elif selector == "effect_topk":
        centered_value = value - batch["exact_visual_output"].unsqueeze(2)
        priority = exact_exp * torch.linalg.vector_norm(centered_value, dim=-1)
    else:
        raise ValueError(f"unknown exact-boundary selector: {selector}")

    selected_indices = torch.topk(
        priority, k=selected_count, dim=2, sorted=False
    ).indices
    tail_mask = torch.ones_like(priority, dtype=torch.bool)
    tail_mask.scatter_(2, selected_indices, False)
    all_indices = torch.arange(token_count, device=key.device).reshape(1, 1, -1)
    tail_indices = all_indices.expand_as(tail_mask)[tail_mask].reshape(
        *tail_mask.shape[:2], token_count - selected_count
    )

    selected_exp = torch.gather(exact_exp, 2, selected_indices)
    tail_exp = torch.gather(exact_exp, 2, tail_indices)
    selected_value = gather_tokens(value, selected_indices)
    exact_selected_z = selected_exp.sum(dim=2)
    exact_selected_n = torch.einsum("bhk,bhkd->bhd", selected_exp, selected_value)
    exact_tail_z = tail_exp.sum(dim=2)
    if torch.any(exact_tail_z <= 0):
        raise RuntimeError("oracle boundary left non-positive exact tail mass")
    return {
        "tail_key": gather_tokens(key, tail_indices),
        "tail_value": gather_tokens(value, tail_indices),
        "exact_selected_z": exact_selected_z,
        "exact_selected_n": exact_selected_n,
        "exact_tail_z": exact_tail_z,
    }


def query_feature(model: PositiveFeatureState, query: torch.Tensor) -> torch.Tensor:
    normalized_query = query / model.query_rms[None, :, :]
    return model.positive(
        torch.einsum("bhd,hdr->bhr", normalized_query, model.query_weight)
        + model.query_bias[None, :, :]
    )


def hybrid_forward(
    model: PositiveFeatureState,
    batch: dict[str, torch.Tensor],
    *,
    selector: str,
    exact_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    partition = oracle_partition(
        batch, selector=selector, exact_fraction=exact_fraction
    )
    state, normalizer = model.visual_state(
        partition["tail_key"], partition["tail_value"]
    )
    feature = query_feature(model, batch["query_scaled"])
    tail_z = torch.einsum("bhr,bhr->bh", feature, normalizer)
    tail_n = torch.einsum("bhr,bhrd->bhd", feature, state)
    scale = torch.exp(model.log_visual_scale).unsqueeze(0)
    tail_z = tail_z * scale
    tail_n = tail_n * scale.unsqueeze(-1)

    visual_z = partition["exact_selected_z"] + tail_z
    visual_n = partition["exact_selected_n"] + tail_n
    visual_output = visual_n / visual_z.unsqueeze(-1).clamp_min(1e-8)
    full_output = (visual_n + batch["nonvisual_n"]) / (
        visual_z + batch["nonvisual_z"]
    ).unsqueeze(-1).clamp_min(1e-8)
    return visual_output, full_output, tail_z, partition["exact_tail_z"]


def exact_only_forward(
    batch: dict[str, torch.Tensor],
    *,
    selector: str,
    exact_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    partition = oracle_partition(
        batch, selector=selector, exact_fraction=exact_fraction
    )
    visual = partition["exact_selected_n"] / partition["exact_selected_z"].unsqueeze(
        -1
    ).clamp_min(1e-8)
    full = (partition["exact_selected_n"] + batch["nonvisual_n"]) / (
        partition["exact_selected_z"] + batch["nonvisual_z"]
    ).unsqueeze(-1).clamp_min(1e-8)
    return visual, full


def initialize_tail_scale(
    model: PositiveFeatureState,
    examples: list[dict[str, torch.Tensor]],
    *,
    selector: str,
    exact_fraction: float,
    device: torch.device,
) -> None:
    differences = []
    with torch.no_grad():
        for start in range(0, len(examples), 2):
            indices = np.arange(start, min(start + 2, len(examples)))
            batch = batch_examples(examples, indices, device=device)
            partition = oracle_partition(
                batch, selector=selector, exact_fraction=exact_fraction
            )
            _, normalizer = model.visual_state(
                partition["tail_key"], partition["tail_value"]
            )
            raw_z = torch.einsum(
                "bhr,bhr->bh",
                query_feature(model, batch["query_scaled"]),
                normalizer,
            )
            differences.append(
                torch.log(partition["exact_tail_z"].clamp_min(1e-8))
                - torch.log(raw_z.clamp_min(1e-8))
            )
        model.log_visual_scale.copy_(torch.cat(differences).mean(dim=0))


def objective(
    model: PositiveFeatureState,
    batch: dict[str, torch.Tensor],
    *,
    selector: str,
    exact_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visual, full, tail_z, exact_tail_z = hybrid_forward(
        model, batch, selector=selector, exact_fraction=exact_fraction
    )
    visual_error = relative_square_error(visual, batch["exact_visual_output"])
    full_error = relative_square_error(full, batch["exact_full_output"])
    log_tail_mass_error = (
        torch.log(tail_z.clamp_min(1e-8)) - torch.log(exact_tail_z.clamp_min(1e-8))
    ).square()
    loss = visual_error.mean() + full_error.mean() + 0.05 * log_tail_mass_error.mean()
    return loss, {
        "visual_error": visual_error,
        "full_error": full_error,
        "log_tail_mass_error": log_tail_mass_error,
    }


def evaluate(
    model: PositiveFeatureState | None,
    examples: list[dict[str, torch.Tensor]],
    *,
    layer_index: int,
    selector: str,
    exact_fraction: float,
    method: str,
    device: torch.device,
) -> list[dict[str, object]]:
    rows = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for index in range(len(examples)):
            batch = batch_examples(examples, np.asarray([index]), device=device)
            if method == "exact_only":
                visual, full = exact_only_forward(
                    batch, selector=selector, exact_fraction=exact_fraction
                )
                log_tail_mass = math.nan
            elif method == "learned_tail" and model is not None:
                visual, full, tail_z, exact_tail_z = hybrid_forward(
                    model, batch, selector=selector, exact_fraction=exact_fraction
                )
                log_tail_mass = float(
                    (
                        torch.log(tail_z.clamp_min(1e-8))
                        - torch.log(exact_tail_z.clamp_min(1e-8))
                    )
                    .abs()
                    .mean()
                    .item()
                )
            else:
                raise ValueError("invalid exact-boundary evaluation method")
            head_visual = relative_square_error(
                visual, batch["exact_visual_output"]
            ).sqrt()
            visual_relative = torch.linalg.vector_norm(
                visual - batch["exact_visual_output"]
            ) / torch.linalg.vector_norm(batch["exact_visual_output"]).clamp_min(1e-8)
            full_relative = torch.linalg.vector_norm(
                full - batch["exact_full_output"]
            ) / torch.linalg.vector_norm(batch["exact_full_output"]).clamp_min(1e-8)
            rows.append(
                {
                    "split": "development",
                    "position": index + 73,
                    "layer_index": layer_index,
                    "selector": selector,
                    "method": method,
                    "exact_fraction": exact_fraction,
                    "visual_relative_l2": float(visual_relative.item()),
                    "visual_worst_head_relative_l2": float(head_visual.max().item()),
                    "full_relative_l2": float(full_relative.item()),
                    "log_tail_mass_absolute_mean": log_tail_mass,
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    visual = np.asarray([float(row["visual_relative_l2"]) for row in rows])
    full = np.asarray([float(row["full_relative_l2"]) for row in rows])
    return {
        "cell_count": len(rows),
        "visual_mean": float(visual.mean()),
        "visual_p95": float(np.quantile(visual, 0.95)),
        "visual_worst": float(visual.max()),
        "visual_worst_head": max(
            float(row["visual_worst_head_relative_l2"]) for row in rows
        ),
        "full_mean": float(full.mean()),
        "full_p95": float(np.quantile(full, 0.95)),
        "full_worst": float(full.max()),
    }


def selector_decision(
    *,
    exact_only: dict[str, float | int],
    learned: dict[str, float | int],
    active_state_ratio: float,
) -> str:
    improvement = 1.0 - float(learned["visual_mean"]) / float(exact_only["visual_mean"])
    common = (
        int(learned["cell_count"]) == 72
        and improvement >= 0.25
        and active_state_ratio >= 2.0
    )
    if (
        common
        and float(learned["visual_mean"]) <= 0.005
        and float(learned["visual_p95"]) <= 0.01
        and float(learned["visual_worst"]) <= 0.02
        and float(learned["full_mean"]) <= 0.0025
        and float(learned["full_p95"]) <= 0.005
    ):
        return "BOUNDARY_ADDITIVE_TAIL_ORACLE_GO"
    if (
        common
        and float(learned["visual_mean"]) <= 0.01
        and float(learned["visual_p95"]) <= 0.02
        and float(learned["visual_worst"]) <= 0.05
        and float(learned["full_mean"]) <= 0.005
        and float(learned["full_p95"]) <= 0.01
    ):
        return "BOUNDARY_ADDITIVE_TAIL_CAPACITY_SIGNAL"
    return "NO_BOUNDARY_ADDITIVE_TAIL_PATH"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty exact-boundary rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if (
        args.exact_fraction != 0.25
        or args.feature_width != 32
        or args.steps != 1000
        or args.batch_size != 2
        or args.learning_rate != 1e-3
        or args.evaluation_interval != 100
        or args.seed != 20260830
    ):
        raise ValueError("registered exact-boundary additive-tail identity changed")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("exact-boundary additive-tail output must be empty")
    paths = capture_paths(args.capture_dir)
    capture_summary = json.loads(
        (args.capture_dir / "summary.json").read_text(encoding="utf-8")
    )
    if capture_summary["protocol_id"] != PROTOCOL_ID:
        raise ValueError("capture protocol identity mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    exact_only_rows: list[dict[str, object]] = []
    learned_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for selector in SELECTORS:
        for layer_index in LAYERS:
            train_examples = load_examples(
                paths, positions=TRAIN_POSITIONS, layer_index=layer_index
            )
            development_examples = load_examples(
                paths, positions=DEVELOPMENT_POSITIONS, layer_index=layer_index
            )
            head_count, head_dim = train_examples[0]["query_scaled"].shape
            model = PositiveFeatureState(
                head_count=head_count,
                head_dim=head_dim,
                feature_width=args.feature_width,
            ).to(device)
            fit_normalization(model, train_examples)
            initialize_tail_scale(
                model,
                train_examples,
                selector=selector,
                exact_fraction=args.exact_fraction,
                device=device,
            )
            exact_only_rows.extend(
                evaluate(
                    None,
                    development_examples,
                    layer_index=layer_index,
                    selector=selector,
                    exact_fraction=args.exact_fraction,
                    method="exact_only",
                    device=device,
                )
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=1e-4
            )
            best_visual = math.inf
            best_state = cpu_state_dict(model)
            for step in range(1, args.steps + 1):
                model.train()
                indices = rng.integers(0, len(train_examples), size=args.batch_size)
                batch = batch_examples(train_examples, indices, device=device)
                loss, metrics = objective(
                    model,
                    batch,
                    selector=selector,
                    exact_fraction=args.exact_fraction,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if step == 1 or step % args.evaluation_interval == 0:
                    development_rows = evaluate(
                        model,
                        development_examples,
                        layer_index=layer_index,
                        selector=selector,
                        exact_fraction=args.exact_fraction,
                        method="learned_tail",
                        device=device,
                    )
                    development = summarize(development_rows)
                    history_rows.append(
                        {
                            "selector": selector,
                            "layer_index": layer_index,
                            "step": step,
                            "train_loss": float(loss.item()),
                            "train_visual_error": float(
                                metrics["visual_error"].mean().sqrt().item()
                            ),
                            "development_visual_mean": development["visual_mean"],
                            "development_visual_p95": development["visual_p95"],
                            "development_full_mean": development["full_mean"],
                        }
                    )
                    if float(development["visual_mean"]) < best_visual:
                        best_visual = float(development["visual_mean"])
                        best_state = cpu_state_dict(model)
            model.load_state_dict(best_state)
            learned_rows.extend(
                evaluate(
                    model,
                    development_examples,
                    layer_index=layer_index,
                    selector=selector,
                    exact_fraction=args.exact_fraction,
                    method="learned_tail",
                    device=device,
                )
            )
            torch.save(
                {
                    "selector": selector,
                    "layer_index": layer_index,
                    "feature_width": args.feature_width,
                    "exact_fraction": args.exact_fraction,
                    "best_state": best_state,
                },
                args.out_dir / f"{selector}_layer_{layer_index:02d}.pt",
            )

    token_count = 8 * 196
    head_dim = 128
    dense_state = 2 * token_count * head_dim
    active_state = args.exact_fraction * dense_state + args.feature_width * (
        head_dim + 1
    )
    active_state_ratio = dense_state / active_state
    selector_summaries = {}
    decisions = []
    for selector in SELECTORS:
        exact_only = summarize(
            [row for row in exact_only_rows if row["selector"] == selector]
        )
        learned = summarize(
            [row for row in learned_rows if row["selector"] == selector]
        )
        decision = selector_decision(
            exact_only=exact_only,
            learned=learned,
            active_state_ratio=active_state_ratio,
        )
        decisions.append(decision)
        selector_summaries[selector] = {
            "decision": decision,
            "exact_only": exact_only,
            "learned_tail": learned,
            "visual_mean_relative_improvement": 1.0
            - float(learned["visual_mean"]) / float(exact_only["visual_mean"]),
        }
    if "BOUNDARY_ADDITIVE_TAIL_ORACLE_GO" in decisions:
        decision = "BOUNDARY_ADDITIVE_TAIL_ORACLE_GO"
    elif "BOUNDARY_ADDITIVE_TAIL_CAPACITY_SIGNAL" in decisions:
        decision = "BOUNDARY_ADDITIVE_TAIL_CAPACITY_SIGNAL"
    else:
        decision = "NO_BOUNDARY_ADDITIVE_TAIL_PATH"

    write_csv(args.out_dir / "exact_only_development_rows.csv", exact_only_rows)
    write_csv(args.out_dir / "learned_tail_development_rows.csv", learned_rows)
    write_csv(args.out_dir / "training_history.csv", history_rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "selectors": list(SELECTORS),
        "selector_summaries": selector_summaries,
        "exact_fraction": args.exact_fraction,
        "feature_width": args.feature_width,
        "steps": args.steps,
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
            "Calibration-development oracle support capacity only. Both selectors "
            "read exact current-query statistics; confirmation, official selection/"
            "formal, deployable support generation, task behavior, writer cost, and "
            "latency remain untested."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
