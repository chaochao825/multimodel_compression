from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mvbench_onevision_utils import (
    build_prompt_batch,
    decode_video_frames,
    first_token_logits_from_features,
    load_onevision_model,
    uniform_frame_indices,
)
from probe_vsi_onevision_cmrq_stage_b import feature_path_for_sample
from probe_vsi_onevision_query_fixed_measure_remainder import SelectedLayerCapture
from probe_vsi_onevision_query_fixed_positive_gaussian_measure import (
    LAYERS,
    GaussianComponents,
    build_gaussian_components,
    hierarchical_group_offsets,
)
from probe_vsi_onevision_reader_risk_stage_a import select_calibration_questions
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    set_language_attention_eager,
)
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


CLUSTER_FAMILIES = ("key", "key_value")
PROTOTYPE_COUNTS = (32, 64, 128)
SELECTORS = ("prototype_mass", "oracle_local")
ACTIVE_TOKEN_BUDGET = 392
KMEANS_ITERATIONS = 4
ROLE = "exposed_query_fixed_prototype_measure"


@dataclass
class PrototypeState:
    exact_visual_output: torch.Tensor
    exact_full_output: torch.Tensor
    exact_cluster_z: torch.Tensor
    exact_cluster_n: torch.Tensor
    coarse_cluster_z: torch.Tensor
    coarse_cluster_n: torch.Tensor
    prototype_mass_priority: torch.Tensor
    oracle_priority: torch.Tensor
    cluster_counts: torch.Tensor
    nonvisual_z: torch.Tensor
    nonvisual_n: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gaussian-summary", type=Path, required=True)
    parser.add_argument("--progressive-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--residual-greedy-diagnostic", action="store_true")
    parser.add_argument("--reverse-greedy-diagnostic", action="store_true")
    return parser.parse_args()


def clustering_features(
    key: torch.Tensor, value: torch.Tensor, *, family: str
) -> torch.Tensor:
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("prototype K/V must share [heads, tokens, dim]")
    key_scale = key.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    normalized_key = key / key_scale
    if family == "key":
        return normalized_key
    if family == "key_value":
        value_scale = (
            value.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        )
        return torch.cat((normalized_key, value / value_scale), dim=-1)
    raise ValueError(f"unregistered cluster family: {family}")


def deterministic_kmeans(
    features: torch.Tensor, *, cluster_count: int, iterations: int
) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("k-means features must be [heads, tokens, dim]")
    head_count, token_count, feature_dim = features.shape
    if not 1 < cluster_count < token_count:
        raise ValueError("cluster count must lie inside the token count")
    if iterations <= 0:
        raise ValueError("k-means iterations must be positive")

    squared_norm = features.square().sum(dim=-1)
    initial = torch.empty(
        (head_count, cluster_count), device=features.device, dtype=torch.long
    )
    initial[:, 0] = squared_norm.argmax(dim=1)
    chosen = torch.zeros(
        (head_count, token_count), device=features.device, dtype=torch.bool
    )
    chosen.scatter_(1, initial[:, :1], True)
    first = features.gather(
        1,
        initial[:, :1].unsqueeze(-1).expand(head_count, 1, feature_dim),
    )
    minimum_distance = (
        squared_norm
        + first.square().sum(dim=-1)
        - 2.0 * torch.einsum("hnd,hpd->hnp", features, first).squeeze(-1)
    )
    for index in range(1, cluster_count):
        candidate_distance = torch.where(
            chosen,
            torch.full_like(minimum_distance, -torch.inf),
            minimum_distance,
        )
        initial[:, index] = candidate_distance.argmax(dim=1)
        chosen.scatter_(1, initial[:, index : index + 1], True)
        center = features.gather(
            1,
            initial[:, index : index + 1]
            .unsqueeze(-1)
            .expand(head_count, 1, feature_dim),
        )
        distance = (
            squared_norm
            + center.square().sum(dim=-1)
            - 2.0 * torch.einsum("hnd,hpd->hnp", features, center).squeeze(-1)
        )
        minimum_distance = torch.minimum(minimum_distance, distance)
    centers = features.gather(
        1, initial.unsqueeze(-1).expand(head_count, cluster_count, feature_dim)
    )

    assignment = torch.zeros(
        (head_count, token_count), device=features.device, dtype=torch.long
    )
    for _ in range(iterations):
        distance = (
            features.square().sum(dim=-1, keepdim=True)
            + centers.square().sum(dim=-1).unsqueeze(1)
            - 2.0 * torch.einsum("hnd,hpd->hnp", features, centers)
        )
        assignment = distance.argmin(dim=-1)
        membership = F.one_hot(assignment, num_classes=cluster_count).to(features.dtype)
        counts = membership.sum(dim=1)
        updated = torch.einsum("hnp,hnd->hpd", membership, features) / counts.clamp_min(
            1.0
        ).unsqueeze(-1)
        centers = torch.where((counts > 0).unsqueeze(-1), updated, centers)
    return assignment


def build_prototype_state(
    components: GaussianComponents,
    *,
    family: str,
    prototype_count: int,
) -> PrototypeState:
    member_key = components.member_key.flatten(1, 2)
    member_value = components.member_value.flatten(1, 2)
    features = clustering_features(member_key, member_value, family=family)
    assignment = deterministic_kmeans(
        features,
        cluster_count=prototype_count,
        iterations=KMEANS_ITERATIONS,
    )
    membership = F.one_hot(assignment, num_classes=prototype_count).float()
    counts = membership.sum(dim=1)
    mean_key = torch.einsum("hnp,hnd->hpd", membership, member_key) / counts.clamp_min(
        1.0
    ).unsqueeze(-1)
    mean_value = torch.einsum(
        "hnp,hnd->hpd", membership, member_value
    ) / counts.clamp_min(1.0).unsqueeze(-1)

    scores = torch.einsum("hd,hnd->hn", components.query_scaled, member_key)
    token_z = torch.exp(scores - components.maximum)
    exact_cluster_z = torch.einsum("hnp,hn->hp", membership, token_z)
    exact_cluster_n = torch.einsum("hnp,hn,hnd->hpd", membership, token_z, member_value)
    prototype_scores = torch.einsum("hd,hpd->hp", components.query_scaled, mean_key)
    coarse_cluster_z = counts * torch.exp(prototype_scores - components.maximum)
    coarse_cluster_n = coarse_cluster_z.unsqueeze(-1) * mean_value

    exact_visual_z = exact_cluster_z.sum(dim=1)
    exact_z_normalized = exact_cluster_z / exact_visual_z.unsqueeze(-1)
    exact_n_normalized = exact_cluster_n / exact_visual_z[:, None, None]
    coarse_z_normalized = coarse_cluster_z / exact_visual_z.unsqueeze(-1)
    coarse_n_normalized = coarse_cluster_n / exact_visual_z[:, None, None]
    local_defect = (
        exact_n_normalized
        - coarse_n_normalized
        - components.exact_visual_output[:, None, :]
        * (exact_z_normalized - coarse_z_normalized).unsqueeze(-1)
    )
    return PrototypeState(
        exact_visual_output=components.exact_visual_output,
        exact_full_output=components.exact_full_output,
        exact_cluster_z=exact_cluster_z,
        exact_cluster_n=exact_cluster_n,
        coarse_cluster_z=coarse_cluster_z,
        coarse_cluster_n=coarse_cluster_n,
        prototype_mass_priority=coarse_cluster_z,
        oracle_priority=torch.linalg.vector_norm(local_defect, dim=-1),
        cluster_counts=counts,
        nonvisual_z=components.nonvisual_z,
        nonvisual_n=components.nonvisual_n,
    )


def select_under_budget(
    priority: torch.Tensor,
    cluster_counts: torch.Tensor,
    *,
    active_token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if priority.shape != cluster_counts.shape:
        raise ValueError("priority and cluster counts must share shape")
    nonempty = cluster_counts > 0
    base_cost = nonempty.sum(dim=1)
    if bool((base_cost > active_token_budget).any().item()):
        raise ValueError("prototype state exceeds the active-token budget")
    incremental = (cluster_counts - 1.0).clamp_min(0.0)
    score = torch.where(
        nonempty,
        priority / incremental.clamp_min(1.0),
        torch.full_like(priority, -torch.inf),
    )
    order = torch.argsort(score, dim=1, descending=True, stable=True)
    selected = torch.zeros_like(nonempty)
    active = base_cost.clone()
    for position in range(order.shape[1]):
        index = order[:, position]
        cost = incremental.gather(1, index.unsqueeze(-1)).squeeze(-1).long()
        valid = nonempty.gather(1, index.unsqueeze(-1)).squeeze(-1)
        accept = valid & (active + cost <= active_token_budget)
        selected.scatter_(1, index.unsqueeze(-1), accept.unsqueeze(-1))
        active = active + torch.where(accept, cost, torch.zeros_like(cost))
    return selected, active


def select_residual_greedy_under_budget(
    state: PrototypeState,
    *,
    active_token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nonempty = state.cluster_counts > 0
    active = nonempty.sum(dim=1)
    if bool((active > active_token_budget).any().item()):
        raise ValueError("prototype state exceeds the active-token budget")

    incremental = (state.cluster_counts - 1.0).clamp_min(0.0).long()
    delta_z = state.exact_cluster_z - state.coarse_cluster_z
    delta_n = state.exact_cluster_n - state.coarse_cluster_n
    current_z = state.coarse_cluster_z.sum(dim=1)
    current_n = state.coarse_cluster_n.sum(dim=1)
    selected = torch.zeros_like(nonempty)

    for _ in range(state.cluster_counts.shape[1]):
        current_output = current_n / current_z.unsqueeze(-1)
        current_error = torch.linalg.vector_norm(
            current_output - state.exact_visual_output, dim=-1
        )
        candidate_z = current_z.unsqueeze(1) + delta_z
        candidate_n = current_n.unsqueeze(1) + delta_n
        candidate_output = candidate_n / candidate_z.unsqueeze(-1)
        candidate_error = torch.linalg.vector_norm(
            candidate_output - state.exact_visual_output.unsqueeze(1), dim=-1
        )
        fits = (
            nonempty
            & ~selected
            & incremental.gt(0)
            & (active.unsqueeze(1) + incremental <= active_token_budget)
        )
        score = torch.where(
            fits,
            (current_error.unsqueeze(1) - candidate_error) / incremental.clamp_min(1),
            torch.full_like(candidate_error, -torch.inf),
        )
        best_score, best_index = score.max(dim=1)
        accept = torch.isfinite(best_score) & best_score.gt(0)
        if not bool(accept.any().item()):
            break

        head_index = torch.nonzero(accept, as_tuple=False).flatten()
        cluster_index = best_index.index_select(0, head_index)
        selected[head_index, cluster_index] = True
        current_z[head_index] += delta_z[head_index, cluster_index]
        current_n[head_index] += delta_n[head_index, cluster_index]
        active[head_index] += incremental[head_index, cluster_index]
    return selected, active


def select_reverse_greedy_under_budget(
    state: PrototypeState,
    *,
    active_token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nonempty = state.cluster_counts > 0
    selected = nonempty.clone()
    active = state.cluster_counts.sum(dim=1).long()
    incremental = (state.cluster_counts - 1.0).clamp_min(0.0).long()
    delta_z = state.exact_cluster_z - state.coarse_cluster_z
    delta_n = state.exact_cluster_n - state.coarse_cluster_n
    current_z = state.exact_cluster_z.sum(dim=1)
    current_n = state.exact_cluster_n.sum(dim=1)

    for _ in range(state.cluster_counts.shape[1]):
        needs_reduction = active > active_token_budget
        if not bool(needs_reduction.any().item()):
            break
        current_output = current_n / current_z.unsqueeze(-1)
        current_error = torch.linalg.vector_norm(
            current_output - state.exact_visual_output, dim=-1
        )
        candidate_z = current_z.unsqueeze(1) - delta_z
        candidate_n = current_n.unsqueeze(1) - delta_n
        candidate_output = candidate_n / candidate_z.unsqueeze(-1)
        candidate_error = torch.linalg.vector_norm(
            candidate_output - state.exact_visual_output.unsqueeze(1), dim=-1
        )
        removable = selected & incremental.gt(0) & needs_reduction.unsqueeze(1)
        damage_per_saved_token = torch.where(
            removable,
            (candidate_error - current_error.unsqueeze(1)) / incremental.clamp_min(1),
            torch.full_like(candidate_error, torch.inf),
        )
        best_score, best_index = damage_per_saved_token.min(dim=1)
        accept = torch.isfinite(best_score) & needs_reduction
        if not bool(accept.any().item()):
            raise RuntimeError(
                "reverse greedy could not satisfy the active-token budget"
            )

        head_index = torch.nonzero(accept, as_tuple=False).flatten()
        cluster_index = best_index.index_select(0, head_index)
        selected[head_index, cluster_index] = False
        current_z[head_index] -= delta_z[head_index, cluster_index]
        current_n[head_index] -= delta_n[head_index, cluster_index]
        active[head_index] -= incremental[head_index, cluster_index]
    if bool((active > active_token_budget).any().item()):
        raise RuntimeError("reverse greedy exceeded its registered iteration bound")
    return selected, active


def evaluate_prototype_state(
    state: PrototypeState,
    selected: torch.Tensor,
    active_tokens: torch.Tensor,
) -> dict[str, float]:
    visual_z = torch.where(selected, state.exact_cluster_z, state.coarse_cluster_z).sum(
        dim=1
    )
    visual_n = torch.where(
        selected.unsqueeze(-1), state.exact_cluster_n, state.coarse_cluster_n
    ).sum(dim=1)
    visual_output = visual_n / visual_z.unsqueeze(-1)
    full_output = (visual_n + state.nonvisual_n) / (
        visual_z + state.nonvisual_z
    ).unsqueeze(-1)
    head_error = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output, dim=-1
    ) / torch.linalg.vector_norm(state.exact_visual_output, dim=-1).clamp_min(1e-12)
    visual_relative = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output
    ) / torch.linalg.vector_norm(state.exact_visual_output).clamp_min(1e-12)
    full_relative = torch.linalg.vector_norm(
        full_output - state.exact_full_output
    ) / torch.linalg.vector_norm(state.exact_full_output).clamp_min(1e-12)
    exact_tokens = (selected * state.cluster_counts).sum(dim=1)
    exact_mass = (selected * state.exact_cluster_z).sum(
        dim=1
    ) / state.exact_cluster_z.sum(dim=1)
    dense_tokens = state.cluster_counts.sum(dim=1)
    return {
        "visual_relative_l2": float(visual_relative.item()),
        "visual_worst_head_relative_l2": float(head_error.max().item()),
        "full_relative_l2": float(full_relative.item()),
        "active_token_count_mean": float(active_tokens.float().mean().item()),
        "active_token_count_max": int(active_tokens.max().item()),
        "active_read_ratio": float(
            (dense_tokens / active_tokens.clamp_min(1)).float().mean().item()
        ),
        "exact_token_fraction_mean": float(
            (exact_tokens / dense_tokens).float().mean().item()
        ),
        "selected_visual_mass_mean": float(exact_mass.mean().item()),
    }


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["cluster_family"]),
                int(row["prototype_count"]),
                str(row["selector"]),
            )
            for row in rows
        }
    )
    summaries = []
    for family, prototype_count, selector in keys:
        selected = [
            row
            for row in rows
            if row["cluster_family"] == family
            and row["prototype_count"] == prototype_count
            and row["selector"] == selector
        ]
        visual = np.asarray([float(row["visual_relative_l2"]) for row in selected])
        full = np.asarray([float(row["full_relative_l2"]) for row in selected])
        summaries.append(
            {
                "cluster_family": family,
                "prototype_count": prototype_count,
                "selector": selector,
                "cell_count": len(selected),
                "active_token_count_mean": float(
                    np.mean([float(row["active_token_count_mean"]) for row in selected])
                ),
                "active_token_count_max": int(
                    max(int(row["active_token_count_max"]) for row in selected)
                ),
                "active_read_ratio": float(
                    np.mean([float(row["active_read_ratio"]) for row in selected])
                ),
                "exact_token_fraction_mean": float(
                    np.mean(
                        [float(row["exact_token_fraction_mean"]) for row in selected]
                    )
                ),
                "selected_visual_mass_mean": float(
                    np.mean(
                        [float(row["selected_visual_mass_mean"]) for row in selected]
                    )
                ),
                "visual_mean": float(visual.mean()),
                "visual_p95": float(np.quantile(visual, 0.95)),
                "visual_worst": float(visual.max()),
                "visual_worst_head": float(
                    max(float(row["visual_worst_head_relative_l2"]) for row in selected)
                ),
                "full_mean": float(full.mean()),
                "full_p95": float(np.quantile(full, 0.95)),
                "full_worst": float(full.max()),
            }
        )
    return summaries


def classify_outcome(
    summaries: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    eligible = [
        row
        for row in summaries
        if float(row["active_read_ratio"]) >= 3.8
        and int(row["active_token_count_max"]) <= ACTIVE_TOKEN_BUDGET
    ]
    if not eligible:
        raise ValueError("prototype Gate produced no eligible candidate")

    def deployable(row: dict[str, object]) -> bool:
        return (
            row["selector"] == "prototype_mass"
            and float(row["visual_mean"]) <= 0.01
            and float(row["visual_p95"]) <= 0.02
            and float(row["visual_worst"]) <= 0.05
            and float(row["full_mean"]) <= 0.005
            and float(row["full_p95"]) <= 0.01
        )

    def capacity(row: dict[str, object]) -> bool:
        return (
            row["selector"] == "oracle_local"
            and float(row["visual_mean"]) <= 0.005
            and float(row["visual_p95"]) <= 0.01
            and float(row["visual_worst"]) <= 0.02
            and float(row["full_mean"]) <= 0.0025
            and float(row["full_p95"]) <= 0.005
        )

    deployable_rows = [row for row in eligible if deployable(row)]
    capacity_rows = [row for row in eligible if capacity(row)]
    best = min(eligible, key=lambda row: float(row["visual_mean"]))
    diagnostics = {
        "deployable_pass_count": len(deployable_rows),
        "capacity_pass_count": len(capacity_rows),
        "best_eligible": best,
    }
    if deployable_rows:
        return "PROTOTYPE_MIXTURE_DEPLOYABLE_PATH", diagnostics
    if capacity_rows:
        return "PROTOTYPE_MIXTURE_CAPACITY_ONLY", diagnostics
    return "NO_PROTOTYPE_MIXTURE_PATH", diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty prototype rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    gaussian = json.loads(args.gaussian_summary.read_text(encoding="utf-8"))
    progressive = json.loads(args.progressive_summary.read_text(encoding="utf-8"))
    if gaussian["decision"] != "NO_POSITIVE_GAUSSIAN_MEASURE_PATH":
        raise ValueError("positive-Gaussian prerequisite changed")
    if progressive["decision"] != "NO_PROGRESSIVE_EXACT_PAGE_PATH":
        raise ValueError("progressive-exact prerequisite changed")
    expected_count = 1 if args.smoke else 24
    if args.sample_offset != 72 or args.sample_count != expected_count:
        raise ValueError(
            "registered Gate requires position 73 for smoke or positions 73-96 formal"
        )
    if args.frame_budget != 8:
        raise ValueError("registered frame budget changed")
    if args.residual_greedy_diagnostic and args.reverse_greedy_diagnostic:
        raise ValueError("support-sensitivity modes are mutually exclusive")

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    samples = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=args.sample_offset + args.sample_count,
    )[args.sample_offset : args.sample_offset + args.sample_count]

    if args.residual_greedy_diagnostic or args.reverse_greedy_diagnostic:
        cluster_families = ("key",)
        prototype_counts = (128,)
        selectors = (
            ("oracle_reverse_greedy",)
            if args.reverse_greedy_diagnostic
            else ("oracle_residual_greedy",)
        )
    else:
        cluster_families = CLUSTER_FAMILIES
        prototype_counts = PROTOTYPE_COUNTS
        selectors = SELECTORS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    set_language_attention_eager(model)
    model_dtype = next(model.parameters()).dtype
    capture = SelectedLayerCapture(model, LAYERS)
    rows: list[dict[str, object]] = []
    maximum_replay_error = 0.0
    started = time.perf_counter()

    for sample_position, sample in enumerate(samples, start=args.sample_offset + 1):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0], args.frame_budget
        )
        reference = (
            payload["features"]
            .index_select(0, torch.tensor(selected_positions, dtype=torch.long))
            .to(device=device, dtype=model_dtype)
        )
        selected_frame_indices = [
            payload["pool_indices"][index] for index in selected_positions
        ]
        frames, _, _ = decode_video_frames(sample.video_path, selected_frame_indices)
        prompt_batch = build_prompt_batch(
            processor,
            sample,
            np.stack(frames),
            device=device,
            dtype=model_dtype,
        )
        capture.clear()
        with torch.inference_mode():
            first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            )
        if set(capture.captures) != set(LAYERS):
            raise RuntimeError("not every registered attention layer was captured")

        video_mask = prompt_batch["input_ids"][0] == model.config.video_token_index
        placeholder_positions = torch.nonzero(video_mask, as_tuple=False).flatten()
        visual_start = int(placeholder_positions[0].item())
        visual_token_count = reference.shape[0] * reference.shape[1]
        offsets = hierarchical_group_offsets(
            frame_count=reference.shape[0],
            token_count=reference.shape[1],
            topology="spatial_7x7",
            device=device,
        )
        for layer_index in LAYERS:
            components = build_gaussian_components(
                capture.captures[layer_index],
                visual_start=visual_start,
                visual_token_count=visual_token_count,
                group_offsets=offsets,
                max_rank=16,
            )
            maximum_replay_error = max(maximum_replay_error, components.replay_error)
            for family in cluster_families:
                for prototype_count in prototype_counts:
                    state = build_prototype_state(
                        components,
                        family=family,
                        prototype_count=prototype_count,
                    )
                    for selector in selectors:
                        if selector == "oracle_residual_greedy":
                            selected, active_tokens = (
                                select_residual_greedy_under_budget(
                                    state,
                                    active_token_budget=ACTIVE_TOKEN_BUDGET,
                                )
                            )
                        elif selector == "oracle_reverse_greedy":
                            selected, active_tokens = (
                                select_reverse_greedy_under_budget(
                                    state,
                                    active_token_budget=ACTIVE_TOKEN_BUDGET,
                                )
                            )
                        else:
                            priority = (
                                state.prototype_mass_priority
                                if selector == "prototype_mass"
                                else state.oracle_priority
                            )
                            selected, active_tokens = select_under_budget(
                                priority,
                                state.cluster_counts,
                                active_token_budget=ACTIVE_TOKEN_BUDGET,
                            )
                        metrics = evaluate_prototype_state(
                            state, selected, active_tokens
                        )
                        rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "sample_position": sample_position,
                                "layer_index": layer_index,
                                "cluster_family": family,
                                "prototype_count": prototype_count,
                                "selector": selector,
                                "active_token_budget": ACTIVE_TOKEN_BUDGET,
                                **metrics,
                            }
                        )
        print(
            json.dumps(
                {
                    "event": "prototype_measure_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "maximum_replay_error": maximum_replay_error,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    capture.remove()
    if maximum_replay_error > 1e-4:
        raise RuntimeError("captured Q/K/V did not reconstruct attention output")
    summaries = summarize_rows(rows)
    if args.residual_greedy_diagnostic or args.reverse_greedy_diagnostic:
        decision = (
            "POSTHOC_REVERSE_GREEDY_SENSITIVITY"
            if args.reverse_greedy_diagnostic
            else "POSTHOC_RESIDUAL_GREEDY_SENSITIVITY"
        )
        diagnostics = {
            "best_diagnostic": min(summaries, key=lambda row: row["visual_mean"])
        }
    else:
        decision, diagnostics = classify_outcome(summaries)
    write_csv(args.out_dir / "prototype_measure_rows.csv", rows)
    write_csv(args.out_dir / "prototype_measure_summary.csv", summaries)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": (
            f"{ROLE}_reverse_greedy_diagnostic"
            if args.reverse_greedy_diagnostic
            else f"{ROLE}_residual_greedy_diagnostic"
            if args.residual_greedy_diagnostic
            else f"{ROLE}_smoke"
            if args.smoke
            else ROLE
        ),
        "decision": decision,
        "sample_positions": [
            args.sample_offset + 1,
            args.sample_offset + args.sample_count,
        ],
        "sample_count": len(samples),
        "layers": list(LAYERS),
        "cluster_families": list(cluster_families),
        "prototype_counts": list(prototype_counts),
        "selectors": list(selectors),
        "active_token_budget": ACTIVE_TOKEN_BUDGET,
        "kmeans_iterations": KMEANS_ITERATIONS,
        "maximum_replay_error": maximum_replay_error,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Post-hoc exposed-only support-optimizer sensitivity; it cannot alter the "
            "registered prototype Gate decision or support a hidden-set, writer, "
            "accuracy, TTFT, or latency claim."
            if args.residual_greedy_diagnostic or args.reverse_greedy_diagnostic
            else "Fixed-query, single-layer, target-free writer capacity diagnostic "
            "on exposed calibration positions 73-96. K-means writer cost, cold exact "
            "storage, reader accuracy, TTFT, and latency are not claimed."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
