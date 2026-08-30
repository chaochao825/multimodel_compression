from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_utils import (
    build_prompt_batch,
    first_token_logits_from_features,
    load_onevision_model,
)
from mvbench_utils import decode_video_frames, uniform_frame_indices
from probe_vsi_onevision_cmrq_stage_b import feature_path_for_sample
from probe_vsi_onevision_reader_risk_stage_a import select_calibration_questions
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    set_language_attention_eager,
)
from probe_vsi_onevision_true_2x2_geometry import spatial_2x2_means_and_offsets
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


LAYERS = (0, 13, 27)
METHODS = (
    "analytic_remainder",
    "attention_mass",
    "exact_local_score",
    "exact_greedy_oracle",
    "fixed_random",
)
BUDGETS = (0, 49, 98, 147, 196, 392)
ROLE = "exposed_query_fixed_measure_remainder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--m0-summary", type=Path, required=True)
    parser.add_argument("--m1-summary", type=Path, required=True)
    parser.add_argument("--geometry-summary", type=Path, required=True)
    parser.add_argument("--ppe-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@dataclass
class AttentionCapture:
    module: torch.nn.Module
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_mask: torch.Tensor | None
    unprojected_output: torch.Tensor


class SelectedLayerCapture:
    def __init__(self, model: torch.nn.Module, layer_indices: tuple[int, ...]) -> None:
        self.captures: dict[int, AttentionCapture] = {}
        self.position_embeddings: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.attention_masks: dict[int, torch.Tensor | None] = {}
        self.projections: dict[int, dict[str, torch.Tensor]] = {}
        self.unprojected_outputs: dict[int, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        layers = model.model.language_model.layers
        if max(layer_indices) >= len(layers):
            raise ValueError("registered attention layer does not exist")
        for layer_index in layer_indices:
            attention = layers[layer_index].self_attn
            self.handles.append(
                attention.register_forward_pre_hook(
                    self._pre_hook(layer_index), with_kwargs=True
                )
            )
            for projection_name, projection in (
                ("query", attention.q_proj),
                ("key", attention.k_proj),
                ("value", attention.v_proj),
            ):
                self.handles.append(
                    projection.register_forward_hook(
                        self._projection_hook(layer_index, projection_name)
                    )
                )
            self.handles.append(
                attention.o_proj.register_forward_pre_hook(
                    self._output_projection_pre_hook(layer_index)
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    self._post_hook(layer_index), with_kwargs=True
                )
            )

    def _pre_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            _args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> None:
            hidden_states = kwargs["hidden_states"]
            position_embeddings = kwargs["position_embeddings"]
            attention_mask = kwargs["attention_mask"]
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError("attention hidden states are not a tensor")
            if not isinstance(position_embeddings, tuple):
                raise TypeError("attention position embeddings are not a tuple")
            if attention_mask is not None and not isinstance(
                attention_mask, torch.Tensor
            ):
                raise TypeError("selected attention mask is not a tensor")
            self.position_embeddings[layer_index] = position_embeddings
            self.attention_masks[layer_index] = attention_mask
            self.projections[layer_index] = {}

        return hook

    def _projection_hook(self, layer_index: int, projection_name: str):
        def hook(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            output: torch.Tensor,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError("attention projection output is not a tensor")
            self.projections[layer_index][projection_name] = output.detach()

        return hook

    def _output_projection_pre_hook(self, layer_index: int):
        def hook(
            _module: torch.nn.Module,
            args: tuple[object, ...],
        ) -> None:
            output_projection_input = args[0]
            if not isinstance(output_projection_input, torch.Tensor):
                raise TypeError("attention output projection input is not a tensor")
            self.unprojected_outputs[layer_index] = output_projection_input[
                0, -1
            ].detach()

        return hook

    def _post_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            _args: tuple[object, ...],
            _kwargs: dict[str, object],
            output: tuple[torch.Tensor, torch.Tensor | None],
        ) -> None:
            projections = self.projections[layer_index]
            if set(projections) != {"query", "key", "value"}:
                raise RuntimeError("not every attention projection was captured")
            input_shape = projections["query"].shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = projections["query"].view(hidden_shape).transpose(1, 2)
            key = projections["key"].view(hidden_shape).transpose(1, 2)
            value = projections["value"].view(hidden_shape).transpose(1, 2)
            cos, sin = self.position_embeddings[layer_index]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
            attention_mask = self.attention_masks[layer_index]
            self.captures[layer_index] = AttentionCapture(
                module=module,
                query=query[0].detach(),
                key=key[0].detach(),
                value=value[0].detach(),
                attention_mask=(
                    attention_mask[0, :, :, : key.shape[-2]].detach()
                    if attention_mask is not None
                    else None
                ),
                unprojected_output=self.unprojected_outputs[layer_index],
            )

        return hook

    def clear(self) -> None:
        self.captures.clear()
        self.position_embeddings.clear()
        self.attention_masks.clear()
        self.projections.clear()
        self.unprojected_outputs.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@dataclass
class MeasureState:
    exact_visual_output: torch.Tensor
    exact_full_output: torch.Tensor
    exact_visual_z: torch.Tensor
    exact_group_z: torch.Tensor
    exact_group_n: torch.Tensor
    coarse_group_z: torch.Tensor
    coarse_group_n: torch.Tensor
    local_bound: torch.Tensor
    analytic_priority: torch.Tensor
    mass_priority: torch.Tensor
    oracle_priority: torch.Tensor
    certificate_valid: torch.Tensor
    exact_projected_relative_error: float
    local_z_bound_violations: int
    local_n_bound_violations: int


def replay_attention(capture: AttentionCapture) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(capture.query, capture.key.transpose(-2, -1))
    scores = scores * float(capture.module.scaling)
    if capture.attention_mask is not None:
        scores = scores + capture.attention_mask
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(capture.query.dtype)
    head_output = torch.matmul(weights, capture.value)
    flattened = (
        head_output.transpose(0, 1).contiguous().reshape(head_output.shape[1], -1)
    )
    relative_error = float(
        (
            torch.linalg.vector_norm(
                flattened[-1].float() - capture.unprojected_output.float()
            )
            / torch.linalg.vector_norm(capture.unprojected_output.float()).clamp_min(
                1e-12
            )
        ).item()
    )
    return head_output[:, -1].float(), torch.tensor(relative_error)


def grouped_measure_state(
    capture: AttentionCapture,
    *,
    visual_start: int,
    visual_token_count: int,
    group_offsets: torch.Tensor,
) -> MeasureState:
    _, projected_error = replay_attention(capture)
    query = capture.query[:, -1].float()
    key = capture.key.float()
    value = capture.value.float()
    scores = torch.einsum("hd,hsd->hs", query, key) * float(capture.module.scaling)
    if capture.attention_mask is not None:
        scores = scores + capture.attention_mask[:, -1].float()
    maximum = scores.max(dim=1, keepdim=True).values
    exponentials = torch.exp(scores - maximum)

    visual_stop = visual_start + visual_token_count
    visual_scores = scores[:, visual_start:visual_stop]
    visual_exp = exponentials[:, visual_start:visual_stop]
    visual_value = value[:, visual_start:visual_stop]
    nonvisual_mask = torch.ones(scores.shape[1], device=scores.device, dtype=torch.bool)
    nonvisual_mask[visual_start:visual_stop] = False
    nonvisual_exp = exponentials[:, nonvisual_mask]
    nonvisual_value = value[:, nonvisual_mask]
    nonvisual_z = nonvisual_exp.sum(dim=1)
    nonvisual_n = torch.einsum("hs,hsd->hd", nonvisual_exp, nonvisual_value)

    head_count, _, head_dim = visual_value.shape
    group_count, group_size = group_offsets.shape
    if group_count != 392 or group_size != 4:
        raise ValueError("registered visual group topology changed")
    offsets = group_offsets.to(device=key.device)
    member_scores = visual_scores.index_select(1, offsets.flatten()).reshape(
        head_count, group_count, group_size
    )
    member_exp = visual_exp.index_select(1, offsets.flatten()).reshape(
        head_count, group_count, group_size
    )
    member_value = visual_value.index_select(1, offsets.flatten()).reshape(
        head_count, group_count, group_size, head_dim
    )
    member_key = (
        key[:, visual_start:visual_stop]
        .index_select(1, offsets.flatten())
        .reshape(head_count, group_count, group_size, head_dim)
    )

    exact_group_z = member_exp.sum(dim=2)
    exact_group_n = torch.einsum("hgm,hgmd->hgd", member_exp, member_value)
    key_center = member_key.mean(dim=2)
    value_center = member_value.mean(dim=2)
    score_center = torch.einsum("hd,hgd->hg", query, key_center) * float(
        capture.module.scaling
    )
    shifted_center = score_center - maximum
    coarse_group_z = group_size * torch.exp(shifted_center)
    coarse_group_n = coarse_group_z.unsqueeze(-1) * value_center

    score_radius = (member_scores - score_center.unsqueeze(-1)).abs().max(dim=2).values
    value_radius = (
        torch.linalg.vector_norm(member_value - value_center.unsqueeze(2), dim=-1)
        .max(dim=2)
        .values
    )
    epsilon_z = coarse_group_z * torch.expm1(score_radius)
    epsilon_n = (
        group_size * torch.exp(shifted_center + score_radius) * value_radius
        + torch.linalg.vector_norm(value_center, dim=-1) * epsilon_z
    )
    exact_z_defect = (exact_group_z - coarse_group_z).abs()
    exact_n_defect = torch.linalg.vector_norm(exact_group_n - coarse_group_n, dim=-1)
    z_tolerance = 1e-6 + 1e-5 * exact_group_z.abs()
    n_tolerance = 1e-6 + 1e-5 * torch.linalg.vector_norm(exact_group_n, dim=-1)
    local_z_bound_violations = int(
        (exact_z_defect > epsilon_z + z_tolerance).sum().item()
    )
    local_n_bound_violations = int(
        (exact_n_defect > epsilon_n + n_tolerance).sum().item()
    )

    exact_visual_z = exact_group_z.sum(dim=1)
    exact_visual_n = exact_group_n.sum(dim=1)
    exact_visual_output = exact_visual_n / exact_visual_z.unsqueeze(-1)
    exact_full_z = exact_visual_z + nonvisual_z
    exact_full_n = exact_visual_n + nonvisual_n
    exact_full_output_from_groups = exact_full_n / exact_full_z.unsqueeze(-1)

    coarse_z = coarse_group_z.sum(dim=1)
    coarse_n = coarse_group_n.sum(dim=1)
    z_floor = coarse_z - epsilon_z.sum(dim=1)
    certificate_valid = z_floor > 0
    safe_floor = z_floor.clamp_min(1e-12)
    numerator_upper = (
        torch.linalg.vector_norm(coarse_group_n, dim=-1) + epsilon_n
    ).sum(dim=1)
    local_bound = epsilon_n / safe_floor.unsqueeze(-1) + numerator_upper.unsqueeze(
        -1
    ) * epsilon_z / safe_floor.square().unsqueeze(-1)
    coarse_output = coarse_n / coarse_z.unsqueeze(-1)
    analytic_priority = epsilon_n / coarse_z.unsqueeze(-1).clamp_min(
        1e-12
    ) + torch.linalg.vector_norm(coarse_output, dim=-1).unsqueeze(
        -1
    ) * epsilon_z / coarse_z.unsqueeze(-1).clamp_min(1e-12)
    analytic_priority = torch.where(
        certificate_valid.unsqueeze(-1), local_bound, analytic_priority
    )
    mass_priority = exact_group_z / exact_visual_z.unsqueeze(-1)
    local_output_defect = (exact_group_n - coarse_group_n) / exact_visual_z[
        :, None, None
    ] - exact_visual_output[:, None, :] * (exact_group_z - coarse_group_z)[
        :, :, None
    ] / exact_visual_z[:, None, None]
    oracle_priority = torch.linalg.vector_norm(local_output_defect, dim=-1)

    return MeasureState(
        exact_visual_output=exact_visual_output,
        exact_full_output=exact_full_output_from_groups,
        exact_visual_z=exact_visual_z,
        exact_group_z=exact_group_z,
        exact_group_n=exact_group_n,
        coarse_group_z=coarse_group_z,
        coarse_group_n=coarse_group_n,
        local_bound=local_bound,
        analytic_priority=analytic_priority,
        mass_priority=mass_priority,
        oracle_priority=oracle_priority,
        certificate_valid=certificate_valid,
        exact_projected_relative_error=float(projected_error.item()),
        local_z_bound_violations=local_z_bound_violations,
        local_n_bound_violations=local_n_bound_violations,
    )


def greedy_analytic_order(priority: torch.Tensor) -> torch.Tensor:
    if priority.ndim != 2:
        raise ValueError("analytic priority must have shape [heads, groups]")
    head_count, group_count = priority.shape
    remaining = priority.sum(dim=1)
    selected = torch.zeros(group_count, device=priority.device, dtype=torch.bool)
    order: list[int] = []
    for _ in range(group_count):
        candidate_remaining = remaining[:, None] - priority
        candidate_norm = torch.linalg.vector_norm(candidate_remaining, dim=0)
        candidate_norm[selected] = torch.inf
        next_group = int(torch.argmin(candidate_norm).item())
        selected[next_group] = True
        remaining = remaining - priority[:, next_group]
        order.append(next_group)
    if len(set(order)) != group_count:
        raise RuntimeError("analytic greedy order contains duplicate groups")
    return torch.tensor(order, device=priority.device, dtype=torch.long)


def selector_orders(
    state: MeasureState,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return {
        "analytic_remainder": greedy_analytic_order(state.analytic_priority),
        "attention_mass": torch.argsort(
            state.mass_priority.sum(dim=0), descending=True, stable=True
        ),
        "exact_local_score": torch.argsort(
            torch.linalg.vector_norm(state.oracle_priority, dim=0),
            descending=True,
            stable=True,
        ),
        "exact_greedy_oracle": exact_greedy_order(state),
        "fixed_random": torch.randperm(
            state.exact_group_z.shape[1], generator=generator
        ).to(state.exact_group_z.device),
    }


def exact_greedy_order(state: MeasureState) -> torch.Tensor:
    group_count = state.exact_group_z.shape[1]
    delta_z = state.exact_group_z - state.coarse_group_z
    delta_n = state.exact_group_n - state.coarse_group_n
    current_z = state.coarse_group_z.sum(dim=1)
    current_n = state.coarse_group_n.sum(dim=1)
    selected = torch.zeros(group_count, device=current_z.device, dtype=torch.bool)
    order: list[int] = []
    for _ in range(group_count):
        candidate_z = current_z[:, None] + delta_z
        candidate_n = current_n[:, None, :] + delta_n
        candidate_output = candidate_n / candidate_z.unsqueeze(-1)
        candidate_error = torch.linalg.vector_norm(
            candidate_output - state.exact_visual_output[:, None, :], dim=(0, 2)
        )
        candidate_error[selected] = torch.inf
        next_group = int(torch.argmin(candidate_error).item())
        selected[next_group] = True
        current_z = current_z + delta_z[:, next_group]
        current_n = current_n + delta_n[:, next_group]
        order.append(next_group)
    if len(set(order)) != group_count:
        raise RuntimeError("exact greedy order contains duplicate groups")
    return torch.tensor(order, device=current_z.device, dtype=torch.long)


def evaluate_selection(
    state: MeasureState,
    selected_indices: torch.Tensor,
    *,
    nonvisual_z: torch.Tensor,
    nonvisual_n: torch.Tensor,
) -> dict[str, object]:
    group_count = state.exact_group_z.shape[1]
    selected = torch.zeros(
        group_count, device=selected_indices.device, dtype=torch.bool
    )
    selected[selected_indices] = True
    visual_z = torch.where(
        selected.unsqueeze(0), state.exact_group_z, state.coarse_group_z
    ).sum(dim=1)
    visual_n = torch.where(
        selected[None, :, None], state.exact_group_n, state.coarse_group_n
    ).sum(dim=1)
    visual_output = visual_n / visual_z.unsqueeze(-1)
    full_output = (visual_n + nonvisual_n) / (visual_z + nonvisual_z).unsqueeze(-1)
    visual_head_error = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output, dim=-1
    ) / torch.linalg.vector_norm(state.exact_visual_output, dim=-1).clamp_min(1e-12)
    full_head_error = torch.linalg.vector_norm(
        full_output - state.exact_full_output, dim=-1
    ) / torch.linalg.vector_norm(state.exact_full_output, dim=-1).clamp_min(1e-12)
    visual_relative = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output
    ) / torch.linalg.vector_norm(state.exact_visual_output).clamp_min(1e-12)
    full_relative = torch.linalg.vector_norm(
        full_output - state.exact_full_output
    ) / torch.linalg.vector_norm(state.exact_full_output).clamp_min(1e-12)

    remaining_bound = state.local_bound[:, ~selected].sum(dim=1)
    certificate_absolute = torch.linalg.vector_norm(remaining_bound)
    certificate_relative = certificate_absolute / torch.linalg.vector_norm(
        state.exact_visual_output
    ).clamp_min(1e-12)
    actual_absolute = torch.linalg.vector_norm(
        visual_output - state.exact_visual_output
    )
    valid_heads = state.certificate_valid
    bound_violation = int(
        bool(valid_heads.all().item()) and actual_absolute > certificate_absolute + 1e-5
    )
    return {
        "visual_relative_l2": float(visual_relative.item()),
        "visual_worst_head_relative_l2": float(visual_head_error.max().item()),
        "full_relative_l2": float(full_relative.item()),
        "full_worst_head_relative_l2": float(full_head_error.max().item()),
        "certificate_relative": (
            float(certificate_relative.item())
            if bool(valid_heads.all().item())
            else math.nan
        ),
        "certificate_valid_head_fraction": float(valid_heads.float().mean().item()),
        "bound_violation": bound_violation,
        "head_visual_relative_l2": visual_head_error,
        "head_certificate_relative": remaining_bound
        / torch.linalg.vector_norm(state.exact_visual_output, dim=-1).clamp_min(1e-12),
    }


def nonvisual_measure(
    capture: AttentionCapture,
    *,
    visual_start: int,
    visual_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = capture.query[:, -1].float()
    key = capture.key.float()
    value = capture.value.float()
    scores = torch.einsum("hd,hsd->hs", query, key) * float(capture.module.scaling)
    if capture.attention_mask is not None:
        scores = scores + capture.attention_mask[:, -1].float()
    maximum = scores.max(dim=1, keepdim=True).values
    exponentials = torch.exp(scores - maximum)
    mask = torch.ones(scores.shape[1], device=scores.device, dtype=torch.bool)
    mask[visual_start : visual_start + visual_token_count] = False
    selected_exp = exponentials[:, mask]
    selected_value = value[:, mask]
    return selected_exp.sum(dim=1), torch.einsum(
        "hs,hsd->hd", selected_exp, selected_value
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty query-fixed rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for method in METHODS:
        method_summary: dict[str, dict[str, float | int]] = {}
        for budget in BUDGETS:
            selected = [
                row
                for row in rows
                if row["method"] == method and int(row["exact_group_count"]) == budget
            ]
            visual = np.asarray([float(row["visual_relative_l2"]) for row in selected])
            full = np.asarray([float(row["full_relative_l2"]) for row in selected])
            method_summary[str(budget)] = {
                "cell_count": len(selected),
                "visual_relative_l2_mean": float(visual.mean()),
                "visual_relative_l2_p95": float(np.quantile(visual, 0.95)),
                "visual_relative_l2_worst": float(visual.max()),
                "visual_worst_head_relative_l2_worst": max(
                    float(row["visual_worst_head_relative_l2"]) for row in selected
                ),
                "full_relative_l2_mean": float(full.mean()),
                "full_relative_l2_p95": float(np.quantile(full, 0.95)),
                "full_relative_l2_worst": float(full.max()),
                "bound_violation_count": sum(
                    int(row["bound_violation"]) for row in selected
                ),
                "certificate_valid_head_fraction_mean": float(
                    np.mean(
                        [
                            float(row["certificate_valid_head_fraction"])
                            for row in selected
                        ]
                    )
                ),
            }
        summaries[method] = method_summary
    return summaries


def count_path_regressions(rows: list[dict[str, object]]) -> dict[str, int]:
    result: dict[str, int] = {}
    sample_layers = sorted(
        {(str(row["sample_id"]), int(row["layer_index"])) for row in rows}
    )
    for method in METHODS:
        regressions = 0
        for sample_id, layer_index in sample_layers:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method
                    and row["sample_id"] == sample_id
                    and int(row["layer_index"]) == layer_index
                ),
                key=lambda row: int(row["exact_group_count"]),
            )
            regressions += sum(
                float(current["visual_relative_l2"])
                > float(previous["visual_relative_l2"]) + 1e-9
                for previous, current in zip(selected, selected[1:])
            )
        result[method] = regressions
    return result


def classify_outcome(
    summaries: dict[str, dict[str, dict[str, float | int]]],
    *,
    certificate_valid_head_fraction: float,
    certificate_violation_count: int,
    certificate_increase_count: int,
) -> tuple[str, dict[str, object]]:
    analytic = summaries["analytic_remainder"]["196"]
    oracle = summaries["exact_greedy_oracle"]["196"]
    actual_keys = (
        "visual_relative_l2_mean",
        "visual_relative_l2_p95",
        "visual_relative_l2_worst",
        "full_relative_l2_mean",
        "full_relative_l2_p95",
    )

    def actual_pass(metrics: dict[str, float | int]) -> bool:
        thresholds = (0.01, 0.02, 0.05, 0.005, 0.01)
        return all(
            float(metrics[key]) <= threshold
            for key, threshold in zip(actual_keys, thresholds)
        )

    beat_mass_budgets = []
    for budget in (98, 147, 196):
        analytic_mean = float(
            summaries["analytic_remainder"][str(budget)]["visual_relative_l2_mean"]
        )
        mass_mean = float(
            summaries["attention_mass"][str(budget)]["visual_relative_l2_mean"]
        )
        if analytic_mean <= 0.9 * mass_mean:
            beat_mass_budgets.append(budget)
    diagnostics = {
        "analytic_actual_pass": actual_pass(analytic),
        "oracle_actual_pass": actual_pass(oracle),
        "certificate_valid_head_fraction": certificate_valid_head_fraction,
        "certificate_violation_count": certificate_violation_count,
        "certificate_increase_count": certificate_increase_count,
        "analytic_beats_mass_budgets": beat_mass_budgets,
    }
    certified = (
        bool(diagnostics["analytic_actual_pass"])
        and certificate_valid_head_fraction >= 0.95
        and certificate_violation_count == 0
        and certificate_increase_count == 0
        and len(beat_mass_budgets) >= 2
    )
    if certified:
        return "QUERY_FIXED_CERTIFIED_HEADROOM", diagnostics
    if bool(diagnostics["oracle_actual_pass"]):
        return "QUERY_FIXED_CAPACITY_BOUND_LOOSE", diagnostics
    return "NO_REGISTERED_QUERY_FIXED_MEASURE_PATH", diagnostics


def main() -> int:
    args = parse_args()
    required_decisions = (
        (args.m0_summary, "SAME_KERNEL_MASS_VALID"),
        (args.m1_summary, "NO_BATCHED_CURRENT_SUPPORT_PATH"),
        (args.geometry_summary, "TRUE_2X2_DECISION_HEADROOM"),
        (args.ppe_summary, "NO_PPE_HEADROOM"),
    )
    for path, decision in required_decisions:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["decision"] != decision:
            raise ValueError(f"required decision changed for {path.name}")

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    selected = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=args.sample_offset + args.sample_count,
    )[args.sample_offset : args.sample_offset + args.sample_count]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    set_language_attention_eager(model)
    model_dtype = next(model.parameters()).dtype
    capture = SelectedLayerCapture(model, LAYERS)

    path_rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []
    maximum_projected_replay_error = 0.0
    local_z_bound_violations = 0
    local_n_bound_violations = 0
    certificate_violation_count = 0
    certificate_increase_count = 0
    finite_certificate_heads = 0
    total_certificate_heads = 0
    started = time.perf_counter()
    for sample_position, sample in enumerate(
        selected,
        start=args.sample_offset + 1,
    ):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0], args.frame_budget
        )
        position_tensor = torch.tensor(selected_positions, dtype=torch.long)
        reference = (
            payload["features"]
            .index_select(0, position_tensor)
            .to(
                device=device,
                dtype=model_dtype,
            )
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
        _, group_offsets = spatial_2x2_means_and_offsets(reference)

        for layer_index in LAYERS:
            layer_capture = capture.captures[layer_index]
            state = grouped_measure_state(
                layer_capture,
                visual_start=visual_start,
                visual_token_count=visual_token_count,
                group_offsets=group_offsets,
            )
            nonvisual_z, nonvisual_n = nonvisual_measure(
                layer_capture,
                visual_start=visual_start,
                visual_token_count=visual_token_count,
            )
            maximum_projected_replay_error = max(
                maximum_projected_replay_error,
                state.exact_projected_relative_error,
            )
            local_z_bound_violations += state.local_z_bound_violations
            local_n_bound_violations += state.local_n_bound_violations
            finite_certificate_heads += int(state.certificate_valid.sum().item())
            total_certificate_heads += int(state.certificate_valid.numel())
            orders = selector_orders(
                state,
                seed=args.seed + sample_position * 100 + layer_index,
            )
            for method in METHODS:
                previous_certificate = math.inf
                for budget in BUDGETS:
                    selected_indices = orders[method][:budget]
                    metrics = evaluate_selection(
                        state,
                        selected_indices,
                        nonvisual_z=nonvisual_z,
                        nonvisual_n=nonvisual_n,
                    )
                    certificate = float(metrics["certificate_relative"])
                    if math.isfinite(certificate):
                        certificate_increase_count += int(
                            certificate > previous_certificate + 1e-8
                        )
                        previous_certificate = certificate
                    certificate_violation_count += int(metrics["bound_violation"])
                    path_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "sample_position": sample_position,
                            "layer_index": layer_index,
                            "method": method,
                            "exact_group_count": budget,
                            "visual_token_retention": (392 + 3 * budget) / 1568,
                            "visual_relative_l2": metrics["visual_relative_l2"],
                            "visual_worst_head_relative_l2": metrics[
                                "visual_worst_head_relative_l2"
                            ],
                            "full_relative_l2": metrics["full_relative_l2"],
                            "full_worst_head_relative_l2": metrics[
                                "full_worst_head_relative_l2"
                            ],
                            "certificate_relative": metrics["certificate_relative"],
                            "certificate_valid_head_fraction": metrics[
                                "certificate_valid_head_fraction"
                            ],
                            "bound_violation": metrics["bound_violation"],
                        }
                    )
                    if method == "analytic_remainder":
                        selected_mask = torch.zeros(
                            392, device=device, dtype=torch.bool
                        )
                        selected_mask[selected_indices] = True
                        head_error = metrics["head_visual_relative_l2"]
                        head_certificate = metrics["head_certificate_relative"]
                        for head_index in range(head_error.numel()):
                            head_rows.append(
                                {
                                    "sample_id": sample.sample_id,
                                    "sample_position": sample_position,
                                    "layer_index": layer_index,
                                    "head_index": head_index,
                                    "exact_group_count": budget,
                                    "visual_relative_l2": float(
                                        head_error[head_index].item()
                                    ),
                                    "certificate_relative": (
                                        float(head_certificate[head_index].item())
                                        if bool(
                                            state.certificate_valid[head_index].item()
                                        )
                                        else math.nan
                                    ),
                                    "certificate_valid": int(
                                        state.certificate_valid[head_index].item()
                                    ),
                                    "remaining_group_count": int(
                                        (~selected_mask).sum().item()
                                    ),
                                }
                            )

        print(
            json.dumps(
                {
                    "event": "query_fixed_measure_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "maximum_projected_replay_error": maximum_projected_replay_error,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    capture.remove()
    if maximum_projected_replay_error > 1e-4:
        raise RuntimeError("captured Q/K/V did not reconstruct attention output")
    if local_z_bound_violations or local_n_bound_violations:
        raise RuntimeError("analytic local remainder did not bound exact local defect")
    if certificate_increase_count:
        raise RuntimeError("registered analytic certificate was not monotone")

    summaries = summarize_rows(path_rows)
    path_regressions = count_path_regressions(path_rows)
    certificate_valid_head_fraction = finite_certificate_heads / total_certificate_heads
    decision, diagnostics = classify_outcome(
        summaries,
        certificate_valid_head_fraction=certificate_valid_head_fraction,
        certificate_violation_count=certificate_violation_count,
        certificate_increase_count=certificate_increase_count,
    )
    write_csv(args.out_dir / "query_fixed_measure_path_rows.csv", path_rows)
    write_csv(args.out_dir / "query_fixed_measure_head_rows.csv", head_rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "sample_positions": [
            args.sample_offset + 1,
            args.sample_offset + args.sample_count,
        ],
        "sample_count": len(selected),
        "layers": list(LAYERS),
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "attention_implementation": "eager",
        "maximum_projected_replay_error": maximum_projected_replay_error,
        "local_z_bound_violations": local_z_bound_violations,
        "local_n_bound_violations": local_n_bound_violations,
        "certificate_valid_head_fraction": certificate_valid_head_fraction,
        "certificate_violation_count": certificate_violation_count,
        "certificate_increase_count": certificate_increase_count,
        "actual_error_path_regressions": path_regressions,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Fixed-query, single-layer attention-measure capacity and certificate "
            "probe on exposed positions 73-96. It is not a self-attention replacement, "
            "reader endpoint, trained method, latency result, or deployment claim."
        ),
    }
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
