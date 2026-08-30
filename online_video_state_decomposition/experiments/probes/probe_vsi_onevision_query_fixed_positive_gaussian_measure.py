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

from mvbench_onevision_utils import (
    build_prompt_batch,
    decode_video_frames,
    first_token_logits_from_features,
    load_onevision_model,
    uniform_frame_indices,
)
from probe_vsi_onevision_cmrq_stage_b import feature_path_for_sample
from probe_vsi_onevision_query_fixed_measure_remainder import (
    AttentionCapture,
    SelectedLayerCapture,
    replay_attention,
)
from probe_vsi_onevision_reader_risk_stage_a import select_calibration_questions
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    set_language_attention_eager,
)
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


LAYERS = (0, 13, 27)
TOPOLOGIES = ("spatial_7x7", "temporal2_spatial_7x7")
RANKS = (0, 2, 4, 8, 16)
SELECTORS = ("compact_mass", "oracle_local")
EXACT_FRACTIONS = (0.0, 0.125, 0.25, 0.5, 1.0)
ROLE = "exposed_query_fixed_positive_gaussian_measure"


@dataclass
class GaussianComponents:
    exact_visual_output: torch.Tensor
    exact_full_output: torch.Tensor
    exact_group_z: torch.Tensor
    exact_group_n: torch.Tensor
    nonvisual_z: torch.Tensor
    nonvisual_n: torch.Tensor
    mean_value: torch.Tensor
    score_center: torch.Tensor
    maximum: torch.Tensor
    eigenvalues: torch.Tensor
    cross_value_key: torch.Tensor
    query_coordinates: torch.Tensor
    group_size: int
    replay_error: float
    mean_key: torch.Tensor
    key_min: torch.Tensor
    key_max: torch.Tensor
    query_scaled: torch.Tensor
    visual_value_norm_max: torch.Tensor
    member_key: torch.Tensor
    member_value: torch.Tensor


@dataclass
class GaussianMeasureState:
    exact_visual_output: torch.Tensor
    exact_full_output: torch.Tensor
    exact_group_z: torch.Tensor
    exact_group_n: torch.Tensor
    coarse_group_z: torch.Tensor
    coarse_group_n: torch.Tensor
    mass_priority: torch.Tensor
    oracle_priority: torch.Tensor
    score_variance: torch.Tensor
    measure_scale: torch.Tensor
    coarse_log_group_z: torch.Tensor
    coarse_mean_value: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--headwise-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def hierarchical_group_offsets(
    *, frame_count: int, token_count: int, topology: str, device: torch.device
) -> torch.Tensor:
    if frame_count != 8 or token_count != 196:
        raise ValueError("registered visual geometry must be 8 x 14 x 14")
    grid = torch.arange(
        frame_count * token_count, device=device, dtype=torch.long
    ).reshape(frame_count, 14, 14)
    if topology == "spatial_7x7":
        offsets = grid.reshape(8, 2, 7, 2, 7).permute(0, 1, 3, 2, 4).reshape(32, 49)
    elif topology == "temporal2_spatial_7x7":
        offsets = (
            grid.reshape(4, 2, 2, 7, 2, 7).permute(0, 2, 4, 1, 3, 5).reshape(16, 98)
        )
    else:
        raise ValueError(f"unregistered topology: {topology}")
    expected = torch.arange(frame_count * token_count, device=device)
    if not torch.equal(torch.sort(offsets.flatten()).values, expected):
        raise RuntimeError("hierarchical groups do not partition the visual tokens")
    return offsets


def low_rank_gaussian_parameters(
    member_key: torch.Tensor,
    member_value: torch.Tensor,
    *,
    query_scaled: torch.Tensor,
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if member_key.ndim != 4 or member_value.shape != member_key.shape:
        raise ValueError("member K/V must share [heads, groups, members, dim]")
    group_size = member_key.shape[2]
    if max_rank <= 0 or max_rank >= group_size:
        raise ValueError("max rank must lie inside the empirical group rank")
    mean_key = member_key.mean(dim=2)
    mean_value = member_value.mean(dim=2)
    centered_key = member_key - mean_key.unsqueeze(2)
    centered_value = member_value - mean_value.unsqueeze(2)
    _, singular_values, right = torch.linalg.svd(centered_key, full_matrices=False)
    basis = right[..., :max_rank, :].transpose(-1, -2)
    eigenvalues = singular_values[..., :max_rank].square() / group_size
    key_coordinates = torch.einsum("hgmf,hgfr->hgmr", centered_key, basis)
    cross_value_key = (
        torch.einsum("hgme,hgmr->hger", centered_value, key_coordinates) / group_size
    )
    query_coordinates = torch.einsum("hf,hgfr->hgr", query_scaled, basis)
    return (
        mean_key,
        mean_value,
        eigenvalues,
        cross_value_key,
        query_coordinates,
    )


def positive_gaussian_measure(
    components: GaussianComponents, rank: int
) -> GaussianMeasureState:
    if rank not in RANKS:
        raise ValueError("Gaussian rank is not registered")
    if rank == 0:
        score_variance = torch.zeros_like(components.score_center)
        tilted_value = components.mean_value
    else:
        eigenvalues = components.eigenvalues[..., :rank]
        coordinates = components.query_coordinates[..., :rank]
        score_variance = (eigenvalues * coordinates.square()).sum(dim=-1)
        tilted_value = components.mean_value + torch.einsum(
            "hger,hgr->hge",
            components.cross_value_key[..., :rank],
            coordinates,
        )
    log_group_z = (
        math.log(components.group_size)
        + components.score_center
        + 0.5 * score_variance
        - components.maximum
    )
    if not torch.isfinite(log_group_z).all():
        raise RuntimeError("Gaussian measure produced a non-finite log mass")
    # Exact and compact branches may use any shared positive headwise scale.
    # This is the same invariance used by online softmax and avoids overflow
    # when the Gaussian log-MGF exceeds the largest empirical token score.
    log_stabilizer = log_group_z.max(dim=1).values.clamp_min(0.0)
    measure_scale = torch.exp(-log_stabilizer)
    coarse_group_z = torch.exp(log_group_z - log_stabilizer.unsqueeze(-1))
    coarse_group_n = coarse_group_z.unsqueeze(-1) * tilted_value
    if (
        not torch.isfinite(coarse_group_z).all()
        or not torch.isfinite(coarse_group_n).all()
    ):
        raise RuntimeError("Gaussian measure produced a non-finite state")
    if bool((coarse_group_z < 0).any().item()):
        raise RuntimeError("Gaussian measure produced negative mass")
    exact_visual_z = components.exact_group_z.sum(dim=1)
    log_exact_visual_z = torch.log(exact_visual_z.clamp_min(1e-30))
    coarse_z_normalized = torch.exp(
        (log_group_z - log_exact_visual_z.unsqueeze(-1)).clamp(-80.0, 80.0)
    )
    exact_z_normalized = components.exact_group_z / exact_visual_z.unsqueeze(-1)
    exact_n_normalized = components.exact_group_n / exact_visual_z[:, None, None]
    coarse_n_normalized = coarse_z_normalized.unsqueeze(-1) * tilted_value
    local_output_defect = (
        exact_n_normalized
        - coarse_n_normalized
        - components.exact_visual_output[:, None, :]
        * (exact_z_normalized - coarse_z_normalized)[:, :, None]
    )
    return GaussianMeasureState(
        exact_visual_output=components.exact_visual_output,
        exact_full_output=components.exact_full_output,
        exact_group_z=components.exact_group_z,
        exact_group_n=components.exact_group_n,
        coarse_group_z=coarse_group_z,
        coarse_group_n=coarse_group_n,
        mass_priority=coarse_group_z,
        oracle_priority=torch.linalg.vector_norm(local_output_defect, dim=-1),
        score_variance=score_variance,
        measure_scale=measure_scale,
        coarse_log_group_z=log_group_z,
        coarse_mean_value=tilted_value,
    )


def evaluate_gaussian_support(
    state: GaussianMeasureState,
    selected_indices: torch.Tensor,
    *,
    nonvisual_z: torch.Tensor,
    nonvisual_n: torch.Tensor,
) -> dict[str, float]:
    head_count, group_count = state.exact_group_z.shape
    selected = torch.zeros(
        (head_count, group_count),
        device=state.exact_group_z.device,
        dtype=torch.bool,
    )
    selected.scatter_(1, selected_indices, True)
    compact_active = ~selected
    active_compact_log_z = torch.where(
        compact_active,
        state.coarse_log_group_z,
        torch.full_like(state.coarse_log_group_z, -torch.inf),
    )
    compact_shift = active_compact_log_z.max(dim=1).values
    active_shift = torch.where(
        compact_active.any(dim=1),
        compact_shift.clamp_min(0.0),
        torch.zeros_like(compact_shift),
    )
    exact_scale = torch.exp(-active_shift)
    exact_group_z = state.exact_group_z * exact_scale.unsqueeze(-1)
    exact_group_n = state.exact_group_n * exact_scale[:, None, None]
    coarse_group_z = torch.exp(state.coarse_log_group_z - active_shift.unsqueeze(-1))
    coarse_group_n = coarse_group_z.unsqueeze(-1) * state.coarse_mean_value
    visual_z = torch.where(selected, exact_group_z, coarse_group_z).sum(dim=1)
    visual_n = torch.where(selected.unsqueeze(-1), exact_group_n, coarse_group_n).sum(
        dim=1
    )
    scaled_nonvisual_z = nonvisual_z * exact_scale
    scaled_nonvisual_n = nonvisual_n * exact_scale.unsqueeze(-1)
    visual_output = visual_n / visual_z.unsqueeze(-1)
    full_output = (visual_n + scaled_nonvisual_n) / (
        visual_z + scaled_nonvisual_z
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
    return {
        "visual_relative_l2": float(visual_relative.item()),
        "visual_worst_head_relative_l2": float(head_error.max().item()),
        "full_relative_l2": float(full_relative.item()),
    }


def build_gaussian_components(
    capture: AttentionCapture,
    *,
    visual_start: int,
    visual_token_count: int,
    group_offsets: torch.Tensor,
    max_rank: int,
) -> GaussianComponents:
    _, projected_error = replay_attention(capture)
    query = capture.query[:, -1].float()
    key = capture.key.float()
    value = capture.value.float()
    query_scaled = query * float(capture.module.scaling)
    scores = torch.einsum("hd,hsd->hs", query_scaled, key)
    if capture.attention_mask is not None:
        scores = scores + capture.attention_mask[:, -1].float()
    maximum = scores.max(dim=1, keepdim=True).values
    exponentials = torch.exp(scores - maximum)

    visual_stop = visual_start + visual_token_count
    visual_key = key[:, visual_start:visual_stop]
    visual_value = value[:, visual_start:visual_stop]
    visual_scores = scores[:, visual_start:visual_stop]
    visual_exp = exponentials[:, visual_start:visual_stop]
    offsets = group_offsets.to(device=key.device)
    head_count, _, head_dim = visual_value.shape
    group_count, group_size = offsets.shape
    flat_offsets = offsets.flatten()
    member_key = visual_key.index_select(1, flat_offsets).reshape(
        head_count, group_count, group_size, head_dim
    )
    member_value = visual_value.index_select(1, flat_offsets).reshape(
        head_count, group_count, group_size, head_dim
    )
    member_scores = visual_scores.index_select(1, flat_offsets).reshape(
        head_count, group_count, group_size
    )
    member_exp = visual_exp.index_select(1, flat_offsets).reshape(
        head_count, group_count, group_size
    )
    exact_group_z = member_exp.sum(dim=2)
    exact_group_n = torch.einsum("hgm,hgmd->hgd", member_exp, member_value)

    (
        mean_key,
        mean_value,
        eigenvalues,
        cross_value_key,
        query_coordinates,
    ) = low_rank_gaussian_parameters(
        member_key,
        member_value,
        query_scaled=query_scaled,
        max_rank=max_rank,
    )
    score_center = member_scores.mean(dim=2)

    nonvisual_mask = torch.ones(scores.shape[1], device=key.device, dtype=torch.bool)
    nonvisual_mask[visual_start:visual_stop] = False
    nonvisual_exp = exponentials[:, nonvisual_mask]
    nonvisual_value = value[:, nonvisual_mask]
    nonvisual_z = nonvisual_exp.sum(dim=1)
    nonvisual_n = torch.einsum("hs,hsd->hd", nonvisual_exp, nonvisual_value)
    exact_visual_z = exact_group_z.sum(dim=1)
    exact_visual_n = exact_group_n.sum(dim=1)
    exact_visual_output = exact_visual_n / exact_visual_z.unsqueeze(-1)
    exact_full_output = (exact_visual_n + nonvisual_n) / (
        exact_visual_z + nonvisual_z
    ).unsqueeze(-1)
    return GaussianComponents(
        exact_visual_output=exact_visual_output,
        exact_full_output=exact_full_output,
        exact_group_z=exact_group_z,
        exact_group_n=exact_group_n,
        nonvisual_z=nonvisual_z,
        nonvisual_n=nonvisual_n,
        mean_value=mean_value,
        score_center=score_center,
        maximum=maximum,
        eigenvalues=eigenvalues,
        cross_value_key=cross_value_key,
        query_coordinates=query_coordinates,
        group_size=group_size,
        # The measure reference intentionally recomputes captured BF16 Q/K/V in
        # FP32. Capture validity is checked against the model's native BF16
        # replay before this distinct numerical reference is constructed.
        replay_error=float(projected_error.item()),
        mean_key=mean_key,
        key_min=member_key.amin(dim=2),
        key_max=member_key.amax(dim=2),
        query_scaled=query_scaled,
        visual_value_norm_max=torch.linalg.vector_norm(visual_value, dim=-1).amax(
            dim=1
        ),
        member_key=member_key,
        member_value=member_value,
    )


def active_read_ratio(
    *,
    group_count: int,
    group_size: int,
    head_dim: int,
    rank: int,
    exact_group_count: int,
) -> tuple[int, int, int, float]:
    dense = group_count * group_size * 2 * head_dim
    moment = group_count * (2 * head_dim + rank * (2 * head_dim + 1))
    exact = exact_group_count * group_size * 2 * head_dim
    total = moment + exact
    return dense, moment, exact, dense / total


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["topology"]),
                int(row["rank"]),
                str(row["selector"]),
                float(row["exact_fraction"]),
            )
            for row in rows
        }
    )
    summaries: list[dict[str, object]] = []
    for topology, rank, selector, exact_fraction in keys:
        selected = [
            row
            for row in rows
            if row["topology"] == topology
            and row["rank"] == rank
            and row["selector"] == selector
            and row["exact_fraction"] == exact_fraction
        ]
        visual = np.asarray([float(row["visual_relative_l2"]) for row in selected])
        full = np.asarray([float(row["full_relative_l2"]) for row in selected])
        summaries.append(
            {
                "topology": topology,
                "rank": rank,
                "selector": selector,
                "exact_fraction": exact_fraction,
                "exact_group_count": int(selected[0]["exact_group_count"]),
                "cell_count": len(selected),
                "active_read_ratio": float(selected[0]["active_read_ratio"]),
                "visual_mean": float(visual.mean()),
                "visual_p95": float(np.quantile(visual, 0.95)),
                "visual_worst": float(visual.max()),
                "visual_worst_head": max(
                    float(row["visual_worst_head_relative_l2"]) for row in selected
                ),
                "full_mean": float(full.mean()),
                "full_p95": float(np.quantile(full, 0.95)),
                "full_worst": float(full.max()),
            }
        )
    return summaries


def compact_pass(summary: dict[str, object]) -> bool:
    return (
        int(summary["cell_count"]) == 72
        and float(summary["exact_fraction"]) <= 0.25
        and float(summary["active_read_ratio"]) >= 2.0
        and float(summary["visual_mean"]) <= 0.01
        and float(summary["visual_p95"]) <= 0.02
        and float(summary["visual_worst"]) <= 0.05
        and float(summary["full_mean"]) <= 0.005
        and float(summary["full_p95"]) <= 0.01
    )


def oracle_pass(summary: dict[str, object]) -> bool:
    return (
        int(summary["cell_count"]) == 72
        and float(summary["exact_fraction"]) <= 0.25
        and float(summary["active_read_ratio"]) >= 2.0
        and float(summary["visual_mean"]) <= 0.005
        and float(summary["visual_p95"]) <= 0.01
        and float(summary["visual_worst"]) <= 0.02
        and float(summary["full_mean"]) <= 0.0025
        and float(summary["full_p95"]) <= 0.005
    )


def classify_outcome(
    summaries: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    compact = [
        row
        for row in summaries
        if row["selector"] == "compact_mass" and compact_pass(row)
    ]
    oracle = [
        row
        for row in summaries
        if row["selector"] == "oracle_local" and oracle_pass(row)
    ]
    eligible = [
        row
        for row in summaries
        if float(row["exact_fraction"]) <= 0.25
        and float(row["active_read_ratio"]) >= 2.0
    ]
    best = min(eligible, key=lambda row: float(row["visual_mean"]))
    diagnostics = {
        "compact_pass_count": len(compact),
        "oracle_pass_count": len(oracle),
        "best_eligible": best,
    }
    if compact:
        return "POSITIVE_GAUSSIAN_COMPACT_PATH", diagnostics
    if oracle:
        return "POSITIVE_GAUSSIAN_CAPACITY_ONLY", diagnostics
    return "NO_POSITIVE_GAUSSIAN_MEASURE_PATH", diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty Gaussian measure rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    previous = json.loads(args.headwise_summary.read_text(encoding="utf-8"))
    if previous["decision"] != "HEADWISE_SUPPORT_PARTIAL":
        raise ValueError("headwise-support prerequisite decision changed")
    if args.sample_offset != 72 or args.sample_count != 24:
        raise ValueError("registered Gate is restricted to calibration positions 73-96")
    if args.frame_budget != 8:
        raise ValueError("registered frame budget changed")

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    selected_samples = select_calibration_questions(
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
    rows: list[dict[str, object]] = []
    maximum_replay_error = 0.0
    started = time.perf_counter()

    for sample_position, sample in enumerate(
        selected_samples, start=args.sample_offset + 1
    ):
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
        for topology in TOPOLOGIES:
            offsets = hierarchical_group_offsets(
                frame_count=reference.shape[0],
                token_count=reference.shape[1],
                topology=topology,
                device=device,
            )
            group_count, group_size = offsets.shape
            for layer_index in LAYERS:
                components = build_gaussian_components(
                    capture.captures[layer_index],
                    visual_start=visual_start,
                    visual_token_count=visual_token_count,
                    group_offsets=offsets,
                    max_rank=max(RANKS),
                )
                maximum_replay_error = max(
                    maximum_replay_error, components.replay_error
                )
                head_dim = components.exact_group_n.shape[-1]
                for rank in RANKS:
                    state = positive_gaussian_measure(components, rank)
                    orders = {
                        "compact_mass": torch.argsort(
                            state.mass_priority,
                            dim=1,
                            descending=True,
                            stable=True,
                        ),
                        "oracle_local": torch.argsort(
                            state.oracle_priority,
                            dim=1,
                            descending=True,
                            stable=True,
                        ),
                    }
                    for selector in SELECTORS:
                        for exact_fraction in EXACT_FRACTIONS:
                            exact_group_count = int(round(group_count * exact_fraction))
                            metrics = evaluate_gaussian_support(
                                state,
                                orders[selector][:, :exact_group_count],
                                nonvisual_z=components.nonvisual_z,
                                nonvisual_n=components.nonvisual_n,
                            )
                            dense, moment, exact, ratio = active_read_ratio(
                                group_count=group_count,
                                group_size=group_size,
                                head_dim=head_dim,
                                rank=rank,
                                exact_group_count=exact_group_count,
                            )
                            rows.append(
                                {
                                    "sample_id": sample.sample_id,
                                    "sample_position": sample_position,
                                    "layer_index": layer_index,
                                    "topology": topology,
                                    "group_count": group_count,
                                    "group_size": group_size,
                                    "rank": rank,
                                    "selector": selector,
                                    "exact_fraction": exact_fraction,
                                    "exact_group_count": exact_group_count,
                                    "dense_read_floats_per_head": dense,
                                    "moment_read_floats_per_head": moment,
                                    "exact_read_floats_per_head": exact,
                                    "active_read_ratio": ratio,
                                    **metrics,
                                }
                            )
        print(
            json.dumps(
                {
                    "event": "positive_gaussian_measure_sample_ok",
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
    full_exact_rows = [row for row in rows if row["exact_fraction"] == 1.0]
    if (
        not full_exact_rows
        or max(
            max(
                float(row["visual_relative_l2"]),
                float(row["full_relative_l2"]),
            )
            for row in full_exact_rows
        )
        > 1e-7
    ):
        raise RuntimeError("full exact replacement did not reproduce the measure")
    summaries = summarize_rows(rows)
    decision, diagnostics = classify_outcome(summaries)
    write_csv(args.out_dir / "positive_gaussian_measure_rows.csv", rows)
    write_csv(args.out_dir / "positive_gaussian_measure_summary.csv", summaries)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "sample_positions": [
            args.sample_offset + 1,
            args.sample_offset + args.sample_count,
        ],
        "sample_count": len(selected_samples),
        "layers": list(LAYERS),
        "topologies": list(TOPOLOGIES),
        "ranks": list(RANKS),
        "selectors": list(SELECTORS),
        "exact_fractions": list(EXACT_FRACTIONS),
        "maximum_replay_error": maximum_replay_error,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Fixed-query, single-layer positive Gaussian attention-measure "
            "diagnostic on exposed calibration positions 73-96. Active-read "
            "ratios are arithmetic proxies, not reader, TTFT, or latency claims."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
