from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_utils import (
    build_prompt_batch,
    first_token_logits_from_features,
    load_onevision_model,
)
from mvbench_reader_quotient_support_oracle import candidate_token_ids
from mvbench_utils import decode_video_frames, uniform_frame_indices
from onevision_reader_quotient_stage_a import descending_eigenspace
from probe_vsi_onevision_cmrq_stage_b import candidate_kl, feature_path_for_sample
from probe_vsi_onevision_group_compaction_geometry import (
    REPRESENTATIVE_OFFSET,
    compact_group_tokens_and_offsets,
    first_token_logits_from_positioned_video_tokens,
)
from probe_vsi_onevision_query_group_fallback_transfer import summarize_raw
from probe_vsi_onevision_reader_risk_stage_a import (
    candidate_margins,
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import (
    contiguous_group_means,
    normalized_adverse_group_risk,
)
from probe_vsi_onevision_target_risk_budget_frontier import (
    REGISTERED_GROUP_COUNTS,
    hybrid_token_count,
    risk_mass_capture,
)
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


LAYOUTS = ("positioned_equal_mass", "positioned_group_mass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--margin-floor", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def compact_group_masses(
    *,
    group_count: int,
    group_size: int,
    selected_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    selected = torch.zeros(group_count, device=device, dtype=torch.bool)
    selected[selected_indices] = True
    pieces = [
        torch.ones(group_size, device=device, dtype=torch.float32)
        if bool(selected[index].item())
        else torch.tensor([float(group_size)], device=device)
        for index in range(group_count)
    ]
    return torch.cat(pieces)


def additive_causal_mass_mask(
    token_masses: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if token_masses.ndim != 1 or token_masses.numel() == 0:
        raise ValueError("token masses must be a non-empty vector")
    if not bool((token_masses > 0).all().item()):
        raise ValueError("token masses must be positive")
    length = token_masses.numel()
    allowed = torch.ones(
        (length, length),
        device=token_masses.device,
        dtype=torch.bool,
    ).tril_()
    mask = torch.full(
        (length, length),
        torch.finfo(dtype).min,
        device=token_masses.device,
        dtype=dtype,
    )
    key_bias = torch.log(token_masses.to(dtype=dtype)).unsqueeze(0).expand(length, -1)
    mask[allowed] = key_bias[allowed]
    return mask.unsqueeze(0).unsqueeze(0)


def first_token_logits_from_measure_tokens(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_tokens: torch.Tensor,
    video_position_offsets: torch.Tensor,
    video_token_masses: torch.Tensor,
    full_video_token_count: int,
) -> torch.Tensor:
    if input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError("measure readout requires a single prompt batch")
    if video_tokens.ndim != 2:
        raise ValueError("video tokens must have shape [tokens, hidden]")
    if video_position_offsets.shape != (video_tokens.shape[0],):
        raise ValueError("video offsets have the wrong shape")
    if video_token_masses.shape != (video_tokens.shape[0],):
        raise ValueError("video masses have the wrong shape")
    if video_position_offsets.numel() > 1 and not bool(
        (video_position_offsets[1:] > video_position_offsets[:-1]).all().item()
    ):
        raise ValueError("video offsets must be strictly increasing")

    video_mask = input_ids[0] == model.config.video_token_index
    placeholder_positions = torch.nonzero(video_mask, as_tuple=False).flatten()
    if placeholder_positions.numel() != full_video_token_count + 1:
        raise ValueError("full prompt video span does not match the registered length")
    start = int(placeholder_positions[0].item())
    stop = int(placeholder_positions[-1].item()) + 1
    if placeholder_positions.numel() != stop - start:
        raise ValueError("video placeholders must form one contiguous span")

    inputs_embeds = model.get_input_embeddings()(input_ids)
    newline = model.model.image_newline[None, :].to(
        device=video_tokens.device,
        dtype=video_tokens.dtype,
    )
    inserted = torch.cat((video_tokens, newline), dim=0).to(inputs_embeds.dtype)
    variable_embeds = torch.cat(
        (
            inputs_embeds[:, :start],
            inserted.unsqueeze(0),
            inputs_embeds[:, stop:],
        ),
        dim=1,
    )
    prefix_positions = torch.arange(start, device=input_ids.device, dtype=torch.long)
    visual_positions = start + video_position_offsets.to(input_ids.device)
    newline_position = torch.tensor(
        [start + full_video_token_count],
        device=input_ids.device,
        dtype=torch.long,
    )
    suffix_positions = torch.arange(
        stop,
        input_ids.shape[1],
        device=input_ids.device,
        dtype=torch.long,
    )
    position_ids = torch.cat(
        (
            prefix_positions,
            visual_positions,
            newline_position,
            suffix_positions,
        )
    ).unsqueeze(0)
    all_masses = torch.cat(
        (
            torch.ones(start, device=input_ids.device),
            video_token_masses.to(input_ids.device),
            torch.ones(1, device=input_ids.device),
            torch.ones(input_ids.shape[1] - stop, device=input_ids.device),
        )
    )
    if position_ids.shape[1] != variable_embeds.shape[1]:
        raise ValueError("position IDs and variable embeddings differ")
    prepared_mask = additive_causal_mass_mask(
        all_masses,
        dtype=inputs_embeds.dtype,
    )

    outputs = model(
        inputs_embeds=variable_embeds,
        attention_mask=prepared_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    return outputs.logits[0, -1]


def measure_decision(
    summaries: dict[str, dict[int, dict[str, float | int]]],
) -> tuple[str, int | None]:
    for group_count in sorted(summaries["positioned_group_mass"]):
        if group_count > 196:
            continue
        weighted = summaries["positioned_group_mass"][group_count]
        equal = summaries["positioned_equal_mass"][group_count]
        fidelity = (
            int(weighted["mismatch_count"]) == 0
            and int(weighted["harmful_count"]) == 0
            and float(weighted["candidate_kl_mean"]) <= 0.01
            and float(weighted["candidate_kl_p95"]) <= 0.02
            and float(weighted["candidate_kl_mean"])
            <= 0.8 * float(equal["candidate_kl_mean"])
        )
        if fidelity:
            return "MASS_FIDELITY_RECOVERY", group_count
    for group_count in sorted(summaries["positioned_group_mass"]):
        if group_count > 196:
            continue
        weighted = summaries["positioned_group_mass"][group_count]
        if (
            int(weighted["mismatch_count"]) == 0
            and int(weighted["harmful_count"]) == 0
        ):
            return "MASS_DECISION_ONLY", group_count
    return "NO_MASS_RECOVERY", None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.group_size != 4 or args.frame_budget != 8:
        raise ValueError("registered diagnostic requires eight frames and group_size=4")
    if args.sample_offset != 72 or args.sample_count != 24:
        raise ValueError("registered diagnostic is restricted to positions 73-96")

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    selected = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=args.sample_offset + args.sample_count,
    )
    samples = selected[args.sample_offset :]
    expected_calibration_ids = {
        str(scene["sample_id"]) for scene in split["roles"]["calibration"]
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    moments = calibration_moments(
        args.feature_dir,
        expected_sample_ids=expected_calibration_ids,
        device=device,
    )
    _, basis = descending_eigenspace(moments.covariance, rank=args.rank)
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    model_dtype = next(model.parameters()).dtype

    full_video_token_count = args.frame_budget * 196
    group_count = full_video_token_count // args.group_size
    rows: list[dict[str, object]] = []
    maximum_dense_equivalence_error = {layout: 0.0 for layout in LAYOUTS}
    maximum_equal_mask_equivalence_error = 0.0
    started = time.perf_counter()

    for sample_position, sample in enumerate(samples, start=1):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0], args.frame_budget
        )
        positions = torch.tensor(selected_positions, dtype=torch.long)
        reference = payload["features"].index_select(0, positions).to(
            device=device,
            dtype=model_dtype,
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

        probe = reference.detach().float().requires_grad_(True)
        reference_logits = first_token_logits_from_features(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            features=probe,
        )
        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        reference_candidates = reference_logits.detach().float().index_select(
            0, token_tensor
        )
        teacher_index = int(torch.argmax(reference_candidates).item())
        competitor_indices = [
            index for index in range(len(token_ids)) if index != teacher_index
        ]
        margins = candidate_margins(
            reference_logits,
            token_ids,
            teacher_index=teacher_index,
            competitor_indices=competitor_indices,
        )
        gradients = []
        for competitor_position, margin in enumerate(margins):
            gradients.append(
                torch.autograd.grad(
                    margin,
                    probe,
                    retain_graph=competitor_position + 1 < len(margins),
                )[0]
                .detach()
                .reshape(-1, reference.shape[-1])
            )
        gradient_tensor = torch.stack(gradients).float()
        approximate = reconstruct(reference, mean=moments.mean, basis=basis).to(
            model_dtype
        )
        exact_groups, _ = contiguous_group_means(
            reference,
            group_size=args.group_size,
        )
        _, approximate_means = contiguous_group_means(
            approximate,
            group_size=args.group_size,
        )
        risk = normalized_adverse_group_risk(
            gradient_tensor,
            exact_groups,
            approximate_means,
            margins.detach(),
            margin_floor=args.margin_floor,
        )
        order = torch.argsort(risk, descending=True, stable=True)
        baseline_correct = teacher_index == sample.answer_index

        with torch.inference_mode():
            for refined_group_count in REGISTERED_GROUP_COUNTS:
                selected_indices = order[:refined_group_count].sort().values
                video_tokens, position_offsets = compact_group_tokens_and_offsets(
                    exact_groups,
                    approximate_means,
                    selected_indices,
                    representative_offset=REPRESENTATIVE_OFFSET,
                )
                group_masses = compact_group_masses(
                    group_count=group_count,
                    group_size=args.group_size,
                    selected_indices=selected_indices,
                    device=device,
                )
                equal_masses = torch.ones_like(group_masses)
                positioned_logits = first_token_logits_from_positioned_video_tokens(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    video_tokens=video_tokens,
                    video_position_offsets=position_offsets,
                    full_video_token_count=full_video_token_count,
                )
                logits_by_layout = {
                    "positioned_equal_mass": first_token_logits_from_measure_tokens(
                        model=model,
                        input_ids=prompt_batch["input_ids"],
                        attention_mask=prompt_batch["attention_mask"],
                        video_tokens=video_tokens,
                        video_position_offsets=position_offsets,
                        video_token_masses=equal_masses,
                        full_video_token_count=full_video_token_count,
                    ),
                    "positioned_group_mass": first_token_logits_from_measure_tokens(
                        model=model,
                        input_ids=prompt_batch["input_ids"],
                        attention_mask=prompt_batch["attention_mask"],
                        video_tokens=video_tokens,
                        video_position_offsets=position_offsets,
                        video_token_masses=group_masses,
                        full_video_token_count=full_video_token_count,
                    ),
                }
                equal_mask_error = float(
                    (
                        logits_by_layout["positioned_equal_mass"]
                        .float()
                        .index_select(0, token_tensor)
                        - positioned_logits.float().index_select(0, token_tensor)
                    )
                    .abs()
                    .max()
                    .item()
                )
                maximum_equal_mask_equivalence_error = max(
                    maximum_equal_mask_equivalence_error,
                    equal_mask_error,
                )
                for layout, logits in logits_by_layout.items():
                    candidate_logits = logits.float().index_select(0, token_tensor)
                    approximate_index = int(torch.argmax(candidate_logits).item())
                    sorted_candidates = torch.sort(
                        candidate_logits,
                        descending=True,
                    ).values
                    dense_equivalence_error = (
                        float(
                            (candidate_logits - reference_candidates).abs().max().item()
                        )
                        if refined_group_count == group_count
                        else 0.0
                    )
                    maximum_dense_equivalence_error[layout] = max(
                        maximum_dense_equivalence_error[layout],
                        dense_equivalence_error,
                    )
                    token_count = hybrid_token_count(
                        group_count=group_count,
                        group_size=args.group_size,
                        refined_group_count=refined_group_count,
                    )
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "sample_position": args.sample_offset + sample_position,
                            "layout": layout,
                            "refined_group_count": refined_group_count,
                            "teacher_index": teacher_index,
                            "answer_index": sample.answer_index,
                            "approximate_index": approximate_index,
                            "candidate_kl": candidate_kl(
                                reference_candidates,
                                candidate_logits,
                            ),
                            "prediction_match": int(
                                approximate_index == teacher_index
                            ),
                            "baseline_correct": int(baseline_correct),
                            "approximate_correct": int(
                                approximate_index == sample.answer_index
                            ),
                            "compressed_top1_margin": float(
                                (sorted_candidates[0] - sorted_candidates[1]).item()
                            ),
                            "target_risk_mass_capture": risk_mass_capture(
                                risk,
                                selected_indices,
                            ),
                            "token_count": token_count,
                            "token_retention": token_count / full_video_token_count,
                            "equal_mask_equivalence_error": equal_mask_error,
                            "dense_equivalence_error": dense_equivalence_error,
                        }
                    )

        print(
            json.dumps(
                {
                    "event": "measure_preserving_compaction_ok",
                    "position": sample_position,
                    "total": len(samples),
                    "sample_id": sample.sample_id,
                }
            ),
            flush=True,
        )

    if maximum_equal_mask_equivalence_error > 1e-5:
        raise RuntimeError(
            "all-mass-one mask changed positioned logits by "
            f"{maximum_equal_mask_equivalence_error}"
        )
    if max(maximum_dense_equivalence_error.values()) > 1e-5:
        raise RuntimeError(
            "dense measure endpoint changed candidate logits: "
            f"{maximum_dense_equivalence_error}"
        )

    summaries = {
        layout: {
            group_count_value: {
                **summarize_raw(
                    [
                        row
                        for row in rows
                        if row["layout"] == layout
                        and int(row["refined_group_count"]) == group_count_value
                    ]
                ),
                "token_count": hybrid_token_count(
                    group_count=group_count,
                    group_size=args.group_size,
                    refined_group_count=group_count_value,
                ),
                "token_retention": hybrid_token_count(
                    group_count=group_count,
                    group_size=args.group_size,
                    refined_group_count=group_count_value,
                )
                / full_video_token_count,
                "mean_target_risk_mass_capture": float(
                    np.mean(
                        [
                            float(row["target_risk_mass_capture"])
                            for row in rows
                            if row["layout"] == layout
                            and int(row["refined_group_count"])
                            == group_count_value
                        ]
                    )
                ),
            }
            for group_count_value in REGISTERED_GROUP_COUNTS
        }
        for layout in LAYOUTS
    }
    decision, first_passing_group_count = measure_decision(summaries)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "exposed_prospective_measure_preserving_compaction",
        "sample_positions": [73, 96],
        "sample_count": len(samples),
        "representative_offset": REPRESENTATIVE_OFFSET,
        "quotient_mass": args.group_size,
        "layouts": list(LAYOUTS),
        "group_counts": list(REGISTERED_GROUP_COUNTS),
        "layout_summaries": {
            layout: {str(key): value for key, value in values.items()}
            for layout, values in summaries.items()
        },
        "decision": decision,
        "first_passing_group_count": first_passing_group_count,
        "maximum_equal_mask_equivalence_error": (
            maximum_equal_mask_equivalence_error
        ),
        "maximum_dense_equivalence_error": maximum_dense_equivalence_error,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Exploratory mechanism diagnostic on already exposed calibration "
            "positions 73-96. Proportional attention is prior art. Positions 97-120, "
            "selection, and formal remain unread."
        ),
    }
    write_csv(args.out_dir / "measure_rows.csv", rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
