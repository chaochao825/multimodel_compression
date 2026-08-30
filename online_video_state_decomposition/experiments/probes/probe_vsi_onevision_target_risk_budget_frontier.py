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
    first_token_logits_from_variable_video_tokens,
    load_onevision_model,
)
from mvbench_reader_quotient_support_oracle import candidate_token_ids
from mvbench_utils import decode_video_frames, uniform_frame_indices
from onevision_reader_quotient_stage_a import descending_eigenspace
from probe_vsi_onevision_cmrq_stage_b import candidate_kl, feature_path_for_sample
from probe_vsi_onevision_query_group_fallback_transfer import summarize_raw
from probe_vsi_onevision_reader_risk_stage_a import (
    candidate_margins,
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import (
    contiguous_group_means,
    hybrid_group_tokens,
    normalized_adverse_group_risk,
)
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


REGISTERED_GROUP_COUNTS = (0, 49, 98, 128, 160, 196, 245, 294, 343, 392)


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


def hybrid_token_count(
    *,
    group_count: int,
    group_size: int,
    refined_group_count: int,
) -> int:
    if not 0 <= refined_group_count <= group_count:
        raise ValueError("refined group count is outside the available groups")
    return group_count + refined_group_count * (group_size - 1)


def risk_mass_capture(risk: torch.Tensor, selected: torch.Tensor) -> float:
    total = risk.sum()
    if selected.numel() == 0:
        return 0.0
    return float(
        (risk.index_select(0, selected).sum() / total.clamp_min(1e-12)).item()
    )


def budget_decision(
    summaries: dict[int, dict[str, float | int]],
) -> tuple[str, int | None]:
    strong = [
        group_count
        for group_count, metrics in summaries.items()
        if group_count <= 196
        and int(metrics["mismatch_count"]) == 0
        and int(metrics["harmful_count"]) == 0
        and float(metrics["candidate_kl_mean"]) <= 0.01
        and float(metrics["candidate_kl_p95"]) <= 0.02
    ]
    if strong:
        return "STRONG_CAPACITY_WINDOW", min(strong)
    weak = [
        group_count
        for group_count, metrics in summaries.items()
        if group_count <= 294
        and int(metrics["mismatch_count"]) == 0
        and int(metrics["harmful_count"]) == 0
    ]
    if weak:
        return "WEAK_CAPACITY_WINDOW", min(weak)
    return "NO_USEFUL_CAPACITY_WINDOW", None


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
        raise ValueError("registered frontier requires eight frames and group_size=4")
    if args.sample_offset != 72 or args.sample_count != 24:
        raise ValueError("registered frontier is restricted to positions 73-96")
    if args.sample_offset + args.sample_count > 96:
        raise ValueError("frontier cannot read calibration positions after 96")

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

    full_token_count = args.frame_budget * 196
    group_count = full_token_count // args.group_size
    if group_count != REGISTERED_GROUP_COUNTS[-1]:
        raise ValueError("registered group-count frontier does not match feature shape")

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    maximum_dense_equivalence_error = 0.0
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
                logits = first_token_logits_from_variable_video_tokens(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    video_tokens=hybrid_group_tokens(
                        exact_groups,
                        approximate_means,
                        selected_indices,
                    ),
                )
                candidate_logits = logits.float().index_select(0, token_tensor)
                approximate_index = int(torch.argmax(candidate_logits).item())
                sorted_candidates = torch.sort(candidate_logits, descending=True).values
                token_count = hybrid_token_count(
                    group_count=group_count,
                    group_size=args.group_size,
                    refined_group_count=refined_group_count,
                )
                dense_equivalence_error = (
                    float((candidate_logits - reference_candidates).abs().max().item())
                    if refined_group_count == group_count
                    else 0.0
                )
                maximum_dense_equivalence_error = max(
                    maximum_dense_equivalence_error,
                    dense_equivalence_error,
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "sample_position": args.sample_offset + sample_position,
                        "refined_group_count": refined_group_count,
                        "teacher_index": teacher_index,
                        "answer_index": sample.answer_index,
                        "approximate_index": approximate_index,
                        "candidate_kl": candidate_kl(
                            reference_candidates,
                            candidate_logits,
                        ),
                        "prediction_match": int(approximate_index == teacher_index),
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
                        "token_retention": token_count / full_token_count,
                        "dense_equivalence_error": dense_equivalence_error,
                    }
                )

        print(
            json.dumps(
                {
                    "event": "target_risk_budget_frontier_ok",
                    "position": sample_position,
                    "total": len(samples),
                    "sample_id": sample.sample_id,
                }
            ),
            flush=True,
        )

    if maximum_dense_equivalence_error > 1e-5:
        raise RuntimeError(
            "dense frontier endpoint changed candidate logits by "
            f"{maximum_dense_equivalence_error}"
        )

    summaries = {
        group_count_value: {
            **summarize_raw(
                [
                    row
                    for row in rows
                    if int(row["refined_group_count"]) == group_count_value
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
            / full_token_count,
            "mean_target_risk_mass_capture": float(
                np.mean(
                    [
                        float(row["target_risk_mass_capture"])
                        for row in rows
                        if int(row["refined_group_count"]) == group_count_value
                    ]
                )
            ),
        }
        for group_count_value in REGISTERED_GROUP_COUNTS
    }
    decision, first_passing_group_count = budget_decision(summaries)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "exposed_prospective_target_risk_budget_frontier",
        "sample_positions": [73, 96],
        "sample_count": len(samples),
        "group_size": args.group_size,
        "group_counts": list(REGISTERED_GROUP_COUNTS),
        "budget_summaries": {str(key): value for key, value in summaries.items()},
        "decision": decision,
        "first_passing_group_count": first_passing_group_count,
        "maximum_dense_equivalence_error": maximum_dense_equivalence_error,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Exploratory capacity diagnostic on already exposed calibration positions "
            "73-96. Positions 97-120, selection, and formal remain unread. Target "
            "gradients are unavailable at deployment."
        ),
    }
    write_csv(args.out_dir / "frontier_rows.csv", rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
