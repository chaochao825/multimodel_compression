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
from probe_vsi_onevision_progressive_evidence_retrieval import (
    question_conditioned_frame_scores,
)
from probe_vsi_onevision_reader_risk_stage_a import (
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import (
    contiguous_group_means,
    hybrid_group_tokens,
)
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fit-count", type=int, default=24)
    parser.add_argument("--evaluation-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--refined-group-count", type=int, default=98)
    parser.add_argument("--max-fallback-rate", type=float, default=0.15)
    parser.add_argument("--max-token-retention", type=float, default=0.53)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def compressed_top1_margin(candidate_logits: torch.Tensor) -> float:
    if candidate_logits.numel() < 2:
        raise ValueError("at least two candidate logits are required")
    top = torch.topk(candidate_logits.float(), k=2).values
    return float((top[0] - top[1]).item())


def calibrate_mismatch_threshold(rows: list[dict[str, object]]) -> float:
    mismatch_margins = [
        float(row["compressed_top1_margin"])
        for row in rows
        if int(row["prediction_match"]) == 0
    ]
    if not mismatch_margins:
        return float("-inf")
    return max(mismatch_margins)


def summarize_raw(rows: list[dict[str, object]]) -> dict[str, float | int]:
    count = len(rows)
    if count == 0:
        raise ValueError("cannot summarize empty rows")
    matches = sum(int(row["prediction_match"]) for row in rows)
    harmful = sum(
        int(row["baseline_correct"]) == 1
        and int(row["approximate_correct"]) == 0
        for row in rows
    )
    return {
        "sample_count": count,
        "agreement": matches / count,
        "mismatch_count": count - matches,
        "harmful_count": int(harmful),
        "baseline_accuracy": sum(int(row["baseline_correct"]) for row in rows)
        / count,
        "candidate_accuracy": sum(
            int(row["approximate_correct"]) for row in rows
        )
        / count,
        "candidate_kl_mean": float(
            np.mean([float(row["candidate_kl"]) for row in rows])
        ),
        "candidate_kl_p95": float(
            np.quantile([float(row["candidate_kl"]) for row in rows], 0.95)
        ),
    }


def apply_fallback(
    rows: list[dict[str, object]],
    *,
    threshold: float,
    hybrid_token_count: int,
    full_token_count: int,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    delivered = []
    for row in rows:
        fallback = float(row["compressed_top1_margin"]) <= threshold
        delivered.append(
            {
                **row,
                "fallback": int(fallback),
                "delivered_prediction_match": 1
                if fallback
                else int(row["prediction_match"]),
                "delivered_correct": int(row["baseline_correct"])
                if fallback
                else int(row["approximate_correct"]),
                "delivered_candidate_kl": 0.0
                if fallback
                else float(row["candidate_kl"]),
            }
        )
    count = len(delivered)
    fallback_count = sum(int(row["fallback"]) for row in delivered)
    agreement = sum(
        int(row["delivered_prediction_match"]) for row in delivered
    ) / count
    remaining_mismatch_count = sum(
        1 - int(row["delivered_prediction_match"]) for row in delivered
    )
    harmful = sum(
        int(row["baseline_correct"]) == 1 and int(row["delivered_correct"]) == 0
        for row in delivered
    )
    baseline_accuracy = sum(
        int(row["baseline_correct"]) for row in delivered
    ) / count
    delivered_accuracy = sum(int(row["delivered_correct"]) for row in delivered) / count
    effective_tokens = (
        (count - fallback_count) * hybrid_token_count
        + fallback_count * full_token_count
    ) / count
    summary = {
        "sample_count": count,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / count,
        "delivered_agreement": agreement,
        "remaining_mismatch_count": remaining_mismatch_count,
        "remaining_harmful_count": int(harmful),
        "baseline_accuracy": baseline_accuracy,
        "delivered_accuracy": delivered_accuracy,
        "delivered_candidate_kl_mean": float(
            np.mean([float(row["delivered_candidate_kl"]) for row in delivered])
        ),
        "effective_token_count": effective_tokens,
        "effective_token_retention": effective_tokens / full_token_count,
        "ideal_reader_token_reduction": full_token_count / effective_tokens,
    }
    return delivered, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    total_count = args.fit_count + args.evaluation_count
    samples = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=total_count,
    )
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
    base_group_count = full_token_count // args.group_size
    hybrid_token_count = base_group_count + args.refined_group_count * (
        args.group_size - 1
    )
    rows = []
    started = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
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
        query_scores = question_conditioned_frame_scores(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            pooled_frames=approximate_means[:, None, :],
        )
        selected_indices = torch.topk(
            query_scores,
            k=args.refined_group_count,
        ).indices.sort().values
        hybrid_tokens = hybrid_group_tokens(
            exact_groups,
            approximate_means,
            selected_indices,
        )
        with torch.inference_mode():
            reference_logits = first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            )
            hybrid_logits = first_token_logits_from_variable_video_tokens(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                video_tokens=hybrid_tokens,
            )
        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        reference_candidates = reference_logits.float().index_select(0, token_tensor)
        hybrid_candidates = hybrid_logits.float().index_select(0, token_tensor)
        teacher_index = int(torch.argmax(reference_candidates).item())
        approximate_index = int(torch.argmax(hybrid_candidates).item())
        role = "fit" if position <= args.fit_count else "evaluation"
        rows.append(
            {
                "sample_id": sample.sample_id,
                "role": role,
                "teacher_index": teacher_index,
                "answer_index": sample.answer_index,
                "approximate_index": approximate_index,
                "compressed_top1_margin": compressed_top1_margin(hybrid_candidates),
                "candidate_kl": candidate_kl(
                    reference_candidates,
                    hybrid_candidates,
                ),
                "prediction_match": int(approximate_index == teacher_index),
                "baseline_correct": int(teacher_index == sample.answer_index),
                "approximate_correct": int(
                    approximate_index == sample.answer_index
                ),
            }
        )
        print(
            json.dumps(
                {
                    "event": "query_group_fallback_sample_ok",
                    "position": position,
                    "total": len(samples),
                    "sample_id": sample.sample_id,
                    "role": role,
                }
            ),
            flush=True,
        )

    fit_rows = [row for row in rows if row["role"] == "fit"]
    evaluation_rows = [row for row in rows if row["role"] == "evaluation"]
    threshold = calibrate_mismatch_threshold(fit_rows)
    fit_delivered, fit_progressive = apply_fallback(
        fit_rows,
        threshold=threshold,
        hybrid_token_count=hybrid_token_count,
        full_token_count=full_token_count,
    )
    evaluation_delivered, evaluation_progressive = apply_fallback(
        evaluation_rows,
        threshold=threshold,
        hybrid_token_count=hybrid_token_count,
        full_token_count=full_token_count,
    )
    conditions = {
        "delivered_agreement_at_least_98pct": float(
            evaluation_progressive["delivered_agreement"]
        )
        >= 0.98,
        "remaining_harmful_zero": int(
            evaluation_progressive["remaining_harmful_count"]
        )
        == 0,
        "fallback_rate_within_budget": float(
            evaluation_progressive["fallback_rate"]
        )
        <= args.max_fallback_rate,
        "effective_token_retention_within_budget": float(
            evaluation_progressive["effective_token_retention"]
        )
        <= args.max_token_retention,
        "task_accuracy_loss_at_most_one_point": float(
            evaluation_progressive["delivered_accuracy"]
        )
        >= float(evaluation_progressive["baseline_accuracy"]) - 0.01,
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "calibration_fit_to_fresh_calibration_evaluation",
        "fit_count": args.fit_count,
        "evaluation_count": args.evaluation_count,
        "rank": args.rank,
        "group_size": args.group_size,
        "refined_group_count": args.refined_group_count,
        "compressed_margin_threshold": threshold,
        "raw_fit": summarize_raw(fit_rows),
        "raw_evaluation": summarize_raw(evaluation_rows),
        "progressive_fit": fit_progressive,
        "progressive_evaluation": evaluation_progressive,
        "transfer_gate": {
            "decision": "GO" if all(conditions.values()) else "NO_GO",
            "conditions": conditions,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Calibration-to-calibration transfer only. The margin threshold is fit "
            "on the first 24 calibration questions and evaluated on the next 24; "
            "selection and formal roles remain unread. Token counts are an ideal "
            "reader-prefill proxy, not measured latency."
        ),
    }
    write_csv(args.out_dir / "raw_sample_metrics.csv", rows)
    write_csv(
        args.out_dir / "delivered_sample_metrics.csv",
        fit_delivered + evaluation_delivered,
    )
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
