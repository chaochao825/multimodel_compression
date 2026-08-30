from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
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
from probe_vsi_onevision_reader_risk_stage_a import (
    candidate_margins,
    reconstruct,
    select_calibration_questions,
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
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--pooled-side", type=int, default=7)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--exact-frame-budget", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-token-retention", type=float, default=0.35)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def spatial_pool_frames(features: torch.Tensor, *, pooled_side: int) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("features must have shape [frames, tokens, hidden]")
    source_side = math.isqrt(features.shape[1])
    if source_side * source_side != features.shape[1]:
        raise ValueError("spatial token count must be square")
    if pooled_side <= 0 or source_side % pooled_side:
        raise ValueError("pooled side must divide the source spatial side")
    factor = source_side // pooled_side
    shaped = features.reshape(
        features.shape[0],
        pooled_side,
        factor,
        pooled_side,
        factor,
        features.shape[-1],
    )
    return shaped.mean(dim=(2, 4)).reshape(
        features.shape[0], pooled_side * pooled_side, features.shape[-1]
    )


def variable_logits(
    *,
    model: torch.nn.Module,
    prompt_batch: dict[str, torch.Tensor],
    frames: list[torch.Tensor],
) -> torch.Tensor:
    return first_token_logits_from_variable_video_tokens(
        model=model,
        input_ids=prompt_batch["input_ids"],
        attention_mask=prompt_batch["attention_mask"],
        video_tokens=torch.cat(frames, dim=0),
    )


def question_conditioned_frame_scores(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooled_frames: torch.Tensor,
) -> torch.Tensor:
    if input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError("question-conditioned scoring requires one prompt")
    video_positions = torch.nonzero(
        input_ids[0] == model.config.video_token_index,
        as_tuple=False,
    ).flatten()
    if video_positions.numel() == 0:
        raise ValueError("prompt contains no video placeholder")
    suffix_start = int(video_positions[-1].item()) + 1
    suffix_mask = attention_mask[0, suffix_start:].bool()
    suffix_ids = input_ids[0, suffix_start:][suffix_mask]
    if suffix_ids.numel() == 0:
        raise ValueError("prompt has no question suffix after video placeholders")
    text = model.get_input_embeddings()(suffix_ids).float().mean(dim=0)
    text = torch.nn.functional.normalize(text, dim=0)
    visual = torch.nn.functional.normalize(pooled_frames.float(), dim=-1)
    return torch.einsum("fth,h->ft", visual, text).amax(dim=1)


def evidence_combinations(frame_budget: int, exact_frame_budget: int) -> list[tuple[int, ...]]:
    if exact_frame_budget <= 0 or exact_frame_budget > frame_budget:
        raise ValueError("exact frame budget must be within the frame budget")
    return list(itertools.combinations(range(frame_budget), exact_frame_budget))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("method summary requires rows")
    kl = np.asarray([float(row["candidate_kl"]) for row in rows])
    match = np.asarray([int(row["prediction_match"]) for row in rows])
    baseline_correct = np.asarray([int(row["baseline_correct"]) for row in rows])
    approximate_correct = np.asarray(
        [int(row["approximate_correct"]) for row in rows]
    )
    return {
        "sample_count": len(rows),
        "candidate_kl_mean": float(kl.mean()),
        "candidate_kl_p95": float(np.quantile(kl, 0.95)),
        "agreement": float(match.mean()),
        "mismatch_count": int(len(rows) - match.sum()),
        "harmful_count": int(
            np.logical_and(baseline_correct == 1, approximate_correct == 0).sum()
        ),
        "baseline_accuracy": float(baseline_correct.mean()),
        "candidate_accuracy": float(approximate_correct.mean()),
        "token_count": int(rows[0]["token_count"]),
        "token_retention": float(rows[0]["token_retention"]),
    }


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    samples = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=args.sample_count,
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
    rows = []
    frame_rows = []
    full_token_count = args.frame_budget * 196
    pooled_token_count = args.frame_budget * args.pooled_side**2
    retrieved_token_count = pooled_token_count + args.exact_frame_budget * (
        196 - args.pooled_side**2
    )
    combinations = evidence_combinations(
        args.frame_budget,
        args.exact_frame_budget,
    )
    method_suffix = f"frame{args.exact_frame_budget}"
    started = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        pool_features = payload["features"]
        selected_positions = uniform_frame_indices(
            pool_features.shape[0], args.frame_budget
        )
        positions = torch.tensor(selected_positions, dtype=torch.long)
        reference = pool_features.index_select(0, positions).to(
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
        with torch.inference_mode():
            reference_logits = first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            )
            variable_reference_logits = variable_logits(
                model=model,
                prompt_batch=prompt_batch,
                frames=[reference[index] for index in range(args.frame_budget)],
            )
        equivalence_error = float(
            (reference_logits.float() - variable_reference_logits.float())
            .abs()
            .max()
            .item()
        )
        if equivalence_error > 1e-5:
            raise RuntimeError(
                f"variable-token full path changed logits by {equivalence_error}"
            )

        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        reference_candidate_logits = reference_logits.float().index_select(
            0, token_tensor
        )
        teacher_index = int(torch.argmax(reference_candidate_logits).item())
        competitor_indices = [
            index for index in range(len(token_ids)) if index != teacher_index
        ]
        margins = candidate_margins(
            reference_logits,
            token_ids,
            teacher_index=teacher_index,
            competitor_indices=competitor_indices,
        )
        baseline_correct = teacher_index == sample.answer_index

        approximate = reconstruct(reference, mean=moments.mean, basis=basis).to(
            model_dtype
        )
        exact_pooled = spatial_pool_frames(reference, pooled_side=args.pooled_side)
        approximate_pooled = spatial_pool_frames(
            approximate, pooled_side=args.pooled_side
        )
        with torch.inference_mode():
            exact_pooled_logits = variable_logits(
                model=model,
                prompt_batch=prompt_batch,
                frames=[
                    exact_pooled[index] for index in range(args.frame_budget)
                ],
            )
            quotient_pooled_logits = variable_logits(
                model=model,
                prompt_batch=prompt_batch,
                frames=[
                    approximate_pooled[index] for index in range(args.frame_budget)
                ],
            )
            retrieved_logits = {}
            for exact_indices in combinations:
                exact_index_set = set(exact_indices)
                retrieved_frames = [
                    reference[index]
                    if index in exact_index_set
                    else approximate_pooled[index]
                    for index in range(args.frame_budget)
                ]
                retrieved_logits[exact_indices] = variable_logits(
                    model=model,
                    prompt_batch=prompt_batch,
                    frames=retrieved_frames,
                )

        combination_kls = []
        residual_energy = torch.stack(
            [
                torch.linalg.vector_norm(
                    reference[index].float() - approximate[index].float()
                )
                for index in range(args.frame_budget)
            ]
        )
        query_scores = question_conditioned_frame_scores(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            pooled_frames=approximate_pooled,
        )
        for exact_indices, logits in retrieved_logits.items():
            candidate_logits = logits.float().index_select(0, token_tensor)
            value = candidate_kl(reference_candidate_logits, candidate_logits)
            combination_kls.append(value)
            frame_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "frame_indices": ";".join(map(str, exact_indices)),
                    "source_frame_indices": ";".join(
                        str(selected_frame_indices[index]) for index in exact_indices
                    ),
                    "candidate_kl": value,
                    "residual_energy_sum": float(
                        residual_energy[list(exact_indices)].sum().item()
                    ),
                    "query_score_sum": float(
                        query_scores[list(exact_indices)].sum().item()
                    ),
                }
            )
        oracle_indices = combinations[int(np.argmin(combination_kls))]
        residual_indices = tuple(
            sorted(
                torch.topk(residual_energy, k=args.exact_frame_budget).indices.tolist()
            )
        )
        query_indices = tuple(
            sorted(
                torch.topk(query_scores, k=args.exact_frame_budget).indices.tolist()
            )
        )
        methods = (
            ("exact_pool49", exact_pooled_logits, pooled_token_count, ()),
            ("quotient_pool49", quotient_pooled_logits, pooled_token_count, ()),
            (
                f"residual_energy_{method_suffix}",
                retrieved_logits[residual_indices],
                retrieved_token_count,
                residual_indices,
            ),
            (
                f"query_score_{method_suffix}",
                retrieved_logits[query_indices],
                retrieved_token_count,
                query_indices,
            ),
            (
                f"oracle_{method_suffix}",
                retrieved_logits[oracle_indices],
                retrieved_token_count,
                oracle_indices,
            ),
        )
        for method, logits, token_count, retrieved_indices in methods:
            candidate_logits = logits.float().index_select(0, token_tensor)
            approximate_index = int(torch.argmax(candidate_logits).item())
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "method": method,
                    "teacher_index": teacher_index,
                    "answer_index": sample.answer_index,
                    "approximate_index": approximate_index,
                    "candidate_kl": candidate_kl(
                        reference_candidate_logits, candidate_logits
                    ),
                    "minimum_margin": float(margins.min().item()),
                    "prediction_match": int(approximate_index == teacher_index),
                    "baseline_correct": int(baseline_correct),
                    "approximate_correct": int(
                        approximate_index == sample.answer_index
                    ),
                    "token_count": token_count,
                    "token_retention": token_count / full_token_count,
                    "retrieved_frame_indices": ";".join(
                        map(str, retrieved_indices)
                    ),
                    "variable_path_equivalence_error": equivalence_error,
                }
            )
        print(
            json.dumps(
                {
                    "event": "progressive_evidence_sample_ok",
                    "position": position,
                    "total": len(samples),
                    "sample_id": sample.sample_id,
                }
            ),
            flush=True,
        )

    method_names = (
        "exact_pool49",
        "quotient_pool49",
        f"residual_energy_{method_suffix}",
        f"query_score_{method_suffix}",
        f"oracle_{method_suffix}",
    )
    summaries = {
        method: method_summary([row for row in rows if row["method"] == method])
        for method in method_names
    }
    quotient_kl = float(summaries["quotient_pool49"]["candidate_kl_mean"])
    residual_kl = float(
        summaries[f"residual_energy_{method_suffix}"]["candidate_kl_mean"]
    )
    query_kl = float(summaries[f"query_score_{method_suffix}"]["candidate_kl_mean"])
    oracle_kl = float(summaries[f"oracle_{method_suffix}"]["candidate_kl_mean"])
    oracle_improvement = quotient_kl - oracle_kl
    selector_recovery = (
        (quotient_kl - residual_kl) / oracle_improvement
        if oracle_improvement > 0.0
        else 0.0
    )
    query_selector_recovery = (
        (quotient_kl - query_kl) / oracle_improvement
        if oracle_improvement > 0.0
        else 0.0
    )
    oracle = summaries[f"oracle_{method_suffix}"]
    conditions = {
        "oracle_halves_quotient_kl": oracle_kl <= 0.5 * quotient_kl,
        "oracle_agreement_at_least_98pct": float(oracle["agreement"]) >= 0.98,
        "oracle_harmful_zero": int(oracle["harmful_count"]) == 0,
        "retrieved_token_retention_within_budget": float(oracle["token_retention"])
        <= args.max_token_retention,
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "calibration_only_progressive_evidence_capacity",
        "sample_count": len(samples),
        "rank": args.rank,
        "pooled_side": args.pooled_side,
        "exact_frame_budget": args.exact_frame_budget,
        "max_token_retention": args.max_token_retention,
        "full_token_count": full_token_count,
        "method_summaries": summaries,
        "residual_selector_oracle_improvement_recovery": selector_recovery,
        "query_selector_oracle_improvement_recovery": query_selector_recovery,
        "capacity_gate": {
            "decision": "GO" if all(conditions.values()) else "NO_GO",
            "conditions": conditions,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            f"Calibration capacity only. oracle_{method_suffix} reads exact outcomes and is not "
            "deployable; no selection, formal, task-generalization, or speed claim."
        ),
    }
    write_csv(args.out_dir / "sample_metrics.csv", rows)
    write_csv(args.out_dir / "frame_metrics.csv", frame_rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
