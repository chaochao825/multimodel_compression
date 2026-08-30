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
    method_summary,
    question_conditioned_frame_scores,
)
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
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--refined-group-count", type=int, default=98)
    parser.add_argument("--margin-floor", type=float, default=0.05)
    parser.add_argument("--max-token-retention", type=float, default=0.45)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def contiguous_group_means(
    features: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 3:
        raise ValueError("features must have shape [frames, tokens, hidden]")
    if group_size <= 0 or features.shape[1] % group_size:
        raise ValueError("group size must divide tokens per frame")
    groups = features.reshape(
        features.shape[0],
        features.shape[1] // group_size,
        group_size,
        features.shape[-1],
    ).reshape(-1, group_size, features.shape[-1])
    return groups, groups.mean(dim=1)


def hybrid_group_tokens(
    exact_groups: torch.Tensor,
    approximate_means: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    if exact_groups.ndim != 3 or approximate_means.ndim != 2:
        raise ValueError("invalid exact-group or approximate-mean shape")
    if exact_groups.shape[0] != approximate_means.shape[0]:
        raise ValueError("group counts differ")
    selected = torch.zeros(
        exact_groups.shape[0],
        device=exact_groups.device,
        dtype=torch.bool,
    )
    selected[selected_indices] = True
    pieces = [
        exact_groups[index]
        if bool(selected[index].item())
        else approximate_means[index : index + 1]
        for index in range(exact_groups.shape[0])
    ]
    return torch.cat(pieces, dim=0)


def normalized_adverse_group_risk(
    gradients: torch.Tensor,
    exact_groups: torch.Tensor,
    approximate_means: torch.Tensor,
    margins: torch.Tensor,
    *,
    margin_floor: float,
) -> torch.Tensor:
    if gradients.ndim != 3:
        raise ValueError("gradients must have shape [competitors, tokens, hidden]")
    delta = approximate_means[:, None, :].float() - exact_groups.float()
    grouped_gradients = gradients.reshape(
        gradients.shape[0],
        exact_groups.shape[0],
        exact_groups.shape[1],
        exact_groups.shape[2],
    )
    shifts = torch.einsum("cgth,gth->cg", grouped_gradients, delta)
    denominators = margins.float().clamp_min(margin_floor)[:, None]
    return ((-shifts).clamp_min(0.0) / denominators).amax(dim=0)


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
    group_rows = []
    full_token_count = args.frame_budget * 196
    base_group_count = full_token_count // args.group_size
    hybrid_token_count = base_group_count + args.refined_group_count * (
        args.group_size - 1
    )
    if args.refined_group_count <= 0 or args.refined_group_count > base_group_count:
        raise ValueError("refined group count is outside the available groups")
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
        probe = reference.detach().float().requires_grad_(True)
        reference_logits = first_token_logits_from_features(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            features=probe,
        )
        with torch.inference_mode():
            variable_reference_logits = first_token_logits_from_variable_video_tokens(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                video_tokens=reference.reshape(-1, reference.shape[-1]),
            )
        equivalence_error = float(
            (reference_logits.detach().float() - variable_reference_logits.float())
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
        reference_candidate_logits = reference_logits.detach().float().index_select(
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
        gradients = []
        for competitor_position, margin in enumerate(margins):
            gradient = torch.autograd.grad(
                margin,
                probe,
                retain_graph=competitor_position + 1 < len(margins),
            )[0]
            gradients.append(gradient.detach().reshape(-1, gradient.shape[-1]))
        gradient_tensor = torch.stack(gradients).float()

        approximate = reconstruct(reference, mean=moments.mean, basis=basis).to(
            model_dtype
        )
        exact_groups, exact_means = contiguous_group_means(
            reference,
            group_size=args.group_size,
        )
        _, approximate_means = contiguous_group_means(
            approximate,
            group_size=args.group_size,
        )
        delta = approximate_means[:, None, :].float() - exact_groups.float()
        residual_scores = torch.linalg.vector_norm(delta, dim=(1, 2))
        query_scores = question_conditioned_frame_scores(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            pooled_frames=approximate_means[:, None, :],
        )
        risk_scores = normalized_adverse_group_risk(
            gradient_tensor,
            exact_groups,
            approximate_means,
            margins.detach(),
            margin_floor=args.margin_floor,
        )
        selections = {
            "residual_energy_groups": torch.topk(
                residual_scores, k=args.refined_group_count
            ).indices.sort().values,
            "query_score_groups": torch.topk(
                query_scores, k=args.refined_group_count
            ).indices.sort().values,
            "target_gradient_risk_groups": torch.topk(
                risk_scores, k=args.refined_group_count
            ).indices.sort().values,
        }
        with torch.inference_mode():
            method_logits = {
                "exact_group4": first_token_logits_from_variable_video_tokens(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    video_tokens=exact_means,
                ),
                "quotient_group4": first_token_logits_from_variable_video_tokens(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    video_tokens=approximate_means,
                ),
            }
            for method, selected_indices in selections.items():
                method_logits[method] = first_token_logits_from_variable_video_tokens(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    video_tokens=hybrid_group_tokens(
                        exact_groups,
                        approximate_means,
                        selected_indices,
                    ),
                )

        baseline_correct = teacher_index == sample.answer_index
        for method, logits in method_logits.items():
            candidate_logits = logits.float().index_select(0, token_tensor)
            approximate_index = int(torch.argmax(candidate_logits).item())
            token_count = (
                base_group_count
                if method in {"exact_group4", "quotient_group4"}
                else hybrid_token_count
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "method": method,
                    "teacher_index": teacher_index,
                    "answer_index": sample.answer_index,
                    "approximate_index": approximate_index,
                    "candidate_kl": candidate_kl(
                        reference_candidate_logits,
                        candidate_logits,
                    ),
                    "prediction_match": int(approximate_index == teacher_index),
                    "baseline_correct": int(baseline_correct),
                    "approximate_correct": int(
                        approximate_index == sample.answer_index
                    ),
                    "token_count": token_count,
                    "token_retention": token_count / full_token_count,
                    "variable_path_equivalence_error": equivalence_error,
                }
            )
        for group_index in range(base_group_count):
            group_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "group_index": group_index,
                    "frame_index": group_index // (196 // args.group_size),
                    "residual_energy": float(residual_scores[group_index].item()),
                    "query_score": float(query_scores[group_index].item()),
                    "target_gradient_risk": float(risk_scores[group_index].item()),
                    "selected_by_residual": int(
                        group_index in selections["residual_energy_groups"].tolist()
                    ),
                    "selected_by_query": int(
                        group_index in selections["query_score_groups"].tolist()
                    ),
                    "selected_by_risk": int(
                        group_index
                        in selections["target_gradient_risk_groups"].tolist()
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "event": "risk_guided_group_sample_ok",
                    "position": position,
                    "total": len(samples),
                    "sample_id": sample.sample_id,
                }
            ),
            flush=True,
        )

    method_names = (
        "exact_group4",
        "quotient_group4",
        "residual_energy_groups",
        "query_score_groups",
        "target_gradient_risk_groups",
    )
    summaries = {
        method: method_summary([row for row in rows if row["method"] == method])
        for method in method_names
    }
    quotient_kl = float(summaries["quotient_group4"]["candidate_kl_mean"])
    risk = summaries["target_gradient_risk_groups"]
    conditions = {
        "risk_halves_quotient_kl": float(risk["candidate_kl_mean"])
        <= 0.5 * quotient_kl,
        "risk_agreement_at_least_98pct": float(risk["agreement"]) >= 0.98,
        "risk_harmful_zero": int(risk["harmful_count"]) == 0,
        "token_retention_within_budget": float(risk["token_retention"])
        <= args.max_token_retention,
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "calibration_only_target_gradient_group_capacity",
        "sample_count": len(samples),
        "rank": args.rank,
        "group_size": args.group_size,
        "refined_group_count": args.refined_group_count,
        "full_token_count": full_token_count,
        "method_summaries": summaries,
        "capacity_gate": {
            "decision": "GO" if all(conditions.values()) else "NO_GO",
            "conditions": conditions,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Calibration capacity only. target_gradient_risk_groups reads exact "
            "full-reader gradients and is not deployable; no selection, formal, "
            "task-generalization, latency, or speed claim."
        ),
    }
    write_csv(args.out_dir / "sample_metrics.csv", rows)
    write_csv(args.out_dir / "group_metrics.csv", group_rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
