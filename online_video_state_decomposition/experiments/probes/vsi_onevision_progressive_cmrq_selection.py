from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
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
from onevision_reader_quotient_stage_a import FeatureStatistics, centered_covariance
from probe_vsi_onevision_cmrq_stage_b import (
    CodecCandidate,
    build_candidates,
    candidate_kl,
    feature_path_for_sample,
)
from probe_vsi_onevision_reader_risk_stage_a import (
    candidate_margins,
    reconstruct,
    select_role_questions,
)
from reader_quotient_cmrq_stage_b import (
    DomainMoments,
    orthogonality_error,
    summarize_exact_rows,
    summarize_progressive_fallback,
    trace_capture,
)
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


FROZEN_METHODS = (
    "pooled3_pca_r456",
    "vsi_pca_train96_r456",
    "cmrq_risk_atoms32_r456",
    "cmrq_mix_g32_w0p3_r456",
    "permuted_mix_g32_w0p3_r456",
)
METHOD_ALIASES = {"vsi_pca_train96_r456": "vsi_pca_cal120_r456"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--calibration-feature-dir", type=Path, required=True)
    parser.add_argument("--evaluation-feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--spectral-artifact", type=Path, required=True)
    parser.add_argument("--reader-risk-artifact", type=Path, required=True)
    parser.add_argument("--reader-risk-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--margin-floor", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--compressed-state-bytes", type=int, default=2_860_032)
    parser.add_argument("--dense-state-bytes", type=int, default=22_478_848)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calibration_moments(
    feature_dir: Path,
    *,
    expected_sample_ids: set[str],
    device: torch.device,
) -> DomainMoments:
    paths = sorted(feature_dir.glob("*.pt"))
    if len(paths) != len(expected_sample_ids):
        raise ValueError("calibration feature count does not match the frozen split")
    rows = 0
    feature_sum = None
    gram = None
    observed_ids = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["protocol_id"] != PROTOCOL_ID or payload["role"] != "calibration":
            raise ValueError(f"invalid calibration feature payload: {path}")
        sample_id = str(payload["sample_id"])
        if sample_id in observed_ids:
            raise ValueError(f"duplicate calibration feature sample: {sample_id}")
        observed_ids.add(sample_id)
        features = payload["features"].to(device=device, dtype=torch.float32)
        matrix = features.reshape(-1, features.shape[-1])
        if feature_sum is None:
            feature_sum = torch.zeros(
                matrix.shape[-1], device=device, dtype=torch.float32
            )
            gram = torch.zeros(
                matrix.shape[-1],
                matrix.shape[-1],
                device=device,
                dtype=torch.float32,
            )
        rows += matrix.shape[0]
        feature_sum.add_(matrix.sum(dim=0))
        gram.add_(matrix.transpose(0, 1) @ matrix)
    if observed_ids != expected_sample_ids or feature_sum is None or gram is None:
        raise ValueError("calibration feature identities do not match the frozen split")
    statistics = FeatureStatistics(rows=rows, feature_sum=feature_sum, gram=gram)
    return DomainMoments(
        mean=feature_sum / rows,
        covariance=centered_covariance(statistics),
    )


def frozen_candidates(
    spectral: dict[str, dict[str, object]],
    risk_artifact: dict[str, torch.Tensor],
    calibration: DomainMoments,
    *,
    rank: int,
    seed: int,
    device: torch.device,
) -> tuple[list[CodecCandidate], DomainMoments, torch.Tensor]:
    candidates, pooled, risk = build_candidates(
        spectral,
        risk_artifact,
        calibration,
        rank=rank,
        atom_counts=(32,),
        null_atom_counts=(32,),
        mix_atom_count=32,
        mix_weights=(0.3,),
        seed=seed,
        device=device,
    )
    by_name = {candidate.name: candidate for candidate in candidates}
    frozen = []
    for name in FROZEN_METHODS:
        if name not in by_name:
            raise ValueError(f"missing frozen CMRQ candidate: {name}")
        candidate = by_name[name]
        if name in METHOD_ALIASES:
            candidate = replace(candidate, name=METHOD_ALIASES[name])
        frozen.append(candidate)
    return frozen, pooled, risk


def paired_bootstrap_delta(
    left: list[float],
    right: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if len(left) != len(right) or not left or replicates <= 0:
        raise ValueError("paired bootstrap inputs are invalid")
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, delta.size, size=(replicates, delta.size))
    means = delta[indices].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def progressive_task_summary(
    rows: list[dict[str, object]],
    *,
    margin_threshold: float,
    compressed_state_bytes: int,
    dense_state_bytes: int,
) -> dict[str, float | int]:
    summary = summarize_progressive_fallback(
        rows,
        margin_threshold=margin_threshold,
        compressed_state_bytes=compressed_state_bytes,
        dense_state_bytes=dense_state_bytes,
    )
    fallback = [
        float(row["approximate_top1_margin"]) <= margin_threshold for row in rows
    ]
    delivered_match = [
        1 if exact else int(row["prediction_match"])
        for row, exact in zip(rows, fallback, strict=True)
    ]
    delivered_correct = [
        int(row["baseline_correct"])
        if exact
        else int(row["approximate_correct"])
        for row, exact in zip(rows, fallback, strict=True)
    ]
    baseline_correct = [int(row["baseline_correct"]) for row in rows]
    summary.update(
        {
            "delivered_agreement": float(np.mean(delivered_match)),
            "baseline_accuracy": float(np.mean(baseline_correct)),
            "delivered_accuracy": float(np.mean(delivered_correct)),
            "delivered_accuracy_delta": float(
                np.mean(delivered_correct) - np.mean(baseline_correct)
            ),
        }
    )
    return summary


def selection_gate(
    *,
    method_summaries: dict[str, dict[str, float | int]],
    progressive: dict[str, float | int],
    paired_delta: dict[str, float],
) -> dict[str, object]:
    mix = method_summaries["cmrq_mix_g32_w0p3_r456"]
    permuted = method_summaries["permuted_mix_g32_w0p3_r456"]
    vsi = method_summaries["vsi_pca_cal120_r456"]
    conditions = {
        "mix_beats_permuted_mean_with_paired_ci": paired_delta["ci95_high"] < 0.0,
        "mix_beats_permuted_p95": mix["candidate_kl_p95"] < permuted["candidate_kl_p95"],
        "mix_not_worse_than_vsi_mean": mix["candidate_kl_mean"] <= vsi["candidate_kl_mean"],
        "mix_not_worse_than_vsi_p95": mix["candidate_kl_p95"] <= vsi["candidate_kl_p95"],
        "progressive_harmful_zero": progressive["remaining_harmful_count"] == 0,
        "progressive_agreement_at_least_98pct": progressive["delivered_agreement"] >= 0.98,
        "fallback_at_most_15pct": progressive["fallback_rate"] <= 0.15,
        "conservative_transfer_at_least_4x": progressive["conservative_transfer_ratio"] >= 4.0,
        "task_accuracy_loss_at_most_1point": progressive["delivered_accuracy_delta"] >= -0.01,
    }
    return {
        "decision": "GO" if all(conditions.values()) else "NO_GO",
        "conditions": conditions,
    }


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    calibration_samples = select_role_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        role="calibration",
        sample_count=None,
    )
    evaluation = select_role_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        role="selection",
        sample_count=None,
    )
    calibration_scene_ids = {str(scene["sample_id"]) for scene in split["roles"]["calibration"]}
    evaluation_scene_ids = {str(scene["sample_id"]) for scene in split["roles"]["selection"]}
    if calibration_scene_ids & evaluation_scene_ids:
        raise ValueError("calibration and selection scenes overlap")

    risk_summary = json.loads(args.reader_risk_summary.read_text(encoding="utf-8"))
    if risk_summary["protocol_id"] != PROTOCOL_ID:
        raise ValueError("reader-risk protocol identity mismatch")
    calibration_question_ids = {sample.sample_id for sample in calibration_samples}
    if not set(risk_summary["sample_ids"]) <= calibration_question_ids:
        raise ValueError("reader-risk artifact contains a non-calibration question")
    if int(risk_summary["sample_count"]) != 72:
        raise ValueError("selection requires the frozen 72-question risk artifact")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    calibration = calibration_moments(
        args.calibration_feature_dir,
        expected_sample_ids=calibration_scene_ids,
        device=device,
    )
    evaluation_paths = {
        feature_path_for_sample(args.evaluation_feature_dir, sample)
        for sample in evaluation
    }
    if len(evaluation_paths) != len(evaluation):
        raise ValueError("selection feature identities are incomplete")
    spectral = torch.load(args.spectral_artifact, map_location="cpu", weights_only=False)
    risk_artifact = torch.load(
        args.reader_risk_artifact,
        map_location="cpu",
        weights_only=True,
    )
    candidates, pooled, risk = frozen_candidates(
        spectral,
        risk_artifact,
        calibration,
        rank=args.rank,
        seed=args.seed,
        device=device,
    )
    candidate_metadata = {
        candidate.name: {
            "rank": candidate.basis.shape[1],
            "atom_family": candidate.atom_family,
            "atom_count": candidate.atom_count,
            "pooled_feature_capture": trace_capture(pooled.covariance, candidate.basis),
            "reader_risk_capture": trace_capture(risk, candidate.basis),
            "orthogonality_error": orthogonality_error(candidate.basis),
        }
        for candidate in candidates
    }

    processor, model = load_onevision_model(args.model_dir, device=args.device)
    model_dtype = next(model.parameters()).dtype
    rows = []
    started = time.perf_counter()
    for position, sample in enumerate(evaluation, start=1):
        payload = torch.load(
            feature_path_for_sample(args.evaluation_feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        if payload["protocol_id"] != PROTOCOL_ID or payload["role"] != "selection":
            raise ValueError(f"invalid selection feature payload: {sample.sample_id}")
        pool_features = payload["features"]
        selected_positions = uniform_frame_indices(pool_features.shape[0], args.frame_budget)
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
        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        reference_candidate_logits = reference_logits.float().index_select(0, token_tensor)
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
        for candidate in candidates:
            approximate = reconstruct(
                reference,
                mean=candidate.mean,
                basis=candidate.basis,
            ).to(model_dtype)
            with torch.inference_mode():
                approximate_logits = first_token_logits_from_features(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    features=approximate,
                )
            approximate_candidate_logits = approximate_logits.float().index_select(
                0, token_tensor
            )
            approximate_index = int(torch.argmax(approximate_candidate_logits).item())
            top_two = torch.topk(approximate_candidate_logits, k=2).values
            approximate_margins = candidate_margins(
                approximate_logits,
                token_ids,
                teacher_index=teacher_index,
                competitor_indices=competitor_indices,
            )
            exact_shifts = approximate_margins - margins
            normalized_adverse = (-exact_shifts).clamp_min(0) / margins.clamp_min(
                args.margin_floor
            )
            approximate_correct = approximate_index == sample.answer_index
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "method": candidate.name,
                    "teacher_index": teacher_index,
                    "answer_index": sample.answer_index,
                    "approximate_index": approximate_index,
                    "approximate_top1_margin": float((top_two[0] - top_two[1]).item()),
                    "minimum_margin": float(margins.min().item()),
                    "feature_relative_l2": float(
                        (
                            torch.linalg.vector_norm(
                                approximate.float() - reference.float()
                            )
                            / torch.linalg.vector_norm(reference.float()).clamp_min(
                                torch.finfo(torch.float32).eps
                            )
                        ).item()
                    ),
                    "candidate_kl": candidate_kl(
                        reference_candidate_logits,
                        approximate_candidate_logits,
                    ),
                    "maximum_normalized_adverse_shift": float(
                        normalized_adverse.max().item()
                    ),
                    "prediction_match": int(approximate_index == teacher_index),
                    "baseline_correct": int(baseline_correct),
                    "approximate_correct": int(approximate_correct),
                    "harmful": int(baseline_correct and not approximate_correct),
                    "beneficial": int(not baseline_correct and approximate_correct),
                }
            )
        print(
            json.dumps(
                {
                    "event": "cmrq_selection_reader_ok",
                    "position": position,
                    "total": len(evaluation),
                    "sample_id": sample.sample_id,
                }
            ),
            flush=True,
        )

    method_summaries = {}
    for candidate in candidates:
        method_rows = [row for row in rows if row["method"] == candidate.name]
        exact = summarize_exact_rows(method_rows, margin_floor=args.margin_floor)
        exact.update(
            {
                "baseline_accuracy": float(
                    np.mean([int(row["baseline_correct"]) for row in method_rows])
                ),
                "candidate_accuracy": float(
                    np.mean([int(row["approximate_correct"]) for row in method_rows])
                ),
            }
        )
        method_summaries[candidate.name] = {
            **candidate_metadata[candidate.name],
            **exact,
        }
    mix_rows = [
        row for row in rows if row["method"] == "cmrq_mix_g32_w0p3_r456"
    ]
    progressive = progressive_task_summary(
        mix_rows,
        margin_threshold=args.margin_threshold,
        compressed_state_bytes=args.compressed_state_bytes,
        dense_state_bytes=args.dense_state_bytes,
    )
    by_method_sample = {
        method: {
            str(row["sample_id"]): float(row["candidate_kl"])
            for row in rows
            if row["method"] == method
        }
        for method in (
            "cmrq_mix_g32_w0p3_r456",
            "permuted_mix_g32_w0p3_r456",
        )
    }
    sample_order = sorted(by_method_sample["cmrq_mix_g32_w0p3_r456"])
    if sample_order != sorted(by_method_sample["permuted_mix_g32_w0p3_r456"]):
        raise ValueError("paired selection identities do not match")
    paired_delta = paired_bootstrap_delta(
        [
            by_method_sample["cmrq_mix_g32_w0p3_r456"][sample_id]
            for sample_id in sample_order
        ],
        [
            by_method_sample["permuted_mix_g32_w0p3_r456"][sample_id]
            for sample_id in sample_order
        ],
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    gate = selection_gate(
        method_summaries=method_summaries,
        progressive=progressive,
        paired_delta=paired_delta,
    )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "selection_only_frozen_progressive_cmrq",
        "selection_role_scene_count": len(split["roles"]["selection"]),
        "selection_evaluated_question_count": len(evaluation),
        "selection_sample_ids": [sample.sample_id for sample in evaluation],
        "calibration_feature_scene_count": len(calibration_scene_ids),
        "reader_risk_question_count": int(risk_summary["sample_count"]),
        "rank": args.rank,
        "frozen_methods": [candidate.name for candidate in candidates],
        "frozen_margin_threshold": args.margin_threshold,
        "method_summaries": method_summaries,
        "progressive_mix": progressive,
        "mix_vs_permuted_paired_kl": paired_delta,
        "selection_gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection behavior and state-transfer only; no formal, long-video token, "
            "reader-compute, TTFT, or latency claim."
        ),
    }
    write_csv(args.out_dir / "sample_metrics.csv", rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
