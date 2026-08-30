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
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import contiguous_group_means
from probe_vsi_onevision_target_risk_budget_frontier import (
    REGISTERED_GROUP_COUNTS,
    hybrid_token_count,
)
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


ROLE = "exposed_reader_aligned_singleton_marginal"


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
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def stable_benefit_order(benefits: list[float]) -> list[int]:
    return sorted(range(len(benefits)), key=lambda index: (-benefits[index], index))


def candidate_top1_margin(candidate_logits: torch.Tensor) -> float:
    values = torch.sort(candidate_logits.float(), descending=True).values
    if values.numel() < 2:
        raise ValueError("at least two candidate logits are required")
    return float((values[0] - values[1]).item())


def path_regressions(
    rows: list[dict[str, object]],
    *,
    through_group_count: int,
) -> tuple[int, int]:
    selected = sorted(
        (
            row
            for row in rows
            if int(row["refined_group_count"]) <= through_group_count
        ),
        key=lambda row: int(row["refined_group_count"]),
    )
    match_regressions = 0
    mean_kl_regressions = 0
    for previous, current in zip(selected, selected[1:]):
        if int(previous["prediction_match"]) and not int(current["prediction_match"]):
            match_regressions += 1
        if float(current["candidate_kl"]) > float(previous["candidate_kl"]) + 1e-6:
            mean_kl_regressions += 1
    return match_regressions, mean_kl_regressions


def diagnostic_decision(
    summaries: dict[int, dict[str, float | int]],
    sample_rows: list[dict[str, object]],
) -> tuple[str, int | None]:
    for group_count in sorted(summaries):
        if group_count > 196:
            continue
        metrics = summaries[group_count]
        per_sample_regressions = sum(
            path_regressions(
                [
                    row
                    for row in sample_rows
                    if row["sample_id"] == sample_id
                ],
                through_group_count=group_count,
            )[0]
            for sample_id in {row["sample_id"] for row in sample_rows}
        )
        mean_path = [summaries[count] for count in sorted(summaries) if count <= group_count]
        aggregate_kl_regressions = sum(
            float(current["candidate_kl_mean"])
            > float(previous["candidate_kl_mean"]) + 1e-6
            for previous, current in zip(mean_path, mean_path[1:])
        )
        if (
            int(metrics["mismatch_count"]) == 0
            and int(metrics["harmful_count"]) == 0
            and float(metrics["candidate_kl_mean"]) <= 0.01
            and float(metrics["candidate_kl_p95"]) <= 0.02
            and per_sample_regressions == 0
            and aggregate_kl_regressions == 0
        ):
            return "STRICT_STATIC_READER_PATH", group_count

    for group_count in sorted(summaries):
        if group_count > 245:
            continue
        metrics = summaries[group_count]
        per_sample_regressions = sum(
            path_regressions(
                [
                    row
                    for row in sample_rows
                    if row["sample_id"] == sample_id
                ],
                through_group_count=group_count,
            )[0]
            for sample_id in {row["sample_id"] for row in sample_rows}
        )
        if (
            int(metrics["mismatch_count"]) == 0
            and int(metrics["harmful_count"]) == 0
            and float(metrics["candidate_kl_mean"]) <= 0.02
            and float(metrics["candidate_kl_p95"]) <= 0.05
            and per_sample_regressions == 0
        ):
            return "READER_PATH_BOUNDARY", group_count
    return "NO_STATIC_READER_PATH", None


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
    )[args.sample_offset : args.sample_offset + args.sample_count]
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
    if model.training:
        raise RuntimeError("reader model must be in evaluation mode")
    model_dtype = next(model.parameters()).dtype

    sample_rows: list[dict[str, object]] = []
    singleton_rows: list[dict[str, object]] = []
    maximum_dense_equivalence_error = 0.0
    maximum_repeatability_error = 0.0
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
        reference = payload["features"].index_select(0, position_tensor).to(
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
        group_count = exact_groups.shape[0]
        full_video_token_count = reference.shape[0] * reference.shape[1]
        if group_count != 392 or full_video_token_count != 1568:
            raise ValueError("registered group or token count changed")

        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        with torch.inference_mode():
            dense_logits = first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            )
            dense_candidate_logits = dense_logits.float().index_select(0, token_tensor)
            teacher_index = int(torch.argmax(dense_candidate_logits).item())
            empty_selection = torch.empty(0, device=device, dtype=torch.long)
            base_tokens, base_offsets = compact_group_tokens_and_offsets(
                exact_groups,
                approximate_means,
                empty_selection,
                representative_offset=REPRESENTATIVE_OFFSET,
            )
            base_logits = first_token_logits_from_positioned_video_tokens(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                video_tokens=base_tokens,
                video_position_offsets=base_offsets,
                full_video_token_count=full_video_token_count,
            )
            repeated_base_logits = first_token_logits_from_positioned_video_tokens(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                video_tokens=base_tokens,
                video_position_offsets=base_offsets,
                full_video_token_count=full_video_token_count,
            )

        base_candidate_logits = base_logits.float().index_select(0, token_tensor)
        repeated_base_candidates = repeated_base_logits.float().index_select(
            0, token_tensor
        )
        base_repeatability_error = float(
            (base_candidate_logits - repeated_base_candidates).abs().max().item()
        )
        maximum_repeatability_error = max(
            maximum_repeatability_error,
            base_repeatability_error,
        )
        if base_repeatability_error > 1e-5:
            raise RuntimeError(
                f"base positioned path changed candidate logits by {base_repeatability_error}"
            )

        base_kl = candidate_kl(dense_candidate_logits, base_candidate_logits)
        base_teacher_margin = float(
            base_candidate_logits[teacher_index]
            - torch.cat(
                (
                    base_candidate_logits[:teacher_index],
                    base_candidate_logits[teacher_index + 1 :],
                )
            ).max()
        )
        singleton_benefits: list[float] = []
        singleton_logits: list[torch.Tensor] = []
        for group_index in range(group_count):
            selected_index = torch.tensor(
                [group_index],
                device=device,
                dtype=torch.long,
            )
            singleton_tokens, singleton_offsets = compact_group_tokens_and_offsets(
                exact_groups,
                approximate_means,
                selected_index,
                representative_offset=REPRESENTATIVE_OFFSET,
            )
            with torch.inference_mode():
                logits = first_token_logits_from_positioned_video_tokens(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    video_tokens=singleton_tokens,
                    video_position_offsets=singleton_offsets,
                    full_video_token_count=full_video_token_count,
                )
            candidates = logits.float().index_select(0, token_tensor)
            singleton_kl = candidate_kl(dense_candidate_logits, candidates)
            benefit = base_kl - singleton_kl
            singleton_benefits.append(benefit)
            singleton_logits.append(candidates)
            teacher_margin = float(
                candidates[teacher_index]
                - torch.cat(
                    (
                        candidates[:teacher_index],
                        candidates[teacher_index + 1 :],
                    )
                ).max()
            )
            singleton_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_position": sample_position,
                    "group_index": group_index,
                    "frame_index": group_index // 49,
                    "base_candidate_kl": base_kl,
                    "singleton_candidate_kl": singleton_kl,
                    "singleton_kl_benefit": benefit,
                    "base_teacher_margin": base_teacher_margin,
                    "singleton_teacher_margin": teacher_margin,
                    "singleton_margin_benefit": teacher_margin - base_teacher_margin,
                }
            )

        first_singleton_repeat = singleton_logits[0]
        first_tokens, first_offsets = compact_group_tokens_and_offsets(
            exact_groups,
            approximate_means,
            torch.tensor([0], device=device, dtype=torch.long),
            representative_offset=REPRESENTATIVE_OFFSET,
        )
        with torch.inference_mode():
            repeated_first_logits = first_token_logits_from_positioned_video_tokens(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                video_tokens=first_tokens,
                video_position_offsets=first_offsets,
                full_video_token_count=full_video_token_count,
            )
        repeated_first_candidates = repeated_first_logits.float().index_select(
            0, token_tensor
        )
        singleton_repeatability_error = float(
            (first_singleton_repeat - repeated_first_candidates).abs().max().item()
        )
        maximum_repeatability_error = max(
            maximum_repeatability_error,
            singleton_repeatability_error,
        )
        if singleton_repeatability_error > 1e-5:
            raise RuntimeError(
                "singleton positioned path changed candidate logits by "
                f"{singleton_repeatability_error}"
            )

        order = stable_benefit_order(singleton_benefits)
        baseline_correct = teacher_index == sample.answer_index
        for refined_group_count in REGISTERED_GROUP_COUNTS:
            selected_indices = torch.tensor(
                sorted(order[:refined_group_count]),
                device=device,
                dtype=torch.long,
            )
            if refined_group_count == 0:
                candidate_logits = base_candidate_logits
            else:
                path_tokens, path_offsets = compact_group_tokens_and_offsets(
                    exact_groups,
                    approximate_means,
                    selected_indices,
                    representative_offset=REPRESENTATIVE_OFFSET,
                )
                with torch.inference_mode():
                    logits = first_token_logits_from_positioned_video_tokens(
                        model=model,
                        input_ids=prompt_batch["input_ids"],
                        attention_mask=prompt_batch["attention_mask"],
                        video_tokens=path_tokens,
                        video_position_offsets=path_offsets,
                        full_video_token_count=full_video_token_count,
                    )
                candidate_logits = logits.float().index_select(0, token_tensor)

            dense_equivalence_error = (
                float((dense_candidate_logits - candidate_logits).abs().max().item())
                if refined_group_count == group_count
                else 0.0
            )
            maximum_dense_equivalence_error = max(
                maximum_dense_equivalence_error,
                dense_equivalence_error,
            )
            if refined_group_count == group_count and dense_equivalence_error > 1e-5:
                raise RuntimeError(
                    "fully refined positioned path changed candidate logits by "
                    f"{dense_equivalence_error}"
                )
            approximate_index = int(torch.argmax(candidate_logits).item())
            token_count = hybrid_token_count(
                group_count=group_count,
                group_size=args.group_size,
                refined_group_count=refined_group_count,
            )
            sample_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_position": sample_position,
                    "refined_group_count": refined_group_count,
                    "teacher_index": teacher_index,
                    "answer_index": sample.answer_index,
                    "approximate_index": approximate_index,
                    "candidate_kl": candidate_kl(
                        dense_candidate_logits,
                        candidate_logits,
                    ),
                    "prediction_match": int(approximate_index == teacher_index),
                    "baseline_correct": int(baseline_correct),
                    "approximate_correct": int(
                        approximate_index == sample.answer_index
                    ),
                    "compressed_top1_margin": candidate_top1_margin(candidate_logits),
                    "token_count": token_count,
                    "token_retention": token_count / full_video_token_count,
                    "dense_equivalence_error": dense_equivalence_error,
                    "base_repeatability_error": base_repeatability_error,
                    "singleton_repeatability_error": singleton_repeatability_error,
                }
            )

        print(
            json.dumps(
                {
                    "event": "reader_aligned_singleton_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "base_kl": base_kl,
                    "positive_singleton_fraction": sum(
                        benefit > 0 for benefit in singleton_benefits
                    )
                    / group_count,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summaries: dict[int, dict[str, float | int]] = {}
    for group_count in REGISTERED_GROUP_COUNTS:
        rows = [
            row
            for row in sample_rows
            if int(row["refined_group_count"]) == group_count
        ]
        metrics = summarize_raw(rows)
        summaries[group_count] = {
            **metrics,
            "token_count": int(rows[0]["token_count"]),
            "token_retention": float(rows[0]["token_retention"]),
        }

    decision, first_passing_group_count = diagnostic_decision(
        summaries,
        sample_rows,
    )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "sample_positions": [73, 96],
        "sample_count": len(selected),
        "group_size": args.group_size,
        "group_counts": list(REGISTERED_GROUP_COUNTS),
        "decision": decision,
        "first_passing_group_count": first_passing_group_count,
        "maximum_dense_equivalence_error": maximum_dense_equivalence_error,
        "maximum_repeatability_error": maximum_repeatability_error,
        "budget_summaries": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Exploratory reader-aligned capacity diagnostic on already exposed "
            "positions 73-96. The singleton teacher reads candidate logits and is "
            "not deployable. Positions 97-120, selection, and formal remain unread."
        ),
    }
    write_csv(args.out_dir / "singleton_group_marginals.csv", singleton_rows)
    write_csv(args.out_dir / "reader_aligned_path_rows.csv", sample_rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
