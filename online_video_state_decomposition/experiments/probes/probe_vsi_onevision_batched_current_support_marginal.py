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
)
from probe_vsi_onevision_reader_aligned_singleton_marginal import (
    candidate_top1_margin,
)
from probe_vsi_onevision_reader_risk_stage_a import (
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import contiguous_group_means
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    compact_video_masses,
    explicit_eager_mask,
    logits_from_positioned_inputs,
    positioned_inputs,
    set_language_attention_eager,
)
from probe_vsi_onevision_target_risk_budget_frontier import hybrid_token_count
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


MODES = ("positioned_equal_mass", "positioned_group_mass")
PATH_GROUP_COUNTS = (0, 49, 98, 147, 196)
SELECTION_BATCH = 49
ROLE = "exposed_batched_current_support_marginal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--m0-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def stable_top_batch(
    benefits: dict[int, float],
    *,
    count: int,
) -> list[int]:
    if count > len(benefits):
        raise ValueError("selection batch exceeds the remaining groups")
    return sorted(benefits, key=lambda index: (-benefits[index], index))[:count]


def sequence_masses(
    *,
    prompt_input_ids: torch.Tensor,
    model: torch.nn.Module,
    variable_length: int,
    video_masses: torch.Tensor,
) -> torch.Tensor:
    video_start = int(
        torch.nonzero(
            prompt_input_ids[0] == model.config.video_token_index,
            as_tuple=False,
        )[0].item()
    )
    suffix_with_newline = variable_length - video_start - video_masses.numel()
    if suffix_with_newline <= 0:
        raise RuntimeError("variable sequence mass accounting failed")
    return torch.cat(
        (
            torch.ones(video_start, device=video_masses.device),
            video_masses,
            torch.ones(suffix_with_newline, device=video_masses.device),
        )
    )


def support_inputs(
    *,
    mode: str,
    support: list[int],
    model: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    exact_groups: torch.Tensor,
    approximate_means: torch.Tensor,
    full_video_token_count: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode not in MODES:
        raise ValueError("unknown mass mode")
    device = exact_groups.device
    selected_indices = torch.tensor(
        sorted(support),
        device=device,
        dtype=torch.long,
    )
    video_tokens, video_offsets = compact_group_tokens_and_offsets(
        exact_groups,
        approximate_means,
        selected_indices,
        representative_offset=REPRESENTATIVE_OFFSET,
    )
    variable_embeds, variable_mask, position_ids = positioned_inputs(
        model=model,
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        video_tokens=video_tokens,
        video_position_offsets=video_offsets,
        full_video_token_count=full_video_token_count,
    )
    if mode == "positioned_equal_mass":
        masses = torch.ones(
            variable_embeds.shape[1],
            device=device,
            dtype=torch.float32,
        )
    else:
        video_masses = compact_video_masses(
            group_count=exact_groups.shape[0],
            group_size=group_size,
            selected_indices=selected_indices,
            device=device,
        )
        masses = sequence_masses(
            prompt_input_ids=prompt_input_ids,
            model=model,
            variable_length=variable_embeds.shape[1],
            video_masses=video_masses,
        )
    explicit_mask = explicit_eager_mask(
        model=model,
        variable_embeds=variable_embeds,
        variable_mask=variable_mask,
        position_ids=position_ids,
        token_masses=masses,
    )
    return variable_embeds, explicit_mask, position_ids


def batched_candidate_logits(
    *,
    mode: str,
    current_support: list[int],
    candidate_groups: list[int],
    model: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    exact_groups: torch.Tensor,
    approximate_means: torch.Tensor,
    full_video_token_count: int,
    group_size: int,
    candidate_token_tensor: torch.Tensor,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    outputs: dict[int, torch.Tensor] = {}
    for start in range(0, len(candidate_groups), batch_size):
        group_batch = candidate_groups[start : start + batch_size]
        embeds_batch = []
        mask_batch = []
        position_batch = []
        for group_index in group_batch:
            variable_embeds, explicit_mask, position_ids = support_inputs(
                mode=mode,
                support=[*current_support, group_index],
                model=model,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                exact_groups=exact_groups,
                approximate_means=approximate_means,
                full_video_token_count=full_video_token_count,
                group_size=group_size,
            )
            embeds_batch.append(variable_embeds)
            mask_batch.append(explicit_mask)
            position_batch.append(position_ids)
        with torch.inference_mode():
            model_outputs = model(
                inputs_embeds=torch.cat(embeds_batch, dim=0),
                attention_mask=torch.cat(mask_batch, dim=0),
                position_ids=torch.cat(position_batch, dim=0),
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        candidate_logits = model_outputs.logits[:, -1].float().index_select(
            1,
            candidate_token_tensor,
        )
        for row_index, group_index in enumerate(group_batch):
            outputs[group_index] = candidate_logits[row_index]
    return outputs


def path_regressions(
    rows: list[dict[str, object]],
    *,
    through_group_count: int,
) -> tuple[int, int]:
    selected = sorted(
        (
            row
            for row in rows
            if int(row["selected_group_count"]) <= through_group_count
        ),
        key=lambda row: int(row["selected_group_count"]),
    )
    match_regressions = 0
    kl_regressions = 0
    for previous, current in zip(selected, selected[1:]):
        if int(previous["prediction_match"]) and not int(current["prediction_match"]):
            match_regressions += 1
        if float(current["candidate_kl"]) > float(previous["candidate_kl"]) + 1e-6:
            kl_regressions += 1
    return match_regressions, kl_regressions


def strict_budget(
    summaries: dict[int, dict[str, float | int]],
    rows: list[dict[str, object]],
    *,
    sample_ids: set[str],
) -> int | None:
    for group_count in PATH_GROUP_COUNTS:
        metrics = summaries[group_count]
        match_regression_count = sum(
            path_regressions(
                [row for row in rows if row["sample_id"] == sample_id],
                through_group_count=group_count,
            )[0]
            for sample_id in sample_ids
        )
        mean_path = [
            summaries[count] for count in PATH_GROUP_COUNTS if count <= group_count
        ]
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
            and match_regression_count == 0
            and aggregate_kl_regressions == 0
        ):
            return group_count
    return None


def classify_outcome(
    summaries: dict[str, dict[int, dict[str, float | int]]],
    strict_budgets: dict[str, int | None],
) -> str:
    mass_budget = strict_budgets["positioned_group_mass"]
    equal_budget = strict_budgets["positioned_equal_mass"]
    if mass_budget is not None:
        mass_kl = float(
            summaries["positioned_group_mass"][mass_budget]["candidate_kl_mean"]
        )
        equal_metrics = summaries["positioned_equal_mass"][mass_budget]
        if equal_budget is None or mass_kl <= 0.8 * float(
            equal_metrics["candidate_kl_mean"]
        ):
            return "MASS_CURRENT_SUPPORT_HEADROOM"
    if mass_budget is not None or equal_budget is not None:
        return "CURRENT_SUPPORT_HEADROOM"
    for mode in MODES:
        metrics = summaries[mode][196]
        if int(metrics["mismatch_count"]) == 0 and int(metrics["harmful_count"]) == 0:
            return "DECISION_ONLY_BOUNDARY"
    return "NO_BATCHED_CURRENT_SUPPORT_PATH"


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
        raise ValueError("registered Gate requires eight frames and group_size=4")
    if args.sample_offset != 72 or args.sample_count != 24:
        raise ValueError("registered Gate is restricted to positions 73-96")
    if args.candidate_batch_size != 8:
        raise ValueError("registered candidate batch size must remain eight")

    m0_summary = json.loads(args.m0_summary.read_text(encoding="utf-8"))
    if m0_summary["decision"] != "SAME_KERNEL_MASS_VALID":
        raise ValueError("M0 same-kernel mass Gate did not pass")
    m0_error_keys = (
        "maximum_full_vocab_equal_mass_error",
        "maximum_full_vocab_repeatability_error",
        "maximum_full_vocab_dense_equivalence_error",
    )
    if any(float(m0_summary[key]) > 1e-5 for key in m0_error_keys):
        raise ValueError("M0 equivalence error exceeds the registered threshold")

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
    set_language_attention_eager(model)
    model_dtype = next(model.parameters()).dtype

    path_rows: list[dict[str, object]] = []
    marginal_rows: list[dict[str, object]] = []
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
            ).float()
        dense_candidates = dense_logits.index_select(0, token_tensor)
        teacher_index = int(torch.argmax(dense_candidates).item())
        baseline_correct = teacher_index == sample.answer_index

        for mode in MODES:
            current_support: list[int] = []
            for selected_count in PATH_GROUP_COUNTS:
                if len(current_support) != selected_count:
                    raise RuntimeError("current support size drifted from the path")
                variable_embeds, explicit_mask, position_ids = support_inputs(
                    mode=mode,
                    support=current_support,
                    model=model,
                    prompt_input_ids=prompt_batch["input_ids"],
                    prompt_attention_mask=prompt_batch["attention_mask"],
                    exact_groups=exact_groups,
                    approximate_means=approximate_means,
                    full_video_token_count=full_video_token_count,
                    group_size=args.group_size,
                )
                with torch.inference_mode():
                    path_logits = logits_from_positioned_inputs(
                        model=model,
                        variable_embeds=variable_embeds,
                        attention_mask=explicit_mask,
                        position_ids=position_ids,
                    )
                path_candidates = path_logits.index_select(0, token_tensor)
                path_kl = candidate_kl(dense_candidates, path_candidates)
                approximate_index = int(torch.argmax(path_candidates).item())
                token_count = hybrid_token_count(
                    group_count=group_count,
                    group_size=args.group_size,
                    refined_group_count=selected_count,
                )
                path_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "sample_position": sample_position,
                        "mode": mode,
                        "selected_group_count": selected_count,
                        "teacher_index": teacher_index,
                        "answer_index": sample.answer_index,
                        "approximate_index": approximate_index,
                        "candidate_kl": path_kl,
                        "prediction_match": int(approximate_index == teacher_index),
                        "baseline_correct": int(baseline_correct),
                        "approximate_correct": int(
                            approximate_index == sample.answer_index
                        ),
                        "compressed_top1_margin": candidate_top1_margin(
                            path_candidates
                        ),
                        "token_count": token_count,
                        "token_retention": token_count / full_video_token_count,
                    }
                )
                if selected_count == PATH_GROUP_COUNTS[-1]:
                    continue

                support_set = set(current_support)
                remaining = [
                    index for index in range(group_count) if index not in support_set
                ]
                candidate_outputs = batched_candidate_logits(
                    mode=mode,
                    current_support=current_support,
                    candidate_groups=remaining,
                    model=model,
                    prompt_input_ids=prompt_batch["input_ids"],
                    prompt_attention_mask=prompt_batch["attention_mask"],
                    exact_groups=exact_groups,
                    approximate_means=approximate_means,
                    full_video_token_count=full_video_token_count,
                    group_size=args.group_size,
                    candidate_token_tensor=token_tensor,
                    batch_size=args.candidate_batch_size,
                )
                benefits: dict[int, float] = {}
                candidate_kls: dict[int, float] = {}
                for group_index in remaining:
                    group_kl = candidate_kl(
                        dense_candidates,
                        candidate_outputs[group_index],
                    )
                    candidate_kls[group_index] = group_kl
                    benefits[group_index] = path_kl - group_kl
                next_groups = stable_top_batch(benefits, count=SELECTION_BATCH)
                next_group_set = set(next_groups)
                for group_index in remaining:
                    marginal_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "sample_position": sample_position,
                            "mode": mode,
                            "current_group_count": selected_count,
                            "group_index": group_index,
                            "frame_index": group_index // 49,
                            "current_candidate_kl": path_kl,
                            "singleton_candidate_kl": candidate_kls[group_index],
                            "conditional_kl_benefit": benefits[group_index],
                            "selected_next_batch": int(
                                group_index in next_group_set
                            ),
                        }
                    )
                current_support = sorted([*current_support, *next_groups])

        print(
            json.dumps(
                {
                    "event": "batched_current_support_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summaries: dict[str, dict[int, dict[str, float | int]]] = {}
    strict_budgets: dict[str, int | None] = {}
    sample_ids = {str(row["sample_id"]) for row in path_rows}
    for mode in MODES:
        mode_rows = [row for row in path_rows if row["mode"] == mode]
        mode_summaries: dict[int, dict[str, float | int]] = {}
        for selected_count in PATH_GROUP_COUNTS:
            selected_rows = [
                row
                for row in mode_rows
                if int(row["selected_group_count"]) == selected_count
            ]
            kl_values = np.asarray(
                [float(row["candidate_kl"]) for row in selected_rows],
                dtype=np.float64,
            )
            mode_summaries[selected_count] = {
                "sample_count": len(selected_rows),
                "mismatch_count": sum(
                    not int(row["prediction_match"]) for row in selected_rows
                ),
                "harmful_count": sum(
                    int(row["baseline_correct"])
                    and not int(row["approximate_correct"])
                    for row in selected_rows
                ),
                "agreement": sum(
                    int(row["prediction_match"]) for row in selected_rows
                )
                / len(selected_rows),
                "candidate_kl_mean": float(kl_values.mean()),
                "candidate_kl_p95": float(np.quantile(kl_values, 0.95)),
                "token_retention": float(selected_rows[0]["token_retention"]),
            }
        summaries[mode] = mode_summaries
        strict_budgets[mode] = strict_budget(
            mode_summaries,
            mode_rows,
            sample_ids=sample_ids,
        )

    decision = classify_outcome(summaries, strict_budgets)
    write_csv(args.out_dir / "current_support_path_rows.csv", path_rows)
    write_csv(args.out_dir / "current_support_marginals.csv", marginal_rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "sample_positions": [73, 96],
        "sample_count": len(selected),
        "modes": list(MODES),
        "path_group_counts": list(PATH_GROUP_COUNTS),
        "selection_batch": SELECTION_BATCH,
        "candidate_batch_size": args.candidate_batch_size,
        "attention_implementation": "eager",
        "summaries": {
            mode: {str(count): metrics for count, metrics in values.items()}
            for mode, values in summaries.items()
        },
        "strict_budgets": strict_budgets,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Transductive batched current-support capacity diagnostic on already "
            "exposed positions 73-96. It is not exact sequential greedy or a "
            "deployable router. Positions 97-120, selection, and formal remain unread."
        ),
    }
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
