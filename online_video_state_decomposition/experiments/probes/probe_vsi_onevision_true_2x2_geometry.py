from __future__ import annotations

import argparse
import csv
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
    load_onevision_model,
)
from mvbench_reader_quotient_support_oracle import candidate_token_ids
from mvbench_utils import decode_video_frames, uniform_frame_indices
from onevision_reader_quotient_stage_a import descending_eigenspace
from probe_vsi_onevision_batched_current_support_marginal import sequence_masses
from probe_vsi_onevision_cmrq_stage_b import candidate_kl, feature_path_for_sample
from probe_vsi_onevision_group_compaction_geometry import REPRESENTATIVE_OFFSET
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    explicit_eager_mask,
    logits_from_positioned_inputs,
    positioned_inputs,
    set_language_attention_eager,
)
from probe_vsi_onevision_reader_risk_stage_a import (
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import contiguous_group_means
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


GEOMETRIES = ("flat_contiguous_4", "spatial_2x2")
MODES = ("positioned_equal_mass", "positioned_group_mass")
ROLE = "exposed_true_2x2_geometry_control"


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
    parser.add_argument("--m1-path-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=72)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def spatial_2x2_means_and_offsets(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 3:
        raise ValueError("features must have shape [frames, tokens, hidden]")
    frame_count, token_count, hidden_size = features.shape
    side = math.isqrt(token_count)
    if side != 14 or side * side != token_count:
        raise ValueError("registered control requires a 14x14 token grid")
    if side % 2:
        raise ValueError("spatial token side must be even")

    groups = (
        features.reshape(frame_count, side // 2, 2, side // 2, 2, hidden_size)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, 4, hidden_size)
    )
    local_offsets = (
        torch.arange(token_count, device=features.device)
        .reshape(side // 2, 2, side // 2, 2)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4)
    )
    frame_offsets = (
        torch.arange(frame_count, device=features.device).reshape(-1, 1, 1)
        * token_count
    )
    offsets = (local_offsets.unsqueeze(0) + frame_offsets).reshape(-1, 4)
    return groups.mean(dim=1), offsets


def flat_means_and_offsets(
    features: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, means = contiguous_group_means(features, group_size=group_size)
    offsets = torch.arange(
        features.shape[0] * features.shape[1],
        device=features.device,
    ).reshape(-1, group_size)
    return means, offsets


def quotient_inputs(
    *,
    mode: str,
    model: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    quotient_means: torch.Tensor,
    group_offsets: torch.Tensor,
    full_video_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode not in MODES:
        raise ValueError("unknown mass mode")
    representative_offsets = group_offsets[:, REPRESENTATIVE_OFFSET]
    variable_embeds, variable_mask, position_ids = positioned_inputs(
        model=model,
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        video_tokens=quotient_means,
        video_position_offsets=representative_offsets,
        full_video_token_count=full_video_token_count,
    )
    if mode == "positioned_equal_mass":
        masses = torch.ones(
            variable_embeds.shape[1],
            device=quotient_means.device,
            dtype=torch.float32,
        )
    else:
        video_masses = torch.full(
            (quotient_means.shape[0],),
            float(group_offsets.shape[1]),
            device=quotient_means.device,
            dtype=torch.float32,
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


def classify_geometry_headroom(
    comparisons: dict[str, dict[str, float | int]],
) -> str:
    for metrics in comparisons.values():
        if (
            int(metrics["mismatch_reduction"]) >= 2
            and int(metrics["harmful_delta"]) <= 0
            and float(metrics["mean_kl_ratio"]) <= 0.8
            and float(metrics["p95_kl_ratio"]) <= 0.8
        ):
            return "TRUE_2X2_GEOMETRY_HEADROOM"
    for metrics in comparisons.values():
        if (
            int(metrics["mismatch_reduction"]) >= 2
            and int(metrics["harmful_delta"]) <= 0
            and float(metrics["mean_kl_ratio"]) <= 1.05
        ):
            return "TRUE_2X2_DECISION_HEADROOM"
    return "NO_TRUE_2X2_GEOMETRY_HEADROOM"


def read_m1_empty_support(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["selected_group_count"]) == 0]
    if len(selected) != 24 * len(MODES):
        raise ValueError("M1 empty-support row count changed")
    return {(row["sample_id"], row["mode"]): row for row in selected}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.sample_offset != 72 or args.sample_count != 24:
        raise ValueError("registered control is restricted to positions 73-96")
    if args.frame_budget != 8 or args.rank != 456 or args.group_size != 4:
        raise ValueError("registered feature and geometry identity changed")

    m0_summary = json.loads(args.m0_summary.read_text(encoding="utf-8"))
    if m0_summary["decision"] != "SAME_KERNEL_MASS_VALID":
        raise ValueError("M0 same-kernel mass Gate did not pass")
    m1_summary = json.loads(args.m1_summary.read_text(encoding="utf-8"))
    if m1_summary["decision"] != "NO_BATCHED_CURRENT_SUPPORT_PATH":
        raise ValueError("M1 decision identity changed")
    m1_rows = read_m1_empty_support(args.m1_path_rows)

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

    rows: list[dict[str, object]] = []
    maximum_flat_kl_repeat_error = 0.0
    flat_prediction_repeat_mismatches = 0
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
        geometry_payloads = {
            "flat_contiguous_4": flat_means_and_offsets(
                approximate,
                group_size=args.group_size,
            ),
            "spatial_2x2": spatial_2x2_means_and_offsets(approximate),
        }
        if any(means.shape[0] != 392 for means, _ in geometry_payloads.values()):
            raise ValueError("registered quotient token count changed")
        full_video_token_count = reference.shape[0] * reference.shape[1]

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

        for geometry in GEOMETRIES:
            quotient_means, group_offsets = geometry_payloads[geometry]
            for mode in MODES:
                variable_embeds, explicit_mask, position_ids = quotient_inputs(
                    mode=mode,
                    model=model,
                    prompt_input_ids=prompt_batch["input_ids"],
                    prompt_attention_mask=prompt_batch["attention_mask"],
                    quotient_means=quotient_means,
                    group_offsets=group_offsets,
                    full_video_token_count=full_video_token_count,
                )
                with torch.inference_mode():
                    logits = logits_from_positioned_inputs(
                        model=model,
                        variable_embeds=variable_embeds,
                        attention_mask=explicit_mask,
                        position_ids=position_ids,
                    )
                candidates = logits.index_select(0, token_tensor)
                approximate_index = int(torch.argmax(candidates).item())
                kl_value = candidate_kl(dense_candidates, candidates)
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "sample_position": sample_position,
                        "geometry": geometry,
                        "mode": mode,
                        "teacher_index": teacher_index,
                        "answer_index": sample.answer_index,
                        "approximate_index": approximate_index,
                        "candidate_kl": kl_value,
                        "prediction_match": int(approximate_index == teacher_index),
                        "baseline_correct": int(baseline_correct),
                        "approximate_correct": int(
                            approximate_index == sample.answer_index
                        ),
                        "harmful": int(
                            baseline_correct
                            and approximate_index != sample.answer_index
                        ),
                        "quotient_token_count": quotient_means.shape[0],
                        "token_retention": quotient_means.shape[0]
                        / full_video_token_count,
                    }
                )
                if geometry == "flat_contiguous_4":
                    previous = m1_rows[(sample.sample_id, mode)]
                    maximum_flat_kl_repeat_error = max(
                        maximum_flat_kl_repeat_error,
                        abs(kl_value - float(previous["candidate_kl"])),
                    )
                    flat_prediction_repeat_mismatches += (
                        approximate_index != int(previous["approximate_index"])
                    )

        print(
            json.dumps(
                {
                    "event": "true_2x2_geometry_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if maximum_flat_kl_repeat_error > 1e-6:
        raise RuntimeError("flat geometry did not reproduce M1 candidate KL")
    if flat_prediction_repeat_mismatches:
        raise RuntimeError("flat geometry did not reproduce M1 predictions")

    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for mode in MODES:
        mode_summaries: dict[str, dict[str, float | int]] = {}
        for geometry in GEOMETRIES:
            selected_rows = [
                row
                for row in rows
                if row["mode"] == mode and row["geometry"] == geometry
            ]
            kl_values = np.asarray(
                [float(row["candidate_kl"]) for row in selected_rows],
                dtype=np.float64,
            )
            mode_summaries[geometry] = {
                "sample_count": len(selected_rows),
                "agreement": sum(
                    int(row["prediction_match"]) for row in selected_rows
                )
                / len(selected_rows),
                "mismatch_count": sum(
                    not int(row["prediction_match"]) for row in selected_rows
                ),
                "harmful_count": sum(int(row["harmful"]) for row in selected_rows),
                "candidate_kl_mean": float(kl_values.mean()),
                "candidate_kl_p95": float(np.quantile(kl_values, 0.95)),
                "token_retention": float(selected_rows[0]["token_retention"]),
            }
        summaries[mode] = mode_summaries

    comparisons: dict[str, dict[str, float | int]] = {}
    for mode in MODES:
        flat = summaries[mode]["flat_contiguous_4"]
        spatial = summaries[mode]["spatial_2x2"]
        flat_rows = {
            str(row["sample_id"]): row
            for row in rows
            if row["mode"] == mode and row["geometry"] == "flat_contiguous_4"
        }
        spatial_rows = [
            row
            for row in rows
            if row["mode"] == mode and row["geometry"] == "spatial_2x2"
        ]
        comparisons[mode] = {
            "mismatch_reduction": int(flat["mismatch_count"])
            - int(spatial["mismatch_count"]),
            "harmful_delta": int(spatial["harmful_count"])
            - int(flat["harmful_count"]),
            "mean_kl_ratio": float(spatial["candidate_kl_mean"])
            / float(flat["candidate_kl_mean"]),
            "p95_kl_ratio": float(spatial["candidate_kl_p95"])
            / float(flat["candidate_kl_p95"]),
            "paired_kl_wins": sum(
                float(row["candidate_kl"])
                < float(flat_rows[str(row["sample_id"])]["candidate_kl"])
                for row in spatial_rows
            ),
            "prediction_match_wins": sum(
                int(row["prediction_match"])
                > int(flat_rows[str(row["sample_id"])]["prediction_match"])
                for row in spatial_rows
            ),
            "prediction_match_losses": sum(
                int(row["prediction_match"])
                < int(flat_rows[str(row["sample_id"])]["prediction_match"])
                for row in spatial_rows
            ),
        }

    decision = classify_geometry_headroom(comparisons)
    write_csv(args.out_dir / "true_2x2_geometry_rows.csv", rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "sample_positions": [73, 96],
        "sample_count": len(selected),
        "geometries": list(GEOMETRIES),
        "modes": list(MODES),
        "attention_implementation": "eager",
        "representative_offset": REPRESENTATIVE_OFFSET,
        "maximum_flat_kl_repeat_error": maximum_flat_kl_repeat_error,
        "flat_prediction_repeat_mismatches": flat_prediction_repeat_mismatches,
        "summaries": summaries,
        "comparisons": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Empty-support topology control on exposed positions 73-96. It does "
            "not test PPE, current-support routing, selection, formal, or latency."
        ),
    }
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
