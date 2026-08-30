from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers.masking_utils import create_causal_mask

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
from probe_vsi_onevision_query_group_fallback_transfer import summarize_raw
from probe_vsi_onevision_reader_risk_stage_a import (
    reconstruct,
    select_calibration_questions,
)
from probe_vsi_onevision_risk_guided_exact_groups import contiguous_group_means
from vsi_onevision_progressive_cmrq_selection import calibration_moments
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


REGISTERED_GROUP_COUNTS = (0, 196, 392)
ROLE = "exposed_same_kernel_mass_equivalence"


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


def deterministic_support(
    *,
    group_count: int,
    selected_count: int,
    device: torch.device,
) -> torch.Tensor:
    if not 0 <= selected_count <= group_count:
        raise ValueError("selected count is outside the group range")
    if selected_count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    indices = [
        index * group_count // selected_count for index in range(selected_count)
    ]
    selected = torch.tensor(indices, device=device, dtype=torch.long)
    if torch.unique(selected).numel() != selected_count:
        raise RuntimeError("deterministic support contains duplicate groups")
    return selected


def compact_video_masses(
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


def set_language_attention_eager(model: torch.nn.Module) -> None:
    model.config._attn_implementation = "eager"
    model.config.text_config._attn_implementation = "eager"
    model.model.config._attn_implementation = "eager"
    model.model.config.text_config._attn_implementation = "eager"
    language_model = model.model.language_model
    language_model.config._attn_implementation = "eager"
    if any(
        layer.self_attn.config._attn_implementation != "eager"
        for layer in language_model.layers
    ):
        raise RuntimeError("not every language attention layer uses eager attention")


def positioned_inputs(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_tokens: torch.Tensor,
    video_position_offsets: torch.Tensor,
    full_video_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError("same-kernel readout requires one prompt batch")
    if video_tokens.ndim != 2:
        raise ValueError("video tokens must have shape [tokens, hidden]")
    if video_position_offsets.shape != (video_tokens.shape[0],):
        raise ValueError("video offsets have the wrong shape")
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
    inserted_mask = torch.ones(
        (1, inserted.shape[0]),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    variable_mask = torch.cat(
        (
            attention_mask[:, :start],
            inserted_mask,
            attention_mask[:, stop:],
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
    if position_ids.shape != variable_mask.shape:
        raise ValueError("position IDs and variable attention mask differ")
    return variable_embeds, variable_mask, position_ids


def explicit_eager_mask(
    *,
    model: torch.nn.Module,
    variable_embeds: torch.Tensor,
    variable_mask: torch.Tensor,
    position_ids: torch.Tensor,
    token_masses: torch.Tensor,
) -> torch.Tensor:
    sequence_length = variable_embeds.shape[1]
    if token_masses.shape != (sequence_length,):
        raise ValueError("token masses do not match the variable sequence")
    if not bool((token_masses > 0).all().item()):
        raise ValueError("token masses must be positive")
    cache_position = torch.arange(
        sequence_length,
        device=variable_embeds.device,
        dtype=torch.long,
    )
    causal_mask = create_causal_mask(
        config=model.model.language_model.config,
        input_embeds=variable_embeds,
        attention_mask=variable_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    if causal_mask is None or not isinstance(causal_mask, torch.Tensor):
        raise RuntimeError("eager attention did not produce an explicit tensor mask")
    if causal_mask.ndim != 4:
        raise RuntimeError("eager causal mask is not four-dimensional")
    key_bias = torch.log(token_masses.to(causal_mask.dtype))[None, None, None, :]
    return causal_mask + key_bias


def logits_from_positioned_inputs(
    *,
    model: torch.nn.Module,
    variable_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        inputs_embeds=variable_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    return outputs.logits[0, -1].float()


def outcome(maximum_error: float, maximum_repeatability_error: float) -> str:
    if maximum_error <= 1e-5 and maximum_repeatability_error <= 1e-5:
        return "SAME_KERNEL_MASS_VALID"
    return "INVALID_KERNEL_EQUIVALENCE"


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
    maximum_equal_mass_error = 0.0
    maximum_repeatability_error = 0.0
    maximum_dense_equivalence_error = 0.0
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

        for selected_count in REGISTERED_GROUP_COUNTS:
            support = deterministic_support(
                group_count=group_count,
                selected_count=selected_count,
                device=device,
            )
            video_tokens, video_offsets = compact_group_tokens_and_offsets(
                exact_groups,
                approximate_means,
                support,
                representative_offset=REPRESENTATIVE_OFFSET,
            )
            variable_embeds, variable_mask, position_ids = positioned_inputs(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                video_tokens=video_tokens,
                video_position_offsets=video_offsets,
                full_video_token_count=full_video_token_count,
            )
            all_one_masses = torch.ones(
                variable_embeds.shape[1],
                device=device,
                dtype=torch.float32,
            )
            equal_mask = explicit_eager_mask(
                model=model,
                variable_embeds=variable_embeds,
                variable_mask=variable_mask,
                position_ids=position_ids,
                token_masses=all_one_masses,
            )
            video_masses = compact_video_masses(
                group_count=group_count,
                group_size=args.group_size,
                selected_indices=support,
                device=device,
            )
            prefix_length = variable_embeds.shape[1] - video_tokens.shape[0] - 1
            video_start = int(
                torch.nonzero(
                    prompt_batch["input_ids"][0] == model.config.video_token_index,
                    as_tuple=False,
                )[0].item()
            )
            suffix_length = prefix_length - video_start
            if suffix_length < 0:
                raise RuntimeError("variable sequence prefix/suffix accounting failed")
            sequence_masses = torch.cat(
                (
                    torch.ones(video_start, device=device),
                    video_masses,
                    torch.ones(1 + suffix_length, device=device),
                )
            )
            weighted_mask = explicit_eager_mask(
                model=model,
                variable_embeds=variable_embeds,
                variable_mask=variable_mask,
                position_ids=position_ids,
                token_masses=sequence_masses,
            )

            with torch.inference_mode():
                ordinary_logits = logits_from_positioned_inputs(
                    model=model,
                    variable_embeds=variable_embeds,
                    attention_mask=variable_mask,
                    position_ids=position_ids,
                )
                equal_logits = logits_from_positioned_inputs(
                    model=model,
                    variable_embeds=variable_embeds,
                    attention_mask=equal_mask,
                    position_ids=position_ids,
                )
                weighted_logits = (
                    equal_logits
                    if selected_count == group_count
                    else logits_from_positioned_inputs(
                        model=model,
                        variable_embeds=variable_embeds,
                        attention_mask=weighted_mask,
                        position_ids=position_ids,
                    )
                )
                repeated_equal_logits = (
                    logits_from_positioned_inputs(
                        model=model,
                        variable_embeds=variable_embeds,
                        attention_mask=equal_mask,
                        position_ids=position_ids,
                    )
                    if selected_count == 0
                    else equal_logits
                )

            equal_mass_error = float(
                (ordinary_logits - equal_logits).abs().max().item()
            )
            repeatability_error = float(
                (equal_logits - repeated_equal_logits).abs().max().item()
            )
            dense_equivalence_error = (
                float((dense_logits - ordinary_logits).abs().max().item())
                if selected_count == group_count
                else 0.0
            )
            maximum_equal_mass_error = max(
                maximum_equal_mass_error,
                equal_mass_error,
            )
            maximum_repeatability_error = max(
                maximum_repeatability_error,
                repeatability_error,
            )
            maximum_dense_equivalence_error = max(
                maximum_dense_equivalence_error,
                dense_equivalence_error,
            )
            weighted_candidates = weighted_logits.index_select(0, token_tensor)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_position": sample_position,
                    "selected_group_count": selected_count,
                    "reader_token_count": video_tokens.shape[0],
                    "full_vocab_equal_mass_error": equal_mass_error,
                    "full_vocab_repeatability_error": repeatability_error,
                    "full_vocab_dense_equivalence_error": dense_equivalence_error,
                    "weighted_candidate_kl": candidate_kl(
                        dense_candidates,
                        weighted_candidates,
                    ),
                    "weighted_prediction_match": int(
                        int(torch.argmax(weighted_candidates).item()) == teacher_index
                    ),
                }
            )

        print(
            json.dumps(
                {
                    "event": "same_kernel_mass_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "maximum_equal_mass_error": maximum_equal_mass_error,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    maximum_error = max(
        maximum_equal_mass_error,
        maximum_dense_equivalence_error,
    )
    decision = outcome(maximum_error, maximum_repeatability_error)
    if decision != "SAME_KERNEL_MASS_VALID":
        raise RuntimeError(
            "same-kernel equivalence guard failed: "
            f"equal={maximum_equal_mass_error}, "
            f"dense={maximum_dense_equivalence_error}, "
            f"repeat={maximum_repeatability_error}"
        )

    weighted_summaries: dict[str, dict[str, float | int]] = {}
    for selected_count in REGISTERED_GROUP_COUNTS:
        selected_rows = [
            row
            for row in rows
            if int(row["selected_group_count"]) == selected_count
        ]
        weighted_summaries[str(selected_count)] = {
            "sample_count": len(selected_rows),
            "agreement": sum(
                int(row["weighted_prediction_match"]) for row in selected_rows
            )
            / len(selected_rows),
            "candidate_kl_mean": sum(
                float(row["weighted_candidate_kl"]) for row in selected_rows
            )
            / len(selected_rows),
            "candidate_kl_p95": summarize_raw(
                [float(row["weighted_candidate_kl"]) for row in selected_rows]
            )["p95"],
        }

    write_csv(args.out_dir / "same_kernel_mass_rows.csv", rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "sample_positions": [73, 96],
        "sample_count": len(selected),
        "registered_group_counts": list(REGISTERED_GROUP_COUNTS),
        "attention_implementation": "eager",
        "maximum_full_vocab_equal_mass_error": maximum_equal_mass_error,
        "maximum_full_vocab_repeatability_error": maximum_repeatability_error,
        "maximum_full_vocab_dense_equivalence_error": (
            maximum_dense_equivalence_error
        ),
        "weighted_diagnostics": weighted_summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Engineering equivalence Gate on already exposed positions 73-96. "
            "The eager harness is not a latency candidate. Weighted metrics are "
            "diagnostic only. Positions 97-120, selection, and formal remain unread."
        ),
    }
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
