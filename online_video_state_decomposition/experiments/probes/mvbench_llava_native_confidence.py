from __future__ import annotations

import json
import time

import torch

import mvbench_llava_compressed_feature_memory as experiment
from mvbench_llava_feature_memory_anchor import (
    build_prompt_inputs,
    use_cached_image_features,
)
from mvbench_utils import parse_choice_output


DIAGNOSTIC_VERSION = 1


def first_token_diagnostics(logits: torch.Tensor) -> dict[str, float | int]:
    values = logits.detach().float().flatten()
    if values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("first-token logits must contain at least two finite values")
    log_probabilities = torch.log_softmax(values, dim=0)
    probabilities = log_probabilities.exp()
    top_values, top_indices = torch.topk(log_probabilities, k=2)
    return {
        "native_first_token_id": int(top_indices[0].item()),
        "native_first_token_logprob": float(top_values[0].item()),
        "native_first_token_margin": float((top_values[0] - top_values[1]).item()),
        "native_first_token_entropy": float(
            (-(probabilities * log_probabilities).sum()).item()
        ),
    }


def candidate_letter_diagnostics(
    logits: torch.Tensor,
    *,
    tokenizer: object,
    candidate_count: int,
) -> dict[str, object]:
    if candidate_count < 2 or candidate_count > 26:
        raise ValueError("candidate count must be between 2 and 26")
    token_ids = []
    for index in range(candidate_count):
        label = chr(ord("A") + index)
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"candidate label {label} is not a single token: {encoded}")
        token_ids.append(int(encoded[0]))
    indices = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    candidate_logits = logits.detach().float().flatten().index_select(0, indices)
    candidate_log_probabilities = torch.log_softmax(candidate_logits, dim=0)
    top_values, _ = torch.topk(candidate_log_probabilities, k=2)
    return {
        "native_candidate_token_ids_json": json.dumps(token_ids),
        "native_candidate_logprobs_json": json.dumps(
            [float(value) for value in candidate_log_probabilities.tolist()]
        ),
        "native_candidate_margin": float((top_values[0] - top_values[1]).item()),
    }


def read_with_native_confidence(
    *,
    sample: object,
    policy: str,
    selected_frame_indices: list[int],
    selected_features: torch.Tensor,
    selected_image_sizes: list[tuple[int, int]],
    tokenizer: object,
    model: torch.nn.Module,
    max_new_tokens: int,
    include_subtitle: bool,
) -> dict[str, object]:
    input_ids, attention_mask = build_prompt_inputs(
        sample=sample,
        image_count=len(selected_frame_indices),
        tokenizer=tokenizer,
        model=model,
        include_subtitle=include_subtitle,
    )
    dummy_images = torch.empty(
        (len(selected_frame_indices), 3, 1, 1),
        device=model.device,
        dtype=torch.float16,
    )
    experiment.synchronize()
    read_start = time.perf_counter()
    with use_cached_image_features(model, selected_features):
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=dummy_images,
                image_sizes=selected_image_sizes,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
    experiment.synchronize()
    read_seconds = time.perf_counter() - read_start
    if not generated.scores:
        raise RuntimeError("generation did not return first-token scores")
    output = tokenizer.batch_decode(
        generated.sequences,
        skip_special_tokens=True,
    )[0].strip()
    predicted = parse_choice_output(output, sample.candidates)
    first_logits = generated.scores[0][0]
    return {
        "policy": policy,
        "raw_output": output,
        "predicted_index": predicted,
        "prediction": sample.candidates[predicted] if predicted is not None else "",
        "parsed": int(predicted is not None),
        "correct": int(predicted == int(sample.answer_index)),
        "inference_seconds": read_seconds,
        **first_token_diagnostics(first_logits),
        **candidate_letter_diagnostics(
            first_logits,
            tokenizer=tokenizer,
            candidate_count=len(sample.candidates),
        ),
    }


def diagnostic_config(args: object) -> dict[str, object]:
    config = original_config_from_args(args)
    config["native_confidence_diagnostic_version"] = DIAGNOSTIC_VERSION
    return config


original_config_from_args = experiment.config_from_args
experiment.config_from_args = diagnostic_config
experiment.read_from_native_feature_pool = read_with_native_confidence


if __name__ == "__main__":
    raise SystemExit(experiment.main())
