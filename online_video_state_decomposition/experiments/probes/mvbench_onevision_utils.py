from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mvbench_utils import (
    choice_prompt,
    decode_video_frames,
    uniform_frame_indices,
    video_metadata,
)


def load_onevision_model(
    model_dir: Path,
    *,
    device: str,
) -> tuple[object, torch.nn.Module]:
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_fast=False,
    )
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.requires_grad_(False)
    return processor, model


def pool_and_recent_positions(
    total_frames: int,
    *,
    sampled_frames: int,
    feature_pool_frames: int,
    frame_budget: int,
) -> tuple[list[int], list[int]]:
    if not 0 < frame_budget <= feature_pool_frames <= sampled_frames:
        raise ValueError("frame budgets must be positive and nested")
    sampled = uniform_frame_indices(total_frames, sampled_frames)
    if len(sampled) < feature_pool_frames:
        raise ValueError("video has fewer sampled frames than the feature pool")
    pool_indices = sampled[-feature_pool_frames:]
    selected_positions = list(
        range(feature_pool_frames - frame_budget, feature_pool_frames)
    )
    return pool_indices, selected_positions


def decode_feature_pool(
    sample: object,
    *,
    sampled_frames: int,
    feature_pool_frames: int,
    frame_budget: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    total_frames, _ = video_metadata(sample.video_path)
    pool_indices, selected_positions = pool_and_recent_positions(
        total_frames,
        sampled_frames=sampled_frames,
        feature_pool_frames=feature_pool_frames,
        frame_budget=frame_budget,
    )
    frames, _, decoded_total = decode_video_frames(sample.video_path, pool_indices)
    if decoded_total != total_frames:
        raise RuntimeError("video frame count changed during decoding")
    return np.stack(frames), pool_indices, selected_positions


def preprocess_video(
    processor: object,
    frames: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch = processor.video_processor(videos=frames, return_tensors="pt")
    return batch["pixel_values_videos"].to(device=device, dtype=dtype)


def encode_video_features(
    model: torch.nn.Module,
    pixel_values_videos: torch.Tensor,
) -> torch.Tensor:
    features = model.get_video_features(
        pixel_values_videos,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError("expected one video feature batch")
    frames = int(pixel_values_videos.shape[1])
    if features.shape[1] % frames:
        raise ValueError("video feature count is not divisible by frame count")
    return features[0].reshape(frames, features.shape[1] // frames, features.shape[2])


def build_prompt_batch(
    processor: object,
    sample: object,
    selected_frames: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": choice_prompt(sample, include_subtitle=False),
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )
    batch = processor(
        text=prompt,
        videos=selected_frames,
        return_tensors="pt",
    )
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "pixel_values_videos": batch["pixel_values_videos"].to(
            device=device,
            dtype=dtype,
        ),
    }


def video_features_with_newline(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("video features must have shape [frames, tokens, hidden]")
    flattened = features.reshape(-1, features.shape[-1])
    newline = model.model.image_newline[None, :].to(
        device=features.device,
        dtype=features.dtype,
    )
    return torch.cat((flattened, newline), dim=0)


def first_token_logits_from_features(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    inserted = video_features_with_newline(model, features)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    video_mask = input_ids == model.config.video_token_index
    if int(video_mask.sum().item()) != inserted.shape[0]:
        raise ValueError("video placeholder and feature counts differ")
    expanded_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        expanded_mask,
        inserted.to(inputs_embeds.dtype),
    )
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    return outputs.logits[0, -1]


def direct_first_token_logits(
    *,
    model: torch.nn.Module,
    prompt_batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        **prompt_batch,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    return outputs.logits[0, -1]


def expected_feature_state_bytes(
    *,
    frames: int,
    tokens_per_frame: int,
    hidden_size: int,
    rank: int,
    residual_tokens_per_frame: int,
) -> dict[str, int | float]:
    dense = frames * tokens_per_frame * hidden_size * 2
    latent = frames * tokens_per_frame * rank * 2
    residual_values = frames * residual_tokens_per_frame * hidden_size * 2
    residual_indices = frames * residual_tokens_per_frame * 2
    state = latent + residual_values + residual_indices
    return {
        "dense_state_bytes_bf16": dense,
        "latent_bytes_fp16": latent,
        "residual_value_bytes_fp16": residual_values,
        "residual_index_bytes_int16": residual_indices,
        "compressed_state_bytes": state,
        "compression_ratio": dense / state,
    }
