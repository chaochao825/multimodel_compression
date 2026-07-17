#!/usr/bin/env python3
"""Run the minimal TileSpec-Ex quality and oracle-risk experiment.

This script evaluates an explicit five-image adapter: one global thumbnail and
four independently encoded TL/TR/BL/BR crops.  Compression applies only to the
four crop grids.  Reconstructed full-length embeddings are injected into the
unchanged VLM for task-quality measurement; this path is not a latency claim.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import types
from typing import Any, Iterator, Sequence

from PIL import Image, ImageOps
import torch

from tilespec_ex.core import (
    METHODS,
    RETENTION_RATES,
    block_query_relevance,
    compress_crop_tiles,
    compress_risk_structure_variant,
    cosine_similarity,
    crop_boundary_mse,
    enumerate_blocks,
    normalized_mse,
)
from tilespec_ex.metrics import dataset_score, normalize_answer


DEFAULT_MODEL = (
    "/home/wangmeiqi/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/"
    "66285546d2b821cf421d4f5eb2576359d3770cd3"
)
TILE_PIXELS = 448
PROMPT_PREFIX = (
    "The five images show one scene. Image 1 is a global thumbnail. "
    "Images 2, 3, 4, and 5 are respectively the top-left, top-right, "
    "bottom-left, and bottom-right high-resolution crops. Use all images. "
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def _sample_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["dataset"]), int(record["dataset_index"])


def _load_manifest(
    manifest: Path, data_root: Path, samples_per_dataset: int
) -> list[dict[str, Any]]:
    output = []
    counts: dict[str, int] = {}
    for record in _read_jsonl(manifest):
        dataset = str(record["dataset"])
        if counts.get(dataset, 0) >= samples_per_dataset:
            continue
        image_path = (data_root / record["image_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {image_path}")
        copied = dict(record)
        copied["resolved_image_path"] = str(image_path)
        output.append(copied)
        counts[dataset] = counts.get(dataset, 0) + 1
    required = {"gqa", "textvqa", "chartqa"}
    if set(counts) != required or any(counts[name] != samples_per_dataset for name in required):
        raise RuntimeError(f"manifest sample counts do not match contract: {counts}")
    return output


def _multi_tile_images(path: str) -> list[Image.Image]:
    image = Image.open(path).convert("RGB")
    thumbnail = ImageOps.pad(
        image,
        (TILE_PIXELS, TILE_PIXELS),
        method=Image.Resampling.BICUBIC,
        color=(127, 127, 127),
        centering=(0.5, 0.5),
    )
    canvas = ImageOps.pad(
        image,
        (2 * TILE_PIXELS, 2 * TILE_PIXELS),
        method=Image.Resampling.BICUBIC,
        color=(127, 127, 127),
        centering=(0.5, 0.5),
    )
    crops = [
        canvas.crop((0, 0, TILE_PIXELS, TILE_PIXELS)),
        canvas.crop((TILE_PIXELS, 0, 2 * TILE_PIXELS, TILE_PIXELS)),
        canvas.crop((0, TILE_PIXELS, TILE_PIXELS, 2 * TILE_PIXELS)),
        canvas.crop(
            (TILE_PIXELS, TILE_PIXELS, 2 * TILE_PIXELS, 2 * TILE_PIXELS)
        ),
    ]
    return [thumbnail, *crops]


def _messages(question: str, answer: str | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(5)]
    content.append(
        {
            "type": "text",
            "text": f"{PROMPT_PREFIX}Question: {question} Answer with only the short answer.",
        }
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


def _processor_inputs(processor: Any, images: Sequence[Image.Image], question: str) -> Any:
    text = processor.apply_chat_template(
        _messages(question), tokenize=False, add_generation_prompt=True
    )
    return processor(
        text=[text], images=list(images), padding=True, return_tensors="pt"
    )


def _teacher_inputs(
    processor: Any,
    images: Sequence[Image.Image],
    question: str,
    answer: str,
) -> tuple[Any, torch.Tensor]:
    prompt = _processor_inputs(processor, images, question)
    full_text = processor.apply_chat_template(
        _messages(question, answer), tokenize=False, add_generation_prompt=False
    )
    full = processor(
        text=[full_text], images=list(images), padding=True, return_tensors="pt"
    )
    prompt_ids = prompt.input_ids[0]
    full_ids = full.input_ids[0]
    common = 0
    limit = min(len(prompt_ids), len(full_ids))
    while common < limit and prompt_ids[common] == full_ids[common]:
        common += 1
    if common < max(1, len(prompt_ids) - 2):
        raise RuntimeError(
            f"teacher prompt prefix mismatch: common={common}, prompt={len(prompt_ids)}"
        )
    labels = full.input_ids.clone()
    labels[:, :common] = -100
    if int((labels != -100).sum()) == 0:
        raise RuntimeError("teacher-forced answer has no supervised tokens")
    return full, labels


@contextmanager
def _cached_visual_forward(visual: Any, output: torch.Tensor) -> Iterator[None]:
    original = visual.forward

    def cached(_self: Any, _hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        expected = int(
            (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item()
            // (_self.spatial_merge_size**2)
        )
        if output.shape[0] != expected:
            raise RuntimeError(
                f"cached visual rows {output.shape[0]} != grid contract {expected}"
            )
        return output

    visual.forward = types.MethodType(cached, visual)
    try:
        yield
    finally:
        visual.forward = original


def _split_visual_features(
    visual_features: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    if image_grid_thw.shape != (5, 3):
        raise RuntimeError(f"expected five image grids, got {image_grid_thw.tolist()}")
    grids = []
    chunks = []
    offset = 0
    for temporal, height, width in image_grid_thw.tolist():
        if temporal != 1 or height % spatial_merge_size or width % spatial_merge_size:
            raise RuntimeError(f"unsupported image grid: {(temporal, height, width)}")
        out_height = height // spatial_merge_size
        out_width = width // spatial_merge_size
        count = out_height * out_width
        chunks.append(visual_features[offset : offset + count])
        grids.append((out_height, out_width))
        offset += count
    if offset != visual_features.shape[0] or len(set(grids)) != 1:
        raise RuntimeError(f"inconsistent visual feature split: grids={grids}")
    height, width = grids[0]
    thumbnail = chunks[0]
    crops = torch.stack([chunk.reshape(height, width, -1) for chunk in chunks[1:]])
    return thumbnail, crops, (height, width)


def _join_visual_features(thumbnail: torch.Tensor, crops: torch.Tensor) -> torch.Tensor:
    return torch.cat((thumbnail, crops.reshape(-1, crops.shape[-1])), dim=0)


def _query_embedding(model: Any, processor: Any, question: str, device: str) -> torch.Tensor:
    tokenized = processor.tokenizer(
        question, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    with torch.inference_mode():
        return model.model.embed_tokens(tokenized).float().mean(dim=(0, 1))


def _variant_specs() -> list[tuple[str, float | None, str]]:
    specs = [("none", None, "main")]
    for rate in RETENTION_RATES:
        specs.extend((method, rate, "main") for method in METHODS if method != "none")
        specs.extend(
            (
                variant,
                rate,
                "structure_ablation",
            )
            for variant in ("risk_token_unstructured", "risk_block_fixed_slots")
        )
    return specs


def _build_variants(
    thumbnail: torch.Tensor,
    crops: torch.Tensor,
    query: torch.Tensor,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    features: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    original_crop_tokens = crops.shape[0] * crops.shape[1] * crops.shape[2]
    thumbnail_tokens = thumbnail.shape[0]
    for method, rate, scope in _variant_specs():
        if method in ("risk_token_unstructured", "risk_block_fixed_slots"):
            result = compress_risk_structure_variant(
                crops, method, float(rate), query_embedding=query
            )
        else:
            effective_rate = 1.0 if rate is None else float(rate)
            result = compress_crop_tiles(
                crops,
                method,
                effective_rate,
                query_embedding=query if method == "tile_risk_exception" else None,
            )
        combined = _join_visual_features(thumbnail, result.reconstructed)
        features.append(combined)
        metadata.append(
            {
                "method": method,
                "retention_rate": rate,
                "scope": scope,
                "retained_crop_tokens": result.retained_tokens,
                "base_tokens": result.base_tokens,
                "exception_tokens": result.exception_tokens,
                "thumbnail_tokens": thumbnail_tokens,
                "compact_visual_tokens": thumbnail_tokens + result.retained_tokens,
                "effective_total_retention": (
                    thumbnail_tokens + result.retained_tokens
                )
                / (thumbnail_tokens + original_crop_tokens),
                "selected_blocks": [list(item) for item in result.selected_blocks],
                "feature_nmse": normalized_mse(crops, result.reconstructed),
                "feature_cosine": cosine_similarity(crops, result.reconstructed),
                "boundary_mse": crop_boundary_mse(crops, result.reconstructed),
            }
        )
    return features, metadata


def _generate_predictions(
    model: Any,
    processor: Any,
    inputs: Any,
    variants: Sequence[torch.Tensor],
    *,
    variant_batch_size: int,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    predictions: list[str] = []
    prompt_length = inputs.input_ids.shape[1]
    for start in range(0, len(variants), variant_batch_size):
        chunk = list(variants[start : start + variant_batch_size])
        batch = len(chunk)
        cached = torch.cat(chunk, dim=0)
        input_ids = inputs.input_ids.to(device).repeat(batch, 1)
        attention_mask = inputs.attention_mask.to(device).repeat(batch, 1)
        image_grid = inputs.image_grid_thw.to(device).repeat(batch, 1)
        pixel_values = inputs.pixel_values.to(device)
        with _cached_visual_forward(model.visual, cached), torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        decoded = processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        predictions.extend(item.strip() for item in decoded)
    if len(predictions) != len(variants):
        raise AssertionError("prediction count differs from variant count")
    return predictions


def _oracle_record(
    model: Any,
    processor: Any,
    inputs: Any,
    images: Sequence[Image.Image],
    record: dict[str, Any],
    visual_features: torch.Tensor,
    crops: torch.Tensor,
    query: torch.Tensor,
    *,
    device: str,
) -> dict[str, Any]:
    teacher, labels = _teacher_inputs(
        processor, images, str(record["question"]), str(record["answers"][0])
    )
    feature_variable = visual_features.detach().clone().requires_grad_(True)
    teacher_grid = teacher.image_grid_thw.to(device)
    if not torch.equal(teacher_grid.cpu(), inputs.image_grid_thw.cpu()):
        raise RuntimeError("teacher and generation image grids differ")
    with _cached_visual_forward(model.visual, feature_variable):
        outputs = model(
            input_ids=teacher.input_ids.to(device),
            attention_mask=teacher.attention_mask.to(device),
            pixel_values=teacher.pixel_values.to(device),
            image_grid_thw=teacher_grid,
            labels=labels.to(device),
            use_cache=False,
            return_dict=True,
        )
        outputs.loss.backward()
    if feature_variable.grad is None:
        raise RuntimeError("oracle loss did not produce a visual-feature gradient")
    _, crop_gradients, _ = _split_visual_features(
        feature_variable.grad,
        teacher_grid,
        model.visual.spatial_merge_size,
    )
    rate_records = []
    original_tokens = crops.shape[0] * crops.shape[1] * crops.shape[2]
    for rate in RETENTION_RATES:
        risk_result = compress_crop_tiles(
            crops,
            "tile_risk_exception",
            rate,
            query_embedding=query,
        )
        base_rate = risk_result.base_tokens / original_tokens
        base = compress_crop_tiles(crops, "tile_lowpass", base_rate).reconstructed
        residual = crops - base
        residual_blocks, locations, energy = enumerate_blocks(residual)
        gradient_blocks, gradient_locations, _ = enumerate_blocks(crop_gradients)
        if locations != gradient_locations:
            raise AssertionError("gradient and residual block layouts differ")
        relevance = block_query_relevance(residual_blocks, query)
        risk = energy * relevance
        oracle = (residual_blocks.float() * gradient_blocks.float()).sum(
            dim=(1, 2, 3)
        ).abs()
        rate_records.append(
            {
                "retention_rate": rate,
                "base_tokens": risk_result.base_tokens,
                "exception_tokens": risk_result.exception_tokens,
                "locations": [list(item) for item in locations],
                "energy": energy.cpu().tolist(),
                "relevance": relevance.cpu().tolist(),
                "risk": risk.cpu().tolist(),
                "oracle_first_order_abs": oracle.cpu().tolist(),
            }
        )
    model.zero_grad(set_to_none=True)
    return {
        "dataset": record["dataset"],
        "dataset_index": record["dataset_index"],
        "sample_id": record["sample_id"],
        "teacher_forced_answer": record["answers"][0],
        "teacher_forced_loss": float(outputs.loss.detach().item()),
        "rates": rate_records,
    }


def _environment(model_dir: str, manifest: Path, args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model_dir": model_dir,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "samples_per_dataset": args.samples_per_dataset,
        "oracle_samples_per_dataset": args.oracle_samples_per_dataset,
        "retention_rates": list(RETENTION_RATES),
        "main_methods": list(METHODS),
        "tile_adapter": "one 448x448 thumbnail + four independent 448x448 crops",
        "quality_path_keeps_original_visual_token_length": True,
        "quality_path_is_latency_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    parser.add_argument("--oracle-samples-per-dataset", type=int, default=16)
    parser.add_argument("--variant-batch-size", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.samples_per_dataset <= 0 or args.oracle_samples_per_dataset < 0:
        raise SystemExit("sample counts must be non-negative, with quality count > 0")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(20260717)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = output_dir / "quality_samples.jsonl"
    oracle_path = output_dir / "oracle_blocks.jsonl"
    completed_quality = {_sample_key(item) for item in _read_jsonl(quality_path)}
    completed_oracle = {_sample_key(item) for item in _read_jsonl(oracle_path)}
    records = _load_manifest(
        args.manifest.resolve(), args.data_root.resolve(), args.samples_per_dataset
    )

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=False,
        min_pixels=TILE_PIXELS * TILE_PIXELS,
        max_pixels=TILE_PIXELS * TILE_PIXELS,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    environment = _environment(args.model_dir, args.manifest, args)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    oracle_counts: dict[str, int] = {}
    started = time.perf_counter()
    for sequence_index, record in enumerate(records):
        key = _sample_key(record)
        needs_quality = key not in completed_quality
        oracle_target = int(record["dataset_index"]) < args.oracle_samples_per_dataset
        needs_oracle = oracle_target and key not in completed_oracle
        if not needs_quality and not needs_oracle:
            continue

        sample_started = time.perf_counter()
        images = _multi_tile_images(record["resolved_image_path"])
        inputs = _processor_inputs(processor, images, str(record["question"]))
        pixel_values = inputs.pixel_values.to(args.device, dtype=model.visual.dtype)
        image_grid = inputs.image_grid_thw.to(args.device)
        with torch.inference_mode():
            visual_features = model.visual(pixel_values, grid_thw=image_grid)
        thumbnail, crops, crop_grid = _split_visual_features(
            visual_features, image_grid, model.visual.spatial_merge_size
        )
        query = _query_embedding(model, processor, str(record["question"]), args.device)

        if needs_oracle:
            oracle_record = _oracle_record(
                model,
                processor,
                inputs,
                images,
                record,
                visual_features,
                crops,
                query,
                device=args.device,
            )
            _append_jsonl(oracle_path, oracle_record)
            completed_oracle.add(key)
            oracle_counts[str(record["dataset"])] = (
                oracle_counts.get(str(record["dataset"]), 0) + 1
            )

        if needs_quality:
            variants, metadata = _build_variants(thumbnail, crops, query)
            predictions = _generate_predictions(
                model,
                processor,
                inputs,
                variants,
                variant_batch_size=args.variant_batch_size,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            baseline_prediction = predictions[0]
            baseline_normalized = normalize_answer(baseline_prediction)
            variant_records = []
            for details, prediction in zip(metadata, predictions):
                details = dict(details)
                details.update(
                    {
                        "prediction": prediction,
                        "normalized_prediction": normalize_answer(prediction),
                        "score": dataset_score(
                            str(record["dataset"]), prediction, record["answers"]
                        ),
                        "agrees_with_full": float(
                            normalize_answer(prediction) == baseline_normalized
                        ),
                    }
                )
                variant_records.append(details)
            payload = {
                "dataset": record["dataset"],
                "dataset_index": record["dataset_index"],
                "sample_id": record["sample_id"],
                "image_sha256": record["image_sha256"],
                "question": record["question"],
                "answers": record["answers"],
                "crop_grid_hw": list(crop_grid),
                "full_visual_tokens": int(visual_features.shape[0]),
                "thumbnail_tokens": int(thumbnail.shape[0]),
                "crop_tokens": int(crops.shape[0] * crops.shape[1] * crops.shape[2]),
                "variants": variant_records,
                "elapsed_seconds": time.perf_counter() - sample_started,
            }
            _append_jsonl(quality_path, payload)
            completed_quality.add(key)

        elapsed = time.perf_counter() - sample_started
        print(
            json.dumps(
                {
                    "progress": f"{sequence_index + 1}/{len(records)}",
                    "dataset": record["dataset"],
                    "dataset_index": record["dataset_index"],
                    "quality": needs_quality,
                    "oracle": needs_oracle,
                    "seconds": round(elapsed, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    run_summary = {
        "quality_records": len(completed_quality),
        "oracle_records": len(completed_oracle),
        "new_oracle_records_by_dataset": oracle_counts,
        "elapsed_seconds": time.perf_counter() - started,
        "complete": len(completed_quality) == len(records),
    }
    (output_dir / "quality_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
