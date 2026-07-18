#!/usr/bin/env python3
"""Run answer quality and teacher-forced NLL for TileLogic-RVQ variants.

All codecs reconstruct the original 1,280 visual-token sequence before Qwen
execution.  This script is task-quality evidence and explicitly not latency
or compact-prefill evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from scripts.run_tilespec_ex_quality import (
    DEFAULT_MODEL,
    TILE_PIXELS,
    _cached_visual_forward,
    _generate_predictions,
    _join_visual_features,
    _load_manifest,
    _multi_tile_images,
    _processor_inputs,
    _teacher_inputs,
)
from tilespec_ex.cache import (
    load_cache_manifest,
    load_cache_payload,
    manifest_sha256,
    validate_split_contract,
)
from tilespec_ex.metrics import dataset_score, normalize_answer
from tilespec_ex.tilelogic_methods import (
    build_tilelogic_variants,
    load_tilelogic_artifacts,
)


QUALITY_FORMAT = "tilelogic_quality_evaluation_v1"


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


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _teacher_nll(
    model: Any,
    processor: Any,
    images: Sequence[Any],
    question: str,
    answer: str,
    visual_variants: Sequence[torch.Tensor],
    *,
    batch_size: int,
    device: str,
) -> tuple[list[float], int]:
    teacher, labels = _teacher_inputs(processor, images, question, answer)
    supervised_tokens = int((labels != -100).sum().item())
    if supervised_tokens <= 0:
        raise RuntimeError("teacher target contains no supervised tokens")
    losses: list[float] = []
    for start in range(0, len(visual_variants), batch_size):
        chunk = list(visual_variants[start : start + batch_size])
        batch = len(chunk)
        cached = torch.cat(chunk, dim=0)
        input_ids = teacher.input_ids.to(device).repeat(batch, 1)
        attention_mask = teacher.attention_mask.to(device).repeat(batch, 1)
        image_grid = teacher.image_grid_thw.to(device).repeat(batch, 1)
        repeated_labels = labels.to(device).repeat(batch, 1)
        with _cached_visual_forward(model.visual, cached), torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=teacher.pixel_values.to(device),
                image_grid_thw=image_grid,
                use_cache=False,
                return_dict=True,
            )
        shift_logits = outputs.logits[:, :-1].float()
        shift_labels = repeated_labels[:, 1:]
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(batch, -1)
        mask = shift_labels != -100
        sequence_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        losses.extend(float(value) for value in sequence_loss.cpu())
        del outputs, shift_logits, token_loss
    if len(losses) != len(visual_variants):
        raise AssertionError("teacher NLL count differs from variant count")
    return losses, supervised_tokens


def _feature_eval_keys(feature_eval_dir: Path) -> set[tuple[str, int]]:
    summary = json.loads(
        (feature_eval_dir / "feature_eval_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("format") != "tilelogic_feature_evaluation_v1" or not summary.get("complete"):
        raise RuntimeError("feature evaluation must complete before task quality")
    records = _read_jsonl(feature_eval_dir / "feature_samples.jsonl")
    keys = {(str(item["dataset"]), int(item["dataset_index"])) for item in records}
    if len(keys) != len(records):
        raise RuntimeError("feature evaluation contains duplicate samples")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--feature-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant-batch-size", type=int, default=4)
    parser.add_argument("--nll-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if min(args.variant_batch_size, args.nll_batch_size, args.max_new_tokens) <= 0:
        raise SystemExit("batch sizes and max_new_tokens must be positive")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import transformers

    torch.manual_seed(20260718)
    cache_dir = args.cache_dir.resolve()
    training_dir = args.training_dir.resolve()
    feature_eval_dir = args.feature_eval_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_entries = load_cache_manifest(cache_dir)
    validate_split_contract(
        cache_entries,
        calibration_per_dataset=80,
        evaluation_per_dataset=120,
        oracle_per_dataset_split=16,
    )
    evaluation_entries = {
        entry.key: entry for entry in cache_entries if entry.split == "evaluation"
    }
    if len(evaluation_entries) != 360:
        raise RuntimeError("quality path requires exactly 360 evaluation entries")
    if _feature_eval_keys(feature_eval_dir) != set(evaluation_entries):
        raise RuntimeError("quality and feature evaluation sample sets differ")
    records = _load_manifest(args.manifest.resolve(), args.data_root.resolve(), 200)
    source_records = {
        (str(record["dataset"]), int(record["dataset_index"])): record
        for record in records
    }
    if not set(evaluation_entries) <= set(source_records):
        raise RuntimeError("cache evaluation keys are absent from source manifest")

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
    artifacts = load_tilelogic_artifacts(training_dir).to(args.device)

    quality_path = output_dir / "quality_samples.jsonl"
    existing = _read_jsonl(quality_path)
    completed = {
        (str(record["dataset"]), int(record["dataset_index"])) for record in existing
    }
    if len(completed) != len(existing) or not completed <= set(evaluation_entries):
        raise RuntimeError("quality resume state is duplicated or outside evaluation")

    environment = {
        "format": QUALITY_FORMAT,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model_dir": args.model_dir,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "cache_manifest_sha256": manifest_sha256(cache_dir),
        "feature_eval_summary_sha256": hashlib.sha256(
            (feature_eval_dir / "feature_eval_summary.json").read_bytes()
        ).hexdigest(),
        "evaluation_samples": len(evaluation_entries),
        "teacher_forced_target_policy": "first_manifest_answer_verbatim",
        "answer_score_uses_all_manifest_answers": True,
        "quality_path_keeps_original_visual_token_length": True,
        "quality_path_is_latency_evidence": False,
    }
    (output_dir / "quality_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    new_records = 0
    ordered_entries = sorted(
        evaluation_entries.values(), key=lambda entry: (entry.dataset, entry.dataset_index)
    )
    for sequence_index, entry in enumerate(ordered_entries):
        if entry.key in completed:
            continue
        sample_started = time.perf_counter()
        source = source_records[entry.key]
        images = _multi_tile_images(source["resolved_image_path"])
        prompt_inputs = _processor_inputs(processor, images, str(source["question"]))
        payload = load_cache_payload(cache_dir, entry, device=args.device)
        variants = build_tilelogic_variants(
            payload["thumbnail"], payload["crops"], payload["query"], artifacts
        )
        visual_variants = [
            _join_visual_features(payload["thumbnail"], variant.reconstructed)
            for variant in variants
        ]
        predictions = _generate_predictions(
            model,
            processor,
            prompt_inputs,
            visual_variants,
            variant_batch_size=args.variant_batch_size,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        nll, supervised_tokens = _teacher_nll(
            model,
            processor,
            images,
            str(source["question"]),
            str(source["answers"][0]),
            visual_variants,
            batch_size=args.nll_batch_size,
            device=args.device,
        )
        baseline = normalize_answer(predictions[0])
        variant_records = []
        for variant, prediction, loss in zip(variants, predictions, nll):
            normalized = normalize_answer(prediction)
            variant_records.append(
                {
                    "method": variant.method,
                    "retention_rate": variant.rate,
                    "scope": variant.scope,
                    "prediction": prediction,
                    "normalized_prediction": normalized,
                    "score": dataset_score(entry.dataset, prediction, source["answers"]),
                    "agrees_with_full": float(normalized == baseline),
                    "teacher_forced_nll": loss,
                    "supervised_tokens": supervised_tokens,
                }
            )
        record = {
            "format": QUALITY_FORMAT,
            "dataset": entry.dataset,
            "dataset_index": entry.dataset_index,
            "sample_id": entry.sample_id,
            "image_sha256": entry.image_sha256,
            "split": entry.split,
            "question": source["question"],
            "answers": source["answers"],
            "variants": variant_records,
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        _append_jsonl(quality_path, record)
        completed.add(entry.key)
        new_records += 1
        print(
            json.dumps(
                {
                    "progress": f"{sequence_index + 1}/{len(ordered_entries)}",
                    "dataset": entry.dataset,
                    "dataset_index": entry.dataset_index,
                    "seconds": round(time.perf_counter() - sample_started, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "format": QUALITY_FORMAT,
        "records": len(completed),
        "new_records": new_records,
        "expected_records": len(ordered_entries),
        "variants_per_record": 23,
        "complete": len(completed) == len(ordered_entries),
        "elapsed_seconds": time.perf_counter() - started,
        "quality_path_keeps_original_visual_token_length": True,
        "quality_path_is_latency_evidence": False,
    }
    (output_dir / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
