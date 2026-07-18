#!/usr/bin/env python3
"""Cache leakage-safe Qwen visual features for TileLogic-RVQ.

The cache is split before any codebook or router fitting.  Oracle gradients are
stored for disjoint calibration and evaluation subsets; downstream trainers
must explicitly request the calibration split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import torch

from scripts.run_tilespec_ex_quality import (
    DEFAULT_MODEL,
    TILE_PIXELS,
    _cached_visual_forward,
    _load_manifest,
    _multi_tile_images,
    _processor_inputs,
    _query_embedding,
    _split_visual_features,
    _teacher_inputs,
)


CACHE_FORMAT = "tilelogic_feature_cache_v1"
SPLIT_SEED = "tilelogic-rvq-split-20260718"
ORACLE_SEED = "tilelogic-rvq-oracle-20260718"


def _stable_digest(seed: str, record: dict[str, Any]) -> str:
    payload = ":".join(
        (
            seed,
            str(record["dataset"]),
            str(record["sample_id"]),
            str(record["image_sha256"]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_splits(
    records: Iterable[dict[str, Any]], calibration_per_dataset: int
) -> list[dict[str, Any]]:
    records = [dict(item) for item in records]
    if calibration_per_dataset <= 0:
        raise ValueError("calibration_per_dataset must be positive")
    output: list[dict[str, Any]] = []
    for dataset in ("gqa", "textvqa", "chartqa"):
        selected = [item for item in records if item["dataset"] == dataset]
        if calibration_per_dataset >= len(selected):
            raise ValueError("calibration split must leave evaluation examples")
        selected.sort(key=lambda item: _stable_digest(SPLIT_SEED, item))
        for rank, record in enumerate(selected):
            record["split"] = (
                "calibration" if rank < calibration_per_dataset else "evaluation"
            )
            record["split_rank"] = rank
            output.append(record)
    output.sort(key=lambda item: (item["dataset"], int(item["dataset_index"])))
    return output


def assign_oracle_subsets(
    records: list[dict[str, Any]], oracle_per_dataset_split: int
) -> None:
    if oracle_per_dataset_split < 0:
        raise ValueError("oracle count cannot be negative")
    for dataset in ("gqa", "textvqa", "chartqa"):
        for split in ("calibration", "evaluation"):
            selected = [
                item
                for item in records
                if item["dataset"] == dataset and item["split"] == split
            ]
            if oracle_per_dataset_split > len(selected):
                raise ValueError("oracle subset exceeds its split")
            selected.sort(key=lambda item: _stable_digest(ORACLE_SEED, item))
            oracle_keys = {
                (item["dataset"], int(item["dataset_index"]))
                for item in selected[:oracle_per_dataset_split]
            }
            for record in selected:
                key = (record["dataset"], int(record["dataset_index"]))
                record["oracle"] = key in oracle_keys


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(cache_dir: Path, record: dict[str, Any]) -> Path:
    return cache_dir / "features" / str(record["dataset"]) / (
        f"{int(record['dataset_index']):04d}.pt"
    )


def _gradient_record(
    model: Any,
    processor: Any,
    images: list[Any],
    record: dict[str, Any],
    visual_features: torch.Tensor,
    prompt_inputs: Any,
    *,
    device: str,
) -> tuple[torch.Tensor, float, int]:
    teacher, labels = _teacher_inputs(
        processor,
        images,
        str(record["question"]),
        str(record["answers"][0]),
    )
    teacher_grid = teacher.image_grid_thw.to(device)
    if not torch.equal(teacher_grid.cpu(), prompt_inputs.image_grid_thw.cpu()):
        raise RuntimeError("teacher and generation image grids differ")
    variable = visual_features.detach().clone().requires_grad_(True)
    with _cached_visual_forward(model.visual, variable):
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
    if variable.grad is None:
        raise RuntimeError("teacher loss produced no visual-feature gradient")
    _, crop_gradient, _ = _split_visual_features(
        variable.grad,
        teacher_grid,
        model.visual.spatial_merge_size,
    )
    supervised_tokens = int((labels != -100).sum().item())
    loss = float(outputs.loss.detach().item())
    model.zero_grad(set_to_none=True)
    return crop_gradient.detach(), loss, supervised_tokens


def _split_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": record["dataset"],
        "dataset_index": int(record["dataset_index"]),
        "sample_id": record["sample_id"],
        "image_sha256": record["image_sha256"],
        "split": record["split"],
        "split_rank": int(record["split_rank"]),
        "oracle": bool(record["oracle"]),
    }


def _write_split_manifest(cache_dir: Path, records: list[dict[str, Any]]) -> str:
    path = cache_dir / "split_manifest.jsonl"
    payload = "".join(
        json.dumps(_split_manifest_record(record), sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_completed(
    completed: list[dict[str, Any]],
    records: list[dict[str, Any]],
    cache_dir: Path,
    *,
    source_manifest_sha256: str,
    model_revision: str,
) -> set[tuple[str, int]]:
    expected = {
        (str(item["dataset"]), int(item["dataset_index"])): item for item in records
    }
    keys: set[tuple[str, int]] = set()
    for item in completed:
        key = (str(item["dataset"]), int(item["dataset_index"]))
        if key in keys:
            raise RuntimeError(f"duplicate cache manifest entry: {key}")
        if key not in expected:
            raise RuntimeError(f"cache manifest entry is outside split contract: {key}")
        source = expected[key]
        for field in ("split", "image_sha256", "sample_id"):
            if item[field] != source[field]:
                raise RuntimeError(f"cache manifest mismatch for {key}: {field}")
        if item.get("source_manifest_sha256") != source_manifest_sha256:
            raise RuntimeError(f"cache manifest source hash mismatch for {key}")
        if item.get("model_revision") != model_revision:
            raise RuntimeError(f"cache manifest model revision mismatch for {key}")
        if not isinstance(item.get("tensor_dtypes"), dict):
            raise RuntimeError(f"cache manifest tensor dtypes missing for {key}")
        feature_path = cache_dir / item["cache_file"]
        if not feature_path.is_file() or feature_path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"missing or truncated cache file: {feature_path}")
        keys.add(key)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    parser.add_argument("--calibration-per-dataset", type=int, default=80)
    parser.add_argument("--oracle-per-dataset-split", type=int, default=16)
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import transformers

    torch.manual_seed(20260718)
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = _load_manifest(
        args.manifest.resolve(),
        args.data_root.resolve(),
        args.samples_per_dataset,
    )
    records = assign_splits(records, args.calibration_per_dataset)
    assign_oracle_subsets(records, args.oracle_per_dataset_split)
    split_manifest_sha256 = _write_split_manifest(cache_dir, records)
    source_manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    model_revision = Path(args.model_dir).resolve().name

    cache_manifest = cache_dir / "cache_manifest.jsonl"
    completed_records = _read_jsonl(cache_manifest)
    completed = _validate_completed(
        completed_records,
        records,
        cache_dir,
        source_manifest_sha256=source_manifest_sha256,
        model_revision=model_revision,
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

    environment = {
        "format": CACHE_FORMAT,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model_dir": str(Path(args.model_dir)),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": source_manifest_sha256,
        "model_revision": model_revision,
        "split_manifest_sha256": split_manifest_sha256,
        "split_seed": SPLIT_SEED,
        "oracle_seed": ORACLE_SEED,
        "samples_per_dataset": args.samples_per_dataset,
        "calibration_per_dataset": args.calibration_per_dataset,
        "evaluation_per_dataset": (
            args.samples_per_dataset - args.calibration_per_dataset
        ),
        "oracle_per_dataset_split": args.oracle_per_dataset_split,
        "cache_dtype": "float16",
        "tile_adapter": "one 448x448 thumbnail + four 448x448 crops",
    }
    (cache_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    written = 0
    for sequence_index, record in enumerate(records):
        key = (str(record["dataset"]), int(record["dataset_index"]))
        if key in completed:
            continue
        sample_started = time.perf_counter()
        images = _multi_tile_images(record["resolved_image_path"])
        inputs = _processor_inputs(processor, images, str(record["question"]))
        pixel_values = inputs.pixel_values.to(args.device, dtype=model.visual.dtype)
        image_grid = inputs.image_grid_thw.to(args.device)
        with torch.inference_mode():
            visual_features = model.visual(pixel_values, grid_thw=image_grid)
        thumbnail, crops, crop_grid = _split_visual_features(
            visual_features,
            image_grid,
            model.visual.spatial_merge_size,
        )
        query = _query_embedding(
            model, processor, str(record["question"]), args.device
        )

        crop_gradient = None
        teacher_loss = None
        supervised_tokens = None
        if record["oracle"]:
            crop_gradient, teacher_loss, supervised_tokens = _gradient_record(
                model,
                processor,
                images,
                record,
                visual_features,
                inputs,
                device=args.device,
            )

        payload: dict[str, Any] = {
            "format": CACHE_FORMAT,
            "dataset": str(record["dataset"]),
            "dataset_index": int(record["dataset_index"]),
            "sample_id": str(record["sample_id"]),
            "image_sha256": str(record["image_sha256"]),
            "source_manifest_sha256": source_manifest_sha256,
            "model_revision": model_revision,
            "split": str(record["split"]),
            "split_rank": int(record["split_rank"]),
            "oracle": bool(record["oracle"]),
            "crop_grid_hw": tuple(int(item) for item in crop_grid),
            "thumbnail": thumbnail.detach().cpu().to(torch.float16),
            "crops": crops.detach().cpu().to(torch.float16),
            "query": query.detach().cpu().to(torch.float16),
        }
        if crop_gradient is not None:
            payload["crop_gradient"] = crop_gradient.cpu().to(torch.float16)
            payload["teacher_forced_loss"] = float(teacher_loss)
            payload["supervised_tokens"] = int(supervised_tokens)
        tensor_dtypes = {
            name: str(value.dtype).removeprefix("torch.")
            for name, value in payload.items()
            if isinstance(value, torch.Tensor)
            and name in {"thumbnail", "crops", "query", "crop_gradient"}
        }
        payload["tensor_dtypes"] = tensor_dtypes

        destination = _cache_path(cache_dir, record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.partial")
        torch.save(payload, temporary)
        temporary.replace(destination)
        relative = destination.relative_to(cache_dir).as_posix()
        manifest_record = {
            **_split_manifest_record(record),
            "format": CACHE_FORMAT,
            "cache_file": relative,
            "bytes": destination.stat().st_size,
            "sha256": _file_sha256(destination),
            "source_manifest_sha256": source_manifest_sha256,
            "model_revision": model_revision,
            "tensor_dtypes": tensor_dtypes,
            "thumbnail_shape": list(payload["thumbnail"].shape),
            "crop_shape": list(payload["crops"].shape),
            "query_shape": list(payload["query"].shape),
            "gradient_shape": (
                list(payload["crop_gradient"].shape)
                if "crop_gradient" in payload
                else None
            ),
        }
        _append_jsonl(cache_manifest, manifest_record)
        completed.add(key)
        written += 1
        print(
            json.dumps(
                {
                    "progress": f"{sequence_index + 1}/{len(records)}",
                    "dataset": record["dataset"],
                    "dataset_index": record["dataset_index"],
                    "split": record["split"],
                    "oracle": record["oracle"],
                    "cache_mib": round(destination.stat().st_size / 2**20, 3),
                    "seconds": round(time.perf_counter() - sample_started, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del visual_features, thumbnail, crops, query, payload
        if sequence_index % 20 == 0:
            torch.cuda.empty_cache()

    summary = {
        "format": CACHE_FORMAT,
        "records": len(completed),
        "new_records": written,
        "expected_records": len(records),
        "split_manifest_sha256": split_manifest_sha256,
        "cache_manifest_sha256": _file_sha256(cache_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "model_revision": model_revision,
        "per_entry_tensor_dtypes": True,
        "complete": len(completed) == len(records),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (cache_dir / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
