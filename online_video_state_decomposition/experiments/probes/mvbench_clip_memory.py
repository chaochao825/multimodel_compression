from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mvbench_utils import (
    clip_candidate_prompts,
    decode_video_frames,
    load_mvbench_samples,
    parse_csv_list,
    parse_int_list,
    shard_samples,
    uniform_frame_indices,
    video_metadata,
)
from task_memory import (
    METHODS,
    make_memory,
    softmax_pool_scores,
    state_accounting,
)


DEFAULT_TASKS = (
    "object_existence",
    "state_change",
    "scene_transition",
    "action_sequence",
    "moving_direction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--capacities", default="4,8,16")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--instant-slots", type=int, default=3)
    parser.add_argument("--storage-bits", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pool-temperature", type=float, default=10.0)
    parser.add_argument("--oja-lr", type=float, default=0.5)
    parser.add_argument("--prototype-scale", type=float, default=0.75)
    parser.add_argument("--slot-min-lr", type=float, default=0.05)
    parser.add_argument(
        "--slot-replace-similarity",
        type=float,
        default=0.75,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-full-sequence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "tasks": parse_csv_list(args.tasks),
        "samples_per_task": args.samples_per_task,
        "selection_seed": args.selection_seed,
        "frames": args.frames,
        "capacities": parse_int_list(args.capacities),
        "methods": parse_csv_list(args.methods),
        "instant_slots": args.instant_slots,
        "storage_bits": args.storage_bits,
        "pool_temperature": args.pool_temperature,
        "oja_lr": args.oja_lr,
        "prototype_scale": args.prototype_scale,
        "slot_min_lr": args.slot_min_lr,
        "slot_replace_similarity": args.slot_replace_similarity,
        "include_full_sequence": not args.skip_full_sequence,
    }


def fingerprint(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values)
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8],
        "little",
    )


def encode_images(
    model: torch.nn.Module,
    image_processor: object,
    frames: list[np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    dtype = next(model.parameters()).dtype
    for start in range(0, len(frames), batch_size):
        images = [
            Image.fromarray(frame)
            for frame in frames[start : start + batch_size]
        ]
        inputs = image_processor(
            images=images,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            vision = model.vision_model(pixel_values=pixel_values)
            features = model.visual_projection(vision.pooler_output)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        chunks.append(features.cpu())
    return torch.cat(chunks, dim=0).numpy()


def encode_text(
    model: torch.nn.Module,
    tokenizer: object,
    texts: list[str],
    *,
    device: torch.device,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(model.config.text_config.max_position_embeddings),
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        text = model.text_model(**inputs)
        features = model.text_projection(text.pooler_output)
        features = torch.nn.functional.normalize(features.float(), dim=-1)
    return features.cpu().numpy()


def prediction_row(
    *,
    sample: object,
    method: str,
    capacity: int,
    scores: np.ndarray,
    accounting: dict[str, object],
    frames: int,
    frame_indices: list[int],
    fps: float,
    total_frames: int,
    decode_seconds: float,
    image_encode_seconds: float,
    text_encode_seconds: float,
) -> dict[str, object]:
    predicted = int(np.argmax(scores))
    answer_index = int(sample.answer_index)
    return {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "sample_index": sample.index,
        "video": str(sample.video_path),
        "question": sample.question,
        "candidates_json": json.dumps(
            sample.candidates,
            ensure_ascii=False,
        ),
        "answer": sample.answer,
        "answer_index": answer_index,
        "method": method,
        "capacity": capacity,
        "predicted_index": predicted,
        "prediction": sample.candidates[predicted],
        "correct": int(predicted == answer_index),
        "scores_json": json.dumps(
            [float(value) for value in scores],
            separators=(",", ":"),
        ),
        "sampled_frames": frames,
        "frame_indices_json": json.dumps(frame_indices),
        "fps": fps,
        "total_frames": total_frames,
        "decode_seconds": decode_seconds,
        "image_encode_seconds": image_encode_seconds,
        "text_encode_seconds": text_encode_seconds,
        **accounting,
    }


def evaluate_sample(
    sample: object,
    *,
    model: torch.nn.Module,
    image_processor: object,
    tokenizer: object,
    device: torch.device,
    config: dict[str, object],
    image_batch_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    total_frames, fps = video_metadata(sample.video_path)
    frame_indices = uniform_frame_indices(total_frames, int(config["frames"]))
    start = time.perf_counter()
    frames, decoded_fps, decoded_total = decode_video_frames(
        sample.video_path,
        frame_indices,
    )
    decode_seconds = time.perf_counter() - start
    if decoded_total != total_frames:
        raise RuntimeError("video frame count changed during decoding")
    if decoded_fps > 0:
        fps = decoded_fps

    start = time.perf_counter()
    image_vectors = encode_images(
        model,
        image_processor,
        frames,
        batch_size=image_batch_size,
        device=device,
    )
    image_encode_seconds = time.perf_counter() - start

    start = time.perf_counter()
    text_vectors = encode_text(
        model,
        tokenizer,
        clip_candidate_prompts(sample),
        device=device,
    )
    text_encode_seconds = time.perf_counter() - start

    rows: list[dict[str, object]] = []
    hidden_dim = int(image_vectors.shape[1])
    for capacity in config["capacities"]:
        instant_capacity = min(int(config["instant_slots"]), int(capacity))
        for method in config["methods"]:
            memory = make_memory(
                str(method),
                capacity=int(capacity),
                hidden_dim=hidden_dim,
                seed=stable_seed(
                    config["selection_seed"],
                    sample.sample_id,
                    method,
                    capacity,
                ),
                instant_capacity=instant_capacity,
                oja_lr=float(config["oja_lr"]),
                prototype_scale=float(config["prototype_scale"]),
                slot_min_lr=float(config["slot_min_lr"]),
                slot_replace_similarity=float(
                    config["slot_replace_similarity"]
                ),
            )
            update_start = time.perf_counter()
            for image_vector in image_vectors:
                memory.update(image_vector[None])
            update_seconds = time.perf_counter() - update_start
            read_start = time.perf_counter()
            scores = memory.score(
                text_vectors,
                temperature=float(config["pool_temperature"]),
            )
            read_seconds = time.perf_counter() - read_start
            account = asdict(
                state_accounting(
                    str(method),
                    capacity=int(capacity),
                    hidden_dim=hidden_dim,
                    storage_bits=int(config["storage_bits"]),
                    instant_capacity=instant_capacity,
                )
            )
            account.update(
                {
                    "memory_update_seconds": update_seconds,
                    "memory_read_seconds": read_seconds,
                }
            )
            rows.append(
                prediction_row(
                    sample=sample,
                    method=str(method),
                    capacity=int(capacity),
                    scores=scores,
                    accounting=account,
                    frames=len(frames),
                    frame_indices=frame_indices,
                    fps=fps,
                    total_frames=total_frames,
                    decode_seconds=decode_seconds,
                    image_encode_seconds=image_encode_seconds,
                    text_encode_seconds=text_encode_seconds,
                )
            )

    if bool(config["include_full_sequence"]):
        scores = softmax_pool_scores(
            text_vectors,
            image_vectors,
            temperature=float(config["pool_temperature"]),
        )
        payload_bytes = (
            len(image_vectors)
            * hidden_dim
            * int(config["storage_bits"])
            // 8
        )
        rows.append(
            prediction_row(
                sample=sample,
                method="full_sequence",
                capacity=len(image_vectors),
                scores=scores,
                accounting={
                    "payload_bytes": payload_bytes,
                    "metadata_bytes": 16,
                    "total_state_bytes": payload_bytes + 16,
                    "read_flops_per_option": (
                        2 * len(image_vectors) * hidden_dim
                    ),
                    "update_flops_per_frame": 0,
                    "memory_update_seconds": 0.0,
                    "memory_read_seconds": 0.0,
                },
                frames=len(frames),
                frame_indices=frame_indices,
                fps=fps,
                total_frames=total_frames,
                decode_seconds=decode_seconds,
                image_encode_seconds=image_encode_seconds,
                text_encode_seconds=text_encode_seconds,
            )
        )
    metadata = {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "frame_indices": frame_indices,
        "total_frames": total_frames,
        "fps": fps,
        "image_embedding_shape": list(image_vectors.shape),
        "text_embedding_shape": list(text_vectors.shape),
    }
    return rows, metadata


def load_checkpoint(
    path: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if payload.get("configuration_fingerprint") != expected_fingerprint:
        return None
    if not isinstance(payload.get("rows"), list):
        return None
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                row["task"],
                row["method"],
                row["capacity"],
                row["total_state_bytes"],
            )
        ].append(row)
    output = []
    for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
        values = grouped[key]
        output.append(
            {
                "task": key[0],
                "method": key[1],
                "capacity": key[2],
                "total_state_bytes": key[3],
                "samples": len(values),
                "accuracy": float(
                    np.mean([int(row["correct"]) for row in values])
                ),
                "mean_memory_update_seconds": float(
                    np.mean(
                        [
                            float(row["memory_update_seconds"])
                            for row in values
                        ]
                    )
                ),
                "mean_memory_read_seconds": float(
                    np.mean(
                        [
                            float(row["memory_read_seconds"])
                            for row in values
                        ]
                    )
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = configuration(args)
    config_fingerprint = fingerprint(config)
    tasks = [str(value) for value in config["tasks"]]
    samples = load_mvbench_samples(
        args.dataset_root,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        selection_seed=args.selection_seed,
    )
    samples = shard_samples(
        samples,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.out_dir / "configuration.json",
        {
            **config,
            "configuration_fingerprint": config_fingerprint,
            "argv": sys.argv,
        },
    )

    from transformers import (
        CLIPImageProcessor,
        CLIPModel,
        CLIPTokenizer,
    )

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    image_processor = CLIPImageProcessor.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        use_fast=False,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
    )

    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for position, sample in enumerate(samples, start=1):
        checkpoint_path = (
            args.out_dir / "checkpoints" / f"{sample.sample_id}.json"
        )
        existing = (
            None
            if args.overwrite
            else load_checkpoint(
                checkpoint_path,
                expected_fingerprint=config_fingerprint,
            )
        )
        if existing is not None:
            all_rows.extend(existing["rows"])
            print(
                json.dumps(
                    {
                        "event": "resume",
                        "sample": sample.sample_id,
                        "position": position,
                        "total": len(samples),
                    }
                ),
                flush=True,
            )
            continue
        sample_start = time.perf_counter()
        try:
            rows, metadata = evaluate_sample(
                sample,
                model=model,
                image_processor=image_processor,
                tokenizer=tokenizer,
                device=device,
                config=config,
                image_batch_size=args.image_batch_size,
            )
            payload = {
                "configuration_fingerprint": config_fingerprint,
                "metadata": metadata,
                "rows": rows,
                "seconds": time.perf_counter() - sample_start,
            }
            write_json_atomic(checkpoint_path, payload)
            all_rows.extend(rows)
            print(
                json.dumps(
                    {
                        "event": "sample_ok",
                        "sample": sample.sample_id,
                        "position": position,
                        "total": len(samples),
                        "seconds": payload["seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "sample": sample.sample_id,
                "task": sample.task,
                "video": str(sample.video_path),
                "type": type(exc).__name__,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            write_json_atomic(
                args.out_dir
                / "errors"
                / f"{sample.sample_id}.json",
                {
                    "configuration_fingerprint": config_fingerprint,
                    **failure,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "sample_fail",
                        "sample": sample.sample_id,
                        "type": type(exc).__name__,
                        "error": repr(exc),
                    }
                ),
                flush=True,
            )
            if args.fail_fast:
                raise

    shard_name = f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    write_csv(args.out_dir / f"{shard_name}_predictions.csv", all_rows)
    summary = summarize(all_rows)
    write_csv(args.out_dir / f"{shard_name}_summary.csv", summary)
    write_json_atomic(
        args.out_dir / f"{shard_name}_status.json",
        {
            "configuration_fingerprint": config_fingerprint,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "assigned_samples": len(samples),
            "completed_samples": len(
                {str(row["sample_id"]) for row in all_rows}
            ),
            "prediction_rows": len(all_rows),
            "failure_count": len(failures),
            "failures": failures,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "torch": torch.__version__,
        },
    )
    print(
        json.dumps(
            {
                "event": "shard_complete",
                "shard": shard_name,
                "samples": len(
                    {str(row["sample_id"]) for row in all_rows}
                ),
                "rows": len(all_rows),
                "failures": len(failures),
            }
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
