from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from mvbench_utils import (
    choice_prompt,
    decode_video_frames,
    hybrid_frame_indices,
    load_mvbench_samples,
    load_mvbench_samples_by_indices,
    parse_choice_output,
    parse_csv_list,
    recent_frame_indices,
    shard_samples,
    uniform_frame_indices,
    video_metadata,
)


DEFAULT_TASKS = (
    "object_existence",
    "state_change",
    "scene_transition",
    "action_sequence",
    "moving_direction",
)
POLICIES = ("uniform", "recent", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llava-source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, default=None)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--recent-frames", type=int, default=3)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--pool-grid", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--include-subtitle", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    selection_manifest_sha256 = None
    if args.selection_manifest is not None:
        selection_manifest_sha256 = hashlib.sha256(
            args.selection_manifest.read_bytes()
        ).hexdigest()
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "llava_source": str(args.llava_source.resolve()),
        "selection_manifest": (
            str(args.selection_manifest.resolve())
            if args.selection_manifest is not None
            else None
        ),
        "selection_manifest_sha256": selection_manifest_sha256,
        "tasks": parse_csv_list(args.tasks),
        "samples_per_task": args.samples_per_task,
        "selection_seed": args.selection_seed,
        "frame_budget": args.frame_budget,
        "recent_frames": args.recent_frames,
        "policies": parse_csv_list(args.policies),
        "pool_grid": args.pool_grid,
        "max_new_tokens": args.max_new_tokens,
        "include_subtitle": args.include_subtitle,
    }


def fingerprint(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def install_visual_pooling(model: torch.nn.Module, grid: int) -> None:
    if grid <= 0:
        return
    original_encode = model.encode_images

    def pooled_encode(_self: object, images: torch.Tensor) -> torch.Tensor:
        features = original_encode(images)
        if features.ndim != 3:
            return features
        batch, tokens, hidden = features.shape
        side = int(round(tokens**0.5))
        if side * side != tokens:
            raise RuntimeError(
                f"cannot spatially pool {tokens} non-square visual tokens"
            )
        spatial = features.transpose(1, 2).reshape(
            batch,
            hidden,
            side,
            side,
        )
        pooled = functional.adaptive_avg_pool2d(spatial, (grid, grid))
        return pooled.flatten(2).transpose(1, 2).contiguous()

    model.encode_images = types.MethodType(pooled_encode, model)


def install_transformers_forward_compatibility(
    model: torch.nn.Module,
) -> None:
    original_forward = model.forward

    def compatible_forward(
        _self: object,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.Tensor | None = None,
        image_sizes: object | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> object:
        # Transformers >=4.46 passes this cache hint, while LLaVA-1.5's
        # forward override predates the argument.
        del cache_position
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            image_sizes=image_sizes,
            return_dict=return_dict,
        )

    model.forward = types.MethodType(compatible_forward, model)


def select_indices(
    policy: str,
    *,
    total_frames: int,
    frame_budget: int,
    recent_frames: int,
    external_indices: list[int] | None = None,
) -> list[int]:
    if external_indices is not None:
        selected = sorted(set(int(index) for index in external_indices))
        if len(selected) != frame_budget:
            raise ValueError(
                f"external policy {policy} selected {len(selected)} frames, "
                f"expected {frame_budget}"
            )
        if selected[0] < 0 or selected[-1] >= total_frames:
            raise IndexError(
                f"external policy {policy} contains an out-of-range frame"
            )
        return selected
    if policy == "uniform":
        return uniform_frame_indices(total_frames, frame_budget)
    if policy == "recent":
        return recent_frame_indices(total_frames, frame_budget)
    if policy == "hybrid":
        return hybrid_frame_indices(
            total_frames,
            frame_budget,
            recent_count=recent_frames,
        )
    raise ValueError(f"unknown LLaVA frame policy: {policy}")


def run_inference(
    *,
    sample: object,
    policy: str,
    tokenizer: object,
    model: torch.nn.Module,
    image_processor: object,
    frame_budget: int,
    recent_frames: int,
    pool_grid: int,
    max_new_tokens: int,
    include_subtitle: bool,
    external_indices: list[int] | None,
    policy_accounting: dict[str, object] | None,
) -> dict[str, object]:
    from llava import conversation as conversation_lib
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import process_images, tokenizer_image_token

    total_frames, fps = video_metadata(sample.video_path)
    frame_indices = select_indices(
        policy,
        total_frames=total_frames,
        frame_budget=frame_budget,
        recent_frames=recent_frames,
        external_indices=external_indices,
    )
    decode_start = time.perf_counter()
    frames, decoded_fps, decoded_total = decode_video_frames(
        sample.video_path,
        frame_indices,
    )
    decode_seconds = time.perf_counter() - decode_start
    if decoded_total != total_frames:
        raise RuntimeError("video frame count changed during decoding")
    if decoded_fps > 0:
        fps = decoded_fps
    images = [Image.fromarray(frame) for frame in frames]
    image_sizes = [image.size for image in images]
    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
        )
    question = choice_prompt(
        sample,
        include_subtitle=include_subtitle,
    )
    query = "\n".join([image_token] * len(images)) + "\n" + question
    conversation = conversation_lib.conv_templates["llava_v1"].copy()
    conversation.append_message(conversation.roles[0], query)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    image_tensor = process_images(
        images,
        image_processor,
        model.config,
    ).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    inference_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    output = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()
    predicted = parse_choice_output(output, sample.candidates)
    answer_index = int(sample.answer_index)
    visual_tokens = len(images) * pool_grid * pool_grid
    accounting = policy_accounting or {}
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
        "policy": policy,
        "frame_budget": len(images),
        "recent_frames": (
            recent_frames
            if policy == "hybrid" and external_indices is None
            else 0
        ),
        "pool_grid": pool_grid,
        "visual_tokens": visual_tokens,
        "frame_indices_json": json.dumps(frame_indices),
        "fps": fps,
        "total_frames": total_frames,
        "raw_output": output,
        "predicted_index": predicted,
        "prediction": (
            sample.candidates[predicted] if predicted is not None else ""
        ),
        "parsed": int(predicted is not None),
        "correct": int(predicted == answer_index),
        "decode_seconds": decode_seconds,
        "inference_seconds": inference_seconds,
        "selection_state_proxy_bytes": int(
            accounting.get(
                "total_state_bytes",
                len(images) * 768 * 2,
            )
        ),
        "selection_online_bounded": int(
            accounting.get("online_bounded", policy == "recent")
        ),
        "selection_query_conditioned": int(
            accounting.get("query_conditioned", 0)
        ),
        "selection_retrieval_flops": int(
            accounting.get("estimated_retrieval_flops", 0)
        ),
        "visual_evidence_cache_counted": int(
            accounting.get("visual_evidence_cache_counted", 0)
        ),
        "llm_visual_token_bytes": visual_tokens * 4096 * 2,
    }


def load_manifest_samples(
    dataset_root: Path,
    manifest: dict[str, object],
    *,
    tasks: list[str],
    samples_per_task: int,
    selection_seed: int,
) -> list[object]:
    available: dict[str, list[int]] = {task: [] for task in tasks}
    for sample in manifest["samples"].values():
        task = str(sample["task"])
        if task in available:
            available[task].append(int(sample["sample_index"]))
    rng = np.random.default_rng(selection_seed)
    selected: dict[str, list[int]] = {}
    for task in tasks:
        values = sorted(set(available[task]))
        if samples_per_task > 0 and samples_per_task < len(values):
            values = sorted(
                int(value)
                for value in rng.choice(
                    values,
                    size=samples_per_task,
                    replace=False,
                )
            )
        selected[task] = values
    return load_mvbench_samples_by_indices(
        dataset_root,
        indices_by_task=selected,
    )


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(str(row["task"]), str(row["policy"]))].append(row)
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        output.append(
            {
                "task": key[0],
                "policy": key[1],
                "samples": len(values),
                "parsed_rate": float(
                    np.mean([int(row["parsed"]) for row in values])
                ),
                "accuracy": float(
                    np.mean([int(row["correct"]) for row in values])
                ),
                "mean_inference_seconds": float(
                    np.mean(
                        [float(row["inference_seconds"]) for row in values]
                    )
                ),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.llava_source))
    config = config_from_args(args)
    config_fingerprint = fingerprint(config)
    selection_manifest = (
        json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        if args.selection_manifest is not None
        else None
    )
    if selection_manifest is None:
        samples = load_mvbench_samples(
            args.dataset_root,
            tasks=[str(value) for value in config["tasks"]],
            samples_per_task=args.samples_per_task,
            selection_seed=args.selection_seed,
        )
    else:
        samples = load_manifest_samples(
            args.dataset_root,
            selection_manifest,
            tasks=[str(value) for value in config["tasks"]],
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

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(str(args.model_dir))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        str(args.model_dir),
        None,
        model_name,
        device_map="auto",
        device=args.device,
        torch_dtype=torch.float16,
    )
    install_transformers_forward_compatibility(model)
    install_visual_pooling(model, args.pool_grid)
    model.eval()

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    policies = [str(value) for value in config["policies"]]
    if selection_manifest is not None:
        missing = sorted(
            (
                sample.sample_id,
                policy,
            )
            for sample in samples
            for policy in policies
            if policy
            not in selection_manifest["samples"][sample.sample_id][
                "policies"
            ]
        )
        if missing:
            raise KeyError(f"selection manifest is missing policies: {missing}")
    for position, sample in enumerate(samples, start=1):
        checkpoint_path = (
            args.out_dir / "checkpoints" / f"{sample.sample_id}.json"
        )
        if checkpoint_path.exists() and not args.overwrite:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if payload.get("configuration_fingerprint") == config_fingerprint:
                rows.extend(payload["rows"])
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
        sample_rows = []
        try:
            for policy in policies:
                external_indices = (
                    selection_manifest["samples"][sample.sample_id][
                        "policies"
                    ][policy]
                    if selection_manifest is not None
                    else None
                )
                policy_accounting = (
                    selection_manifest.get("policy_accounting", {}).get(
                        policy
                    )
                    if selection_manifest is not None
                    else None
                )
                policy_start = time.perf_counter()
                row = run_inference(
                    sample=sample,
                    policy=policy,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    frame_budget=args.frame_budget,
                    recent_frames=args.recent_frames,
                    pool_grid=args.pool_grid,
                    max_new_tokens=args.max_new_tokens,
                    include_subtitle=args.include_subtitle,
                    external_indices=external_indices,
                    policy_accounting=policy_accounting,
                )
                row["policy_seconds"] = time.perf_counter() - policy_start
                sample_rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "policy_ok",
                            "sample": sample.sample_id,
                            "policy": policy,
                            "correct": row["correct"],
                            "parsed": row["parsed"],
                            "seconds": row["policy_seconds"],
                        }
                    ),
                    flush=True,
                )
            write_json_atomic(
                checkpoint_path,
                {
                    "configuration_fingerprint": config_fingerprint,
                    "rows": sample_rows,
                },
            )
            rows.extend(sample_rows)
        except Exception as exc:
            failure = {
                "sample": sample.sample_id,
                "task": sample.task,
                "type": type(exc).__name__,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            write_json_atomic(
                args.out_dir / "errors" / f"{sample.sample_id}.json",
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
    write_csv(args.out_dir / f"{shard_name}_predictions.csv", rows)
    write_csv(args.out_dir / f"{shard_name}_summary.csv", summarize(rows))
    write_json_atomic(
        args.out_dir / f"{shard_name}_status.json",
        {
            "configuration_fingerprint": config_fingerprint,
            "assigned_samples": len(samples),
            "completed_samples": len(
                {str(row["sample_id"]) for row in rows}
            ),
            "rows": len(rows),
            "failures": failures,
            "context_len": int(context_len),
            "model_name": model_name,
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
