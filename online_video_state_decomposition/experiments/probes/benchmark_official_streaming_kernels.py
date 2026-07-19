from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from packaging.version import InvalidVersion, Version


SOURCE_COMMITS = {
    "streamingtom": "6c66b05065692bc3fa4c6ec7fa9cad84d3b0cd75",
    "stc": "cf53f781d8740df5c07d7924756acc429641ffd0",
}
METHOD_SOURCE = {
    "streamingtom_ctr": "streamingtom",
    "streamingtom_oqm_write": "streamingtom",
    "streamingtom_oqm_select": "streamingtom",
    "stc_pruner": "stc",
}
REQUIRED_PATHS = {
    "streamingtom": (
        Path("streamingtom/modules/ctr.py"),
        Path("streamingtom/modules/oqm.py"),
        Path("streamingtom/modules/attention_processor.py"),
        Path("streamingtom/main.py"),
    ),
    "stc": (
        Path("stc/pruner/pruner.py"),
        Path("stc/pruner/scoring.py"),
        Path("stc/config.py"),
        Path("models/rekv/model/abstract_rekv.py"),
    ),
}
DEFAULT_FRAMES = {
    "streamingtom_ctr": 64,
    "streamingtom_oqm_write": 64,
    "streamingtom_oqm_select": 256,
    "stc_pruner": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pinned official streaming-video core modules on CUDA."
    )
    parser.add_argument("--method", choices=sorted(METHOD_SOURCE), required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help=(
            "Zero selects a method-specific default. OQM selection uses 256 "
            "frames so the official top-k branch is exercised."
        ),
    )
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--stc-tokens-per-frame", type=int, default=64)
    parser.add_argument("--max-idle-memory-mib", type=int, default=4096)
    parser.add_argument("--max-idle-utilization", type=int, default=20)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {output}")
    return completed.stdout.strip()


def _cache_only_status(line: str) -> bool:
    paths = line[3:].split(" -> ")
    return all(
        "__pycache__/" in path.replace("\\", "/") or path.endswith(".pyc")
        for path in paths
    )


def validate_checkout(
    checkout: Path,
    *,
    source_name: str,
) -> dict[str, Any]:
    if not (checkout / ".git").is_dir():
        raise FileNotFoundError(f"official checkout not found: {checkout}")
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=checkout)
    expected = SOURCE_COMMITS[source_name]
    if commit != expected:
        raise ValueError(
            f"{source_name} commit mismatch: expected {expected}, found {commit}"
        )
    missing = [
        str(path)
        for path in REQUIRED_PATHS[source_name]
        if not (checkout / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"official checkout is missing files: {missing}")
    status = _run_text(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=checkout
    )
    non_cache = [
        line for line in status.splitlines() if line and not _cache_only_status(line)
    ]
    if non_cache:
        raise ValueError(f"official checkout has source changes: {non_cache}")
    return {
        "name": source_name,
        "path": str(checkout),
        "commit": commit,
        "required_paths": [str(path) for path in REQUIRED_PATHS[source_name]],
        "code_clean": True,
    }


def query_gpu_state(gpu_index: int) -> dict[str, Any]:
    output = _run_text(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    fields = [field.strip() for field in output.splitlines()[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi output: {output}")
    return {
        "name": fields[0],
        "uuid": fields[1],
        "driver_version": fields[2],
        "memory_used_mib": int(fields[3]),
        "memory_total_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
    }


def higher_quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * (len(ordered) - 1)))
    return ordered[index]


def summarize_distribution(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": higher_quantile(values, 0.50),
        "p95": higher_quantile(values, 0.95),
        "p99": higher_quantile(values, 0.99),
        "max": max(values),
    }


def resolve_frames(method: str, requested: int) -> int:
    if requested < 0:
        raise ValueError("frames must be non-negative")
    return requested or DEFAULT_FRAMES[method]


def retrieval_group_budget(
    max_tokens: int, group_size: int, available_groups: int
) -> int:
    if max_tokens <= 0 or group_size <= 0 or available_groups <= 0:
        raise ValueError("retrieval budget inputs must be positive")
    return min(available_groups, math.ceil(max_tokens / group_size))


def base_version_equals(observed: str | None, expected: str) -> bool:
    if observed is None:
        return False
    try:
        return Version(observed).base_version == Version(expected).base_version
    except InvalidVersion:
        return False


def version_at_least(observed: str | None, minimum: str) -> bool:
    if observed is None:
        return False
    try:
        return Version(observed).base_version >= Version(minimum).base_version
    except InvalidVersion:
        return False


def configure_streamingtom_environment() -> dict[str, str]:
    values = {
        "CTR_BETA": "0.6",
        "CTR_K": "7",
        "CTR_RETAIN_TOKENS": "50",
        "CTR_SIMILARITY_THRESHOLD": "0.9",
        "OQM_ENABLE_QUANTIZATION": "1",
        "OQM_GROUP_SIZE": "50",
        "OQM_INIT_TOKEN_COUNT": "14",
        "OQM_QUANTIZATION_BITS": "4",
        "OQM_RETRIEVAL_MAX_TOKENS": "12544",
        "OQM_SLIDING_WINDOW_SIZE": "4800",
        "STREAMING_ENCODER_BATCH_SIZE": "32",
    }
    os.environ.update(values)
    return values


def _dtype(torch: Any, name: str) -> Any:
    return torch.float16 if name == "float16" else torch.bfloat16


def _randn(
    torch: Any, shape: tuple[int, ...], *, device: Any, dtype: Any, seed: int
) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def _build_correlated_ctr_inputs(
    torch: Any,
    *,
    device: Any,
    dtype: Any,
    frames: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    static_tokens = 147
    previous = _randn(
        torch, (1, 196, 3584), device=device, dtype=dtype, seed=seed
    )
    frame_list = []
    prior = previous[0]
    for frame_idx in range(frames):
        correlated_noise = _randn(
            torch,
            (196, 3584),
            device=device,
            dtype=dtype,
            seed=seed + 10 + frame_idx * 2,
        )
        dynamic = _randn(
            torch,
            (196 - static_tokens, 3584),
            device=device,
            dtype=dtype,
            seed=seed + 11 + frame_idx * 2,
        )
        current = prior + correlated_noise * 0.05
        current = torch.cat([current[:static_tokens], dynamic], dim=0)
        frame_list.append(current)
        prior = current
    features = torch.stack(frame_list)
    attention = _randn(
        torch, (frames, 196), device=device, dtype=dtype, seed=seed + 1000
    )
    return features, attention, previous


def build_streamingtom_ctr(
    torch: Any,
    *,
    checkout: Path,
    device: Any,
    dtype: Any,
    frames: int,
    seed: int,
) -> tuple[Callable[[], Any], Callable[[Any], Any], Callable[[], dict[str, Any]]]:
    sys.path.insert(0, str(checkout))
    from streamingtom.modules.ctr import CTR

    module = CTR()
    features, attention, previous = _build_correlated_ctr_inputs(
        torch,
        device=device,
        dtype=dtype,
        frames=frames,
        seed=seed,
    )
    batch_size = int(os.environ["STREAMING_ENCODER_BATCH_SIZE"])

    def prepare() -> dict[str, Any]:
        return {"ctr_state": {"last_frame_tokens": previous}}

    def invoke(state: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        outputs = []
        for start in range(0, frames, batch_size):
            stop = min(frames, start + batch_size)
            compressed, state = module.compress_features(
                features[start:stop], state, attention[start:stop]
            )
            outputs.append(compressed)
        return torch.cat(outputs, dim=0), state

    def quality() -> dict[str, Any]:
        classification_state = {"last_frame_tokens": previous}
        static_masks, dynamic_masks = module._classify_tokens(
            features, classification_state
        )
        static_per_frame = static_masks.sum(dim=1)
        dynamic_per_frame = dynamic_masks.sum(dim=1)
        torch.manual_seed(seed + 100)
        torch.cuda.manual_seed_all(seed + 100)
        first, first_state = invoke(prepare())
        torch.manual_seed(seed + 100)
        torch.cuda.manual_seed_all(seed + 100)
        second, second_state = invoke(prepare())
        torch.manual_seed(seed + 100)
        torch.cuda.manual_seed_all(seed + 100)
        one_shot, one_shot_state = module.compress_features(
            features,
            prepare(),
            attention,
        )
        exact = torch.equal(first, second)
        one_shot_exact = torch.equal(first, one_shot)
        final_state_exact = bool(
            torch.equal(
                first_state["ctr_state"]["last_frame_tokens"],
                features[-1:],
            )
            and torch.equal(
                second_state["ctr_state"]["last_frame_tokens"],
                one_shot_state["ctr_state"]["last_frame_tokens"],
            )
        )
        finite = bool(torch.isfinite(first).all().item())
        expected_shape = [frames * 50, 3584]
        batch_count = math.ceil(frames / batch_size)
        mixed_every_frame = bool(
            ((static_per_frame > 0) & (dynamic_per_frame > 0)).all().item()
        )
        return {
            "passed": (
                exact
                and one_shot_exact
                and final_state_exact
                and finite
                and mixed_every_frame
                and batch_count >= 2
                and list(first.shape) == expected_shape
            ),
            "deterministic_exact": exact,
            "incremental_matches_one_shot_exact": one_shot_exact,
            "final_state_matches_last_frame_exact": final_state_exact,
            "finite": finite,
            "output_shape": list(first.shape),
            "expected_shape": expected_shape,
            "streaming_batch_size_frames": batch_size,
            "streaming_batch_count": batch_count,
            "cross_batch_state_required": True,
            "static_tokens": int(static_masks.sum().item()),
            "dynamic_tokens": int(dynamic_masks.sum().item()),
            "mixed_static_dynamic_every_frame": mixed_every_frame,
            "dpc_path_exercised": bool((static_per_frame > 0).all().item()),
            "attention_path_exercised": bool((dynamic_per_frame > 0).all().item()),
        }

    return prepare, invoke, quality


def _build_oqm_inputs(
    torch: Any,
    *,
    device: Any,
    dtype: Any,
    frames: int,
    seed: int,
) -> dict[str, Any]:
    vision_tokens = frames * 50
    return {
        "prompt_k": _randn(
            torch, (1, 4, 14, 128), device=device, dtype=dtype, seed=seed
        ),
        "prompt_v": _randn(
            torch, (1, 4, 14, 128), device=device, dtype=dtype, seed=seed + 1
        ),
        "vision_k": _randn(
            torch,
            (1, 4, vision_tokens, 128),
            device=device,
            dtype=dtype,
            seed=seed + 2,
        ),
        "vision_v": _randn(
            torch,
            (1, 4, vision_tokens, 128),
            device=device,
            dtype=dtype,
            seed=seed + 3,
        ),
    }


def _reset_oqm_retrieval_context(context: Any, oqm: Any) -> None:
    context.retrieved_layers.clear()
    context.selected_vision_group_indices_per_layer.clear()
    context.mode = "retrieve"
    context.video_id = "benchmark"
    context.set_oqm(oqm)
    context.should_store_keys = False
    context.retrieval_info = {
        "video_id": "benchmark",
        "budget": oqm.retrieval_max_tokens,
    }


def _populate_oqm_incremental(
    oqm: Any,
    inputs: dict[str, Any],
    layers: int,
    *,
    processor: Any,
    context: Any,
    batch_size_frames: int,
) -> int:
    for layer in range(layers):
        oqm.store_system_prompt(
            "benchmark", layer, inputs["prompt_k"], inputs["prompt_v"]
        )

    config = SimpleNamespace(num_attention_heads=inputs["vision_k"].shape[1])
    tokens_per_batch = batch_size_frames * oqm.group_size
    total_tokens = inputs["vision_k"].shape[2]
    batch_count = math.ceil(total_tokens / tokens_per_batch)
    for batch_idx, start in enumerate(range(0, total_tokens, tokens_per_batch)):
        stop = min(total_tokens, start + tokens_per_batch)
        context.set_encode_mode("benchmark", batch_idx)
        context.set_oqm(oqm)
        context.should_store_keys = True
        for layer in range(layers):
            key_batch = inputs["vision_k"][:, :, start:stop, :]
            value_batch = inputs["vision_v"][:, :, start:stop, :]
            processor.process(
                layer,
                key_batch,
                key_batch,
                value_batch,
                config,
            )
            oqm.store_kv_cache("benchmark", layer, key_batch, value_batch)
        context.should_store_keys = False
    return batch_count


def _selected_group_reference(
    tensor: Any, selected: Any, group_size: int
) -> Any:
    batch, heads, tokens, width = tensor.shape
    grouped = tensor.reshape(batch, heads, tokens // group_size, group_size, width)
    return grouped[:, :, selected].reshape(
        batch, heads, len(selected) * group_size, width
    )


def _oqm_quality(
    torch: Any,
    oqm: Any,
    inputs: dict[str, Any],
    processor: Any,
    context: Any,
    queries: Any,
    layers: int,
    *,
    require_topk: bool,
) -> dict[str, Any]:
    available_groups = inputs["vision_k"].shape[2] // oqm.group_size
    expected_groups = retrieval_group_budget(
        oqm.retrieval_max_tokens,
        oqm.group_size,
        available_groups,
    )
    expected_shape = [
        1,
        4,
        oqm.init_token_count + expected_groups * oqm.group_size,
        128,
    ]
    prompt_exact_all = True
    finite_all = True
    sorted_unique_all = True
    group_keys_valid_all = True
    selected_counts: list[int] = []
    key_max = 0.0
    value_max = 0.0
    key_bound = 0.0
    value_bound = 0.0
    shapes: list[list[int]] = []
    config = SimpleNamespace(num_attention_heads=inputs["vision_k"].shape[1])
    _reset_oqm_retrieval_context(context, oqm)

    for layer in range(layers):
        group_keys = oqm.get_group_keys("benchmark", layer)
        group_keys_valid = (
            group_keys is not None
            and list(group_keys.shape) == [available_groups, 512]
            and bool(torch.isfinite(group_keys).all().item())
        )
        group_keys_valid_all = group_keys_valid_all and group_keys_valid
        if group_keys is None:
            continue
        query_states = queries[layer].reshape(
            1,
            inputs["vision_k"].shape[1],
            1,
            inputs["vision_k"].shape[-1],
        )
        keys, values = processor.process(
            layer,
            query_states,
            inputs["prompt_k"],
            inputs["prompt_v"],
            config,
        )
        selected = context.selected_vision_group_indices_per_layer[layer]
        selected_counts.append(len(selected))
        sorted_unique = bool(
            (selected[1:] > selected[:-1]).all().item()
            if len(selected) > 1
            else True
        ) and len(torch.unique(selected)) == len(selected)
        sorted_unique_all = sorted_unique_all and sorted_unique
        shapes.append(list(keys.shape))
        prompt_exact = torch.equal(
            keys[:, :, : oqm.init_token_count], inputs["prompt_k"]
        ) and torch.equal(
            values[:, :, : oqm.init_token_count], inputs["prompt_v"]
        )
        prompt_exact_all = prompt_exact_all and prompt_exact
        finite_all = finite_all and bool(
            torch.isfinite(keys).all().item()
            and torch.isfinite(values).all().item()
        )
        expected_k = _selected_group_reference(
            inputs["vision_k"], selected, oqm.group_size
        )
        expected_v = _selected_group_reference(
            inputs["vision_v"], selected, oqm.group_size
        )
        key_max = max(
            key_max,
            float(
                (keys[:, :, oqm.init_token_count :] - expected_k)
                .abs()
                .amax()
                .item()
            ),
        )
        value_max = max(
            value_max,
            float(
                (values[:, :, oqm.init_token_count :] - expected_v)
                .abs()
                .amax()
                .item()
            ),
        )
        storage = oqm.quantized_storage["benchmark"][layer]
        key_bound = max(
            key_bound,
            max(float(scales.amax().item()) for scales in storage["keys_scales"])
            / 2.0
            + 2e-3,
        )
        value_bound = max(
            value_bound,
            max(
                float(scales.amax().item())
                for scales in storage["values_scales"]
            )
            / 2.0
            + 2e-3,
        )
        del keys, values, expected_k, expected_v

    count_exact = selected_counts == [expected_groups] * layers
    shapes_exact = shapes == [expected_shape] * layers
    topk_exercised = expected_groups < available_groups
    context_control_flow_exercised = (
        context.retrieved_layers == set(range(layers))
        and set(context.selected_vision_group_indices_per_layer) == set(range(layers))
    )
    passed = (
        len(selected_counts) == layers
        and count_exact
        and shapes_exact
        and prompt_exact_all
        and finite_all
        and sorted_unique_all
        and group_keys_valid_all
        and key_max <= key_bound
        and value_max <= value_bound
        and context_control_flow_exercised
        and (topk_exercised or not require_topk)
    )
    return {
        "passed": passed,
        "checked_layers": len(selected_counts),
        "prompt_bit_exact_all_layers": prompt_exact_all,
        "finite_all_layers": finite_all,
        "sorted_unique_indices_all_layers": sorted_unique_all,
        "group_keys_valid_all_layers": group_keys_valid_all,
        "key_max_abs_error": key_max,
        "key_scale_half_bound": key_bound,
        "value_max_abs_error": value_max,
        "value_scale_half_bound": value_bound,
        "selected_output_shape": expected_shape,
        "selected_groups": expected_groups,
        "available_groups": available_groups,
        "retrieval_max_tokens": oqm.retrieval_max_tokens,
        "topk_branch_exercised": topk_exercised,
        "topk_branch_required": require_topk,
        "official_context_control_flow_exercised": context_control_flow_exercised,
    }


def build_streamingtom_oqm_write(
    torch: Any,
    *,
    checkout: Path,
    device: Any,
    dtype: Any,
    frames: int,
    layers: int,
    seed: int,
) -> tuple[Callable[[], Any], Callable[[Any], Any], Callable[[], dict[str, Any]]]:
    sys.path.insert(0, str(checkout))
    from streamingtom.modules.attention_processor import (
        StreamingTOMAttentionProcessor,
    )
    from streamingtom.modules.oqm import OQM
    from streamingtom.modules.streamingtom_context import StreamingTOMContext

    inputs = _build_oqm_inputs(
        torch, device=device, dtype=dtype, frames=frames, seed=seed
    )
    queries = _randn(
        torch, (layers, 512), device=device, dtype=dtype, seed=seed + 5
    )

    batch_size_frames = int(os.environ["STREAMING_ENCODER_BATCH_SIZE"])

    def prepare() -> dict[str, Any]:
        oqm = OQM()
        context = StreamingTOMContext()
        return {
            "oqm": oqm,
            "context": context,
            "processor": StreamingTOMAttentionProcessor(context, oqm),
        }

    def invoke(state: dict[str, Any]) -> Any:
        batch_count = _populate_oqm_incremental(
            state["oqm"],
            inputs,
            layers,
            processor=state["processor"],
            context=state["context"],
            batch_size_frames=batch_size_frames,
        )
        return state["oqm"], batch_count

    def quality() -> dict[str, Any]:
        state = prepare()
        oqm, batch_count = invoke(state)
        result = _oqm_quality(
            torch,
            oqm,
            inputs,
            state["processor"],
            state["context"],
            queries,
            layers,
            require_topk=False,
        )
        result.update(
            {
                "layers": layers,
                "streaming_batch_size_frames": batch_size_frames,
                "streaming_batch_count": batch_count,
                "incremental_append_required": True,
            }
        )
        result["passed"] = result["passed"] and batch_count >= 2
        return result

    return prepare, invoke, quality


def build_streamingtom_oqm_select(
    torch: Any,
    *,
    checkout: Path,
    device: Any,
    dtype: Any,
    frames: int,
    layers: int,
    seed: int,
) -> tuple[Callable[[], Any], Callable[[Any], Any], Callable[[], dict[str, Any]]]:
    sys.path.insert(0, str(checkout))
    from streamingtom.modules.attention_processor import (
        StreamingTOMAttentionProcessor,
    )
    from streamingtom.modules.oqm import OQM
    from streamingtom.modules.streamingtom_context import StreamingTOMContext

    inputs = _build_oqm_inputs(
        torch, device=device, dtype=dtype, frames=frames, seed=seed
    )
    oqm = OQM()
    context = StreamingTOMContext()
    processor = StreamingTOMAttentionProcessor(context, oqm)
    batch_size_frames = int(os.environ["STREAMING_ENCODER_BATCH_SIZE"])
    batch_count = _populate_oqm_incremental(
        oqm,
        inputs,
        layers,
        processor=processor,
        context=context,
        batch_size_frames=batch_size_frames,
    )
    queries = _randn(
        torch, (layers, 512), device=device, dtype=dtype, seed=seed + 5
    )

    state = {
        "oqm": oqm,
        "context": context,
        "processor": processor,
    }

    def prepare() -> dict[str, Any]:
        return state

    def invoke(active: dict[str, Any]) -> Any:
        active_oqm = active["oqm"]
        active_context = active["context"]
        active_processor = active["processor"]
        config = SimpleNamespace(
            num_attention_heads=inputs["vision_k"].shape[1]
        )
        _reset_oqm_retrieval_context(active_context, active_oqm)
        last_selected = None
        for layer in range(layers):
            query_states = queries[layer].reshape(
                1,
                inputs["vision_k"].shape[1],
                1,
                inputs["vision_k"].shape[-1],
            )
            keys, values = active_processor.process(
                layer,
                query_states,
                inputs["prompt_k"],
                inputs["prompt_v"],
                config,
            )
            last_selected = (
                active_context.selected_vision_group_indices_per_layer[layer]
            )
            del keys, values
        return last_selected

    def quality() -> dict[str, Any]:
        result = _oqm_quality(
            torch,
            oqm,
            inputs,
            processor,
            context,
            queries,
            layers,
            require_topk=True,
        )
        result.update(
            {
                "layers": layers,
                "retrieval_max_tokens": oqm.retrieval_max_tokens,
                "streaming_batch_size_frames": batch_size_frames,
                "streaming_batch_count": batch_count,
                "incremental_append_exercised": batch_count >= 2,
            }
        )
        result["passed"] = result["passed"] and batch_count >= 2
        return result

    return prepare, invoke, quality


def build_stc_pruner(
    torch: Any,
    *,
    checkout: Path,
    device: Any,
    dtype: Any,
    frames: int,
    tokens_per_frame: int,
    seed: int,
) -> tuple[Callable[[], Any], Callable[[Any], Any], Callable[[], dict[str, Any]]]:
    sys.path.insert(0, str(checkout))
    from stc.pruner.pruner import STCPruner

    module = STCPruner()
    features = _randn(
        torch, (frames, 196, 3584), device=device, dtype=dtype, seed=seed
    )

    def run_sequence(pruner: Any, budget: int, *, collect: bool) -> Any:
        outputs = []
        last_output = None
        for frame_features in features.unbind(0):
            last_output = pruner.compress(
                frame_features,
                model="llava_ov",
                tokens_per_frame=budget,
                score_strategy="gaussian",
            )
            if collect:
                outputs.append(last_output)
        if collect:
            return torch.cat(outputs, dim=0)
        return last_output

    def prepare() -> Any:
        module.reset()
        return module

    def invoke(pruner: Any) -> Any:
        return run_sequence(pruner, tokens_per_frame, collect=False)

    def quality() -> dict[str, Any]:
        module.reset()
        identity = run_sequence(module, 196, collect=True)
        identity_state_updates = len(module.past_memory_mean_token)
        module.reset()
        compressed = run_sequence(module, tokens_per_frame, collect=True)
        compressed_state_updates = len(module.past_memory_mean_token)
        exact = torch.equal(identity, features.reshape(frames * 196, 3584))
        finite = bool(torch.isfinite(compressed).all().item())
        expected_shape = [frames * tokens_per_frame, 3584]
        return {
            "passed": (
                exact
                and finite
                and identity_state_updates == frames
                and compressed_state_updates == frames
                and list(compressed.shape) == expected_shape
            ),
            "identity_196_bit_exact": exact,
            "finite": finite,
            "compressed_shape": list(compressed.shape),
            "expected_shape": expected_shape,
            "frames": frames,
            "compress_calls_per_sample": frames,
            "identity_state_updates": identity_state_updates,
            "compressed_state_updates": compressed_state_updates,
            "state_preserved_across_frames": compressed_state_updates == frames,
            "score_strategy": "gaussian_fixed_source_fast_path",
        }

    return prepare, invoke, quality


def measure(
    torch: Any,
    *,
    prepare: Callable[[], Any],
    invoke: Callable[[Any], Any],
    warmup: int,
    repeat: int,
) -> list[dict[str, float | int]]:
    for _ in range(warmup):
        context = prepare()
        output = invoke(context)
        torch.cuda.synchronize()
        del output, context

    rows = []
    for iteration in range(repeat):
        context = prepare()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start_event.record()
        output = invoke(context)
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
        cuda_ms = start_event.elapsed_time(end_event)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        rows.append(
            {
                "iteration": iteration,
                "wall_ms": wall_ms,
                "cuda_event_ms": cuda_ms,
                "baseline_allocated_mib": baseline_allocated / (1024**2),
                "baseline_reserved_mib": baseline_reserved / (1024**2),
                "peak_allocated_mib": peak_allocated / (1024**2),
                "peak_reserved_mib": peak_reserved / (1024**2),
                "peak_allocated_delta_mib": (peak_allocated - baseline_allocated)
                / (1024**2),
                "peak_reserved_delta_mib": (peak_reserved - baseline_reserved)
                / (1024**2),
            }
        )
        del output, context, start_event, end_event
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runtime_versions(method: str, torch: Any) -> dict[str, Any]:
    packages = {}
    for name in ("transformers", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": packages,
    }
    if method.startswith("streamingtom_"):
        expected = {
            "torch": "2.5.1",
            "transformers": "4.53.3",
            "flash-attn": "2.8.0.post2",
        }
        observed = {
            "torch": result["torch"],
            "transformers": packages["transformers"],
            "flash-attn": packages["flash-attn"],
        }
        mismatches = {
            name: {"expected": version, "observed": observed[name]}
            for name, version in expected.items()
            if not base_version_equals(observed[name], version)
        }
        if mismatches:
            raise ValueError(f"StreamingTOM runtime version mismatch: {mismatches}")
    else:
        minimums = {
            "python": "3.10",
            "torch": "2.1",
        }
        observed = {
            "python": result["python"],
            "torch": result["torch"],
        }
        mismatches = {
            name: {"minimum": version, "observed": observed[name]}
            for name, version in minimums.items()
            if not version_at_least(observed[name], version)
        }
        if mismatches:
            raise ValueError(f"STC runtime version mismatch: {mismatches}")
    return result


def main() -> int:
    args = parse_args()
    frames = resolve_frames(args.method, args.frames)
    if frames < 1 or args.layers < 1 or args.warmup < 0 or args.repeat < 1:
        raise ValueError(
            "frames/layers/repeat must be positive and warmup non-negative"
        )
    source_name = METHOD_SOURCE[args.method]
    directory_name = "StreamingTOM" if source_name == "streamingtom" else "STC"
    checkout = (args.external_root / directory_name).resolve()
    source = validate_checkout(checkout, source_name=source_name)
    if args.method == "stc_pruner":
        scope_warning = (
            "Official STC-Pruner core latency only; this does not execute "
            "STC-Cacher, ReKV, ViT encoding, or LLM prefill."
        )
    else:
        scope_warning = (
            "Official StreamingTOM core-module latency with incremental "
            "32-frame state flow; not end-to-end Video-LLM latency."
        )
    spec = {
        "format_version": 2,
        "evidence_tier": "official_core_gpu_microbenchmark",
        "method": args.method,
        "source": source,
        "frames": frames,
        "layers": args.layers,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "dtype": args.dtype,
        "stc_tokens_per_frame": args.stc_tokens_per_frame,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "scope_warning": scope_warning,
    }
    if args.dry_run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return 0

    gpu_before = query_gpu_state(args.gpu_index)
    if not args.allow_busy_gpu and (
        gpu_before["memory_used_mib"] > args.max_idle_memory_mib
        or gpu_before["utilization_percent"] > args.max_idle_utilization
    ):
        raise RuntimeError(f"GPU idle gate failed: {gpu_before}")
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.pop("PREFIX", None)
    method_environment = (
        configure_streamingtom_environment() if source_name == "streamingtom" else {}
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected Python environment")
    runtime = runtime_versions(args.method, torch)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    dtype = _dtype(torch, args.dtype)
    if args.method == "streamingtom_ctr":
        prepare, invoke, quality = build_streamingtom_ctr(
            torch,
            checkout=checkout,
            device=device,
            dtype=dtype,
            frames=frames,
            seed=args.seed,
        )
    elif args.method == "streamingtom_oqm_write":
        prepare, invoke, quality = build_streamingtom_oqm_write(
            torch,
            checkout=checkout,
            device=device,
            dtype=dtype,
            frames=frames,
            layers=args.layers,
            seed=args.seed,
        )
    elif args.method == "streamingtom_oqm_select":
        prepare, invoke, quality = build_streamingtom_oqm_select(
            torch,
            checkout=checkout,
            device=device,
            dtype=dtype,
            frames=frames,
            layers=args.layers,
            seed=args.seed,
        )
    else:
        prepare, invoke, quality = build_stc_pruner(
            torch,
            checkout=checkout,
            device=device,
            dtype=dtype,
            frames=frames,
            tokens_per_frame=args.stc_tokens_per_frame,
            seed=args.seed,
        )

    quality_result = quality()
    if not quality_result["passed"]:
        raise RuntimeError(f"quality gate failed before timing: {quality_result}")
    out_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    rows = measure(
        torch,
        prepare=prepare,
        invoke=invoke,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    wall = [float(row["wall_ms"]) for row in rows]
    cuda = [float(row["cuda_event_ms"]) for row in rows]
    allocated = [float(row["peak_allocated_mib"]) for row in rows]
    reserved = [float(row["peak_reserved_mib"]) for row in rows]
    allocated_delta = [float(row["peak_allocated_delta_mib"]) for row in rows]
    reserved_delta = [float(row["peak_reserved_delta_mib"]) for row in rows]
    summary = {
        **spec,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_before,
        "runtime": runtime,
        "method_environment": method_environment,
        "quality_gate": quality_result,
        "wall_ms": summarize_distribution(wall),
        "cuda_event_ms": summarize_distribution(cuda),
        "peak_allocated_mib": summarize_distribution(allocated),
        "peak_reserved_mib": summarize_distribution(reserved),
        "peak_allocated_delta_mib": summarize_distribution(allocated_delta),
        "peak_reserved_delta_mib": summarize_distribution(reserved_delta),
        "tail_latency_protocol": {
            "quantile_method": "higher",
            "global_cuda_synchronize_per_iteration": True,
            "input_preparation_timed": False,
            "model_loading_timed": False,
        },
    }
    write_rows(out_dir / "samples.csv", rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
