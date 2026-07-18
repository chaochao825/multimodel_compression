#!/usr/bin/env python3
"""Benchmark TileLogic-RVQ codec components and honest end-to-end TTFT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Sequence

import torch

from scripts.run_tilespec_ex_quality import (
    DEFAULT_MODEL,
    TILE_PIXELS,
    _cached_visual_forward,
    _join_visual_features,
    _load_manifest,
    _multi_tile_images,
    _processor_inputs,
    _query_embedding,
    _split_visual_features,
)
from tilespec_ex.cache import (
    CacheEntry,
    load_cache_manifest,
    load_cache_payload,
    validate_split_contract,
)
from tilespec_ex.core import RETENTION_RATES, enumerate_blocks
from tilespec_ex.tilelogic_codec import (
    decode_residual_payload,
    encode_base_vq,
    extract_tile_coefficients,
)
from tilespec_ex.tilelogic_methods import (
    build_dynamic_logic_variant,
    build_fixed_logic_variant,
    load_tilelogic_artifacts,
)


LATENCY_FORMAT = "tilelogic_latency_v1"


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _cuda_benchmark(
    function: Callable[[], Any],
    *,
    warmup: int,
    trials: int,
) -> tuple[dict[str, float], int, int]:
    if warmup < 0 or trials <= 0:
        raise ValueError("invalid CUDA benchmark counts")
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return (
        _summary(samples),
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
    )


def _wall_benchmark(
    function: Callable[[], Any],
    *,
    warmup: int,
    trials: int,
) -> tuple[dict[str, float], int, int]:
    for _ in range(warmup):
        function()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(trials):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return (
        _summary(samples),
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
    )


def _row(
    entry: CacheEntry,
    *,
    scope: str,
    component: str,
    method: str,
    rate: float | None,
    timing: dict[str, float],
    peak_allocated: int,
    peak_reserved: int,
    trials: int,
    includes: dict[str, bool],
) -> dict[str, Any]:
    return {
        "dataset": entry.dataset,
        "dataset_index": entry.dataset_index,
        "sample_id": entry.sample_id,
        "scope": scope,
        "component": component,
        "method": method,
        "retention_rate": rate,
        "trials": trials,
        **timing,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        **{f"includes_{name}": value for name, value in includes.items()},
    }


def _component_rows(
    entry: CacheEntry,
    payload: dict[str, Any],
    artifacts: Any,
    *,
    warmup: int,
    trials: int,
) -> list[dict[str, Any]]:
    crops = payload["crops"]
    thumbnail = payload["thumbnail"]
    query = payload["query"]
    rows = []
    component_includes = {
        "image_preprocess": False,
        "visual_encoder": False,
        "codec_encode": True,
        "codec_decode": False,
        "native_positions": False,
        "language_prefill": False,
        "first_token": False,
    }
    for rate in RETENTION_RATES:
        retained = round(1024 * rate)
        exception_tokens = round(retained * 0.25 / 4) * 4
        base_tokens = retained - exception_tokens
        coefficients = extract_tile_coefficients(crops, base_tokens)
        timing, allocated, reserved = _cuda_benchmark(
            lambda: artifacts.base.encode(coefficients), warmup=warmup, trials=trials
        )
        rows.append(
            _row(
                entry,
                scope="component",
                component="base_vq_search",
                method="base_vq",
                rate=rate,
                timing=timing,
                peak_allocated=allocated,
                peak_reserved=reserved,
                trials=trials,
                includes=component_includes,
            )
        )
        base = encode_base_vq(crops, base_tokens, artifacts.base).reconstructed
        residual = crops - base
        blocks, locations, _ = enumerate_blocks(residual)
        flat_blocks = blocks.reshape(blocks.shape[0], -1)
        stage_indices, scale_indices = artifacts.residual_fisher.encode(flat_blocks)
        timing, allocated, reserved = _cuda_benchmark(
            lambda: artifacts.residual_fisher.encode(flat_blocks),
            warmup=warmup,
            trials=trials,
        )
        rows.append(
            _row(
                entry,
                scope="component",
                component="residual_rvq_search",
                method="base_vq_residual_rvq",
                rate=rate,
                timing=timing,
                peak_allocated=allocated,
                peak_reserved=reserved,
                trials=trials,
                includes=component_includes,
            )
        )
        bundle = artifacts.routers[rate]
        from tilespec_ex.routing import block_router_features

        features, _ = block_router_features(
            crops,
            residual,
            query,
            thumbnail,
            curvature_prior=bundle.curvature_prior,
        )
        timing, allocated, reserved = _cuda_benchmark(
            lambda: bundle.logic.predict(features), warmup=warmup, trials=trials
        )
        rows.append(
            _row(
                entry,
                scope="component",
                component="logic_router",
                method="base_vq_logic_router",
                rate=rate,
                timing=timing,
                peak_allocated=allocated,
                peak_reserved=reserved,
                trials=trials,
                includes=component_includes,
            )
        )
        dynamic_variant = build_dynamic_logic_variant(
            thumbnail, crops, query, artifacts, rate
        )
        fixed_variant = build_fixed_logic_variant(
            thumbnail, crops, query, artifacts, rate
        )
        if dynamic_variant.residual_modes is None:
            raise AssertionError("dynamic logic variant has no residual modes")
        if fixed_variant.residual_modes is None:
            raise AssertionError("fixed-slot variant has no residual modes")
        active_dynamic = dynamic_variant.residual_modes > 0
        fixed_mask = bundle.fixed_slot_mask
        for method, active in (
            ("base_vq_logic_router", active_dynamic),
            ("logic_router_fixed_slots", fixed_mask),
        ):
            timing, allocated, reserved = _cuda_benchmark(
                lambda active=active: (
                    flat_blocks[active],
                    stage_indices[active],
                ),
                warmup=warmup,
                trials=trials,
            )
            rows.append(
                _row(
                    entry,
                    scope="component",
                    component="layout_pack",
                    method=method,
                    rate=rate,
                    timing=timing,
                    peak_allocated=allocated,
                    peak_reserved=reserved,
                    trials=trials,
                    includes=component_includes,
                )
            )
        decode_includes = dict(component_includes)
        decode_includes["codec_encode"] = False
        decode_includes["codec_decode"] = True
        for method, variant in (
            ("base_vq_logic_router", dynamic_variant),
            ("logic_router_fixed_slots", fixed_variant),
        ):
            timing, allocated, reserved = _cuda_benchmark(
                lambda modes=variant.residual_modes: decode_residual_payload(
                    base,
                    blocks,
                    locations,
                    artifacts.residual_fisher,
                    stage_indices,
                    scale_indices,
                    modes,
                    output_dtype=crops.dtype,
                ),
                warmup=warmup,
                trials=trials,
            )
            rows.append(
                _row(
                    entry,
                    scope="component",
                    component="residual_decode_scatter",
                    method=method,
                    rate=rate,
                    timing=timing,
                    peak_allocated=allocated,
                    peak_reserved=reserved,
                    trials=trials,
                    includes=decode_includes,
                )
            )
        roundtrip_includes = dict(component_includes)
        roundtrip_includes["codec_decode"] = True
        for method, builder in (
            (
                "base_vq_logic_router",
                lambda: build_dynamic_logic_variant(
                    thumbnail, crops, query, artifacts, rate
                ),
            ),
            (
                "logic_router_fixed_slots",
                lambda: build_fixed_logic_variant(
                    thumbnail, crops, query, artifacts, rate
                ),
            ),
        ):
            timing, allocated, reserved = _cuda_benchmark(
                builder,
                warmup=warmup,
                trials=trials,
            )
            rows.append(
                _row(
                    entry,
                    scope="component",
                    component="codec_roundtrip",
                    method=method,
                    rate=rate,
                    timing=timing,
                    peak_allocated=allocated,
                    peak_reserved=reserved,
                    trials=trials,
                    includes=roundtrip_includes,
                )
            )
    return rows


def _language_forward(
    model: Any,
    inputs: Any,
    cached_visual: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    with _cached_visual_forward(model.visual, cached_visual), torch.inference_mode():
        outputs = model(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            pixel_values=inputs.pixel_values.to(device),
            image_grid_thw=inputs.image_grid_thw.to(device),
            use_cache=True,
            return_dict=True,
        )
    return outputs.logits[:, -1]


def _model_rows(
    entry: CacheEntry,
    source: dict[str, Any],
    payload: dict[str, Any],
    artifacts: Any,
    model: Any,
    processor: Any,
    *,
    device: str,
    warmup: int,
    trials: int,
    ttft_warmup: int,
    ttft_trials: int,
) -> list[dict[str, Any]]:
    images = _multi_tile_images(source["resolved_image_path"])
    inputs = _processor_inputs(processor, images, str(source["question"]))
    reference = _join_visual_features(payload["thumbnail"], payload["crops"])
    rows = []
    prefill_includes = {
        "image_preprocess": False,
        "visual_encoder": False,
        "codec_encode": False,
        "codec_decode": False,
        "native_positions": True,
        "language_prefill": True,
        "first_token": True,
    }
    timing, allocated, reserved = _cuda_benchmark(
        lambda: _language_forward(model, inputs, reference, device=device),
        warmup=warmup,
        trials=trials,
    )
    rows.append(
        _row(
            entry,
            scope="model",
            component="prefill_first_logits",
            method="none",
            rate=None,
            timing=timing,
            peak_allocated=allocated,
            peak_reserved=reserved,
            trials=trials,
            includes=prefill_includes,
        )
    )
    def cached_logic_variant(method: str, rate: float) -> Any:
        arguments = (
            payload["thumbnail"],
            payload["crops"],
            payload["query"],
            artifacts,
            rate,
        )
        if method == "base_vq_logic_router":
            return build_dynamic_logic_variant(*arguments)
        if method == "logic_router_fixed_slots":
            return build_fixed_logic_variant(*arguments)
        raise ValueError(f"unsupported latency method: {method}")

    for rate in RETENTION_RATES:
        for method in ("base_vq_logic_router", "logic_router_fixed_slots"):
            variant = cached_logic_variant(method, rate)
            cached = _join_visual_features(
                payload["thumbnail"], variant.reconstructed
            )
            timing, allocated, reserved = _cuda_benchmark(
                lambda cached=cached: _language_forward(
                    model, inputs, cached, device=device
                ),
                warmup=warmup,
                trials=trials,
            )
            rows.append(
                _row(
                    entry,
                    scope="model",
                    component="prefill_first_logits",
                    method=method,
                    rate=rate,
                    timing=timing,
                    peak_allocated=allocated,
                    peak_reserved=reserved,
                    trials=trials,
                    includes=prefill_includes,
                )
            )
            combined_includes = dict(prefill_includes)
            combined_includes["codec_encode"] = True
            combined_includes["codec_decode"] = True

            def codec_plus_prefill(
                method: str = method, rate: float = rate
            ) -> torch.Tensor:
                encoded = cached_logic_variant(method, rate)
                visual = _join_visual_features(
                    payload["thumbnail"], encoded.reconstructed
                )
                return _language_forward(model, inputs, visual, device=device)

            timing, allocated, reserved = _cuda_benchmark(
                codec_plus_prefill, warmup=warmup, trials=trials
            )
            rows.append(
                _row(
                    entry,
                    scope="model",
                    component="codec_plus_prefill_first_logits",
                    method=method,
                    rate=rate,
                    timing=timing,
                    peak_allocated=allocated,
                    peak_reserved=reserved,
                    trials=trials,
                    includes=combined_includes,
                )
            )

    ttft_includes = {
        "image_preprocess": True,
        "visual_encoder": True,
        "codec_encode": False,
        "codec_decode": False,
        "native_positions": True,
        "language_prefill": True,
        "first_token": True,
    }

    def end_to_end(method: str, rate: float | None) -> torch.Tensor:
        trial_images = _multi_tile_images(source["resolved_image_path"])
        trial_inputs = _processor_inputs(
            processor, trial_images, str(source["question"])
        )
        pixel = trial_inputs.pixel_values.to(device, dtype=model.visual.dtype)
        grid = trial_inputs.image_grid_thw.to(device)
        with torch.inference_mode():
            visual_features = model.visual(pixel, grid_thw=grid)
        thumbnail, crops, _ = _split_visual_features(
            visual_features, grid, model.visual.spatial_merge_size
        )
        if method == "none":
            cached = visual_features
        else:
            query = _query_embedding(
                model, processor, str(source["question"]), device
            )
            arguments = (thumbnail, crops, query, artifacts, float(rate))
            if method == "base_vq_logic_router":
                variant = build_dynamic_logic_variant(*arguments)
            elif method == "logic_router_fixed_slots":
                variant = build_fixed_logic_variant(*arguments)
            else:
                raise ValueError(f"unsupported end-to-end method: {method}")
            cached = _join_visual_features(thumbnail, variant.reconstructed)
        with _cached_visual_forward(model.visual, cached), torch.inference_mode():
            return model.generate(
                input_ids=trial_inputs.input_ids.to(device),
                attention_mask=trial_inputs.attention_mask.to(device),
                pixel_values=trial_inputs.pixel_values.to(device),
                image_grid_thw=grid,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

    timing, allocated, reserved = _wall_benchmark(
        lambda: end_to_end("none", None), warmup=ttft_warmup, trials=ttft_trials
    )
    rows.append(
        _row(
            entry,
            scope="end_to_end",
            component="ttft",
            method="none",
            rate=None,
            timing=timing,
            peak_allocated=allocated,
            peak_reserved=reserved,
            trials=ttft_trials,
            includes=ttft_includes,
        )
    )
    for rate in RETENTION_RATES:
        for method in ("base_vq_logic_router", "logic_router_fixed_slots"):
            includes = dict(ttft_includes)
            includes["codec_encode"] = True
            includes["codec_decode"] = True
            timing, allocated, reserved = _wall_benchmark(
                lambda method=method, rate=rate: end_to_end(method, rate),
                warmup=ttft_warmup,
                trials=ttft_trials,
            )
            rows.append(
                _row(
                    entry,
                    scope="end_to_end",
                    component="ttft",
                    method=method,
                    rate=rate,
                    timing=timing,
                    peak_allocated=allocated,
                    peak_reserved=reserved,
                    trials=ttft_trials,
                    includes=includes,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--ttft-warmup", type=int, default=1)
    parser.add_argument("--ttft-trials", type=int, default=5)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    if min(args.samples_per_dataset, args.trials, args.ttft_trials) <= 0:
        raise SystemExit("sample/trial counts must be positive")

    cache_dir = args.cache_dir.resolve()
    training_dir = args.training_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_cache_manifest(cache_dir)
    validate_split_contract(
        entries,
        calibration_per_dataset=80,
        evaluation_per_dataset=120,
        oracle_per_dataset_split=16,
    )
    selected = []
    for dataset in ("gqa", "textvqa", "chartqa"):
        candidates = [
            entry
            for entry in entries
            if entry.dataset == dataset and entry.split == "evaluation"
        ]
        selected.extend(candidates[: args.samples_per_dataset])
    artifacts = load_tilelogic_artifacts(training_dir).to(args.device)
    source_records = {
        (str(record["dataset"]), int(record["dataset_index"])): record
        for record in _load_manifest(
            args.manifest.resolve(), args.data_root.resolve(), 200
        )
    }

    model = None
    processor = None
    if not args.skip_model:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

    rows = []
    for entry in selected:
        payload = load_cache_payload(cache_dir, entry, device=args.device)
        rows.extend(
            _component_rows(
                entry,
                payload,
                artifacts,
                warmup=args.warmup,
                trials=args.trials,
            )
        )
        if model is not None and processor is not None:
            rows.extend(
                _model_rows(
                    entry,
                    source_records[entry.key],
                    payload,
                    artifacts,
                    model,
                    processor,
                    device=args.device,
                    warmup=max(1, args.warmup // 2),
                    trials=max(3, args.trials // 5),
                    ttft_warmup=args.ttft_warmup,
                    ttft_trials=args.ttft_trials,
                )
            )
        print(
            json.dumps(
                {
                    "dataset": entry.dataset,
                    "dataset_index": entry.dataset_index,
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    csv_path = output_dir / "latency_samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    environment = {
        "format": LATENCY_FORMAT,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model_dir": args.model_dir if not args.skip_model else None,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "samples": len(selected),
        "component_trials": args.trials,
        "ttft_trials": args.ttft_trials,
        "quality_equivalent_sequence_length": 1280,
        "native_compact_visual_sequence": False,
        "fixed_codec_ttft_expands_to_full_visual_tokens": True,
        "paired_dynamic_fixed_methods": [
            "base_vq_logic_router",
            "logic_router_fixed_slots",
        ],
        "claim_boundary": (
            "End-to-end TTFT includes image preprocessing, the visual encoder, "
            "codec encode/decode, native full-length multimodal positions, "
            "language prefill, and first-token generation. The codec still "
            "expands to 1,280 visual tokens; no native compact-kernel speedup is claimed."
        ),
    }
    (output_dir / "latency_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
