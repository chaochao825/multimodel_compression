#!/usr/bin/env python3
"""Benchmark aligned base-plus-exception layouts with real Qwen weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from tilespec_ex.core import RETENTION_RATES


DEFAULT_MODEL = (
    "/home/wangmeiqi/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/"
    "66285546d2b821cf421d4f5eb2576359d3770cd3"
)
ORIGINAL_CROP_TOKENS = 1024
THUMBNAIL_TOKENS = 256
TEXT_TOKENS = 64
EXCEPTION_FRACTION = 0.25


def _block_layout(raster: torch.Tensor) -> torch.Tensor:
    """Convert [B,4,16,16,C] raster tiles to [B,256,4,C] blocks."""

    batch, tiles, height, width, channels = raster.shape
    if (tiles, height, width) != (4, 16, 16):
        raise ValueError(f"unexpected raster shape: {tuple(raster.shape)}")
    return (
        raster.reshape(batch, tiles, 8, 2, 8, 2, channels)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .contiguous()
        .reshape(batch, 256, 4, channels)
    )


def compression_budget(rate: float) -> tuple[int, int, int, int]:
    """Return retained, base, exception-token, and exception-block counts."""

    retained = round(ORIGINAL_CROP_TOKENS * rate)
    exception_tokens = round(retained * EXCEPTION_FRACTION / 4) * 4
    base_tokens = retained - exception_tokens
    if retained % 4 or base_tokens % 4 or exception_tokens <= 0:
        raise ValueError(f"unsupported retention rate: {rate}")
    return retained, base_tokens, exception_tokens, exception_tokens // 4


def _gather_tokens(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        values,
        1,
        indices.unsqueeze(2).expand(-1, -1, values.shape[-1]),
    )


def _gather_blocks(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        values,
        1,
        indices[:, :, None, None].expand(-1, -1, 4, values.shape[-1]),
    )


def _token_risk(flat: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    work = flat.float()
    energy = work.square().sum(dim=2)
    vectors = torch.nn.functional.normalize(work, dim=2)
    normalized_query = torch.nn.functional.normalize(query, dim=0)
    relevance = ((vectors @ normalized_query + 1.0) * 0.5).clamp(0, 1)
    return energy * relevance


def _block_risk(blocks: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    work = blocks.float()
    energy = work.square().sum(dim=(2, 3))
    vectors = torch.nn.functional.normalize(work.mean(dim=2), dim=2)
    normalized_query = torch.nn.functional.normalize(query, dim=0)
    relevance = ((vectors @ normalized_query + 1.0) * 0.5).clamp(0, 1)
    return energy * relevance


def _fixed_per_tile_indices(scores: torch.Tensor, block_count: int) -> torch.Tensor:
    batch, blocks = scores.shape
    if blocks != 256 or block_count % 4:
        raise ValueError("fixed-slot selection requires 256 blocks and a four-tile budget")
    per_tile = block_count // 4
    local = torch.topk(scores.reshape(batch, 4, 64), k=per_tile, dim=2).indices
    offsets = torch.arange(4, device=scores.device).view(1, 4, 1) * 64
    return (local + offsets).reshape(batch, block_count)


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    trials: int,
    inner_repeats: int,
) -> tuple[dict[str, float], tuple[int, ...], int]:
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    expected_shape = tuple(output.shape)
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_repeats):
            output = function()
        end.record()
        end.synchronize()
        if tuple(output.shape) != expected_shape:
            raise RuntimeError("benchmark output shape changed between trials")
        samples.append(start.elapsed_time(end) / inner_repeats)
    peak = max(0, torch.cuda.max_memory_allocated() - baseline_memory)
    array = np.asarray(samples, dtype=np.float64)
    metrics = {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "std_ms": float(array.std()),
        "minimum_ms": float(array.min()),
        "maximum_ms": float(array.max()),
    }
    return metrics, expected_shape, int(peak)


def _compile_or_eager(
    function: Callable[[], torch.Tensor], enabled: bool
) -> tuple[Callable[[], torch.Tensor], str, str | None]:
    if not enabled:
        return function, "eager", None
    try:
        compiled = torch.compile(function, fullgraph=True, mode="reduce-overhead")
        compiled()
        torch.cuda.synchronize()
        return compiled, "torch_compile", None
    except Exception as error:  # pragma: no cover - environment dependent
        return function, "eager_fallback", f"{type(error).__name__}: {error}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("latency rows are empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--inner-repeats", type=int, default=10)
    parser.add_argument("--prefill-warmup", type=int, default=3)
    parser.add_argument("--prefill-trials", type=int, default=20)
    parser.add_argument("--prefill-max-batch", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    from transformers import Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(20260717)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    projection = model.model.layers[0].self_attn.q_proj
    channels = int(projection.in_features)
    projection_out = int(projection.out_features)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows: list[dict[str, Any]] = []
    compile_failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for batch_size in args.batch_sizes:
        raster = torch.randn(
            batch_size,
            4,
            16,
            16,
            channels,
            device=args.device,
            dtype=torch.bfloat16,
        )
        flat = raster.reshape(batch_size, ORIGINAL_CROP_TOKENS, channels)
        blocked = _block_layout(raster)
        thumbnail = torch.randn(
            batch_size,
            THUMBNAIL_TOKENS,
            channels,
            device=args.device,
            dtype=torch.bfloat16,
        )
        text = torch.randn(
            batch_size,
            TEXT_TOKENS,
            channels,
            device=args.device,
            dtype=torch.bfloat16,
        )
        query = torch.randn(channels, device=args.device, dtype=torch.float32)

        def full_q_proj() -> torch.Tensor:
            return projection(flat)

        full_function, full_execution, full_compile_error = _compile_or_eager(
            full_q_proj, not args.no_compile
        )
        if full_compile_error:
            compile_failures.append(
                {
                    "batch_size": str(batch_size),
                    "retention_rate": "1.0",
                    "layout": "full_uncompressed",
                    "component": "full_q_proj",
                    "error": full_compile_error,
                }
            )
        full_metrics, full_shape, full_peak = _measure(
            full_function,
            warmup=args.warmup,
            trials=args.trials,
            inner_repeats=args.inner_repeats,
        )
        full_row: dict[str, Any] = {
            "batch_size": batch_size,
            "retention_rate": 1.0,
            "retained_crop_tokens": ORIGINAL_CROP_TOKENS,
            "layout": "full_uncompressed",
            "component": "full_q_proj",
            "execution": full_execution,
            "output_shape": "x".join(str(item) for item in full_shape),
            "peak_incremental_bytes": full_peak,
        }
        full_row.update(full_metrics)
        rows.append(full_row)
        print(json.dumps(full_row), flush=True)

        for rate in RETENTION_RATES:
            retained, base_tokens, exception_tokens, exception_blocks = (
                compression_budget(rate)
            )
            base_compact = torch.randn(
                batch_size,
                base_tokens,
                channels,
                device=args.device,
                dtype=torch.bfloat16,
            )
            token_indices = torch.topk(
                _token_risk(flat, query), k=exception_tokens, dim=1
            ).indices
            block_scores = _block_risk(blocked, query)
            block_indices = torch.topk(
                block_scores, k=exception_blocks, dim=1
            ).indices
            fixed_indices = _fixed_per_tile_indices(block_scores, exception_blocks)

            def arbitrary_gather() -> torch.Tensor:
                exceptions = _gather_tokens(flat, token_indices)
                return torch.cat((base_compact, exceptions), dim=1)

            def block_gather() -> torch.Tensor:
                exceptions = _gather_blocks(blocked, block_indices).reshape(
                    batch_size, exception_tokens, channels
                )
                return torch.cat((base_compact, exceptions), dim=1)

            def fixed_slot_gather() -> torch.Tensor:
                exceptions = _gather_blocks(blocked, fixed_indices).reshape(
                    batch_size, exception_tokens, channels
                )
                return torch.cat((base_compact, exceptions), dim=1)

            def block_with_layout() -> torch.Tensor:
                converted = _block_layout(raster)
                exceptions = _gather_blocks(converted, block_indices).reshape(
                    batch_size, exception_tokens, channels
                )
                return torch.cat((base_compact, exceptions), dim=1)

            def fixed_with_layout() -> torch.Tensor:
                converted = _block_layout(raster)
                exceptions = _gather_blocks(converted, fixed_indices).reshape(
                    batch_size, exception_tokens, channels
                )
                return torch.cat((base_compact, exceptions), dim=1)

            gather_functions = {
                "arbitrary_token": arbitrary_gather,
                "block_preblocked": block_gather,
                "fixed_slots_preblocked": fixed_slot_gather,
                "block_layout_included": block_with_layout,
                "fixed_slots_layout_included": fixed_with_layout,
            }
            for layout, gather_function in gather_functions.items():
                for component, raw_function in (
                    ("gather_layout", gather_function),
                    (
                        "gather_plus_q_proj",
                        lambda gather_function=gather_function: projection(
                            gather_function()
                        ),
                    ),
                ):
                    function, execution, compile_error = _compile_or_eager(
                        raw_function, not args.no_compile
                    )
                    if compile_error:
                        compile_failures.append(
                            {
                                "batch_size": str(batch_size),
                                "retention_rate": str(rate),
                                "layout": layout,
                                "component": component,
                                "error": compile_error,
                            }
                        )
                    metrics, shape, peak = _measure(
                        function,
                        warmup=args.warmup,
                        trials=args.trials,
                        inner_repeats=args.inner_repeats,
                    )
                    row: dict[str, Any] = {
                        "batch_size": batch_size,
                        "retention_rate": rate,
                        "retained_crop_tokens": retained,
                        "layout": layout,
                        "component": component,
                        "execution": execution,
                        "output_shape": "x".join(str(item) for item in shape),
                        "peak_incremental_bytes": peak,
                    }
                    row.update(metrics)
                    rows.append(row)
                    print(json.dumps(row), flush=True)

            def energy_selector() -> torch.Tensor:
                energy = blocked.float().square().sum(dim=(2, 3))
                return torch.topk(energy, k=exception_blocks, dim=1).indices

            def risk_selector() -> torch.Tensor:
                return torch.topk(
                    _block_risk(blocked, query), k=exception_blocks, dim=1
                ).indices

            def risk_fixed_selector() -> torch.Tensor:
                scores = _block_risk(blocked, query)
                return _fixed_per_tile_indices(scores, exception_blocks)

            def risk_token_selector() -> torch.Tensor:
                return torch.topk(
                    _token_risk(flat, query), k=exception_tokens, dim=1
                ).indices

            for selector_name, raw_function in (
                ("energy_selector", energy_selector),
                ("risk_selector", risk_selector),
                ("risk_fixed_selector", risk_fixed_selector),
                ("risk_token_selector", risk_token_selector),
            ):
                function, execution, compile_error = _compile_or_eager(
                    raw_function, not args.no_compile
                )
                if compile_error:
                    compile_failures.append(
                        {
                            "batch_size": str(batch_size),
                            "retention_rate": str(rate),
                            "layout": selector_name,
                            "component": "score_plus_topk",
                            "error": compile_error,
                        }
                    )
                metrics, shape, peak = _measure(
                    function,
                    warmup=args.warmup,
                    trials=args.trials,
                    inner_repeats=args.inner_repeats,
                )
                row = {
                    "batch_size": batch_size,
                    "retention_rate": rate,
                    "retained_crop_tokens": retained,
                    "layout": selector_name,
                    "component": "score_plus_topk",
                    "execution": execution,
                    "output_shape": "x".join(str(item) for item in shape),
                    "peak_incremental_bytes": peak,
                }
                row.update(metrics)
                rows.append(row)
                print(json.dumps(row), flush=True)

            if batch_size <= args.prefill_max_batch:
                sequence_tokens = THUMBNAIL_TOKENS + retained + TEXT_TOKENS
                attention_mask = torch.ones(
                    batch_size,
                    sequence_tokens,
                    device=args.device,
                    dtype=torch.long,
                )
                position_ids = (
                    torch.arange(sequence_tokens, device=args.device, dtype=torch.long)
                    .view(1, 1, sequence_tokens)
                    .expand(3, batch_size, sequence_tokens)
                )

                def arbitrary_pack() -> torch.Tensor:
                    indices = torch.topk(
                        _token_risk(flat, query), k=exception_tokens, dim=1
                    ).indices
                    return torch.cat(
                        (base_compact, _gather_tokens(flat, indices)), dim=1
                    )

                def block_pack() -> torch.Tensor:
                    converted = _block_layout(raster)
                    indices = torch.topk(
                        _block_risk(converted, query), k=exception_blocks, dim=1
                    ).indices
                    exceptions = _gather_blocks(converted, indices).reshape(
                        batch_size, exception_tokens, channels
                    )
                    return torch.cat((base_compact, exceptions), dim=1)

                def fixed_pack() -> torch.Tensor:
                    converted = _block_layout(raster)
                    scores = _block_risk(converted, query)
                    indices = _fixed_per_tile_indices(scores, exception_blocks)
                    exceptions = _gather_blocks(converted, indices).reshape(
                        batch_size, exception_tokens, channels
                    )
                    return torch.cat((base_compact, exceptions), dim=1)

                def prefill(pack: Callable[[], torch.Tensor]) -> torch.Tensor:
                    sequence = torch.cat((thumbnail, pack(), text), dim=1)
                    hidden = model.model(
                        inputs_embeds=sequence,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state
                    return model.lm_head(hidden[:, -1])

                for layout, pack in (
                    ("arbitrary_token", arbitrary_pack),
                    ("block_layout_included", block_pack),
                    ("fixed_slots_layout_included", fixed_pack),
                ):
                    function = lambda pack=pack: prefill(pack)
                    metrics, shape, peak = _measure(
                        function,
                        warmup=args.prefill_warmup,
                        trials=args.prefill_trials,
                        inner_repeats=1,
                    )
                    row = {
                        "batch_size": batch_size,
                        "retention_rate": rate,
                        "retained_crop_tokens": retained,
                        "layout": layout,
                        "component": "compact_prefill_plus_logits",
                        "execution": "eager",
                        "output_shape": "x".join(str(item) for item in shape),
                        "peak_incremental_bytes": peak,
                    }
                    row.update(metrics)
                    rows.append(row)
                    print(json.dumps(row), flush=True)

    _write_csv(output_dir / "latency_samples.csv", rows)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model_dir": args.model_dir,
        "projection": "model.model.layers[0].self_attn.q_proj",
        "projection_shape": [projection_out, channels],
        "dtype": "bfloat16",
        "batch_sizes": args.batch_sizes,
        "retention_rates": list(RETENTION_RATES),
        "warmup": args.warmup,
        "trials": args.trials,
        "inner_repeats": args.inner_repeats,
        "prefill_warmup": args.prefill_warmup,
        "prefill_trials": args.prefill_trials,
        "prefill_max_batch": args.prefill_max_batch,
        "thumbnail_tokens": THUMBNAIL_TOKENS,
        "text_tokens": TEXT_TOKENS,
        "exception_fraction": EXCEPTION_FRACTION,
        "budget_contract": [
            {
                "retention_rate": rate,
                "retained_crop_tokens": compression_budget(rate)[0],
                "base_tokens": compression_budget(rate)[1],
                "exception_tokens": compression_budget(rate)[2],
                "exception_blocks": compression_budget(rate)[3],
            }
            for rate in RETENTION_RATES
        ],
        "selector_operates_on_exception_budget_only": True,
        "prefill_sequence_includes_thumbnail_and_text": True,
        "structured_gate_validated": False,
        "compile_requested": not args.no_compile,
        "compile_failures": compile_failures,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Aligned GPU diagnostics with exact 75% base plus 25% exception "
            "budgets, a real Qwen q_proj weight, and compact decoder prefill "
            "plus first-token logits. The visual encoder, native multimodal "
            "position construction, and end-to-end TTFT are not measured."
        ),
    }
    (output_dir / "latency_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
