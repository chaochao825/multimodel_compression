#!/usr/bin/env python3
"""Screen exact temporal scheduling for the Wan VAE on frozen rCM latents."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import run_wan_rcm_baseline as baseline
import run_wan_rcm_exact_runtime as exact_runtime
from experiment_artifacts import JsonlEventLog, atomic_write_json, require_fresh_output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("f17-screen", "f81-component")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--f17-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-053" or config["gate_id"] != "G-032":
        raise ValueError("config is not the proposed EXP-053/G-032 configuration")
    if tuple(config["methods"]) != ("rcm4",):
        raise ValueError("EXP-053 permits only the exact resident rCM4 incumbent")
    chunk_sizes = tuple(int(value) for value in config["screen_chunk_sizes"])
    if chunk_sizes != (1, 2, 4, 8):
        raise ValueError("EXP-053 requires the frozen chunk grid 1/2/4/8")
    if tuple(config["timing_prompt_indices"]) != (0, 1, 2, 3):
        raise ValueError("EXP-053 requires all four frozen EXP-047 prompts")

    base_path = (PROJECT_ROOT / config["base_config"]).resolve()
    base_config = baseline.load_config(base_path)
    if config["timing_seed"] not in base_config["generation"]["formal_seeds"]:
        raise ValueError("EXP-053 timing seed must remain inside the EXP-047 plan")
    return config, base_config


def temporal_decode_ranges(
    num_latent_frames: int, chunk_size: int
) -> tuple[tuple[int, int], ...]:
    """Keep the sentinel first frame separate, then emit regular causal chunks."""
    if num_latent_frames < 1:
        raise ValueError("Wan VAE decode requires at least one latent frame")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    ranges = [(0, 1)]
    ranges.extend(
        (start, min(start + chunk_size, num_latent_frames))
        for start in range(1, num_latent_frames, chunk_size)
    )
    return tuple(ranges)


def decode_inner_temporal_chunks(
    model: Any,
    z: Any,
    scale: Any,
    chunk_size: int,
    torch: Any,
) -> Any:
    """Run the unchanged decoder with fewer causal temporal dispatch groups."""
    model.clear_cache()
    try:
        if isinstance(scale[0], torch.Tensor):
            z = (
                z / scale[1].view(1, model.z_dim, 1, 1, 1)
                + scale[0].view(1, model.z_dim, 1, 1, 1)
            )
        else:
            z = z / scale[1] + scale[0]
        x = model.conv2(z)
        outputs = []
        for start, end in temporal_decode_ranges(int(z.shape[2]), chunk_size):
            model._conv_idx = [0]
            outputs.append(
                model.decoder(
                    x[:, :, start:end, :, :],
                    feat_cache=model._feat_map,
                    feat_idx=model._conv_idx,
                )
            )
        return torch.cat(outputs, dim=2)
    finally:
        model.clear_cache()


def decode_temporal_chunks(
    tokenizer: Any, latent: Any, chunk_size: int, torch: Any
) -> Any:
    """Match WanVAE.decode dtype/autocast semantics around the candidate schedule."""
    outer = tokenizer.model
    in_dtype = latent.dtype
    with outer.context:
        if not outer.is_amp:
            latent = latent.to(outer.dtype)
        video = decode_inner_temporal_chunks(
            outer.model,
            latent,
            outer.scale,
            chunk_size,
            torch,
        )
    return video.to(in_dtype)


def tensor_comparison(torch: Any, reference: Any, candidate: Any) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"decode shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    delta = candidate.float() - reference.float()
    reference_norm = float(torch.linalg.vector_norm(reference.float()).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": delta_norm / max(reference_norm, 1e-12),
    }


def timed_decode(
    torch: Any, device: Any, decode: Callable[[], Any]
) -> tuple[Any, float, float]:
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = decode()
    end.record()
    end.synchronize()
    seconds = float(start.elapsed_time(end)) / 1000.0
    peak_reserved_mib = float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
    return output, seconds, peak_reserved_mib


def summarize_seconds(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def benchmark_latent(
    tokenizer: Any,
    latent: Any,
    chunk_sizes: tuple[int, ...],
    repeats: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("decode repeats must be positive")

    reference = tokenizer.decode(latent)
    baseline.synchronize(torch, device)
    exactness: dict[int, dict[str, Any]] = {}
    for chunk_size in chunk_sizes:
        candidate = decode_temporal_chunks(tokenizer, latent, chunk_size, torch)
        baseline.synchronize(torch, device)
        exactness[chunk_size] = tensor_comparison(torch, reference, candidate)
        del candidate

    # Warm both dispatch paths before collecting CUDA-event samples.
    tokenizer.decode(latent)
    for chunk_size in chunk_sizes:
        decode_temporal_chunks(tokenizer, latent, chunk_size, torch)
    baseline.synchronize(torch, device)

    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: dict[int, list[dict[str, Any]]] = {
        chunk_size: [] for chunk_size in chunk_sizes
    }
    for repeat in range(repeats):
        output, seconds, peak = timed_decode(
            torch, device, lambda: tokenizer.decode(latent)
        )
        baseline_rows.append(
            {
                "repeat": repeat,
                "seconds": seconds,
                "peak_reserved_mib": peak,
            }
        )
        del output
        for chunk_size in chunk_sizes:
            output, seconds, peak = timed_decode(
                torch,
                device,
                lambda chunk_size=chunk_size: decode_temporal_chunks(
                    tokenizer, latent, chunk_size, torch
                ),
            )
            candidate_rows[chunk_size].append(
                {
                    "repeat": repeat,
                    "seconds": seconds,
                    "peak_reserved_mib": peak,
                }
            )
            del output

    baseline_median = summarize_seconds(baseline_rows, "seconds")
    candidates = []
    for chunk_size in chunk_sizes:
        candidate_median = summarize_seconds(candidate_rows[chunk_size], "seconds")
        candidates.append(
            {
                "chunk_size": chunk_size,
                "exactness": exactness[chunk_size],
                "rows": candidate_rows[chunk_size],
                "median_seconds": candidate_median,
                "speedup": baseline_median / candidate_median,
            }
        )
    exact_candidates = [
        row for row in candidates if row["exactness"]["bitwise_equal"]
    ]
    selected = min(exact_candidates, key=lambda row: row["median_seconds"]) if exact_candidates else None
    del reference
    return {
        "baseline_rows": baseline_rows,
        "baseline_median_seconds": baseline_median,
        "candidates": candidates,
        "selected_chunk_size": (
            int(selected["chunk_size"]) if selected is not None else None
        ),
        "selected_median_seconds": (
            float(selected["median_seconds"]) if selected is not None else None
        ),
        "selected_speedup": float(selected["speedup"]) if selected is not None else None,
    }


def make_latent(
    base_config: dict[str, Any],
    network: Any,
    tokenizer: Any,
    runtime: dict[str, Any],
    device: Any,
    text_policy: exact_runtime.TextEncoderPolicy,
    prompt: str,
    prompt_index: int,
    seed: int,
    num_frames: int,
    output_dir: Path,
) -> tuple[Any, dict[str, int]]:
    spec = exact_runtime.make_spec(
        base_config,
        "rcm4",
        prompt,
        prompt_index,
        seed,
        num_frames,
        output_dir,
    )
    condition, uncondition = text_policy.encode_condition(prompt, need_negative=False)
    if uncondition is not None:
        raise RuntimeError("rcm4 unexpectedly produced a CFG negative condition")
    noise, generator = baseline.make_initial_noise(
        base_config, spec, tokenizer, runtime, device
    )
    samples, calls = baseline.denoise_rcm(
        base_config,
        network,
        noise,
        generator,
        condition,
        runtime,
        device,
    )
    return samples, {"network_forward_calls": calls}


def run_f17_screen(
    config: dict[str, Any],
    base_config: dict[str, Any],
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    checkpoint_path, checkpoint_identity = baseline.verify_checkpoint(
        base_config, "rcm4"
    )
    network, tokenizer, load_info = baseline.load_pipeline(
        base_config, "rcm4", runtime, device
    )
    text_policy = exact_runtime.TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=False,
    )
    try:
        text_policy.warm(baseline.load_formal_prompts(base_config)[0], False)
        torch.manual_seed(int(base_config["generation"]["smoke_seed"]))
        torch.cuda.manual_seed_all(int(base_config["generation"]["smoke_seed"]))
        latent, latent_info = make_latent(
            base_config,
            network,
            tokenizer,
            runtime,
            device,
            text_policy,
            base_config["generation"]["smoke_prompt"],
            -1,
            int(base_config["generation"]["smoke_seed"]),
            17,
            output_dir,
        )
        benchmark = benchmark_latent(
            tokenizer,
            latent,
            tuple(int(value) for value in config["screen_chunk_sizes"]),
            int(config["screen_repeats"]),
            torch,
            device,
        )
    finally:
        text_policy.close()
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": checkpoint_identity,
        "load": load_info,
        "latent": latent_info,
        "benchmark": benchmark,
        "selected_chunk_size": benchmark["selected_chunk_size"],
        "advance": benchmark["selected_chunk_size"] is not None,
    }


def read_selected_chunk(path: Path, config: dict[str, Any]) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["experiment_id"] != "EXP-053" or manifest["gate_id"] != "G-032":
        raise ValueError("F17 selection manifest belongs to another experiment")
    if manifest["stage"] != "f17-screen":
        raise ValueError("F81 confirmation requires an F17 screen manifest")
    chunk_size = manifest["result"]["selected_chunk_size"]
    if chunk_size is None:
        raise ValueError("F17 screen did not find a bitwise-exact candidate")
    if int(chunk_size) not in config["screen_chunk_sizes"]:
        raise ValueError("selected chunk is outside the frozen candidate grid")
    return int(chunk_size)


def run_f81_component(
    config: dict[str, Any],
    base_config: dict[str, Any],
    selected_chunk: int,
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    checkpoint_path, checkpoint_identity = baseline.verify_checkpoint(
        base_config, "rcm4"
    )
    network, tokenizer, load_info = baseline.load_pipeline(
        base_config, "rcm4", runtime, device
    )
    prompts = baseline.load_formal_prompts(base_config)
    text_policy = exact_runtime.TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=False,
    )
    rows = []
    try:
        text_policy.warm(base_config["generation"]["smoke_prompt"], False)
        for prompt_index in config["timing_prompt_indices"]:
            seed = int(config["timing_seed"])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            latent, latent_info = make_latent(
                base_config,
                network,
                tokenizer,
                runtime,
                device,
                text_policy,
                prompts[prompt_index],
                int(prompt_index),
                seed,
                int(base_config["generation"]["num_frames"]),
                output_dir,
            )
            benchmark = benchmark_latent(
                tokenizer,
                latent,
                (selected_chunk,),
                int(config["confirm_repeats"]),
                torch,
                device,
            )
            rows.append(
                {
                    "prompt_index": int(prompt_index),
                    "prompt": prompts[prompt_index],
                    "seed": seed,
                    "latent": latent_info,
                    "benchmark": benchmark,
                }
            )
            atomic_write_json(output_dir / "rows.partial.json", rows)

    finally:
        text_policy.close()

    exact = all(
        row["benchmark"]["candidates"][0]["exactness"]["bitwise_equal"]
        for row in rows
    )
    baseline_samples = [
        sample["seconds"]
        for row in rows
        for sample in row["benchmark"]["baseline_rows"]
    ]
    candidate_samples = [
        sample["seconds"]
        for row in rows
        for sample in row["benchmark"]["candidates"][0]["rows"]
    ]
    baseline_median = float(statistics.median(baseline_samples))
    candidate_median = float(statistics.median(candidate_samples))
    vae_speedup = baseline_median / candidate_median
    projected_seconds = (
        float(config["incumbent_warm_seconds"])
        - float(config["incumbent_vae_seconds"])
        + candidate_median
    )
    projected_speedup = float(config["incumbent_warm_seconds"]) / projected_seconds
    advance = (
        exact
        and vae_speedup >= float(config["min_vae_speedup"])
        and projected_speedup >= float(config["min_projected_warm_speedup"])
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": checkpoint_identity,
        "load": load_info,
        "selected_chunk_size": selected_chunk,
        "rows": rows,
        "summary": {
            "bitwise_exact": exact,
            "baseline_vae_median_seconds": baseline_median,
            "candidate_vae_median_seconds": candidate_median,
            "vae_speedup": vae_speedup,
            "projected_warm_seconds": projected_seconds,
            "projected_incremental_warm_speedup": projected_speedup,
            "min_vae_speedup": float(config["min_vae_speedup"]),
            "min_projected_warm_speedup": float(
                config["min_projected_warm_speedup"]
            ),
        },
        "advance": advance,
    }


def resolve_output_dir(
    config: dict[str, Any], stage: str, override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    return (Path(config["remote_output_root"]) / stage).resolve()


def main() -> None:
    args = parse_args()
    config, base_config = load_configs(args.config.resolve())
    if args.stage == "f17-screen" and args.f17_manifest is not None:
        raise ValueError("F17 screen cannot consume a previous selection manifest")
    if args.stage == "f81-component" and args.f17_manifest is None:
        raise ValueError("F81 component confirmation requires --f17-manifest")

    output_dir = resolve_output_dir(config, args.stage, args.output_dir)
    require_fresh_output_dir(output_dir)
    log = JsonlEventLog(output_dir / "events.jsonl", f"EXP-053-{args.stage}")
    log.emit("run_start", stage=args.stage)
    runtime, device, source_commit = exact_runtime.setup_runtime(
        base_config, args.device
    )
    if args.stage == "f17-screen":
        result = run_f17_screen(config, base_config, output_dir, runtime, device)
    else:
        if args.f17_manifest is None:
            raise RuntimeError("F81 manifest validation was bypassed")
        selected_chunk = read_selected_chunk(args.f17_manifest.resolve(), config)
        result = run_f81_component(
            config,
            base_config,
            selected_chunk,
            output_dir,
            runtime,
            device,
        )

    manifest = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "stage": args.stage,
        "source_commit": source_commit,
        "result": result,
        "environment": exact_runtime.runtime_environment(runtime["torch"], device),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {"status": "complete", "stage": args.stage},
    )
    log.emit("run_complete", stage=args.stage, advance=result["advance"])
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
