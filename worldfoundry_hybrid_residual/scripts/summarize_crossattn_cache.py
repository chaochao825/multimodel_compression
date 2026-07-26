#!/usr/bin/env python3
"""Pair baseline and exact cross-attention-cache generations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from compare_paired_videos import compare, read_video
from summarize_cfg_parallel import bootstrap_mean_interval, finite_mean, read_rows, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[int, int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_rows(run_dir / "generation_runs.csv"):
        if row.get("status") == "ok":
            key = (int(row["prompt_index"]), int(row["seed"]), int(row["repeat"]))
            grouped[key][row["method"]] = row

    paired: list[dict[str, object]] = []
    for key, methods in sorted(grouped.items()):
        if not {"baseline", "crossattn_kv_cache"}.issubset(methods):
            continue
        baseline = methods["baseline"]
        cached = methods["crossattn_kv_cache"]
        baseline_video, _ = read_video(run_dir / baseline["video_file"])
        cached_video, _ = read_video(run_dir / cached["video_file"])
        video_metrics = compare(baseline_video, cached_video)
        baseline_latent = torch.load(
            run_dir / baseline["latent_file"], map_location="cpu", weights_only=True
        )["latent"].float()
        cached_latent = torch.load(
            run_dir / cached["latent_file"], map_location="cpu", weights_only=True
        )["latent"].float()
        delta = cached_latent - baseline_latent
        baseline_seconds = float(baseline["seconds_including_text_and_vae"])
        cached_seconds = float(cached["seconds_including_text_and_vae"])
        paired.append(
            {
                "prompt_index": key[0],
                "seed": key[1],
                "repeat": key[2],
                "prompt": baseline["prompt"],
                "baseline_order_index": int(baseline["method_order_index"]),
                "cached_order_index": int(cached["method_order_index"]),
                "baseline_seconds": baseline_seconds,
                "cached_seconds": cached_seconds,
                "speedup": baseline_seconds / cached_seconds,
                "baseline_peak_allocated_mib": float(baseline["peak_allocated_mib"]),
                "cached_peak_allocated_mib": float(cached["peak_allocated_mib"]),
                "cache_hits": int(cached["crossattn_cache_hits"]),
                "cache_misses": int(cached["crossattn_cache_misses"]),
                "cache_bytes": int(cached["crossattn_cache_bytes"]),
                "latent_max_abs": float(delta.abs().max()),
                "latent_relative_l2": float(
                    delta.norm() / baseline_latent.norm().clamp_min(1e-30)
                ),
                "latent_exact_fraction": float((delta == 0).float().mean()),
                **video_metrics,
            }
        )
    if not paired:
        raise RuntimeError("no successful baseline/cache pairs found")
    write_rows(out_dir / "crossattn_cache_paired_metrics.csv", paired)
    speedups = [float(row["speedup"]) for row in paired]
    ci_low, ci_high = bootstrap_mean_interval(speedups, 10_000, 20260726)
    exact = all(
        float(row["latent_max_abs"]) == 0.0
        and float(row["latent_relative_l2"]) == 0.0
        and float(row["latent_exact_fraction"]) == 1.0
        for row in paired
    )
    summary = {
        "pairs": len(paired),
        "unique_prompts": len({str(row["prompt"]) for row in paired}),
        "unique_seeds": len({int(row["seed"]) for row in paired}),
        "speedup_mean": finite_mean(speedups),
        "speedup_median": float(np.median(speedups)),
        "speedup_bootstrap_95ci_low": ci_low,
        "speedup_bootstrap_95ci_high": ci_high,
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "cache_bytes_max": max(int(row["cache_bytes"]) for row in paired),
        "peak_memory_delta_mib_mean": finite_mean(
            [
                float(row["cached_peak_allocated_mib"])
                - float(row["baseline_peak_allocated_mib"])
                for row in paired
            ]
        ),
        "cache_hits_min": min(int(row["cache_hits"]) for row in paired),
        "cache_misses_max": max(int(row["cache_misses"]) for row in paired),
        "latent_max_abs_max": max(float(row["latent_max_abs"]) for row in paired),
        "frame_ssim_min": min(float(row["frame_ssim_min"]) for row in paired),
        "latent_bit_exact_all_pairs": exact,
        "exact_gate_passed": exact,
    }
    (out_dir / "crossattn_cache_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.require_exact and not exact:
        raise RuntimeError("cross-attention cache exactness gate failed")


if __name__ == "__main__":
    main()
