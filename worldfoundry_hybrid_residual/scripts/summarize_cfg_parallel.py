#!/usr/bin/env python3
"""Pair Wan sequential and exact CFG-parallel runs and summarize fidelity/speed."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from compare_paired_videos import compare, read_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Generation directory. Repeat to aggregate independent seed runs.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Fail after writing metrics unless every paired final latent is bit exact.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def bootstrap_mean_interval(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1 or samples <= 0:
        value = float(finite[0])
        return value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, finite.size, size=(samples, finite.size))
    means = finite[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def pair_run(run_dir: Path) -> list[dict[str, object]]:
    source_rows = read_rows(run_dir / "generation_runs.csv")
    grouped: dict[tuple[int, int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in source_rows:
        if row.get("status") != "ok" or not row.get("video_file"):
            continue
        key = (int(row["prompt_index"]), int(row["seed"]), int(row["repeat"]))
        grouped[key][row["method"]] = row

    paired: list[dict[str, object]] = []
    for key, methods in sorted(grouped.items()):
        if "sequential" not in methods or "cfg_parallel" not in methods:
            continue
        sequential = methods["sequential"]
        parallel = methods["cfg_parallel"]
        sequential_video, sequential_fps = read_video(run_dir / sequential["video_file"])
        parallel_video, parallel_fps = read_video(run_dir / parallel["video_file"])
        metrics = compare(sequential_video, parallel_video)
        sequential_latent = torch.load(
            run_dir / sequential["latent_file"], map_location="cpu", weights_only=True
        )["latent"].float()
        parallel_latent = torch.load(
            run_dir / parallel["latent_file"], map_location="cpu", weights_only=True
        )["latent"].float()
        if sequential_latent.shape != parallel_latent.shape:
            raise ValueError(
                f"latent shape mismatch: {sequential_latent.shape} versus "
                f"{parallel_latent.shape}"
            )
        latent_delta = parallel_latent - sequential_latent
        latent_reference_norm = float(torch.linalg.vector_norm(sequential_latent).item())
        latent_metrics = {
            "latent_max_abs": float(latent_delta.abs().max().item()),
            "latent_rmse": float(latent_delta.square().mean().sqrt().item()),
            "latent_relative_l2": float(
                torch.linalg.vector_norm(latent_delta).item()
                / max(latent_reference_norm, 1e-30)
            ),
            "latent_exact_fraction": float((latent_delta == 0).float().mean().item()),
        }
        sequential_seconds = float(sequential["seconds_including_text_and_vae"])
        parallel_seconds = float(parallel["seconds_including_text_and_vae"])
        paired.append(
            {
                "source_run": run_dir.name,
                "source_run_dir": str(run_dir),
                "prompt_index": key[0],
                "seed": key[1],
                "repeat": key[2],
                "prompt": sequential["prompt"],
                "sequential_order_index": int(sequential["method_order_index"]),
                "cfg_parallel_order_index": int(parallel["method_order_index"]),
                "sequential_seconds": sequential_seconds,
                "cfg_parallel_seconds": parallel_seconds,
                "speedup": sequential_seconds / parallel_seconds,
                "sequential_fps": sequential_fps,
                "cfg_parallel_fps": parallel_fps,
                "sequential_video": sequential["video_file"],
                "cfg_parallel_video": parallel["video_file"],
                "sequential_latent": sequential["latent_file"],
                "cfg_parallel_latent": parallel["latent_file"],
                **latent_metrics,
                **metrics,
            }
        )
    return paired


def main() -> None:
    args = parse_args()
    run_dirs = [path.resolve() for path in args.run_dir]
    if len(run_dirs) > 1 and args.out_dir is None:
        raise ValueError("--out-dir is required when multiple --run-dir values are used")
    out_dir = (args.out_dir or run_dirs[0]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paired: list[dict[str, object]] = []
    for run_dir in run_dirs:
        paired.extend(pair_run(run_dir))

    if not paired:
        raise RuntimeError("no successful sequential/cfg_parallel pairs found")
    write_rows(out_dir / "cfg_parallel_paired_metrics.csv", paired)
    speedups = [float(row["speedup"]) for row in paired]
    speedup_ci_low, speedup_ci_high = bootstrap_mean_interval(
        speedups, args.bootstrap_samples, args.bootstrap_seed
    )
    sequential_seconds = [float(row["sequential_seconds"]) for row in paired]
    parallel_seconds = [float(row["cfg_parallel_seconds"]) for row in paired]
    latent_exact = all(
        float(row["latent_max_abs"]) == 0.0
        and float(row["latent_relative_l2"]) == 0.0
        and float(row["latent_exact_fraction"]) == 1.0
        for row in paired
    )
    pixel_exact = all(
        float(row["pixel_max_abs"]) == 0.0
        and float(row["exact_pixel_fraction"]) == 1.0
        for row in paired
    )
    summary = {
        "run_directories": [str(path) for path in run_dirs],
        "runs": len(run_dirs),
        "pairs": len(paired),
        "unique_prompts": len({str(row["prompt"]) for row in paired}),
        "unique_seeds": len({int(row["seed"]) for row in paired}),
        "sequential_first_pairs": sum(
            int(row["sequential_order_index"]) == 0 for row in paired
        ),
        "cfg_parallel_first_pairs": sum(
            int(row["cfg_parallel_order_index"]) == 0 for row in paired
        ),
        "sequential_seconds_mean": finite_mean(sequential_seconds),
        "cfg_parallel_seconds_mean": finite_mean(parallel_seconds),
        "aggregate_wall_time_speedup": sum(sequential_seconds) / sum(parallel_seconds),
        "speedup_mean": finite_mean(speedups),
        "speedup_median": float(np.median(speedups)),
        "speedup_bootstrap_95ci_low": speedup_ci_low,
        "speedup_bootstrap_95ci_high": speedup_ci_high,
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "frame_ssim_mean": finite_mean(
            [float(row["frame_ssim_mean"]) for row in paired]
        ),
        "frame_ssim_min": min(float(row["frame_ssim_min"]) for row in paired),
        "pixel_max_abs_max": max(float(row["pixel_max_abs"]) for row in paired),
        "latent_max_abs_max": max(float(row["latent_max_abs"]) for row in paired),
        "latent_relative_l2_max": max(
            float(row["latent_relative_l2"]) for row in paired
        ),
        "latent_exact_fraction_mean": finite_mean(
            [float(row["latent_exact_fraction"]) for row in paired]
        ),
        "exact_pixel_fraction_mean": finite_mean(
            [float(row["exact_pixel_fraction"]) for row in paired]
        ),
        "latent_bit_exact_all_pairs": latent_exact,
        "decoded_pixel_exact_all_pairs": pixel_exact,
        "exact_gate_passed": latent_exact,
        "interpretation": (
            "The implementation is model-exact by construction only if the paired "
            "final-latent metrics confirm numerical equivalence."
        ),
    }
    (out_dir / "cfg_parallel_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.require_exact and not latent_exact:
        raise RuntimeError("strict exactness gate failed for one or more paired latents")


if __name__ == "__main__":
    main()
