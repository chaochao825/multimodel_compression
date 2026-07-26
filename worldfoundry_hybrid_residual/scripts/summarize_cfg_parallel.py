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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
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


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
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
                "prompt_index": key[0],
                "seed": key[1],
                "repeat": key[2],
                "prompt": sequential["prompt"],
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

    if not paired:
        raise RuntimeError("no successful sequential/cfg_parallel pairs found")
    write_rows(out_dir / "cfg_parallel_paired_metrics.csv", paired)
    summary = {
        "pairs": len(paired),
        "speedup_mean": finite_mean([float(row["speedup"]) for row in paired]),
        "speedup_min": min(float(row["speedup"]) for row in paired),
        "speedup_max": max(float(row["speedup"]) for row in paired),
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
        "interpretation": (
            "The implementation is model-exact by construction only if the paired "
            "final-latent metrics confirm numerical equivalence."
        ),
    }
    (out_dir / "cfg_parallel_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
