#!/usr/bin/env python3
"""Pair videos by prompt/seed and summarize hybrid residual quality and speed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_paired_videos import compare, read_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-method", default="dense")
    parser.add_argument("--dense-cache-method", default="dense_cache008")
    parser.add_argument("--primary-method", default="hybrid")
    parser.add_argument("--primary-cache-method", default="hybrid_cache008")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    clean = finite(values)
    return statistics.fmean(clean) if clean else math.nan


def geometric_mean(values: list[float]) -> float:
    clean = [value for value in finite(values) if value > 0]
    return math.exp(statistics.fmean(math.log(value) for value in clean)) if clean else math.nan


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generation = read_rows(run_dir / "generation_runs.csv")
    successful = [row for row in generation if row.get("status") == "ok"]
    references: dict[tuple[int, int], dict[str, str]] = {}
    for row in successful:
        if row["method"] == args.reference_method:
            references[(int(row["prompt_index"]), int(row["seed"]))] = row

    pairs: list[dict[str, object]] = []
    for row in successful:
        if row["method"] == args.reference_method:
            continue
        key = (int(row["prompt_index"]), int(row["seed"]))
        reference = references.get(key)
        if reference is None:
            pairs.append(
                {
                    "prompt_index": key[0],
                    "seed": key[1],
                    "method": row["method"],
                    "status": "missing_reference",
                }
            )
            continue
        reference_video, reference_fps = read_video(run_dir / reference["video_file"])
        candidate_video, candidate_fps = read_video(run_dir / row["video_file"])
        result: dict[str, object] = {
            "prompt_index": key[0],
            "seed": key[1],
            "method": row["method"],
            "reference_method": args.reference_method,
            "status": "ok" if candidate_fps == reference_fps else "fps_mismatch",
            "reference_video": reference["video_file"],
            "candidate_video": row["video_file"],
            "reference_seconds": float(reference["seconds_including_text_and_vae"]),
            "candidate_seconds": float(row["seconds_including_text_and_vae"]),
            "paired_speedup": float(reference["seconds_including_text_and_vae"])
            / float(row["seconds_including_text_and_vae"]),
            "cached_model_forward_fraction": float(
                row.get("cached_model_forward_fraction") or 0.0
            ),
            "hybrid_residual_calls": int(row.get("hybrid_residual_calls") or 0),
            "peak_allocated_mib": float(row.get("peak_allocated_mib") or math.nan),
        }
        try:
            result.update(compare(reference_video, candidate_video))
        except Exception as error:
            result.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        pairs.append(result)
        print(
            f"PAIR prompt={key[0]} seed={key[1]} method={row['method']} "
            f"speedup={result['paired_speedup']:.4f} "
            f"ssim={result.get('frame_ssim_mean', math.nan):.6f}",
            flush=True,
        )

    write_rows(args.out_dir / "paired_metrics.csv", pairs)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pairs:
        grouped[str(row["method"])].append(row)
    summaries: list[dict[str, object]] = []
    for method, rows in sorted(grouped.items()):
        ok = [row for row in rows if row.get("status") == "ok"]
        summaries.append(
            {
                "method": method,
                "pairs": len(rows),
                "successful_pairs": len(ok),
                "paired_speedup_geomean": geometric_mean(
                    [float(row["paired_speedup"]) for row in ok]
                ),
                "paired_speedup_min": min(
                    [float(row["paired_speedup"]) for row in ok], default=math.nan
                ),
                "candidate_seconds_mean": mean(
                    [float(row["candidate_seconds"]) for row in ok]
                ),
                "frame_ssim_mean": mean(
                    [float(row["frame_ssim_mean"]) for row in ok]
                ),
                "frame_ssim_min": min(
                    [float(row["frame_ssim_mean"]) for row in ok], default=math.nan
                ),
                "frame_ssim_p05_mean": mean(
                    [float(row["frame_ssim_p05"]) for row in ok]
                ),
                "pixel_psnr_db_mean": mean(
                    [float(row["pixel_psnr_db"]) for row in ok]
                ),
                "cached_model_forward_fraction_mean": mean(
                    [float(row["cached_model_forward_fraction"]) for row in ok]
                ),
                "hybrid_residual_calls_mean": mean(
                    [float(row["hybrid_residual_calls"]) for row in ok]
                ),
                "peak_allocated_mib_mean": mean(
                    [float(row["peak_allocated_mib"]) for row in ok]
                ),
            }
        )
    write_rows(args.out_dir / "method_summary.csv", summaries)
    by_method = {str(row["method"]): row for row in summaries}
    expected_pairs = len(references)

    def complete(method: str) -> bool:
        row = by_method.get(method)
        return bool(row and int(row["successful_pairs"]) == expected_pairs)

    hybrid = by_method.get(args.primary_method, {})
    hybrid_cache = by_method.get(args.primary_cache_method, {})
    dense_cache = by_method.get(args.dense_cache_method, {})
    expected_candidates = (
        args.dense_cache_method,
        "fp8",
        args.primary_method,
        args.primary_cache_method,
    )
    correctness = expected_pairs > 0 and all(
        complete(method) for method in expected_candidates
    )
    hybrid_quality = (
        complete(args.primary_method)
        and float(hybrid.get("frame_ssim_mean", 0.0)) >= 0.98
        and float(hybrid.get("pixel_psnr_db_mean", 0.0)) >= 30.0
    )
    cache_quality = (
        complete(args.primary_cache_method)
        and complete(args.dense_cache_method)
        and float(hybrid_cache.get("frame_ssim_mean", 0.0)) >= 0.85
        and float(hybrid_cache.get("frame_ssim_mean", 0.0))
        >= float(dense_cache.get("frame_ssim_mean", 0.0)) - 0.02
    )
    hybrid_runtime = (
        complete(args.primary_method)
        and float(hybrid.get("paired_speedup_geomean", 0.0)) >= 1.0 / 1.15
    )
    cache_runtime = (
        complete(args.primary_cache_method)
        and complete(args.primary_method)
        and float(hybrid_cache.get("candidate_seconds_mean", math.inf))
        < float(hybrid.get("candidate_seconds_mean", -math.inf))
    )
    acceptance = {
        "reference_pairs": expected_pairs,
        "primary_method": args.primary_method,
        "primary_cache_method": args.primary_cache_method,
        "correctness": correctness,
        "hybrid_quality": hybrid_quality,
        "cache_quality": cache_quality,
        "hybrid_runtime_within_15_percent": hybrid_runtime,
        "cache_faster_than_uncached_hybrid": cache_runtime,
        "promote_to_f81": all(
            (correctness, hybrid_quality, cache_quality, hybrid_runtime, cache_runtime)
        ),
        "warning": "PSNR and SSIM are paired deviations from dense output, not standalone semantic quality",
    }
    (args.out_dir / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {"methods": summaries, "acceptance": acceptance},
            indent=2,
            sort_keys=True,
            default=lambda value: float(value) if isinstance(value, np.floating) else value,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(acceptance, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
