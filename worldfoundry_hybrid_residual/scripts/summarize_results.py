#!/usr/bin/env python3
"""Aggregate probe and existing World Foundry evidence into one JSON file."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values) -> float:
    return float(statistics.mean(values))


def decomposition_summary(probe: Path) -> dict:
    rows = read_csv(probe / "decomposition_wan9_v1" / "decomposition_metrics.csv")
    by_key = defaultdict(dict)
    for row in rows:
        by_key[row["weight_key"]][row["method"]] = row
    method_means = {}
    for method in (
        "ternary_no_hadamard",
        "ternary_hadamard",
        "robuq_weight_svd_r8",
        "robuq_weight_svd_r16",
        "robuq_weight_svd_r32",
        "ternary_qer_svd_r16",
        "fp8_hadamard",
        "fp8_qer_svd_r16",
    ):
        selected = [methods[method] for methods in by_key.values() if method in methods]
        method_means[method] = {
            "count": len(selected),
            "activation_relative_l2_mean": mean(float(row["activation_relative_l2"]) for row in selected),
            "relative_fro_error_mean": mean(float(row["relative_fro_error"]) for row in selected),
        }

    hybrid_minus_qer = []
    hybrid_minus_robuq = []
    hybrid_wins = 0
    for methods in by_key.values():
        candidates = [
            row
            for row in methods.values()
            if row["main_kind"] == "ternary"
            and 0.90 <= float(row["budget_ratio_vs_lr16"]) <= 1.01
            and "budget_lr16" in row["method"]
        ]
        best = min(candidates, key=lambda row: float(row["activation_relative_l2"]))
        best_error = float(best["activation_relative_l2"])
        qer_error = float(methods["ternary_qer_svd_r16"]["activation_relative_l2"])
        robuq_error = float(methods["robuq_weight_svd_r16"]["activation_relative_l2"])
        hybrid_minus_qer.append(best_error - qer_error)
        hybrid_minus_robuq.append(best_error - robuq_error)
        hybrid_wins += int(best_error < qer_error)

    structure = read_csv(probe / "decomposition_wan9_v1" / "structure_stats.csv")
    bcm = {}
    for block_size in (64, 128, 256):
        selected = [row for row in structure if row.get("bcm_energy_ratio") and int(row["block_size"]) == block_size]
        bcm[str(block_size)] = {
            "capture_mean": mean(float(row["bcm_energy_ratio"]) for row in selected),
            "increment_after_lowrank_mean": mean(float(row["bcm_after_lowrank_energy_ratio"]) for row in selected),
        }
    sparse = {}
    for kind in ("row_sparse", "tile_sparse"):
        selected = [row for row in structure if row.get("structure") == kind]
        sparse[kind] = mean(float(row["captured_residual_energy_ratio"]) for row in selected)
    return {
        "weight_count": len(by_key),
        "method_means": method_means,
        "best_matched_hybrid_minus_fixed_error_rank16_mean": mean(hybrid_minus_qer),
        "best_matched_hybrid_minus_robuq_rank16_mean": mean(hybrid_minus_robuq),
        "best_matched_hybrid_wins_vs_fixed_error_rank16": hybrid_wins,
        "bcm_energy": bcm,
        "sparse_energy_after_rank8": sparse,
    }


def alternating_summary(probe: Path) -> dict:
    rows = read_csv(probe / "alternating_wan9_v1" / "alternating_metrics.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["weight_key"], row["method"])].append(row)
    keys = sorted({key for key, _ in grouped})
    pure_gain = []
    differences = defaultdict(list)
    wins = defaultdict(int)
    for key in keys:
        methods = {method: values for (row_key, method), values in grouped.items() if row_key == key}
        pure_name = next(method for method in methods if method == "alternating_svd_r16")
        initial = float(min(methods[pure_name], key=lambda row: int(row["iteration"]))["activation_relative_l2"])
        pure_best = min(float(row["activation_relative_l2"]) for row in methods[pure_name])
        pure_gain.append(initial - pure_best)
        for family, needle in (("bcm", "alternating_bcm"), ("row_sparse", "rowsparse"), ("tile_sparse", "tilesparse")):
            method = next(name for name in methods if needle in name)
            candidate = min(float(row["activation_relative_l2"]) for row in methods[method])
            differences[family].append(candidate - pure_best)
            wins[family] += int(candidate < pure_best)
    return {
        "weight_count": len(keys),
        "pure_rank16_mean_improvement_after_three_updates": mean(pure_gain),
        "candidate_minus_pure_rank16_mean": {key: mean(value) for key, value in differences.items()},
        "candidate_wins": dict(wins),
    }


def benchmark_summary(probe: Path) -> dict:
    sources = {
        "q_projection": probe / "h200_q_f17_f81_v1" / "h200_benchmark.csv",
        "ffn_up": probe / "h200_ffn_up_v1" / "h200_benchmark.csv",
        "ffn_down": probe / "h200_ffn_down_v1" / "h200_benchmark.csv",
    }
    result = {}
    for name, path in sources.items():
        rows = read_csv(path)
        selected = [row for row in rows if int(row["rows"]) == 7800 and row.get("status") == "ok"]
        result[name] = {
            row["method"]: {
                "latency_ms": float(row["latency_ms_median"]),
                "speedup_vs_bf16": float(row["speedup_vs_bf16"]),
                "output_relative_l2": float(row["output_relative_l2"]),
                "scope": row["measurement_scope"],
            }
            for row in selected
        }
    q_rows = read_csv(sources["q_projection"])
    result["q_projection_f81"] = {
        row["method"]: {
            "latency_ms": float(row["latency_ms_median"]),
            "speedup_vs_bf16": float(row["speedup_vs_bf16"]),
            "output_relative_l2": float(row["output_relative_l2"]),
            "scope": row["measurement_scope"],
        }
        for row in q_rows
        if int(row["rows"]) == 32760 and row.get("status") == "ok"
    }
    sparse = read_csv(probe / "h200_sparse24_v1" / "sparse24_benchmark.csv")
    result["sparse24"] = {
        row["rows"]: {
            "latency_ms": float(row["latency_ms_median"]),
            "speedup_vs_dense": float(row["speedup_vs_dense"]),
            "value_budget_ratio_vs_lr16": float(row["value_budget_ratio_vs_lr16"]),
        }
        for row in sparse
        if row["method"] == "cusparselt_2to4"
    }
    return result


def worldfoundry_summary(root: Path) -> dict:
    attention = read_csv(root / "attention_real_qkv_h200_v1" / "attention_benchmark.csv")
    attention_methods = {
        row["method"]: {
            "latency_ms": float(row["milliseconds"]),
            "output_relative_l2": float(row["output_rel_l2"]),
        }
        for row in attention
        if row["method"] in {"torch_sdpa", "fa3_bf16", "fa3_fp8", "sage_sm90_no_smooth"}
    }
    generation = read_csv(root / "wan_generation_f81_hybrid_20step_v1" / "generation_runs.csv")
    generation_seconds = {
        row["method"]: float(row["seconds_including_text_and_vae"])
        for row in generation
        if row.get("status") == "ok"
    }
    pilot = []
    for part in ("pilot8_teacache_f81_part_a_v1", "pilot8_teacache_f81_part_b_v1"):
        pilot.extend(read_csv(root / part / "data" / "generation_runs.csv"))
    baseline = [float(row["seconds_including_text_and_vae"]) for row in pilot if row["method"] == "fa3_bf16"]
    candidate = [float(row["seconds_including_text_and_vae"]) for row in pilot if "teacache" in row["method"]]
    quality = read_csv(root / "pilot8_teacache_f81_quality_v1" / "paired_video_summary.csv")[0]
    return {
        "attention_f17": attention_methods,
        "generation_f81_20step_seconds": generation_seconds,
        "teacache_f81_eight_prompt": {
            "speedup": mean(baseline) / mean(candidate),
            "cached_model_forward_fraction": float(quality["cached_model_forward_fraction_mean"]),
            "frame_ssim_mean": float(quality["frame_ssim_mean_mean"]),
            "pixel_psnr_db_mean": float(quality["pixel_psnr_db_mean"]),
        },
    }


def nfe_summary(probe: Path, worldfoundry: Path) -> dict:
    baseline = next(
        row
        for row in read_csv(
            worldfoundry / "wan_generation_paired_20step_v1" / "generation_runs.csv"
        )
        if row["method"] == "fa3_bf16"
    )
    baseline_seconds = float(baseline["seconds_including_text_and_vae"])
    unipc_metrics = {
        int(row["video"].removeprefix("step").removesuffix(".mp4")): row
        for row in read_csv(
            probe / "nfe_sweep_f17_quality_v1" / "metrics" / "paired_video_metrics.csv"
        )
        if row["video"].startswith("step")
    }
    unipc = {
        "20": {
            "seconds": baseline_seconds,
            "speedup": 1.0,
            "frame_ssim": 1.0,
            "pixel_psnr_db": None,
        }
    }
    for steps in (12, 8, 6, 4):
        run = read_csv(probe / f"nfe_sweep_f17_step{steps}_v1" / "generation_runs.csv")[0]
        metric = unipc_metrics[steps]
        seconds = float(run["seconds_including_text_and_vae"])
        unipc[str(steps)] = {
            "seconds": seconds,
            "speedup": baseline_seconds / seconds,
            "frame_ssim": float(metric["frame_ssim_mean"]),
            "pixel_psnr_db": float(metric["pixel_psnr_db"]),
        }
    dpm_metrics = {
        int(row["video"].removeprefix("dpm").removesuffix(".mp4")): row
        for row in read_csv(
            probe / "nfe_sweep_f17_dpm_quality_v1" / "metrics" / "paired_video_metrics.csv"
        )
        if row["video"].startswith("dpm")
    }
    dpm = {}
    for steps in (12, 8):
        run = read_csv(probe / f"nfe_sweep_f17_dpm_step{steps}_v1" / "generation_runs.csv")[0]
        metric = dpm_metrics[steps]
        seconds = float(run["seconds_including_text_and_vae"])
        dpm[str(steps)] = {
            "seconds": seconds,
            "speedup": baseline_seconds / seconds,
            "frame_ssim": float(metric["frame_ssim_mean"]),
            "pixel_psnr_db": float(metric["pixel_psnr_db"]),
        }
    return {"reference": "20-step UniPC FA3 BF16", "unipc": unipc, "dpm++": dpm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-results-root", required=True)
    parser.add_argument("--worldfoundry-results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    probe = Path(args.probe_results_root)
    worldfoundry = Path(args.worldfoundry_results_root)
    summary = {
        "decomposition": decomposition_summary(probe),
        "alternating": alternating_summary(probe),
        "h200_probe": benchmark_summary(probe),
        "worldfoundry_existing": worldfoundry_summary(worldfoundry),
        "nfe_sweep": nfe_summary(probe, worldfoundry),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
