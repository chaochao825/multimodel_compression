#!/usr/bin/env python3
"""Visualize World Foundry bottlenecks and residual-branch H200 results."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )


def tea_cache_points(root: Path) -> list[dict]:
    baseline_rows = read_csv(root / "teacache_f17_screen_v1" / "data" / "generation_runs.csv")
    baseline = next(float(row["seconds_including_text_and_vae"]) for row in baseline_rows if row["method"] == "fa3_bf16")
    points = []
    for version in ("teacache_f17_screen_v2", "teacache_f17_screen_v3"):
        run_path = root / version / "data" / "generation_runs.csv"
        metric_path = root / version / "paired_metrics" / "paired_video_metrics.csv"
        if not run_path.exists() or not metric_path.exists():
            continue
        timings = read_csv(run_path)
        metrics = read_csv(metric_path)
        by_threshold = {}
        for row in metrics:
            match = re.search(r"teacache_(\d+)", row.get("video", ""))
            if match:
                by_threshold[int(match.group(1)) / 100.0] = float(row["frame_ssim_mean"])
        for row in timings:
            threshold = row.get("teacache_threshold")
            if not threshold:
                continue
            threshold_value = float(threshold)
            if threshold_value not in by_threshold:
                continue
            points.append(
                {
                    "scope": "single_prompt_screen",
                    "threshold": threshold_value,
                    "speedup": baseline / float(row["seconds_including_text_and_vae"]),
                    "ssim": by_threshold[threshold_value],
                    "cached_fraction": float(row["cached_model_forward_fraction"]),
                }
            )
    # The eight-prompt F81 point is the robust conservative setting.
    parts = []
    for part in ("pilot8_teacache_f81_part_a_v1", "pilot8_teacache_f81_part_b_v1"):
        path = root / part / "data" / "generation_runs.csv"
        if path.exists():
            parts.extend(read_csv(path))
    summary_path = root / "pilot8_teacache_f81_quality_v1" / "paired_video_summary.csv"
    if parts and summary_path.exists():
        baseline_times = [float(row["seconds_including_text_and_vae"]) for row in parts if row["method"] == "fa3_bf16"]
        candidate_times = [float(row["seconds_including_text_and_vae"]) for row in parts if "teacache" in row["method"]]
        summary = read_csv(summary_path)[0]
        points.append(
            {
                "scope": "eight_prompt_f81",
                "threshold": 0.08,
                "speedup": (sum(baseline_times) / len(baseline_times)) / (sum(candidate_times) / len(candidate_times)),
                "ssim": float(summary["frame_ssim_mean_mean"]),
                "cached_fraction": float(summary["cached_model_forward_fraction_mean"]),
            }
        )
    dedup = {}
    for point in points:
        dedup[(point["scope"], point["threshold"])] = point
    return list(dedup.values())


def plot_worldfoundry(root: Path, output: Path) -> None:
    linear = read_csv(root / "worldfoundry_linear_wan_h200_v1" / "worldfoundry_linear_benchmark.csv")
    attention = read_csv(root / "attention_real_qkv_h200_v1" / "attention_benchmark.csv")
    generation = read_csv(root / "wan_generation_f81_hybrid_20step_v1" / "generation_runs.csv")
    tea = tea_cache_points(root)
    figure_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.2))
    axis = axes[0, 0]
    fp8 = [row for row in linear if row["method"] == "worldfoundry_fp8"]
    labels = [row["shape"].replace("_", " ") for row in fp8]
    values = [float(row["speedup_vs_bf16"]) for row in fp8]
    axis.barh(range(len(labels)), values, color="#00798C")
    axis.set_yticks(range(len(labels)), labels)
    axis.axvline(1.0, color="#D1495B", linestyle="--")
    axis.set_xlabel("Speedup vs BF16")
    axis.set_title("A. Dynamic FP8 linear: quantization overhead dominates")
    for row in fp8:
        figure_rows.append({"panel": "linear", **row})

    axis = axes[0, 1]
    wanted = ["torch_sdpa", "fa3_bf16", "fa3_fp8", "sage_sm90_no_smooth"]
    selected = [next(row for row in attention if row["method"] == method) for method in wanted]
    bars = axis.bar(
        range(len(selected)),
        [float(row["milliseconds"]) for row in selected],
        color=["#667085", "#30638E", "#D1495B", "#6A994E"],
    )
    axis.set_xticks(range(len(selected)), [row["method"].replace("_", "\n") for row in selected], fontsize=8)
    axis.set_ylabel("Latency (ms)")
    axis.set_title("B. Real F17 attention: kernel choice matters")
    for bar, row in zip(bars, selected):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"err {float(row['output_rel_l2']):.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        figure_rows.append({"panel": "attention", **row})

    axis = axes[1, 0]
    generation = [row for row in generation if row.get("status") == "ok"]
    axis.barh(
        range(len(generation)),
        [float(row["seconds_including_text_and_vae"]) for row in generation],
        color="#30638E",
    )
    axis.set_yticks(range(len(generation)), [row["method"].replace("fa3_", "") for row in generation], fontsize=8)
    axis.set_xlabel("End-to-end seconds, lower is better")
    axis.set_title("C. F81 / 20-step generation on H200")
    for row in generation:
        figure_rows.append({"panel": "generation", **row})

    axis = axes[1, 1]
    screen = [point for point in tea if point["scope"] == "single_prompt_screen"]
    robust = [point for point in tea if point["scope"] == "eight_prompt_f81"]
    axis.plot(
        [point["speedup"] for point in sorted(screen, key=lambda p: p["threshold"])],
        [point["ssim"] for point in sorted(screen, key=lambda p: p["threshold"])],
        marker="o",
        color="#D1495B",
        label="single-prompt threshold screen",
    )
    for point in screen:
        axis.annotate(f"t={point['threshold']:.2f}", (point["speedup"], point["ssim"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    if robust:
        point = robust[0]
        axis.scatter(point["speedup"], point["ssim"], marker="*", s=180, color="#6A994E", label="8-prompt F81")
    axis.axvline(1.0, color="#667085", linewidth=1)
    axis.set_xlabel("End-to-end speedup")
    axis.set_ylabel("Frame SSIM vs BF16")
    axis.set_title("D. TeaCache: aggressive skipping is quality-limited")
    axis.legend(frameon=False, fontsize=8)
    for point in tea:
        figure_rows.append({"panel": "teacache", **point})

    fig.suptitle("World Foundry / Wan H200 bottleneck map", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output / "worldfoundry_bottleneck_map.png", bbox_inches="tight")
    plt.close(fig)
    write_csv(output / "worldfoundry_bottleneck_map_data.csv", figure_rows)


def canonical_method(method: str) -> str | None:
    if method == "bf16_dense":
        return "BF16"
    if method == "fp8_dynamic_main":
        return "FP8 dyn"
    if method == "fp8_static_input_main":
        return "FP8 static LB"
    if "dynamic_svd_r16" in method:
        return "FP8 dyn + LR"
    if "static_input_svd_r16" in method:
        return "FP8 static + LR"
    if "dynamic_bcm" in method:
        return "FP8 dyn + BCM/LR"
    if "static_input_bcm" in method:
        return "FP8 static + BCM/LR"
    if "dynamic_svd_r8_rowsparse" in method:
        return "FP8 dyn + LR/sparse"
    if "static_input_svd_r8_rowsparse" in method:
        return "FP8 static + LR/sparse"
    return None


def plot_probe_shapes(probe_root: Path, output: Path) -> None:
    sources = {
        "Q projection": probe_root / "h200_q_f17_f81_v1" / "h200_benchmark.csv",
        "FFN up": probe_root / "h200_ffn_up_v1" / "h200_benchmark.csv",
        "FFN down": probe_root / "h200_ffn_down_v1" / "h200_benchmark.csv",
    }
    methods = [
        "BF16",
        "FP8 dyn",
        "FP8 static LB",
        "FP8 dyn + LR",
        "FP8 static + LR",
        "FP8 dyn + BCM/LR",
        "FP8 static + BCM/LR",
        "FP8 dyn + LR/sparse",
        "FP8 static + LR/sparse",
    ]
    matrix = np.full((len(sources), len(methods)), np.nan)
    data_rows = []
    for row_index, (shape, path) in enumerate(sources.items()):
        rows = [row for row in read_csv(path) if int(row["rows"]) == 7800 and row.get("status") == "ok"]
        for row in rows:
            method = canonical_method(row["method"])
            if method is None:
                continue
            matrix[row_index, methods.index(method)] = float(row["speedup_vs_bf16"])
            data_rows.append({"shape_family": shape, "canonical_method": method, **row})
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=max(1.8, float(np.nanmax(matrix))), aspect="auto")
    ax.set_yticks(range(len(sources)), list(sources.keys()))
    ax.set_xticks(range(len(methods)), methods, rotation=32, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}x", ha="center", va="center", fontsize=8)
    ax.set_title("Measured H200 speedup at M=7800 (static input entries are lower bounds)")
    fig.colorbar(image, ax=ax, label="Speedup vs BF16")
    fig.tight_layout()
    fig.savefig(output / "h200_residual_shape_speedup.png")
    plt.close(fig)
    write_csv(output / "h200_residual_shape_speedup_data.csv", data_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worldfoundry-results-root", required=True)
    parser.add_argument("--probe-results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    setup()
    plot_worldfoundry(Path(args.worldfoundry_results_root), output)
    plot_probe_shapes(Path(args.probe_results_root), output)
    print(f"wrote system figures to {output}")


if __name__ == "__main__":
    main()
