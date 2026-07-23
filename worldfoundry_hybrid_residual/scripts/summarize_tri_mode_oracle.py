#!/usr/bin/env python3
"""Pair tri-mode rollouts with dense references and build Oracle diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from compare_paired_videos import compare, read_video


DEFAULT_THRESHOLDS = (0.98, 0.97, 0.95, 0.90)


def route_decision(
    oracle_speed: float, universal_speed: float
) -> tuple[bool, bool, str]:
    stop_high_fidelity = oracle_speed < 1.2
    focus_adaptive = oracle_speed >= 1.4 and universal_speed <= 1.08
    if stop_high_fidelity:
        next_gate = (
            "stop schedule-combination search for the current training-free action "
            "implementations; change the operator set or relax/redefine the quality gate"
        )
    elif focus_adaptive:
        next_gate = "prioritize a sample-adaptive controller and quantify its oracle gap"
    else:
        next_gate = "verify beam-search combinations with real rollouts"
    return stop_high_fidelity, focus_adaptive, next_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--paired-csv", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = [dict(row) for row in rows]
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def as_float(row: Mapping[str, object], field: str) -> float:
    return float(row[field])


def as_float_default(
    row: Mapping[str, object], field: str, default: float = 0.0
) -> float:
    value = row.get(field, default)
    return float(value) if value not in (None, "") else float(default)


def sample_key(row: Mapping[str, object]) -> tuple[int, int, int]:
    return int(row["prompt_index"]), int(row["seed"]), int(row["repeat"])


def sample_name(key: tuple[int, int, int]) -> str:
    prompt, seed, repeat = key
    return f"p{prompt:02d}_seed{seed}_r{repeat}"


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.any(array <= 0):
        return math.nan
    return float(np.exp(np.mean(np.log(array))))


def collect_paired_rows(run_dirs: list[Path]) -> list[dict[str, object]]:
    paired: list[dict[str, object]] = []
    for run_dir in run_dirs:
        generation_path = run_dir / "generation_runs.csv"
        generation = read_csv(generation_path)
        references = {
            sample_key(row): row
            for row in generation
            if row.get("status") == "ok" and row.get("method") == "dense"
        }
        missing = {
            sample_key(row)
            for row in generation
            if row.get("status") == "ok"
        } - set(references)
        if missing:
            raise ValueError(
                f"{generation_path} lacks dense references for {sorted(missing)}"
            )

        reference_cache: dict[tuple[int, int, int], np.ndarray] = {}
        reference_fps: dict[tuple[int, int, int], float] = {}
        for row in generation:
            if row.get("status") != "ok":
                continue
            key = sample_key(row)
            reference_row = references[key]
            reference_path = run_dir / reference_row["video_file"]
            candidate_path = run_dir / row["video_file"]
            if key not in reference_cache:
                reference_cache[key], reference_fps[key] = read_video(reference_path)
            candidate, candidate_fps = read_video(candidate_path)
            metrics = compare(reference_cache[key], candidate)
            dense_seconds = as_float(reference_row, "seconds_including_text_and_vae")
            candidate_seconds = as_float(row, "seconds_including_text_and_vae")
            metadata = json.loads(row.get("schedule_metadata") or "{}")
            paired.append(
                dict(row)
                | {
                    "source_run_dir": str(run_dir.resolve()),
                    "logical_sample": sample_name(key),
                    "dense_video_file": reference_row["video_file"],
                    "dense_seconds": dense_seconds,
                    "candidate_seconds": candidate_seconds,
                    "end_to_end_speedup": dense_seconds / candidate_seconds,
                    "candidate_fps": candidate_fps,
                    "reference_fps": reference_fps[key],
                    "fps_match": candidate_fps == reference_fps[key],
                    "step_start": metadata.get("step_start", ""),
                    "step_end": metadata.get("step_end", ""),
                    "block_start": metadata.get("block_start", ""),
                    "block_end": metadata.get("block_end", ""),
                    "schedule_branches": json.dumps(
                        metadata.get("branches", [0, 1]), sort_keys=True
                    ),
                    "forecast_scale": metadata.get("forecast_scale", 0.0),
                    **metrics,
                }
            )
            print(
                f"PAIR {sample_name(key)} {row['method']:28s} "
                f"speed={dense_seconds / candidate_seconds:.4f} "
                f"SSIM={metrics['frame_ssim_mean']:.6f}",
                flush=True,
            )
    return paired


def aggregate_methods(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary: list[dict[str, object]] = []
    for method, method_rows in sorted(grouped.items()):
        speeds = [as_float(row, "end_to_end_speedup") for row in method_rows]
        ssims = [as_float(row, "frame_ssim_mean") for row in method_rows]
        psnrs = [as_float(row, "pixel_psnr_db") for row in method_rows]
        summary.append(
            {
                "method": method,
                "schedule_family": method_rows[0].get("schedule_family", ""),
                "schedule_action": method_rows[0].get("schedule_action", ""),
                "schedule_cell_count": method_rows[0].get("schedule_cell_count", ""),
                "samples": len(method_rows),
                "speedup_geomean": geometric_mean(speeds),
                "speedup_mean": float(np.mean(speeds)),
                "speedup_min": float(np.min(speeds)),
                "speedup_max": float(np.max(speeds)),
                "ssim_mean": float(np.mean(ssims)),
                "ssim_min": float(np.min(ssims)),
                "ssim_p05": float(np.quantile(ssims, 0.05)),
                "psnr_mean_db": float(np.mean(psnrs)),
                "requested_quant_mean": float(
                    np.mean([as_float(row, "tri_mode_requested_quant") for row in method_rows])
                ),
                "requested_cache_mean": float(
                    np.mean([as_float(row, "tri_mode_requested_cache") for row in method_rows])
                ),
                "executed_cache_mean": float(
                    np.mean([as_float(row, "tri_mode_executed_cache") for row in method_rows])
                ),
                "cache_fallbacks_mean": float(
                    np.mean([as_float(row, "tri_mode_cache_fallbacks") for row in method_rows])
                ),
                "cache_refreshes_mean": float(
                    np.mean(
                        [
                            as_float_default(row, "tri_mode_cache_refreshes")
                            for row in method_rows
                        ]
                    )
                ),
            }
        )
    return summary


def pareto_frontier(method_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    for candidate in method_rows:
        speed = as_float(candidate, "speedup_geomean")
        quality = as_float(candidate, "ssim_min")
        dominated = any(
            as_float(other, "speedup_geomean") >= speed
            and as_float(other, "ssim_min") >= quality
            and (
                as_float(other, "speedup_geomean") > speed
                or as_float(other, "ssim_min") > quality
            )
            for other in method_rows
            if other is not candidate
        )
        if not dominated:
            frontier.append(dict(candidate) | {"pareto": True})
    return sorted(frontier, key=lambda row: as_float(row, "speedup_geomean"))


def aggregate_method_samples(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["logical_sample"]))].append(row)
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (method, sample), values in grouped.items():
        result[method][sample] = {
            "speedup": geometric_mean(as_float(row, "end_to_end_speedup") for row in values),
            "ssim": min(as_float(row, "frame_ssim_mean") for row in values),
        }
    return dict(result)


def oracle_thresholds(
    rows: list[dict[str, object]], thresholds: list[float]
) -> list[dict[str, object]]:
    by_method = aggregate_method_samples(rows)
    samples = sorted({sample for values in by_method.values() for sample in values})
    output: list[dict[str, object]] = []
    for threshold in thresholds:
        selected: dict[str, str] = {}
        selected_speeds: list[float] = []
        for sample in samples:
            feasible = [
                (values[sample]["speedup"], method)
                for method, values in by_method.items()
                if sample in values and values[sample]["ssim"] >= threshold
            ]
            if not feasible:
                continue
            speed, method = max(feasible)
            selected[sample] = method
            selected_speeds.append(speed)

        universal: list[tuple[float, str]] = []
        for method, values in by_method.items():
            if all(
                sample in values and values[sample]["ssim"] >= threshold
                for sample in samples
            ):
                universal.append(
                    (geometric_mean(values[sample]["speedup"] for sample in samples), method)
                )
        universal_speed, universal_method = max(universal) if universal else (math.nan, "")
        output.append(
            {
                "ssim_threshold": threshold,
                "sample_count": len(samples),
                "oracle_feasible_samples": len(selected),
                "per_sample_oracle_geomean_speedup": geometric_mean(selected_speeds),
                "per_sample_oracle_min_speedup": min(selected_speeds) if selected_speeds else math.nan,
                "universal_best_method": universal_method,
                "universal_geomean_speedup": universal_speed,
                "selected_methods_json": json.dumps(selected, sort_keys=True),
                "evidence_scope": "best measured candidate; lower bound on restricted grouped oracle",
            }
        )
    return output


def cell_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, int, int, int, int, str, float], list[dict[str, object]]
    ] = defaultdict(list)
    for row in rows:
        if row.get("schedule_family") != "single_cell":
            continue
        key = (
            str(row["schedule_action"]),
            int(row["step_start"]),
            int(row["step_end"]),
            int(row["block_start"]),
            int(row["block_end"]),
            str(row.get("schedule_branches") or "[0, 1]"),
            float(row.get("forecast_scale") or 0.0),
        )
        grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        (
            action,
            step_start,
            step_end,
            block_start,
            block_end,
            branches,
            forecast_scale,
        ) = key
        output.append(
            {
                "action": action,
                "step_start": step_start,
                "step_end": step_end,
                "block_start": block_start,
                "block_end": block_end,
                "branches": branches,
                "forecast_scale": forecast_scale,
                "samples": len(values),
                "speedup_geomean": geometric_mean(
                    as_float(row, "end_to_end_speedup") for row in values
                ),
                "ssim_mean": float(
                    np.mean([as_float(row, "frame_ssim_mean") for row in values])
                ),
                "ssim_min": min(as_float(row, "frame_ssim_mean") for row in values),
                "quality_loss_worst": max(
                    0.0, 1.0 - min(as_float(row, "frame_ssim_mean") for row in values)
                ),
                "requested_quant_mean": float(
                    np.mean([as_float(row, "tri_mode_requested_quant") for row in values])
                ),
                "requested_cache_mean": float(
                    np.mean([as_float(row, "tri_mode_requested_cache") for row in values])
                ),
                "executed_cache_mean": float(
                    np.mean([as_float(row, "tri_mode_executed_cache") for row in values])
                ),
                "cache_refreshes_mean": float(
                    np.mean(
                        [
                            as_float_default(row, "tri_mode_cache_refreshes")
                            for row in values
                        ]
                    )
                ),
            }
        )
    return output


def make_plots(
    out_dir: Path,
    cells: list[dict[str, object]],
    methods: list[dict[str, object]],
    frontier: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    action_groups = sorted(
        {
            (
                str(row["action"]),
                str(row.get("branches") or "[0, 1]"),
                float(row.get("forecast_scale") or 0.0),
            )
            for row in cells
        }
    )
    if cells:
        figure, axes = plt.subplots(len(action_groups), 2, figsize=(11, 4.2 * len(action_groups)), squeeze=False)
        for action_index, (action, branches, forecast_scale) in enumerate(action_groups):
            selected = [
                row
                for row in cells
                if row["action"] == action
                and str(row.get("branches") or "[0, 1]") == branches
                and float(row.get("forecast_scale") or 0.0) == forecast_scale
            ]
            steps = sorted({int(row["step_start"]) for row in selected})
            blocks = sorted({int(row["block_start"]) for row in selected})
            for metric_index, (field, title, cmap) in enumerate(
                (("speedup_geomean", "End-to-end speedup", "RdYlGn"), ("ssim_min", "Worst paired SSIM", "viridis"))
            ):
                matrix = np.full((len(steps), len(blocks)), np.nan)
                for row in selected:
                    matrix[steps.index(int(row["step_start"])), blocks.index(int(row["block_start"]))] = float(row[field])
                axis = axes[action_index, metric_index]
                image = axis.imshow(matrix, aspect="auto", cmap=cmap)
                axis.set_title(
                    f"{action} branches={branches} forecast={forecast_scale:g}: {title}"
                )
                axis.set_xlabel("block-group start")
                axis.set_ylabel("step-group start")
                axis.set_xticks(range(len(blocks)), blocks)
                axis.set_yticks(range(len(steps)), steps)
                for row_index in range(matrix.shape[0]):
                    for column_index in range(matrix.shape[1]):
                        axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.3f}", ha="center", va="center", fontsize=7)
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(out_dir / "single_cell_heatmaps.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.scatter(
        [as_float(row, "speedup_geomean") for row in methods],
        [as_float(row, "ssim_min") for row in methods],
        color="#9aa0a6",
        alpha=0.45,
        s=24,
        label="measured schedules",
    )
    if frontier:
        x = [as_float(row, "speedup_geomean") for row in frontier]
        y = [as_float(row, "ssim_min") for row in frontier]
        axis.plot(x, y, "o-", color="#b23a2b", linewidth=1.8, label="Pareto frontier")
        for row in frontier:
            axis.annotate(str(row["method"]), (as_float(row, "speedup_geomean"), as_float(row, "ssim_min")), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axis.axhline(0.98, color="#234e70", linestyle="--", linewidth=1.2, label="SSIM 0.98 gate")
    axis.axvline(1.2, color="#6b705c", linestyle=":", linewidth=1.2, label="1.2x stop gate")
    axis.set_xlabel("end-to-end speedup vs paired dense")
    axis.set_ylabel("worst paired frame-mean SSIM")
    axis.set_title("Restricted tri-mode measured frontier")
    axis.grid(alpha=0.22)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "oracle_frontier.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    thresholds = [float(token) for token in args.thresholds.split(",") if token.strip()]
    run_dirs = [path.resolve() for path in args.run_dir]
    paired_csvs = [path.resolve() for path in args.paired_csv]
    if not run_dirs and not paired_csvs:
        raise ValueError("provide at least one --run-dir or --paired-csv")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paired = collect_paired_rows(run_dirs)
    for path in paired_csvs:
        paired.extend(read_csv(path))
    methods = aggregate_methods(paired)
    frontier = pareto_frontier(methods)
    thresholds_rows = oracle_thresholds(paired, thresholds)
    cells = cell_rows(paired)
    write_csv(out_dir / "paired_metrics.csv", paired)
    write_csv(out_dir / "method_summary.csv", methods)
    write_csv(out_dir / "oracle_frontier.csv", frontier)
    write_csv(out_dir / "oracle_thresholds.csv", thresholds_rows)
    write_csv(out_dir / "tri_mode_cell_metrics.csv", cells)

    gate = min(thresholds_rows, key=lambda row: abs(float(row["ssim_threshold"]) - 0.98))
    oracle_speed = as_float(gate, "per_sample_oracle_geomean_speedup")
    universal_speed = as_float(gate, "universal_geomean_speedup")
    stop_high_fidelity, focus_adaptive, next_gate = route_decision(
        oracle_speed, universal_speed
    )
    decision = {
        "status": "preliminary" if len({row["logical_sample"] for row in paired}) < 6 else "multi-sample",
        "evidence_scope": "restricted grouped schedules with actual rollout; measured best is an oracle lower bound, not an exhaustive upper bound",
        "ssim_gate": float(gate["ssim_threshold"]),
        "per_sample_oracle_geomean_speedup": oracle_speed,
        "universal_geomean_speedup": universal_speed,
        "stop_training_free_high_fidelity": stop_high_fidelity,
        "focus_sample_adaptive_controller": focus_adaptive,
        "next_gate": next_gate,
        "run_dirs": [str(path) for path in run_dirs],
        "paired_csvs": [str(path) for path in paired_csvs],
    }
    (out_dir / "oracle_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.no_plots:
        make_plots(out_dir, cells, methods, frontier)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
