from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probes.spectral_event_trigger import (  # noqa: E402
    CAMERA_SCENARIOS,
    NEGATIVE_SCENARIOS,
    OBJECT_EVENT_SCENARIOS,
    RARE_EVENT_SCENARIOS,
    SCENARIOS,
    generate_controlled_sequence,
    trace_controlled_sequence,
)


METHOD_LABELS = {
    "frame_delta": "Frame delta",
    "causalmem_residual_proxy": "CausalMem residual proxy",
    "single_oja_residual": "Single Oja residual",
    "dual_slow_residual": "Dual-state residual only",
    "dual_spectral": "Dual-timescale spectral",
}

METHOD_COLORS = {
    "frame_delta": "#7A7A7A",
    "causalmem_residual_proxy": "#0072B2",
    "single_oja_residual": "#56B4E9",
    "dual_slow_residual": "#E69F00",
    "dual_spectral": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "spectral_event_trigger.yaml"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile from an empty list")
    return float(np.quantile(np.asarray(values), q, method="higher"))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _binary_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    values = np.asarray(positive + negative, dtype=np.float64)
    labels = np.asarray([1] * len(positive) + [0] * len(negative))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    positive_count = len(positive)
    negative_count = len(negative)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def _run_split(
    cfg: dict[str, object],
    *,
    seeds: list[int],
    split: str,
) -> list[dict[str, object]]:
    synthetic = cfg["synthetic"]
    state = cfg["state"]
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for scenario in SCENARIOS:
            sequence = generate_controlled_sequence(
                scenario=scenario,
                seed=seed,
                frames=int(synthetic["frames"]),
                height=int(synthetic["height"]),
                width=int(synthetic["width"]),
                hidden_dim=int(synthetic["hidden_dim"]),
                base_rank=int(synthetic["base_rank"]),
                event_frame=int(synthetic["event_frame"]),
                event_block_size=int(synthetic["event_block_size"]),
                event_amplitude=float(synthetic["event_amplitude"]),
                noise_std=float(synthetic["noise_std"]),
            )
            for rank_budget in state["total_rank_budgets"]:
                traces = trace_controlled_sequence(
                    sequence,
                    total_rank_budget=int(rank_budget),
                    storage_bits=int(state["storage_bits"]),
                    fast_beta=float(state["fast_beta"]),
                    slow_beta=float(state["slow_beta"]),
                    fast_learning_rate=float(state["fast_learning_rate"]),
                    slow_learning_rate=float(state["slow_learning_rate"]),
                    causal_activity_beta=float(state["causal_activity_beta"]),
                    causal_max_new_directions=int(
                        state["causal_max_new_directions"]
                    ),
                )
                rows.extend({"split": split, **row} for row in traces)
    return rows


def _fit_calibration(
    rows: list[dict[str, object]],
    *,
    warmup_frames: int,
    target_false_trigger_rate: float,
    component_scale_quantile: float,
) -> dict[str, object]:
    calibration: dict[str, object] = {}
    ranks = sorted({int(row["total_rank_budget"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    for rank_budget in ranks:
        negative = [
            row
            for row in rows
            if int(row["total_rank_budget"]) == rank_budget
            and str(row["scenario"]) in NEGATIVE_SCENARIOS
            and int(row["frame"]) >= warmup_frames
        ]
        dual = [row for row in negative if row["method"] == "dual_spectral"]
        scales = {
            component: max(
                _quantile(
                    [abs(float(row[component])) for row in dual],
                    component_scale_quantile,
                ),
                1e-9,
            )
            for component in (
                "residual_component",
                "angle_component",
                "spectrum_component",
            )
        }
        method_rows: dict[str, object] = {}
        for method in methods:
            selected = [row for row in negative if row["method"] == method]
            scores = [_score_row(row, scales) for row in selected]
            threshold = _quantile(scores, 1.0 - target_false_trigger_rate)
            method_rows[method] = {
                "threshold": threshold,
                "negative_samples": len(scores),
                "empirical_false_trigger_rate": float(
                    np.mean(np.asarray(scores) > threshold)
                ),
            }
        calibration[str(rank_budget)] = {
            "component_scales": scales,
            "methods": method_rows,
        }
    return calibration


def _score_row(
    row: dict[str, object],
    scales: dict[str, float],
) -> float:
    if row["method"] != "dual_spectral":
        return float(row["raw_score"])
    return sum(
        float(row[component]) / scales[component]
        for component in (
            "residual_component",
            "angle_component",
            "spectrum_component",
        )
    )


def _apply_detection(
    rows: list[dict[str, object]],
    calibration: dict[str, object],
    *,
    warmup_frames: int,
    detection_tolerance: int,
    cooldown_frames: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    last_trigger: dict[tuple[int, str, int, str], int] = {}
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["total_rank_budget"]),
            str(row["method"]),
            int(row["seed"]),
            str(row["scenario"]),
            int(row["frame"]),
        ),
    )
    for row in ordered:
        rank_budget = int(row["total_rank_budget"])
        rank_calibration = calibration[str(rank_budget)]
        scales = rank_calibration["component_scales"]
        method_calibration = rank_calibration["methods"][str(row["method"])]
        score = _score_row(row, scales)
        threshold = float(method_calibration["threshold"])
        frame = int(row["frame"])
        key = (
            rank_budget,
            str(row["method"]),
            int(row["seed"]),
            str(row["scenario"]),
        )
        eligible = frame >= warmup_frames and score > threshold
        triggered = eligible and (
            key not in last_trigger
            or frame - last_trigger[key] > cooldown_frames
        )
        if triggered:
            last_trigger[key] = frame
        event_frame = int(row.get("event_frame", -1))
        output.append(
            {
                **row,
                "score": score,
                "threshold": threshold,
                "normalized_score": score / max(threshold, 1e-12),
                "triggered": int(triggered),
                "positive_window": int(
                    event_frame >= 0
                    and event_frame <= frame <= event_frame + detection_tolerance
                ),
            }
        )
    return output


def _attach_event_frame(
    rows: list[dict[str, object]],
    event_frame: int,
) -> None:
    for row in rows:
        row["event_frame"] = (
            -1 if str(row["scenario"]) in NEGATIVE_SCENARIOS else event_frame
        )


def _event_outcomes(
    rows: list[dict[str, object]],
    *,
    detection_tolerance: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, int, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in rows:
        if int(row["event_frame"]) < 0:
            continue
        key = (
            int(row["total_rank_budget"]),
            str(row["method"]),
            int(row["seed"]),
            str(row["scenario"]),
        )
        grouped[key].append(row)
    outcomes: list[dict[str, object]] = []
    for (rank_budget, method, seed, scenario), selected in grouped.items():
        onset = int(selected[0]["event_frame"])
        hits = sorted(
            int(row["frame"])
            for row in selected
            if bool(row["triggered"])
            and onset <= int(row["frame"]) <= onset + detection_tolerance
        )
        outcomes.append(
            {
                "total_rank_budget": rank_budget,
                "method": method,
                "seed": seed,
                "scenario": scenario,
                "event_frame": onset,
                "detected": int(bool(hits)),
                "detection_delay": hits[0] - onset if hits else None,
            }
        )
    return outcomes


def _summarize(
    rows: list[dict[str, object]],
    outcomes: list[dict[str, object]],
    calibration: dict[str, object],
    *,
    warmup_frames: int,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    groups = sorted(
        {
            (int(row["total_rank_budget"]), str(row["method"]))
            for row in rows
        }
    )
    for rank_budget, method in groups:
        selected = [
            row
            for row in rows
            if int(row["total_rank_budget"]) == rank_budget
            and row["method"] == method
            and int(row["frame"]) >= warmup_frames
        ]
        negative = [
            row for row in selected if row["scenario"] in NEGATIVE_SCENARIOS
        ]
        camera = [row for row in selected if row["scenario"] in CAMERA_SCENARIOS]
        periodic = [
            row for row in selected if row["scenario"] == "periodic_motion"
        ]
        event_rows = [
            row
            for row in outcomes
            if int(row["total_rank_budget"]) == rank_budget
            and row["method"] == method
        ]
        rare = [
            row for row in event_rows if row["scenario"] in RARE_EVENT_SCENARIOS
        ]
        objects = [
            row
            for row in event_rows
            if row["scenario"] in OBJECT_EVENT_SCENARIOS
        ]
        scene = [row for row in event_rows if row["scenario"] == "scene_cut"]
        delays = [
            float(row["detection_delay"])
            for row in event_rows
            if row["detection_delay"] is not None
        ]
        positive_scores = [
            float(row["score"]) for row in selected if bool(row["positive_window"])
        ]
        negative_scores = [float(row["score"]) for row in negative]
        latency = [float(row["update_us"]) for row in selected]
        method_calibration = calibration[str(rank_budget)]["methods"][method]
        summary.append(
            {
                "total_rank_budget": rank_budget,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "state_bytes": int(selected[0]["state_bytes"]),
                "estimated_update_flops": int(
                    selected[0]["estimated_update_flops"]
                ),
                "threshold": float(method_calibration["threshold"]),
                "calibration_false_trigger_rate": float(
                    method_calibration["empirical_false_trigger_rate"]
                ),
                "false_trigger_rate": float(
                    np.mean([bool(row["triggered"]) for row in negative])
                ),
                "camera_false_trigger_rate": float(
                    np.mean([bool(row["triggered"]) for row in camera])
                ),
                "periodic_false_trigger_rate": float(
                    np.mean([bool(row["triggered"]) for row in periodic])
                ),
                "event_recall": float(
                    np.mean([bool(row["detected"]) for row in event_rows])
                ),
                "rare_event_recall": float(
                    np.mean([bool(row["detected"]) for row in rare])
                ),
                "object_change_recall": float(
                    np.mean([bool(row["detected"]) for row in objects])
                ),
                "scene_cut_recall": float(
                    np.mean([bool(row["detected"]) for row in scene])
                ),
                "mean_detection_delay": _mean(delays),
                "scene_cut_mean_delay": _mean(
                    [
                        float(row["detection_delay"])
                        for row in scene
                        if row["detection_delay"] is not None
                    ]
                ),
                "frame_auc": _binary_auc(positive_scores, negative_scores),
                "writes_per_100_frames": 100.0
                * float(np.mean([bool(row["triggered"]) for row in selected])),
                "update_p50_us": _percentile(latency, 50),
                "update_p95_us": _percentile(latency, 95),
                "update_p99_us": _percentile(latency, 99),
            }
        )
    return summary


def _summarize_scenarios(
    rows: list[dict[str, object]],
    outcomes: list[dict[str, object]],
    *,
    warmup_frames: int,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    groups = sorted(
        {
            (
                int(row["total_rank_budget"]),
                str(row["method"]),
                str(row["scenario"]),
            )
            for row in rows
        }
    )
    for rank_budget, method, scenario in groups:
        selected = [
            row
            for row in rows
            if int(row["total_rank_budget"]) == rank_budget
            and row["method"] == method
            and row["scenario"] == scenario
            and int(row["frame"]) >= warmup_frames
        ]
        scenario_outcomes = [
            row
            for row in outcomes
            if int(row["total_rank_budget"]) == rank_budget
            and row["method"] == method
            and row["scenario"] == scenario
        ]
        delays = [
            float(row["detection_delay"])
            for row in scenario_outcomes
            if row["detection_delay"] is not None
        ]
        is_negative = scenario in NEGATIVE_SCENARIOS
        summary.append(
            {
                "total_rank_budget": rank_budget,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "scenario": scenario,
                "scenario_type": "negative_control" if is_negative else "event",
                "seed_count": len({int(row["seed"]) for row in selected}),
                "eligible_frame_count": len(selected),
                "trigger_count": sum(bool(row["triggered"]) for row in selected),
                "false_trigger_rate": (
                    float(np.mean([bool(row["triggered"]) for row in selected]))
                    if is_negative
                    else None
                ),
                "event_count": len(scenario_outcomes),
                "detected_count": sum(
                    bool(row["detected"]) for row in scenario_outcomes
                ),
                "event_recall": (
                    float(
                        np.mean(
                            [bool(row["detected"]) for row in scenario_outcomes]
                        )
                    )
                    if scenario_outcomes
                    else None
                ),
                "mean_detection_delay": _mean(delays),
            }
        )
    return summary


def _paired_false_trigger_bootstrap(
    rows: list[dict[str, object]],
    *,
    primary_rank: int,
    warmup_frames: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    methods = (
        "causalmem_residual_proxy",
        "single_oja_residual",
        "dual_slow_residual",
        "dual_spectral",
    )
    selected = [
        row
        for row in rows
        if int(row["total_rank_budget"]) == primary_rank
        and row["scenario"] in NEGATIVE_SCENARIOS
        and int(row["frame"]) >= warmup_frames
        and row["method"] in methods
    ]
    grouped: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for row in selected:
        grouped[(str(row["method"]), int(row["seed"]))].append(
            bool(row["triggered"])
        )
    seeds = sorted(
        set.intersection(
            *(
                {seed for method, seed in grouped if method == target}
                for target in methods
            )
        )
    )
    if not seeds:
        raise ValueError("paired false-trigger analysis has no common seeds")
    rates = {
        method: np.asarray(
            [float(np.mean(grouped[(method, seed)])) for seed in seeds],
            dtype=np.float64,
        )
        for method in methods
    }
    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(
        0,
        len(seeds),
        size=(bootstrap_samples, len(seeds)),
    )
    comparisons: list[dict[str, object]] = []
    by_seed: list[dict[str, object]] = []
    dual = rates["dual_spectral"]
    for baseline_method in methods[:-1]:
        baseline = rates[baseline_method]
        differences = dual - baseline
        bootstrapped = np.mean(differences[sample_indices], axis=1)
        baseline_mean = float(np.mean(baseline))
        dual_mean = float(np.mean(dual))
        comparisons.append(
            {
                "total_rank_budget": primary_rank,
                "candidate_method": "dual_spectral",
                "baseline_method": baseline_method,
                "seed_count": len(seeds),
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
                "candidate_false_trigger_rate": dual_mean,
                "baseline_false_trigger_rate": baseline_mean,
                "candidate_minus_baseline_pp": 100.0
                * (dual_mean - baseline_mean),
                "paired_bootstrap_ci95_low_pp": 100.0
                * float(np.percentile(bootstrapped, 2.5)),
                "paired_bootstrap_ci95_high_pp": 100.0
                * float(np.percentile(bootstrapped, 97.5)),
                "relative_reduction": (
                    (baseline_mean - dual_mean) / baseline_mean
                    if baseline_mean > 0.0
                    else None
                ),
            }
        )
        for index, seed in enumerate(seeds):
            by_seed.append(
                {
                    "total_rank_budget": primary_rank,
                    "seed": seed,
                    "candidate_method": "dual_spectral",
                    "baseline_method": baseline_method,
                    "candidate_false_trigger_rate": float(dual[index]),
                    "baseline_false_trigger_rate": float(baseline[index]),
                    "candidate_minus_baseline_pp": 100.0
                    * float(differences[index]),
                }
            )
    return comparisons, by_seed


def _find_summary(
    summary: list[dict[str, object]], rank: int, method: str
) -> dict[str, object]:
    return next(
        row
        for row in summary
        if int(row["total_rank_budget"]) == rank and row["method"] == method
    )


def _evaluate_gates(
    summary: list[dict[str, object]],
    cfg: dict[str, object],
) -> dict[str, object]:
    primary_rank = int(cfg["detection"]["primary_rank_budget"])
    gates = cfg["gates"]
    dual = _find_summary(summary, primary_rank, "dual_spectral")
    causal = _find_summary(summary, primary_rank, "causalmem_residual_proxy")
    residual = _find_summary(summary, primary_rank, "dual_slow_residual")
    checks = {
        "camera_false_trigger_rate": {
            "value": dual["camera_false_trigger_rate"],
            "threshold": float(gates["max_camera_false_trigger_rate"]),
            "passed": float(dual["camera_false_trigger_rate"])
            <= float(gates["max_camera_false_trigger_rate"]),
        },
        "scene_cut_delay": {
            "value": dual["scene_cut_mean_delay"],
            "threshold": float(gates["max_scene_cut_delay_frames"]),
            "passed": dual["scene_cut_mean_delay"] is not None
            and float(dual["scene_cut_mean_delay"])
            <= float(gates["max_scene_cut_delay_frames"]),
        },
        "event_recall_vs_causalmem_proxy": {
            "value": float(dual["event_recall"])
            - float(causal["event_recall"]),
            "threshold": float(gates["min_event_recall_gain"]),
            "passed": float(dual["event_recall"])
            - float(causal["event_recall"])
            >= float(gates["min_event_recall_gain"]),
        },
        "rare_recall_vs_residual_only": {
            "value": float(dual["rare_event_recall"])
            - float(residual["rare_event_recall"]),
            "threshold": float(gates["min_rare_recall_gain"]),
            "passed": float(dual["rare_event_recall"])
            - float(residual["rare_event_recall"])
            >= float(gates["min_rare_recall_gain"]),
        },
        "writer_p95_us": {
            "value": dual["update_p95_us"],
            "threshold": float(gates["max_writer_p95_us"]),
            "passed": float(dual["update_p95_us"])
            <= float(gates["max_writer_p95_us"]),
        },
        "matched_basis_state_bytes": {
            "value": float(dual["state_bytes"]) / float(causal["state_bytes"]),
            "threshold": float(gates["max_state_ratio_vs_causalmem"]),
            "passed": float(dual["state_bytes"])
            / float(causal["state_bytes"])
            <= float(gates["max_state_ratio_vs_causalmem"]),
        },
    }
    return {
        "primary_rank_budget": primary_rank,
        "checks": checks,
        "all_passed": all(bool(row["passed"]) for row in checks.values()),
    }


def _plot_tradeoff(summary: list[dict[str, object]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ranks = sorted({int(row["total_rank_budget"]) for row in summary})
    marker_cycle = ("o", "s", "^", "D", "P", "X")
    rank_markers = {
        rank: marker_cycle[index % len(marker_cycle)]
        for index, rank in enumerate(ranks)
    }
    for method in METHOD_LABELS:
        selected = sorted(
            (row for row in summary if row["method"] == method),
            key=lambda row: int(row["total_rank_budget"]),
        )
        ax.plot(
            [100 * float(row["false_trigger_rate"]) for row in selected],
            [100 * float(row["event_recall"]) for row in selected],
            linewidth=1.7,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for row in selected:
            rank = int(row["total_rank_budget"])
            ax.scatter(
                100 * float(row["false_trigger_rate"]),
                100 * float(row["event_recall"]),
                marker=rank_markers[rank],
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.6,
                s=52,
                zorder=3,
            )
    ax.set_xlabel("Negative-control false-trigger rate (%)")
    ax.set_ylabel("Event recall within tolerance (%)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    method_legend = ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    ax.add_artist(method_legend)
    rank_handles = [
        Line2D(
            [0],
            [0],
            marker=rank_markers[rank],
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=7,
            label=f"rank {rank}",
        )
        for rank in sorted(rank_markers)
    ]
    ax.legend(
        handles=rank_handles,
        title="Basis budget",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        loc="center right",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"event_recall_vs_false_trigger_rate.{suffix}", dpi=300)
    plt.close(fig)


def _plot_traces(
    rows: list[dict[str, object]],
    out_dir: Path,
    *,
    primary_rank: int,
    seed: int,
) -> None:
    scenarios = ("camera_fast", "scene_cut", "one_frame_ocr", "object_disappear")
    methods = ("frame_delta", "causalmem_residual_proxy", "dual_spectral")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True)
    for ax, scenario in zip(axes.flat, scenarios):
        for method in methods:
            selected = sorted(
                (
                    row
                    for row in rows
                    if int(row["total_rank_budget"]) == primary_rank
                    and int(row["seed"]) == seed
                    and row["scenario"] == scenario
                    and row["method"] == method
                ),
                key=lambda row: int(row["frame"]),
            )
            ax.plot(
                [int(row["frame"]) for row in selected],
                [float(row["normalized_score"]) for row in selected],
                color=METHOD_COLORS[method],
                linewidth=1.5,
                label=METHOD_LABELS[method],
            )
        event_frame = int(selected[0]["event_frame"]) if selected else -1
        if event_frame >= 0:
            ax.axvline(event_frame, color="#CC3311", linestyle="--", linewidth=1.0)
        ax.axhline(1.0, color="#333333", linestyle=":", linewidth=1.0)
        ax.text(0.02, 0.92, scenario.replace("_", " "), transform=ax.transAxes)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Frame")
    axes[1, 1].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Score / threshold")
    axes[1, 0].set_ylabel("Score / threshold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"spectral_trigger_traces.{suffix}", dpi=300)
    plt.close(fig)


def _plot_latency(
    summary: list[dict[str, object]],
    out_dir: Path,
    *,
    primary_rank: int,
) -> None:
    selected = [
        row for row in summary if int(row["total_rank_budget"]) == primary_rank
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(selected))
    ax.bar(
        positions,
        [float(row["update_p95_us"]) for row in selected],
        color=[METHOD_COLORS[str(row["method"])] for row in selected],
    )
    ax.set_xticks(
        positions,
        [METHOD_LABELS[str(row["method"])] for row in selected],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("CPU update P95 (us/frame)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"spectral_trigger_update_latency.{suffix}", dpi=300)
    plt.close(fig)


def _plot_scenario_metrics(
    scenario_summary: list[dict[str, object]],
    out_dir: Path,
    *,
    primary_rank: int,
) -> None:
    selected = [
        row
        for row in scenario_summary
        if int(row["total_rank_budget"]) == primary_rank
    ]
    plot_specs = (
        (
            "event",
            "event_recall",
            "Event recall within tolerance (%)",
            "scenario_event_recall",
        ),
        (
            "negative_control",
            "false_trigger_rate",
            "False-trigger rate (%)",
            "scenario_false_trigger_rate",
        ),
    )
    for scenario_type, metric, ylabel, stem in plot_specs:
        scenarios = [
            scenario
            for scenario in SCENARIOS
            if any(
                row["scenario"] == scenario
                and row["scenario_type"] == scenario_type
                for row in selected
            )
        ]
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        positions = np.arange(len(scenarios), dtype=np.float64)
        width = 0.16
        for method_index, method in enumerate(METHOD_LABELS):
            method_rows = {
                str(row["scenario"]): row
                for row in selected
                if row["method"] == method
                and row["scenario_type"] == scenario_type
            }
            values = [
                100.0 * float(method_rows[scenario][metric])
                for scenario in scenarios
            ]
            offset = (method_index - (len(METHOD_LABELS) - 1) / 2.0) * width
            ax.bar(
                positions + offset,
                values,
                width=width,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_xticks(
            positions,
            [scenario.replace("_", " ") for scenario in scenarios],
            rotation=18,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, ncol=3)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300)
        plt.close(fig)


def _format_percent(value: object) -> str:
    return f"{100 * float(value):.1f}%"


def _write_report(
    path: Path,
    summary: list[dict[str, object]],
    scenario_summary: list[dict[str, object]],
    paired_comparisons: list[dict[str, object]],
    gate_results: dict[str, object],
    cfg: dict[str, object],
) -> None:
    rank = int(gate_results["primary_rank_budget"])
    selected = [
        row for row in summary if int(row["total_rank_budget"]) == rank
    ]
    scenario_lookup = {
        (str(row["method"]), str(row["scenario"])): row
        for row in scenario_summary
        if int(row["total_rank_budget"]) == rank
    }
    report_methods = (
        "causalmem_residual_proxy",
        "dual_slow_residual",
        "dual_spectral",
    )
    lines = [
        "# Controlled Spectral Event Trigger Analysis",
        "",
        "## Evidence Boundary",
        "",
        "This is a deterministic synthetic feature-stream experiment. It tests",
        "event-trigger behavior and CPU proxy cost, not Video-LLM answer quality.",
        "`causalmem_residual_proxy` is an independent mechanism proxy and not the",
        "official CausalMem implementation.",
        "",
        "## Primary Matched-Rank Result",
        "",
        f"Total basis-rank budget: `{rank}`. Detection tolerance: "
        f"`{cfg['detection']['detection_tolerance_frames']}` frames.",
        "",
        "| Method | State | Event recall | Rare recall | Camera FTR | Scene delay | Frame AUC | Update P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        scene_delay = (
            "missed"
            if row["scene_cut_mean_delay"] is None
            else f"{float(row['scene_cut_mean_delay']):.2f}"
        )
        auc = "N/A" if row["frame_auc"] is None else f"{float(row['frame_auc']):.3f}"
        lines.append(
            "| {label} | {state:,} B | {event} | {rare} | {camera} | "
            "{delay} | {auc} | {p95:.1f} us |".format(
                label=row["method_label"],
                state=int(row["state_bytes"]),
                event=_format_percent(row["event_recall"]),
                rare=_format_percent(row["rare_event_recall"]),
                camera=_format_percent(row["camera_false_trigger_rate"]),
                delay=scene_delay,
                auc=auc,
                p95=float(row["update_p95_us"]),
            )
        )
    lines.extend(
        [
            "",
            "## Scenario Localization",
            "",
            "| Event scenario | CausalMem proxy | Dual residual | Dual spectral |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        if scenario in NEGATIVE_SCENARIOS:
            continue
        recalls = [
            _format_percent(scenario_lookup[(method, scenario)]["event_recall"])
            for method in report_methods
        ]
        lines.append(
            f"| {scenario.replace('_', ' ')} | {recalls[0]} | "
            f"{recalls[1]} | {recalls[2]} |"
        )
    lines.extend(
        [
            "",
            "| Negative scenario | CausalMem proxy FTR | Dual residual FTR | Dual spectral FTR |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        if scenario not in NEGATIVE_SCENARIOS:
            continue
        rates = [
            _format_percent(
                scenario_lookup[(method, scenario)]["false_trigger_rate"]
            )
            for method in report_methods
        ]
        lines.append(
            f"| {scenario.replace('_', ' ')} | {rates[0]} | "
            f"{rates[1]} | {rates[2]} |"
        )
    lines.extend(
        [
            "",
            "## Paired Selectivity Analysis",
            "",
            "Paired bootstrap intervals resample the eight evaluation seeds. "
            "Differences are dual spectral minus baseline false-trigger rate; "
            "negative values favor dual spectral. This is secondary evidence, "
            "not a replacement for the preregistered recall gate.",
            "",
            "| Baseline | Baseline FTR | Dual FTR | Relative reduction | Difference (pp) | Paired 95% CI (pp) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in paired_comparisons:
        reduction = comparison["relative_reduction"]
        reduction_text = (
            "N/A" if reduction is None else _format_percent(reduction)
        )
        lines.append(
            "| {baseline} | {baseline_ftr} | {dual_ftr} | {reduction} | "
            "{difference:.3f} | [{low:.3f}, {high:.3f}] |".format(
                baseline=METHOD_LABELS[str(comparison["baseline_method"])],
                baseline_ftr=_format_percent(
                    comparison["baseline_false_trigger_rate"]
                ),
                dual_ftr=_format_percent(
                    comparison["candidate_false_trigger_rate"]
                ),
                reduction=reduction_text,
                difference=float(comparison["candidate_minus_baseline_pp"]),
                low=float(comparison["paired_bootstrap_ci95_low_pp"]),
                high=float(comparison["paired_bootstrap_ci95_high_pp"]),
            )
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
        ]
    )
    for name, gate in gate_results["checks"].items():
        lines.append(
            f"- `{name}`: value `{gate['value']}`, threshold `{gate['threshold']}`, "
            f"passed `{gate['passed']}`."
        )
    verdict = "passes" if gate_results["all_passed"] else "does not pass"
    rare_gate = gate_results["checks"]["rare_recall_vs_residual_only"]
    disappear = scenario_lookup[("dual_spectral", "object_disappear")]
    lines.extend(
        [
            "",
            "## Failure Localization",
            "",
            f"- Rare-event recall gain was `{100 * float(rare_gate['value']):.1f}` "
            f"percentage points versus the required "
            f"`{100 * float(rare_gate['threshold']):.1f}`.",
            "- Dual spectral object-disappearance recall was "
            f"`{_format_percent(disappear['event_recall'])}`; the spectral terms "
            "did not recover this missing event class.",
            "- The observed advantage is lower false-trigger rate at unchanged "
            "event recall, not improved rare-event recall.",
            "",
            "## Verdict",
            "",
            f"The dual-timescale trigger **{verdict}** the preregistered synthetic gate.",
            "A pass only promotes the mechanism to native-feature validation. A",
            "failure keeps spectral state as a diagnostic rather than a memory writer.",
            "Single-Oja task-memory results remain rejected regardless of this trigger",
            "outcome.",
            "",
            "## Files",
            "",
            "- `summary.csv`: method/rank metrics and latency percentiles.",
            "- `per_scenario.csv`: event recall and negative-control FTR by scenario.",
            "- `paired_false_trigger_bootstrap.csv`: paired seed-bootstrap comparisons.",
            "- `paired_false_trigger_by_seed.csv`: raw seed-level paired rates.",
            "- `per_frame.csv`: complete score and trigger traces.",
            "- `event_outcomes.csv`: event-level recall and delay.",
            "- `calibration.json`: negative-only scales and thresholds.",
            "- `event_recall_vs_false_trigger_rate.*`: quality trade-off.",
            "- `spectral_trigger_traces.*`: representative normalized traces.",
            "- `spectral_trigger_update_latency.*`: CPU P95 update cost.",
            "- `scenario_event_recall.*`: event recall localized by scenario.",
            "- `scenario_false_trigger_rate.*`: false triggers by negative scenario.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detection = cfg["detection"]
    synthetic = cfg["synthetic"]

    calibration_rows = _run_split(
        cfg,
        seeds=[int(value) for value in cfg["calibration_seeds"]],
        split="calibration",
    )
    evaluation_rows = _run_split(
        cfg,
        seeds=[int(value) for value in cfg["evaluation_seeds"]],
        split="evaluation",
    )
    event_frame = int(synthetic["event_frame"])
    _attach_event_frame(calibration_rows, event_frame)
    _attach_event_frame(evaluation_rows, event_frame)
    calibration = _fit_calibration(
        calibration_rows,
        warmup_frames=int(detection["warmup_frames"]),
        target_false_trigger_rate=float(detection["target_false_trigger_rate"]),
        component_scale_quantile=float(
            detection["component_scale_quantile"]
        ),
    )
    detected_rows = _apply_detection(
        evaluation_rows,
        calibration,
        warmup_frames=int(detection["warmup_frames"]),
        detection_tolerance=int(detection["detection_tolerance_frames"]),
        cooldown_frames=int(detection["cooldown_frames"]),
    )
    outcomes = _event_outcomes(
        detected_rows,
        detection_tolerance=int(detection["detection_tolerance_frames"]),
    )
    summary = _summarize(
        detected_rows,
        outcomes,
        calibration,
        warmup_frames=int(detection["warmup_frames"]),
    )
    scenario_summary = _summarize_scenarios(
        detected_rows,
        outcomes,
        warmup_frames=int(detection["warmup_frames"]),
    )
    analysis_cfg = cfg["analysis"]
    paired_comparisons, paired_by_seed = _paired_false_trigger_bootstrap(
        detected_rows,
        primary_rank=int(detection["primary_rank_budget"]),
        warmup_frames=int(detection["warmup_frames"]),
        bootstrap_samples=int(analysis_cfg["bootstrap_samples"]),
        bootstrap_seed=int(analysis_cfg["bootstrap_seed"]),
    )
    gate_results = _evaluate_gates(summary, cfg)
    payload = {
        "format_version": 2,
        "config": str(args.config.resolve()),
        "summary": summary,
        "scenario_summary": scenario_summary,
        "paired_false_trigger_comparisons": paired_comparisons,
        "gates": gate_results,
        "claim_boundary": (
            "controlled synthetic trigger evidence; not model quality or official "
            "CausalMem/StreamingTOM/STC latency"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.out_dir / "summary.csv", summary)
    _write_csv(args.out_dir / "per_scenario.csv", scenario_summary)
    _write_csv(
        args.out_dir / "paired_false_trigger_bootstrap.csv",
        paired_comparisons,
    )
    _write_csv(
        args.out_dir / "paired_false_trigger_by_seed.csv",
        paired_by_seed,
    )
    _write_csv(args.out_dir / "per_frame.csv", detected_rows)
    _write_csv(args.out_dir / "event_outcomes.csv", outcomes)
    _plot_tradeoff(summary, args.out_dir)
    _plot_traces(
        detected_rows,
        args.out_dir,
        primary_rank=int(detection["primary_rank_budget"]),
        seed=int(cfg["evaluation_seeds"][0]),
    )
    _plot_latency(
        summary,
        args.out_dir,
        primary_rank=int(detection["primary_rank_budget"]),
    )
    _plot_scenario_metrics(
        scenario_summary,
        args.out_dir,
        primary_rank=int(detection["primary_rank_budget"]),
    )
    _write_report(
        args.out_dir / "CONTROLLED_SPECTRAL_TRIGGER_ANALYSIS.md",
        summary,
        scenario_summary,
        paired_comparisons,
        gate_results,
        cfg,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if gate_results["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
