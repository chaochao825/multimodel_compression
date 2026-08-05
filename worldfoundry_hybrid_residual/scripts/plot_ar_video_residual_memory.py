"""Create publication-oriented figures from the LongLive residual-memory probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "post_rope": "#D55E00",
    "phase_aligned": "#0072B2",
    "adaptive_rank_oracle": "#009E73",
    "frozen_calibration_basis_oracle_coefficients": "#CC79A7",
    "none": "#4D4D4D",
}

SCOPE_SPLITS = {
    "calibration": {"calibration"},
    "validation": {"validation"},
    "test": {"test"},
    "held_out": {"validation", "test"},
    "all": {"calibration", "validation", "test"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=tuple(SCOPE_SPLITS), default="held_out"
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows available for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def mode_from_method(method: str) -> str:
    if method.startswith("phasealigned_"):
        return "phase_aligned"
    if method.startswith("postrope_"):
        return "post_rope"
    return "drop_middle"


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    best_error = math.inf
    for row in sorted(rows, key=lambda item: float(item["reduction"]), reverse=True):
        error = float(row["aggregate_error_percent"])
        if error < best_error:
            frontier.append(row)
            best_error = error
    return sorted(frontier, key=lambda item: float(item["reduction"]))


def plot_quality_cost(
    summaries: list[dict[str, Any]],
    primary: str,
    primary_rank: int,
    output_dir: Path,
    scope: str,
) -> None:
    rows = []
    for item in summaries:
        if item["scope"] != scope:
            continue
        correction = item["correction"]
        rank = int(item["rank"])
        if not (
            (correction == "none" and rank == 0)
            or (correction != "none" and rank == primary_rank)
        ):
            continue
        rows.append(
            {
                "method": item["method"],
                "mode": mode_from_method(item["method"]),
                "correction": correction,
                "rank": rank,
                "reduction": float(item["mean_arithmetic_reduction"]),
                "aggregate_error_percent": 100 * float(item["aggregate_relative_av_l2"]),
                "worst_error_percent": 100 * float(item["worst_head_relative_av_l2"]),
                "is_primary": item["method"] == primary,
                "scope": scope,
            }
        )
    write_csv(output_dir / "quality_cost_frontier.csv", rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = {"none": "o", "adaptive_rank_oracle": "^", "frozen_calibration_basis_oracle_coefficients": "s"}
    for correction in markers:
        subset = [row for row in rows if row["correction"] == correction]
        for mode in ("drop_middle", "post_rope", "phase_aligned"):
            points = [row for row in subset if row["mode"] == mode]
            if not points:
                continue
            color = COLORS.get(mode, "#777777")
            ax.scatter(
                [row["reduction"] for row in points],
                [row["aggregate_error_percent"] for row in points],
                marker=markers[correction],
                s=44,
                alpha=0.78,
                color=color,
                label=f"{mode.replace('_', ' ')} / {correction.replace('_', ' ')}",
            )
    frontier = pareto_frontier(rows)
    if len(frontier) > 1:
        ax.plot(
            [row["reduction"] for row in frontier],
            [row["aggregate_error_percent"] for row in frontier],
            color="#111111",
            linewidth=1.2,
            linestyle="--",
            label="observed Pareto envelope",
        )
    for row in rows:
        if row["is_primary"] and row["correction"] != "none":
            ax.annotate(
                f"primary {row['correction'].split('_')[0]}",
                (row["reduction"], row["aggregate_error_percent"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axvline(1.5, color="#666666", linewidth=1, linestyle=":")
    ax.axhline(0.5, color="#666666", linewidth=1, linestyle=":")
    ax.set_xlabel("Arithmetic key reduction (higher is better)")
    ax.set_ylabel(f"{scope.replace('_', ' ').title()} aggregate AV error (%)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, output_dir, "quality_cost_frontier")


def parse_structured_method(method: str) -> tuple[str, int, float] | None:
    match = re.fullmatch(
        r"(postrope|phasealigned)_recency_g(\d+)_event_([0-9p]+)", method
    )
    if not match:
        return None
    mode = "post_rope" if match.group(1) == "postrope" else "phase_aligned"
    return mode, int(match.group(2)), float(match.group(3).replace("p", "."))


def plot_phase_alignment(
    summaries: list[dict[str, Any]],
    primary_rank: int,
    output_dir: Path,
    scope: str,
) -> None:
    rows = []
    for item in summaries:
        parsed = parse_structured_method(item["method"])
        if item["scope"] != scope or parsed is None:
            continue
        correction = item["correction"]
        rank = int(item["rank"])
        if not (
            (correction == "none" and rank == 0)
            or (correction == "adaptive_rank_oracle" and rank == primary_rank)
        ):
            continue
        mode, groups, event_fraction = parsed
        rows.append(
            {
                "mode": mode,
                "summary_groups": groups,
                "event_fraction": event_fraction,
                "correction": correction,
                "rank": rank,
                "aggregate_error_percent": 100 * float(item["aggregate_relative_av_l2"]),
                "worst_error_percent": 100 * float(item["worst_head_relative_av_l2"]),
                "reduction": float(item["mean_arithmetic_reduction"]),
                "scope": scope,
            }
        )
    write_csv(output_dir / "phase_alignment_ablation.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for ax, correction in zip(axes, ("none", "adaptive_rank_oracle")):
        subset = [row for row in rows if row["correction"] == correction]
        fractions = sorted({row["event_fraction"] for row in subset})
        for mode in ("post_rope", "phase_aligned"):
            for fraction in fractions:
                line = sorted(
                    [
                        row
                        for row in subset
                        if row["mode"] == mode and row["event_fraction"] == fraction
                    ],
                    key=lambda row: row["summary_groups"],
                )
                if not line:
                    continue
                ax.plot(
                    [row["summary_groups"] for row in line],
                    [row["aggregate_error_percent"] for row in line],
                    marker="o" if mode == "post_rope" else "s",
                    color=COLORS[mode],
                    alpha=0.52 + 0.42 * (fraction / max(fractions or [1.0])),
                    label=f"{mode.replace('_', ' ')}, event={fraction:.0%}",
                )
        ax.set_xlabel("Temporal summary groups")
        ax.set_ylabel(f"{scope.replace('_', ' ').title()} aggregate AV error (%)")
        ax.text(0.02, 0.97, correction.replace("_", " "), transform=ax.transAxes, va="top")
        ax.legend(fontsize=7)
    save_figure(fig, output_dir, "phase_alignment_ablation")


def aggregate_metric_rows(rows: list[dict[str, str]]) -> float:
    numerator = sum(float(row["numerator_sq"]) for row in rows)
    denominator = sum(float(row["denominator_sq"]) for row in rows)
    return 100 * math.sqrt(numerator / denominator)


def plot_primary_heatmap(
    metrics: list[dict[str, str]],
    primary: str,
    primary_rank: int,
    output_dir: Path,
    scope: str,
) -> None:
    selected = [
        row
        for row in metrics
        if row["split"] in SCOPE_SPLITS[scope]
        and row["method"] == primary
        and row["correction"] == "adaptive_rank_oracle"
        and int(row["rank"]) == primary_rank
    ]
    layers = sorted({int(row["layer"]) for row in selected})
    columns = sorted(
        {
            (
                int(row["current_start_frame"]),
                int(row["denoising_call_index"]),
                int(row["denoising_timestep"]),
            )
            for row in selected
        }
    )
    cells = []
    matrix = np.full((len(layers), len(columns)), np.nan)
    for row_index, layer in enumerate(layers):
        for column_index, (start, call, timestep) in enumerate(columns):
            group = [
                row
                for row in selected
                if int(row["layer"]) == layer
                and int(row["current_start_frame"]) == start
                and int(row["denoising_call_index"]) == call
            ]
            value = aggregate_metric_rows(group)
            matrix[row_index, column_index] = value
            cells.append(
                {
                    "layer": layer,
                    "current_start_frame": start,
                    "denoising_call_index": call,
                    "denoising_timestep": timestep,
                    "aggregate_error_percent": value,
                    "scope": scope,
                }
            )
    write_csv(output_dir / "primary_cell_heatmap.csv", cells)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xticks(
        range(len(columns)),
        [f"frame {start}\ncall {call}, t={timestep}" for start, call, timestep in columns],
    )
    ax.set_ylabel("Transformer layer")
    ax.set_xlabel("Autoregressive start / denoising call")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Aggregate AV error (%)")
    save_figure(fig, output_dir, "primary_cell_heatmap")


def plot_rank_transfer(
    summaries: list[dict[str, Any]],
    primary: str,
    output_dir: Path,
    scope: str,
) -> None:
    primary_rows = [
        item
        for item in summaries
        if item["scope"] == scope and item["method"] == primary
    ]
    raw = next(item for item in primary_rows if item["correction"] == "none")
    rows = [
        {
            "correction": "none",
            "rank": 0,
            "aggregate_error_percent": 100 * float(raw["aggregate_relative_av_l2"]),
            "worst_error_percent": 100 * float(raw["worst_head_relative_av_l2"]),
            "scope": scope,
        }
    ]
    for item in primary_rows:
        if item["correction"] == "none":
            continue
        rows.append(
            {
                "correction": item["correction"],
                "rank": int(item["rank"]),
                "aggregate_error_percent": 100 * float(item["aggregate_relative_av_l2"]),
                "worst_error_percent": 100 * float(item["worst_head_relative_av_l2"]),
                "scope": scope,
            }
        )
    write_csv(output_dir / "primary_rank_transfer.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for correction in (
        "adaptive_rank_oracle",
        "frozen_calibration_basis_oracle_coefficients",
    ):
        correction_rows = [row for row in rows if row["correction"] == correction]
        if not correction_rows:
            continue
        line = [rows[0]] + sorted(
            correction_rows,
            key=lambda row: row["rank"],
        )
        label = "adaptive per-record oracle" if correction.startswith("adaptive") else "calibration-frozen basis"
        color = COLORS[correction]
        axes[0].plot(
            [row["rank"] for row in line],
            [row["aggregate_error_percent"] for row in line],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].plot(
            [row["rank"] for row in line],
            [row["worst_error_percent"] for row in line],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].axhline(0.5, color="#666666", linestyle=":")
    axes[1].axhline(1.0, color="#666666", linestyle=":")
    scope_label = scope.replace("_", " ").title()
    axes[0].set_ylabel(f"{scope_label} aggregate AV error (%)")
    axes[1].set_ylabel(f"{scope_label} worst-head AV error (%)")
    for ax in axes:
        ax.set_xlabel("Residual rank")
        ax.set_xticks(sorted({row["rank"] for row in rows}))
        ax.legend(fontsize=8)
    save_figure(fig, output_dir, "primary_rank_transfer")


def main() -> None:
    args = parse_args()
    required = [
        args.result_dir / "metrics.csv",
        args.result_dir / "summary.json",
        args.result_dir / "gate_decision.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing result artifacts: " + ", ".join(map(str, missing)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty figure directory {args.output_dir}")

    summary_payload = json.loads(required[1].read_text(encoding="utf-8"))
    gate = json.loads(required[2].read_text(encoding="utf-8"))
    summaries = summary_payload["summaries"]
    metrics = read_csv(required[0])
    primary = gate["primary_method"]
    primary_rank = int(gate["primary_rank"])
    configure_style()
    plot_quality_cost(summaries, primary, primary_rank, args.output_dir, args.scope)
    plot_phase_alignment(summaries, primary_rank, args.output_dir, args.scope)
    plot_primary_heatmap(metrics, primary, primary_rank, args.output_dir, args.scope)
    plot_rank_transfer(summaries, primary, args.output_dir, args.scope)
    print(f"[figures] wrote bound CSV/PNG/PDF artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
