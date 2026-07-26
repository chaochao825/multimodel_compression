#!/usr/bin/env python3
"""Plot checkpoint-faithful Wan FFN exact-path H200 results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PATH_ORDER = (
    "cuda_graph_eager_static",
    "compile_default",
    "compile_reduce-overhead",
    "compile_max-autotune",
)
PATH_LABELS = {
    "cuda_graph_eager_static": "CUDA Graph\n(eager static)",
    "compile_default": "compile\ndefault",
    "compile_reduce-overhead": "compile\nreduce-overhead",
    "compile_max-autotune": "compile\nmax-autotune",
}
CASE_ORDER = ("F17", "F81")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("FFN exact-path CSV is empty")
    return rows


def harmonic_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return math.nan
    return len(values) / sum(1.0 / value for value in values)


def build_plot_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        case = row.get("case", "")
        path = row.get("path", "")
        if case in CASE_ORDER and path in PATH_ORDER and row.get("status") == "ok":
            groups[(case, path)].append(row)

    output: list[dict[str, object]] = []
    for case in CASE_ORDER:
        for path in PATH_ORDER:
            group = groups.get((case, path), [])
            if not group:
                raise ValueError(f"missing result group: {case}/{path}")
            median_speedups = [float(row["median_speedup"]) for row in group]
            p95_speedups = [float(row["p95_speedup"]) for row in group]
            amortized_speedups = [float(row["amortized_speedup"]) for row in group]
            setup_ms = [float(row["setup_ms"]) for row in group]
            relative_l2 = [float(row["relative_l2"]) for row in group]
            output.append(
                {
                    "case": case,
                    "path": path,
                    "layers": len(group),
                    "median_speedup_hmean": harmonic_mean(median_speedups),
                    "median_speedup_min": min(median_speedups),
                    "median_speedup_max": max(median_speedups),
                    "p95_speedup_min": min(p95_speedups),
                    "amortized_speedup_min": min(amortized_speedups),
                    "amortized_speedup_hmean": harmonic_mean(amortized_speedups),
                    "setup_ms_median": sorted(setup_ms)[len(setup_ms) // 2],
                    "setup_ms_max": max(setup_ms),
                    "relative_l2_max": max(relative_l2),
                    "bitwise_exact_all": all(
                        row.get("bitwise_equal", "").lower() == "true" for row in group
                    ),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def annotate_heatmap(axis: plt.Axes, values: list[list[float]], fmt: str) -> None:
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#111111",
            )


def speedup_matrix(
    plot_rows: list[dict[str, object]], key: str
) -> list[list[float]]:
    lookup = {(str(row["case"]), str(row["path"])): row for row in plot_rows}
    return [
        [float(lookup[(case, path)][key]) for case in CASE_ORDER]
        for path in PATH_ORDER
    ]


def plot(plot_rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.4))
    norm = TwoSlopeNorm(vmin=0.84, vcenter=1.0, vmax=1.10)

    steady = speedup_matrix(plot_rows, "median_speedup_hmean")
    axes[0].imshow(steady, cmap="RdYlGn", norm=norm, aspect="auto")
    annotate_heatmap(axes[0], steady, ".3f")
    axes[0].set_xticks(range(len(CASE_ORDER)), CASE_ORDER)
    axes[0].set_yticks(range(len(PATH_ORDER)), [PATH_LABELS[path] for path in PATH_ORDER])
    axes[0].set_title("(a) Steady-state median speedup")
    axes[0].set_xlabel("Token shape")

    amortized = speedup_matrix(plot_rows, "amortized_speedup_hmean")
    axes[1].imshow(amortized, cmap="RdYlGn", norm=norm, aspect="auto")
    annotate_heatmap(axes[1], amortized, ".3f")
    axes[1].set_xticks(range(len(CASE_ORDER)), CASE_ORDER)
    axes[1].set_yticks(range(len(PATH_ORDER)), [PATH_LABELS[path] for path in PATH_ORDER])
    axes[1].set_title("(b) 40-call amortized speedup")
    axes[1].set_xlabel("Token shape")

    colors = {"F17": "#0077BB", "F81": "#EE7733"}
    markers = {
        "cuda_graph_eager_static": "o",
        "compile_default": "s",
        "compile_reduce-overhead": "^",
        "compile_max-autotune": "D",
    }
    for row in plot_rows:
        case = str(row["case"])
        path = str(row["path"])
        error_percent = 100.0 * float(row["relative_l2_max"])
        setup_ms = float(row["setup_ms_median"])
        axes[2].scatter(
            setup_ms,
            error_percent,
            color=colors[case],
            marker=markers[path],
            s=58,
            edgecolor="#222222",
            linewidth=0.5,
        )
    for case, color in colors.items():
        axes[2].scatter([], [], color=color, label=case, s=42)
    for path, marker in markers.items():
        axes[2].scatter(
            [], [], color="#777777", marker=marker, label=PATH_LABELS[path].replace("\n", " "), s=42
        )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Median setup/capture cost (ms, log scale)")
    axes[2].set_ylabel("Worst relative L2 error across layers (%)")
    axes[2].set_title("(c) Setup cost and numerical fidelity")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=6.4, frameon=False, ncol=2, loc="upper left")

    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"ffn_exact_paths.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plot_rows = build_plot_rows(read_rows(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ffn_exact_plot_data.csv", plot_rows)
    plot(plot_rows, args.output_dir)


if __name__ == "__main__":
    main()
