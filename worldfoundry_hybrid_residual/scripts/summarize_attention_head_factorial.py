#!/usr/bin/env python3
"""Summarize prompt, seed, step, and CFG-branch stability of attention head roles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_attention_head_stability import read_heads, summarize_pair


COMPARISON_ORDER = ("seed", "prompt", "mixed_sample", "step", "branch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entropy-correlation-gate", type=float, default=0.90)
    parser.add_argument("--geometry-correlation-gate", type=float, default=0.90)
    parser.add_argument("--class-agreement-gate", type=float, default=0.75)
    parser.add_argument("--localized-jaccard-gate", type=float, default=0.50)
    return parser.parse_args()


def read_index(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError("head-stat index is empty")
    rows: list[dict[str, object]] = []
    labels: set[str] = set()
    for row in raw:
        label = row["label"]
        if label in labels:
            raise ValueError(f"duplicate label in head-stat index: {label}")
        labels.add(label)
        rows.append(
            {
                **row,
                "prompt_index": int(row["prompt_index"]),
                "seed": int(row["seed"]),
                "sampling_step": int(row["sampling_step"]),
                "layer": int(row["layer"]),
                "head_csv": str(Path(row["head_csv"]).resolve()),
            }
        )
    return rows


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def pair_type(left: dict[str, object], right: dict[str, object]) -> str:
    same_prompt = left["prompt_index"] == right["prompt_index"]
    same_seed = left["seed"] == right["seed"]
    if same_prompt and not same_seed:
        return "seed"
    if same_seed and not same_prompt:
        return "prompt"
    return "mixed_sample"


def add_comparison(
    output: list[dict[str, object]],
    comparison_type: str,
    left: dict[str, object],
    right: dict[str, object],
    runs: dict[str, dict[int, dict[str, float]]],
    gates: dict[str, float],
) -> None:
    pair, _ = summarize_pair(
        str(left["label"]),
        runs[str(left["label"])],
        str(right["label"]),
        runs[str(right["label"])],
        entropy_correlation_gate=gates["entropy"],
        geometry_correlation_gate=gates["geometry"],
        class_agreement_gate=gates["class"],
        localized_jaccard_gate=gates["jaccard"],
    )
    output.append(
        {
            "comparison_type": comparison_type,
            "left": left["label"],
            "right": right["label"],
            "layer": left["layer"],
            "left_step": left["sampling_step"],
            "right_step": right["sampling_step"],
            "left_branch": left["branch"],
            "right_branch": right["branch"],
            "left_prompt_index": left["prompt_index"],
            "right_prompt_index": right["prompt_index"],
            "left_seed": left["seed"],
            "right_seed": right["seed"],
            "entropy_correlation": pair["entropy_correlation"],
            "geometry_correlation": pair["geometry_correlation"],
            "top64_mass_correlation": pair["top64_mass_correlation"],
            "participation_correlation": pair["participation_correlation"],
            "class_agreement": pair["class_agreement"],
            "localized_jaccard": pair["localized_jaccard"],
            "router_class_pilot_go": pair["router_class_pilot_go"],
        }
    )


def build_comparisons(
    index_rows: list[dict[str, object]],
    runs: dict[str, dict[int, dict[str, float]]],
    gates: dict[str, float],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    sample_groups: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in index_rows:
        sample_groups[(int(row["sampling_step"]), str(row["branch"]), int(row["layer"]))].append(row)
    for group in sample_groups.values():
        ordered = sorted(group, key=lambda row: (int(row["prompt_index"]), int(row["seed"])))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                add_comparison(output, pair_type(left, right), left, right, runs, gates)

    temporal_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in index_rows:
        temporal_groups[(str(row["sample_id"]), str(row["branch"]), int(row["layer"]))].append(row)
    for group in temporal_groups.values():
        ordered = sorted(group, key=lambda row: int(row["sampling_step"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                add_comparison(output, "step", left, right, runs, gates)

    branch_groups: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in index_rows:
        branch_groups[(str(row["sample_id"]), int(row["sampling_step"]), int(row["layer"]))].append(row)
    for group in branch_groups.values():
        if len(group) != 2:
            raise ValueError("each sample/step/layer cell must contain two CFG branches")
        left, right = sorted(group, key=lambda row: str(row["branch"]))
        add_comparison(output, "branch", left, right, runs, gates)
    return output


def summarize_types(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["comparison_type"])].append(row)
    output = []
    for comparison_type in COMPARISON_ORDER:
        group = groups.get(comparison_type, [])
        if not group:
            continue
        metrics = {
            name: [float(row[name]) for row in group]
            for name in (
                "entropy_correlation",
                "geometry_correlation",
                "class_agreement",
                "localized_jaccard",
            )
        }
        output.append(
            {
                "comparison_type": comparison_type,
                "pairs": len(group),
                "go_fraction": sum(bool(row["router_class_pilot_go"]) for row in group) / len(group),
                **{
                    f"{name}_{suffix}": statistic(values)
                    for name, values in metrics.items()
                    for suffix, statistic in (
                        ("min", min),
                        ("p05", lambda items: quantile(items, 0.05)),
                        ("median", lambda items: quantile(items, 0.50)),
                    )
                },
            }
        )
    return output


def summarize_layers(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["comparison_type"]), int(row["layer"]))].append(row)
    output = []
    for comparison_type in COMPARISON_ORDER:
        layers = sorted(layer for kind, layer in groups if kind == comparison_type)
        for layer in layers:
            group = groups[(comparison_type, layer)]
            metrics = {
                name: [float(row[name]) for row in group]
                for name in (
                    "entropy_correlation",
                    "geometry_correlation",
                    "class_agreement",
                    "localized_jaccard",
                )
            }
            output.append(
                {
                    "comparison_type": comparison_type,
                    "layer": layer,
                    "pairs": len(group),
                    "go_fraction": sum(
                        bool(row["router_class_pilot_go"]) for row in group
                    )
                    / len(group),
                    **{
                        f"{name}_{suffix}": statistic(values)
                        for name, values in metrics.items()
                        for suffix, statistic in (
                            ("min", min),
                            ("p05", lambda items: quantile(items, 0.05)),
                            ("median", lambda items: quantile(items, 0.50)),
                        )
                    },
                }
            )
    return output


def summarize_step_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["comparison_type"] != "step":
            continue
        groups[
            (int(row["layer"]), int(row["left_step"]), int(row["right_step"]))
        ].append(row)
    output = []
    for (layer, left_step, right_step), group in sorted(groups.items()):
        output.append(
            {
                "layer": layer,
                "left_step": left_step,
                "right_step": right_step,
                "pairs": len(group),
                "go_fraction": sum(
                    bool(row["router_class_pilot_go"]) for row in group
                )
                / len(group),
                "entropy_correlation_min": min(
                    float(row["entropy_correlation"]) for row in group
                ),
                "entropy_correlation_median": quantile(
                    [float(row["entropy_correlation"]) for row in group], 0.50
                ),
                "geometry_correlation_min": min(
                    float(row["geometry_correlation"]) for row in group
                ),
                "geometry_correlation_median": quantile(
                    [float(row["geometry_correlation"]) for row in group], 0.50
                ),
                "class_agreement_min": min(
                    float(row["class_agreement"]) for row in group
                ),
                "class_agreement_median": quantile(
                    [float(row["class_agreement"]) for row in group], 0.50
                ),
                "localized_jaccard_min": min(
                    float(row["localized_jaccard"]) for row in group
                ),
                "localized_jaccard_median": quantile(
                    [float(row["localized_jaccard"]) for row in group], 0.50
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cell_minimum(
    rows: list[dict[str, object]], metric: str, layers: list[int], steps: list[int]
) -> list[list[float]]:
    values: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["comparison_type"] in {"seed", "prompt", "mixed_sample"}:
            values[(int(row["layer"]), int(row["left_step"]))].append(float(row[metric]))
    return [[min(values[(layer, step)]) for step in steps] for layer in layers]


def annotate(axis: plt.Axes, matrix: list[list[float]]) -> None:
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)


def plot_dashboard(
    comparisons: list[dict[str, object]],
    type_summary: list[dict[str, object]],
    output_dir: Path,
) -> None:
    layers = sorted({int(row["layer"]) for row in comparisons})
    steps = sorted(
        {int(row["left_step"]) for row in comparisons if row["comparison_type"] != "step"}
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    for axis, metric, title in (
        (axes[0, 0], "class_agreement", "(a) Worst sample class agreement"),
        (axes[0, 1], "entropy_correlation", "(b) Worst sample entropy correlation"),
        (axes[1, 0], "geometry_correlation", "(c) Worst sample geometry correlation"),
    ):
        matrix = cell_minimum(comparisons, metric, layers, steps)
        axis.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        annotate(axis, matrix)
        axis.set_xticks(range(len(steps)), [str(step) for step in steps])
        axis.set_yticks(range(len(layers)), [str(layer) for layer in layers])
        axis.set_xlabel("Sampling step")
        axis.set_ylabel("Wan layer")
        axis.set_title(title)

    labels = [str(row["comparison_type"]) for row in type_summary]
    positions = list(range(len(labels)))
    go = [float(row["go_fraction"]) for row in type_summary]
    class_p05 = [float(row["class_agreement_p05"]) for row in type_summary]
    width = 0.36
    axes[1, 1].bar([value - width / 2 for value in positions], go, width, label="GO fraction", color="#0077BB")
    axes[1, 1].bar(
        [value + width / 2 for value in positions],
        class_p05,
        width,
        label="class agreement P05",
        color="#EE7733",
    )
    axes[1, 1].axhline(0.75, color="#333333", linestyle=":", linewidth=1.0)
    axes[1, 1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].set_title("(d) Factor-specific stability gates")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)

    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"attention_head_factorial.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    index_rows = read_index(args.index)
    runs = {
        str(row["label"]): read_heads(Path(str(row["head_csv"]))) for row in index_rows
    }
    gates = {
        "entropy": args.entropy_correlation_gate,
        "geometry": args.geometry_correlation_gate,
        "class": args.class_agreement_gate,
        "jaccard": args.localized_jaccard_gate,
    }
    comparisons = build_comparisons(index_rows, runs, gates)
    type_summary = summarize_types(comparisons)
    layer_summary = summarize_layers(comparisons)
    step_pair_summary = summarize_step_pairs(comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "attention_head_factorial_pairs.csv", comparisons)
    write_csv(args.output_dir / "attention_head_factorial_summary.csv", type_summary)
    write_csv(args.output_dir / "attention_head_factorial_layer_summary.csv", layer_summary)
    write_csv(
        args.output_dir / "attention_head_factorial_step_pair_summary.csv",
        step_pair_summary,
    )
    summary_by_type = {str(row["comparison_type"]): row for row in type_summary}

    def passes(name: str) -> bool:
        row = summary_by_type.get(name)
        return bool(row) and float(row["go_fraction"]) >= 0.95 and float(row["class_agreement_p05"]) >= gates["class"]

    layer_summary_by_key = {
        (str(row["comparison_type"]), int(row["layer"])): row
        for row in layer_summary
    }

    def layer_passes(name: str, layer: int) -> bool:
        row = layer_summary_by_key.get((name, layer))
        return bool(row) and float(row["go_fraction"]) >= 0.95 and float(
            row["class_agreement_p05"]
        ) >= gates["class"]

    layer_decisions = {
        str(layer): {
            "fixed_layer_step_router_go": layer_passes("seed", layer)
            and layer_passes("prompt", layer),
            "step_agnostic_router_go": layer_passes("step", layer),
            "branch_shared_router_go": layer_passes("branch", layer),
        }
        for layer in sorted({int(row["layer"]) for row in index_rows})
    }

    payload = {
        "scope": "F81 attention head-role prompt/seed/step/CFG factorial gate",
        "runs": len(index_rows),
        "comparisons": len(comparisons),
        "gates": gates,
        "type_summary": type_summary,
        "layer_summary": layer_summary,
        "step_pair_summary": step_pair_summary,
        "layer_decisions": layer_decisions,
        "decisions": {
            "fixed_layer_step_router_go": passes("seed") and passes("prompt"),
            "step_agnostic_router_go": passes("step"),
            "branch_shared_router_go": passes("branch"),
        },
        "interpretation": (
            "Comparisons never equate head indices across different layers. A GO supports "
            "using stable role statistics to choose an operator; it does not validate a fixed "
            "token mask, a frozen correction basis, final video quality, or kernel speedup."
        ),
    }
    (args.output_dir / "attention_head_factorial_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_dashboard(comparisons, type_summary, args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
