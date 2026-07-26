#!/usr/bin/env python3
"""Summarize cross-run stability of attention head classes and features."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES = (
    "actual_normalized_entropy_mean",
    "geometry_mass_mean",
    "actual_top64_mass_mean",
    "actual_participation_support_fraction_mean",
)
CLASS_ORDER = ("localized", "transitional", "diffuse")


def parse_labeled_path(text: str) -> tuple[str, Path]:
    label, separator, raw_path = text.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-csv", type=parse_labeled_path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entropy-correlation-gate", type=float, default=0.90)
    parser.add_argument("--geometry-correlation-gate", type=float, default=0.90)
    parser.add_argument("--class-agreement-gate", type=float, default=0.75)
    parser.add_argument("--localized-jaccard-gate", type=float, default=0.50)
    return parser.parse_args()


def read_heads(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: dict[int, dict[str, float]] = {}
    for row in rows:
        head = int(row["head"])
        if head in parsed:
            raise ValueError(f"duplicate head {head} in {path}")
        parsed[head] = {feature: float(row[feature]) for feature in FEATURES}
    if not parsed:
        raise ValueError(f"head CSV is empty: {path}")
    return parsed


def classify_head(row: dict[str, float]) -> str:
    entropy = row["actual_normalized_entropy_mean"]
    geometry = row["geometry_mass_mean"]
    if entropy <= 0.55 and geometry >= 0.80:
        return "localized"
    if entropy <= 0.80 and geometry >= 0.50:
        return "transitional"
    return "diffuse"


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson correlation requires equal vectors of length >= 2")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0.0:
        return 1.0 if left == right else 0.0
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def summarize_pair(
    left_label: str,
    left: dict[int, dict[str, float]],
    right_label: str,
    right: dict[int, dict[str, float]],
    *,
    entropy_correlation_gate: float,
    geometry_correlation_gate: float,
    class_agreement_gate: float,
    localized_jaccard_gate: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if set(left) != set(right):
        raise ValueError("head IDs differ across runs")
    heads = sorted(left)
    left_classes = {head: classify_head(left[head]) for head in heads}
    right_classes = {head: classify_head(right[head]) for head in heads}
    correlations = {
        feature: pearson(
            [left[head][feature] for head in heads],
            [right[head][feature] for head in heads],
        )
        for feature in FEATURES
    }
    agreement = sum(
        left_classes[head] == right_classes[head] for head in heads
    ) / len(heads)
    localized_overlap = jaccard(
        {head for head in heads if left_classes[head] == "localized"},
        {head for head in heads if right_classes[head] == "localized"},
    )
    decision = (
        correlations["actual_normalized_entropy_mean"]
        >= entropy_correlation_gate
        and correlations["geometry_mass_mean"] >= geometry_correlation_gate
        and agreement >= class_agreement_gate
        and localized_overlap >= localized_jaccard_gate
    )
    pair = {
        "left": left_label,
        "right": right_label,
        "heads": len(heads),
        "entropy_correlation": correlations["actual_normalized_entropy_mean"],
        "geometry_correlation": correlations["geometry_mass_mean"],
        "top64_mass_correlation": correlations["actual_top64_mass_mean"],
        "participation_correlation": correlations[
            "actual_participation_support_fraction_mean"
        ],
        "class_agreement": agreement,
        "localized_jaccard": localized_overlap,
        "router_class_pilot_go": decision,
    }
    head_rows = []
    for head in heads:
        row: dict[str, object] = {
            "left": left_label,
            "right": right_label,
            "head": head,
            "left_class": left_classes[head],
            "right_class": right_classes[head],
            "class_equal": left_classes[head] == right_classes[head],
        }
        for feature in FEATURES:
            row[f"left_{feature}"] = left[head][feature]
            row[f"right_{feature}"] = right[head][feature]
            row[f"abs_delta_{feature}"] = abs(left[head][feature] - right[head][feature])
        head_rows.append(row)
    return pair, head_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def plot_pair(
    output_dir: Path,
    pair: dict[str, object],
    head_rows: list[dict[str, object]],
) -> None:
    colors = {
        "localized": "#CC3311",
        "transitional": "#EE7733",
        "diffuse": "#0077BB",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    left_label = str(pair["left"])
    right_label = str(pair["right"])
    for axis, feature, title in (
        (axes[0], "actual_normalized_entropy_mean", "(a) Normalized entropy"),
        (axes[1], "geometry_mass_mean", "(b) Temporal-tile mass"),
    ):
        left_values = [float(row[f"left_{feature}"]) for row in head_rows]
        right_values = [float(row[f"right_{feature}"]) for row in head_rows]
        lower = min(left_values + right_values)
        upper = max(left_values + right_values)
        margin = max((upper - lower) * 0.08, 0.01)
        axis.plot(
            [lower - margin, upper + margin],
            [lower - margin, upper + margin],
            color="#555555",
            linestyle=":",
            linewidth=1.0,
        )
        for row, left_value, right_value in zip(head_rows, left_values, right_values):
            head_class = str(row["left_class"])
            axis.scatter(left_value, right_value, color=colors[head_class], s=34)
            axis.annotate(
                str(row["head"]),
                (left_value, right_value),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_xlabel(left_label)
        axis.set_ylabel(right_label)
        axis.set_title(title)
        axis.grid(alpha=0.2)

    class_codes = {name: index for index, name in enumerate(CLASS_ORDER)}
    matrix = [
        [class_codes[str(row["left_class"])] for row in head_rows],
        [class_codes[str(row["right_class"])] for row in head_rows],
    ]
    axes[2].imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdYlBu", vmin=0, vmax=2)
    axes[2].set_yticks([0, 1], [left_label, right_label])
    axes[2].set_xticks(range(len(head_rows)), [str(row["head"]) for row in head_rows])
    axes[2].set_xlabel("Head index")
    axes[2].set_title("(c) Rule-based head class")
    axes[2].set_xticks([index - 0.5 for index in range(1, len(head_rows))], minor=True)
    axes[2].grid(which="minor", axis="x", color="white", linewidth=0.5)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"head_class_stability.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if len(args.head_csv) < 2:
        raise ValueError("provide at least two labeled head CSV files")
    labels = [label for label, _ in args.head_csv]
    if len(set(labels)) != len(labels):
        raise ValueError("head CSV labels must be unique")
    runs = {label: read_heads(path) for label, path in args.head_csv}
    pair_rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []
    for left_label, right_label in itertools.combinations(labels, 2):
        pair, heads = summarize_pair(
            left_label,
            runs[left_label],
            right_label,
            runs[right_label],
            entropy_correlation_gate=args.entropy_correlation_gate,
            geometry_correlation_gate=args.geometry_correlation_gate,
            class_agreement_gate=args.class_agreement_gate,
            localized_jaccard_gate=args.localized_jaccard_gate,
        )
        pair_rows.append(pair)
        head_rows.extend(heads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "head_stability_pairs.csv", pair_rows)
    write_csv(args.output_dir / "head_stability_heads.csv", head_rows)
    payload = {
        "scope": "cross-seed head-class pilot on sampled attention probabilities",
        "thresholds": {
            "localized": "entropy <= 0.55 and geometry mass >= 0.80",
            "transitional": "entropy <= 0.80 and geometry mass >= 0.50",
            "entropy_correlation_gate": args.entropy_correlation_gate,
            "geometry_correlation_gate": args.geometry_correlation_gate,
            "class_agreement_gate": args.class_agreement_gate,
            "localized_jaccard_gate": args.localized_jaccard_gate,
        },
        "pairs": pair_rows,
        "all_pairs_go": all(bool(row["router_class_pilot_go"]) for row in pair_rows),
        "warning": (
            "A two-seed layer-0/step-0 result is a seed-stability screen only. "
            "Prompt, deeper-layer, and later-step stability remain untested."
        ),
    }
    write_json(args.output_dir / "head_stability_summary.json", payload)
    if len(pair_rows) == 1:
        plot_pair(args.output_dir, pair_rows[0], head_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
