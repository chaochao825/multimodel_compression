from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-selector", default="exact_recent")
    parser.add_argument(
        "--candidate-selector",
        default="learned_recent_query_topk",
    )
    parser.add_argument("--reference-variant", default="full")
    parser.add_argument("--candidate-variant", default="pca_r256_s4")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    output = {}
    for row in rows:
        key = (
            row["sample_id"],
            row["selection_policy"],
            row["memory_variant"],
        )
        if key in output:
            raise ValueError(f"duplicate prediction row: {key}")
        output[key] = row
    return output


def transition(reference_correct: int, candidate_correct: int) -> str:
    if reference_correct == candidate_correct:
        return "stable_correct" if reference_correct else "stable_wrong"
    return "better" if candidate_correct else "worse"


def rank_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable += 1.0
            elif positive == negative:
                favorable += 0.5
    return favorable / (len(positives) * len(negatives))


def format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def unique_in_order(values: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(values))


def comparison_events(
    rows: list[dict[str, str]],
    *,
    selector: str,
    reference_variant: str,
    candidate_variant: str,
) -> list[dict[str, object]]:
    indexed = index_rows(rows)
    sample_ids = sorted(
        row["sample_id"]
        for row in rows
        if row["selection_policy"] == selector
        and row["memory_variant"] == reference_variant
    )
    output = []
    for sample_id in sample_ids:
        reference = indexed[(sample_id, selector, reference_variant)]
        candidate = indexed[(sample_id, selector, candidate_variant)]
        changed = int(
            reference["predicted_index"] != candidate["predicted_index"]
        )
        output.append(
            {
                "sample_id": sample_id,
                "task": reference["task"],
                "selector": selector,
                "reference_variant": reference_variant,
                "candidate_variant": candidate_variant,
                "transition": transition(
                    int(reference["correct"]),
                    int(candidate["correct"]),
                ),
                "prediction_changed": changed,
                "reference_correct": int(reference["correct"]),
                "candidate_correct": int(candidate["correct"]),
                "reference_prediction": reference["prediction"],
                "candidate_prediction": candidate["prediction"],
                "answer": reference["answer"],
                "question": reference["question"],
                "frame_indices": reference["frame_indices"],
                "pool_reconstruction_error": float(
                    candidate["pool_reconstruction_relative_error"]
                ),
                "selected_reconstruction_error": float(
                    candidate["selected_reconstruction_relative_error"]
                ),
            }
        )
    return output


def selector_events(
    rows: list[dict[str, str]],
    *,
    reference_selector: str,
    candidate_selector: str,
    variant: str,
) -> list[dict[str, object]]:
    indexed = index_rows(rows)
    sample_ids = sorted(
        row["sample_id"]
        for row in rows
        if row["selection_policy"] == reference_selector
        and row["memory_variant"] == variant
    )
    output = []
    for sample_id in sample_ids:
        reference = indexed[(sample_id, reference_selector, variant)]
        candidate = indexed[(sample_id, candidate_selector, variant)]
        output.append(
            {
                "sample_id": sample_id,
                "task": reference["task"],
                "memory_variant": variant,
                "transition": transition(
                    int(reference["correct"]),
                    int(candidate["correct"]),
                ),
                "prediction_changed": int(
                    reference["predicted_index"]
                    != candidate["predicted_index"]
                ),
                "reference_correct": int(reference["correct"]),
                "candidate_correct": int(candidate["correct"]),
                "reference_prediction": reference["prediction"],
                "candidate_prediction": candidate["prediction"],
                "answer": reference["answer"],
                "question": reference["question"],
                "reference_frames": reference["frame_indices"],
                "candidate_frames": candidate["frame_indices"],
            }
        )
    return output


def accuracy_by_task(
    rows: list[dict[str, str]],
    *,
    selector: str,
    variant: str,
) -> dict[str, tuple[int, int]]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if (
            row["selection_policy"] != selector
            or row["memory_variant"] != variant
        ):
            continue
        totals[row["task"]][0] += int(row["correct"])
        totals[row["task"]][1] += 1
    return {task: (values[0], values[1]) for task, values in totals.items()}


def top_error_capture(
    events: list[dict[str, object]],
    *,
    fraction: float,
) -> dict[str, float | int]:
    ordered = sorted(
        events,
        key=lambda row: float(row["selected_reconstruction_error"]),
        reverse=True,
    )
    retained = max(1, math.ceil(len(ordered) * fraction))
    top = ordered[:retained]
    changed_total = sum(int(row["prediction_changed"]) for row in ordered)
    worse_total = sum(row["transition"] == "worse" for row in ordered)
    changed_top = sum(int(row["prediction_changed"]) for row in top)
    worse_top = sum(row["transition"] == "worse" for row in top)
    return {
        "fraction": fraction,
        "samples": retained,
        "changed_capture": (
            changed_top / changed_total if changed_total else 0.0
        ),
        "worse_capture": worse_top / worse_total if worse_total else 0.0,
    }


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: object, out_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"{name}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )


def plot_task_stages(
    task_rows: list[dict[str, object]],
    out_dir: Path,
    *,
    reference_selector: str,
    candidate_selector: str,
    reference_variant: str,
    candidate_variant: str,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    stages = [
        (
            f"{reference_selector} / {reference_variant}",
            "exact_full",
            "#4C78A8",
        ),
        (
            f"{candidate_selector} / {reference_variant}",
            "learned_full",
            "#F58518",
        ),
        (
            f"{candidate_selector} / {candidate_variant}",
            "learned_compressed",
            "#54A24B",
        ),
    ]
    tasks = [str(row["task"]) for row in task_rows]
    positions = list(range(len(tasks)))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    for offset, (label, key, color) in enumerate(stages):
        values = [float(row[key]) * 100.0 for row in task_rows]
        axis.bar(
            [position + (offset - 1) * width for position in positions],
            values,
            width=width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, [task.replace("_", "\n") for task in tasks])
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0.0, 80.0)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    save_figure(figure, out_dir, "task_stage_decomposition")
    plt.close(figure)


def plot_reconstruction_risk(
    compression_events: list[dict[str, object]],
    out_dir: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    colors = {
        "stable_correct": "#4C78A8",
        "stable_wrong": "#BAB0AC",
        "better": "#54A24B",
        "worse": "#E45756",
    }
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    for axis, selector in zip(
        axes,
        sorted({str(row["selector"]) for row in compression_events}),
    ):
        subset = [row for row in compression_events if row["selector"] == selector]
        for transition_name in colors:
            values = [
                float(row["selected_reconstruction_error"])
                for row in subset
                if row["transition"] == transition_name
            ]
            if not values:
                continue
            axis.scatter(
                values,
                [transition_name] * len(values),
                s=18,
                alpha=0.7,
                color=colors[transition_name],
                edgecolors="none",
            )
        axis.set_xlabel("Selected-feature relative error")
        axis.set_title(selector.replace("_", " "), fontsize=10)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_ylabel("Paired outcome vs full cache")
    figure.tight_layout()
    save_figure(figure, out_dir, "reconstruction_error_outcomes")
    plt.close(figure)


def overall_accuracy(
    rows: list[dict[str, str]],
    *,
    selector: str,
    variant: str,
) -> tuple[int, int]:
    subset = [
        row
        for row in rows
        if row["selection_policy"] == selector
        and row["memory_variant"] == variant
    ]
    return sum(int(row["correct"]) for row in subset), len(subset)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    compression_events = []
    for selector in unique_in_order(
        (args.reference_selector, args.candidate_selector)
    ):
        compression_events.extend(
            comparison_events(
                rows,
                selector=selector,
                reference_variant=args.reference_variant,
                candidate_variant=args.candidate_variant,
            )
        )
    selector_full = selector_events(
        rows,
        reference_selector=args.reference_selector,
        candidate_selector=args.candidate_selector,
        variant=args.reference_variant,
    )
    selector_compressed = selector_events(
        rows,
        reference_selector=args.reference_selector,
        candidate_selector=args.candidate_selector,
        variant=args.candidate_variant,
    )

    task_exact = accuracy_by_task(
        rows,
        selector=args.reference_selector,
        variant=args.reference_variant,
    )
    task_learned = accuracy_by_task(
        rows,
        selector=args.candidate_selector,
        variant=args.reference_variant,
    )
    task_compressed = accuracy_by_task(
        rows,
        selector=args.candidate_selector,
        variant=args.candidate_variant,
    )
    task_rows = []
    for task in sorted(task_exact):
        exact_correct, total = task_exact[task]
        learned_correct, learned_total = task_learned[task]
        compressed_correct, compressed_total = task_compressed[task]
        if len({total, learned_total, compressed_total}) != 1:
            raise ValueError(f"task sample counts differ for {task}")
        task_rows.append(
            {
                "task": task,
                "samples": total,
                "exact_full": exact_correct / total,
                "learned_full": learned_correct / total,
                "learned_compressed": compressed_correct / total,
                "selector_gain_points": 100.0
                * (learned_correct - exact_correct)
                / total,
                "compression_delta_points": 100.0
                * (compressed_correct - learned_correct)
                / total,
            }
        )

    learned_compression = [
        row
        for row in compression_events
        if row["selector"] == args.candidate_selector
    ]
    scores = [
        float(row["selected_reconstruction_error"])
        for row in learned_compression
    ]
    changed_labels = [
        int(row["prediction_changed"]) for row in learned_compression
    ]
    worse_labels = [
        int(row["transition"] == "worse") for row in learned_compression
    ]
    exact_correct, sample_count = overall_accuracy(
        rows,
        selector=args.reference_selector,
        variant=args.reference_variant,
    )
    learned_correct, learned_count = overall_accuracy(
        rows,
        selector=args.candidate_selector,
        variant=args.reference_variant,
    )
    compressed_correct, compressed_count = overall_accuracy(
        rows,
        selector=args.candidate_selector,
        variant=args.candidate_variant,
    )
    if len({sample_count, learned_count, compressed_count}) != 1:
        raise ValueError("overall sample counts differ")

    summary = {
        "reference_selector": args.reference_selector,
        "candidate_selector": args.candidate_selector,
        "reference_variant": args.reference_variant,
        "candidate_variant": args.candidate_variant,
        "samples": sample_count,
        "exact_full_correct": exact_correct,
        "learned_full_correct": learned_correct,
        "learned_compressed_correct": compressed_correct,
        "exact_full_accuracy": exact_correct / sample_count,
        "learned_full_accuracy": learned_correct / sample_count,
        "learned_compressed_accuracy": compressed_correct / sample_count,
        "selector_net_gain_samples": learned_correct - exact_correct,
        "compression_net_delta_samples": compressed_correct - learned_correct,
        "learned_compression_prediction_changes": sum(changed_labels),
        "learned_compression_worse_samples": sum(worse_labels),
        "reconstruction_error_auc_for_prediction_change": rank_auc(
            scores,
            changed_labels,
        ),
        "reconstruction_error_auc_for_worse_event": rank_auc(
            scores,
            worse_labels,
        ),
        "top_error_capture": [
            top_error_capture(learned_compression, fraction=fraction)
            for fraction in (0.05, 0.10, 0.20)
        ],
        "selector_full_transitions": {
            name: sum(row["transition"] == name for row in selector_full)
            for name in ("better", "worse", "stable_correct", "stable_wrong")
        },
        "selector_compressed_transitions": {
            name: sum(
                row["transition"] == name for row in selector_compressed
            )
            for name in ("better", "worse", "stable_correct", "stable_wrong")
        },
    }

    write_csv(args.out_dir / "compression_pair_events.csv", compression_events)
    write_csv(args.out_dir / "selector_full_pair_events.csv", selector_full)
    write_csv(
        args.out_dir / "selector_compressed_pair_events.csv",
        selector_compressed,
    )
    write_csv(args.out_dir / "task_stage_decomposition.csv", task_rows)
    (args.out_dir / "loss_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_task_stages(
        task_rows,
        args.out_dir,
        reference_selector=args.reference_selector,
        candidate_selector=args.candidate_selector,
        reference_variant=args.reference_variant,
        candidate_variant=args.candidate_variant,
    )
    plot_reconstruction_risk(compression_events, args.out_dir)

    losses = [
        row
        for row in learned_compression
        if row["transition"] == "worse"
    ]
    loss_lines = [
        f"- `{row['sample_id']}` ({row['task']}): "
        f"{row['reference_prediction']} -> {row['candidate_prediction']}; "
        f"answer `{row['answer']}`, selected error "
        f"{float(row['selected_reconstruction_error']):.4f}."
        for row in losses
    ] or ["- No full-correct/compressed-wrong events."]
    report = [
        "# Compressed Memory Loss Probe",
        "",
        "## Stage Attribution",
        "",
        f"- `{args.reference_selector}` with `{args.reference_variant}`: "
        f"{exact_correct}/{sample_count} "
        f"({100.0 * exact_correct / sample_count:.1f}%).",
        f"- `{args.candidate_selector}` with `{args.reference_variant}`: "
        f"{learned_correct}/{sample_count} "
        f"({100.0 * learned_correct / sample_count:.1f}%), net "
        f"{learned_correct - exact_correct:+d} samples.",
        f"- `{args.candidate_selector}` with `{args.candidate_variant}`: "
        f"{compressed_correct}/{sample_count} "
        f"({100.0 * compressed_correct / sample_count:.1f}%), net "
        f"{compressed_correct - learned_correct:+d} sample versus "
        f"`{args.reference_variant}`.",
        "",
        f"The candidate selector changes net correctness by "
        f"{learned_correct - exact_correct:+d} samples at the reference "
        f"variant. Changing memory from `{args.reference_variant}` to "
        f"`{args.candidate_variant}` changes it by "
        f"{compressed_correct - learned_correct:+d} samples. These are "
        "descriptive stage deltas, not causal attribution.",
        "",
        "## Compression Loss Events",
        "",
        *loss_lines,
        "",
        "## Reconstruction Error Diagnostic",
        "",
        f"- AUC for predicting any answer change: "
        f"{format_optional(summary['reconstruction_error_auc_for_prediction_change'])}.",
        f"- AUC for predicting a correctness loss: "
        f"{format_optional(summary['reconstruction_error_auc_for_worse_event'])}.",
        "",
        "These AUCs are descriptive because answer changes and losses are "
        "rare. They test whether global reconstruction error is a useful "
        "allocation score; they do not establish causality.",
        "",
        "![Task-stage decomposition](task_stage_decomposition.png)",
        "",
        "![Reconstruction error outcomes](reconstruction_error_outcomes.png)",
        "",
    ]
    (args.out_dir / "LOSS_PROBE_ANALYSIS.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
