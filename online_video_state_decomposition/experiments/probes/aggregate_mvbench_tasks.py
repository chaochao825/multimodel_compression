from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--plot-capacity", type=int, default=8)
    return parser.parse_args()


def load_rows(run_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    fingerprints: set[str] = set()
    for path in sorted((run_dir / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fingerprints.add(str(payload["configuration_fingerprint"]))
        rows.extend(payload["rows"])
    if not rows:
        raise FileNotFoundError(f"no completed checkpoints under {run_dir}")
    return rows, sorted(fingerprints)


def wilson_interval(
    correct: int,
    total: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def task_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                row["task"],
                row["method"],
                int(row["capacity"]),
                int(row["total_state_bytes"]),
            )
        ].append(row)
    output = []
    for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
        values = grouped[key]
        correct = sum(int(row["correct"]) for row in values)
        low, high = wilson_interval(correct, len(values))
        output.append(
            {
                "task": key[0],
                "method": key[1],
                "capacity": key[2],
                "total_state_bytes": key[3],
                "samples": len(values),
                "correct": correct,
                "accuracy": correct / len(values),
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "mean_decode_seconds": float(
                    np.mean(
                        [float(row["decode_seconds"]) for row in values]
                    )
                ),
                "mean_image_encode_seconds": float(
                    np.mean(
                        [
                            float(row["image_encode_seconds"])
                            for row in values
                        ]
                    )
                ),
                "mean_memory_update_seconds": float(
                    np.mean(
                        [
                            float(row["memory_update_seconds"])
                            for row in values
                        ]
                    )
                ),
                "mean_memory_read_seconds": float(
                    np.mean(
                        [
                            float(row["memory_read_seconds"])
                            for row in values
                        ]
                    )
                ),
            }
        )
    return output


def overall_summary(
    task_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in task_rows:
        grouped[
            (
                row["method"],
                row["capacity"],
                row["total_state_bytes"],
            )
        ].append(row)
    output = []
    for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
        values = grouped[key]
        total = sum(int(row["samples"]) for row in values)
        correct = sum(int(row["correct"]) for row in values)
        low, high = wilson_interval(correct, total)
        output.append(
            {
                "method": key[0],
                "capacity": key[1],
                "total_state_bytes": key[2],
                "tasks": len(values),
                "samples": total,
                "micro_accuracy": correct / total,
                "micro_accuracy_ci95_low": low,
                "micro_accuracy_ci95_high": high,
                "macro_task_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in values])
                ),
            }
        )
    recent = {
        int(row["capacity"]): float(row["macro_task_accuracy"])
        for row in output
        if row["method"] == "recent_window"
    }
    for row in output:
        baseline = recent.get(int(row["capacity"]))
        row["macro_gain_vs_recent"] = (
            float(row["macro_task_accuracy"]) - baseline
            if baseline is not None
            else float("nan")
        )
    return output


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    probability = 0.0
    observed = math.comb(trials, successes) / (2**trials)
    for value in range(trials + 1):
        mass = math.comb(trials, value) / (2**trials)
        if mass <= observed + 1e-15:
            probability += mass
    return min(probability, 1.0)


def paired_comparisons(
    rows: list[dict[str, object]],
    *,
    seed: int = 42,
    bootstrap_samples: int = 10000,
) -> list[dict[str, object]]:
    by_key = {
        (
            str(row["sample_id"]),
            str(row["method"]),
            int(row["capacity"]),
        ): int(row["correct"])
        for row in rows
    }
    methods = sorted(
        {
            (str(row["method"]), int(row["capacity"]))
            for row in rows
            if row["method"] not in {"recent_window", "full_sequence"}
        }
    )
    rng = np.random.default_rng(seed)
    output = []
    for method, capacity in methods:
        sample_ids = sorted(
            {
                str(row["sample_id"])
                for row in rows
                if row["method"] == method
                and int(row["capacity"]) == capacity
                and (
                    str(row["sample_id"]),
                    "recent_window",
                    capacity,
                )
                in by_key
            }
        )
        differences = np.asarray(
            [
                by_key[(sample_id, method, capacity)]
                - by_key[(sample_id, "recent_window", capacity)]
                for sample_id in sample_ids
            ],
            dtype=np.float64,
        )
        if differences.size == 0:
            continue
        better = int(np.sum(differences > 0))
        worse = int(np.sum(differences < 0))
        ties = int(np.sum(differences == 0))
        resampled = rng.choice(
            differences,
            size=(bootstrap_samples, differences.size),
            replace=True,
        ).mean(axis=1)
        low, high = np.quantile(resampled, [0.025, 0.975])
        output.append(
            {
                "method": method,
                "capacity": capacity,
                "paired_samples": int(differences.size),
                "accuracy_gain_vs_recent": float(np.mean(differences)),
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "better_samples": better,
                "worse_samples": worse,
                "tied_samples": ties,
                "mcnemar_exact_p": exact_binomial_two_sided(
                    min(better, worse),
                    better + worse,
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_frontier(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "recent_window": "#6c757d",
        "uniform_reservoir": "#2a6f97",
        "adaptive_slots": "#e76f51",
        "oja_subspace": "#2a9d8f",
        "instant_oja": "#8d5a97",
        "full_sequence": "#264653",
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        selected.sort(key=lambda row: int(row["total_state_bytes"]))
        x = np.asarray(
            [int(row["total_state_bytes"]) / 1024 for row in selected]
        )
        y = np.asarray(
            [float(row["macro_task_accuracy"]) for row in selected]
        )
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=method,
            color=colors.get(method),
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("State size (KiB, log2)")
    axis.set_ylim(0.40, 0.60)
    axis.set_ylabel("Macro task accuracy (CLIP proxy)")
    axis.set_title(
        "MVBench CLIP proxy accuracy vs matched state budget"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_task_heatmap(
    rows: list[dict[str, object]],
    path: Path,
    *,
    capacity: int,
) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if int(row["capacity"]) == capacity
        and row["method"] != "full_sequence"
    ]
    if not selected:
        return
    tasks = sorted({str(row["task"]) for row in selected})
    methods = [
        method
        for method in (
            "recent_window",
            "uniform_reservoir",
            "adaptive_slots",
            "oja_subspace",
            "instant_oja",
        )
        if any(row["method"] == method for row in selected)
    ]
    lookup = {
        (str(row["task"]), str(row["method"])): float(row["accuracy"])
        for row in selected
    }
    matrix = np.asarray(
        [[lookup.get((task, method), np.nan) for method in methods]
         for task in tasks]
    )
    fig, axis = plt.subplots(
        figsize=(1.55 * len(methods) + 2.5, 0.55 * len(tasks) + 2.1)
    )
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    axis.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    axis.set_yticks(range(len(tasks)), tasks)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                )
    axis.set_title(f"CLIP proxy task accuracy at capacity={capacity}")
    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_paired_gains(
    rows: list[dict[str, object]],
    path: Path,
    *,
    capacity: int,
) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row for row in rows if int(row["capacity"]) == capacity
    ]
    if not selected:
        return
    selected.sort(key=lambda row: float(row["accuracy_gain_vs_recent"]))
    labels = [str(row["method"]) for row in selected]
    values = np.asarray(
        [float(row["accuracy_gain_vs_recent"]) for row in selected]
    )
    low = np.asarray(
        [float(row["bootstrap_ci95_low"]) for row in selected]
    )
    high = np.asarray(
        [float(row["bootstrap_ci95_high"]) for row in selected]
    )
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in values]
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(labels))
    axis.bar(x, values, color=colors)
    axis.errorbar(
        x,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="none",
        ecolor="#343a40",
        capsize=4,
    )
    axis.axhline(0.0, color="#343a40", linewidth=1)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Paired accuracy gain vs recent")
    axis.set_title(f"Matched-sample gain at capacity={capacity}")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or args.run_dir / "aggregate"
    rows, fingerprints = load_rows(args.run_dir)
    task_rows = task_summary(rows)
    overall_rows = overall_summary(task_rows)
    paired_rows = paired_comparisons(rows)
    write_csv(out_dir / "predictions.csv", rows)
    write_csv(out_dir / "task_accuracy.csv", task_rows)
    write_csv(out_dir / "overall_accuracy.csv", overall_rows)
    write_csv(out_dir / "paired_vs_recent.csv", paired_rows)
    plot_accuracy_frontier(
        overall_rows,
        out_dir / "accuracy_vs_state_budget.png",
    )
    plot_task_heatmap(
        task_rows,
        out_dir / "task_accuracy_heatmap.png",
        capacity=args.plot_capacity,
    )
    plot_paired_gains(
        paired_rows,
        out_dir / "paired_gain_vs_recent.png",
        capacity=args.plot_capacity,
    )
    summary = {
        "checkpoint_count": len(
            list((args.run_dir / "checkpoints").glob("*.json"))
        ),
        "prediction_rows": len(rows),
        "tasks": sorted({str(row["task"]) for row in rows}),
        "methods": sorted({str(row["method"]) for row in rows}),
        "fingerprints": fingerprints,
        "plot_capacity": args.plot_capacity,
    }
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
