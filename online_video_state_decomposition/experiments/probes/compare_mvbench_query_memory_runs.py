from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_POLICIES = (
    "recent_pool_query_topk",
    "recent_pool_query_mmr",
    "learned_recent_query_topk",
    "offline_uniform",
    "offline_full_query_mmr",
)

POLICY_LABELS = {
    "recent_pool_query_topk": "Recent pool + top-k",
    "recent_pool_query_mmr": "Recent pool + MMR",
    "learned_recent_query_topk": "Learned recent-pool top-k",
    "offline_uniform": "Offline uniform",
    "offline_full_query_mmr": "Full history + MMR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", default="exact_recent")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["split"] == "evaluation"
        ]


def paired_deltas(
    rows: list[dict[str, str]],
    *,
    policy: str,
    reference: str,
) -> tuple[np.ndarray, list[str]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    tasks: dict[str, str] = {}
    for row in rows:
        if row["policy"] not in {policy, reference}:
            continue
        grouped[row["sample_id"]][row["policy"]] = int(row["correct"])
        tasks[row["sample_id"]] = row["task"]
    sample_ids = sorted(
        sample_id
        for sample_id, values in grouped.items()
        if policy in values and reference in values
    )
    deltas = np.asarray(
        [
            grouped[sample_id][policy] - grouped[sample_id][reference]
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )
    return deltas, [tasks[sample_id] for sample_id in sample_ids]


def bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    chunk_size = min(samples, 1_000)
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        current = min(chunk_size, remaining)
        indices = generator.integers(
            0,
            values.size,
            size=(current, values.size),
        )
        means.append(values[indices].mean(axis=1))
        remaining -= current
    estimates = np.concatenate(means)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_summary(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int]:
    low, high = bootstrap_interval(
        values,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "samples": int(values.size),
        "gain": float(values.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "better": int(np.sum(values > 0)),
        "worse": int(np.sum(values < 0)),
        "tied": int(np.sum(values == 0)),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_paired_gains(
    rows: list[dict[str, object]],
    *,
    policies: list[str],
    out_dir: Path,
) -> None:
    run_order = ("formal", "confirmation", "pooled_descriptive")
    colors = {
        "formal": "#3d5a80",
        "confirmation": "#2a9d8f",
        "pooled_descriptive": "#e09f3e",
    }
    labels = {
        "formal": "Formal evaluation",
        "confirmation": "Untouched reserve",
        "pooled_descriptive": "Pooled descriptive",
    }
    lookup = {
        (str(row["run"]), str(row["policy"])): row
        for row in rows
    }
    x = np.arange(len(policies), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(12.5, 5.8))
    for index, run in enumerate(run_order):
        selected = [lookup[(run, policy)] for policy in policies]
        gains = np.asarray([float(row["gain"]) for row in selected])
        lower = gains - np.asarray(
            [float(row["ci95_low"]) for row in selected]
        )
        upper = np.asarray(
            [float(row["ci95_high"]) for row in selected]
        ) - gains
        axis.errorbar(
            x + (index - 1) * width,
            gains,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=1.8,
            color=colors[run],
            label=labels[run],
        )
    axis.axhline(0.0, color="#333333", linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [POLICY_LABELS.get(policy, policy) for policy in policies],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("Paired accuracy gain vs exact recent")
    axis.set_title("Query-memory transfer across disjoint evaluations")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"formal_confirmation_paired_gain.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_task_gains(
    rows: list[dict[str, object]],
    *,
    policies: list[str],
    out_dir: Path,
) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    row_keys = [
        (run, policy)
        for run in ("formal", "confirmation")
        for policy in policies
    ]
    lookup = {
        (str(row["run"]), str(row["policy"]), str(row["task"])): float(
            row["gain"]
        )
        for row in rows
    }
    matrix = np.asarray(
        [
            [lookup[(run, policy, task)] for task in tasks]
            for run, policy in row_keys
        ]
    )
    limit = max(0.05, float(np.max(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(tasks)))
    axis.set_xticklabels(tasks, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(row_keys)))
    axis.set_yticklabels(
        [
            f"{run}: {POLICY_LABELS.get(policy, policy)}"
            for run, policy in row_keys
        ]
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_title("Task-level gain transfer versus exact recent")
    figure.colorbar(image, ax=axis, label="Accuracy gain")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"formal_confirmation_task_gain.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


def render_report(
    paired_rows: list[dict[str, object]],
    *,
    policies: list[str],
) -> str:
    lookup = {
        (str(row["run"]), str(row["policy"])): row
        for row in paired_rows
    }
    lines = [
        "# Formal-to-Confirmation Query-Memory Analysis",
        "",
        "The formal evaluation and untouched-reserve confirmation contain "
        "200 disjoint records each. Pooled values below are descriptive and "
        "do not erase the post-hoc status of policies selected after the "
        "formal evaluation.",
        "",
        "## Paired Results",
        "",
        "| Policy | Formal gain | Reserve gain | Pooled gain | Pooled flips |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in policies:
        formal = lookup[("formal", policy)]
        confirmation = lookup[("confirmation", policy)]
        pooled = lookup[("pooled_descriptive", policy)]
        lines.append(
            f"| {POLICY_LABELS.get(policy, policy)} "
            f"| {float(formal['gain']):+.2%} "
            f"[{float(formal['ci95_low']):+.2%}, "
            f"{float(formal['ci95_high']):+.2%}] "
            f"| {float(confirmation['gain']):+.2%} "
            f"[{float(confirmation['ci95_low']):+.2%}, "
            f"{float(confirmation['ci95_high']):+.2%}] "
            f"| {float(pooled['gain']):+.2%} "
            f"[{float(pooled['ci95_low']):+.2%}, "
            f"{float(pooled['ci95_high']):+.2%}] "
            f"| {pooled['better']} better / {pooled['worse']} worse |"
        )
    learned = lookup.get(
        ("pooled_descriptive", "learned_recent_query_topk")
    )
    primary = lookup.get(
        ("pooled_descriptive", "recent_pool_query_topk")
    )
    lines.extend(["", "## Interpretation", ""])
    if learned is not None:
        lines.append(
            "- The frozen learned readout is the only tested bounded policy "
            "whose paired point gain is positive in both disjoint "
            f"evaluations; its descriptive pooled gain is "
            f"{float(learned['gain']):+.2%}."
        )
    if primary is not None:
        lines.append(
            "- The frozen top-k confirmation primary also remains positive "
            f"when pooled ({float(primary['gain']):+.2%}), but its reserve "
            "interval crosses zero and the answer changes are sparse."
        )
    lines.extend(
        [
            "- Task-level plots must be checked because a positive aggregate "
            "can still be concentrated in scene-transition or "
            "action-sequence examples.",
            "- These CLIP results justify only a paired raw-frame VLM anchor, "
            "not a claim of bounded deployment or a native learned memory.",
            "",
            "## Figures",
            "",
            "![Paired gains](formal_confirmation_paired_gain.png)",
            "",
            "![Task gains](formal_confirmation_task_gain.png)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    policies = [
        value.strip()
        for value in args.policies.split(",")
        if value.strip()
    ]
    runs = {
        "formal": read_predictions(args.formal_dir / "predictions.csv"),
        "confirmation": read_predictions(
            args.confirmation_dir / "predictions.csv"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    values_by_policy: dict[str, list[np.ndarray]] = defaultdict(list)
    for run_index, (run, rows) in enumerate(runs.items()):
        for policy_index, policy in enumerate(policies):
            values, tasks = paired_deltas(
                rows,
                policy=policy,
                reference=args.reference,
            )
            if values.size == 0:
                raise ValueError(f"missing paired rows for {run}/{policy}")
            values_by_policy[policy].append(values)
            summary = paired_summary(
                values,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + run_index * 100 + policy_index,
            )
            paired_rows.append(
                {"run": run, "policy": policy, **summary}
            )
            grouped: dict[str, list[float]] = defaultdict(list)
            for task, value in zip(tasks, values):
                grouped[task].append(float(value))
            for task in sorted(grouped):
                task_rows.append(
                    {
                        "run": run,
                        "policy": policy,
                        "task": task,
                        "gain": float(np.mean(grouped[task])),
                        "samples": len(grouped[task]),
                    }
                )
    for policy_index, policy in enumerate(policies):
        pooled = np.concatenate(values_by_policy[policy])
        paired_rows.append(
            {
                "run": "pooled_descriptive",
                "policy": policy,
                **paired_summary(
                    pooled,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 1_000 + policy_index,
                ),
            }
        )
    write_csv(
        args.out_dir / "formal_confirmation_paired.csv",
        paired_rows,
        [
            "run",
            "policy",
            "samples",
            "gain",
            "ci95_low",
            "ci95_high",
            "better",
            "worse",
            "tied",
        ],
    )
    write_csv(
        args.out_dir / "formal_confirmation_task_gain.csv",
        task_rows,
        ["run", "policy", "task", "gain", "samples"],
    )
    plot_paired_gains(
        paired_rows,
        policies=policies,
        out_dir=args.out_dir,
    )
    bounded_policies = [
        policy
        for policy in policies
        if policy
        in {
            "recent_pool_query_topk",
            "recent_pool_query_mmr",
            "learned_recent_query_topk",
        }
    ]
    plot_task_gains(
        task_rows,
        policies=bounded_policies,
        out_dir=args.out_dir,
    )
    report = render_report(paired_rows, policies=policies)
    report_path = args.out_dir / "FORMAL_CONFIRMATION_ANALYSIS.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
