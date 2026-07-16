from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


POLICY_LABELS = {
    "uniform": "uniform",
    "recent": "recent",
    "hybrid": "hybrid",
    "exact_recent": "exact recent",
    "recent_pool_query_topk": "recent pool + top-k",
    "recent_pool_query_mmr": "recent pool + MMR",
    "learned_recent_query_topk": "learned recent-pool top-k",
    "reservoir_recent_query_mmr": "reservoir + recent + MMR",
    "diverse_recent_query_mmr": "diverse + recent + MMR",
    "offline_full_query_mmr": "full history + MMR",
}

POLICY_COLORS = {
    "uniform": "#8d5a97",
    "recent": "#6c757d",
    "hybrid": "#2a6f97",
    "exact_recent": "#264653",
    "recent_pool_query_topk": "#2a9d8f",
    "recent_pool_query_mmr": "#457b9d",
    "learned_recent_query_topk": "#b07aa1",
    "reservoir_recent_query_mmr": "#59a14f",
    "diverse_recent_query_mmr": "#f28e2b",
    "offline_full_query_mmr": "#9c755f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--reference", default="uniform")
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
    logged_seconds: dict[tuple[str, str], float] = {}
    for path in sorted((run_dir / "logs").glob("shard_*.log")):
        for line in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            marker = line.find('{"event": "policy_ok"')
            if marker < 0:
                continue
            try:
                payload = json.loads(line[marker:])
            except json.JSONDecodeError:
                continue
            logged_seconds[
                (str(payload["sample"]), str(payload["policy"]))
            ] = float(payload["seconds"])
    for row in rows:
        key = (str(row["sample_id"]), str(row["policy"]))
        if "policy_seconds" not in row and key in logged_seconds:
            row["policy_seconds"] = logged_seconds[key]
    return rows, sorted(fingerprints)


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.96
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


def summarize_task(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(str(row["task"]), str(row["policy"]))].append(row)
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        correct = sum(int(row["correct"]) for row in values)
        parsed = sum(int(row["parsed"]) for row in values)
        low, high = wilson_interval(correct, len(values))
        output.append(
            {
                "task": key[0],
                "policy": key[1],
                "samples": len(values),
                "parsed": parsed,
                "parsed_rate": parsed / len(values),
                "correct": correct,
                "accuracy": correct / len(values),
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "mean_decode_seconds": float(
                    np.mean(
                        [float(row["decode_seconds"]) for row in values]
                    )
                ),
                "mean_inference_seconds": float(
                    np.mean(
                        [float(row["inference_seconds"]) for row in values]
                    )
                ),
                "mean_policy_seconds": float(
                    np.mean(
                        [
                            float(
                                row.get(
                                    "policy_seconds",
                                    float(row["decode_seconds"])
                                    + float(row["inference_seconds"]),
                                )
                            )
                            for row in values
                        ]
                    )
                ),
                "mean_feature_cache_write_seconds": float(
                    np.mean(
                        [
                            float(
                                row.get(
                                    "feature_cache_write_seconds",
                                    0.0,
                                )
                            )
                            for row in values
                        ]
                    )
                ),
                "mean_preprocess_seconds": float(
                    np.mean(
                        [
                            float(row.get("preprocess_seconds", 0.0))
                            for row in values
                        ]
                    )
                ),
                "mean_vision_encode_seconds": float(
                    np.mean(
                        [
                            float(
                                row.get("vision_encode_seconds", 0.0)
                            )
                            for row in values
                        ]
                    )
                ),
                "visual_tokens": int(values[0]["visual_tokens"]),
                "selection_state_proxy_bytes": int(
                    values[0]["selection_state_proxy_bytes"]
                ),
                "llm_visual_token_bytes": int(
                    values[0]["llm_visual_token_bytes"]
                ),
            }
        )
    return output


def summarize_overall(
    task_rows: list[dict[str, object]],
    *,
    reference: str = "uniform",
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in task_rows:
        grouped[str(row["policy"])].append(row)
    output = []
    for policy in sorted(grouped):
        values = grouped[policy]
        samples = sum(int(row["samples"]) for row in values)
        correct = sum(int(row["correct"]) for row in values)
        parsed = sum(int(row["parsed"]) for row in values)
        low, high = wilson_interval(correct, samples)
        output.append(
            {
                "policy": policy,
                "tasks": len(values),
                "samples": samples,
                "parsed_rate": parsed / samples,
                "micro_accuracy": correct / samples,
                "micro_accuracy_ci95_low": low,
                "micro_accuracy_ci95_high": high,
                "macro_task_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in values])
                ),
                "mean_inference_seconds": (
                    sum(
                        float(row["mean_inference_seconds"])
                        * int(row["samples"])
                        for row in values
                    )
                    / samples
                ),
                "mean_decode_seconds": (
                    sum(
                        float(row.get("mean_decode_seconds", 0.0))
                        * int(row["samples"])
                        for row in values
                    )
                    / samples
                ),
                "mean_policy_seconds": (
                    sum(
                        float(
                            row.get(
                                "mean_policy_seconds",
                                float(
                                    row.get("mean_decode_seconds", 0.0)
                                )
                                + float(row["mean_inference_seconds"]),
                            )
                        )
                        * int(row["samples"])
                        for row in values
                    )
                    / samples
                ),
                "mean_feature_cache_write_seconds": (
                    sum(
                        float(
                            row.get(
                                "mean_feature_cache_write_seconds",
                                0.0,
                            )
                        )
                        * int(row["samples"])
                        for row in values
                    )
                    / samples
                ),
                "mean_preprocess_seconds": (
                    sum(
                        float(row.get("mean_preprocess_seconds", 0.0))
                        * int(row["samples"])
                        for row in values
                    )
                    / samples
                ),
                "mean_vision_encode_seconds": (
                    sum(
                        float(
                            row.get(
                                "mean_vision_encode_seconds",
                                0.0,
                            )
                        )
                        * int(row["samples"])
                        for row in values
                    )
                    / samples
                ),
            }
        )
    reference_accuracy = next(
        (
            float(row["macro_task_accuracy"])
            for row in output
            if row["policy"] == reference
        ),
        None,
    )
    uniform_accuracy = next(
        (
            float(row["macro_task_accuracy"])
            for row in output
            if row["policy"] == "uniform"
        ),
        None,
    )
    for row in output:
        row["reference_policy"] = reference
        row["macro_gain_vs_reference"] = (
            float(row["macro_task_accuracy"]) - reference_accuracy
            if reference_accuracy is not None
            else float("nan")
        )
        row["macro_gain_vs_uniform"] = (
            float(row["macro_task_accuracy"]) - uniform_accuracy
            if uniform_accuracy is not None
            else float("nan")
        )
    return output


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    observed = math.comb(trials, successes) / (2**trials)
    probability = 0.0
    for value in range(trials + 1):
        mass = math.comb(trials, value) / (2**trials)
        if mass <= observed + 1e-15:
            probability += mass
    return min(probability, 1.0)


def paired_policy_comparisons(
    rows: list[dict[str, object]],
    *,
    reference: str = "uniform",
    seed: int = 42,
    bootstrap_samples: int = 10000,
) -> list[dict[str, object]]:
    lookup = {
        (str(row["sample_id"]), str(row["policy"])): int(row["correct"])
        for row in rows
    }
    policies = sorted(
        {str(row["policy"]) for row in rows if row["policy"] != reference}
    )
    rng = np.random.default_rng(seed)
    output = []
    for policy in policies:
        sample_ids = sorted(
            {
                str(row["sample_id"])
                for row in rows
                if row["policy"] == policy
                and (str(row["sample_id"]), reference) in lookup
            }
        )
        differences = np.asarray(
            [
                lookup[(sample_id, policy)]
                - lookup[(sample_id, reference)]
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
                "policy": policy,
                "reference": reference,
                "paired_samples": int(differences.size),
                "accuracy_gain": float(np.mean(differences)),
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


def plot_task_accuracy(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    tasks = sorted({str(row["task"]) for row in rows})
    preferred = (
        "uniform",
        "recent",
        "hybrid",
        "exact_recent",
        "recent_pool_query_topk",
        "recent_pool_query_mmr",
        "learned_recent_query_topk",
        "reservoir_recent_query_mmr",
        "diverse_recent_query_mmr",
        "offline_full_query_mmr",
    )
    available = {str(row["policy"]) for row in rows}
    policies = [policy for policy in preferred if policy in available]
    policies.extend(sorted(available - set(policies)))
    lookup = {
        (str(row["task"]), str(row["policy"])): float(row["accuracy"])
        for row in rows
    }
    x = np.arange(len(tasks))
    width = 0.8 / max(len(policies), 1)
    fig, axis = plt.subplots(figsize=(1.55 * len(tasks) + 2.7, 4.7))
    for index, policy in enumerate(policies):
        values = [lookup.get((task, policy), np.nan) for task in tasks]
        offset = (index - (len(policies) - 1) / 2) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            label=POLICY_LABELS.get(policy, policy),
            color=POLICY_COLORS.get(policy),
        )
    axis.set_xticks(x, tasks, rotation=25, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title("LLaVA pooled multi-frame anchor by frame policy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_latency(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    policies = [str(row["policy"]) for row in rows]
    latency = [float(row["mean_policy_seconds"]) for row in rows]
    accuracy = [float(row["macro_task_accuracy"]) for row in rows]
    colors = [POLICY_COLORS.get(policy, "#457b9d") for policy in policies]
    label_offsets = {
        "hybrid": (8, 10),
        "recent": (8, -14),
        "uniform": (8, 8),
        "exact_recent": (8, -18),
        "recent_pool_query_topk": (8, 10),
        "recent_pool_query_mmr": (8, -12),
        "learned_recent_query_topk": (8, 14),
        "reservoir_recent_query_mmr": (8, 10),
        "diverse_recent_query_mmr": (8, -14),
        "offline_full_query_mmr": (8, 8),
    }
    fig, axis = plt.subplots(figsize=(6.3, 4.5))
    axis.scatter(latency, accuracy, s=90, c=colors)
    for policy, x_value, y_value in zip(
        policies,
        latency,
        accuracy,
        strict=True,
    ):
        axis.annotate(
            POLICY_LABELS.get(policy, policy),
            (x_value, y_value),
            xytext=label_offsets.get(policy, (5, 5)),
            textcoords="offset points",
        )
    axis.set_xlabel("Mean end-to-end policy latency (s/sample)")
    axis.set_ylabel("Macro task accuracy")
    axis.set_ylim(0.35, 0.60)
    axis.set_title("LLaVA frame-policy accuracy-latency anchor")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_paired_gains(
    rows: list[dict[str, object]],
    path: Path,
    *,
    reference: str = "uniform",
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    labels = [
        POLICY_LABELS.get(str(row["policy"]), str(row["policy"]))
        for row in rows
    ]
    values = np.asarray([float(row["accuracy_gain"]) for row in rows])
    low = np.asarray(
        [float(row["bootstrap_ci95_low"]) for row in rows]
    )
    high = np.asarray(
        [float(row["bootstrap_ci95_high"]) for row in rows]
    )
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in values]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(7.4, 4.5))
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
    axis.set_xticks(x, labels, rotation=15, ha="right")
    reference_label = POLICY_LABELS.get(reference, reference)
    axis.set_ylabel(f"Paired accuracy gain vs {reference_label}")
    axis.set_title("LLaVA frame-policy paired comparison")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or args.run_dir / "aggregate"
    reference_slug = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in args.reference
    )
    rows, fingerprints = load_rows(args.run_dir)
    task_rows = summarize_task(rows)
    overall_rows = summarize_overall(
        task_rows,
        reference=args.reference,
    )
    paired_rows = paired_policy_comparisons(
        rows,
        reference=args.reference,
    )
    write_csv(out_dir / "predictions.csv", rows)
    write_csv(out_dir / "task_accuracy.csv", task_rows)
    write_csv(out_dir / "overall_accuracy.csv", overall_rows)
    write_csv(
        out_dir / f"paired_vs_{reference_slug}.csv",
        paired_rows,
    )
    plot_task_accuracy(
        task_rows,
        out_dir / "task_accuracy_by_policy.png",
    )
    plot_latency(
        overall_rows,
        out_dir / "accuracy_latency.png",
    )
    plot_paired_gains(
        paired_rows,
        out_dir / f"paired_gain_vs_{reference_slug}.png",
        reference=args.reference,
    )
    summary = {
        "checkpoint_count": len(
            list((args.run_dir / "checkpoints").glob("*.json"))
        ),
        "prediction_rows": len(rows),
        "tasks": sorted({str(row["task"]) for row in rows}),
        "policies": sorted({str(row["policy"]) for row in rows}),
        "fingerprints": fingerprints,
        "reference": args.reference,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
