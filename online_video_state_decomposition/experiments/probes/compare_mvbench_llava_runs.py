from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(run_dir: Path) -> list[dict[str, str]]:
    aggregate_path = run_dir / "aggregate" / "predictions.csv"
    if aggregate_path.exists():
        with aggregate_path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = []
    for path in sorted((run_dir / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    if not rows:
        raise FileNotFoundError(f"no predictions found under {run_dir}")
    return rows


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


def summarize_run(
    rows: list[dict[str, str]],
    *,
    label: str,
) -> list[dict[str, object]]:
    output = []
    for policy in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy]
        output.append(
            {
                "run": label,
                "policy": policy,
                "samples": len(selected),
                "accuracy": float(
                    np.mean([int(row["correct"]) for row in selected])
                ),
                "parsed_rate": float(
                    np.mean([int(row["parsed"]) for row in selected])
                ),
                "mean_inference_seconds": float(
                    np.mean(
                        [
                            float(row["inference_seconds"])
                            for row in selected
                        ]
                    )
                ),
                "visual_tokens": int(selected[0]["visual_tokens"]),
                "selection_state_proxy_bytes": int(
                    selected[0]["selection_state_proxy_bytes"]
                ),
                "llm_visual_token_bytes": int(
                    selected[0]["llm_visual_token_bytes"]
                ),
            }
        )
    return output


def paired_comparison(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    *,
    label_a: str,
    label_b: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    lookup_a = {
        (row["sample_id"], row["policy"]): int(row["correct"])
        for row in rows_a
    }
    lookup_b = {
        (row["sample_id"], row["policy"]): int(row["correct"])
        for row in rows_b
    }
    rng = np.random.default_rng(seed)
    output = []
    policies = sorted(
        {policy for _, policy in lookup_a} & {policy for _, policy in lookup_b}
    )
    for policy in policies:
        keys = sorted(
            key
            for key in lookup_a
            if key[1] == policy and key in lookup_b
        )
        differences = np.asarray(
            [lookup_b[key] - lookup_a[key] for key in keys],
            dtype=np.float64,
        )
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
                "run_a": label_a,
                "run_b": label_b,
                "paired_samples": int(differences.size),
                "accuracy_gain_b_vs_a": float(np.mean(differences)),
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


def sensitivity_summary(
    rows: list[dict[str, str]],
    *,
    label: str,
) -> dict[str, object]:
    by_sample: dict[str, dict[str, int]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], {})[row["policy"]] = int(
            row["predicted_index"]
        )
    changed = sum(
        len(set(predictions.values())) > 1
        for predictions in by_sample.values()
    )
    return {
        "run": label,
        "samples": len(by_sample),
        "policy_changed_samples": changed,
        "policy_changed_fraction": changed / len(by_sample),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(
    rows: list[dict[str, object]],
    path: Path,
    *,
    labels: tuple[str, str],
) -> None:
    import matplotlib.pyplot as plt

    policies = sorted({str(row["policy"]) for row in rows})
    lookup = {
        (str(row["run"]), str(row["policy"])): float(row["accuracy"])
        for row in rows
    }
    x = np.arange(len(policies))
    width = 0.34
    colors = ("#457b9d", "#e76f51")
    fig, axis = plt.subplots(figsize=(7.0, 4.6))
    for index, label in enumerate(labels):
        values = [lookup[(label, policy)] for policy in policies]
        offset = (index - 0.5) * width
        bars = axis.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=colors[index],
        )
        axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_xticks(x, policies)
    axis.set_ylim(0.35, 0.60)
    axis.set_ylabel("Accuracy")
    axis.set_title("LLaVA visual-token fidelity sensitivity")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_paired_gain(
    rows: list[dict[str, object]],
    path: Path,
    *,
    label_a: str,
    label_b: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["policy"]) for row in rows]
    values = np.asarray(
        [float(row["accuracy_gain_b_vs_a"]) for row in rows]
    )
    low = np.asarray(
        [float(row["bootstrap_ci95_low"]) for row in rows]
    )
    high = np.asarray(
        [float(row["bootstrap_ci95_high"]) for row in rows]
    )
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(6.4, 4.3))
    axis.bar(x, values, color="#2a9d8f")
    axis.errorbar(
        x,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="none",
        ecolor="#343a40",
        capsize=4,
    )
    axis.axhline(0.0, color="#343a40", linewidth=1)
    axis.set_xticks(x, labels)
    axis.set_ylabel(f"Paired accuracy gain: {label_b} vs {label_a}")
    axis.set_title("Visual-token fidelity paired comparison")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows_a = load_predictions(args.run_a)
    rows_b = load_predictions(args.run_b)
    sample_ids_a = {row["sample_id"] for row in rows_a}
    sample_ids_b = {row["sample_id"] for row in rows_b}
    if sample_ids_a != sample_ids_b:
        raise ValueError("runs do not contain the same sample IDs")
    summary_rows = summarize_run(rows_a, label=args.label_a)
    summary_rows.extend(summarize_run(rows_b, label=args.label_b))
    paired_rows = paired_comparison(
        rows_a,
        rows_b,
        label_a=args.label_a,
        label_b=args.label_b,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    sensitivity_rows = [
        sensitivity_summary(rows_a, label=args.label_a),
        sensitivity_summary(rows_b, label=args.label_b),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "run_summary.csv", summary_rows)
    write_csv(args.out_dir / "paired_b_vs_a.csv", paired_rows)
    write_csv(
        args.out_dir / "policy_sensitivity.csv",
        sensitivity_rows,
    )
    plot_accuracy(
        summary_rows,
        args.out_dir / "pooling_accuracy_comparison.png",
        labels=(args.label_a, args.label_b),
    )
    plot_paired_gain(
        paired_rows,
        args.out_dir / "pooling_paired_gain.png",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    summary = {
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "sample_count": len(sample_ids_a),
        "policies": sorted({row["policy"] for row in rows_a}),
    }
    (args.out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
