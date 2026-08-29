from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from aggregate_compressed_feature_memory import clopper_pearson_upper


EXPECTED_TASKS = (
    "fine_grained_action",
    "action_antonym",
    "unexpected_action",
    "counterfactual_inference",
    "action_count",
)
EXPECTED_EXCLUDED_BY_TASK = {
    "fine_grained_action": 0,
    "action_antonym": 0,
    "unexpected_action": 0,
    "counterfactual_inference": 5,
    "action_count": 0,
}
NUMERIC_ROW_FIELDS = (
    "answer_logprob",
    "answer_logprob_delta",
    "candidate_kl",
    "dense_native_feature_bytes",
    "feature_relative_l2",
    "inference_seconds",
    "injection_max_abs",
    "native_feature_state_bytes",
    "state_compression_ratio",
    "vision_seconds",
    "vocabulary_kl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def classify_confirmation(summary: dict[str, float | int]) -> str:
    structural = (
        int(summary["samples"]) == 500
        and int(summary["failed_samples"]) == 0
        and int(summary["duplicate_samples"]) == 0
        and int(summary["nonfinite_metrics"]) == 0
        and float(summary["max_injection_abs"]) <= 1e-3
        and int(summary["max_state_bytes"]) <= 2_867_328
        and float(summary["min_compression_ratio"]) >= 7.8
    )
    if not structural:
        return "INVALID"
    aggregate_loss = float(summary["reference_accuracy"]) - float(
        summary["candidate_accuracy"]
    )
    pass_guards = (
        float(summary["reference_accuracy"]) >= 0.35
        and aggregate_loss <= 0.02
        and float(summary["harmful_upper_95"]) <= 0.02
        and float(summary["prediction_agreement"]) >= 0.98
        and float(summary["minimum_task_accuracy_delta"]) >= -0.05
    )
    if pass_guards:
        return "PASS"
    adverse = (
        aggregate_loss > 0.05
        or float(summary["prediction_agreement"]) < 0.95
        or float(summary["harmful_rate"]) > 0.05
        or float(summary["minimum_task_accuracy_delta"]) < -0.10
    )
    return "ADVERSE" if adverse else "BOUNDARY"


def validate_configuration(
    configuration: dict[str, object],
    codec_metadata: dict[str, object],
) -> list[str]:
    if tuple(configuration["tasks"]) != EXPECTED_TASKS:
        raise ValueError("run tasks differ from the frozen confirmation tasks")
    expected_fields = {
        "candidate": "pca_r456_s0",
        "claim_tier": "onevision_pca_r456_untouched_task_confirmation",
        "eligibility": "2_to_26_candidates_and_answer_in_candidates",
        "samples_per_task": 100,
        "selection_seed": 20260829,
        "sampled_frames": 32,
        "feature_pool_frames": 16,
        "frame_budget": 8,
    }
    for field, expected in expected_fields.items():
        if configuration[field] != expected:
            raise ValueError(
                f"configuration field {field} differs from the frozen protocol"
            )
    if configuration["excluded_by_task"] != EXPECTED_EXCLUDED_BY_TASK:
        raise ValueError("eligibility exclusions differ from the frozen protocol")

    expected_ids = [str(sample_id) for sample_id in configuration["sample_ids"]]
    if len(expected_ids) != 500 or len(set(expected_ids)) != 500:
        raise ValueError("configuration must contain 500 unique sample IDs")
    task_counts = Counter(
        task
        for sample_id in expected_ids
        for task in EXPECTED_TASKS
        if sample_id.startswith(f"{task}_")
    )
    if task_counts != Counter({task: 100 for task in EXPECTED_TASKS}):
        raise ValueError("configuration must contain 100 IDs for every frozen task")

    codec_fields = {
        "rank": 456,
        "sampled_frames": 32,
        "feature_pool_frames": 16,
        "frame_budget": 8,
    }
    for field, expected in codec_fields.items():
        if codec_metadata[field] != expected:
            raise ValueError(f"codec field {field} differs from the frozen protocol")
    return expected_ids


def bootstrap_accuracy_delta(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, reference.size, size=reference.size)
        deltas[index] = np.mean(candidate[selected] - reference[selected])
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    configuration = json.loads(
        (args.run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    codec_metadata = json.loads(
        (args.run_dir / "codec_metadata.json").read_text(encoding="utf-8")
    )
    expected_ids = validate_configuration(configuration, codec_metadata)
    failures = []
    for path in sorted(args.run_dir.glob("failures_shard_*.json")):
        failures.extend(json.loads(path.read_text(encoding="utf-8")))
    rows = []
    for path in sorted((args.run_dir / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload["row"])
    counts = Counter(str(row["sample_id"]) for row in rows)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    observed_ids = set(counts)
    missing_ids = sorted(set(expected_ids) - observed_ids)
    unexpected_ids = sorted(observed_ids - set(expected_ids))
    if unexpected_ids:
        raise ValueError(f"unexpected sample IDs: {unexpected_ids[:5]}")
    nonfinite_metrics = sum(
        not math.isfinite(float(row[field]))
        for row in rows
        for field in NUMERIC_ROW_FIELDS
    )

    reference = np.asarray(
        [int(row["reference_candidate_correct"]) for row in rows], dtype=np.float64
    )
    candidate = np.asarray(
        [int(row["candidate_correct"]) for row in rows], dtype=np.float64
    )
    low, high = bootstrap_accuracy_delta(
        reference,
        candidate,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    task_rows = []
    for task in EXPECTED_TASKS:
        selected = [row for row in rows if row["task"] == task]
        task_reference = np.asarray(
            [int(row["reference_candidate_correct"]) for row in selected],
            dtype=np.float64,
        )
        task_candidate = np.asarray(
            [int(row["candidate_correct"]) for row in selected],
            dtype=np.float64,
        )
        task_rows.append(
            {
                "task": task,
                "samples": len(selected),
                "reference_accuracy": float(np.mean(task_reference)),
                "candidate_accuracy": float(np.mean(task_candidate)),
                "accuracy_delta": float(np.mean(task_candidate - task_reference)),
                "harmful_flips": sum(int(row["harmful_flip"]) for row in selected),
                "beneficial_flips": sum(
                    int(row["beneficial_flip"]) for row in selected
                ),
                "prediction_agreement": float(
                    np.mean([int(row["prediction_match"]) for row in selected])
                ),
                "candidate_kl_mean": float(
                    np.mean([float(row["candidate_kl"]) for row in selected])
                ),
            }
        )

    harmful = sum(int(row["harmful_flip"]) for row in rows)
    metrics = {
        "samples": len(rows),
        "expected_samples": len(expected_ids),
        "failed_samples": len(
            set(missing_ids)
            | {str(failure["sample_id"]) for failure in failures}
        ),
        "duplicate_samples": duplicates,
        "nonfinite_metrics": nonfinite_metrics,
        "reference_accuracy": float(np.mean(reference)),
        "candidate_accuracy": float(np.mean(candidate)),
        "accuracy_delta": float(np.mean(candidate - reference)),
        "accuracy_delta_ci95_low": low,
        "accuracy_delta_ci95_high": high,
        "prediction_agreement": float(
            np.mean([int(row["prediction_match"]) for row in rows])
        ),
        "harmful_flips": harmful,
        "beneficial_flips": sum(int(row["beneficial_flip"]) for row in rows),
        "harmful_rate": harmful / len(rows),
        "harmful_upper_95": clopper_pearson_upper(
            harmful,
            len(rows),
            alpha=0.05,
        ),
        "candidate_kl_mean": float(
            np.mean([float(row["candidate_kl"]) for row in rows])
        ),
        "candidate_kl_median": float(
            np.median([float(row["candidate_kl"]) for row in rows])
        ),
        "candidate_kl_p95": float(
            np.quantile([float(row["candidate_kl"]) for row in rows], 0.95)
        ),
        "vocabulary_kl_mean": float(
            np.mean([float(row["vocabulary_kl"]) for row in rows])
        ),
        "feature_relative_l2_mean": float(
            np.mean([float(row["feature_relative_l2"]) for row in rows])
        ),
        "max_injection_abs": max(float(row["injection_max_abs"]) for row in rows),
        "max_state_bytes": max(int(row["native_feature_state_bytes"]) for row in rows),
        "min_compression_ratio": min(
            float(row["state_compression_ratio"]) for row in rows
        ),
        "minimum_task_accuracy_delta": min(
            float(row["accuracy_delta"]) for row in task_rows
        ),
        "vision_seconds_mean": float(
            np.mean([float(row["vision_seconds"]) for row in rows])
        ),
        "inference_seconds_mean": float(
            np.mean([float(row["inference_seconds"]) for row in rows])
        ),
    }
    decision = classify_confirmation(metrics)
    summary = {
        "decision": decision,
        "claim_tier": "onevision_pca_r456_untouched_task_confirmation",
        "candidate": "pca_r456_s0",
        "metrics": metrics,
        "missing_sample_ids": missing_ids,
        "task_summary": task_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", summary)
    write_csv(args.out_dir / "paired_samples.csv", rows)
    write_csv(args.out_dir / "task_summary.csv", task_rows)

    report = [
        "# OneVision PCA-r456 Untouched-Task Confirmation",
        "",
        f"Decision: **{decision}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {len(rows)} |",
        f"| Full accuracy | {metrics['reference_accuracy']:.2%} |",
        f"| PCA-r456 accuracy | {metrics['candidate_accuracy']:.2%} |",
        f"| Paired delta | {metrics['accuracy_delta']:+.2%} [{low:+.2%}, {high:+.2%}] |",
        f"| Prediction agreement | {metrics['prediction_agreement']:.2%} |",
        f"| Harmful / beneficial flips | {harmful} / {metrics['beneficial_flips']} |",
        f"| Harmful one-sided upper 95% | {metrics['harmful_upper_95']:.3%} |",
        f"| Candidate KL mean / P95 | {metrics['candidate_kl_mean']:.6f} / {metrics['candidate_kl_p95']:.6f} |",
        f"| Feature relative L2 | {metrics['feature_relative_l2_mean']:.2%} |",
        f"| State bytes / minimum compression | {metrics['max_state_bytes']:,} / {metrics['min_compression_ratio']:.3f}x |",
        "",
        "This result evaluates reader preservation only. It does not establish",
        "serialization, prefill, TTFT, or end-to-end speedup.",
    ]
    (args.out_dir / "RESULTS_ANALYSIS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(decision == "INVALID")


if __name__ == "__main__":
    raise SystemExit(main())
