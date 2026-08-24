from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_VARIANTS = ("pca_only", "euclidean_s4", "fisher_s4", "mixed_s4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--baseline-variant", default="euclidean_s4")
    parser.add_argument("--candidate-variants", default="fisher_s4,mixed_s4")
    parser.add_argument(
        "--claim-tier", default="transductive_native_fisher_support_oracle"
    )
    parser.add_argument("--title", default="Reader-Quotient Sparse-Support Oracle")
    parser.add_argument(
        "--boundary-note",
        default=(
            "The result is a transductive support oracle. It cannot be read as an "
            "online writer, task-accuracy result, strong-reader replication, or "
            "latency claim."
        ),
    )
    parser.add_argument(
        "--decision-rule",
        choices=("reader_quotient_v1", "onevision_replication"),
        default="reader_quotient_v1",
    )
    return parser.parse_args()


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    failure_paths = sorted(run_dir.glob("failures_shard_*.json"))
    if (run_dir / "failures.json").exists():
        failure_paths.append(run_dir / "failures.json")
    for path in failure_paths:
        failures = json.loads(path.read_text(encoding="utf-8"))
        if failures:
            raise ValueError(f"run contains failures in {path}: {len(failures)}")
    rows = []
    for path in sorted((run_dir / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    if not rows:
        raise ValueError("no checkpoint rows found")
    return rows


def validate_rows(
    rows: list[dict[str, object]],
    variants: tuple[str, ...],
) -> dict[str, dict[str, dict[str, object]]]:
    samples: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        sample_id = str(row["sample_id"])
        variant = str(row["variant"])
        if variant not in variants:
            raise ValueError(f"unknown variant: {variant}")
        if variant in samples[sample_id]:
            raise ValueError(f"duplicate row: {sample_id}/{variant}")
        numeric_fields = (
            "candidate_kl",
            "vocabulary_kl",
            "feature_relative_l2",
            "instrumentation_max_abs",
        )
        if not all(math.isfinite(float(row[field])) for field in numeric_fields):
            raise ValueError(f"non-finite metric: {sample_id}/{variant}")
        if float(row["instrumentation_max_abs"]) > 1e-3:
            raise ValueError(f"instrumentation mismatch: {sample_id}/{variant}")
        if "injection_max_abs" in row:
            injection_error = float(row["injection_max_abs"])
            if not math.isfinite(injection_error) or injection_error > 1e-3:
                raise ValueError(f"feature injection mismatch: {sample_id}/{variant}")
        samples[sample_id][variant] = row
    expected = set(variants)
    for sample_id, variants in samples.items():
        if set(variants) != expected:
            raise ValueError(f"incomplete variants for {sample_id}: {sorted(variants)}")
        state_bytes = {
            int(row["native_feature_state_bytes"])
            for variant, row in variants.items()
            if variant != "pca_only"
        }
        if len(state_bytes) != 1:
            raise ValueError(f"s4 payload mismatch for {sample_id}: {state_bytes}")
    return dict(samples)


def bootstrap_reduction(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(samples, len(baseline)))
    baseline_sum = baseline[indices].sum(axis=1)
    candidate_sum = candidate[indices].sum(axis=1)
    reductions = 1.0 - candidate_sum / np.maximum(
        baseline_sum, np.finfo(np.float64).tiny
    )
    return tuple(float(value) for value in np.quantile(reductions, [0.025, 0.975]))


def classify(summary: dict[str, float]) -> str:
    reduction = summary["aggregate_candidate_kl_reduction"]
    tail_ratio = summary["candidate_kl_p95_ratio"]
    top1_delta = summary["candidate_top1_match_delta"]
    if reduction >= 0.25 and tail_ratio <= 1.0 and top1_delta >= -0.01:
        return "GO"
    if reduction >= 0.10 and tail_ratio <= 1.0 and top1_delta >= -0.01:
        return "WEAK"
    if reduction < -0.10 or tail_ratio > 1.10 or top1_delta < -0.01:
        return "ADVERSE"
    return "NULL"


def classify_onevision_replication(
    summary: dict[str, float],
    *,
    positive_tasks: int,
) -> str:
    reduction = summary["aggregate_candidate_kl_reduction"]
    tail_ratio = summary["candidate_kl_p95_ratio"]
    if reduction < -0.10 or tail_ratio > 1.10:
        return "ADVERSE"
    if reduction >= 0.25 and tail_ratio <= 1.0 and positive_tasks >= 4:
        return "GO"
    if reduction > 0.0:
        return "BOUNDARY"
    return "NULL"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    candidate_variants = tuple(
        value.strip() for value in args.candidate_variants.split(",") if value.strip()
    )
    if args.baseline_variant not in variants:
        raise ValueError("baseline variant is not present in --variants")
    if not candidate_variants or not set(candidate_variants).issubset(variants):
        raise ValueError("candidate variants must be a non-empty subset of --variants")
    rows = load_rows(args.run_dir)
    samples = validate_rows(rows, variants)
    sample_ids = sorted(samples)
    baseline = np.asarray(
        [float(samples[sample_id][args.baseline_variant]["candidate_kl"]) for sample_id in sample_ids]
    )
    baseline_p95 = float(np.quantile(baseline, 0.95))
    baseline_vocabulary = np.asarray(
        [float(samples[sample_id][args.baseline_variant]["vocabulary_kl"]) for sample_id in sample_ids]
    )
    baseline_top1 = float(
        np.mean(
            [
                int(samples[sample_id][args.baseline_variant]["candidate_top1_match"])
                for sample_id in sample_ids
            ]
        )
    )
    summaries = []
    for variant_index, variant in enumerate(variants):
        values = np.asarray(
            [float(samples[sample_id][variant]["candidate_kl"]) for sample_id in sample_ids]
        )
        top1 = float(
            np.mean(
                [int(samples[sample_id][variant]["candidate_top1_match"]) for sample_id in sample_ids]
            )
        )
        reduction = 1.0 - float(values.sum()) / max(
            float(baseline.sum()), np.finfo(np.float64).tiny
        )
        reduction_ci = bootstrap_reduction(
            baseline,
            values,
            samples=args.bootstrap_samples,
            seed=args.seed + variant_index,
        )
        p95 = float(np.quantile(values, 0.95))
        vocabulary_values = np.asarray(
            [float(samples[sample_id][variant]["vocabulary_kl"]) for sample_id in sample_ids]
        )
        summary = {
            "variant": variant,
            "samples": len(sample_ids),
            "candidate_kl_sum": float(values.sum()),
            "candidate_kl_mean": float(values.mean()),
            "candidate_kl_median": float(np.median(values)),
            "candidate_kl_p95": p95,
            "aggregate_candidate_kl_reduction": reduction,
            "reduction_ci95_low": reduction_ci[0],
            "reduction_ci95_high": reduction_ci[1],
            "candidate_kl_p95_ratio": p95
            / max(baseline_p95, np.finfo(np.float64).tiny),
            "candidate_top1_match": top1,
            "candidate_top1_match_delta": top1 - baseline_top1,
            "vocabulary_kl_sum": float(vocabulary_values.sum()),
            "aggregate_vocabulary_kl_reduction": 1.0
            - float(vocabulary_values.sum())
            / max(float(baseline_vocabulary.sum()), np.finfo(np.float64).tiny),
            "mean_answer_logprob_delta": float(
                np.mean(
                    [
                        float(samples[sample_id][variant]["answer_logprob_delta"])
                        for sample_id in sample_ids
                    ]
                )
            ),
            "mean_feature_relative_l2": float(
                np.mean(
                    [
                        float(samples[sample_id][variant]["feature_relative_l2"])
                        for sample_id in sample_ids
                    ]
                )
            ),
            "mean_support_overlap_with_euclidean": float(
                np.mean(
                    [
                        float(
                            samples[sample_id][variant][
                                "support_overlap_with_euclidean"
                            ]
                        )
                        for sample_id in sample_ids
                    ]
                )
            ),
            "paired_kl_wins": int(np.sum(values < baseline)),
            "paired_kl_ties": int(np.sum(values == baseline)),
            "paired_kl_losses": int(np.sum(values > baseline)),
        }
        summary["decision"] = (
            (
                classify(summary)
                if args.decision_rule == "reader_quotient_v1"
                else "PENDING"
            )
            if variant in candidate_variants
            else ("REFERENCE" if variant == args.baseline_variant else "ABLATION")
        )
        summaries.append(summary)

    task_rows = []
    tasks = sorted(
        {
            str(rows_for_sample[args.baseline_variant]["task"])
            for rows_for_sample in samples.values()
        }
    )
    for task in tasks:
        task_ids = [
            sample_id
            for sample_id in sample_ids
            if str(samples[sample_id][args.baseline_variant]["task"]) == task
        ]
        task_baseline = sum(
            float(samples[sample_id][args.baseline_variant]["candidate_kl"])
            for sample_id in task_ids
        )
        for variant in variants:
            if variant == args.baseline_variant:
                continue
            task_candidate = sum(
                float(samples[sample_id][variant]["candidate_kl"])
                for sample_id in task_ids
            )
            task_rows.append(
                {
                    "task": task,
                    "variant": variant,
                    "samples": len(task_ids),
                    "candidate_kl_sum": task_candidate,
                    "euclidean_candidate_kl_sum": task_baseline,
                    "aggregate_reduction": 1.0
                    - task_candidate / max(task_baseline, np.finfo(np.float64).tiny),
                }
            )

    positive_tasks_by_variant = {
        variant: sum(
            float(row["aggregate_reduction"]) > 0.0
            for row in task_rows
            if row["variant"] == variant
        )
        for variant in variants
    }
    for summary in summaries:
        variant = str(summary["variant"])
        positive_tasks = positive_tasks_by_variant[variant]
        summary["positive_task_count"] = positive_tasks
        if (
            args.decision_rule == "onevision_replication"
            and variant in candidate_variants
        ):
            summary["decision"] = classify_onevision_replication(
                summary,
                positive_tasks=positive_tasks,
            )

    paired_rows = []
    for sample_id in sample_ids:
        baseline_row = samples[sample_id][args.baseline_variant]
        for variant in variants:
            if variant == args.baseline_variant:
                continue
            candidate_row = samples[sample_id][variant]
            paired_rows.append(
                {
                    "sample_id": sample_id,
                    "task": baseline_row["task"],
                    "variant": variant,
                    "euclidean_candidate_kl": baseline_row["candidate_kl"],
                    "candidate_kl": candidate_row["candidate_kl"],
                    "candidate_kl_reduction": 1.0
                    - float(candidate_row["candidate_kl"])
                    / max(
                        float(baseline_row["candidate_kl"]),
                        np.finfo(np.float64).tiny,
                    ),
                    "support_overlap_with_euclidean": candidate_row[
                        "support_overlap_with_euclidean"
                    ],
                }
            )

    candidates = [
        summary for summary in summaries if summary["variant"] in candidate_variants
    ]
    decision_order = {
        "GO": 4,
        "BOUNDARY": 3,
        "WEAK": 2,
        "NULL": 1,
        "ADVERSE": 0,
    }
    decision = max(candidates, key=lambda item: decision_order[str(item["decision"])])[
        "decision"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "claim_tier": args.claim_tier,
                "baseline_variant": args.baseline_variant,
                "candidate_variants": list(candidate_variants),
                "samples": len(sample_ids),
                "variant_summary": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "variant_summary.csv", summaries)
    write_csv(args.output_dir / "task_summary.csv", task_rows)
    write_csv(args.output_dir / "paired_samples.csv", paired_rows)

    lines = [
        f"# {args.title}",
        "",
        f"Decision: `{decision}`; samples: `{len(sample_ids)}`.",
        "",
        "| Variant | Candidate KL reduction | Vocab KL reduction | 95% CI | P95 ratio | Top-1 delta | W/T/L | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {variant} | {reduction:+.2%} | {vocab:+.2%} | [{low:+.2%}, {high:+.2%}] | "
            "{tail:.3f} | {top1:+.2%} | {wins}/{ties}/{losses} | {decision} |".format(
                variant=summary["variant"],
                reduction=summary["aggregate_candidate_kl_reduction"],
                vocab=summary["aggregate_vocabulary_kl_reduction"],
                low=summary["reduction_ci95_low"],
                high=summary["reduction_ci95_high"],
                tail=summary["candidate_kl_p95_ratio"],
                top1=summary["candidate_top1_match_delta"],
                wins=summary["paired_kl_wins"],
                ties=summary["paired_kl_ties"],
                losses=summary["paired_kl_losses"],
                decision=summary["decision"],
            )
        )
    lines.extend(
        [
            "",
            args.boundary_note,
        ]
    )
    (args.output_dir / "RESULTS_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "samples": len(sample_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
