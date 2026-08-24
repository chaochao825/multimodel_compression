from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ANCHOR_VARIANT = "euclidean_r384_s4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def bootstrap_reduction(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(samples, len(baseline)))
    reductions = 1.0 - candidate[indices].sum(axis=1) / np.maximum(
        baseline[indices].sum(axis=1),
        np.finfo(np.float64).tiny,
    )
    return tuple(float(value) for value in np.quantile(reductions, [0.025, 0.975]))


def classify_candidate(
    *,
    reduction: float,
    p95_ratio: float,
    positive_tasks: int,
    top1_delta: float,
    absolute_kl_ratio: float,
    state_bytes: int,
) -> str:
    if reduction < -0.10 or p95_ratio > 1.10 or top1_delta < -0.01:
        return "ADVERSE"
    if (
        reduction >= 0.25
        and p95_ratio <= 1.0
        and positive_tasks >= 4
        and top1_delta >= 0.0
        and absolute_kl_ratio <= 1.0
        and state_bytes <= 2_867_328
    ):
        return "GO"
    if reduction > 0.0:
        return "BOUNDARY"
    return "NULL"


def load_rows(run_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    configuration = json.loads(
        (run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    for path in sorted(run_dir.glob("failures_shard_*.json")):
        failures = json.loads(path.read_text(encoding="utf-8"))
        if failures:
            raise ValueError(f"run contains failures in {path}: {len(failures)}")
    rows = []
    for path in sorted((run_dir / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    if not rows:
        raise ValueError("no checkpoint rows found")
    return rows, configuration


def validate_rows(
    rows: list[dict[str, object]],
    configuration: dict[str, object],
) -> dict[str, dict[str, dict[str, object]]]:
    expected_variants = set(configuration["variants"])
    expected_samples = set(configuration["sample_ids"])
    samples: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        sample_id = str(row["sample_id"])
        variant = str(row["variant"])
        if sample_id not in expected_samples or variant not in expected_variants:
            raise ValueError(f"unexpected row: {sample_id}/{variant}")
        if variant in samples[sample_id]:
            raise ValueError(f"duplicate row: {sample_id}/{variant}")
        for field in (
            "candidate_kl",
            "vocabulary_kl",
            "feature_relative_l2",
            "injection_max_abs",
            "instrumentation_max_abs",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite {field}: {sample_id}/{variant}")
        if float(row["injection_max_abs"]) > 1e-3:
            raise ValueError(f"feature injection mismatch: {sample_id}/{variant}")
        if float(row["instrumentation_max_abs"]) > 1e-3:
            raise ValueError(f"instrumentation mismatch: {sample_id}/{variant}")
        samples[sample_id][variant] = row
    if set(samples) != expected_samples:
        raise ValueError("checkpoint sample set differs from configuration")
    for sample_id, variants in samples.items():
        if set(variants) != expected_variants:
            raise ValueError(f"incomplete variants for {sample_id}")
    return dict(samples)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows, configuration = load_rows(args.run_dir)
    samples = validate_rows(rows, configuration)
    sample_ids = sorted(samples)
    anchor = np.asarray(
        [float(samples[sample_id][ANCHOR_VARIANT]["candidate_kl"]) for sample_id in sample_ids]
    )
    anchor_sum = float(anchor.sum())
    summaries = []
    task_rows = []
    paired_rows = []
    for allocation_index, (rank, support) in enumerate(configuration["allocations"]):
        identifier = f"r{rank}_s{support}"
        baseline_variant = f"euclidean_{identifier}"
        baseline = np.asarray(
            [float(samples[sample_id][baseline_variant]["candidate_kl"]) for sample_id in sample_ids]
        )
        baseline_p95 = float(np.quantile(baseline, 0.95))
        baseline_top1 = float(
            np.mean(
                [int(samples[sample_id][baseline_variant]["candidate_top1_match"]) for sample_id in sample_ids]
            )
        )
        baseline_accuracy = float(
            np.mean(
                [
                    int(samples[sample_id][baseline_variant]["candidate_correct"])
                    for sample_id in sample_ids
                ]
            )
        )
        families = ("euclidean",) if support == 0 else ("euclidean", "fisher", "mixed")
        for family_index, family in enumerate(families):
            variant = f"{family}_{identifier}"
            values = np.asarray(
                [float(samples[sample_id][variant]["candidate_kl"]) for sample_id in sample_ids]
            )
            p95 = float(np.quantile(values, 0.95))
            top1 = float(
                np.mean(
                    [int(samples[sample_id][variant]["candidate_top1_match"]) for sample_id in sample_ids]
                )
            )
            tasks = sorted({str(samples[sample_id][variant]["task"]) for sample_id in sample_ids})
            positive_tasks = 0
            for task in tasks:
                task_ids = [
                    sample_id
                    for sample_id in sample_ids
                    if str(samples[sample_id][variant]["task"]) == task
                ]
                task_baseline = sum(
                    float(samples[sample_id][baseline_variant]["candidate_kl"])
                    for sample_id in task_ids
                )
                task_candidate = sum(
                    float(samples[sample_id][variant]["candidate_kl"])
                    for sample_id in task_ids
                )
                task_reduction = 1.0 - task_candidate / max(
                    task_baseline,
                    np.finfo(np.float64).tiny,
                )
                if family != "euclidean" and task_reduction > 0.0:
                    positive_tasks += 1
                task_rows.append(
                    {
                        "allocation_id": identifier,
                        "rank": rank,
                        "residual_tokens_per_frame": support,
                        "task": task,
                        "variant": variant,
                        "samples": len(task_ids),
                        "paired_reduction": task_reduction,
                    }
                )
            reduction = 1.0 - float(values.sum()) / max(
                float(baseline.sum()),
                np.finfo(np.float64).tiny,
            )
            ci_low, ci_high = bootstrap_reduction(
                baseline,
                values,
                samples=args.bootstrap_samples,
                seed=args.seed + allocation_index * 3 + family_index,
            )
            state_bytes_values = {
                int(samples[sample_id][variant]["native_feature_state_bytes"])
                for sample_id in sample_ids
            }
            if len(state_bytes_values) != 1:
                raise ValueError(f"state bytes differ across samples for {variant}")
            state_bytes = state_bytes_values.pop()
            absolute_kl_ratio = float(values.sum()) / max(
                anchor_sum,
                np.finfo(np.float64).tiny,
            )
            leave_one_out_reductions = [
                1.0
                - (float(values.sum()) - float(candidate_value))
                / max(
                    float(baseline.sum()) - float(baseline_value),
                    np.finfo(np.float64).tiny,
                )
                for baseline_value, candidate_value in zip(
                    baseline,
                    values,
                    strict=True,
                )
            ]
            candidate_accuracy = float(
                np.mean(
                    [
                        int(samples[sample_id][variant]["candidate_correct"])
                        for sample_id in sample_ids
                    ]
                )
            )
            decision = "REFERENCE"
            if family != "euclidean":
                decision = classify_candidate(
                    reduction=reduction,
                    p95_ratio=p95 / max(baseline_p95, np.finfo(np.float64).tiny),
                    positive_tasks=positive_tasks,
                    top1_delta=top1 - baseline_top1,
                    absolute_kl_ratio=absolute_kl_ratio,
                    state_bytes=state_bytes,
                )
            summaries.append(
                {
                    "allocation_id": identifier,
                    "rank": rank,
                    "residual_tokens_per_frame": support,
                    "variant": variant,
                    "state_bytes": state_bytes,
                    "compression_ratio": float(
                        samples[sample_ids[0]][variant]["state_compression_ratio"]
                    ),
                    "candidate_kl_sum": float(values.sum()),
                    "paired_candidate_kl_reduction": reduction,
                    "reduction_ci95_low": ci_low,
                    "reduction_ci95_high": ci_high,
                    "candidate_kl_p95_ratio": p95
                    / max(baseline_p95, np.finfo(np.float64).tiny),
                    "absolute_kl_ratio_to_r384_s4_euclidean": absolute_kl_ratio,
                    "positive_task_count": positive_tasks,
                    "candidate_top1_match": top1,
                    "candidate_top1_match_delta": top1 - baseline_top1,
                    "candidate_accuracy": candidate_accuracy,
                    "candidate_accuracy_delta": candidate_accuracy
                    - baseline_accuracy,
                    "minimum_leave_one_out_paired_reduction": min(
                        leave_one_out_reductions
                    ),
                    "mean_feature_relative_l2": float(
                        np.mean(
                            [float(samples[sample_id][variant]["feature_relative_l2"]) for sample_id in sample_ids]
                        )
                    ),
                    "mean_support_overlap_with_euclidean": float(
                        np.mean(
                            [float(samples[sample_id][variant]["support_overlap_with_euclidean"]) for sample_id in sample_ids]
                        )
                    ),
                    "paired_kl_wins": int(np.sum(values < baseline)),
                    "paired_kl_ties": int(np.sum(values == baseline)),
                    "paired_kl_losses": int(np.sum(values > baseline)),
                    "decision": decision,
                }
            )
            if family != "euclidean":
                for sample_id, baseline_value, candidate_value in zip(
                    sample_ids,
                    baseline,
                    values,
                    strict=True,
                ):
                    paired_rows.append(
                        {
                            "sample_id": sample_id,
                            "task": samples[sample_id][variant]["task"],
                            "allocation_id": identifier,
                            "variant": variant,
                            "paired_euclidean_kl": baseline_value,
                            "candidate_kl": candidate_value,
                            "paired_reduction": 1.0
                            - candidate_value
                            / max(baseline_value, np.finfo(np.float64).tiny),
                        }
                    )

    candidates = [row for row in summaries if row["decision"] != "REFERENCE"]
    go_candidates = [row for row in candidates if row["decision"] == "GO"]
    if go_candidates:
        selected = min(go_candidates, key=lambda row: float(row["candidate_kl_sum"]))
        decision = "GO"
    else:
        selected = min(candidates, key=lambda row: float(row["candidate_kl_sum"]))
        decision = "BOUNDARY" if any(
            row["decision"] == "BOUNDARY" for row in candidates
        ) else max(
            (str(row["decision"]) for row in candidates),
            key={"ADVERSE": 0, "NULL": 1}.get,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "variant_summary.csv", summaries)
    write_csv(args.output_dir / "task_summary.csv", task_rows)
    write_csv(args.output_dir / "paired_samples.csv", paired_rows)
    summary = {
        "decision": decision,
        "claim_tier": "onevision_equal_budget_rank_support_selection",
        "samples": len(sample_ids),
        "anchor_variant": ANCHOR_VARIANT,
        "selected_variant": selected["variant"],
        "confirmation_authorized": decision == "GO",
        "variant_summary": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# OneVision Equal-Budget Rank/Support Allocation",
        "",
        f"Decision: `{decision}`; selected diagnostic endpoint: `{selected['variant']}`; "
        f"fresh confirmation authorized: `{decision == 'GO'}`.",
        "",
        "| Variant | Bytes | KL sum | Paired reduction | Min LOO | 95% CI | P95 ratio | Absolute/anchor | Positive tasks | Top-1 delta | Accuracy delta | Feature L2 | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {variant} | {state_bytes:,} | {candidate_kl_sum:.6f} | {paired_candidate_kl_reduction:+.2%} | {minimum_leave_one_out_paired_reduction:+.2%} | "
            "[{reduction_ci95_low:+.2%},{reduction_ci95_high:+.2%}] | {candidate_kl_p95_ratio:.3f} | "
            "{absolute_kl_ratio_to_r384_s4_euclidean:.3f} | {positive_task_count}/5 | "
            "{candidate_top1_match_delta:+.2%} | {candidate_accuracy_delta:+.2%} | {mean_feature_relative_l2:.2%} | {decision} |".format(**row)
        )
    lines.extend(
        [
            "",
            "This is an allocation-selection result on observed samples. Only a GO may enter a separately frozen, untouched-task confirmation; no scorer or deployment claim is authorized here.",
        ]
    )
    (args.output_dir / "RESULTS_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "selected_variant": selected["variant"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
