from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aggregate_mvbench_llava import paired_policy_comparisons


POLICY_LABELS = {
    "exact_recent": "Exact recent",
    "recent_pool_query_topk": "Recent pool + top-k",
    "recent_pool_query_mmr": "Recent pool + MMR",
    "learned_recent_query_topk": "Learned recent-pool top-k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--reference", default="exact_recent")
    parser.add_argument("--primary-policy", default="recent_pool_query_topk")
    parser.add_argument(
        "--learned-policy",
        default="learned_recent_query_topk",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def task_gains(
    rows: list[dict[str, str]],
    *,
    policy: str,
    reference: str,
) -> dict[str, float]:
    lookup = {
        (row["task"], row["policy"]): float(row["accuracy"])
        for row in rows
    }
    tasks = sorted(
        task
        for task, candidate in lookup
        if candidate == reference
    )
    return {
        task: lookup[(task, policy)] - lookup[(task, reference)]
        for task in tasks
    }


def transfer_gate(gains: dict[str, float], overall_gain: float) -> bool:
    positive_tasks = sum(gain > 0.0 for gain in gains.values())
    return overall_gain > 0.0 and positive_tasks >= 2


def comparison_is_significant(
    row: dict[str, object],
    *,
    alpha: float = 0.05,
) -> bool:
    return float(row["mcnemar_exact_p"]) < alpha


def format_interval(row: dict[str, str]) -> str:
    return (
        f"[{float(row['bootstrap_ci95_low']):+.2%}, "
        f"{float(row['bootstrap_ci95_high']):+.2%}]"
    )


def main() -> int:
    args = parse_args()
    aggregate_dir = args.aggregate_dir
    overall = read_csv(aggregate_dir / "overall_accuracy.csv")
    tasks = read_csv(aggregate_dir / "task_accuracy.csv")
    paired = read_csv(
        aggregate_dir / f"paired_vs_{args.reference}.csv"
    )
    predictions = read_csv(aggregate_dir / "predictions.csv")
    summary = json.loads(
        (aggregate_dir / "aggregate_summary.json").read_text(
            encoding="utf-8"
        )
    )
    configuration_path = aggregate_dir.parent / "configuration.json"
    configuration = (
        json.loads(configuration_path.read_text(encoding="utf-8"))
        if configuration_path.exists()
        else {}
    )
    overall_lookup = {row["policy"]: row for row in overall}
    paired_lookup = {row["policy"]: row for row in paired}
    native_feature_memory = bool(
        configuration.get("native_feature_memory", False)
    )
    validation_path = aggregate_dir / "full_validation.json"
    full_validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.exists()
        else {}
    )
    accounting = {}
    for row in tasks:
        accounting.setdefault(row["policy"], row)

    report_policies = [
        policy
        for policy in (
            args.reference,
            args.primary_policy,
            "recent_pool_query_mmr",
            args.learned_policy,
        )
        if policy in overall_lookup
    ]
    primary_paired = paired_lookup[args.primary_policy]
    learned_paired = paired_lookup[args.learned_policy]
    learned_vs_topk = next(
        row
        for row in paired_policy_comparisons(
            predictions,
            reference=args.primary_policy,
            seed=20260718,
        )
        if row["policy"] == args.learned_policy
    )
    learned_vs_mmr = next(
        row
        for row in paired_policy_comparisons(
            predictions,
            reference="recent_pool_query_mmr",
            seed=20260718,
        )
        if row["policy"] == args.learned_policy
    )
    primary_gains = task_gains(
        tasks,
        policy=args.primary_policy,
        reference=args.reference,
    )
    learned_gains = task_gains(
        tasks,
        policy=args.learned_policy,
        reference=args.reference,
    )
    primary_pass = transfer_gate(
        primary_gains,
        float(primary_paired["accuracy_gain"]),
    )
    learned_pass = transfer_gate(
        learned_gains,
        float(learned_paired["accuracy_gain"]),
    )
    learned_significant_vs_reference = comparison_is_significant(
        learned_paired
    )
    learned_significant_vs_topk = comparison_is_significant(
        learned_vs_topk
    )
    learned_significant_vs_mmr = comparison_is_significant(
        learned_vs_mmr
    )

    lines = [
        "# MVBench Query-Memory LLaVA Anchor",
        "",
        "## Validity",
        "",
        f"- Completed checkpoints: {summary['checkpoint_count']}.",
        f"- Prediction rows: {summary['prediction_rows']}.",
        f"- Configuration fingerprints: "
        f"{len(summary.get('fingerprints', []))}.",
        f"- Selection-manifest SHA256: "
        f"`{configuration.get('selection_manifest_sha256', 'unknown')}`.",
        (
            "- All policies are evaluated on the same examples with the "
            "same cached-feature and visual-token budgets."
            if native_feature_memory
            else "- All policies are evaluated on the same raw-frame "
            "examples with the same visual-token budget."
        ),
        "",
        "## Overall Results",
        "",
        "| Policy | Accuracy | Gain vs recent | Paired 95% CI | "
        "Better / worse | McNemar p | Parse rate | End-to-end | Model only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in report_policies:
        row = overall_lookup[policy]
        if policy == args.reference:
            gain = "reference"
            interval = "reference"
            flips = "reference"
            p_value = "reference"
        else:
            comparison = paired_lookup[policy]
            gain = f"{float(comparison['accuracy_gain']):+.2%}"
            interval = format_interval(comparison)
            flips = (
                f"{comparison['better_samples']} / "
                f"{comparison['worse_samples']}"
            )
            p_value = f"{float(comparison['mcnemar_exact_p']):.4f}"
        lines.append(
            f"| {POLICY_LABELS.get(policy, policy)} "
            f"| {float(row['macro_task_accuracy']):.2%} "
            f"| {gain} | {interval} | {flips} | {p_value} "
            f"| {float(row['parsed_rate']):.2%} "
            f"| {float(row['mean_policy_seconds']):.2f} s "
            f"| {float(row['mean_inference_seconds']):.2f} s |"
        )

    lines.extend(
        [
            "",
            "## Direct Selector Comparisons",
            "",
            f"- Learned versus recent-pool top-k: "
            f"{float(learned_vs_topk['accuracy_gain']):+.2%}, "
            f"95% CI "
            f"[{float(learned_vs_topk['bootstrap_ci95_low']):+.2%}, "
            f"{float(learned_vs_topk['bootstrap_ci95_high']):+.2%}], "
            f"{learned_vs_topk['better_samples']} better / "
            f"{learned_vs_topk['worse_samples']} worse.",
            f"- Learned versus recent-pool MMR: "
            f"{float(learned_vs_mmr['accuracy_gain']):+.2%}, "
            f"95% CI "
            f"[{float(learned_vs_mmr['bootstrap_ci95_low']):+.2%}, "
            f"{float(learned_vs_mmr['bootstrap_ci95_high']):+.2%}], "
            f"{learned_vs_mmr['better_samples']} better / "
            f"{learned_vs_mmr['worse_samples']} worse.",
        ]
    )
    if (
        learned_significant_vs_reference
        and not learned_significant_vs_topk
        and not learned_significant_vs_mmr
    ):
        lines.append(
            "- The learned readout is significant versus exact recent, but "
            "not versus the two query-only recent-pool controls."
        )
    elif learned_significant_vs_reference:
        lines.append(
            "- The learned readout is significant versus exact recent. "
            "At least one direct selector comparison is also significant."
        )
    else:
        lines.append(
            "- The learned readout is not statistically significant versus "
            "exact recent or either query-only recent-pool control."
        )

    if native_feature_memory and full_validation:
        policy_rows = int(full_validation.get("policy_rows", 0))
        prediction_mismatches = int(
            full_validation.get("prediction_mismatches_vs_raw", 0)
        )
        frame_mismatches = int(
            full_validation.get(
                "selected_frame_mismatches_vs_raw",
                0,
            )
        )
        correctness_mismatches = int(
            full_validation.get("correctness_mismatches_vs_raw", 0)
        )
        if policy_rows:
            lines.extend(
                [
                    "",
                    "## Native/Raw Path Consistency",
                    "",
                    f"- Selected-frame agreement with the raw-frame "
                    f"anchor: {policy_rows - frame_mismatches}/{policy_rows}.",
                    f"- Prediction agreement: "
                    f"{policy_rows - prediction_mismatches}/{policy_rows}.",
                    f"- Correctness agreement: "
                    f"{policy_rows - correctness_mismatches}/{policy_rows}.",
                    "- The native path is task-equivalent at this scale but "
                    "not bit-exact; FP16 visual encoding and different batch "
                    "shapes can change borderline generations.",
                ]
            )

    selected_tasks = sorted(primary_gains)
    lines.extend(
        [
            "",
            "## Task Breakdown",
            "",
            "| Task | Exact recent | Top-k gain | MMR gain | "
            "Learned gain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    task_lookup = {
        (row["task"], row["policy"]): float(row["accuracy"])
        for row in tasks
    }
    mmr_gains = (
        task_gains(
            tasks,
            policy="recent_pool_query_mmr",
            reference=args.reference,
        )
        if "recent_pool_query_mmr" in overall_lookup
        else {}
    )
    for task in selected_tasks:
        lines.append(
            f"| {task} "
            f"| {task_lookup[(task, args.reference)]:.2%} "
            f"| {primary_gains[task]:+.2%} "
            f"| {mmr_gains.get(task, float('nan')):+.2%} "
            f"| {learned_gains[task]:+.2%} |"
        )

    visual_tokens = int(accounting[args.reference]["visual_tokens"])
    llm_visual_bytes = int(
        accounting[args.reference]["llm_visual_token_bytes"]
    )
    reference_state_bytes = int(
        accounting[args.reference]["selection_state_proxy_bytes"]
    )
    learned_state_bytes = int(
        accounting[args.learned_policy]["selection_state_proxy_bytes"]
    )
    matched_state = reference_state_bytes == learned_state_bytes
    lines.extend(
        [
            "",
            "## Budget",
            "",
            f"- Visual tokens per example: {visual_tokens}.",
            f"- LLM visual-token activation proxy: "
            f"{llm_visual_bytes / 1024:.2f} KiB.",
            "- End-to-end policy latency includes selection bookkeeping, "
            "video decoding, image preprocessing, and model generation. "
            "The model-only column measures `model.generate`.",
        ]
    )
    if matched_state:
        lines.append(
            f"- Matched provisioned persistent state: "
            f"{reference_state_bytes / 1024:.2f} KiB per policy."
        )
    else:
        lines.extend(
            [
                f"- Exact-recent selection state: "
                f"{reference_state_bytes / 1024:.2f} KiB.",
                f"- Learned recent-pool selection state: "
                f"{learned_state_bytes / 1024:.2f} KiB.",
                "- The query policies retain a 16-vector pool while exact "
                "recent retains 8 vectors, so this anchor is not a "
                "matched-state deployment comparison.",
            ]
        )
    if native_feature_memory:
        learned_overall = overall_lookup[args.learned_policy]
        lines.extend(
            [
                "- The 16-frame projected visual-feature cache is included "
                "in persistent-state bytes.",
                "- Query-time answers read cached visual tokens directly; "
                "there is no source-video replay at read time.",
                f"- Mean native write: "
                f"{float(learned_overall['mean_feature_cache_write_seconds']):.2f} s "
                f"(decode "
                f"{float(learned_overall['mean_decode_seconds']):.2f} s, "
                f"preprocess "
                f"{float(learned_overall['mean_preprocess_seconds']):.2f} s, "
                f"vision encode "
                f"{float(learned_overall['mean_vision_encode_seconds']):.2f} s).",
                f"- Mean cached read and generation: "
                f"{float(learned_overall['mean_inference_seconds']):.2f} s.",
            ]
        )
    else:
        lines.append(
            "- Raw-frame replay remains an anchor-only mechanism and is not "
            "a bounded persistent-state result."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Frozen top-k primary transfer gate: "
            f"{'PASS' if primary_pass else 'FAIL'}.",
            f"- Frozen learned-readout transfer gate: "
            f"{'PASS' if learned_pass else 'FAIL'}.",
        ]
    )
    if primary_pass or learned_pass:
        if native_feature_memory and matched_state:
            lines.append(
                "- The matched-state native-feature anchor may proceed to a "
                "trainable writer/readout experiment and second-encoder "
                "replication."
            )
            lines.append(
                "- The current selections still come from a frozen CLIP "
                "ranker, so this result does not yet establish a native "
                "learned memory."
            )
        else:
            lines.append(
                "- A passing policy may proceed to a native learned-memory "
                "experiment, but this anchor alone does not establish a "
                "streaming memory contribution."
            )
            lines.append(
                "- The next experiment must match persistent bytes and "
                "isolate learned ranking from the benefit of retaining a "
                "larger recent pool."
            )
    else:
        lines.append(
            "- No tested selector has positive LLaVA gain across at least "
            "two tasks; stop this frame-retrieval branch before native "
            "memory training."
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Task accuracy](task_accuracy_by_policy.png)",
            "",
            "![Accuracy and latency](accuracy_latency.png)",
            "",
            f"![Paired gains](paired_gain_vs_{args.reference}.png)",
            "",
        ]
    )
    if (
        aggregate_dir / "native_feature_memory_validation.png"
    ).exists():
        lines.extend(
            [
                "![Native feature-memory validation]"
                "(native_feature_memory_validation.png)",
                "",
            ]
        )
    report_path = aggregate_dir / "RESULTS_ANALYSIS.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
