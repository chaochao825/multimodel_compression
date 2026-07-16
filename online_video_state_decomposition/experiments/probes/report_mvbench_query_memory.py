from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


POLICY_LABELS = {
    "exact_recent": "Exact recent",
    "offline_uniform": "Offline uniform",
    "recent_pool_query_topk": "Recent pool + top-k",
    "recent_pool_query_mmr": "Recent pool + MMR",
    "reservoir_recent_query_mmr": "Reservoir + recent + MMR",
    "diverse_recent_query_topk": "Diverse + recent + top-k",
    "diverse_recent_query_mmr": "Diverse + recent + MMR",
    "calibrated_diverse_recent_query_mmr": (
        "Calibrated option-aware MMR"
    ),
    "learned_recent_query_topk": "Learned recent-pool top-k",
    "offline_full_query_mmr": "Full history + MMR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: object, *, signed: bool = False) -> str:
    numeric = float(value) * 100.0
    prefix = "+" if signed and numeric > 0 else ""
    return f"{prefix}{numeric:.2f}%"


def integer(value: object) -> int:
    return int(float(value))


def policy_label(policy: object) -> str:
    value = str(policy)
    return POLICY_LABELS.get(value, value)


def find_policy(
    rows: list[dict[str, str]],
    policy: str,
) -> dict[str, str]:
    return next(row for row in rows if row["policy"] == policy)


def render_method_table(
    overall: list[dict[str, str]],
    paired: dict[str, dict[str, str]],
) -> list[str]:
    lines = [
        "| Policy | Macro accuracy | Gain vs recent | Paired 95% CI | "
        "State KiB | Retrieval MFLOPs | Bounded | Option-aware |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in sorted(
        overall,
        key=lambda value: float(value["macro_task_accuracy"]),
        reverse=True,
    ):
        comparison = paired.get(row["policy"])
        interval = (
            f"[{percent(comparison['bootstrap_ci95_low'], signed=True)}, "
            f"{percent(comparison['bootstrap_ci95_high'], signed=True)}]"
            if comparison
            else "reference"
        )
        lines.append(
            "| "
            f"{policy_label(row['policy'])} | "
            f"{percent(row['macro_task_accuracy'])} | "
            f"{percent(row['macro_gain_vs_exact_recent'], signed=True)} | "
            f"{interval} | "
            f"{integer(row['total_state_bytes']) / 1024:.2f} | "
            f"{integer(row['estimated_retrieval_flops']) / 1e6:.3f} | "
            f"{'yes' if integer(row['online_bounded']) else 'no'} | "
            f"{'yes' if integer(row['option_aware']) else 'no'} |"
        )
    return lines


def render_task_table(
    task_rows: list[dict[str, str]],
    *,
    reference: str,
    policies: list[str],
) -> list[str]:
    tasks = sorted({row["task"] for row in task_rows})
    lookup = {
        (row["task"], row["policy"]): float(row["accuracy"])
        for row in task_rows
    }
    lines = [
        "| Task | "
        + " | ".join(policy_label(policy) for policy in policies)
        + " |",
        "|---|" + "---:|" * len(policies),
    ]
    for task in tasks:
        reference_accuracy = lookup[(task, reference)]
        values = []
        for policy in policies:
            accuracy = lookup[(task, policy)]
            if policy == reference:
                values.append(percent(accuracy))
            else:
                gain = accuracy - reference_accuracy
                values.append(
                    f"{percent(accuracy)} ({percent(gain, signed=True)})"
                )
        lines.append(f"| {task} | " + " | ".join(values) + " |")
    return lines


def main() -> int:
    args = parse_args()
    aggregate_dir = args.aggregate_dir
    out_path = args.out or aggregate_dir / "RESULTS_ANALYSIS.md"
    summary = load_json(aggregate_dir / "aggregate_summary.json")
    validation = load_json(aggregate_dir / "cache_validation.json")
    hyperparameters = load_json(
        aggregate_dir / "selected_hyperparameters.json"
    )
    promotion = load_json(aggregate_dir / "promotion_decision.json")
    overall = [
        row
        for row in read_csv(aggregate_dir / "overall_accuracy.csv")
        if row["split"] == "evaluation"
    ]
    task_rows = [
        row
        for row in read_csv(aggregate_dir / "task_accuracy.csv")
        if row["split"] == "evaluation"
    ]
    paired_rows = read_csv(
        aggregate_dir / "paired_vs_exact_recent.csv"
    )
    paired = {row["policy"]: row for row in paired_rows}
    transfer_path = aggregate_dir / "calibration_to_evaluation.csv"
    transfer_rows = read_csv(transfer_path) if transfer_path.exists() else []
    overlap_path = aggregate_dir / "selection_overlap_vs_reference.csv"
    overlap_rows = read_csv(overlap_path) if overlap_path.exists() else []

    primary = str(promotion["primary_policy"])
    reference = str(promotion.get("reference_policy", "exact_recent"))
    reference_row = find_policy(overall, reference)
    primary_row = find_policy(overall, primary)
    bounded_query_rows = [
        row
        for row in overall
        if integer(row["online_bounded"])
        and integer(row["query_conditioned"])
        and not integer(row["option_aware"])
    ]
    best_bounded_query = max(
        bounded_query_rows,
        key=lambda row: float(row["macro_task_accuracy"]),
    )
    upper_bound = find_policy(overall, "offline_full_query_mmr")
    exploratory_policy = "calibrated_diverse_recent_query_mmr"
    exploratory_row = next(
        (
            row
            for row in overall
            if row["policy"] == exploratory_policy
        ),
        None,
    )
    learned_policy = "learned_recent_query_topk"
    learned_row = next(
        (
            row
            for row in overall
            if row["policy"] == learned_policy
        ),
        None,
    )

    analysis_stage = str(
        summary.get("analysis_stage", "preregistered_mvp")
    )
    primary_label = (
        "Preregistered primary"
        if analysis_stage == "preregistered_mvp"
        else "Frozen confirmatory primary"
    )
    lines = [
        "# MVBench Query-Memory Result Analysis",
        "",
        "## Validity",
        "",
        f"- Cache records: {validation['records']} / "
        f"{validation['expected_records']}.",
        f"- Split and cache checks: "
        f"{'PASS' if validation['valid'] else 'FAIL'}.",
        "- Evaluation labels were not used to select frames or tune "
        "hyperparameters.",
        "- The primary selector uses question-only retrieval. The "
        "calibrated secondary selector uses all candidate options "
        "symmetrically.",
        "- CLIP frame embeddings and raw-frame replay are not counted in "
        "persistent state bytes.",
        "",
        "## Main Result",
        "",
        f"- Exact recent: "
        f"{percent(reference_row['macro_task_accuracy'])} macro accuracy.",
        f"- {primary_label} ({policy_label(primary)}): "
        f"{percent(primary_row['macro_task_accuracy'])}, "
        f"{percent(primary_row['macro_gain_vs_exact_recent'], signed=True)} "
        "versus exact recent.",
        f"- Best bounded question-only selector "
        f"({policy_label(best_bounded_query['policy'])}): "
        f"{percent(best_bounded_query['macro_task_accuracy'])}.",
        f"- Offline full-history upper bound: "
        f"{percent(upper_bound['macro_task_accuracy'])}.",
    ]
    if exploratory_row is not None:
        lines.append(
            f"- Exploratory option-aware selector: "
            f"{percent(exploratory_row['macro_task_accuracy'])}, "
            f"{percent(exploratory_row['macro_gain_vs_exact_recent'], signed=True)} "
            "versus exact recent."
        )
    if learned_row is not None:
        lines.append(
            f"- Exploratory learned readout: "
            f"{percent(learned_row['macro_task_accuracy'])}, "
            f"{percent(learned_row['macro_gain_vs_exact_recent'], signed=True)} "
            "versus exact recent."
        )
    task_policies = list(
        dict.fromkeys(
            [
                reference,
                primary,
                best_bounded_query["policy"],
                exploratory_policy,
                learned_policy,
                "offline_uniform",
                "offline_full_query_mmr",
            ]
        )
    )
    lines.extend(
        [
            f"- Promotion gate: "
            f"{'PASS' if promotion['advance_to_llava_anchor'] else 'FAIL'}.",
            "",
            "## Method Comparison",
            "",
            *render_method_table(overall, paired),
            "",
            "## Task Breakdown",
            "",
            *render_task_table(
                task_rows,
                reference=reference,
                policies=task_policies,
            ),
        ]
    )
    if transfer_rows or overlap_rows:
        lines.extend(["", "## Stability Diagnostics", ""])
    if transfer_rows:
        calibration_order = sorted(
            transfer_rows,
            key=lambda row: float(row["calibration_macro_accuracy"]),
            reverse=True,
        )
        evaluation_order = sorted(
            transfer_rows,
            key=lambda row: float(row["evaluation_macro_accuracy"]),
            reverse=True,
        )
        calibration_rank = {
            row["policy"]: index + 1
            for index, row in enumerate(calibration_order)
        }
        evaluation_rank = {
            row["policy"]: index + 1
            for index, row in enumerate(evaluation_order)
        }
        primary_transfer = next(
            row for row in transfer_rows if row["policy"] == primary
        )
        lines.append(
            f"- {policy_label(primary)} moves from "
            f"{percent(primary_transfer['calibration_macro_accuracy'])} "
            f"(rank {calibration_rank[primary]}/{len(transfer_rows)}) to "
            f"{percent(primary_transfer['evaluation_macro_accuracy'])} "
            f"(rank {evaluation_rank[primary]}/{len(transfer_rows)})."
        )
    primary_overlap = next(
        (
            row
            for row in overlap_rows
            if row["policy"] == primary
        ),
        None,
    )
    if primary_overlap is not None:
        primary_paired = paired[primary]
        lines.append(
            f"- Its selected evidence has mean Jaccard "
            f"{float(primary_overlap['mean_jaccard']):.3f} versus "
            f"{policy_label(reference)}, while "
            f"{primary_paired['tied_samples']}/{primary_paired['paired_samples']} "
            "final predictions remain unchanged."
        )
        lines.append(
            f"- Only {primary_paired['better_samples']} samples improve and "
            f"{primary_paired['worse_samples']} worsen, so the point gain "
            "or loss depends on very few decision flips."
        )
    best_policy = str(best_bounded_query["policy"])
    if best_policy != primary and transfer_rows:
        best_transfer = next(
            row
            for row in transfer_rows
            if row["policy"] == best_policy
        )
        lines.append(
            f"- Exploratory {policy_label(best_policy)} moves from "
            f"{percent(best_transfer['calibration_macro_accuracy'])} "
            f"(rank {calibration_rank[best_policy]}/{len(transfer_rows)}) "
            f"to {percent(best_transfer['evaluation_macro_accuracy'])} "
            f"(rank {evaluation_rank[best_policy]}/{len(transfer_rows)})."
        )
        best_overlap = next(
            (
                row
                for row in overlap_rows
                if row["policy"] == best_policy
            ),
            None,
        )
        best_paired = paired[best_policy]
        if best_overlap is not None:
            lines.append(
                f"- It has mean evidence Jaccard "
                f"{float(best_overlap['mean_jaccard']):.3f}, but "
                f"{best_paired['tied_samples']}/{best_paired['paired_samples']} "
                "predictions are tied; this ranking reversal requires the "
                "untouched-reserve confirmation."
            )
    lines.extend(
        [
            "",
            "## Hyperparameters",
            "",
            f"- Diversity weight: {hyperparameters['diversity_weight']}.",
            f"- Temporal coverage weight: "
            f"{hyperparameters['temporal_weight']}.",
            f"- Option contrast weight: "
            f"{hyperparameters['option_weight']}.",
            f"- Recency feature weight: "
            f"{hyperparameters['recency_weight']}.",
            f"- Novelty feature weight: "
            f"{hyperparameters['novelty_weight']}.",
        ]
    )
    ranker_path = aggregate_dir / "learned_feature_ranker.json"
    if ranker_path.exists():
        ranker = load_json(ranker_path)
        coefficients = ", ".join(
            f"{name}={float(value):+.4f}"
            for name, value in zip(
                ranker["feature_names"],
                ranker["coefficients"],
            )
        )
        lines.append(
            f"- Learned ridge coefficients: {coefficients}."
        )
        lines.append(
            f"- Learned readout parameter bytes: "
            f"{ranker['parameter_bytes_fp32']}."
        )
    hyperparameter_source = summary.get("hyperparameter_source", {})
    if hyperparameter_source.get("mode") == "fixed_before_evaluation":
        lines.extend(
            [
                "- Feature weights were loaded from a frozen file before the "
                "confirmatory evaluation.",
                f"- Frozen hyperparameter SHA256: "
                f"`{hyperparameter_source.get('sha256', 'unknown')}`.",
            ]
        )
    else:
        lines.append(
            "- All values above were selected on the disjoint calibration "
            "split and frozen before evaluation."
        )
    learned_ranker_source = summary.get("learned_ranker_source", {})
    if learned_ranker_source.get("mode") == "fixed_before_evaluation":
        lines.extend(
            [
                "- The learned readout was also frozen before the "
                "confirmatory evaluation.",
                f"- Frozen learned-ranker SHA256: "
                f"`{learned_ranker_source.get('sha256', 'unknown')}`.",
            ]
        )
    lines.extend(["", "## Decision", ""])
    if promotion["advance_to_llava_anchor"]:
        lines.append(
            "The frozen primary bounded question-only selector passes the "
            "CLIP proxy gate and should be checked with the paired LLaVA "
            "raw-frame anchor."
        )
    else:
        lines.append(
            "The frozen primary bounded question-only selector does not "
            "pass the CLIP proxy gate. Do not claim that bounded online "
            "query memory is solved; inspect task-specific failures before "
            "spending GPU time on a formal LLaVA anchor."
        )
    lines.extend(["", "## Figures", ""])
    figures = (
        ("Accuracy versus state", "accuracy_vs_persistent_state.png"),
        ("Task heatmap", "task_accuracy_heatmap.png"),
        ("Paired gains", "paired_gain_vs_exact_recent.png"),
        ("Primary calibration", "calibration_surface.png"),
        ("Feature calibration", "feature_calibration_surface.png"),
        (
            "Calibration transfer",
            "calibration_to_evaluation.png",
        ),
        (
            "Selection overlap",
            "selection_overlap_vs_reference.png",
        ),
        (
            "Temporal selections",
            "selected_frame_temporal_distribution.png",
        ),
    )
    for label, filename in figures:
        if (aggregate_dir / filename).exists():
            lines.extend([f"![{label}]({filename})", ""])
    lines.extend(
        [
            "## Claim Boundary",
            "",
            str(summary["promotion"]["claim_boundary"]),
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
