from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from aggregate_mvbench_llava import (
    load_rows,
    paired_policy_comparisons,
    wilson_interval,
    write_csv,
)


SELECTION_LABELS = {
    "exact_recent": "Exact recent",
    "recent_pool_query_topk": "Recent pool + top-k",
    "recent_pool_query_mmr": "Recent pool + MMR",
    "learned_recent_query_topk": "Learned recent-pool top-k",
}
SELECTION_COLORS = {
    "exact_recent": "#264653",
    "recent_pool_query_topk": "#2A9D8F",
    "recent_pool_query_mmr": "#457B9D",
    "learned_recent_query_topk": "#B07AA1",
}
TASK_ORDER = [
    "object_existence",
    "state_change",
    "scene_transition",
    "action_sequence",
    "moving_direction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--noninferiority-margin",
        type=float,
        default=0.02,
        help="Maximum allowed paired accuracy loss as a fraction.",
    )
    return parser.parse_args()


def summarize_variants(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, object]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["selection_policy"]),
                str(row["memory_variant"]),
            )
        ].append(row)
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        sparse_rates = {
            int(row.get("sparse_residual_tokens_per_frame", 0))
            for row in values
        }
        correct = sum(int(row["correct"]) for row in values)
        parsed = sum(int(row["parsed"]) for row in values)
        samples = len(values)
        low, high = wilson_interval(correct, samples)
        mean_total_state_bytes = float(
            np.mean(
                [
                    int(row["selection_state_proxy_bytes"])
                    for row in values
                ]
            )
        )
        codec_parameter_bytes = int(
            values[0]["codec_parameter_bytes"]
        )
        cold_start_total_state_bytes = mean_total_state_bytes
        if key[1] != "full":
            cold_start_total_state_bytes += codec_parameter_bytes
        output.append(
            {
                "selection_policy": key[0],
                "memory_variant": key[1],
                "samples": samples,
                "parsed_rate": parsed / samples,
                "correct": correct,
                "accuracy": correct / samples,
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "mean_total_state_bytes": mean_total_state_bytes,
                "cold_start_total_state_bytes": (
                    cold_start_total_state_bytes
                ),
                "mean_feature_state_bytes": float(
                    np.mean(
                        [
                            int(row["native_feature_state_bytes"])
                            for row in values
                        ]
                    )
                ),
                "codec_parameter_bytes": codec_parameter_bytes,
                "codec_rank": int(values[0]["codec_rank"]),
                "residual_tokens_per_frame": int(
                    values[0]["residual_tokens_per_frame"]
                ),
                "residual_value_vectors_per_frame": int(
                    values[0].get("residual_value_vectors_per_frame", 0)
                ),
                "residual_value_vector_budget": int(
                    values[0].get("residual_value_vector_budget", 0)
                ),
                "sparse_residual_tokens_per_frame": int(
                    next(iter(sparse_rates)) if len(sparse_rates) == 1 else -1
                ),
                "sparse_residual_token_capacity": int(
                    max(
                        int(row.get("sparse_residual_token_capacity", 0))
                        for row in values
                    )
                ),
                "mean_realized_sparse_residual_tokens": float(
                    np.mean(
                        [
                            int(row.get("realized_sparse_residual_tokens", 0))
                            for row in values
                        ]
                    )
                ),
                "mean_realized_sparse_tokens_per_frame": float(
                    np.mean(
                        [
                            int(row.get("realized_sparse_residual_tokens", 0))
                            / max(1, len(row.get("pool_frame_indices", [])))
                            for row in values
                        ]
                    )
                ),
                "mean_grid_mode_frames": float(
                    np.mean(
                        [int(row.get("grid_mode_frames", 0)) for row in values]
                    )
                ),
                "mean_sparse_mode_frames": float(
                    np.mean(
                        [int(row.get("sparse_mode_frames", 0)) for row in values]
                    )
                ),
                "mean_residual_value_bytes": float(
                    np.mean(
                        [int(row.get("residual_value_bytes", 0)) for row in values]
                    )
                ),
                "mean_residual_index_bytes": float(
                    np.mean(
                        [int(row.get("residual_index_bytes", 0)) for row in values]
                    )
                ),
                "mean_residual_index_slot_bytes": float(
                    np.mean(
                        [
                            int(row.get("residual_index_slot_bytes", 0))
                            for row in values
                        ]
                    )
                ),
                "mean_route_mask_bytes": float(
                    np.mean(
                        [int(row.get("route_mask_bytes", 0)) for row in values]
                    )
                ),
                "mean_feature_state_compression_ratio": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "feature_state_compression_ratio"
                                ]
                            )
                            for row in values
                        ]
                    )
                ),
                "mean_total_state_compression_ratio": float(
                    np.mean(
                        [
                            float(
                                row["total_state_compression_ratio"]
                            )
                            for row in values
                        ]
                    )
                ),
                "mean_pool_reconstruction_error": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "pool_reconstruction_relative_error"
                                ]
                            )
                            for row in values
                        ]
                    )
                ),
                "mean_selected_reconstruction_error": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "selected_reconstruction_relative_error"
                                ]
                            )
                            for row in values
                        ]
                    )
                ),
                "mean_compression_seconds": float(
                    np.mean(
                        [
                            float(row["compression_seconds"])
                            for row in values
                        ]
                    )
                ),
                "mean_reconstruction_seconds": float(
                    np.mean(
                        [
                            float(row["reconstruction_seconds"])
                            for row in values
                        ]
                    )
                ),
                "mean_inference_seconds": float(
                    np.mean(
                        [
                            float(row["inference_seconds"])
                            for row in values
                        ]
                    )
                ),
                "mean_policy_seconds": float(
                    np.mean(
                        [
                            float(row["policy_seconds"])
                            for row in values
                        ]
                    )
                ),
            }
        )
    return output


def summarize_tasks(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, object]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["task"]),
                str(row["selection_policy"]),
                str(row["memory_variant"]),
            )
        ].append(row)
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        correct = sum(int(row["correct"]) for row in values)
        output.append(
            {
                "task": key[0],
                "selection_policy": key[1],
                "memory_variant": key[2],
                "samples": len(values),
                "correct": correct,
                "accuracy": correct / len(values),
            }
        )
    return output


def task_deltas_vs_full(
    task_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    full_accuracy = {
        (str(row["task"]), str(row["selection_policy"])): float(
            row["accuracy"]
        )
        for row in task_rows
        if row["memory_variant"] == "full"
    }
    output = []
    for row in task_rows:
        if row["memory_variant"] == "full":
            continue
        key = (str(row["task"]), str(row["selection_policy"]))
        if key not in full_accuracy:
            continue
        reference = full_accuracy[key]
        output.append(
            {
                "task": row["task"],
                "selection_policy": row["selection_policy"],
                "memory_variant": row["memory_variant"],
                "samples": row["samples"],
                "accuracy": row["accuracy"],
                "full_accuracy": reference,
                "accuracy_delta_vs_full": (
                    float(row["accuracy"]) - reference
                ),
            }
        )
    return output


def paired_vs_full(
    rows: list[dict[str, object]],
    *,
    seed: int,
    noninferiority_margin: float = 0.02,
) -> list[dict[str, object]]:
    if noninferiority_margin < 0.0:
        raise ValueError("noninferiority margin must be non-negative")
    output = []
    selection_policies = sorted(
        {str(row["selection_policy"]) for row in rows}
    )
    for selection_policy in selection_policies:
        subset = [
            row
            for row in rows
            if row["selection_policy"] == selection_policy
        ]
        reference = f"{selection_policy}__full"
        for comparison in paired_policy_comparisons(
            subset,
            reference=reference,
            seed=seed,
        ):
            variant = str(comparison["policy"]).split("__", 1)[1]
            reference_by_sample = {
                str(row["sample_id"]): row
                for row in subset
                if row["policy"] == reference
            }
            variant_policy = f"{selection_policy}__{variant}"
            variant_by_sample = {
                str(row["sample_id"]): row
                for row in subset
                if row["policy"] == variant_policy
            }
            paired_ids = sorted(
                set(reference_by_sample) & set(variant_by_sample)
            )
            prediction_matches = sum(
                str(reference_by_sample[sample_id]["predicted_index"])
                == str(variant_by_sample[sample_id]["predicted_index"])
                for sample_id in paired_ids
            )
            paired_samples = int(comparison["paired_samples"])
            worse_samples = int(comparison["worse_samples"])
            worse_rate_upper = clopper_pearson_upper(
                worse_samples,
                paired_samples,
                alpha=0.05,
            )
            output.append(
                {
                    "selection_policy": selection_policy,
                    "memory_variant": variant,
                    "reference_variant": "full",
                    "noninferiority_margin": noninferiority_margin,
                    "paired_prediction_matches": prediction_matches,
                    "paired_prediction_disagreements": (
                        len(paired_ids) - prediction_matches
                    ),
                    "prediction_agreement_rate": (
                        prediction_matches / len(paired_ids)
                    ),
                    "worse_rate": worse_samples / paired_samples,
                    "worse_rate_upper_95": worse_rate_upper,
                    "noninferior_at_margin": int(
                        worse_rate_upper <= noninferiority_margin
                    ),
                    **{
                        key: value
                        for key, value in comparison.items()
                        if key not in {"policy", "reference"}
                    },
                }
            )
    return output


def paired_selectors_by_variant(
    rows: list[dict[str, object]],
    *,
    seed: int,
    reference_selector: str = "exact_recent",
) -> list[dict[str, object]]:
    output = []
    for variant in sorted(
        {str(row["memory_variant"]) for row in rows}
    ):
        subset = [
            row for row in rows if row["memory_variant"] == variant
        ]
        reference = f"{reference_selector}__{variant}"
        if not any(row["policy"] == reference for row in subset):
            continue
        for comparison in paired_policy_comparisons(
            subset,
            reference=reference,
            seed=seed,
        ):
            candidate_policy = str(comparison["policy"])
            candidate_selector = candidate_policy.rsplit("__", 1)[0]
            reference_by_sample = {
                str(row["sample_id"]): row
                for row in subset
                if row["policy"] == reference
            }
            candidate_by_sample = {
                str(row["sample_id"]): row
                for row in subset
                if row["policy"] == candidate_policy
            }
            paired_ids = sorted(
                set(reference_by_sample) & set(candidate_by_sample)
            )
            prediction_matches = sum(
                str(reference_by_sample[sample_id]["predicted_index"])
                == str(candidate_by_sample[sample_id]["predicted_index"])
                for sample_id in paired_ids
            )
            output.append(
                {
                    "memory_variant": variant,
                    "selection_policy": candidate_selector,
                    "reference_selection_policy": reference_selector,
                    "paired_prediction_matches": prediction_matches,
                    "prediction_agreement_rate": (
                        prediction_matches / len(paired_ids)
                    ),
                    **{
                        key: value
                        for key, value in comparison.items()
                        if key not in {"policy", "reference"}
                    },
                }
            )
    return output


def binomial_cdf(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if probability == 0.0:
        return 1.0
    if probability == 1.0:
        return float(successes == trials)
    return sum(
        math.comb(trials, value)
        * probability**value
        * (1.0 - probability) ** (trials - value)
        for value in range(successes + 1)
    )


def clopper_pearson_upper(
    successes: int,
    trials: int,
    *,
    alpha: float,
) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if successes == trials:
        return 1.0
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)
    lower = successes / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if binomial_cdf(successes, trials, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def plot_accuracy_vs_state(
    rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    for selection_policy in sorted(
        {str(row["selection_policy"]) for row in rows}
    ):
        values = sorted(
            (
                row
                for row in rows
                if row["selection_policy"] == selection_policy
            ),
            key=lambda row: float(row["mean_total_state_bytes"]),
        )
        x = [
            float(row["mean_total_state_bytes"]) / (1024**2)
            for row in values
        ]
        y = [float(row["accuracy"]) for row in values]
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            label=SELECTION_LABELS.get(
                selection_policy,
                selection_policy,
            ),
            color=SELECTION_COLORS.get(selection_policy),
        )
        for x_value, y_value, row in zip(x, y, values):
            axis.annotate(
                str(row["memory_variant"]).replace("pca_r64_", ""),
                (x_value, y_value),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xscale("log")
    axis.set_xlabel("Per-stream persistent state (MiB, log scale)")
    axis.set_ylabel("MVBench accuracy")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"accuracy_vs_state.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_reconstruction_vs_state(
    rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    compressed = [
        row for row in rows if row["memory_variant"] != "full"
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for selection_policy in sorted(
        {str(row["selection_policy"]) for row in compressed}
    ):
        values = sorted(
            (
                row
                for row in compressed
                if row["selection_policy"] == selection_policy
            ),
            key=lambda row: float(row["mean_total_state_bytes"]),
        )
        axis.plot(
            [
                float(row["mean_total_state_bytes"]) / (1024**2)
                for row in values
            ],
            [
                float(row["mean_selected_reconstruction_error"])
                for row in values
            ],
            marker="o",
            linewidth=1.8,
            label=SELECTION_LABELS.get(
                selection_policy,
                selection_policy,
            ),
            color=SELECTION_COLORS.get(selection_policy),
        )
    axis.set_xlabel("Per-stream persistent state (MiB)")
    axis.set_ylabel("Selected-feature relative reconstruction error")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"reconstruction_vs_state.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_task_delta_heatmap(
    rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    tasks = [
        task
        for task in TASK_ORDER
        if any(row["task"] == task for row in rows)
    ]
    combinations = sorted(
        {
            (
                str(row["selection_policy"]),
                str(row["memory_variant"]),
            )
            for row in rows
        }
    )
    lookup = {
        (
            str(row["selection_policy"]),
            str(row["memory_variant"]),
            str(row["task"]),
        ): float(row["accuracy_delta_vs_full"])
        for row in rows
    }
    matrix = np.asarray(
        [
            [
                lookup[(policy, variant, task)] * 100
                for task in tasks
            ]
            for policy, variant in combinations
        ],
        dtype=np.float64,
    )
    limit = max(float(np.max(np.abs(matrix))), 5.0)
    figure, axis = plt.subplots(
        figsize=(8.4, 1.1 + 0.55 * len(combinations))
    )
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_xticks(
        range(len(tasks)),
        [task.replace("_", " ") for task in tasks],
        rotation=20,
        ha="right",
    )
    axis.set_yticks(
        range(len(combinations)),
        [
            f"{SELECTION_LABELS.get(policy, policy)} | {variant}"
            for policy, variant in combinations
        ],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if abs(value) > limit * 0.55
                    else "#222222"
                ),
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Accuracy delta vs full cache (points)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"task_delta_vs_full.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_preservation_gate(
    rows: list[dict[str, object]],
    out_dir: Path,
    *,
    noninferiority_margin: float,
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    labels = [
        f"{'Exact' if row['selection_policy'] == 'exact_recent' else 'Learned'}\n"
        f"{str(row['memory_variant']).rsplit('_', 1)[-1]}"
        for row in rows
    ]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(
        positions,
        [float(row["worse_rate_upper_95"]) * 100 for row in rows],
        color=[
            SELECTION_COLORS.get(str(row["selection_policy"]))
            for row in rows
        ],
        alpha=0.88,
    )
    axes[0].axhline(
        noninferiority_margin * 100,
        color="#B23A48",
        linestyle="--",
        linewidth=1.6,
        label=f"{noninferiority_margin:.0%} NI margin",
    )
    axes[0].set_ylabel("One-sided 95% upper degradation rate (%)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].bar(
        positions,
        [float(row["prediction_agreement_rate"]) * 100 for row in rows],
        color=[
            SELECTION_COLORS.get(str(row["selection_policy"]))
            for row in rows
        ],
        alpha=0.88,
    )
    axes[1].set_ylabel("Exact prediction agreement (%)")
    axes[1].set_ylim(0, 101)
    axes[1].grid(axis="y", alpha=0.22)
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            out_dir / f"preservation_gate.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def write_report(
    out_dir: Path,
    *,
    summary_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    selector_rows: list[dict[str, object]],
    checkpoint_count: int,
    fingerprints: list[str],
    noninferiority_margin: float,
) -> None:
    lines = [
        "# Compressed Native Feature-Memory Analysis",
        "",
        f"- Completed checkpoints: {checkpoint_count}.",
        f"- Configuration fingerprints: {len(fingerprints)}.",
        "",
        "## Variant Summary",
        "",
        "| Selector | Memory | Accuracy | Steady-state MiB | "
        "Cold-start MiB | Compression | Selected error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {SELECTION_LABELS.get(str(row['selection_policy']), row['selection_policy'])} "
            f"| {row['memory_variant']} "
            f"| {float(row['accuracy']):.2%} "
            f"| {float(row['mean_total_state_bytes']) / (1024**2):.3f} "
            f"| {float(row['cold_start_total_state_bytes']) / (1024**2):.3f} "
            f"| {float(row['mean_total_state_compression_ratio']):.2f}x "
            f"| {float(row['mean_selected_reconstruction_error']):.4f} |"
        )
    lines.extend(
        [
            "",
        "## Paired Accuracy Versus Full Cache",
        "",
        f"Non-inferiority margin: {noninferiority_margin:.1%}. The "
        "decision uses the one-sided 95% Clopper-Pearson upper bound "
        "on full-correct/compressed-wrong outcomes.",
        "",
        "| Selector | Memory | Gain | 95% CI | Prediction agreement | "
        "Better / worse | Worse upper 95% | Non-inferior |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {SELECTION_LABELS.get(str(row['selection_policy']), row['selection_policy'])} "
            f"| {row['memory_variant']} "
            f"| {float(row['accuracy_gain']):+.2%} "
            f"| [{float(row['bootstrap_ci95_low']):+.2%}, "
            f"{float(row['bootstrap_ci95_high']):+.2%}] "
            f"| {float(row['prediction_agreement_rate']):.2%} "
            f"| {row['better_samples']} / {row['worse_samples']} "
            f"| {float(row['worse_rate_upper_95']):.2%} "
            f"| {'yes' if int(row['noninferior_at_margin']) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Query-Conditioned Selector Gain at Matched State",
            "",
            "| Memory | Candidate versus exact recent | Gain | 95% CI | "
            "Better / worse | McNemar p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in selector_rows:
        lines.append(
            f"| {row['memory_variant']} "
            f"| {SELECTION_LABELS.get(str(row['selection_policy']), row['selection_policy'])} "
            f"| {float(row['accuracy_gain']):+.2%} "
            f"| [{float(row['bootstrap_ci95_low']):+.2%}, "
            f"{float(row['bootstrap_ci95_high']):+.2%}] "
            f"| {row['better_samples']} / {row['worse_samples']} "
            f"| {float(row['mcnemar_exact_p']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- PCA and sparse residual coding are established compression "
            "tools. This experiment tests task preservation and systems "
            "trade-offs, not mathematical novelty.",
            "- Shared codec parameters and per-stream state are reported "
            "separately. Cold-start state includes the shared codec for "
            "compressed variants; steady-state state does not amortize it "
            "into every stream.",
            "- A lower reconstruction error is not sufficient; promotion "
            "requires preserving full-cache LLaVA accuracy.",
            "- The non-inferiority gate is conservative: compressed "
            "improvements do not offset full-correct/compressed-wrong "
            "events.",
            "",
            "## Figures",
            "",
            "![Accuracy versus state](accuracy_vs_state.png)",
            "",
            "![Reconstruction versus state]"
            "(reconstruction_vs_state.png)",
            "",
            "![Task deltas versus full cache]"
            "(task_delta_vs_full.png)",
            "",
            "![Finite-sample preservation gate]"
            "(preservation_gate.png)",
            "",
        ]
    )
    (out_dir / "RESULTS_ANALYSIS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.noninferiority_margin < 0.0:
        raise ValueError("noninferiority margin must be non-negative")
    out_dir = args.out_dir or args.run_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, fingerprints = load_rows(args.run_dir)
    summary_rows = summarize_variants(rows)
    task_rows = summarize_tasks(rows)
    task_delta_rows = task_deltas_vs_full(task_rows)
    paired_rows = paired_vs_full(
        rows,
        seed=args.seed,
        noninferiority_margin=args.noninferiority_margin,
    )
    selector_rows = paired_selectors_by_variant(
        rows,
        seed=args.seed,
    )
    write_csv(out_dir / "predictions.csv", rows)
    write_csv(out_dir / "variant_summary.csv", summary_rows)
    write_csv(out_dir / "task_accuracy.csv", task_rows)
    write_csv(out_dir / "task_delta_vs_full.csv", task_delta_rows)
    write_csv(out_dir / "paired_vs_full.csv", paired_rows)
    write_csv(
        out_dir / "selector_gain_by_variant.csv",
        selector_rows,
    )
    checkpoint_count = len(
        list((args.run_dir / "checkpoints").glob("*.json"))
    )
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps(
            {
                "checkpoint_count": checkpoint_count,
                "prediction_rows": len(rows),
                "fingerprints": fingerprints,
                "selection_policies": sorted(
                    {str(row["selection_policy"]) for row in rows}
                ),
                "memory_variants": sorted(
                    {str(row["memory_variant"]) for row in rows}
                ),
                "noninferiority_margin": args.noninferiority_margin,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_accuracy_vs_state(summary_rows, out_dir)
    plot_reconstruction_vs_state(summary_rows, out_dir)
    plot_task_delta_heatmap(task_delta_rows, out_dir)
    plot_preservation_gate(
        paired_rows,
        out_dir,
        noninferiority_margin=args.noninferiority_margin,
    )
    write_report(
        out_dir,
        summary_rows=summary_rows,
        paired_rows=paired_rows,
        selector_rows=selector_rows,
        checkpoint_count=checkpoint_count,
        fingerprints=fingerprints,
        noninferiority_margin=args.noninferiority_margin,
    )
    print(out_dir / "RESULTS_ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
