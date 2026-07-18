from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_mvbench_query_memory import CacheRecord, load_records
from query_memory import LearnedFeatureRanker, apply_learned_feature_policy
from streaming_baseline_proxies import (
    PROXY_METHODS,
    MemoryAccounting,
    ProxyResult,
    run_proxy,
)
from task_memory import softmax_pool_scores


OURS_METHOD = "ours_learned_recent_selector"
DEFAULT_METHODS = PROXY_METHODS + (OURS_METHOD,)
METHOD_LABELS = {
    "exact_recent": "Exact recent",
    "causalmem_feature_proxy": "CausalMem proxy",
    "streamingtom_feature_proxy": "StreamingTOM proxy",
    "stc_feature_proxy": "STC proxy",
    "selectstream_feature_proxy": "SelectStream proxy",
    "oasis_feature_proxy": "OASIS proxy",
    "statekv_feature_proxy": "StateKV proxy",
    OURS_METHOD: "Ours: learned selector (dev-fitted)",
}
METHOD_COLORS = {
    "exact_recent": "#6B7280",
    "causalmem_feature_proxy": "#D97706",
    "streamingtom_feature_proxy": "#0F766E",
    "stc_feature_proxy": "#2563EB",
    "selectstream_feature_proxy": "#9F1239",
    "oasis_feature_proxy": "#7C3AED",
    "statekv_feature_proxy": "#4D7C0F",
    OURS_METHOD: "#C2410C",
}
ANNOTATION_OFFSETS = {
    "exact_recent": (5, 5),
    "causalmem_feature_proxy": (5, -12),
    "streamingtom_feature_proxy": (5, 5),
    "stc_feature_proxy": (5, 5),
    "selectstream_feature_proxy": (-8, 7),
    "oasis_feature_proxy": (5, 5),
    "statekv_feature_proxy": (5, 5),
    OURS_METHOD: (5, 5),
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--learned-ranker", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--evidence-budget", type=int, default=8)
    parser.add_argument("--pool-capacity", type=int, default=16)
    parser.add_argument("--recent-anchors", type=int, default=3)
    parser.add_argument("--storage-bits", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def _run_ours(
    record: CacheRecord,
    ranker: LearnedFeatureRanker,
    *,
    evidence_budget: int,
    pool_capacity: int,
    recent_anchors: int,
    storage_bits: int,
) -> ProxyResult:
    start = perf_counter()
    result = apply_learned_feature_policy(
        record.image_vectors,
        record.question_vector,
        record.candidate_vectors,
        ranker,
        evidence_budget=evidence_budget,
        pool_capacity=pool_capacity,
        recent_anchors=recent_anchors,
        storage_bits=storage_bits,
    )
    elapsed = perf_counter() - start
    indices = list(result.selected_indices)
    accounting = MemoryAccounting(
        active_state_bytes=result.payload_bytes,
        archive_bytes=0,
        detailed_decode_bytes=0,
        metadata_bytes=result.metadata_bytes,
        shared_parameter_bytes=result.ranker_parameter_bytes,
        total_retained_bytes=(
            result.total_state_bytes + result.ranker_parameter_bytes
        ),
        active_state_bounded=True,
        total_state_bounded=True,
    )
    return ProxyResult(
        method=OURS_METHOD,
        reproduction_tier="project_native_selector_feature_proxy",
        evidence_vectors=record.image_vectors[indices],
        evidence_indices=tuple(indices),
        accounting=accounting,
        write_seconds=0.0,
        read_seconds=elapsed,
        estimated_write_flops=0,
        estimated_read_flops=result.estimated_retrieval_flops,
        query_conditioned=True,
        diagnostics={
            "ranker_parameter_bytes": result.ranker_parameter_bytes,
            "native_llava_selector_validated": True,
            "routed_spatial_codec_represented": False,
        },
    )


def evaluate_record(
    record: CacheRecord,
    method: str,
    ranker: LearnedFeatureRanker,
    args: argparse.Namespace,
) -> dict[str, object]:
    if method == OURS_METHOD:
        result = _run_ours(
            record,
            ranker,
            evidence_budget=args.evidence_budget,
            pool_capacity=args.pool_capacity,
            recent_anchors=args.recent_anchors,
            storage_bits=args.storage_bits,
        )
    else:
        result = run_proxy(
            method,
            record.image_vectors,
            record.question_vector,
            evidence_budget=args.evidence_budget,
            pool_capacity=args.pool_capacity,
            recent_anchors=args.recent_anchors,
            storage_bits=args.storage_bits,
        )
    scores = softmax_pool_scores(
        record.candidate_vectors,
        result.evidence_vectors,
        temperature=args.temperature,
    )
    prediction = int(np.argmax(scores))
    answer_index = int(record.metadata["answer_index"])
    accounting = result.accounting
    return {
        "sample_id": str(record.metadata["sample_id"]),
        "task": str(record.metadata["task"]),
        "method": result.method,
        "method_label": METHOD_LABELS[result.method],
        "reproduction_tier": result.reproduction_tier,
        "prediction": prediction,
        "answer_index": answer_index,
        "correct": int(prediction == answer_index),
        "evidence_count": len(result.evidence_vectors),
        "evidence_indices_json": json.dumps(list(result.evidence_indices)),
        "active_state_bytes": accounting.active_state_bytes,
        "archive_bytes": accounting.archive_bytes,
        "detailed_decode_bytes": accounting.detailed_decode_bytes,
        "metadata_bytes": accounting.metadata_bytes,
        "shared_parameter_bytes": accounting.shared_parameter_bytes,
        "total_retained_bytes": accounting.total_retained_bytes,
        "active_state_bounded": int(accounting.active_state_bounded),
        "total_state_bounded": int(accounting.total_state_bounded),
        "query_conditioned": int(result.query_conditioned),
        "write_seconds": result.write_seconds,
        "read_seconds": result.read_seconds,
        "estimated_write_flops": result.estimated_write_flops,
        "estimated_read_flops": result.estimated_read_flops,
        "diagnostics_json": json.dumps(
            result.diagnostics, sort_keys=True, separators=(",", ":")
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize_overall(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary: list[dict[str, object]] = []
    for method in DEFAULT_METHODS:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        tasks: dict[str, list[int]] = defaultdict(list)
        for row in method_rows:
            tasks[str(row["task"])].append(int(row["correct"]))
        write = [float(row["write_seconds"]) for row in method_rows]
        read = [float(row["read_seconds"]) for row in method_rows]
        summary.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "reproduction_tier": method_rows[0]["reproduction_tier"],
                "samples": len(method_rows),
                "micro_accuracy": float(
                    np.mean([int(row["correct"]) for row in method_rows])
                ),
                "macro_task_accuracy": float(
                    np.mean([np.mean(values) for values in tasks.values()])
                ),
                "mean_evidence_count": float(
                    np.mean([int(row["evidence_count"]) for row in method_rows])
                ),
                "mean_active_state_bytes": float(
                    np.mean(
                        [int(row["active_state_bytes"]) for row in method_rows]
                    )
                ),
                "mean_archive_bytes": float(
                    np.mean([int(row["archive_bytes"]) for row in method_rows])
                ),
                "mean_detailed_decode_bytes": float(
                    np.mean(
                        [int(row["detailed_decode_bytes"]) for row in method_rows]
                    )
                ),
                "mean_metadata_bytes": float(
                    np.mean([int(row["metadata_bytes"]) for row in method_rows])
                ),
                "mean_total_retained_bytes": float(
                    np.mean(
                        [int(row["total_retained_bytes"]) for row in method_rows]
                    )
                ),
                "active_state_bounded": int(
                    all(int(row["active_state_bounded"]) for row in method_rows)
                ),
                "total_state_bounded": int(
                    all(int(row["total_state_bounded"]) for row in method_rows)
                ),
                "query_conditioned": int(method_rows[0]["query_conditioned"]),
                "mean_write_ms": 1000.0 * float(np.mean(write)),
                "p50_write_ms": 1000.0 * _percentile(write, 0.50),
                "p95_write_ms": 1000.0 * _percentile(write, 0.95),
                "p99_write_ms": 1000.0 * _percentile(write, 0.99),
                "mean_read_ms": 1000.0 * float(np.mean(read)),
                "p50_read_ms": 1000.0 * _percentile(read, 0.50),
                "p95_read_ms": 1000.0 * _percentile(read, 0.95),
                "p99_read_ms": 1000.0 * _percentile(read, 0.99),
                "mean_estimated_write_flops": float(
                    np.mean(
                        [int(row["estimated_write_flops"]) for row in method_rows]
                    )
                ),
                "mean_estimated_read_flops": float(
                    np.mean(
                        [int(row["estimated_read_flops"]) for row in method_rows]
                    )
                ),
            }
        )
    return summary


def summarize_tasks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["method"]))].append(
            int(row["correct"])
        )
    return [
        {
            "task": task,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "samples": len(values),
            "accuracy": float(np.mean(values)),
        }
        for (task, method), values in sorted(grouped.items())
    ]


def task_deltas(
    rows: list[dict[str, object]], *, reference: str
) -> list[dict[str, object]]:
    lookup = {
        (str(row["task"]), str(row["method"])): float(row["accuracy"])
        for row in rows
    }
    tasks = sorted({str(row["task"]) for row in rows})
    methods = [
        method
        for method in DEFAULT_METHODS
        if method != reference
        and any((task, method) in lookup for task in tasks)
    ]
    output: list[dict[str, object]] = []
    for task in tasks:
        reference_accuracy = lookup[(task, reference)]
        for method in methods:
            if (task, method) not in lookup:
                continue
            accuracy = lookup[(task, method)]
            output.append(
                {
                    "task": task,
                    "reference": reference,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "accuracy": accuracy,
                    "reference_accuracy": reference_accuracy,
                    "gain": accuracy - reference_accuracy,
                }
            )
    return output


def _mcnemar_exact(better: int, worse: int) -> float:
    discordant = better + worse
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(better, worse) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_comparisons(
    rows: list[dict[str, object]],
    *,
    reference: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    by_method: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        by_method[str(row["method"])][str(row["sample_id"])] = int(
            row["correct"]
        )
    reference_rows = by_method[reference]
    output: list[dict[str, object]] = []
    for method in DEFAULT_METHODS:
        if method == reference or method not in by_method:
            continue
        sample_ids = sorted(set(reference_rows) & set(by_method[method]))
        deltas = np.asarray(
            [
                by_method[method][sample_id] - reference_rows[sample_id]
                for sample_id in sample_ids
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed + sum(map(ord, method)))
        if len(deltas) and bootstrap_samples:
            bootstrap = np.mean(
                deltas[
                    rng.integers(
                        0,
                        len(deltas),
                        size=(bootstrap_samples, len(deltas)),
                    )
                ],
                axis=1,
            )
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        else:
            lower = upper = float("nan")
        better = int(np.sum(deltas > 0))
        worse = int(np.sum(deltas < 0))
        output.append(
            {
                "reference": reference,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "samples": len(deltas),
                "gain": float(np.mean(deltas)) if len(deltas) else float("nan"),
                "bootstrap_95_low": float(lower),
                "bootstrap_95_high": float(upper),
                "better": better,
                "worse": worse,
                "tied": int(np.sum(deltas == 0)),
                "mcnemar_exact_p": _mcnemar_exact(better, worse),
            }
        )
    return output


def _plot_quality_state(
    summary: list[dict[str, object]], out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for row in summary:
        method = str(row["method"])
        marker = "o" if int(row["total_state_bounded"]) else "X"
        ax.scatter(
            float(row["mean_total_retained_bytes"]) / 1024.0,
            100.0 * float(row["micro_accuracy"]),
            s=95,
            marker=marker,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset = ANNOTATION_OFFSETS[method]
        ax.annotate(
            METHOD_LABELS[method],
            (
                float(row["mean_total_retained_bytes"]) / 1024.0,
                100.0 * float(row["micro_accuracy"]),
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="right" if offset[0] < 0 else "left",
            va="top" if offset[1] < 0 else "bottom",
        )
    ax.set_xlabel("Mean retained state (KiB, all counted components)")
    ax.set_ylabel("CLIP-proxy multiple-choice accuracy (%)")
    ax.set_title("Mechanism proxy quality vs retained state")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_vs_state.png", dpi=220)
    fig.savefig(out_dir / "quality_vs_state.pdf")
    plt.close(fig)


def _plot_task_heatmap(
    rows: list[dict[str, object]],
    methods: list[str],
    out_dir: Path,
) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    lookup = {
        (str(row["task"]), str(row["method"])): float(row["accuracy"])
        for row in rows
    }
    matrix = np.asarray(
        [[lookup.get((task, method), np.nan) for method in methods] for task in tasks]
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    image = ax.imshow(matrix, vmin=0.25, vmax=0.75, cmap="YlGnBu", aspect="auto")
    for row_index in range(len(tasks)):
        for column_index in range(len(methods)):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{100 * value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.58 else "#172554",
                )
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=32, ha="right")
    ax.set_yticks(range(len(tasks)), [task.replace("_", " ") for task in tasks])
    ax.set_title("Task accuracy by mechanism proxy")
    fig.colorbar(image, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "task_accuracy_heatmap.png", dpi=220)
    fig.savefig(out_dir / "task_accuracy_heatmap.pdf")
    plt.close(fig)


def _plot_task_delta_heatmap(
    rows: list[dict[str, object]],
    methods: list[str],
    out_dir: Path,
) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    lookup = {
        (str(row["task"]), str(row["method"])): float(row["gain"])
        for row in rows
    }
    methods = [method for method in methods if method != "exact_recent"]
    matrix = np.asarray(
        [
            [lookup.get((task, method), np.nan) for method in methods]
            for task in tasks
        ]
    )
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(0.05, float(np.max(finite)) if finite.size else 0.05)
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    image = ax.imshow(
        matrix,
        vmin=-limit,
        vmax=limit,
        cmap="RdBu",
        aspect="auto",
    )
    for row_index in range(len(tasks)):
        for column_index in range(len(methods)):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{100 * value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.55 * limit else "#172554",
                )
    ax.set_xticks(
        range(len(methods)),
        [METHOD_LABELS[method] for method in methods],
        rotation=32,
        ha="right",
    )
    ax.set_yticks(
        range(len(tasks)), [task.replace("_", " ") for task in tasks]
    )
    ax.set_title("Task accuracy gain relative to exact recent (percentage points)")
    fig.colorbar(image, ax=ax, label="Accuracy gain")
    fig.tight_layout()
    fig.savefig(out_dir / "task_gain_vs_exact_recent.png", dpi=220)
    fig.savefig(out_dir / "task_gain_vs_exact_recent.pdf")
    plt.close(fig)


def _plot_memory_breakdown(
    summary: list[dict[str, object]], out_dir: Path
) -> None:
    methods = [str(row["method"]) for row in summary]
    components = (
        ("mean_active_state_bytes", "Active state", "#0F766E"),
        ("mean_archive_bytes", "Archive", "#D97706"),
        ("mean_detailed_decode_bytes", "Detailed decode", "#9F1239"),
        ("mean_metadata_bytes", "Metadata", "#64748B"),
    )
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    bottom = np.zeros(len(summary), dtype=np.float64)
    for key, label, color in components:
        values = np.asarray([float(row[key]) / 1024.0 for row in summary])
        ax.bar(range(len(summary)), values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=32, ha="right")
    ax.set_ylabel("Mean retained state (KiB)")
    ax.set_title("Retained-state accounting by component")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_breakdown.png", dpi=220)
    fig.savefig(out_dir / "memory_breakdown.pdf")
    plt.close(fig)


def _plot_cpu_latency(
    summary: list[dict[str, object]], out_dir: Path
) -> None:
    methods = [str(row["method"]) for row in summary]
    write = np.asarray([float(row["p95_write_ms"]) for row in summary])
    read = np.asarray([float(row["p95_read_ms"]) for row in summary])
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.bar(range(len(summary)), write, color="#0F766E", label="P95 write")
    ax.bar(range(len(summary)), read, bottom=write, color="#D97706", label="P95 read")
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=32, ha="right")
    ax.set_ylabel("CPU feature-replay latency (ms)")
    ax.set_title("Proxy overhead only: not end-to-end GPU latency")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "cpu_proxy_latency.png", dpi=220)
    fig.savefig(out_dir / "cpu_proxy_latency.pdf")
    plt.close(fig)


def write_report(
    out_dir: Path,
    summary: list[dict[str, object]],
    paired: list[dict[str, object]],
    *,
    cache_records: int,
) -> None:
    best_bounded = max(
        (row for row in summary if int(row["total_state_bounded"])),
        key=lambda row: float(row["micro_accuracy"]),
    )
    lines = [
        "# Streaming Baseline Mechanism-Proxy Comparison",
        "",
        f"This run replays {cache_records} frozen MVBench CLIP caches from a "
        "reused development set. It is a "
        "mechanism and accounting comparison, not an official end-to-end "
        "reproduction of any external method.",
        "",
        "## Overall",
        "",
        "| Method | Tier | Accuracy | Evidence | Active KiB | Archive KiB | "
        "Detailed KiB | Total KiB | Total bounded |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary:
        lines.append(
            "| {label} | `{tier}` | {accuracy:.1f}% | {evidence:.1f} | "
            "{active:.2f} | {archive:.2f} | {detail:.2f} | {total:.2f} | "
            "{bounded} |".format(
                label=row["method_label"],
                tier=row["reproduction_tier"],
                accuracy=100.0 * float(row["micro_accuracy"]),
                evidence=float(row["mean_evidence_count"]),
                active=float(row["mean_active_state_bytes"]) / 1024.0,
                archive=float(row["mean_archive_bytes"]) / 1024.0,
                detail=float(row["mean_detailed_decode_bytes"]) / 1024.0,
                total=float(row["mean_total_retained_bytes"]) / 1024.0,
                bounded="yes" if int(row["total_state_bounded"]) else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Paired Against Exact Recent",
            "",
            "| Method | Gain | 95% bootstrap interval | Better / worse | "
            "McNemar p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            "| {label} | {gain:+.1f} pp | [{low:+.1f}, {high:+.1f}] | "
            "{better} / {worse} | {p:.4f} |".format(
                label=row["method_label"],
                gain=100.0 * float(row["gain"]),
                low=100.0 * float(row["bootstrap_95_low"]),
                high=100.0 * float(row["bootstrap_95_high"]),
                better=row["better"],
                worse=row["worse"],
                p=float(row["mcnemar_exact_p"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            f"The strongest total-bounded proxy is **{best_bounded['method_label']}** "
            f"at {100 * float(best_bounded['micro_accuracy']):.1f}% on this "
            "frozen CLIP proxy. This does not transfer paper-reported quality "
            "between different backbones or benchmarks.",
            "",
            "- The learned selector was fitted on development evidence from this "
            "reused sample pool. Its result is post-hoc and is not an independent "
            "generalization estimate.",
            "- CausalMem is reduced from projected token memory to one frame vector "
            "per observation; its background-token merge is intentionally omitted "
            "because it degenerates at that resolution.",
            "- StreamingTOM includes a 4-bit feature-group archive and bounded active "
            "read, but not its real KV kernels or end-to-end TTFT.",
            "- STC uses frame-group reuse and dynamic/uniform pruning proxies; real "
            "ViT token reuse and visual-token pruning require the official stack.",
            "- SelectStream lacks public code in this audit and its learned segment "
            "encoder, graph attention, calibration, and training are not reproduced.",
            "- OASIS uses vector event centroids instead of MLLM event summaries and "
            "intent-driven tool calls.",
            "- StateKV correctly counts its fixed cstate separately from the growing "
            "detailed decode cache; its apparent quality is not a matched fixed-total-"
            "memory comparison.",
            "- The row for our method evaluates the frozen learned selector only. "
            "The routed low-rank/spatial residual codec is analyzed in the separate "
            "native LLaVA feature-memory confirmation.",
            "",
            "CPU latency plots measure NumPy feature replay only. Official P50/P95/"
            "P99 GPU latency remains a separate reproduction gate.",
        ]
    )
    (out_dir / "RESULTS_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    methods = parse_csv(args.methods)
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.cache_dir)
    ranker_payload = json.loads(args.learned_ranker.read_text(encoding="utf-8"))
    ranker = LearnedFeatureRanker.from_dict(ranker_payload)

    rows: list[dict[str, object]] = []
    for record in records:
        for method in methods:
            rows.append(evaluate_record(record, method, ranker, args))
    overall = summarize_overall(rows)
    tasks = summarize_tasks(rows)
    deltas = task_deltas(tasks, reference="exact_recent")
    paired = paired_comparisons(
        rows,
        reference="exact_recent",
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_csv(args.out_dir / "predictions.csv", rows)
    _write_csv(args.out_dir / "overall_summary.csv", overall)
    _write_csv(args.out_dir / "task_accuracy.csv", tasks)
    _write_csv(args.out_dir / "task_gain_vs_exact_recent.csv", deltas)
    _write_csv(args.out_dir / "paired_vs_exact_recent.csv", paired)
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "evaluation_scope": "frozen_clip_feature_mechanism_proxy",
                "records": len(records),
                "prediction_rows": len(rows),
                "methods": methods,
                "evidence_budget": args.evidence_budget,
                "pool_capacity": args.pool_capacity,
                "recent_anchors": args.recent_anchors,
                "storage_bits": args.storage_bits,
                "overall": overall,
                "paired_vs_exact_recent": paired,
                "task_gain_vs_exact_recent": deltas,
                "ranker": ranker.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    method_order = [method for method in methods if any(row["method"] == method for row in overall)]
    _plot_quality_state(overall, args.out_dir)
    _plot_task_heatmap(tasks, method_order, args.out_dir)
    _plot_task_delta_heatmap(deltas, method_order, args.out_dir)
    _plot_memory_breakdown(overall, args.out_dir)
    _plot_cpu_latency(overall, args.out_dir)
    write_report(
        args.out_dir,
        overall,
        paired,
        cache_records=len(records),
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "methods": methods,
                "out_dir": str(args.out_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
