#!/usr/bin/env python3
"""Aggregate TileLogic-RVQ feature, quality, routing, rate, and latency results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from tilespec_ex.metrics import spearman_correlation
from tilespec_ex.tilelogic_analysis import (
    DATASETS,
    RATES,
    quality_guardrail,
    relative_increase,
    status_from_evidence,
    strict_frontier_extension,
    topk_recall,
    weighted_mean,
)
from tilespec_ex.rate import RATE_STORAGE_POLICY
from tilespec_ex.tilelogic_methods import ABLATION_METHODS, MAIN_TILELOGIC_METHODS


ANALYSIS_FORMAT = "tilelogic_rvq_analysis_v1"
EXPECTED_METHOD_RATES = {
    (method, rate)
    for method in (*MAIN_TILELOGIC_METHODS, *ABLATION_METHODS)
    for rate in RATES
}
EXPECTED_VARIANTS = 1 + len(EXPECTED_METHOD_RATES)
ROUTER_METHOD_KEYS = {
    "mlp": "base_vq_mlp_router",
    "logic": "base_vq_logic_router",
}
DECISION_QUESTIONS = {
    1: "Base VQ extends the full-overhead frontier beyond INT4",
    2: "Fisher RVQ improves on base-only and unweighted RVQ",
    3: "MLP routing beats energy and cosine-risk heuristics",
    4: "Discrete logic retains the MLP routing benefit",
    5: "Fixed slots reduce layout, decoder, and TTFT cost",
    6: "Fully charged exact fallback improves the frontier",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value for {name}: {value}")
    return result


def _rate(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _group_with_all(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
        if "dataset" in keys:
            aggregate = tuple("all" if key == "dataset" else row[key] for key in keys)
            grouped[aggregate].append(row)
    return grouped


def _flatten_feature(samples: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for sample in samples:
        for variant in sample["variants"]:
            rate = _rate(variant["retention_rate"])
            metrics = variant["rate"]
            modes = variant.get("mode_counts") or {}
            row = {
                "dataset": str(sample["dataset"]),
                "dataset_index": int(sample["dataset_index"]),
                "sample_id": str(sample["sample_id"]),
                "oracle": bool(sample["oracle"]),
                "method": str(variant["method"]),
                "scope": str(variant["scope"]),
                "retention_rate": rate,
                "feature_nmse": _finite(variant["feature_nmse"], "feature_nmse"),
                "feature_cosine": _finite(variant["feature_cosine"], "feature_cosine"),
                "boundary_mse": _finite(variant["boundary_mse"], "boundary_mse"),
                "fisher_weighted_nmse": _finite(
                    variant["fisher_weighted_nmse"], "fisher_weighted_nmse"
                ),
                "oracle_first_order_abs": (
                    None
                    if variant.get("oracle_first_order_abs") is None
                    else _finite(variant["oracle_first_order_abs"], "oracle_first_order_abs")
                ),
                "base_tokens": int(variant["base_tokens"]),
                "residual_budget_bits": int(variant["residual_budget_bits"]),
                "residual_spent_bits": int(variant["residual_spent_bits"]),
                "mode_drop": int(modes.get("drop", 0)),
                "mode_rvq1": int(modes.get("rvq1", 0)),
                "mode_rvq2": int(modes.get("rvq2", 0)),
                "mode_exact": int(modes.get("exact", 0)),
                **{name: metrics.get(name) for name in metrics},
            }
            rows.append(row)
            for component in variant["rate_components"]:
                component_rows.append(
                    {
                        "dataset": row["dataset"],
                        "dataset_index": row["dataset_index"],
                        "method": row["method"],
                        "retention_rate": rate,
                        "component": str(component["name"]),
                        "component_scope": str(component["scope"]),
                        "bits": int(component["bits"]),
                    }
                )
    return rows, component_rows


def _flatten_quality(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        for variant in sample["variants"]:
            rows.append(
                {
                    "dataset": str(sample["dataset"]),
                    "dataset_index": int(sample["dataset_index"]),
                    "sample_id": str(sample["sample_id"]),
                    "method": str(variant["method"]),
                    "scope": str(variant["scope"]),
                    "retention_rate": _rate(variant["retention_rate"]),
                    "score": _finite(variant["score"], "score"),
                    "prediction_agreement": _finite(
                        variant["agrees_with_full"], "agrees_with_full"
                    ),
                    "teacher_forced_nll": _finite(
                        variant["teacher_forced_nll"], "teacher_forced_nll"
                    ),
                    "supervised_tokens": int(variant["supervised_tokens"]),
                }
            )
    return rows


def _feature_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_with_all(rows, ("dataset", "scope", "method", "retention_rate"))
    output = []
    for (dataset, scope, method, rate), values in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        oracle_values = [
            float(row["oracle_first_order_abs"])
            for row in values
            if row["oracle_first_order_abs"] is not None
        ]
        budget_rows = [row for row in values if int(row["residual_budget_bits"]) > 0]
        mode_total = sum(
            int(row[name])
            for row in values
            for name in ("mode_drop", "mode_rvq1", "mode_rvq2", "mode_exact")
        )
        result = {
            "dataset": dataset,
            "scope": scope,
            "method": method,
            "retention_rate": rate,
            "samples": len(values),
            "feature_nmse_mean": mean(row["feature_nmse"] for row in values),
            "feature_nmse_median": median(row["feature_nmse"] for row in values),
            "feature_cosine_mean": mean(row["feature_cosine"] for row in values),
            "boundary_mse_mean": mean(row["boundary_mse"] for row in values),
            "fisher_weighted_nmse_mean": mean(
                row["fisher_weighted_nmse"] for row in values
            ),
            "oracle_first_order_abs_mean": mean(oracle_values) if oracle_values else None,
            "oracle_samples": len(oracle_values),
            "stream_bits_mean": mean(float(row["stream_bits"]) for row in values),
            "shared_bits_mean": mean(float(row["shared_bits"]) for row in values),
            "effective_bits_mean": mean(float(row["effective_bits"]) for row in values),
            "stream_bits_per_original_vector_mean": mean(
                float(row["stream_bits_per_original_vector"]) for row in values
            ),
            "effective_bits_per_original_vector_mean": mean(
                float(row["effective_bits_per_original_vector"]) for row in values
            ),
            "stream_bits_per_original_value_mean": mean(
                float(row["stream_bits_per_original_value"]) for row in values
            ),
            "effective_bits_per_original_value_mean": mean(
                float(row["effective_bits_per_original_value"]) for row in values
            ),
            "effective_compression_ratio_mean": mean(
                float(row["effective_compression_ratio"]) for row in values
            ),
            "break_even_samples_max": max(
                (
                    int(row["break_even_samples"])
                    for row in values
                    if row.get("break_even_samples") is not None
                ),
                default=None,
            ),
            "residual_budget_bits_mean": mean(
                int(row["residual_budget_bits"]) for row in values
            ),
            "residual_spent_bits_mean": mean(
                int(row["residual_spent_bits"]) for row in values
            ),
            "budget_error_bits_mean": (
                mean(
                    int(row["residual_spent_bits"]) - int(row["residual_budget_bits"])
                    for row in budget_rows
                )
                if budget_rows
                else 0.0
            ),
            "budget_absolute_relative_error_mean": (
                mean(
                    abs(
                        int(row["residual_spent_bits"])
                        - int(row["residual_budget_bits"])
                    )
                    / int(row["residual_budget_bits"])
                    for row in budget_rows
                )
                if budget_rows
                else 0.0
            ),
            "mode_drop": sum(int(row["mode_drop"]) for row in values),
            "mode_rvq1": sum(int(row["mode_rvq1"]) for row in values),
            "mode_rvq2": sum(int(row["mode_rvq2"]) for row in values),
            "mode_exact": sum(int(row["mode_exact"]) for row in values),
        }
        for mode in ("drop", "rvq1", "rvq2", "exact"):
            result[f"mode_{mode}_fraction"] = (
                result[f"mode_{mode}"] / mode_total if mode_total else 0.0
            )
        vq_mode_total = result["mode_rvq1"] + result["mode_rvq2"]
        weighted_depth = result["mode_rvq1"] + 2 * result["mode_rvq2"]
        result["mean_rvq_depth_all_blocks"] = (
            weighted_depth / mode_total if mode_total else 0.0
        )
        result["mean_rvq_depth_given_vq"] = (
            weighted_depth / vq_mode_total if vq_mode_total else 0.0
        )
        output.append(result)
    return output


def _quality_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_with_all(rows, ("dataset", "scope", "method", "retention_rate"))
    output = []
    for (dataset, scope, method, rate), values in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        losses = [row["teacher_forced_nll"] for row in values]
        tokens = [row["supervised_tokens"] for row in values]
        output.append(
            {
                "dataset": dataset,
                "scope": scope,
                "method": method,
                "retention_rate": rate,
                "samples": len(values),
                "answer_score_mean": mean(row["score"] for row in values),
                "prediction_agreement_mean": mean(
                    row["prediction_agreement"] for row in values
                ),
                "teacher_nll_mean": mean(losses),
                "teacher_nll_token_weighted": weighted_mean(losses, tokens),
                "supervised_tokens": sum(tokens),
            }
        )
    return output


def _component_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_with_all(
        rows,
        (
            "dataset",
            "method",
            "retention_rate",
            "component_scope",
            "component",
        ),
    )
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        dataset, method, rate, scope, component = key
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "retention_rate": rate,
                "component_scope": scope,
                "component": component,
                "samples": len(values),
                "bits_mean": mean(row["bits"] for row in values),
                "bits_min": min(row["bits"] for row in values),
                "bits_max": max(row["bits"] for row in values),
            }
        )
    return output


def _score_array(values: Sequence[Sequence[float]]) -> list[float]:
    return [sum(float(item) for item in row) for row in values]


def _router_metrics(samples: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows = []
    for sample in samples:
        if not sample.get("oracle"):
            continue
        for oracle_record in sample["router_oracle"]:
            rate = float(oracle_record["retention_rate"])
            oracle_matrix = oracle_record["oracle_marginal_benefits"]
            oracle_block = _score_array(oracle_matrix)
            stages = len(oracle_matrix[0])
            top_k = round(1024 * rate * 0.25 / 4)
            router_scores = oracle_record["router_marginal_scores"]
            score_sets: dict[str, tuple[list[float], list[float]]] = {
                "energy": (
                    [float(value) for value in oracle_record["energy"]],
                    [float(value) for value in oracle_record["energy"] for _ in range(stages)],
                ),
                "risk": (
                    [float(value) for value in oracle_record["risk"]],
                    [float(value) for value in oracle_record["risk"] for _ in range(stages)],
                ),
            }
            for display, method in ROUTER_METHOD_KEYS.items():
                if method not in router_scores:
                    raise RuntimeError(f"router oracle is missing {method}")
                matrix = router_scores[method]
                score_sets[display] = (
                    _score_array(matrix),
                    [float(value) for row in matrix for value in row],
                )
            oracle_flat = [float(value) for row in oracle_matrix for value in row]
            for method, (block_scores, marginal_scores) in score_sets.items():
                sample_rows.append(
                    {
                        "dataset": sample["dataset"],
                        "dataset_index": sample["dataset_index"],
                        "sample_id": sample["sample_id"],
                        "retention_rate": rate,
                        "method": method,
                        "top_k": top_k,
                        "block_spearman": spearman_correlation(block_scores, oracle_block),
                        "marginal_spearman": spearman_correlation(
                            marginal_scores, oracle_flat
                        ),
                        "topk_recall": topk_recall(block_scores, oracle_block, top_k),
                    }
                )
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(row["retention_rate"], row["method"])].append(row)
    summary = []
    for (rate, method), values in sorted(grouped.items()):
        summary.append(
            {
                "retention_rate": rate,
                "method": method,
                "samples": len(values),
                "top_k": values[0]["top_k"],
                "block_spearman_mean": mean(row["block_spearman"] for row in values),
                "block_spearman_median": median(
                    row["block_spearman"] for row in values
                ),
                "marginal_spearman_mean": mean(
                    row["marginal_spearman"] for row in values
                ),
                "topk_recall_mean": mean(row["topk_recall"] for row in values),
                "topk_recall_median": median(row["topk_recall"] for row in values),
            }
        )
    return sample_rows, summary


def _latency_summary(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["component"], row["method"], _rate(row["retention_rate"]))].append(row)
    output = []
    for (component, method, rate), values in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        output.append(
            {
                "component": component,
                "method": method,
                "retention_rate": rate,
                "samples": len(values),
                "trials_per_sample": min(int(row["trials"]) for row in values),
                "p50_ms_median": median(float(row["p50_ms"]) for row in values),
                "p50_ms_mean": mean(float(row["p50_ms"]) for row in values),
                "p95_ms_median": median(float(row["p95_ms"]) for row in values),
                "peak_allocated_bytes_max": max(
                    int(row["peak_allocated_bytes"]) for row in values
                ),
                "peak_reserved_bytes_max": max(
                    int(row["peak_reserved_bytes"]) for row in values
                ),
            }
        )
    return output


def _paired_latency(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    dynamic = "base_vq_logic_router"
    fixed = "logic_router_fixed_slots"
    components = ("layout_pack", "residual_decode_scatter", "ttft")
    indexed = {
        (
            row["dataset"],
            int(row["dataset_index"]),
            row["component"],
            row["method"],
            _rate(row["retention_rate"]),
        ): row
        for row in rows
    }
    output = []
    for rate in RATES:
        for component in components:
            keys = sorted(
                (dataset, dataset_index)
                for dataset, dataset_index, item_component, method, item_rate in indexed
                if item_component == component and method == dynamic and item_rate == rate
            )
            differences = []
            reductions = []
            fixed_lower = []
            for dataset, dataset_index in keys:
                left = indexed[(dataset, dataset_index, component, dynamic, rate)]
                right = indexed[(dataset, dataset_index, component, fixed, rate)]
                dynamic_p50 = float(left["p50_ms"])
                fixed_p50 = float(right["p50_ms"])
                differences.append(fixed_p50 - dynamic_p50)
                reductions.append((dynamic_p50 - fixed_p50) / dynamic_p50)
                fixed_lower.append(fixed_p50 < dynamic_p50)
            if not keys:
                continue
            output.append(
                {
                    "retention_rate": rate,
                    "component": component,
                    "paired_samples": len(keys),
                    "dynamic_p50_ms_median": median(
                        float(indexed[(*key, component, dynamic, rate)]["p50_ms"])
                        for key in keys
                    ),
                    "fixed_p50_ms_median": median(
                        float(indexed[(*key, component, fixed, rate)]["p50_ms"])
                        for key in keys
                    ),
                    "paired_delta_ms_median": median(differences),
                    "paired_reduction_fraction_median": median(reductions),
                    "fixed_lower_pair_fraction": mean(fixed_lower),
                    "paired_median_lower": median(differences) < 0,
                }
            )
    return output


def _lookup(
    rows: Sequence[dict[str, Any]],
    *,
    dataset: str,
    method: str,
    rate: float | None,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["method"] == method
        and _rate(row["retention_rate"]) == rate
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one summary row for {dataset}/{method}/{rate}, got {len(matches)}"
        )
    return matches[0]


def _guardrail(
    candidate: str,
    baseline: str,
    rate: float,
    feature: Sequence[dict[str, Any]],
    quality: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    candidate_scores = {
        dataset: _lookup(quality, dataset=dataset, method=candidate, rate=rate)[
            "answer_score_mean"
        ]
        for dataset in DATASETS
    }
    baseline_scores = {
        dataset: _lookup(quality, dataset=dataset, method=baseline, rate=rate)[
            "answer_score_mean"
        ]
        for dataset in DATASETS
    }
    candidate_quality = _lookup(quality, dataset="all", method=candidate, rate=rate)
    baseline_quality = _lookup(quality, dataset="all", method=baseline, rate=rate)
    candidate_feature = _lookup(feature, dataset="all", method=candidate, rate=rate)
    baseline_feature = _lookup(feature, dataset="all", method=baseline, rate=rate)
    result = quality_guardrail(
        candidate_scores,
        baseline_scores,
        candidate_nll=candidate_quality["teacher_nll_token_weighted"],
        baseline_nll=baseline_quality["teacher_nll_token_weighted"],
        candidate_nmse=candidate_feature["feature_nmse_mean"],
        baseline_nmse=baseline_feature["feature_nmse_mean"],
    )
    result.update({"candidate": candidate, "baseline": baseline, "retention_rate": rate})
    return result


def _method_points(
    feature: Sequence[dict[str, Any]], quality: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for feature_row in feature:
        if feature_row["dataset"] != "all":
            continue
        quality_row = _lookup(
            quality,
            dataset="all",
            method=feature_row["method"],
            rate=_rate(feature_row["retention_rate"]),
        )
        rows.append(
            {
                "scope": feature_row["scope"],
                "method": feature_row["method"],
                "retention_rate": feature_row["retention_rate"],
                "effective_bits_per_original_value": feature_row[
                    "effective_bits_per_original_value_mean"
                ],
                "stream_bits_per_original_value": feature_row[
                    "stream_bits_per_original_value_mean"
                ],
                "effective_bits_per_original_vector": feature_row[
                    "effective_bits_per_original_vector_mean"
                ],
                "shared_bits": feature_row["shared_bits_mean"],
                "feature_nmse": feature_row["feature_nmse_mean"],
                "feature_cosine": feature_row["feature_cosine_mean"],
                "fisher_weighted_nmse": feature_row["fisher_weighted_nmse_mean"],
                "teacher_nll": quality_row["teacher_nll_token_weighted"],
                "answer_score": quality_row["answer_score_mean"],
                "prediction_agreement": quality_row["prediction_agreement_mean"],
            }
        )
    return sorted(rows, key=lambda row: (row["method"], str(row["retention_rate"])))


def _point_index(points: Sequence[dict[str, Any]]) -> dict[tuple[str, float | None], dict[str, Any]]:
    return {(row["method"], _rate(row["retention_rate"])): row for row in points}


def _router_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[float, str], dict[str, Any]]:
    return {(float(row["retention_rate"]), row["method"]): row for row in rows}


def _decision_summary(
    feature: Sequence[dict[str, Any]],
    quality: Sequence[dict[str, Any]],
    points: Sequence[dict[str, Any]],
    routers: Sequence[dict[str, Any]],
    paired_latency: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    point = _point_index(points)
    router = _router_index(routers)
    questions: list[dict[str, Any]] = []

    scalar_points = [point[("base_scalar_quant", rate)] for rate in RATES]
    vq_points = [point[("base_vq", rate)] for rate in RATES]
    q1_feature_extensions = [
        strict_frontier_extension(
            item,
            scalar_points,
            rate_key="effective_bits_per_original_value",
            metric_key="feature_nmse",
        )
        for item in vq_points
    ]
    q1_nll_extensions = [
        strict_frontier_extension(
            item,
            scalar_points,
            rate_key="effective_bits_per_original_value",
            metric_key="teacher_nll",
        )
        for item in vq_points
    ]
    q1_guardrails = [
        _guardrail("base_vq", "base_scalar_quant", rate, feature, quality)
        for rate in RATES
    ]
    q1_pass = any(q1_feature_extensions) and any(q1_nll_extensions) and all(
        item["pass"] for item in q1_guardrails
    )
    questions.append(
        {
            "id": 1,
            "question": DECISION_QUESTIONS[1],
            "status": status_from_evidence(available=True, passed=q1_pass),
            "pass": q1_pass,
            "evidence": {
                "feature_frontier_extension_by_rate": dict(zip(RATES, q1_feature_extensions)),
                "nll_frontier_extension_by_rate": dict(zip(RATES, q1_nll_extensions)),
                "guardrails": q1_guardrails,
            },
        }
    )

    q2_rates = []
    for rate in RATES:
        fisher = _lookup(
            feature, dataset="all", method="base_vq_residual_rvq", rate=rate
        )
        base = _lookup(feature, dataset="all", method="base_vq", rate=rate)
        unweighted = _lookup(
            feature,
            dataset="all",
            method="base_vq_residual_rvq_unweighted",
            rate=rate,
        )
        versus_base = _guardrail(
            "base_vq_residual_rvq", "base_vq", rate, feature, quality
        )
        versus_unweighted = _guardrail(
            "base_vq_residual_rvq",
            "base_vq_residual_rvq_unweighted",
            rate,
            feature,
            quality,
        )
        distortion_pass = (
            fisher["fisher_weighted_nmse_mean"]
            < base["fisher_weighted_nmse_mean"]
            and fisher["fisher_weighted_nmse_mean"]
            < unweighted["fisher_weighted_nmse_mean"]
        )
        q2_rates.append(
            {
                "retention_rate": rate,
                "fisher_weighted_nmse": fisher["fisher_weighted_nmse_mean"],
                "base_only_fisher_weighted_nmse": base["fisher_weighted_nmse_mean"],
                "unweighted_fisher_weighted_nmse": unweighted[
                    "fisher_weighted_nmse_mean"
                ],
                "added_effective_bits_per_value_vs_base": fisher[
                    "effective_bits_per_original_value_mean"
                ]
                - base["effective_bits_per_original_value_mean"],
                "distortion_pass": distortion_pass,
                "guardrail_vs_base": versus_base,
                "guardrail_vs_unweighted": versus_unweighted,
                "pass": bool(
                    distortion_pass
                    and versus_base["pass"]
                    and versus_unweighted["pass"]
                ),
            }
        )
    q2_pass = all(item["pass"] for item in q2_rates)
    questions.append(
        {
            "id": 2,
            "question": DECISION_QUESTIONS[2],
            "status": status_from_evidence(available=True, passed=q2_pass),
            "pass": q2_pass,
            "evidence": {"by_rate": q2_rates},
        }
    )

    router_available = all(
        (rate, method) in router
        for rate in RATES
        for method in ("energy", "risk", "mlp", "logic")
    )
    q3_rates = []
    if router_available:
        for rate in RATES:
            heuristic_spearman = max(
                router[(rate, "energy")]["block_spearman_mean"],
                router[(rate, "risk")]["block_spearman_mean"],
            )
            heuristic_recall = max(
                router[(rate, "energy")]["topk_recall_mean"],
                router[(rate, "risk")]["topk_recall_mean"],
            )
            mlp = router[(rate, "mlp")]
            guardrail = _guardrail(
                "base_vq_mlp_router", "base_vq_residual_rvq", rate, feature, quality
            )
            spearman_delta = mlp["block_spearman_mean"] - heuristic_spearman
            recall_delta = mlp["topk_recall_mean"] - heuristic_recall
            q3_rates.append(
                {
                    "retention_rate": rate,
                    "stronger_heuristic_spearman": heuristic_spearman,
                    "stronger_heuristic_topk_recall": heuristic_recall,
                    "mlp_spearman": mlp["block_spearman_mean"],
                    "mlp_topk_recall": mlp["topk_recall_mean"],
                    "spearman_delta": spearman_delta,
                    "topk_recall_delta": recall_delta,
                    "spearman_delta_at_least_0p02": spearman_delta >= 0.02,
                    "topk_delta_at_least_0p02": recall_delta >= 0.02,
                    "guardrail": guardrail,
                    "pass": bool(
                        spearman_delta >= 0.02
                        and recall_delta >= 0.02
                        and guardrail["pass"]
                    ),
                }
            )
    q3_pass = router_available and all(item["pass"] for item in q3_rates)
    questions.append(
        {
            "id": 3,
            "question": DECISION_QUESTIONS[3],
            "status": status_from_evidence(available=router_available, passed=q3_pass),
            "pass": q3_pass,
            "evidence": {"by_rate": q3_rates},
        }
    )

    q4_rates = []
    q4_available = bool(q3_pass and router_available)
    if q4_available:
        for rate in RATES:
            heuristic_spearman = max(
                router[(rate, "energy")]["block_spearman_mean"],
                router[(rate, "risk")]["block_spearman_mean"],
            )
            heuristic_recall = max(
                router[(rate, "energy")]["topk_recall_mean"],
                router[(rate, "risk")]["topk_recall_mean"],
            )
            mlp = router[(rate, "mlp")]
            logic = router[(rate, "logic")]
            spearman_retention = (
                logic["block_spearman_mean"] - heuristic_spearman
            ) / (mlp["block_spearman_mean"] - heuristic_spearman)
            recall_retention = (
                logic["topk_recall_mean"] - heuristic_recall
            ) / (mlp["topk_recall_mean"] - heuristic_recall)
            guardrail = _guardrail(
                "base_vq_logic_router", "base_vq_mlp_router", rate, feature, quality
            )
            q4_rates.append(
                {
                    "retention_rate": rate,
                    "spearman_improvement_retained": spearman_retention,
                    "topk_improvement_retained": recall_retention,
                    "guardrail": guardrail,
                    "pass": bool(
                        spearman_retention >= 0.9
                        and recall_retention >= 0.9
                        and guardrail["pass"]
                    ),
                }
            )
    q4_pass = q4_available and all(item["pass"] for item in q4_rates)
    questions.append(
        {
            "id": 4,
            "question": DECISION_QUESTIONS[4],
            "status": status_from_evidence(available=q4_available, passed=q4_pass),
            "pass": q4_pass,
            "evidence": {
                "prerequisite_question_3_pass": q3_pass,
                "by_rate": q4_rates,
            },
        }
    )

    latency_index = {
        (float(row["retention_rate"]), row["component"]): row
        for row in paired_latency
    }
    q5_available = all(
        (rate, component) in latency_index
        for rate in RATES
        for component in ("layout_pack", "residual_decode_scatter", "ttft")
    )
    q5_rates = []
    if q5_available:
        for rate in RATES:
            components = {
                component: latency_index[(rate, component)]
                for component in ("layout_pack", "residual_decode_scatter", "ttft")
            }
            guardrail = _guardrail(
                "logic_router_fixed_slots",
                "base_vq_logic_router",
                rate,
                feature,
                quality,
            )
            q5_rates.append(
                {
                    "retention_rate": rate,
                    "components": components,
                    "all_paired_medians_lower": all(
                        row["paired_median_lower"] for row in components.values()
                    ),
                    "guardrail": guardrail,
                    "pass": bool(
                        all(row["paired_median_lower"] for row in components.values())
                        and guardrail["pass"]
                    ),
                }
            )
    q5_pass = q5_available and all(item["pass"] for item in q5_rates)
    questions.append(
        {
            "id": 5,
            "question": DECISION_QUESTIONS[5],
            "status": status_from_evidence(available=q5_available, passed=q5_pass),
            "pass": q5_pass,
            "evidence": {
                "by_rate": q5_rates,
                "quality_path_expands_to_1280_visual_tokens": True,
                "native_compact_prefill_speedup_established": False,
            },
        }
    )

    fixed_points = [point[("logic_router_fixed_slots", rate)] for rate in RATES]
    exact_points = [
        point[("logic_router_fixed_slots_exact_fallback", rate)] for rate in RATES
    ]
    q6_rates = []
    for rate, exact in zip(RATES, exact_points):
        feature_extension = strict_frontier_extension(
            exact,
            fixed_points,
            rate_key="effective_bits_per_original_value",
            metric_key="feature_nmse",
        )
        nll_extension = strict_frontier_extension(
            exact,
            fixed_points,
            rate_key="effective_bits_per_original_value",
            metric_key="teacher_nll",
        )
        fixed = point[("logic_router_fixed_slots", rate)]
        feature_improvement = -relative_increase(
            exact["feature_nmse"], fixed["feature_nmse"]
        )
        nll_improvement = -relative_increase(exact["teacher_nll"], fixed["teacher_nll"])
        guardrail = _guardrail(
            "logic_router_fixed_slots_exact_fallback",
            "logic_router_fixed_slots",
            rate,
            feature,
            quality,
        )
        q6_rates.append(
            {
                "retention_rate": rate,
                "feature_frontier_extension": feature_extension,
                "nll_frontier_extension": nll_extension,
                "feature_nmse_improvement": feature_improvement,
                "teacher_nll_improvement": nll_improvement,
                "material_improvement": bool(
                    feature_improvement >= 0.05 or nll_improvement >= 0.01
                ),
                "guardrail": guardrail,
                "pass": bool(
                    feature_extension
                    and nll_extension
                    and (feature_improvement >= 0.05 or nll_improvement >= 0.01)
                    and guardrail["pass"]
                ),
            }
        )
    q6_pass = any(item["pass"] for item in q6_rates)
    questions.append(
        {
            "id": 6,
            "question": DECISION_QUESTIONS[6],
            "status": status_from_evidence(available=True, passed=q6_pass),
            "pass": q6_pass,
            "evidence": {"by_rate": q6_rates},
        }
    )

    return {
        "format": "tilelogic_rvq_decisions_v1",
        "questions": questions,
        "all_questions_pass": all(question["status"] == "PASS" for question in questions),
        "aggregate_positive_claim_allowed": all(
            question["status"] == "PASS" for question in questions
        ),
    }


def _decision_csv(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": item["id"],
            "question": item["question"],
            "status": item["status"],
            "pass": item["pass"],
            "evidence_json": json.dumps(item["evidence"], sort_keys=True),
        }
        for item in decisions["questions"]
    ]


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-4):
            return f"{value:.4e}"
        return f"{value:.6f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_format_value(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _plot_results(
    output_dir: Path,
    points: Sequence[dict[str, Any]],
    routers: Sequence[dict[str, Any]],
    paired_latency: Sequence[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    method_order = list(MAIN_TILELOGIC_METHODS) + list(ABLATION_METHODS)
    colors = plt.get_cmap("tab20")
    outputs = []
    for metric, label, logarithmic in (
        ("feature_nmse", "Feature NMSE", True),
        ("teacher_nll", "Teacher-forced NLL", False),
    ):
        fig, axis = plt.subplots(figsize=(10.5, 6.5))
        for index, method in enumerate(method_order):
            selected = [row for row in points if row["method"] == method]
            if not selected:
                continue
            selected.sort(key=lambda row: row["effective_bits_per_original_value"])
            axis.plot(
                [row["effective_bits_per_original_value"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                linewidth=1.5,
                color=colors(index),
                label=method,
            )
        axis.set_xlabel("Effective bits per original crop scalar (shared overhead / 360)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        if logarithmic:
            axis.set_yscale("log")
        axis.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        name = f"rate_distortion_{metric}.png"
        fig.savefig(figure_dir / name, dpi=180)
        fig.savefig((figure_dir / name).with_suffix(".pdf"))
        plt.close(fig)
        outputs.append(f"figures/{name}")

    if routers:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        methods = ("energy", "risk", "mlp", "logic")
        width = 0.18
        for panel, (key, title) in enumerate(
            (("block_spearman_mean", "Block Spearman"), ("topk_recall_mean", "Top-k recall"))
        ):
            axis = axes[panel]
            for method_index, method in enumerate(methods):
                values = [
                    next(
                        row[key]
                        for row in routers
                        if row["method"] == method and row["retention_rate"] == rate
                    )
                    for rate in RATES
                ]
                axis.bar(
                    [index + (method_index - 1.5) * width for index in range(len(RATES))],
                    values,
                    width=width,
                    label=method,
                )
            axis.set_xticks(range(len(RATES)), [f"{rate:.1%}" for rate in RATES])
            axis.set_title(title)
            axis.grid(True, axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        name = "router_metrics.png"
        fig.savefig(figure_dir / name, dpi=180)
        fig.savefig((figure_dir / name).with_suffix(".pdf"))
        plt.close(fig)
        outputs.append(f"figures/{name}")

    if paired_latency:
        fig, axis = plt.subplots(figsize=(10, 5))
        labels = [
            f"{row['component']}\n{float(row['retention_rate']):.1%}"
            for row in paired_latency
        ]
        positions = list(range(len(labels)))
        axis.bar(
            [position - 0.2 for position in positions],
            [row["dynamic_p50_ms_median"] for row in paired_latency],
            width=0.4,
            label="dynamic logic",
        )
        axis.bar(
            [position + 0.2 for position in positions],
            [row["fixed_p50_ms_median"] for row in paired_latency],
            width=0.4,
            label="fixed slots",
        )
        axis.set_xticks(positions, labels, fontsize=8)
        axis.set_ylabel("Median per-sample p50 latency (ms)")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        name = "dynamic_fixed_latency.png"
        fig.savefig(figure_dir / name, dpi=180)
        fig.savefig((figure_dir / name).with_suffix(".pdf"))
        plt.close(fig)
        outputs.append(f"figures/{name}")
    return outputs


def _report(
    output_dir: Path,
    feature_summary: Sequence[dict[str, Any]],
    quality_summary: Sequence[dict[str, Any]],
    points: Sequence[dict[str, Any]],
    router_summary: Sequence[dict[str, Any]],
    latency_summary: Sequence[dict[str, Any]],
    paired_latency: Sequence[dict[str, Any]],
    decisions: dict[str, Any],
    environments: dict[str, Any],
    figures: Sequence[str],
) -> str:
    decision_table = _markdown_table(
        ("ID", "Question", "Status"),
        (
            (item["id"], item["question"], item["status"])
            for item in decisions["questions"]
        ),
    )
    point_table = _markdown_table(
        (
            "Method",
            "Rate",
            "Eff. bit/value",
            "Stream bit/value",
            "NMSE",
            "Fisher NMSE",
            "NLL",
            "Score",
        ),
        (
            (
                row["method"],
                row["retention_rate"],
                row["effective_bits_per_original_value"],
                row["stream_bits_per_original_value"],
                row["feature_nmse"],
                row["fisher_weighted_nmse"],
                row["teacher_nll"],
                row["answer_score"],
            )
            for row in points
            if row["method"] != "none"
        ),
    )
    quality_index = {
        (row["dataset"], row["method"], _rate(row["retention_rate"])): row
        for row in quality_summary
    }
    dataset_quality_table = _markdown_table(
        (
            "Method",
            "Rate",
            "GQA score",
            "TextVQA score",
            "ChartQA score",
            "Aggregate NLL",
        ),
        (
            (
                row["method"],
                row["retention_rate"],
                quality_index[("gqa", row["method"], _rate(row["retention_rate"]))][
                    "answer_score_mean"
                ],
                quality_index[
                    ("textvqa", row["method"], _rate(row["retention_rate"]))
                ]["answer_score_mean"],
                quality_index[
                    ("chartqa", row["method"], _rate(row["retention_rate"]))
                ]["answer_score_mean"],
                row["teacher_nll"],
            )
            for row in points
        ),
    )
    router_table = _markdown_table(
        ("Rate", "Router", "Spearman", "Marginal rho", "Top-k recall", "N"),
        (
            (
                row["retention_rate"],
                row["method"],
                row["block_spearman_mean"],
                row["marginal_spearman_mean"],
                row["topk_recall_mean"],
                row["samples"],
            )
            for row in router_summary
        ),
    )
    latency_table = _markdown_table(
        ("Rate", "Component", "Dynamic p50 ms", "Fixed p50 ms", "Paired reduction", "Lower?"),
        (
            (
                row["retention_rate"],
                row["component"],
                row["dynamic_p50_ms_median"],
                row["fixed_p50_ms_median"],
                row["paired_reduction_fraction_median"],
                row["paired_median_lower"],
            )
            for row in paired_latency
        ),
    )
    latency_full_table = _markdown_table(
        ("Component", "Method", "Rate", "p50 ms", "p95 ms", "Peak alloc MiB", "Peak reserve MiB"),
        (
            (
                row["component"],
                row["method"],
                row["retention_rate"],
                row["p50_ms_median"],
                row["p95_ms_median"],
                row["peak_allocated_bytes_max"] / (1024 * 1024),
                row["peak_reserved_bytes_max"] / (1024 * 1024),
            )
            for row in latency_summary
        ),
    )
    mode_rows = [
        row
        for row in feature_summary
        if row["dataset"] == "all"
        and row["method"]
        in {
            "base_vq_residual_rvq",
            "base_vq_mlp_router",
            "base_vq_logic_router",
            "logic_router_fixed_slots",
            "logic_router_fixed_slots_exact_fallback",
        }
    ]
    mode_table = _markdown_table(
        (
            "Method",
            "Rate",
            "Drop",
            "RVQ1",
            "RVQ2",
            "Exact",
            "Depth/all",
            "Depth/VQ-active",
            "Budget abs. rel. err.",
        ),
        (
            (
                row["method"],
                row["retention_rate"],
                row["mode_drop_fraction"],
                row["mode_rvq1_fraction"],
                row["mode_rvq2_fraction"],
                row["mode_exact_fraction"],
                row["mean_rvq_depth_all_blocks"],
                row["mean_rvq_depth_given_vq"],
                row["budget_absolute_relative_error_mean"],
            )
            for row in mode_rows
        ),
    )
    figure_lines = "\n".join(f"![{Path(path).stem}]({path})" for path in figures)
    rate_correction = environments["rate_precision_correction"]
    cache_provenance = environments["cache_provenance_backfill"]
    gpu_snapshot = environments["latency"].get("gpu_co_residency_log")
    latency_provenance_note = (
        "A during-run GPU process/pmon snapshot is archived as "
        f"`{gpu_snapshot['file']}` (SHA256 `{gpu_snapshot['sha256']}`). "
        "It records co-residency for diagnostic interpretation and does not "
        "establish an exclusive-GPU performance signoff."
        if gpu_snapshot
        else "No during-run GPU co-residency snapshot was captured."
    )
    all_pass = decisions["all_questions_pass"]
    conclusion = (
        "All six predeclared questions pass."
        if all_pass
        else "The evidence does not support an aggregate positive claim; see the independent decisions."
    )
    text = f"""# TileLogic-RVQ Formal Experiment Report

## 1. Executive Summary

- Feature/rate/oracle evaluation: **PASS**, 360 evaluation samples and 23 variants per sample.
- Quality and teacher-NLL evaluation: **PASS**, paired on the same 360 samples.
- Latency diagnostics: **PASS as diagnostics**, with paired dynamic/fixed component and TTFT rows.
- Aggregate positive claim allowed: **{'YES' if all_pass else 'NO'}**.
- {conclusion}

The quality path reconstructs the original 1,280 visual tokens before Qwen execution. It is information-quality evidence, not native compact-prefill latency evidence. No PPA, kernel-fusion, or physical-hardware claim is made.

## 2. Independent Decision Questions

{decision_table}

Each decision follows `TILELOGIC_RVQ_EXPERIMENT.md`. Missing required evidence is `INCONCLUSIVE`; complete evidence that misses a threshold is `FAIL`. Detailed machine-readable evidence is in `decision_summary.json`.

## 3. Protocol And Leakage Controls

- Datasets: GQA, TextVQA, ChartQA; 80 calibration and 120 evaluation examples per dataset.
- Oracle subsets: 16 calibration and 16 disjoint evaluation examples per dataset.
- Codebooks, Fisher weights, router normalization/MLP/tree, and fixed slots use calibration records only.
- Feature and quality evaluation use exactly 360 evaluation records; calibration records loaded by formal evaluation: 0.
- Shared overhead is amortized over exactly 360 evaluation samples. Stream-only rate and break-even count remain separately reported.
- Rate precision is tied to executed, explicitly serialized, or exact round-tripped logical payloads: FP32 INT4 scales, VQ metric weights, MLP/normalizer state, logic leaves, and curvature priors; FP16 codewords, scale tables, logic thresholds, and exact fallback values.
- The rate correction was checked over {rate_correction['records']} samples and {rate_correction['compared_variants']} variants; all compared non-rate fields are structurally identical, with run-time-only `elapsed_seconds` excluded.
- Cache provenance covers {cache_provenance['records']} entries and records the source-manifest hash, model revision, and tensor dtypes without changing cached tensor payloads.
- Model: `{environments['quality'].get('model_dir', 'unknown')}`.
- Feature GPU: `{environments['feature'].get('gpu', 'unknown')}`; latency GPU: `{environments['latency'].get('gpu', 'unknown')}`.

## 4. Rate-Distortion And Quality

{point_table}

Paired dataset-level answer scores and aggregate token-weighted teacher NLL:

{dataset_quality_table}

`effective_bits_per_original_value` includes codebooks, metric weights, scale tables, router parameters/normalizers/tree state, curvature priors, and fixed-slot metadata amortized over the evaluation set. `rate_components.csv` preserves every stream/shared component. The machine audit loads the stored artifacts and rejects any rate component whose charged precision differs from execution, explicit serialization, or an exact declared-precision round trip.

{figure_lines}

## 5. Router Metrics

Decision metrics use per-block cumulative marginal benefit. Spearman and top-k recall are computed on the 48 disjoint evaluation-oracle samples. Marginal Spearman is retained as a stage-aware diagnostic.

{router_table}

## 6. Mode Usage And Budget Error

{mode_table}

Exact fallback mode includes the complete FP16 2x2xC residual payload. Budget error is reported from the emitted stream fields rather than silently normalized away.

## 7. Measured Latency And Peak Memory

Paired dynamic-versus-fixed decision components:

{latency_table}

All measured component, prefill, and TTFT summaries:

{latency_full_table}

Full component rows, p50/p95/p99, and peak allocated/reserved memory are in `latency_metrics.csv`. TTFT includes image preprocessing, visual encoding, codec reconstruction, native multimodal positions, language prefill, and first-token generation. Both compressed paths expand to 1,280 visual tokens, so these rows do not establish a native compact-sequence prefill benefit.

{latency_provenance_note}

## 8. Evidence Boundaries

- Base/residual codebooks are continuous FP16 tables; the logic router controls path/depth and does not replace the reconstruction datapath.
- Lower precision is never charged for an FP32 value unless the encoder performs that lower-precision round trip before reconstruction or routing.
- MLP is a calibration-only routing upper bound. The logic tree is discrete and uses deployable, quantized-feature thresholds.
- Fisher distortion is a diagonal empirical proxy, not an exact Hessian.
- Teacher-forced NLL uses the first manifest answer verbatim; dataset answer scores use all manifest answers.
- GPU timing is an implementation diagnostic for the current PyTorch path, not ASIC/FPGA PPA.
- Answer scores and teacher NLL cover the stated 360-example protocol only.

## 9. Output Map

- `feature_metrics.csv`: feature, Fisher, rate, mode, and budget aggregates.
- `quality_metrics.csv`: dataset scores, agreement, and teacher NLL.
- `method_points.csv`: overall rate-distortion points.
- `rate_components.csv`: exact stream/shared bit components.
- `router_sample_metrics.csv` and `router_metrics.csv`: oracle routing evidence.
- `latency_metrics.csv` and `latency_paired_dynamic_fixed.csv`: timing and memory.
- `decision_summary.json` and `decision_summary.csv`: six frozen-rule decisions.
- `result_audit_report.md`: independent machine audit.
- `../rate_precision_correction_validation.json`: old/new feature-ledger semantic comparison.
- `../cache_provenance_backfill.json`: hash-linked cache provenance migration.
"""
    path = output_dir / "TILELOGIC_RVQ_FINAL_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-eval-dir", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--latency-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    feature_dir = args.feature_eval_dir.resolve()
    quality_dir = args.quality_dir.resolve()
    latency_dir = args.latency_dir.resolve()
    training_dir = args.training_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_environment = _read_json(feature_dir / "feature_eval_summary.json")
    quality_environment = _read_json(quality_dir / "quality_environment.json")
    quality_run = _read_json(quality_dir / "quality_summary.json")
    latency_environment = _read_json(latency_dir / "latency_environment.json")
    run_dir = feature_dir.parent
    rate_correction = _read_json(run_dir / "rate_precision_correction_validation.json")
    cache_provenance = _read_json(run_dir / "cache_provenance_backfill.json")
    gpu_snapshot_path = latency_dir / "gpu_co_residency_during_run.log"
    if gpu_snapshot_path.is_file():
        latency_environment = dict(latency_environment)
        latency_environment["gpu_co_residency_log"] = {
            "file": gpu_snapshot_path.name,
            "bytes": gpu_snapshot_path.stat().st_size,
            "sha256": hashlib.sha256(gpu_snapshot_path.read_bytes()).hexdigest(),
        }
    training = _read_json(training_dir / "training_summary.json")
    if not feature_environment.get("complete") or feature_environment.get("records") != 360:
        raise RuntimeError("feature evaluation is incomplete")
    if feature_environment.get("rate_storage_policy") != RATE_STORAGE_POLICY:
        raise RuntimeError("feature evaluation uses a stale rate-storage policy")
    if (
        rate_correction.get("records") != 360
        or rate_correction.get("compared_variants") != 360 * 23
        or rate_correction.get("non_rate_semantics_identical") is not True
        or rate_correction.get("errors")
        or rate_correction.get("new_feature_samples_sha256")
        != hashlib.sha256((feature_dir / "feature_samples.jsonl").read_bytes()).hexdigest()
    ):
        raise RuntimeError("rate-precision correction validation is incomplete")
    if (
        cache_provenance.get("records") != 600
        or cache_provenance.get("payload_tensors_unchanged") is not True
        or not cache_provenance.get("source_manifest_sha256")
        or not cache_provenance.get("model_revision")
    ):
        raise RuntimeError("cache provenance migration evidence is incomplete")
    if not quality_run.get("complete") or quality_run.get("records") != 360:
        raise RuntimeError("quality evaluation is incomplete")
    if not training.get("complete") or training.get("evaluation_entries_loaded") != 0:
        raise RuntimeError("training is incomplete or reports evaluation leakage")

    feature_samples = _read_jsonl(feature_dir / "feature_samples.jsonl")
    quality_samples = _read_jsonl(quality_dir / "quality_samples.jsonl")
    latency_rows = _read_csv(latency_dir / "latency_samples.csv")
    feature_rows, component_rows = _flatten_feature(feature_samples)
    quality_rows = _flatten_quality(quality_samples)
    feature_summary = _feature_summary(feature_rows)
    quality_summary = _quality_summary(quality_rows)
    component_summary = _component_summary(component_rows)
    router_samples, router_summary = _router_metrics(feature_samples)
    latency_summary = _latency_summary(latency_rows)
    paired_latency = _paired_latency(latency_rows)
    points = _method_points(feature_summary, quality_summary)
    decisions = _decision_summary(
        feature_summary,
        quality_summary,
        points,
        router_summary,
        paired_latency,
    )

    _write_csv(output_dir / "feature_metrics.csv", feature_summary)
    _write_csv(output_dir / "quality_metrics.csv", quality_summary)
    _write_csv(output_dir / "rate_components.csv", component_summary)
    _write_csv(output_dir / "router_sample_metrics.csv", router_samples)
    _write_csv(output_dir / "router_metrics.csv", router_summary)
    _write_csv(output_dir / "latency_metrics.csv", latency_summary)
    _write_csv(output_dir / "latency_paired_dynamic_fixed.csv", paired_latency)
    _write_csv(output_dir / "method_points.csv", points)
    _write_json(output_dir / "decision_summary.json", decisions)
    _write_csv(output_dir / "decision_summary.csv", _decision_csv(decisions))
    environments = {
        "format": ANALYSIS_FORMAT,
        "feature": feature_environment,
        "quality": quality_environment,
        "quality_run": quality_run,
        "latency": latency_environment,
        "training": training,
        "rate_precision_correction": rate_correction,
        "cache_provenance_backfill": cache_provenance,
    }
    _write_json(output_dir / "analysis_environment.json", environments)
    figures = [] if args.skip_plots else _plot_results(
        output_dir, points, router_summary, paired_latency
    )
    _report(
        output_dir,
        feature_summary,
        quality_summary,
        points,
        router_summary,
        latency_summary,
        paired_latency,
        decisions,
        environments,
        figures,
    )
    print(
        json.dumps(
            {
                "format": ANALYSIS_FORMAT,
                "feature_samples": len(feature_samples),
                "quality_samples": len(quality_samples),
                "latency_rows": len(latency_rows),
                "router_oracle_samples": len({(row['dataset'], row['dataset_index']) for row in router_samples}),
                "decision_statuses": {
                    str(item["id"]): item["status"] for item in decisions["questions"]
                },
                "all_questions_pass": decisions["all_questions_pass"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
