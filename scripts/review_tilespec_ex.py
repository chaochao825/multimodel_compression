#!/usr/bin/env python3
"""Independent result audit for the TileSpec-Ex minimal experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from tilespec_ex.core import METHODS, RETENTION_RATES
from tilespec_ex.metrics import (
    dataset_score,
    normalize_answer,
    paired_bootstrap_interval,
    spearman_correlation,
)


EXPECTED_VARIANTS = 1 + len(RETENTION_RATES) * ((len(METHODS) - 1) + 2)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSON at {path}:{line_number}") from error
    return output


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _add(
    findings: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: str,
    *,
    severity: str = "major",
) -> None:
    findings.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "severity": "none" if passed else severity,
            "evidence": evidence,
        }
    )


def _close(lhs: float, rhs: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(lhs, rhs, rel_tol=tolerance, abs_tol=tolerance)


def _quality_variant_index(
    samples: list[dict[str, Any]],
) -> dict[tuple[str, int, str, float | None], dict[str, Any]]:
    output = {}
    for sample in samples:
        for variant in sample["variants"]:
            key = (
                str(sample["dataset"]),
                int(sample["dataset_index"]),
                str(variant["method"]),
                None
                if variant["retention_rate"] is None
                else float(variant["retention_rate"]),
            )
            if key in output:
                raise RuntimeError(f"duplicate quality variant: {key}")
            output[key] = variant
    return output


def _raw_quality_comparison(
    index: dict[tuple[str, int, str, float | None], dict[str, Any]],
    lhs_method: str,
    rhs_method: str,
    rate: float,
    datasets: set[str],
) -> dict[str, Any]:
    sample_keys = sorted(
        (dataset, dataset_index)
        for dataset, dataset_index, method, item_rate in index
        if dataset in datasets and method == lhs_method and item_rate == rate
    )
    lhs = [index[(*key, lhs_method, rate)] for key in sample_keys]
    rhs = [index[(*key, rhs_method, rate)] for key in sample_keys]
    baselines = [index[(*key, "none", None)] for key in sample_keys]
    lhs_scores = [float(item["score"]) for item in lhs]
    rhs_scores = [float(item["score"]) for item in rhs]
    accuracy_delta, _, _ = paired_bootstrap_interval(lhs_scores, rhs_scores)
    lhs_boundary = [float(item["boundary_mse"]) for item in lhs]
    rhs_boundary = [float(item["boundary_mse"]) for item in rhs]
    _, boundary_lower, _ = paired_bootstrap_interval(rhs_boundary, lhs_boundary)
    lhs_losses = sum(
        float(base["score"]) > float(item["score"]) + 1e-12
        for base, item in zip(baselines, lhs)
    )
    rhs_losses = sum(
        float(base["score"]) > float(item["score"]) + 1e-12
        for base, item in zip(baselines, rhs)
    )
    return {
        "accuracy_delta": accuracy_delta,
        "prediction_loss_event_reduction": 0.0
        if rhs_losses == 0
        else 1.0 - lhs_losses / rhs_losses,
        "boundary_mse_reduction": 1.0
        - mean(lhs_boundary) / max(mean(rhs_boundary), 1e-12),
        "boundary_delta_ci95_lower": boundary_lower,
        "same_budget": all(
            int(left["retained_crop_tokens"]) == int(right["retained_crop_tokens"])
            for left, right in zip(lhs, rhs)
        ),
    }


def _recompute_gate_boole(gates: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    tile = gates["tile_local_better_than_global"]
    tile_rates = []
    for row in tile["by_rate"]:
        expected = bool(
            row["at_least_one_sensitive_win"]
            and row["gqa_not_worse"]
            and row["same_budget"]
        )
        tile_rates.append(expected)
        if expected != row["pass"]:
            errors.append(f"tile rate {row['retention_rate']} boolean mismatch")
    if all(tile_rates) != tile["pass"]:
        errors.append("tile aggregate boolean mismatch")

    risk = gates["risk_exception_better_than_energy"]
    risk_rates = []
    for row in risk["by_rate"]:
        expected = all(
            row[key]
            for key in (
                "task_delta_at_least_2pp",
                "gqa_not_worse",
                "oracle_significantly_better",
                "same_budget",
                "scorer_below_5pct",
            )
        )
        risk_rates.append(expected)
        if expected != row["pass"]:
            errors.append(f"risk rate {row['retention_rate']} boolean mismatch")
    if all(risk_rates) != risk["pass"]:
        errors.append("risk aggregate boolean mismatch")

    structured = gates["structured_block_real_latency_benefit"]
    candidate_values = {}
    for candidate in structured["candidate_pass"]:
        candidate_rows = [
            row
            for row in structured["by_candidate_rate"]
            if row["candidate"] == candidate
        ]
        for row in candidate_rows:
            diagnostic_expected = all(
                row[key]
                for key in (
                    "accuracy_loss_within_0_5pp",
                "gather_reduction_at_least_30pct",
                "total_reduction_at_least_10pct",
                "all_micro_batches_faster",
                "full_decoder_prefill_reduction_at_least_10pct",
                "all_prefill_batches_faster",
                "fixed_output_shape",
                )
            )
            expected = bool(row["validated"] and diagnostic_expected)
            expected_status = (
                "PASS"
                if expected
                else "FAIL"
                if row["validated"]
                else "INCONCLUSIVE"
            )
            if diagnostic_expected != row["diagnostic_pass"]:
                errors.append(
                    f"structured {candidate} rate {row['retention_rate']} diagnostic mismatch"
                )
            if expected != row["pass"]:
                errors.append(
                    f"structured {candidate} rate {row['retention_rate']} mismatch"
                )
            if expected_status != row["status"]:
                errors.append(
                    f"structured {candidate} rate {row['retention_rate']} status mismatch"
                )
        candidate_values[candidate] = all(row["pass"] for row in candidate_rows)
        if candidate_values[candidate] != structured["candidate_pass"][candidate]:
            errors.append(f"structured candidate {candidate} aggregate mismatch")
    expected_structured = bool(
        structured["validated"] and any(candidate_values.values())
    )
    expected_structured_status = (
        "PASS"
        if expected_structured
        else "FAIL"
        if structured["validated"]
        else "INCONCLUSIVE"
    )
    if expected_structured != structured["pass"]:
        errors.append("structured aggregate boolean mismatch")
    if expected_structured_status != structured["status"]:
        errors.append("structured aggregate status mismatch")

    all_three = bool(tile["pass"] and risk["pass"] and structured["pass"])
    if all_three != gates["all_three_pass"]:
        errors.append("all-three boolean mismatch")
    return not errors, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    findings: list[dict[str, Any]] = []

    required = [
        "quality_samples.jsonl",
        "oracle_blocks.jsonl",
        "environment.json",
        "quality_run_summary.json",
        "rescore_summary.json",
        "latency_samples.csv",
        "latency_environment.json",
        "quality_metrics.csv",
        "tile_local_comparison.csv",
        "risk_comparison.csv",
        "oracle_correlations.csv",
        "structured_latency_comparison.csv",
        "structured_task_comparison.csv",
        "gate_summary.json",
        "TILESPEC_EX_MINIMAL_REPORT.md",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    _add(findings, "required_outputs", not missing, f"missing={missing}")
    if missing:
        raise SystemExit("required result files are missing")

    quality = _jsonl(root / "quality_samples.jsonl")
    oracle = _jsonl(root / "oracle_blocks.jsonl")
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    latency_environment = json.loads(
        (root / "latency_environment.json").read_text(encoding="utf-8")
    )
    run_summary = json.loads(
        (root / "quality_run_summary.json").read_text(encoding="utf-8")
    )
    gates = json.loads((root / "gate_summary.json").read_text(encoding="utf-8"))
    latency = _csv(root / "latency_samples.csv")
    oracle_summary = _csv(root / "oracle_correlations.csv")
    structured_latency_summary = _csv(root / "structured_latency_comparison.csv")

    counts = Counter(str(row["dataset"]) for row in quality)
    contract_count_ok = counts == Counter({"gqa": 200, "textvqa": 200, "chartqa": 200})
    _add(
        findings,
        "three_dataset_200_sample_contract",
        contract_count_ok,
        f"counts={dict(counts)}",
    )
    _add(
        findings,
        "quality_run_completed",
        bool(run_summary.get("complete")) and run_summary.get("quality_records") == 600,
        json.dumps(run_summary, sort_keys=True),
    )

    variant_errors = []
    score_errors = []
    derived_answer_errors = []
    budget_errors = []
    grid_errors = []
    method_contract = {
        (method, rate)
        for rate in RETENTION_RATES
        for method in METHODS
        if method != "none"
    }
    for sample in quality:
        variants = sample["variants"]
        if len(variants) != EXPECTED_VARIANTS:
            variant_errors.append(
                f"{sample['dataset']}:{sample['dataset_index']} variants={len(variants)}"
            )
            continue
        main = {
            (item["method"], item["retention_rate"])
            for item in variants
            if item["scope"] == "main" and item["method"] != "none"
        }
        baseline_count = sum(item["method"] == "none" for item in variants)
        if main != method_contract or baseline_count != 1:
            variant_errors.append(
                f"{sample['dataset']}:{sample['dataset_index']} main contract mismatch"
            )
        baseline = next(item for item in variants if item["method"] == "none")
        baseline_normalized = normalize_answer(baseline["prediction"])
        if (
            sample["crop_grid_hw"] != [16, 16]
            or sample["full_visual_tokens"] != 1280
            or sample["thumbnail_tokens"] != 256
            or sample["crop_tokens"] != 1024
        ):
            grid_errors.append(f"{sample['dataset']}:{sample['dataset_index']}")
        for variant in variants:
            recomputed = dataset_score(
                sample["dataset"], variant["prediction"], sample["answers"]
            )
            if not _close(float(variant["score"]), recomputed):
                score_errors.append(
                    f"{sample['dataset']}:{sample['dataset_index']}:{variant['method']}"
                )
            normalized = normalize_answer(variant["prediction"])
            agrees = float(normalized == baseline_normalized)
            if (
                variant.get("normalized_prediction") != normalized
                or not _close(float(variant.get("agrees_with_full", -1.0)), agrees)
            ):
                derived_answer_errors.append(
                    f"{sample['dataset']}:{sample['dataset_index']}:{variant['method']}"
                )
            expected_crop = (
                1024
                if variant["method"] == "none"
                else round(1024 * float(variant["retention_rate"]))
            )
            if (
                variant["retained_crop_tokens"] != expected_crop
                or variant["compact_visual_tokens"] != 256 + expected_crop
                or variant["base_tokens"] + variant["exception_tokens"]
                != expected_crop
            ):
                budget_errors.append(
                    f"{sample['dataset']}:{sample['dataset_index']}:{variant['method']}"
                )
    _add(
        findings,
        "six_main_methods_two_rates",
        not variant_errors,
        f"errors={variant_errors[:5]}, expected_variants={EXPECTED_VARIANTS}",
    )
    _add(
        findings,
        "five_image_multitile_grid",
        not grid_errors,
        f"errors={grid_errors[:5]}",
    )
    _add(
        findings,
        "answer_scores_recomputed",
        not score_errors,
        f"errors={score_errors[:5]}",
    )
    _add(
        findings,
        "answer_derived_fields_recomputed",
        not derived_answer_errors,
        f"errors={derived_answer_errors[:5]}",
    )
    _add(
        findings,
        "exact_equal_budget_accounting",
        not budget_errors,
        f"errors={budget_errors[:5]}",
    )

    oracle_counts = Counter(str(row["dataset"]) for row in oracle)
    oracle_errors = []
    recomputed_rhos: dict[tuple[str, int, float], tuple[float, float]] = {}
    for record in oracle:
        if len(record["rates"]) != 2:
            oracle_errors.append(f"{record['dataset']}:{record['dataset_index']}:rates")
        for rate in record["rates"]:
            arrays = [
                rate["locations"],
                rate["energy"],
                rate["relevance"],
                rate["risk"],
                rate["oracle_first_order_abs"],
            ]
            if any(len(array) != 256 for array in arrays):
                oracle_errors.append(
                    f"{record['dataset']}:{record['dataset_index']}:{rate['retention_rate']}:length"
                )
                continue
            if len({tuple(item) for item in rate["locations"]}) != 256:
                oracle_errors.append(
                    f"{record['dataset']}:{record['dataset_index']}:{rate['retention_rate']}:locations"
                )
            for energy, relevance, risk in zip(
                rate["energy"], rate["relevance"], rate["risk"]
            ):
                if not _close(float(risk), float(energy) * float(relevance), 2e-5):
                    oracle_errors.append(
                        f"{record['dataset']}:{record['dataset_index']}:{rate['retention_rate']}:product"
                    )
                    break
            key = (
                str(record["dataset"]),
                int(record["dataset_index"]),
                float(rate["retention_rate"]),
            )
            recomputed_rhos[key] = (
                spearman_correlation(rate["energy"], rate["oracle_first_order_abs"]),
                spearman_correlation(rate["risk"], rate["oracle_first_order_abs"]),
            )
    _add(
        findings,
        "oracle_16_samples_per_dataset",
        oracle_counts == Counter({"gqa": 16, "textvqa": 16, "chartqa": 16}),
        f"counts={dict(oracle_counts)}",
    )
    _add(
        findings,
        "oracle_arrays_and_risk_product",
        not oracle_errors,
        f"errors={oracle_errors[:5]}",
    )

    rho_errors = []
    for row in oracle_summary:
        key = (
            row["dataset"],
            int(row["dataset_index"]),
            float(row["retention_rate"]),
        )
        expected = recomputed_rhos.get(key)
        if expected is None or not (
            _close(float(row["energy_oracle_spearman"]), expected[0])
            and _close(float(row["risk_oracle_spearman"]), expected[1])
        ):
            rho_errors.append(str(key))
    _add(
        findings,
        "oracle_correlations_recomputed",
        not rho_errors and len(oracle_summary) == 96,
        f"rows={len(oracle_summary)}, errors={rho_errors[:5]}",
    )

    raw_index = _quality_variant_index(quality)
    gate_input_errors = []
    for rate in RETENTION_RATES:
        tile_sensitive = _raw_quality_comparison(
            raw_index,
            "tile_lowpass",
            "global_lowpass",
            rate,
            {"textvqa", "chartqa"},
        )
        tile_gqa = _raw_quality_comparison(
            raw_index, "tile_lowpass", "global_lowpass", rate, {"gqa"}
        )
        tile_row = next(
            row
            for row in gates["tile_local_better_than_global"]["by_rate"]
            if float(row["retention_rate"]) == rate
        )
        tile_win = bool(
            tile_sensitive["accuracy_delta"] >= 0.015
            or tile_sensitive["prediction_loss_event_reduction"] >= 0.30
            or tile_sensitive["boundary_delta_ci95_lower"] > 0.0
        )
        tile_expected = bool(
            tile_win
            and tile_gqa["accuracy_delta"] >= 0.0
            and tile_sensitive["same_budget"]
            and tile_gqa["same_budget"]
        )
        tile_numeric = {
            "task_sensitive_accuracy_delta": tile_sensitive["accuracy_delta"],
            "prediction_loss_event_reduction": tile_sensitive[
                "prediction_loss_event_reduction"
            ],
            "boundary_mse_reduction": tile_sensitive["boundary_mse_reduction"],
            "boundary_delta_ci95_lower": tile_sensitive[
                "boundary_delta_ci95_lower"
            ],
            "gqa_accuracy_delta": tile_gqa["accuracy_delta"],
        }
        for key, expected in tile_numeric.items():
            if not _close(float(tile_row[key]), float(expected)):
                gate_input_errors.append(f"tile {rate} {key}")
        if tile_row["pass"] != tile_expected:
            gate_input_errors.append(f"tile {rate} pass")

        risk_sensitive = _raw_quality_comparison(
            raw_index,
            "tile_risk_exception",
            "tile_energy_exception",
            rate,
            {"textvqa", "chartqa"},
        )
        risk_gqa = _raw_quality_comparison(
            raw_index,
            "tile_risk_exception",
            "tile_energy_exception",
            rate,
            {"gqa"},
        )
        rate_rhos = [
            values
            for key, values in recomputed_rhos.items()
            if float(key[2]) == rate
        ]
        oracle_delta, oracle_lower, oracle_upper = paired_bootstrap_interval(
            [item[1] for item in rate_rhos], [item[0] for item in rate_rhos]
        )
        channels = int(latency_environment["projection_shape"][1])
        output_channels = int(latency_environment["projection_shape"][0])
        scorer_flops = 256 * (4 * 4 * channels + 2 * channels)
        retained = round(1024 * rate)
        saved_projection_flops = (
            (1024 - retained) * 2 * channels * output_channels
        )
        scorer_ratio = scorer_flops / saved_projection_flops
        risk_row = next(
            row
            for row in gates["risk_exception_better_than_energy"]["by_rate"]
            if float(row["retention_rate"]) == rate
        )
        risk_expected = bool(
            risk_sensitive["accuracy_delta"] >= 0.02
            and risk_gqa["accuracy_delta"] >= 0.0
            and oracle_lower > 0.0
            and risk_sensitive["same_budget"]
            and risk_gqa["same_budget"]
            and scorer_ratio < 0.05
        )
        risk_numeric = {
            "task_sensitive_accuracy_delta": risk_sensitive["accuracy_delta"],
            "gqa_accuracy_delta": risk_gqa["accuracy_delta"],
            "oracle_spearman_delta": oracle_delta,
            "oracle_spearman_delta_ci95_lower": oracle_lower,
            "oracle_spearman_delta_ci95_upper": oracle_upper,
            "risk_scorer_flop_ratio_to_saved_projection": scorer_ratio,
        }
        for key, expected in risk_numeric.items():
            if not _close(float(risk_row[key]), float(expected)):
                gate_input_errors.append(f"risk {rate} {key}")
        if risk_row["pass"] != risk_expected:
            gate_input_errors.append(f"risk {rate} pass")
    _add(
        findings,
        "quality_and_oracle_gate_inputs_recomputed_from_raw",
        not gate_input_errors,
        f"errors={gate_input_errors[:10]}",
    )

    expected_latency = set()
    for batch_size in (1, 4, 8, 16):
        expected_latency.add((batch_size, 1.0, "full_uncompressed", "full_q_proj"))
        for rate in RETENTION_RATES:
            for layout in (
                "arbitrary_token",
                "block_preblocked",
                "fixed_slots_preblocked",
                "block_layout_included",
                "fixed_slots_layout_included",
            ):
                for component in ("gather_layout", "gather_plus_q_proj"):
                    expected_latency.add((batch_size, rate, layout, component))
            for selector in (
                "energy_selector",
                "risk_selector",
                "risk_fixed_selector",
                "risk_token_selector",
            ):
                expected_latency.add((batch_size, rate, selector, "score_plus_topk"))
            if batch_size <= 8:
                for layout in (
                    "arbitrary_token",
                    "block_layout_included",
                    "fixed_slots_layout_included",
                ):
                    expected_latency.add(
                        (
                            batch_size,
                            rate,
                            layout,
                            "compact_prefill_plus_logits",
                        )
                    )
    actual_latency = {
        (
            int(row["batch_size"]),
            float(row["retention_rate"]),
            row["layout"],
            row["component"],
        )
        for row in latency
    }
    invalid_latency = [
        row
        for row in latency
        if float(row["p50_ms"]) <= 0
        or float(row["p95_ms"]) < float(row["p50_ms"])
        or float(row["p99_ms"]) < float(row["p95_ms"])
    ]
    _add(
        findings,
        "latency_matrix_complete",
        expected_latency == actual_latency and not invalid_latency,
        f"missing={list(expected_latency - actual_latency)[:5]}, extra={list(actual_latency - expected_latency)[:5]}, invalid={len(invalid_latency)}",
    )
    expected_budgets = [
        {
            "retention_rate": 0.125,
            "retained_crop_tokens": 128,
            "base_tokens": 96,
            "exception_tokens": 32,
            "exception_blocks": 8,
        },
        {
            "retention_rate": 0.25,
            "retained_crop_tokens": 256,
            "base_tokens": 192,
            "exception_tokens": 64,
            "exception_blocks": 16,
        },
    ]
    latency_contract_ok = bool(
        latency_environment.get("budget_contract") == expected_budgets
        and latency_environment.get("selector_operates_on_exception_budget_only")
        is True
        and latency_environment.get("prefill_sequence_includes_thumbnail_and_text")
        is True
        and latency_environment.get("structured_gate_validated") is False
    )
    _add(
        findings,
        "aligned_latency_budget_and_claim_boundary",
        latency_contract_ok,
        json.dumps(
            {
                "budget_contract": latency_environment.get("budget_contract"),
                "structured_gate_validated": latency_environment.get(
                    "structured_gate_validated"
                ),
            },
            sort_keys=True,
        ),
    )

    latency_index = {
        (
            int(row["batch_size"]),
            float(row["retention_rate"]),
            row["layout"],
            row["component"],
        ): row
        for row in latency
    }
    candidate_paths = {
        "risk_block_dynamic": ("block_layout_included", "risk_selector"),
        "risk_block_fixed_slots": (
            "fixed_slots_layout_included",
            "risk_fixed_selector",
        ),
    }
    structured_latency_errors = []
    for row in structured_latency_summary:
        candidate = row["candidate"]
        batch_size = int(row["batch_size"])
        rate = float(row["retention_rate"])
        layout, selector = candidate_paths[candidate]
        arbitrary_gather = latency_index[
            (batch_size, rate, "arbitrary_token", "gather_layout")
        ]
        structured_gather = latency_index[
            (batch_size, rate, layout, "gather_layout")
        ]
        arbitrary_path = latency_index[
            (batch_size, rate, "arbitrary_token", "gather_plus_q_proj")
        ]
        structured_path = latency_index[
            (batch_size, rate, layout, "gather_plus_q_proj")
        ]
        arbitrary_selector = latency_index[
            (batch_size, rate, "risk_token_selector", "score_plus_topk")
        ]
        structured_selector = latency_index[
            (batch_size, rate, selector, "score_plus_topk")
        ]
        arbitrary_total = float(arbitrary_selector["p50_ms"]) + float(
            arbitrary_path["p50_ms"]
        )
        structured_total = float(structured_selector["p50_ms"]) + float(
            structured_path["p50_ms"]
        )
        expected_values = {
            "gather_p50_reduction": 1.0
            - float(structured_gather["p50_ms"])
            / float(arbitrary_gather["p50_ms"]),
            "total_p50_reduction": 1.0 - structured_total / arbitrary_total,
        }
        if batch_size <= 8:
            arbitrary_prefill = latency_index[
                (
                    batch_size,
                    rate,
                    "arbitrary_token",
                    "compact_prefill_plus_logits",
                )
            ]
            structured_prefill = latency_index[
                (
                    batch_size,
                    rate,
                    layout,
                    "compact_prefill_plus_logits",
                )
            ]
            expected_values["decoder_prefill_p50_reduction"] = 1.0 - float(
                structured_prefill["p50_ms"]
            ) / float(arbitrary_prefill["p50_ms"])
        for key, expected in expected_values.items():
            if not _close(float(row[key]), expected):
                structured_latency_errors.append(
                    f"{candidate}:{batch_size}:{rate}:{key}"
                )
        expected_shape_equal = (
            structured_gather["output_shape"] == arbitrary_gather["output_shape"]
        )
        if (row["output_shape_equal"].lower() == "true") != expected_shape_equal:
            structured_latency_errors.append(
                f"{candidate}:{batch_size}:{rate}:output_shape"
            )
    _add(
        findings,
        "structured_latency_recomputed_from_raw",
        not structured_latency_errors,
        f"errors={structured_latency_errors[:10]}",
    )

    structured_gate_errors = []
    method_for_candidate = {
        "risk_block_dynamic": "tile_risk_exception",
        "risk_block_fixed_slots": "risk_block_fixed_slots",
    }
    for gate_row in gates["structured_block_real_latency_benefit"][
        "by_candidate_rate"
    ]:
        candidate = gate_row["candidate"]
        rate = float(gate_row["retention_rate"])
        task = _raw_quality_comparison(
            raw_index,
            method_for_candidate[candidate],
            "risk_token_unstructured",
            rate,
            {"gqa", "textvqa", "chartqa"},
        )
        if not _close(
            float(gate_row["accuracy_loss_vs_unstructured"]),
            -float(task["accuracy_delta"]),
        ):
            structured_gate_errors.append(f"{candidate}:{rate}:accuracy")
        rows = [
            row
            for row in structured_latency_summary
            if row["candidate"] == candidate
            and float(row["retention_rate"]) == rate
        ]
        gather_mean = mean(float(row["gather_p50_reduction"]) for row in rows)
        total_mean = mean(float(row["total_p50_reduction"]) for row in rows)
        prefill_values = [
            float(row["decoder_prefill_p50_reduction"])
            for row in rows
            if row["decoder_prefill_p50_reduction"]
        ]
        prefill_mean = mean(prefill_values)
        for key, expected in (
            ("mean_gather_p50_reduction", gather_mean),
            ("mean_selector_gather_project_p50_reduction", total_mean),
            ("mean_full_decoder_prefill_p50_reduction", prefill_mean),
        ):
            if not _close(float(gate_row[key]), expected):
                structured_gate_errors.append(f"{candidate}:{rate}:{key}")
        if (
            gate_row["validated"] is not False
            or gate_row["status"] != "INCONCLUSIVE"
            or gate_row["pass"] is not False
        ):
            structured_gate_errors.append(f"{candidate}:{rate}:claim_status")
    _add(
        findings,
        "structured_diagnostic_and_status_recomputed",
        not structured_gate_errors,
        f"errors={structured_gate_errors[:10]}",
    )

    boolean_ok, boolean_errors = _recompute_gate_boole(gates)
    _add(
        findings,
        "gate_boole_recomputed",
        boolean_ok,
        f"errors={boolean_errors}",
    )
    _add(
        findings,
        "quality_latency_claim_boundary",
        environment.get("quality_path_keeps_original_visual_token_length") is True
        and environment.get("quality_path_is_latency_evidence") is False,
        json.dumps(
            {
                "keeps_length": environment.get(
                    "quality_path_keeps_original_visual_token_length"
                ),
                "latency_evidence": environment.get("quality_path_is_latency_evidence"),
            }
        ),
    )

    report_text = (root / "TILESPEC_EX_MINIMAL_REPORT.md").read_text(encoding="utf-8")
    forbidden_positive_claims = [
        phrase
        for phrase in (
            "fused kernel speedup achieved",
            "end-to-end TTFT improved",
            "end-to-end prefill improved",
        )
        if phrase.lower() in report_text.lower()
    ]
    _add(
        findings,
        "no_unsupported_system_claim",
        not forbidden_positive_claims
        and "Structured block real latency benefit: **INCONCLUSIVE**"
        in report_text,
        f"matches={forbidden_positive_claims}, structured_inconclusive={gates['structured_block_real_latency_benefit']['status']}",
    )

    major = [
        item
        for item in findings
        if item["status"] == "FAIL" and item["severity"] == "major"
    ]
    minor = [
        item
        for item in findings
        if item["status"] == "FAIL" and item["severity"] == "minor"
    ]
    overall = "PASS" if not major else "FAIL"
    payload = {
        "overall": overall,
        "major_issues": major,
        "minor_issues": minor,
        "findings": findings,
        "scientific_gate_outcome": {
            "tile_local": gates["tile_local_better_than_global"]["pass"],
            "risk_exception": gates["risk_exception_better_than_energy"]["pass"],
            "structured_latency": gates[
                "structured_block_real_latency_benefit"
            ]["status"],
            "all_three": gates["all_three_pass"],
        },
        "review_note": (
            "Review PASS means evidence/accounting consistency, not that a "
            "scientific gate is positive."
        ),
    }
    (root / "review_findings.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# TileSpec-Ex Independent Review Report",
        "",
        f"## Overall: {overall}",
        "",
        payload["review_note"],
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- **{item['status']}** `{item['check']}`: {item['evidence']}"
        for item in findings
    )
    lines.extend(
        [
            "",
            "## Major Issues",
            "",
            *(f"- {item['check']}: {item['evidence']}" for item in major),
            *( ["- None."] if not major else [] ),
            "",
            "## Minor Issues",
            "",
            *(f"- {item['check']}: {item['evidence']}" for item in minor),
            *( ["- None."] if not minor else [] ),
            "",
            "## Recommended Next Step",
            "",
            (
                "Proceed to fused-kernel and end-to-end benchmark work."
                if gates["all_three_pass"]
                else (
                    "Do not start fused-kernel work. Tile and risk gates failed; "
                    "the structured gate needs a native compact multimodal TTFT "
                    "path before it can be decided."
                )
            ),
            "",
        ]
    )
    (root / "review_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
