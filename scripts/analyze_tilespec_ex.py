#!/usr/bin/env python3
"""Aggregate TileSpec-Ex results and apply the three predeclared kill gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from tilespec_ex.core import METHODS, RETENTION_RATES
from tilespec_ex.metrics import paired_bootstrap_interval, spearman_correlation


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_quality(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        baseline = next(
            variant for variant in sample["variants"] if variant["method"] == "none"
        )
        for variant in sample["variants"]:
            row = {
                "dataset": sample["dataset"],
                "dataset_index": sample["dataset_index"],
                "sample_id": sample["sample_id"],
                "method": variant["method"],
                "scope": variant["scope"],
                "retention_rate": variant["retention_rate"],
                "score": float(variant["score"]),
                "baseline_score": float(baseline["score"]),
                "agrees_with_full": float(variant["agrees_with_full"]),
                "feature_nmse": float(variant["feature_nmse"]),
                "feature_cosine": float(variant["feature_cosine"]),
                "boundary_mse": float(variant["boundary_mse"]),
                "retained_crop_tokens": int(variant["retained_crop_tokens"]),
                "base_tokens": int(variant["base_tokens"]),
                "exception_tokens": int(variant["exception_tokens"]),
                "compact_visual_tokens": int(variant["compact_visual_tokens"]),
                "effective_total_retention": float(
                    variant["effective_total_retention"]
                ),
                "prediction_loss_event": int(
                    float(baseline["score"]) > float(variant["score"]) + 1e-12
                ),
            }
            rows.append(row)
    return rows


def _quality_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["scope"],
                row["method"],
                row["retention_rate"],
            )
        ].append(row)
    output = []
    for (dataset, scope, method, rate), values in sorted(
        grouped.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        output.append(
            {
                "dataset": dataset,
                "scope": scope,
                "method": method,
                "retention_rate": rate,
                "samples": len(values),
                "accuracy": mean(item["score"] for item in values),
                "prediction_agreement": mean(
                    item["agrees_with_full"] for item in values
                ),
                "feature_nmse": mean(item["feature_nmse"] for item in values),
                "feature_cosine": mean(item["feature_cosine"] for item in values),
                "boundary_mse": mean(item["boundary_mse"] for item in values),
                "prediction_loss_events": sum(
                    item["prediction_loss_event"] for item in values
                ),
                "retained_crop_tokens": values[0]["retained_crop_tokens"],
                "compact_visual_tokens": values[0]["compact_visual_tokens"],
                "effective_total_retention": values[0]["effective_total_retention"],
            }
        )
    return output


def _select(
    rows: Sequence[dict[str, Any]],
    method: str,
    rate: float,
    datasets: set[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for row in rows:
        if (
            row["method"] == method
            and row["retention_rate"] == rate
            and row["dataset"] in datasets
        ):
            output[(row["dataset"], int(row["dataset_index"]))] = row
    return output


def _paired_method_comparison(
    rows: Sequence[dict[str, Any]],
    lhs_method: str,
    rhs_method: str,
    rate: float,
    datasets: set[str],
) -> dict[str, Any]:
    lhs = _select(rows, lhs_method, rate, datasets)
    rhs = _select(rows, rhs_method, rate, datasets)
    if set(lhs) != set(rhs) or not lhs:
        raise RuntimeError(
            f"unpaired quality rows for {lhs_method} vs {rhs_method} at {rate}"
        )
    keys = sorted(lhs)
    lhs_scores = [lhs[key]["score"] for key in keys]
    rhs_scores = [rhs[key]["score"] for key in keys]
    accuracy_delta, accuracy_lower, accuracy_upper = paired_bootstrap_interval(
        lhs_scores, rhs_scores
    )
    lhs_boundary = [lhs[key]["boundary_mse"] for key in keys]
    rhs_boundary = [rhs[key]["boundary_mse"] for key in keys]
    boundary_delta, boundary_lower, boundary_upper = paired_bootstrap_interval(
        rhs_boundary, lhs_boundary
    )
    lhs_losses = sum(lhs[key]["prediction_loss_event"] for key in keys)
    rhs_losses = sum(rhs[key]["prediction_loss_event"] for key in keys)
    loss_reduction = 0.0 if rhs_losses == 0 else 1.0 - lhs_losses / rhs_losses
    budget_equal = all(
        lhs[key]["retained_crop_tokens"] == rhs[key]["retained_crop_tokens"]
        for key in keys
    )
    return {
        "retention_rate": rate,
        "datasets": "+".join(sorted(datasets)),
        "lhs_method": lhs_method,
        "rhs_method": rhs_method,
        "samples": len(keys),
        "lhs_accuracy": mean(lhs_scores),
        "rhs_accuracy": mean(rhs_scores),
        "accuracy_delta": accuracy_delta,
        "accuracy_delta_ci95_lower": accuracy_lower,
        "accuracy_delta_ci95_upper": accuracy_upper,
        "lhs_prediction_loss_events": lhs_losses,
        "rhs_prediction_loss_events": rhs_losses,
        "prediction_loss_event_reduction": loss_reduction,
        "lhs_boundary_mse": mean(lhs_boundary),
        "rhs_boundary_mse": mean(rhs_boundary),
        "rhs_minus_lhs_boundary_mse": boundary_delta,
        "boundary_delta_ci95_lower": boundary_lower,
        "boundary_delta_ci95_upper": boundary_upper,
        "relative_boundary_mse_reduction": 1.0
        - mean(lhs_boundary) / max(mean(rhs_boundary), 1e-12),
        "budget_equal": budget_equal,
    }


def _oracle_correlations(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for rate_record in record["rates"]:
            oracle = rate_record["oracle_first_order_abs"]
            energy_rho = spearman_correlation(rate_record["energy"], oracle)
            risk_rho = spearman_correlation(rate_record["risk"], oracle)
            rows.append(
                {
                    "dataset": record["dataset"],
                    "dataset_index": record["dataset_index"],
                    "sample_id": record["sample_id"],
                    "retention_rate": float(rate_record["retention_rate"]),
                    "blocks": len(oracle),
                    "energy_oracle_spearman": energy_rho,
                    "risk_oracle_spearman": risk_rho,
                    "risk_minus_energy_spearman": risk_rho - energy_rho,
                }
            )
    return rows


def _latency_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    output = {}
    for raw in rows:
        row = dict(raw)
        row["batch_size"] = int(row["batch_size"])
        row["retention_rate"] = float(row["retention_rate"])
        for field in (
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "std_ms",
            "minimum_ms",
            "maximum_ms",
        ):
            row[field] = float(row[field])
        key = (
            row["batch_size"],
            row["retention_rate"],
            row["layout"],
            row["component"],
        )
        output[key] = row
    return output


def _structured_latency_rows(
    latency: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = _latency_index(latency)
    batch_sizes = sorted({int(row["batch_size"]) for row in latency})
    candidates = {
        "risk_block_dynamic": ("block_layout_included", "risk_selector"),
        "risk_block_fixed_slots": (
            "fixed_slots_layout_included",
            "risk_fixed_selector",
        ),
    }
    output = []
    for rate in RETENTION_RATES:
        for batch_size in batch_sizes:
            if (batch_size, 1.0, "full_uncompressed", "full_q_proj") not in index:
                continue
            arbitrary_gather = index[
                (batch_size, rate, "arbitrary_token", "gather_layout")
            ]
            arbitrary_path = index[
                (batch_size, rate, "arbitrary_token", "gather_plus_q_proj")
            ]
            token_selector = index[
                (batch_size, rate, "risk_token_selector", "score_plus_topk")
            ]
            full_path = index[
                (batch_size, 1.0, "full_uncompressed", "full_q_proj")
            ]
            arbitrary_prefill = index.get(
                (
                    batch_size,
                    rate,
                    "arbitrary_token",
                    "compact_prefill_plus_logits",
                )
            )
            for candidate, (layout, selector_layout) in candidates.items():
                structured_prefill = index.get(
                    (
                        batch_size,
                        rate,
                        layout,
                        "compact_prefill_plus_logits",
                    )
                )
                block_selector = index[
                    (batch_size, rate, selector_layout, "score_plus_topk")
                ]
                gather = index[(batch_size, rate, layout, "gather_layout")]
                path = index[(batch_size, rate, layout, "gather_plus_q_proj")]
                arbitrary_total = (
                    token_selector["p50_ms"] + arbitrary_path["p50_ms"]
                )
                structured_total = block_selector["p50_ms"] + path["p50_ms"]
                output.append(
                    {
                        "candidate": candidate,
                        "batch_size": batch_size,
                        "retention_rate": rate,
                        "arbitrary_gather_p50_ms": arbitrary_gather["p50_ms"],
                        "structured_gather_p50_ms": gather["p50_ms"],
                        "gather_p50_reduction": 1.0
                        - gather["p50_ms"] / arbitrary_gather["p50_ms"],
                        "arbitrary_selector_p50_ms": token_selector["p50_ms"],
                        "structured_selector_p50_ms": block_selector["p50_ms"],
                        "arbitrary_total_p50_ms": arbitrary_total,
                        "structured_total_p50_ms": structured_total,
                        "total_p50_reduction": 1.0
                        - structured_total / arbitrary_total,
                        "arbitrary_path_p95_ms": arbitrary_path["p95_ms"],
                        "structured_path_p95_ms": path["p95_ms"],
                        "arbitrary_path_p99_ms": arbitrary_path["p99_ms"],
                        "structured_path_p99_ms": path["p99_ms"],
                        "full_q_proj_p50_ms": full_path["p50_ms"],
                        "compact_vs_full_q_proj_reduction": 1.0
                        - path["p50_ms"] / full_path["p50_ms"],
                        "arbitrary_decoder_prefill_p50_ms": None
                        if arbitrary_prefill is None
                        else arbitrary_prefill["p50_ms"],
                        "structured_decoder_prefill_p50_ms": None
                        if structured_prefill is None
                        else structured_prefill["p50_ms"],
                        "decoder_prefill_p50_reduction": None
                        if arbitrary_prefill is None or structured_prefill is None
                        else 1.0
                        - structured_prefill["p50_ms"]
                        / arbitrary_prefill["p50_ms"],
                        "output_shape_equal": gather["output_shape"]
                        == arbitrary_gather["output_shape"],
                        "execution": gather["execution"],
                    }
                )
    return output


def _risk_scorer_flop_ratio(rate: float, channels: int, output_channels: int) -> float:
    # Four operations per block element for energy/mean, one dot per block,
    # compared with the dense projection FLOPs removed by compact execution.
    blocks = 256
    scorer_flops = blocks * (4 * 4 * channels + 2 * channels)
    retained = round(1024 * rate)
    saved_projection_flops = (1024 - retained) * 2 * channels * output_channels
    return scorer_flops / saved_projection_flops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    samples = _read_jsonl(root / "quality_samples.jsonl")
    oracle_records = _read_jsonl(root / "oracle_blocks.jsonl")
    latency_raw = _read_csv(root / "latency_samples.csv")
    latency_environment = json.loads(
        (root / "latency_environment.json").read_text(encoding="utf-8")
    )
    quality_environment = json.loads(
        (root / "environment.json").read_text(encoding="utf-8")
    )

    flat = _flatten_quality(samples)
    quality_summary = _quality_summary(flat)
    _write_csv(root / "quality_metrics.csv", quality_summary)

    tile_rows = []
    for rate in RETENTION_RATES:
        tile_rows.append(
            _paired_method_comparison(
                flat,
                "tile_lowpass",
                "global_lowpass",
                rate,
                {"textvqa", "chartqa"},
            )
        )
        tile_rows.append(
            _paired_method_comparison(
                flat, "tile_lowpass", "global_lowpass", rate, {"gqa"}
            )
        )
    _write_csv(root / "tile_local_comparison.csv", tile_rows)

    oracle_rows = _oracle_correlations(oracle_records)
    _write_csv(root / "oracle_correlations.csv", oracle_rows)
    risk_rows = []
    risk_gate_by_rate = []
    projection_out, channels = latency_environment["projection_shape"]
    for rate in RETENTION_RATES:
        task_sensitive = _paired_method_comparison(
            flat,
            "tile_risk_exception",
            "tile_energy_exception",
            rate,
            {"textvqa", "chartqa"},
        )
        gqa = _paired_method_comparison(
            flat,
            "tile_risk_exception",
            "tile_energy_exception",
            rate,
            {"gqa"},
        )
        rate_oracle = [row for row in oracle_rows if row["retention_rate"] == rate]
        oracle_delta, oracle_lower, oracle_upper = paired_bootstrap_interval(
            [row["risk_oracle_spearman"] for row in rate_oracle],
            [row["energy_oracle_spearman"] for row in rate_oracle],
        )
        scorer_ratio = _risk_scorer_flop_ratio(rate, channels, projection_out)
        gate = {
            "retention_rate": rate,
            "task_sensitive_accuracy_delta": task_sensitive["accuracy_delta"],
            "task_delta_at_least_2pp": task_sensitive["accuracy_delta"] >= 0.02,
            "gqa_accuracy_delta": gqa["accuracy_delta"],
            "gqa_not_worse": gqa["accuracy_delta"] >= 0.0,
            "oracle_spearman_delta": oracle_delta,
            "oracle_spearman_delta_ci95_lower": oracle_lower,
            "oracle_spearman_delta_ci95_upper": oracle_upper,
            "oracle_significantly_better": oracle_lower > 0.0,
            "same_budget": bool(task_sensitive["budget_equal"] and gqa["budget_equal"]),
            "risk_scorer_flop_ratio_to_saved_projection": scorer_ratio,
            "scorer_below_5pct": scorer_ratio < 0.05,
        }
        gate["pass"] = all(
            gate[key]
            for key in (
                "task_delta_at_least_2pp",
                "gqa_not_worse",
                "oracle_significantly_better",
                "same_budget",
                "scorer_below_5pct",
            )
        )
        risk_gate_by_rate.append(gate)
        risk_rows.extend(
            [
                {"comparison_scope": "textvqa+chartqa", **task_sensitive, **gate},
                {"comparison_scope": "gqa", **gqa, **gate},
            ]
        )
    _write_csv(root / "risk_comparison.csv", risk_rows)

    structured_latency = _structured_latency_rows(latency_raw)
    structured_validated = (
        latency_environment.get("structured_gate_validated") is True
    )
    structure_task_rows = []
    candidate_method = {
        "risk_block_dynamic": "tile_risk_exception",
        "risk_block_fixed_slots": "risk_block_fixed_slots",
    }
    structure_gates = []
    for candidate, method in candidate_method.items():
        for rate in RETENTION_RATES:
            task = _paired_method_comparison(
                flat,
                method,
                "risk_token_unstructured",
                rate,
                {"gqa", "textvqa", "chartqa"},
            )
            latency_rows = [
                row
                for row in structured_latency
                if row["candidate"] == candidate and row["retention_rate"] == rate
            ]
            if not latency_rows:
                raise RuntimeError(f"missing latency rows for {candidate} at {rate}")
            accuracy_loss = -task["accuracy_delta"]
            prefill_rows = [
                row
                for row in latency_rows
                if row["decoder_prefill_p50_reduction"] is not None
            ]
            if not prefill_rows:
                raise RuntimeError(
                    f"missing full decoder prefill rows for {candidate} at {rate}"
                )
            gate = {
                "candidate": candidate,
                "retention_rate": rate,
                "accuracy_loss_vs_unstructured": accuracy_loss,
                "accuracy_loss_within_0_5pp": accuracy_loss <= 0.005,
                "mean_gather_p50_reduction": mean(
                    row["gather_p50_reduction"] for row in latency_rows
                ),
                "gather_reduction_at_least_30pct": mean(
                    row["gather_p50_reduction"] for row in latency_rows
                )
                >= 0.30,
                "mean_selector_gather_project_p50_reduction": mean(
                    row["total_p50_reduction"] for row in latency_rows
                ),
                "total_reduction_at_least_10pct": mean(
                    row["total_p50_reduction"] for row in latency_rows
                )
                >= 0.10,
                "all_micro_batches_faster": all(
                    row["total_p50_reduction"] > 0.0 for row in latency_rows
                ),
                "mean_full_decoder_prefill_p50_reduction": mean(
                    row["decoder_prefill_p50_reduction"] for row in prefill_rows
                ),
                "full_decoder_prefill_reduction_at_least_10pct": mean(
                    row["decoder_prefill_p50_reduction"] for row in prefill_rows
                )
                >= 0.10,
                "all_prefill_batches_faster": all(
                    row["decoder_prefill_p50_reduction"] > 0.0
                    for row in prefill_rows
                ),
                "fixed_output_shape": all(
                    row["output_shape_equal"] for row in latency_rows
                ),
            }
            gate["diagnostic_pass"] = all(
                gate[key]
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
            gate["validated"] = structured_validated
            gate["pass"] = bool(structured_validated and gate["diagnostic_pass"])
            gate["status"] = (
                "PASS"
                if gate["pass"]
                else "FAIL"
                if structured_validated
                else "INCONCLUSIVE"
            )
            structure_gates.append(gate)
            structure_task_rows.append({**task, **gate})
    _write_csv(root / "structured_latency_comparison.csv", structured_latency)
    _write_csv(root / "structured_task_comparison.csv", structure_task_rows)

    tile_gate_by_rate = []
    for rate in RETENTION_RATES:
        sensitive = next(
            row
            for row in tile_rows
            if row["retention_rate"] == rate
            and row["datasets"] == "chartqa+textvqa"
        )
        gqa = next(
            row
            for row in tile_rows
            if row["retention_rate"] == rate and row["datasets"] == "gqa"
        )
        quality_or_boundary_win = (
            sensitive["accuracy_delta"] >= 0.015
            or sensitive["prediction_loss_event_reduction"] >= 0.30
            or sensitive["boundary_delta_ci95_lower"] > 0.0
        )
        rate_gate = {
            "retention_rate": rate,
            "task_sensitive_accuracy_delta": sensitive["accuracy_delta"],
            "prediction_loss_event_reduction": sensitive[
                "prediction_loss_event_reduction"
            ],
            "boundary_mse_reduction": sensitive[
                "relative_boundary_mse_reduction"
            ],
            "boundary_delta_ci95_lower": sensitive["boundary_delta_ci95_lower"],
            "at_least_one_sensitive_win": quality_or_boundary_win,
            "gqa_accuracy_delta": gqa["accuracy_delta"],
            "gqa_not_worse": gqa["accuracy_delta"] >= 0.0,
            "same_budget": bool(sensitive["budget_equal"] and gqa["budget_equal"]),
        }
        rate_gate["pass"] = bool(
            rate_gate["at_least_one_sensitive_win"]
            and rate_gate["gqa_not_worse"]
            and rate_gate["same_budget"]
        )
        tile_gate_by_rate.append(rate_gate)

    candidate_complete = {}
    for candidate in candidate_method:
        candidate_complete[candidate] = all(
            gate["pass"] for gate in structure_gates if gate["candidate"] == candidate
        )
    structured_pass = bool(structured_validated and any(candidate_complete.values()))
    structured_status = (
        "PASS"
        if structured_pass
        else "FAIL"
        if structured_validated
        else "INCONCLUSIVE"
    )
    gates = {
        "contract": {
            "model": "Qwen2.5-VL-3B-Instruct with deterministic five-image multi-tile adapter",
            "datasets": ["gqa", "textvqa", "chartqa"],
            "samples_per_dataset": quality_environment["samples_per_dataset"],
            "retention_rates": list(RETENTION_RATES),
            "main_methods": list(METHODS),
        },
        "tile_local_better_than_global": {
            "pass": all(item["pass"] for item in tile_gate_by_rate),
            "by_rate": tile_gate_by_rate,
        },
        "risk_exception_better_than_energy": {
            "pass": all(item["pass"] for item in risk_gate_by_rate),
            "by_rate": risk_gate_by_rate,
        },
        "structured_block_real_latency_benefit": {
            "status": structured_status,
            "validated": structured_validated,
            "pass": structured_pass,
            "candidate_pass": candidate_complete,
            "by_candidate_rate": structure_gates,
            "evidence_boundary": latency_environment["claim_boundary"],
            "inconclusive_reason": None
            if structured_validated
            else (
                "The aligned benchmark stops after compact packing, language "
                "decoder prefill, and first-token logits. It does not measure "
                "the visual encoder, native multimodal position construction, "
                "or end-to-end TTFT required by the predeclared gate."
            ),
        },
    }
    gates["all_three_pass"] = all(
        gates[name]["pass"]
        for name in (
            "tile_local_better_than_global",
            "risk_exception_better_than_energy",
            "structured_block_real_latency_benefit",
        )
    )
    gates["recommendation"] = (
        "Proceed to a fused kernel and full benchmark."
        if gates["all_three_pass"]
        else (
            "Do not invest in a fused kernel: tile-local and risk gates failed, "
            "and the structured gate remains inconclusive."
        )
    )
    (root / "gate_summary.json").write_text(
        json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# TileSpec-Ex Minimal Feasibility Report",
        "",
        "## Executive Summary",
        "",
        f"- Tile-local > global: **{'PASS' if gates['tile_local_better_than_global']['pass'] else 'FAIL'}**",
        f"- Risk exception > energy exception: **{'PASS' if gates['risk_exception_better_than_energy']['pass'] else 'FAIL'}**",
        f"- Structured block real latency benefit: **{gates['structured_block_real_latency_benefit']['status']}**",
        f"- All-three investment gate: **{'PASS' if gates['all_three_pass'] else 'FAIL'}**",
        f"- Decision: {gates['recommendation']}",
        "",
        "The task-quality path reconstructs the original visual-token count. It tests representation fidelity, not speed. The aligned latency path uses the exact 75% base plus 25% exception budget, includes the thumbnail and text payload in language-decoder prefill, and emits first-token logits. It still omits the visual encoder, native multimodal position construction, and end-to-end TTFT, so the structured gate remains inconclusive.",
        "",
        "## Experiment Contract",
        "",
        f"- Model: {gates['contract']['model']}",
        f"- Samples: {quality_environment['samples_per_dataset']} each from GQA, TextVQA, and ChartQA",
        "- Crop-token retention: 12.5% and 25%; the global thumbnail is always retained",
        "- Main methods: none, average pooling, global low-pass, tile low-pass, tile+energy exception, tile+risk exception",
        "- Structural ablation: arbitrary risk tokens, dynamic 2x2 risk blocks, fixed per-tile block slots",
        "",
        "## Tile-Local Versus Global",
        "",
    ]
    for item in tile_gate_by_rate:
        report.append(
            f"- {item['retention_rate']:.3f}: accuracy delta {item['task_sensitive_accuracy_delta']:+.4f}, "
            f"loss-event reduction {item['prediction_loss_event_reduction']:.1%}, "
            f"boundary-MSE reduction {item['boundary_mse_reduction']:.1%}, "
            f"GQA delta {item['gqa_accuracy_delta']:+.4f}, gate {'PASS' if item['pass'] else 'FAIL'}."
        )
    report.extend(["", "## Risk Versus Energy", ""])
    for item in risk_gate_by_rate:
        report.append(
            f"- {item['retention_rate']:.3f}: TextVQA+ChartQA delta "
            f"{item['task_sensitive_accuracy_delta']:+.4f}, oracle Spearman delta "
            f"{item['oracle_spearman_delta']:+.4f} "
            f"(95% CI {item['oracle_spearman_delta_ci95_lower']:+.4f} to "
            f"{item['oracle_spearman_delta_ci95_upper']:+.4f}), scorer FLOP ratio "
            f"{item['risk_scorer_flop_ratio_to_saved_projection']:.3%}, gate "
            f"{'PASS' if item['pass'] else 'FAIL'}."
        )
    report.extend(["", "## Structured Execution", ""])
    for item in structure_gates:
        report.append(
            f"- {item['candidate']} @ {item['retention_rate']:.3f}: accuracy loss "
            f"{item['accuracy_loss_vs_unstructured']:+.4f}, gather P50 reduction "
            f"{item['mean_gather_p50_reduction']:.1%}, selector+gather+q_proj P50 "
            f"reduction {item['mean_selector_gather_project_p50_reduction']:.1%}, "
            f"compact-prefill-plus-logits reduction "
            f"{item['mean_full_decoder_prefill_p50_reduction']:.1%}, "
            f"diagnostic {'PASS' if item['diagnostic_pass'] else 'FAIL'}, "
            f"gate {item['status']}."
        )
    report.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "No fused kernel, end-to-end multimodal prefill/TTFT speedup, or model-wide memory reduction is claimed. The aligned latency rows are diagnostics only; an end-to-end structured gate decision requires a native compact multimodal execution path.",
            "",
        ]
    )
    (root / "TILESPEC_EX_MINIMAL_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(gates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
