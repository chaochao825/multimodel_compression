#!/usr/bin/env python3
"""Independent machine audit for the formal TileLogic-RVQ experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from tilespec_ex.cache import load_cache_manifest, manifest_sha256
from tilespec_ex.metrics import dataset_score, normalize_answer
from tilespec_ex.rate import RATE_STORAGE_POLICY
from tilespec_ex.routing import LogicRouter, logic_router_storage_bits
from tilespec_ex.tilelogic_analysis import DATASETS, RATES
from tilespec_ex.tilelogic_methods import ABLATION_METHODS, MAIN_TILELOGIC_METHODS


AUDIT_FORMAT = "tilelogic_rvq_audit_v1"
EXPECTED_VARIANTS = 1 + len(RATES) * (
    len(MAIN_TILELOGIC_METHODS) + len(ABLATION_METHODS)
)
EXPECTED_METHOD_RATES = {
    (method, rate)
    for method in (*MAIN_TILELOGIC_METHODS, *ABLATION_METHODS)
    for rate in RATES
}


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
    return output


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _rate(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _add(
    findings: list[dict[str, Any]],
    check: str,
    passed: bool,
    evidence: Any,
    *,
    severity: str = "major",
) -> None:
    findings.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "severity": "none" if passed else severity,
            "evidence": evidence,
        }
    )


def _key(sample: Mapping[str, Any]) -> tuple[str, int]:
    return str(sample["dataset"]), int(sample["dataset_index"])


def _variant_key(variant: Mapping[str, Any]) -> tuple[str, float | None]:
    return str(variant["method"]), _rate(variant["retention_rate"])


def _tensor_bits(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size() * 8)


def _codebook_state_bits(state: Mapping[str, Any]) -> int:
    return sum(
        _tensor_bits(state[name])
        for name in ("codewords", "scale_levels", "metric_weights")
    )


def _component_bits(variant: Mapping[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for component in variant["rate_components"]:
        name = str(component["name"])
        output[name] = output.get(name, 0) + int(component["bits"])
    return output


def _logic_tree_values_are_fp32_logical_payload(state: Mapping[str, Any]) -> bool:
    state_format = state.get("format")
    if state_format == "tilespec_logic_regression_tree_v2":
        values = state.get("values")
        return (
            isinstance(values, torch.Tensor)
            and values.dtype == torch.float32
            and values.shape == (len(state.get("nodes", [])),)
        )
    if state_format != "tilespec_logic_regression_tree_v1":
        return False
    for node in state.get("nodes", []):
        value = float(node["value"])
        encoded = torch.tensor(value, dtype=torch.float32)
        if not bool(torch.isfinite(encoded)) or float(encoded.item()) != value:
            return False
    return True


def _check_guardrail_contract(
    errors: list[str],
    guardrail: Mapping[str, Any],
    *,
    candidate: str,
    baseline: str,
    context: str,
) -> None:
    if guardrail.get("candidate") != candidate:
        errors.append(
            f"{context} guardrail candidate is {guardrail.get('candidate')}, "
            f"expected {candidate}"
        )
    if guardrail.get("baseline") != baseline:
        errors.append(
            f"{context} guardrail baseline is {guardrail.get('baseline')}, "
            f"expected {baseline}"
        )


def _decision_boolean_audit(decisions: dict[str, Any]) -> list[str]:
    errors = []
    indexed = {int(item["id"]): item for item in decisions["questions"]}
    if set(indexed) != set(range(1, 7)):
        return [f"decision IDs are {sorted(indexed)}"]

    q1 = indexed[1]
    q1_evidence = q1["evidence"]
    q1_expected = bool(
        any(q1_evidence["feature_frontier_extension_by_rate"].values())
        and any(q1_evidence["nll_frontier_extension_by_rate"].values())
        and all(item["pass"] for item in q1_evidence["guardrails"])
    )
    if q1_expected != q1["pass"]:
        errors.append("question 1 boolean mismatch")
    for item in q1_evidence["guardrails"]:
        _check_guardrail_contract(
            errors,
            item,
            candidate="base_vq",
            baseline="base_scalar_quant",
            context="question 1",
        )

    q2 = indexed[2]
    q2_expected = all(item["pass"] for item in q2["evidence"]["by_rate"])
    if q2_expected != q2["pass"]:
        errors.append("question 2 boolean mismatch")
    for item in q2["evidence"]["by_rate"]:
        _check_guardrail_contract(
            errors,
            item["guardrail_vs_base"],
            candidate="base_vq_residual_rvq",
            baseline="base_vq",
            context="question 2 versus base",
        )
        _check_guardrail_contract(
            errors,
            item["guardrail_vs_unweighted"],
            candidate="base_vq_residual_rvq",
            baseline="base_vq_residual_rvq_unweighted",
            context="question 2 versus unweighted",
        )

    q3 = indexed[3]
    q3_rows = q3["evidence"]["by_rate"]
    q3_available = len(q3_rows) == len(RATES)
    q3_expected = q3_available and all(
        item["spearman_delta"] >= 0.02
        and item["topk_recall_delta"] >= 0.02
        and item["guardrail"]["pass"]
        for item in q3_rows
    )
    if q3_expected != q3["pass"]:
        errors.append("question 3 boolean mismatch")
    for item in q3_rows:
        _check_guardrail_contract(
            errors,
            item["guardrail"],
            candidate="base_vq_mlp_router",
            baseline="base_vq_residual_rvq",
            context="question 3",
        )

    q4 = indexed[4]
    q4_rows = q4["evidence"]["by_rate"]
    if not q3["pass"]:
        if q4["status"] != "INCONCLUSIVE" or q4["pass"]:
            errors.append("question 4 must be INCONCLUSIVE when question 3 fails")
    else:
        q4_expected = len(q4_rows) == len(RATES) and all(
            item["spearman_improvement_retained"] >= 0.9
            and item["topk_improvement_retained"] >= 0.9
            and item["guardrail"]["pass"]
            for item in q4_rows
        )
        if q4_expected != q4["pass"]:
            errors.append("question 4 boolean mismatch")
        for item in q4_rows:
            _check_guardrail_contract(
                errors,
                item["guardrail"],
                candidate="base_vq_logic_router",
                baseline="base_vq_mlp_router",
                context="question 4",
            )

    q5 = indexed[5]
    q5_rows = q5["evidence"]["by_rate"]
    q5_available = len(q5_rows) == len(RATES)
    q5_expected = q5_available and all(
        item["all_paired_medians_lower"] and item["guardrail"]["pass"]
        for item in q5_rows
    )
    if q5_expected != q5["pass"]:
        errors.append("question 5 boolean mismatch")
    for item in q5_rows:
        _check_guardrail_contract(
            errors,
            item["guardrail"],
            candidate="logic_router_fixed_slots",
            baseline="base_vq_logic_router",
            context="question 5",
        )

    q6 = indexed[6]
    q6_rows = q6["evidence"]["by_rate"]
    q6_expected = any(
        item["feature_frontier_extension"]
        and item["nll_frontier_extension"]
        and item["material_improvement"]
        and item["guardrail"]["pass"]
        for item in q6_rows
    )
    if q6_expected != q6["pass"]:
        errors.append("question 6 boolean mismatch")
    for item in q6_rows:
        _check_guardrail_contract(
            errors,
            item["guardrail"],
            candidate="logic_router_fixed_slots_exact_fallback",
            baseline="logic_router_fixed_slots",
            context="question 6",
        )

    for question in decisions["questions"]:
        expected_status = "PASS" if question["pass"] else "FAIL"
        if question["id"] == 4 and not q3["pass"]:
            expected_status = "INCONCLUSIVE"
        if question["status"] != expected_status:
            errors.append(f"question {question['id']} status mismatch")
    expected_all = all(item["status"] == "PASS" for item in decisions["questions"])
    if decisions["all_questions_pass"] != expected_all:
        errors.append("all_questions_pass mismatch")
    if decisions["aggregate_positive_claim_allowed"] != expected_all:
        errors.append("aggregate_positive_claim_allowed mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--feature-eval-dir", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--latency-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    training_dir = args.training_dir.resolve()
    feature_dir = args.feature_eval_dir.resolve()
    quality_dir = args.quality_dir.resolve()
    latency_dir = args.latency_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    findings: list[dict[str, Any]] = []

    required = [
        cache_dir.parent / "cache_provenance_backfill.json",
        cache_dir.parent / "rate_precision_correction_validation.json",
        feature_dir / "feature_eval_summary.json",
        feature_dir / "feature_samples.jsonl",
        quality_dir / "quality_environment.json",
        quality_dir / "quality_summary.json",
        quality_dir / "quality_samples.jsonl",
        latency_dir / "latency_environment.json",
        latency_dir / "latency_samples.csv",
        training_dir / "training_summary.json",
        analysis_dir / "feature_metrics.csv",
        analysis_dir / "quality_metrics.csv",
        analysis_dir / "rate_components.csv",
        analysis_dir / "router_sample_metrics.csv",
        analysis_dir / "router_metrics.csv",
        analysis_dir / "latency_metrics.csv",
        analysis_dir / "latency_paired_dynamic_fixed.csv",
        analysis_dir / "method_points.csv",
        analysis_dir / "decision_summary.json",
        analysis_dir / "decision_summary.csv",
        analysis_dir / "analysis_environment.json",
        analysis_dir / "TILELOGIC_RVQ_FINAL_REPORT.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    _add(findings, "required_outputs", not missing, {"missing": missing})
    if missing:
        raise SystemExit("required outputs are missing")

    cache_entries = load_cache_manifest(cache_dir, verify_files=True)
    calibration_keys = {entry.key for entry in cache_entries if entry.split == "calibration"}
    evaluation_keys = {entry.key for entry in cache_entries if entry.split == "evaluation"}
    split_counts = Counter((entry.dataset, entry.split) for entry in cache_entries)
    expected_split_counts = Counter(
        {
            (dataset, "calibration"): 80
            for dataset in DATASETS
        }
        | {(dataset, "evaluation"): 120 for dataset in DATASETS}
    )
    _add(
        findings,
        "cache_split_contract",
        split_counts == expected_split_counts
        and not calibration_keys & evaluation_keys
        and len(cache_entries) == 600,
        {"counts": {str(key): value for key, value in split_counts.items()}},
    )
    cache_environment = _json(cache_dir / "environment.json")
    expected_source_hash = str(cache_environment.get("manifest_sha256", ""))
    expected_model_revision = str(
        cache_environment.get("model_revision")
        or Path(str(cache_environment.get("model_dir", ""))).name
    )
    provenance_errors = []
    for entry in cache_entries:
        expected_dtype_fields = {"thumbnail", "crops", "query"}
        if entry.oracle:
            expected_dtype_fields.add("crop_gradient")
        if (
            entry.source_manifest_sha256 != expected_source_hash
            or entry.model_revision != expected_model_revision
            or set(entry.tensor_dtypes) != expected_dtype_fields
            or set(entry.tensor_dtypes.values()) != {"float16"}
        ):
            provenance_errors.append(str(entry.key))
    _add(
        findings,
        "per_entry_cache_source_model_and_dtype_provenance",
        len(cache_entries) == 600 and not provenance_errors,
        {
            "records": len(cache_entries),
            "source_manifest_sha256": expected_source_hash,
            "model_revision": expected_model_revision,
            "errors": provenance_errors[:10],
        },
    )
    provenance_backfill = _json(cache_dir.parent / "cache_provenance_backfill.json")
    _add(
        findings,
        "cache_provenance_migration_is_hash_linked_and_payload_preserving",
        provenance_backfill.get("records") == 600
        and provenance_backfill.get("new_manifest_sha256")
        == manifest_sha256(cache_dir)
        and provenance_backfill.get("source_manifest_sha256")
        == expected_source_hash
        and provenance_backfill.get("model_revision") == expected_model_revision
        and provenance_backfill.get("payload_tensors_unchanged") is True,
        provenance_backfill,
    )

    training = _json(training_dir / "training_summary.json")
    training_hash_ok = training.get("cache_manifest_sha256") == manifest_sha256(cache_dir)
    artifact_errors = []
    for artifact in training.get("artifacts", {}).values():
        path = training_dir / artifact["file"]
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["bytes"])
            or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
        ):
            artifact_errors.append(str(path))
    _add(
        findings,
        "calibration_only_training_and_artifact_hashes",
        bool(training.get("complete"))
        and training.get("training_entries") == 240
        and training.get("evaluation_entries_loaded") == 0
        and training_hash_ok
        and not artifact_errors,
        {
            "training_entries": training.get("training_entries"),
            "evaluation_entries_loaded": training.get("evaluation_entries_loaded"),
            "artifact_errors": artifact_errors,
        },
    )

    base_state = torch.load(
        training_dir / "base_codebook.pt", map_location="cpu", weights_only=True
    )
    residual_fisher_state = torch.load(
        training_dir / "residual_rvq_fisher.pt",
        map_location="cpu",
        weights_only=True,
    )
    residual_unweighted_state = torch.load(
        training_dir / "residual_rvq_unweighted.pt",
        map_location="cpu",
        weights_only=True,
    )
    router_bundle_errors = []
    router_states: dict[float, dict[str, Any]] = {}
    for rate in RATES:
        suffix = f"rate_{rate:.3f}".replace(".", "p")
        state = torch.load(
            training_dir / f"router_{suffix}.pt", map_location="cpu", weights_only=True
        )
        router_states[rate] = state
        train_keys = {tuple(item) for item in state["router_train_keys"]}
        validation_keys = {tuple(item) for item in state["router_validation_keys"]}
        fixed_mask = state["fixed_slot_mask"].bool()
        expected_slots = round(1024 * rate * 0.25 / 4)
        if (
            not train_keys <= calibration_keys
            or not validation_keys <= calibration_keys
            or train_keys & validation_keys
            or (train_keys | validation_keys) & evaluation_keys
            or int(fixed_mask.sum()) != expected_slots
            or fixed_mask.numel() != 256
        ):
            router_bundle_errors.append(
                {
                    "rate": rate,
                    "train": len(train_keys),
                    "validation": len(validation_keys),
                    "fixed_slots": int(fixed_mask.sum()),
                }
            )
    _add(
        findings,
        "router_training_and_fixed_slots_are_calibration_only",
        not router_bundle_errors,
        {"errors": router_bundle_errors},
    )

    feature_summary = _json(feature_dir / "feature_eval_summary.json")
    quality_summary = _json(quality_dir / "quality_summary.json")
    quality_environment = _json(quality_dir / "quality_environment.json")
    training_summary_sha256 = hashlib.sha256(
        (training_dir / "training_summary.json").read_bytes()
    ).hexdigest()
    feature_summary_sha256 = hashlib.sha256(
        (feature_dir / "feature_eval_summary.json").read_bytes()
    ).hexdigest()
    _add(
        findings,
        "cache_training_feature_quality_hash_chain",
        feature_summary.get("cache_manifest_sha256") == manifest_sha256(cache_dir)
        and feature_summary.get("training_summary_sha256")
        == training_summary_sha256
        and quality_environment.get("cache_manifest_sha256")
        == manifest_sha256(cache_dir)
        and quality_environment.get("feature_eval_summary_sha256")
        == feature_summary_sha256,
        {
            "cache_manifest_sha256": manifest_sha256(cache_dir),
            "training_summary_sha256": training_summary_sha256,
            "feature_summary_sha256": feature_summary_sha256,
        },
    )
    feature = _jsonl(feature_dir / "feature_samples.jsonl")
    quality = _jsonl(quality_dir / "quality_samples.jsonl")
    correction_validation = _json(
        cache_dir.parent / "rate_precision_correction_validation.json"
    )
    _add(
        findings,
        "rate_correction_preserves_all_non_rate_feature_semantics",
        correction_validation.get("records") == 360
        and correction_validation.get("compared_variants") == 360 * EXPECTED_VARIANTS
        and correction_validation.get("non_rate_semantics_identical") is True
        and correction_validation.get("new_feature_samples_sha256")
        == hashlib.sha256((feature_dir / "feature_samples.jsonl").read_bytes()).hexdigest()
        and not correction_validation.get("errors"),
        correction_validation,
    )
    feature_keys = {_key(sample) for sample in feature}
    quality_keys = {_key(sample) for sample in quality}
    sample_counts = Counter(str(sample["dataset"]) for sample in feature)
    _add(
        findings,
        "complete_paired_360_evaluation_samples",
        feature_summary.get("complete")
        and quality_summary.get("complete")
        and feature_summary.get("records") == 360
        and quality_summary.get("records") == 360
        and len(feature) == len(feature_keys) == 360
        and len(quality) == len(quality_keys) == 360
        and feature_keys == quality_keys == evaluation_keys
        and sample_counts == Counter({dataset: 120 for dataset in DATASETS})
        and feature_summary.get("calibration_records_loaded") == 0,
        {
            "feature_records": len(feature),
            "quality_records": len(quality),
            "counts": dict(sample_counts),
            "calibration_records_loaded": feature_summary.get(
                "calibration_records_loaded"
            ),
        },
    )
    _add(
        findings,
        "quality_path_claim_boundary",
        quality_environment.get("quality_path_keeps_original_visual_token_length")
        is True
        and quality_environment.get("quality_path_is_latency_evidence") is False
        and quality_environment.get("teacher_forced_target_policy")
        == "first_manifest_answer_verbatim"
        and quality_environment.get("answer_score_uses_all_manifest_answers") is True,
        quality_environment,
    )

    quality_by_key = {_key(sample): sample for sample in quality}
    matrix_errors = []
    rate_errors = []
    precision_errors = []
    exact_payload_errors = []
    metric_errors = []
    score_errors = []
    expected_base_bits = _codebook_state_bits(base_state)
    expected_residual_bits = {
        "base_vq_residual_rvq_unweighted": _codebook_state_bits(
            residual_unweighted_state
        ),
        "fisher": _codebook_state_bits(residual_fisher_state),
    }
    expected_router_bits: dict[float, dict[str, int]] = {}
    for rate, state in router_states.items():
        logic = LogicRouter.from_state_dict(state["logic"])
        expected_router_bits[rate] = {
            "mlp_router_parameters": sum(
                _tensor_bits(value) for value in state["mlp"]["state_dict"].values()
            ),
            "mlp_router_normalizer": sum(
                _tensor_bits(state["normalizer"][name]) for name in ("mean", "scale")
            ),
            "logic_router": logic_router_storage_bits(logic),
            "router_curvature_prior": _tensor_bits(state["curvature_prior"]),
        }
    artifact_precision_ok = (
        base_state["codewords"].dtype == torch.float16
        and base_state["scale_levels"].dtype == torch.float16
        and base_state["metric_weights"].dtype == torch.float32
        and residual_fisher_state["codewords"].dtype == torch.float16
        and residual_fisher_state["scale_levels"].dtype == torch.float16
        and residual_fisher_state["metric_weights"].dtype == torch.float32
        and residual_unweighted_state["codewords"].dtype == torch.float16
        and residual_unweighted_state["scale_levels"].dtype == torch.float16
        and residual_unweighted_state["metric_weights"].dtype == torch.float32
    )
    logic_tree_formats: set[str] = set()
    for state in router_states.values():
        logic_tree_states = state["logic"]["trees"]
        logic_tree_formats.update(str(tree.get("format")) for tree in logic_tree_states)
        artifact_precision_ok = artifact_precision_ok and (
            all(
                value.dtype == torch.float32
                for value in state["mlp"]["state_dict"].values()
            )
            and state["normalizer"]["mean"].dtype == torch.float32
            and state["normalizer"]["scale"].dtype == torch.float32
            and state["logic"]["binarizer"]["thresholds"].dtype
            == torch.float16
            and state["curvature_prior"].dtype == torch.float32
            and all(
                _logic_tree_values_are_fp32_logical_payload(tree)
                for tree in logic_tree_states
            )
        )
    if not artifact_precision_ok:
        precision_errors.append("codebook artifact dtypes differ from the rate policy")
    if feature_summary.get("rate_storage_policy") != RATE_STORAGE_POLICY:
        precision_errors.append("feature summary rate storage policy mismatch")
    for sample in feature:
        key = _key(sample)
        quality_sample = quality_by_key[key]
        feature_variants = sample["variants"]
        quality_variants = quality_sample["variants"]
        if len(feature_variants) != EXPECTED_VARIANTS or len(quality_variants) != EXPECTED_VARIANTS:
            matrix_errors.append(f"{key}: variant count")
            continue
        feature_index = {_variant_key(item): item for item in feature_variants}
        quality_index = {_variant_key(item): item for item in quality_variants}
        expected = {("none", None), *EXPECTED_METHOD_RATES}
        if set(feature_index) != expected or set(quality_index) != expected:
            matrix_errors.append(f"{key}: method/rate matrix")
            continue
        baseline_prediction = normalize_answer(quality_index[("none", None)]["prediction"])
        for variant_key, variant in feature_index.items():
            metrics = variant["rate"]
            components = variant["rate_components"]
            component_index = _component_bits(variant)
            method, retention_rate = variant_key
            stream = sum(
                int(component["bits"])
                for component in components
                if component["scope"] == "stream"
            )
            shared = sum(
                int(component["bits"])
                for component in components
                if component["scope"] == "shared"
            )
            effective = stream + shared / 360
            original_values = float(metrics["raw_fp_bits"]) / 16
            if (
                stream != int(metrics["stream_bits"])
                or shared != int(metrics["shared_bits"])
                or not _close(effective, float(metrics["effective_bits"]))
                or not _close(
                    effective / original_values,
                    float(metrics["effective_bits_per_original_value"]),
                )
            ):
                rate_errors.append(f"{key}:{variant_key}")
            if method == "base_scalar_quant":
                expected_scale_bits = (
                    int(variant["base_tokens"])
                    * RATE_STORAGE_POLICY["base_scalar_scale_bits"]
                )
                if component_index.get("base_scalar_scales") != expected_scale_bits:
                    precision_errors.append(f"{key}:{variant_key}:scalar_scale")
            vq_methods = {
                "base_vq",
                "base_vq_residual_rvq",
                "base_vq_residual_rvq_unweighted",
                "base_vq_mlp_router",
                "base_vq_logic_router",
                "logic_router_fixed_slots",
                "logic_router_fixed_slots_exact_fallback",
            }
            if method in vq_methods and component_index.get("base_codebook") != expected_base_bits:
                precision_errors.append(f"{key}:{variant_key}:base_codebook")
            if method == "base_vq_residual_rvq_unweighted":
                expected = expected_residual_bits[method]
                if component_index.get("residual_codebooks") != expected:
                    precision_errors.append(f"{key}:{variant_key}:unweighted_codebook")
            elif method in vq_methods - {"base_vq"}:
                expected = expected_residual_bits["fisher"]
                if component_index.get("residual_codebooks") != expected:
                    precision_errors.append(f"{key}:{variant_key}:fisher_codebook")
            if method in {
                "base_vq_mlp_router",
                "base_vq_logic_router",
                "logic_router_fixed_slots",
                "logic_router_fixed_slots_exact_fallback",
            }:
                router_expected = expected_router_bits[float(retention_rate)]
                if (
                    component_index.get("router_curvature_prior")
                    != router_expected["router_curvature_prior"]
                ):
                    precision_errors.append(f"{key}:{variant_key}:curvature")
                if method == "base_vq_mlp_router":
                    for name in ("mlp_router_parameters", "mlp_router_normalizer"):
                        if component_index.get(name) != router_expected[name]:
                            precision_errors.append(f"{key}:{variant_key}:{name}")
                elif component_index.get("logic_router") != router_expected["logic_router"]:
                    precision_errors.append(f"{key}:{variant_key}:logic_router")
            for metric in (
                "feature_nmse",
                "feature_cosine",
                "boundary_mse",
                "fisher_weighted_nmse",
            ):
                if not _finite(variant[metric]):
                    metric_errors.append(f"{key}:{variant_key}:{metric}")
            if variant_key[0] == "logic_router_fixed_slots_exact_fallback":
                exact_count = int((variant.get("mode_counts") or {}).get("exact", 0))
                payload = sum(
                    int(component["bits"])
                    for component in components
                    if component["name"] == "exact_fallback_payload"
                )
                expected_payload = exact_count * 4 * 2048 * 16
                if payload != expected_payload:
                    exact_payload_errors.append(
                        f"{key}:{variant_key}: {payload} != {expected_payload}"
                    )
        for variant_key, variant in quality_index.items():
            if (
                not _finite(variant["teacher_forced_nll"])
                or float(variant["teacher_forced_nll"]) < 0
                or int(variant["supervised_tokens"]) <= 0
            ):
                metric_errors.append(f"{key}:{variant_key}:nll")
            recomputed_score = dataset_score(
                quality_sample["dataset"], variant["prediction"], quality_sample["answers"]
            )
            expected_agreement = float(
                normalize_answer(variant["prediction"]) == baseline_prediction
            )
            if (
                not _close(float(variant["score"]), recomputed_score)
                or variant["normalized_prediction"]
                != normalize_answer(variant["prediction"])
                or not _close(float(variant["agrees_with_full"]), expected_agreement)
            ):
                score_errors.append(f"{key}:{variant_key}")
    _add(
        findings,
        "same_23_variant_matrix_for_feature_and_quality",
        not matrix_errors,
        {"errors": matrix_errors[:10], "expected_variants": EXPECTED_VARIANTS},
    )
    _add(
        findings,
        "exact_stream_shared_and_effective_rate_accounting",
        not rate_errors,
        {"errors": rate_errors[:10]},
    )
    _add(
        findings,
        "rate_precision_matches_executed_serialized_or_roundtripped_payloads",
        not precision_errors,
        {
            "policy": RATE_STORAGE_POLICY,
            "base_codebook_bits": expected_base_bits,
            "residual_fisher_bits": expected_residual_bits["fisher"],
            "residual_unweighted_bits": expected_residual_bits[
                "base_vq_residual_rvq_unweighted"
            ],
            "logic_tree_formats": sorted(logic_tree_formats),
            "legacy_logic_tree_values_require_exact_fp32_roundtrip": True,
            "errors": precision_errors[:10],
        },
    )
    _add(
        findings,
        "exact_fallback_fp16_payload_fully_charged",
        not exact_payload_errors,
        {"errors": exact_payload_errors[:10]},
    )
    _add(
        findings,
        "finite_feature_and_teacher_nll_metrics",
        not metric_errors,
        {"errors": metric_errors[:10]},
    )
    _add(
        findings,
        "answer_scores_and_agreement_recomputed",
        not score_errors,
        {"errors": score_errors[:10]},
    )

    router_errors = []
    oracle_samples = [sample for sample in feature if sample["oracle"]]
    for sample in oracle_samples:
        if len(sample["router_oracle"]) != len(RATES):
            router_errors.append(f"{_key(sample)}: rate count")
            continue
        for record in sample["router_oracle"]:
            locations = record["locations"]
            oracle = record["oracle_marginal_benefits"]
            arrays = (record["energy"], record["query_relevance"], record["risk"])
            scores = record["router_marginal_scores"]
            if (
                len(locations) != 256
                or len({tuple(item) for item in locations}) != 256
                or len(oracle) != 256
                or any(len(row) != 3 for row in oracle)
                or any(len(array) != 256 for array in arrays)
                or not {
                    "base_vq_mlp_router",
                    "base_vq_logic_router",
                }
                <= set(scores)
            ):
                router_errors.append(f"{_key(sample)}:{record['retention_rate']}:shape")
                continue
            for energy, relevance, risk in zip(*arrays):
                if not _close(float(risk), float(energy) * float(relevance), 2e-5):
                    router_errors.append(
                        f"{_key(sample)}:{record['retention_rate']}:risk product"
                    )
                    break
            for method, matrix in scores.items():
                if len(matrix) != 256 or any(len(row) != 3 for row in matrix):
                    router_errors.append(
                        f"{_key(sample)}:{record['retention_rate']}:{method}:shape"
                    )
                if not all(_finite(value) for row in matrix for value in row):
                    router_errors.append(
                        f"{_key(sample)}:{record['retention_rate']}:{method}:finite"
                    )
    _add(
        findings,
        "evaluation_oracle_router_arrays",
        len(oracle_samples) == 48 and not router_errors,
        {"oracle_samples": len(oracle_samples), "errors": router_errors[:10]},
    )

    latency_environment = _json(latency_dir / "latency_environment.json")
    analysis_environment = _json(analysis_dir / "analysis_environment.json")
    latency = _csv(latency_dir / "latency_samples.csv")
    latency_errors = []
    required_latency = {
        (component, method, rate)
        for rate in RATES
        for component, methods in {
            "layout_pack": ("base_vq_logic_router", "logic_router_fixed_slots"),
            "residual_decode_scatter": (
                "base_vq_logic_router",
                "logic_router_fixed_slots",
            ),
            "codec_roundtrip": ("base_vq_logic_router", "logic_router_fixed_slots"),
            "prefill_first_logits": (
                "base_vq_logic_router",
                "logic_router_fixed_slots",
            ),
            "codec_plus_prefill_first_logits": (
                "base_vq_logic_router",
                "logic_router_fixed_slots",
            ),
            "ttft": ("base_vq_logic_router", "logic_router_fixed_slots"),
        }.items()
        for method in methods
    }
    observed_latency = {
        (row["component"], row["method"], _rate(row["retention_rate"]))
        for row in latency
    }
    for row in latency:
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
            if not _finite(row[metric]) or float(row[metric]) < 0:
                latency_errors.append(f"{row['component']}:{row['method']}:{metric}")
        if int(row["peak_allocated_bytes"]) < 0 or int(row["peak_reserved_bytes"]) < 0:
            latency_errors.append(f"{row['component']}:{row['method']}:memory")
        if row["component"] == "ttft":
            required_flags = (
                "includes_image_preprocess",
                "includes_visual_encoder",
                "includes_native_positions",
                "includes_language_prefill",
                "includes_first_token",
            )
            if not all(_truth(row[name]) for name in required_flags):
                latency_errors.append(f"ttft:{row['method']}:inclusion flags")
    _add(
        findings,
        "latency_components_memory_and_inclusion_flags",
        required_latency <= observed_latency
        and not latency_errors
        and latency_environment.get("fixed_codec_ttft_expands_to_full_visual_tokens")
        is True
        and latency_environment.get("paired_dynamic_fixed_methods")
        == ["base_vq_logic_router", "logic_router_fixed_slots"],
        {
            "rows": len(latency),
            "missing": sorted(str(item) for item in required_latency - observed_latency),
            "errors": latency_errors[:10],
        },
    )
    gpu_snapshot_path = latency_dir / "gpu_co_residency_during_run.log"
    gpu_snapshot = analysis_environment.get("latency", {}).get(
        "gpu_co_residency_log", {}
    )
    snapshot_hash = (
        hashlib.sha256(gpu_snapshot_path.read_bytes()).hexdigest()
        if gpu_snapshot_path.is_file()
        else None
    )
    _add(
        findings,
        "latency_gpu_co_residency_provenance",
        gpu_snapshot_path.is_file()
        and gpu_snapshot.get("file") == gpu_snapshot_path.name
        and gpu_snapshot.get("bytes") == gpu_snapshot_path.stat().st_size
        and gpu_snapshot.get("sha256") == snapshot_hash,
        {
            "file": str(gpu_snapshot_path),
            "recorded": gpu_snapshot,
            "recomputed_sha256": snapshot_hash,
        },
    )

    feature_metrics = _csv(analysis_dir / "feature_metrics.csv")
    depth_errors = []
    for row in feature_metrics:
        drop = int(row["mode_drop"])
        rvq1 = int(row["mode_rvq1"])
        rvq2 = int(row["mode_rvq2"])
        exact = int(row["mode_exact"])
        total = drop + rvq1 + rvq2 + exact
        vq_total = rvq1 + rvq2
        weighted = rvq1 + 2 * rvq2
        expected_all = weighted / total if total else 0.0
        expected_active = weighted / vq_total if vq_total else 0.0
        if not _close(float(row["mean_rvq_depth_all_blocks"]), expected_all):
            depth_errors.append(
                f"{row['dataset']}:{row['method']}:{row['retention_rate']}:all"
            )
        if not _close(float(row["mean_rvq_depth_given_vq"]), expected_active):
            depth_errors.append(
                f"{row['dataset']}:{row['method']}:{row['retention_rate']}:active"
            )
    _add(
        findings,
        "router_mode_and_depth_usage",
        bool(feature_metrics) and not depth_errors,
        {"rows": len(feature_metrics), "errors": depth_errors[:10]},
    )

    decisions = _json(analysis_dir / "decision_summary.json")
    decision_errors = _decision_boolean_audit(decisions)
    _add(
        findings,
        "six_predeclared_decisions_recomputed",
        not decision_errors,
        {
            "errors": decision_errors,
            "statuses": {str(item["id"]): item["status"] for item in decisions["questions"]},
        },
    )

    report = (analysis_dir / "TILELOGIC_RVQ_FINAL_REPORT.md").read_text(encoding="utf-8")
    boundary_phrases = (
        "not native compact-prefill latency evidence",
        "No PPA, kernel-fusion, or physical-hardware claim is made",
        "does not support an aggregate positive claim",
    )
    aggregate_negative_needed = not decisions["all_questions_pass"]
    report_ok = all(phrase in report for phrase in boundary_phrases[:2]) and (
        not aggregate_negative_needed or boundary_phrases[2] in report
    )
    _add(
        findings,
        "report_claim_boundaries",
        report_ok,
        {
            "aggregate_negative_needed": aggregate_negative_needed,
            "required_phrases": boundary_phrases,
        },
    )

    major_failures = [
        finding
        for finding in findings
        if finding["status"] == "FAIL" and finding["severity"] == "major"
    ]
    overall = "PASS" if not major_failures else "FAIL"
    payload = {
        "format": AUDIT_FORMAT,
        "overall": overall,
        "major_failures": len(major_failures),
        "checks": findings,
    }
    (analysis_dir / "result_audit_findings.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed_lines = (
        "None."
        if not major_failures
        else "\n".join(
            f"- `{item['check']}`: {json.dumps(item['evidence'], sort_keys=True)}"
            for item in major_failures
        )
    )
    report_text = f"""# TileLogic-RVQ Result Audit

## Status

**{overall}**

## Major Issues

{failed_lines}

## Minor Issues

None recorded by the machine audit.

## Checks

""" + "\n".join(
        f"- **{item['status']}** `{item['check']}`: `{json.dumps(item['evidence'], sort_keys=True)}`"
        for item in findings
    ) + """

## Recommended Next Step

Use exactly one independent Review Agent to inspect the implementation, raw evidence,
decision rules, and claim boundaries. Resolve every major finding before publication.
"""
    (analysis_dir / "result_audit_report.md").write_text(
        report_text, encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if major_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
