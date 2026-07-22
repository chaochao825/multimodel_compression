from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHODS = (
    ("ours", "Ours"),
    ("exact_recent", "Exact recent"),
    ("causalmem", "CausalMem"),
    ("stc", "STC"),
    ("streamingtom", "StreamingTOM"),
    ("selectstream", "SelectStream"),
    ("oasis", "OASIS"),
    ("statekv", "StateKV"),
)

STAGES = (
    ("source", "Source"),
    ("module_smoke", "Module smoke"),
    ("dataset_preflight", "Data preflight"),
    ("runtime_preflight", "Runtime preflight"),
    ("official_quality", "Quality"),
    ("official_latency", "Latency"),
    ("state_accounting", "State accounting"),
    ("independent_replication", "Independent replication"),
)

ALLOWED_STATUSES = {
    "PASS",
    "FAIL",
    "OPEN",
    "QUEUED",
    "RUNNING",
    "UNAVAILABLE",
    "PLACEHOLDER",
    "PAPER_ONLY",
    "PROXY_ONLY",
    "SMOKE_ONLY",
    "NA",
}

STATUS_STYLE = {
    "PASS": ("#009E73", "", "PASS"),
    "FAIL": ("#D55E00", "xx", "FAIL"),
    "OPEN": ("#BDBDBD", "..", "OPEN"),
    "QUEUED": ("#E69F00", "//", "QUEUE"),
    "RUNNING": ("#0072B2", "\\\\", "RUN"),
    "UNAVAILABLE": ("#4D4D4D", "++", "UNAV"),
    "PLACEHOLDER": ("#7F7F7F", "oo", "HOLD"),
    "PAPER_ONLY": ("#56B4E9", "oo", "PAPER"),
    "PROXY_ONLY": ("#CC79A7", "**", "PROXY"),
    "SMOKE_ONLY": ("#F0E442", "--", "SMOKE"),
    "NA": ("#FFFFFF", "", "N/A"),
}

EVIDENCE_FIELDS = (
    "evidence_id",
    "method_id",
    "component_id",
    "comparability_group",
    "evidence_tier",
    "data_reuse",
    "sample_count",
    "gate_id",
    "gate_status",
    "valid_for_positive_claim",
    "metric_name",
    "metric_value",
    "metric_unit",
    "ci95_low",
    "ci95_high",
    "reference",
    "source_path",
    "source_sha256",
    "notes",
)

MATRIX_FIELDS = (
    "method_id",
    "method_label",
    "stage",
    "stage_label",
    "status",
    "detail",
    "source_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an evidence-tiered online-video baseline completion matrix"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--runtime-status", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _load_object(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _select_one(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(field) == expected for field, expected in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        observed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric: {value!r}") from error
    if not math.isfinite(observed):
        raise ValueError(f"{label} must be finite: {value!r}")
    return observed


def _integer(value: Any, *, label: str) -> int:
    observed = _finite(value, label=label)
    if not observed.is_integer() or observed < 0:
        raise ValueError(f"{label} must be a non-negative integer: {value!r}")
    return int(observed)


def _metric(
    name: str,
    value: float | int,
    unit: str,
    *,
    ci95_low: float | None = None,
    ci95_high: float | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "reference": reference,
    }


def _evidence(
    *,
    evidence_id: str,
    method_id: str,
    component_id: str,
    comparability_group: str,
    evidence_tier: str,
    data_reuse: str,
    sample_count: int,
    gate_id: str,
    gate_status: str,
    valid_for_positive_claim: bool,
    metrics: list[dict[str, Any]],
    source_path: str,
    source_sha256: str,
    notes: str,
) -> dict[str, Any]:
    if gate_status not in ALLOWED_STATUSES:
        raise ValueError(f"invalid gate status: {gate_status}")
    return {
        "evidence_id": evidence_id,
        "method_id": method_id,
        "component_id": component_id,
        "comparability_group": comparability_group,
        "evidence_tier": evidence_tier,
        "data_reuse": data_reuse,
        "sample_count": sample_count,
        "gate": {
            "gate_id": gate_id,
            "status": gate_status,
            "valid_for_positive_claim": valid_for_positive_claim,
        },
        "metrics": metrics,
        "source": {"path": source_path, "sha256": source_sha256},
        "notes": notes,
    }


def _artifact(repo_root: Path, relative: str) -> tuple[Path, str]:
    path = repo_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required evidence artifact not found: {path}")
    return path, _sha256(path)


def build_evidence(repo_root: Path) -> list[dict[str, Any]]:
    repo_root = repo_root.resolve()
    evidence: list[dict[str, Any]] = []

    formal_rel = (
        "paper/results/probe_mvp/mvbench_query_formal_20260717_v1/"
        "aggregate/promotion_decision.json"
    )
    formal_path, formal_sha = _artifact(repo_root, formal_rel)
    formal = _load_object(formal_path)
    paired = formal["paired"]
    if formal.get("advance_to_llava_anchor") is not False:
        raise ValueError("formal query-selector gate unexpectedly changed")
    evidence.append(
        _evidence(
            evidence_id="ours_query_selector_formal_200",
            method_id="ours",
            component_id="query_conditioned_selector",
            comparability_group="mvbench_clip_preregistered_200",
            evidence_tier="project_native_preregistered_proxy",
            data_reuse="fresh_preregistered_evaluation",
            sample_count=_integer(paired["paired_samples"], label="formal paired samples"),
            gate_id="query_selector_promotion",
            gate_status="FAIL",
            valid_for_positive_claim=False,
            metrics=[
                _metric(
                    "accuracy_gain",
                    _finite(paired["accuracy_gain"], label="formal selector gain"),
                    "fraction",
                    ci95_low=_finite(paired["bootstrap_ci95_low"], label="formal CI low"),
                    ci95_high=_finite(paired["bootstrap_ci95_high"], label="formal CI high"),
                    reference="exact_recent",
                ),
                _metric(
                    "mcnemar_exact_p",
                    _finite(paired["mcnemar_exact_p"], label="formal McNemar p"),
                    "probability",
                ),
            ],
            source_path=formal_rel,
            source_sha256=formal_sha,
            notes="The preregistered primary did not advance; exploratory readers are separate evidence.",
        )
    )

    native_pair_rel = (
        "paper/results/probe_mvp/mvbench_feature_memory_confirmation_20260718_v1/"
        "aggregate/paired_vs_exact_recent.csv"
    )
    native_pair_path, native_pair_sha = _artifact(repo_root, native_pair_rel)
    native_pair = _select_one(
        _read_csv(native_pair_path), policy="learned_recent_query_topk"
    )
    native_accuracy_rel = (
        "paper/results/probe_mvp/mvbench_feature_memory_confirmation_20260718_v1/"
        "aggregate/overall_accuracy.csv"
    )
    native_accuracy_path, _ = _artifact(repo_root, native_accuracy_rel)
    native_rows = _read_csv(native_accuracy_path)
    native_learned = _select_one(native_rows, policy="learned_recent_query_topk")
    native_recent = _select_one(native_rows, policy="exact_recent")
    native_validation_rel = (
        "paper/results/probe_mvp/mvbench_feature_memory_confirmation_20260718_v1/"
        "aggregate/native_feature_memory_validation.csv"
    )
    native_validation_path, _ = _artifact(repo_root, native_validation_rel)
    state_row = _select_one(
        _read_csv(native_validation_path),
        panel="state",
        metric="total_persistent_state",
        statistic="bytes",
    )
    evidence.append(
        _evidence(
            evidence_id="ours_native_query_memory_200",
            method_id="ours",
            component_id="query_conditioned_native_memory",
            comparability_group="mvbench_llava_native_matched_state_200",
            evidence_tier="project_native_model_level_confirmation",
            data_reuse="frozen_confirmation_split",
            sample_count=_integer(native_pair["paired_samples"], label="native samples"),
            gate_id="native_query_memory_independent_confirmation",
            gate_status="OPEN",
            valid_for_positive_claim=False,
            metrics=[
                _metric(
                    "accuracy",
                    _finite(
                        native_learned["micro_accuracy"],
                        label="native learned accuracy",
                    ),
                    "fraction",
                ),
                _metric(
                    "reference_accuracy",
                    _finite(
                        native_recent["micro_accuracy"],
                        label="native recent accuracy",
                    ),
                    "fraction",
                    reference="exact_recent",
                ),
                _metric(
                    "accuracy_gain",
                    _finite(native_pair["accuracy_gain"], label="native gain"),
                    "fraction",
                    ci95_low=_finite(native_pair["bootstrap_ci95_low"], label="native CI low"),
                    ci95_high=_finite(native_pair["bootstrap_ci95_high"], label="native CI high"),
                    reference="exact_recent",
                ),
                _metric(
                    "persistent_state_bytes",
                    _finite(state_row["value"], label="native state bytes"),
                    "bytes",
                ),
            ],
            source_path=native_pair_rel,
            source_sha256=native_pair_sha,
            notes="Matched provisioned native state gives a positive point estimate, but no independent 400-sample end-to-end promotion exists.",
        )
    )

    spectral_rel = "paper/results/probe_mvp/controlled_spectral_trigger_20260719/summary.json"
    spectral_path, spectral_sha = _artifact(repo_root, spectral_rel)
    spectral = _load_object(spectral_path)
    gates = spectral["gates"]
    if gates.get("all_passed") is not False:
        raise ValueError("controlled spectral gate unexpectedly changed")
    primary_rank = gates["primary_rank_budget"]
    spectral_rows = [
        row
        for row in spectral["summary"]
        if row.get("method") == "dual_spectral"
        and row.get("total_rank_budget") == primary_rank
    ]
    if len(spectral_rows) != 1:
        raise ValueError("missing primary dual-spectral summary")
    spectral_row = spectral_rows[0]
    comparison = next(
        row
        for row in spectral["paired_false_trigger_comparisons"]
        if row["baseline_method"] == "causalmem_residual_proxy"
        and row["total_rank_budget"] == primary_rank
    )
    rare_gate = gates["checks"]["rare_recall_vs_residual_only"]
    evidence.append(
        _evidence(
            evidence_id="ours_dual_spectral_controlled",
            method_id="ours",
            component_id="dual_timescale_spectral_trigger",
            comparability_group="controlled_trigger_matched_rank",
            evidence_tier="controlled_synthetic_trigger",
            data_reuse="disjoint_calibration_and_evaluation_seeds",
            sample_count=8,
            gate_id="spectral_rare_event_recall",
            gate_status="FAIL",
            valid_for_positive_claim=False,
            metrics=[
                _metric("event_recall", spectral_row["event_recall"], "fraction"),
                _metric(
                    "false_trigger_rate", spectral_row["false_trigger_rate"], "fraction"
                ),
                _metric(
                    "false_trigger_delta_vs_causalmem_proxy",
                    comparison["candidate_minus_baseline_pp"],
                    "percentage_points",
                    ci95_low=comparison["paired_bootstrap_ci95_low_pp"],
                    ci95_high=comparison["paired_bootstrap_ci95_high_pp"],
                    reference="causalmem_residual_proxy",
                ),
                _metric("writer_p95", spectral_row["update_p95_us"], "microseconds"),
                _metric("state_bytes", spectral_row["state_bytes"], "bytes"),
                _metric("rare_recall_gain", rare_gate["value"], "fraction"),
            ],
            source_path=spectral_rel,
            source_sha256=spectral_sha,
            notes="Five sub-gates pass, but rare-event recall gain is 0 rather than the preregistered +10-point requirement.",
        )
    )

    variant_rel = (
        "paper/results/probe_mvp/mvbench_compressed_feature_confirmation_rank256_20260718_v1/"
        "aggregate/variant_summary.csv"
    )
    variant_path, _ = _artifact(repo_root, variant_rel)
    variant = _select_one(
        _read_csv(variant_path),
        selection_policy="learned_recent_query_topk",
        memory_variant="pca_r256_s4",
    )
    fixed_pair_rel = (
        "paper/results/probe_mvp/mvbench_compressed_feature_confirmation_rank256_20260718_v1/"
        "aggregate/paired_vs_full.csv"
    )
    fixed_pair_path, fixed_pair_sha = _artifact(repo_root, fixed_pair_rel)
    fixed_pair = _select_one(
        _read_csv(fixed_pair_path),
        selection_policy="learned_recent_query_topk",
        memory_variant="pca_r256_s4",
    )
    if fixed_pair["noninferior_at_margin"] != "0":
        raise ValueError("fixed rank-256+s4 non-inferiority gate unexpectedly changed")
    evidence.append(
        _evidence(
            evidence_id="ours_fixed_rank256_sparse4_200",
            method_id="ours",
            component_id="low_rank_sparse_codec",
            comparability_group="mvbench_llava_codec_confirmation_200",
            evidence_tier="project_native_model_level_confirmation",
            data_reuse="frozen_confirmation_split",
            sample_count=_integer(fixed_pair["paired_samples"], label="codec samples"),
            gate_id="codec_two_point_noninferiority",
            gate_status="FAIL",
            valid_for_positive_claim=False,
            metrics=[
                _metric("accuracy", _finite(variant["accuracy"], label="codec accuracy"), "fraction"),
                _metric(
                    "accuracy_gain",
                    _finite(fixed_pair["accuracy_gain"], label="codec gain"),
                    "fraction",
                    ci95_low=_finite(fixed_pair["bootstrap_ci95_low"], label="codec CI low"),
                    ci95_high=_finite(fixed_pair["bootstrap_ci95_high"], label="codec CI high"),
                    reference="full_feature_cache",
                ),
                _metric(
                    "worse_rate_upper_95",
                    _finite(fixed_pair["worse_rate_upper_95"], label="codec loss bound"),
                    "fraction",
                    reference="noninferiority_margin_0.02",
                ),
                _metric(
                    "steady_state_bytes",
                    _finite(variant["mean_total_state_bytes"], label="codec state"),
                    "bytes",
                ),
                _metric(
                    "cold_start_bytes",
                    _finite(variant["cold_start_total_state_bytes"], label="codec cold state"),
                    "bytes",
                ),
                _metric(
                    "state_compression_ratio",
                    _finite(
                        variant["mean_total_state_compression_ratio"],
                        label="codec compression ratio",
                    ),
                    "ratio",
                ),
            ],
            source_path=fixed_pair_rel,
            source_sha256=fixed_pair_sha,
            notes="The one-sided 95% loss bound is 2.3498%, above the fixed 2% margin.",
        )
    )

    routed_rel = (
        "paper/results/probe_mvp/mvbench_routed_residual_exploratory_200_20260718_v2/"
        "aggregate/paired_vs_full.csv"
    )
    routed_path, routed_sha = _artifact(repo_root, routed_rel)
    routed = _select_one(
        _read_csv(routed_path),
        selection_policy="learned_recent_query_topk",
        memory_variant="pca_r256_route_grid2_s4",
    )
    evidence.append(
        _evidence(
            evidence_id="ours_routed_codec_posthoc_200",
            method_id="ours",
            component_id="routed_low_rank_spatial_sparse_codec",
            comparability_group="mvbench_llava_codec_posthoc_200",
            evidence_tier="project_native_posthoc_same_set",
            data_reuse="posthoc_same_set",
            sample_count=_integer(routed["paired_samples"], label="routed samples"),
            gate_id="routed_codec_independent_noninferiority",
            gate_status="OPEN",
            valid_for_positive_claim=False,
            metrics=[
                _metric(
                    "prediction_agreement_rate",
                    _finite(routed["prediction_agreement_rate"], label="routed agreement"),
                    "fraction",
                ),
                _metric(
                    "worse_rate_upper_95",
                    _finite(routed["worse_rate_upper_95"], label="routed loss bound"),
                    "fraction",
                    reference="noninferiority_margin_0.02",
                ),
                _metric(
                    "accuracy_gain",
                    _finite(routed["accuracy_gain"], label="routed gain"),
                    "fraction",
                    reference="full_feature_cache",
                ),
            ],
            source_path=routed_rel,
            source_sha256=routed_sha,
            notes="The post-hoc router clears the numerical margin on reused data, but has no disjoint training/evaluation confirmation.",
        )
    )

    independent_root = (
        "paper/results/probe_mvp/"
        "mvbench_independent_replication_300_20260722/aggregate/"
    )
    independent_validation_rel = independent_root + "full_validation.json"
    independent_validation_path, _ = _artifact(
        repo_root, independent_validation_rel
    )
    independent_validation = _load_object(independent_validation_path)
    if independent_validation.get("passed") is not True:
        raise ValueError("independent native-memory validation did not pass")
    independent_pair_rel = independent_root + "paired_vs_full.csv"
    independent_pair_path, independent_pair_sha = _artifact(
        repo_root, independent_pair_rel
    )
    independent_pair = _select_one(
        _read_csv(independent_pair_path),
        selection_policy="learned_recent_query_topk",
        memory_variant="pca_r256_route_grid2_s4",
    )
    if independent_pair["noninferior_at_margin"] != "1":
        raise ValueError("independent routed-codec preservation gate did not pass")
    independent_summary_rel = independent_root + "variant_summary.csv"
    independent_summary_path, _ = _artifact(repo_root, independent_summary_rel)
    independent_summary = _select_one(
        _read_csv(independent_summary_path),
        selection_policy="learned_recent_query_topk",
        memory_variant="pca_r256_route_grid2_s4",
    )
    evidence.append(
        _evidence(
            evidence_id="ours_routed_codec_independent_300",
            method_id="ours",
            component_id="routed_low_rank_spatial_sparse_codec",
            comparability_group="mvbench_llava_independent_reserve_300",
            evidence_tier="project_native_frozen_independent_replication",
            data_reuse="untouched_final_reserve_frozen_before_results",
            sample_count=_integer(
                independent_pair["paired_samples"],
                label="independent routed samples",
            ),
            gate_id="routed_codec_two_point_noninferiority",
            gate_status="PASS",
            valid_for_positive_claim=True,
            metrics=[
                _metric(
                    "accuracy",
                    _finite(
                        independent_summary["accuracy"],
                        label="independent routed accuracy",
                    ),
                    "fraction",
                ),
                _metric(
                    "accuracy_gain_vs_full",
                    _finite(
                        independent_pair["accuracy_gain"],
                        label="independent routed gain",
                    ),
                    "fraction",
                    ci95_low=_finite(
                        independent_pair["bootstrap_ci95_low"],
                        label="independent routed CI low",
                    ),
                    ci95_high=_finite(
                        independent_pair["bootstrap_ci95_high"],
                        label="independent routed CI high",
                    ),
                    reference="full_feature_cache",
                ),
                _metric(
                    "prediction_agreement_rate",
                    _finite(
                        independent_pair["prediction_agreement_rate"],
                        label="independent routed agreement",
                    ),
                    "fraction",
                ),
                _metric(
                    "worse_rate_upper_95",
                    _finite(
                        independent_pair["worse_rate_upper_95"],
                        label="independent routed loss bound",
                    ),
                    "fraction",
                    reference="noninferiority_margin_0.02",
                ),
                _metric(
                    "steady_state_bytes",
                    _finite(
                        independent_summary["mean_total_state_bytes"],
                        label="independent routed state",
                    ),
                    "bytes",
                ),
                _metric(
                    "cold_start_bytes",
                    _finite(
                        independent_summary["cold_start_total_state_bytes"],
                        label="independent routed cold state",
                    ),
                    "bytes",
                ),
                _metric(
                    "state_compression_ratio",
                    _finite(
                        independent_summary[
                            "mean_total_state_compression_ratio"
                        ],
                        label="independent routed compression",
                    ),
                    "ratio",
                ),
            ],
            source_path=independent_pair_rel,
            source_sha256=independent_pair_sha,
            notes=(
                "The frozen error-oracle route passes the independent "
                "representation-preservation gate at 7.84x steady-state "
                "compression; this is not a cheap-router or latency result."
            ),
        )
    )

    independent_selector_rel = independent_root + "selector_gain_by_variant.csv"
    independent_selector_path, independent_selector_sha = _artifact(
        repo_root, independent_selector_rel
    )
    independent_selector = _select_one(
        _read_csv(independent_selector_path),
        selection_policy="learned_recent_query_topk",
        memory_variant="pca_r256_route_grid2_s4",
    )
    evidence.append(
        _evidence(
            evidence_id="ours_query_memory_independent_300",
            method_id="ours",
            component_id="query_conditioned_routed_memory",
            comparability_group="mvbench_llava_independent_reserve_300",
            evidence_tier="project_native_frozen_independent_replication",
            data_reuse="untouched_final_reserve_frozen_before_results",
            sample_count=_integer(
                independent_selector["paired_samples"],
                label="independent selector samples",
            ),
            gate_id="query_memory_confirmatory_superiority",
            gate_status="OPEN",
            valid_for_positive_claim=False,
            metrics=[
                _metric(
                    "accuracy_gain",
                    _finite(
                        independent_selector["accuracy_gain"],
                        label="independent selector gain",
                    ),
                    "fraction",
                    ci95_low=_finite(
                        independent_selector["bootstrap_ci95_low"],
                        label="independent selector CI low",
                    ),
                    ci95_high=_finite(
                        independent_selector["bootstrap_ci95_high"],
                        label="independent selector CI high",
                    ),
                    reference="exact_recent_matched_routed_state",
                ),
                _metric(
                    "mcnemar_exact_p",
                    _finite(
                        independent_selector["mcnemar_exact_p"],
                        label="independent selector McNemar p",
                    ),
                    "probability",
                ),
                _metric(
                    "better_samples",
                    _integer(
                        independent_selector["better_samples"],
                        label="independent selector better samples",
                    ),
                    "count",
                ),
                _metric(
                    "worse_samples",
                    _integer(
                        independent_selector["worse_samples"],
                        label="independent selector worse samples",
                    ),
                    "count",
                ),
            ],
            source_path=independent_selector_rel,
            source_sha256=independent_selector_sha,
            notes=(
                "The frozen learned reader gains 2.0 points (8 better, 2 "
                "worse), but its interval touches zero and McNemar p=0.1094."
            ),
        )
    )

    decision_rel = (
        "paper/results/probe_mvp/clip_stratified_formal30_20260717/"
        "formal_probe_decision_metrics.csv"
    )
    decision_path, decision_sha = _artifact(repo_root, decision_rel)
    decisions = [row for row in _read_csv(decision_path) if row["eligible"] == "True"]

    def metric_rows(name: str) -> list[dict[str, str]]:
        matches = [row for row in decisions if row["metric"] == name]
        if not matches:
            raise ValueError(f"missing formal decision metric: {name}")
        return matches

    state_energy = metric_rows("state_rank32_energy")
    causal_error = metric_rows("causal_rank32_projection_error")
    residual_energy = metric_rows("residual_top10_energy")
    residual_recall = metric_rows("residual_top10_change_recall")
    residual_joint = metric_rows("residual_joint_gate")
    bccb_gain = metric_rows("bccb_gain_vs_identity")
    bccb_increment = metric_rows("bccb_increment_vs_bttb")

    def mean(rows: list[dict[str, str]]) -> float:
        return sum(_finite(row["value"], label=row["metric"]) for row in rows) / len(rows)

    evidence.append(
        _evidence(
            evidence_id="ours_low_rank_state_formal30",
            method_id="ours",
            component_id="low_rank_long_term_state",
            comparability_group="video_mme_clip_stratified_formal30",
            evidence_tier="project_native_representation_probe",
            data_reuse="formal_stratified_probe",
            sample_count=30,
            gate_id="p1_low_dimensional_state_two_domains",
            gate_status="OPEN",
            valid_for_positive_claim=False,
            metrics=[
                _metric("rank32_energy_mean", mean(state_energy), "fraction"),
                _metric(
                    "rank32_energy_cells_passing",
                    sum(row["gate_pass"] == "True" for row in state_energy),
                    "count",
                ),
                _metric("causal_rank32_projection_error_mean", mean(causal_error), "fraction"),
            ],
            source_path=decision_rel,
            source_sha256=decision_sha,
            notes="Rank-32 energy is encouraging in one encoder domain, but the preregistered two-domain or matched-byte task gate remains open.",
        )
    )
    evidence.append(
        _evidence(
            evidence_id="ours_sparse_event_residual_formal30",
            method_id="ours",
            component_id="sparse_event_residual",
            comparability_group="video_mme_clip_stratified_formal30",
            evidence_tier="project_native_representation_probe",
            data_reuse="formal_stratified_probe",
            sample_count=30,
            gate_id="p2_sparse_event_joint_gate",
            gate_status="FAIL",
            valid_for_positive_claim=False,
            metrics=[
                _metric("top10_residual_energy_mean", mean(residual_energy), "fraction"),
                _metric("top10_change_recall_mean", mean(residual_recall), "fraction"),
                _metric(
                    "joint_cells_passing",
                    sum(row["gate_pass"] == "True" for row in residual_joint),
                    "count",
                ),
            ],
            source_path=decision_rel,
            source_sha256=decision_sha,
            notes="No category-layer cell passes both the 70% residual-energy and 80% event-recall thresholds.",
        )
    )
    evidence.append(
        _evidence(
            evidence_id="ours_bccb_transport_formal30",
            method_id="ours",
            component_id="bccb_transport",
            comparability_group="video_mme_clip_stratified_formal30",
            evidence_tier="project_native_representation_probe",
            data_reuse="formal_stratified_probe",
            sample_count=30,
            gate_id="p3_bccb_incremental_advantage",
            gate_status="FAIL",
            valid_for_positive_claim=False,
            metrics=[
                _metric("bccb_gain_vs_identity_mean", mean(bccb_gain), "fraction"),
                _metric(
                    "bccb_cells_passing_identity_gate",
                    sum(row["gate_pass"] == "True" for row in bccb_gain),
                    "count",
                ),
                _metric("bccb_increment_vs_bttb_mean", mean(bccb_increment), "fraction"),
            ],
            source_path=decision_rel,
            source_sha256=decision_sha,
            notes="BCCB tracks BTTB and has no demonstrated cyclic or measured-cost advantage.",
        )
    )

    proxy_rel = "paper/results/probe_mvp/streaming_baseline_proxy_20260719/overall_summary.csv"
    proxy_path, proxy_sha = _artifact(repo_root, proxy_rel)
    proxy_rows = _read_csv(proxy_path)
    proxy_method_map = {
        "exact_recent": "exact_recent",
        "causalmem_feature_proxy": "causalmem",
        "streamingtom_feature_proxy": "streamingtom",
        "stc_feature_proxy": "stc",
        "selectstream_feature_proxy": "selectstream",
        "oasis_feature_proxy": "oasis",
        "statekv_feature_proxy": "statekv",
        "ours_learned_recent_selector": "ours",
    }
    for row in proxy_rows:
        method = row["method"]
        if method not in proxy_method_map:
            raise ValueError(f"unknown streaming proxy method: {method}")
        method_id = proxy_method_map[method]
        status = "PASS" if method == "exact_recent" else "PROXY_ONLY"
        evidence.append(
            _evidence(
                evidence_id=f"proxy_{method}",
                method_id=method_id,
                component_id="unified_feature_proxy",
                comparability_group="mvbench_clip_streaming_proxy_200",
                evidence_tier=row["reproduction_tier"],
                data_reuse="frozen_then_reused_development_pool",
                sample_count=_integer(row["samples"], label=f"{method} samples"),
                gate_id="proxy_mechanism_only",
                gate_status=status,
                valid_for_positive_claim=method == "exact_recent",
                metrics=[
                    _metric(
                        "micro_accuracy",
                        _finite(row["micro_accuracy"], label=f"{method} accuracy"),
                        "fraction",
                    ),
                    _metric(
                        "mean_evidence_count",
                        _finite(row["mean_evidence_count"], label=f"{method} evidence"),
                        "count",
                    ),
                    _metric(
                        "mean_total_retained_bytes",
                        _finite(
                            row["mean_total_retained_bytes"],
                            label=f"{method} retained bytes",
                        ),
                        "bytes",
                    ),
                    _metric(
                        "total_state_bounded",
                        _integer(row["total_state_bounded"], label=f"{method} bounded"),
                        "boolean",
                    ),
                ],
                source_path=proxy_rel,
                source_sha256=proxy_sha,
                notes="Comparable only inside the frozen CLIP mechanism-proxy group; not official model quality or GPU latency.",
            )
        )

    official_root = (
        "paper/results/probe_mvp/official_streaming_formal_20260722/"
    )
    official_quality_rel = official_root + "official_quality_formal.csv"
    official_quality_path, official_quality_sha = _artifact(
        repo_root, official_quality_rel
    )
    official_quality = _read_csv(official_quality_path)
    quality_method_ids = {"CausalMem": "causalmem", "OASIS": "oasis"}
    for row in official_quality:
        if row["method"] not in quality_method_ids:
            raise ValueError(f"unknown official quality method: {row['method']}")
        expected = _integer(
            row["expected"], label=f"{row['method']} expected questions"
        )
        scored = _integer(
            row["scored"], label=f"{row['method']} scored questions"
        )
        if scored != expected or _finite(
            row["coverage"], label=f"{row['method']} coverage"
        ) != 1.0:
            raise ValueError(f"incomplete official quality result: {row['method']}")
        method_id = quality_method_ids[row["method"]]
        evidence.append(
            _evidence(
                evidence_id=f"{method_id}_official_quality_50x5",
                method_id=method_id,
                component_id="official_streamingbench_system",
                comparability_group="streamingbench_rtu_official_50x5",
                evidence_tier=row["evidence_tier"],
                data_reuse="official_fixed_50_video_evaluation",
                sample_count=scored,
                gate_id="official_quality_run_complete",
                gate_status="PASS",
                valid_for_positive_claim=True,
                metrics=[
                    _metric(
                        "accuracy",
                        _finite(
                            row["accuracy"],
                            label=f"{row['method']} official accuracy",
                        ),
                        "fraction",
                    ),
                    _metric(
                        "correct",
                        _integer(
                            row["correct"],
                            label=f"{row['method']} official correct",
                        ),
                        "count",
                    ),
                    _metric("coverage", 1.0, "fraction"),
                ],
                source_path=official_quality_rel,
                source_sha256=official_quality_sha,
                notes=(
                    "Official system-level StreamingBench reproduction. "
                    "CausalMem and OASIS use different VLM backbones, so "
                    "their paired scores are not a memory-module ablation."
                ),
            )
        )

    official_runs_rel = official_root + "official_runs.csv"
    official_runs_path, _ = _artifact(repo_root, official_runs_rel)
    official_runs = _read_csv(official_runs_path)
    stc_rel = official_root + "official_stc_stage_latency.csv"
    stc_path, stc_sha = _artifact(repo_root, stc_rel)
    stc_rows = _read_csv(stc_path)
    rekv_stage = _select_one(
        stc_rows, mode="rekv", stage="instrumented_stage_sum_ms"
    )
    stc_stage = _select_one(
        stc_rows, mode="stc", stage="instrumented_stage_sum_ms"
    )
    rekv_run = _select_one(official_runs, method="STC ReKV", variant="rekv")
    stc_run = _select_one(official_runs, method="STC ReKV", variant="stc")
    stc_p50_reduction = 1.0 - _finite(
        stc_stage["p50_ms"], label="STC P50"
    ) / _finite(rekv_stage["p50_ms"], label="ReKV P50")
    stc_mean_reduction = 1.0 - _finite(
        stc_stage["mean_ms"], label="STC mean"
    ) / _finite(rekv_stage["mean_ms"], label="ReKV mean")
    stc_peak_reduction = 1.0 - _finite(
        stc_run["peak_memory_value"], label="STC peak memory"
    ) / _finite(rekv_run["peak_memory_value"], label="ReKV peak memory")
    evidence.append(
        _evidence(
            evidence_id="stc_official_rekv_stage_pair_20",
            method_id="stc",
            component_id="official_rekv_stage_latency",
            comparability_group="stc_rekv_official_model_stage_pair",
            evidence_tier="official_model_stage_latency",
            data_reuse="official_benchmark_pair",
            sample_count=_integer(stc_stage["count"], label="STC stage count"),
            gate_id="official_stage_pair_complete",
            gate_status="PASS",
            valid_for_positive_claim=True,
            metrics=[
                _metric(
                    "rekv_p50",
                    _finite(rekv_stage["p50_ms"], label="ReKV P50"),
                    "milliseconds",
                ),
                _metric(
                    "stc_p50",
                    _finite(stc_stage["p50_ms"], label="STC P50"),
                    "milliseconds",
                ),
                _metric("p50_reduction", stc_p50_reduction, "fraction"),
                _metric("mean_reduction", stc_mean_reduction, "fraction"),
                _metric("peak_memory_reduction", stc_peak_reduction, "fraction"),
            ],
            source_path=stc_rel,
            source_sha256=stc_sha,
            notes=(
                "Matched official ViT-plus-visual-prefill stages only; not "
                "request TTFT, decode, quality, or an end-to-end SLO result."
            ),
        )
    )

    streamingtom_rel = official_root + "official_streamingtom_core_latency.csv"
    streamingtom_path, streamingtom_sha = _artifact(repo_root, streamingtom_rel)
    streamingtom_rows = [
        row
        for row in _read_csv(streamingtom_path)
        if row["timing_basis"] == "cuda_event"
    ]
    streamingtom_metrics = []
    for variant in (
        "streamingtom_ctr",
        "streamingtom_oqm_write",
        "streamingtom_oqm_select",
    ):
        row = _select_one(streamingtom_rows, variant=variant)
        short = variant.removeprefix("streamingtom_")
        streamingtom_metrics.extend(
            [
                _metric(
                    f"{short}_p50",
                    _finite(row["p50_ms"], label=f"{variant} P50"),
                    "milliseconds",
                ),
                _metric(
                    f"{short}_p95",
                    _finite(row["p95_ms"], label=f"{variant} P95"),
                    "milliseconds",
                ),
                _metric(
                    f"{short}_p99",
                    _finite(row["p99_ms"], label=f"{variant} P99"),
                    "milliseconds",
                ),
            ]
        )
    evidence.append(
        _evidence(
            evidence_id="streamingtom_official_core_triplet_200",
            method_id="streamingtom",
            component_id="official_ctr_oqm_core_latency",
            comparability_group="streamingtom_official_cuda_core_microbenchmarks",
            evidence_tier="official_core_gpu_microbenchmark",
            data_reuse="official_pinned_core_benchmark",
            sample_count=200,
            gate_id="official_core_summary_triplet_complete",
            gate_status="PASS",
            valid_for_positive_claim=True,
            metrics=streamingtom_metrics,
            source_path=streamingtom_rel,
            source_sha256=streamingtom_sha,
            notes=(
                "CTR, OQM write, and OQM select have different scopes and "
                "are neither additive nor end-to-end Video-LLM latency."
            ),
        )
    )

    oasis_rel = "paper/results/probe_mvp/oasis_official_smoke_20260719/result.json"
    oasis_path, oasis_sha = _artifact(repo_root, oasis_rel)
    oasis = _load_object(oasis_path)
    oasis_metrics = oasis.get("metrics")
    if oasis.get("status") != "complete" or not isinstance(oasis_metrics, dict):
        raise ValueError("committed OASIS smoke result is not complete")
    if oasis_metrics.get("complete") is not True or oasis_metrics.get("errors") != []:
        raise ValueError("committed OASIS smoke result failed integrity checks")
    evidence.append(
        _evidence(
            evidence_id="oasis_official_smoke_1x5",
            method_id="oasis",
            component_id="official_event_archive",
            comparability_group="oasis_official_smoke_1x5",
            evidence_tier="official_model_level_smoke",
            data_reuse="official_prefix_smoke",
            sample_count=_integer(oasis_metrics["scored_questions"], label="OASIS smoke questions"),
            gate_id="oasis_cuda_and_inference_smoke",
            gate_status="SMOKE_ONLY",
            valid_for_positive_claim=False,
            metrics=[
                _metric("accuracy", oasis_metrics["accuracy"], "fraction"),
                _metric(
                    "whole_run_wall_time",
                    oasis["run_record"]["elapsed_wall_seconds"],
                    "seconds",
                ),
                _metric(
                    "peak_process_memory_sampled",
                    oasis["run_record"]["gpu_monitor"]["gpu_peak_process_mib_sampled"],
                    "MiB",
                ),
            ],
            source_path=oasis_rel,
            source_sha256=oasis_sha,
            notes="One-video smoke validates BF16 CUDA and official inference only; wall time is offline pace=0 and not request latency.",
        )
    )
    return evidence


def base_completion_matrix() -> list[dict[str, str]]:
    statuses = {
        "ours": (
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "OPEN",
            "OPEN",
            "PASS",
            "PASS",
        ),
        "exact_recent": (
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "NA",
            "PASS",
            "PASS",
        ),
        "causalmem": (
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "NA",
            "PROXY_ONLY",
            "PASS",
        ),
        "stc": (
            "PASS",
            "PASS",
            "NA",
            "PASS",
            "NA",
            "PASS",
            "PROXY_ONLY",
            "PASS",
        ),
        "streamingtom": (
            "PASS",
            "PASS",
            "NA",
            "PASS",
            "NA",
            "PASS",
            "PROXY_ONLY",
            "PASS",
        ),
        "selectstream": (
            "UNAVAILABLE",
            "UNAVAILABLE",
            "NA",
            "UNAVAILABLE",
            "PAPER_ONLY",
            "PAPER_ONLY",
            "PROXY_ONLY",
            "OPEN",
        ),
        "oasis": (
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "PASS",
            "UNAVAILABLE",
            "PROXY_ONLY",
            "PASS",
        ),
        "statekv": (
            "PLACEHOLDER",
            "UNAVAILABLE",
            "NA",
            "UNAVAILABLE",
            "PAPER_ONLY",
            "PAPER_ONLY",
            "PROXY_ONLY",
            "OPEN",
        ),
    }
    details = {
        ("ours", "official_quality"): "No independent end-to-end streaming quality run.",
        ("ours", "official_latency"): "No native writer/read/TTFT tail-latency run.",
        ("ours", "independent_replication"): "Frozen 300-sample routed-state preservation gate passed; selector superiority remains open.",
        ("causalmem", "official_quality"): "Strict official 50x5 result: 206/250 correct.",
        ("causalmem", "independent_replication"): "Official evaluator reproduced with complete audited artifacts.",
        ("stc", "official_latency"): "Matched official ReKV/ReKV+STC ViT-plus-prefill stage pair completed.",
        ("stc", "independent_replication"): "Official stage benchmark pair reproduced; not end-to-end TTFT.",
        ("streamingtom", "runtime_preflight"): "Pinned CTR/OQM dry-run preflight triplet passed.",
        ("streamingtom", "official_latency"): "Official-core CTR/OQM CUDA summaries completed; scopes are non-additive.",
        ("streamingtom", "independent_replication"): "Pinned official core microbenchmarks reproduced; not model-level quality.",
        ("selectstream", "source"): "No discoverable public implementation.",
        ("oasis", "official_quality"): "Strict official 50x5 result: 209/250 correct.",
        ("oasis", "official_latency"): "No public paper-table request-latency runner.",
        ("oasis", "independent_replication"): "Official evaluator reproduced with complete audited artifacts.",
        ("statekv", "source"): "Official repository is a README placeholder.",
    }
    sources = {
        "ours": "paper/results/probe_mvp/mvbench_independent_replication_300_20260722/INDEPENDENT_REPLICATION_ANALYSIS.md",
        "exact_recent": "paper/results/probe_mvp/mvbench_independent_replication_300_20260722/aggregate/variant_summary.csv",
        "causalmem": "paper/results/probe_mvp/official_streaming_formal_20260722/official_quality_formal.csv",
        "stc": "paper/results/probe_mvp/official_streaming_formal_20260722/official_stc_stage_latency.csv",
        "streamingtom": "paper/results/probe_mvp/official_streaming_formal_20260722/official_streamingtom_core_latency.csv",
        "selectstream": "paper/results/probe_mvp/STREAMING_BASELINE_REPRODUCTION_AUDIT_20260719.md",
        "oasis": "paper/results/probe_mvp/official_streaming_formal_20260722/official_quality_formal.csv",
        "statekv": "paper/results/probe_mvp/STREAMING_BASELINE_REPRODUCTION_AUDIT_20260719.md",
    }
    method_labels = dict(METHODS)
    stage_labels = dict(STAGES)
    matrix = []
    for method_id, _ in METHODS:
        method_statuses = statuses[method_id]
        if len(method_statuses) != len(STAGES):
            raise ValueError(f"completion status width mismatch for {method_id}")
        for (stage, _), status in zip(STAGES, method_statuses, strict=True):
            matrix.append(
                {
                    "method_id": method_id,
                    "method_label": method_labels[method_id],
                    "stage": stage,
                    "stage_label": stage_labels[stage],
                    "status": status,
                    "detail": details.get((method_id, stage), ""),
                    "source_path": sources[method_id],
                }
            )
    return matrix


def apply_runtime_status(
    matrix: list[dict[str, str]], runtime_status_path: Path | None
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    if runtime_status_path is None:
        return matrix, None
    payload = _load_object(runtime_status_path)
    if payload.get("format_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("runtime status must use format_version=1 and contain records")
    observed_at = payload.get("observed_at")
    if not isinstance(observed_at, str):
        raise ValueError("runtime status lacks observed_at")
    try:
        datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid runtime observed_at: {observed_at}") from error
    index = {(row["method_id"], row["stage"]): row for row in matrix}
    seen: set[tuple[str, str]] = set()
    for record in payload["records"]:
        if not isinstance(record, dict):
            raise ValueError("runtime status records must be objects")
        key = (record.get("method_id"), record.get("stage"))
        if key not in index:
            raise ValueError(f"unknown runtime status target: {key}")
        if key in seen:
            raise ValueError(f"duplicate runtime status target: {key}")
        seen.add(key)
        status = record.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid runtime status for {key}: {status}")
        detail = record.get("detail", "")
        source_path = record.get("source_path", "")
        if not isinstance(detail, str) or not isinstance(source_path, str):
            raise ValueError(f"runtime status detail/source_path must be strings: {key}")
        index[key]["status"] = status
        index[key]["detail"] = detail
        if source_path:
            index[key]["source_path"] = source_path
    return matrix, {
        "path": str(runtime_status_path.resolve()),
        "sha256": _sha256(runtime_status_path),
        "observed_at": observed_at,
        "record_count": len(payload["records"]),
    }


def _flatten_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in evidence:
        for metric in entry["metrics"]:
            rows.append(
                {
                    "evidence_id": entry["evidence_id"],
                    "method_id": entry["method_id"],
                    "component_id": entry["component_id"],
                    "comparability_group": entry["comparability_group"],
                    "evidence_tier": entry["evidence_tier"],
                    "data_reuse": entry["data_reuse"],
                    "sample_count": entry["sample_count"],
                    "gate_id": entry["gate"]["gate_id"],
                    "gate_status": entry["gate"]["status"],
                    "valid_for_positive_claim": entry["gate"][
                        "valid_for_positive_claim"
                    ],
                    "metric_name": metric["name"],
                    "metric_value": metric["value"],
                    "metric_unit": metric["unit"],
                    "ci95_low": metric["ci95_low"],
                    "ci95_high": metric["ci95_high"],
                    "reference": metric["reference"],
                    "source_path": entry["source"]["path"],
                    "source_sha256": entry["source"]["sha256"],
                    "notes": entry["notes"],
                }
            )
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_matrix(matrix: list[dict[str, str]], output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": False,
        }
    )
    index = {(row["method_id"], row["stage"]): row for row in matrix}
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    for y, (method_id, _) in enumerate(METHODS):
        for x, (stage, _) in enumerate(STAGES):
            status = index[(method_id, stage)]["status"]
            color, hatch, label = STATUS_STYLE[status]
            rectangle = Rectangle(
                (x - 0.5, y - 0.5),
                1.0,
                1.0,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch,
            )
            ax.add_patch(rectangle)
            text_color = "white" if status in {"UNAVAILABLE", "RUNNING"} else "black"
            ax.text(x, y, label, ha="center", va="center", color=text_color, fontsize=7.5)
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    ax.set_ylim(len(METHODS) - 0.5, -0.5)
    ax.set_xticks(range(len(STAGES)), [label for _, label in STAGES], rotation=28, ha="right")
    ax.set_yticks(range(len(METHODS)), [label for _, label in METHODS])
    ax.tick_params(length=0)
    ax.set_xlabel("Evidence stage")
    ax.set_ylabel("Method or control")
    observed = {row["status"] for row in matrix}
    handles = [
        Patch(
            facecolor=STATUS_STYLE[status][0],
            edgecolor="black",
            hatch=STATUS_STYLE[status][1],
            label=status.replace("_", " ").title(),
        )
        for status in STATUS_STYLE
        if status in observed
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _repository_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_report(
    path: Path,
    *,
    evidence: list[dict[str, Any]],
    matrix: list[dict[str, str]],
    runtime: dict[str, Any] | None,
) -> None:
    ours = [entry for entry in evidence if entry["method_id"] == "ours" and entry["component_id"] != "unified_feature_proxy"]
    status_counts = Counter(row["status"] for row in matrix)
    lines = [
        "# Online-Video Evidence Completion Matrix",
        "",
        "## Claim Decision",
        "",
        "The frozen routed codec now passes an independent fixed-state "
        "preservation gate. No complete hybrid method has yet passed the "
        "combined end-to-end streaming quality, latency, and SLO gates; "
        "comparability groups remain separate.",
        "",
        "## Our Component Gates",
        "",
        "| Component | Status | Evidence tier | Primary metric |",
        "|---|---:|---|---|",
    ]
    for entry in ours:
        metric = entry["metrics"][0]
        lines.append(
            f"| {entry['component_id']} | {entry['gate']['status']} | "
            f"{entry['evidence_tier']} | {metric['name']}={metric['value']:.6g} {metric['unit']} |"
        )
    lines.extend(
        [
            "",
            "## Completion Status",
            "",
            ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())),
            "",
        ]
    )
    if runtime is not None:
        lines.extend(
            [
                f"Runtime snapshot: `{runtime['observed_at']}` with {runtime['record_count']} overrides.",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "The current evidence supports independent routed-state "
            "preservation on the frozen LLaVA MVBench reserve. It does not "
            "justify claims that BCCB replaces global video attention, that "
            "the error-oracle route is a cheap semantic event detector, or "
            "that proxy/stage latency is official TTFT or SLO latency. "
            "SelectStream and StateKV remain paper/proxy references until "
            "executable official code is available.",
            "",
            "The matrix figure is `streaming_evidence_completion_matrix.png`/`.pdf`; raw rows are preserved in `completion_matrix.csv` and `evidence_metrics.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_matrix(
    *,
    repo_root: Path,
    out_dir: Path,
    runtime_status_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    evidence = build_evidence(repo_root)
    matrix, runtime = apply_runtime_status(
        base_completion_matrix(),
        runtime_status_path.resolve() if runtime_status_path is not None else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "evidence_metrics.csv", EVIDENCE_FIELDS, _flatten_evidence(evidence))
    _write_csv(out_dir / "completion_matrix.csv", MATRIX_FIELDS, matrix)
    _plot_matrix(matrix, out_dir / "streaming_evidence_completion_matrix")
    _write_report(
        out_dir / "EVIDENCE_MATRIX_ANALYSIS.md",
        evidence=evidence,
        matrix=matrix,
        runtime=runtime,
    )
    summary = {
        "format_version": "evidence-completion/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _repository_commit(repo_root),
        "runtime_status": runtime,
        "evidence_count": len(evidence),
        "metric_row_count": len(_flatten_evidence(evidence)),
        "completion_cell_count": len(matrix),
        "status_counts": dict(sorted(Counter(row["status"] for row in matrix).items())),
        "evidence": evidence,
        "completion_matrix": matrix,
        "cautions": [
            "Rows from different comparability_group values must not be ranked together.",
            "PROXY_ONLY, PAPER_ONLY, SMOKE_ONLY, RUNNING, and QUEUED are not completed formal comparisons.",
            "A failed or open component gate prevents a positive claim for the complete hybrid method.",
        ],
    }
    (out_dir / "evidence_matrix.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = build_matrix(
        repo_root=args.repo_root,
        out_dir=args.out_dir.resolve(),
        runtime_status_path=args.runtime_status,
    )
    print(
        json.dumps(
            {
                "evidence_count": summary["evidence_count"],
                "metric_row_count": summary["metric_row_count"],
                "completion_cell_count": summary["completion_cell_count"],
                "status_counts": summary["status_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
