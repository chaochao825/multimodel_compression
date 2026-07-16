from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FILES = (
    "split_manifest.csv",
    "predictor_results.csv",
    "vq_results.csv",
    "controller_results.csv",
    "combined_results.csv",
    "summary.json",
    "RESULT_SUMMARY.md",
    "component_verdicts.json",
    "combined_candidate_points.csv",
    "STREAMING_HYBRID_STATE_V0_ANALYSIS.md",
)

PREDICTORS = {
    "previous",
    "ema_025",
    "ema_050",
    "ema_075",
    "linear",
    "fourier_h4_k1",
    "fourier_h8_k2",
}

FIXED_POLICIES = {
    "always_reuse",
    "always_predict",
    "always_innovation",
    "always_int4_refresh",
}

NUMERIC_METRICS = {
    "mse",
    "nmse",
    "mean_cosine",
    "p05_cosine",
    "payload_bps",
    "effective_bps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent audit for the streaming hybrid-state probe."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric field {key!r}: {row}") from error


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(
        self,
        condition: bool,
        *,
        name: str,
        severity: str = "major",
        detail: str,
    ) -> None:
        self.checks.append(
            {
                "name": name,
                "status": "PASS" if condition else "FAIL",
                "severity": severity,
                "detail": detail,
            }
        )

    @property
    def major_failures(self) -> list[dict[str, Any]]:
        return [
            check
            for check in self.checks
            if check["status"] == "FAIL" and check["severity"] == "major"
        ]

    @property
    def minor_failures(self) -> list[dict[str, Any]]:
        return [
            check
            for check in self.checks
            if check["status"] == "FAIL" and check["severity"] == "minor"
        ]


def finite_rows(
    rows: Iterable[dict[str, str]],
    fields: Iterable[str],
) -> bool:
    for row in rows:
        for field in fields:
            if field in row and row[field] != "":
                if not math.isfinite(float(row[field])):
                    return False
    return True


def audit_split(audit: Audit, rows: list[dict[str, str]]) -> None:
    by_split = defaultdict(set)
    by_category_split: dict[tuple[str, str], int] = Counter()
    for row in rows:
        by_split[row["split"]].add(row["run"])
        by_category_split[(row["category"], row["split"])] += 1
    overlap = (
        (by_split["train"] & by_split["val"])
        | (by_split["train"] & by_split["test"])
        | (by_split["val"] & by_split["test"])
    )
    audit.check(
        not overlap,
        name="clip_split_disjoint",
        detail=f"overlapping runs: {sorted(overlap)}",
    )
    categories = sorted({row["category"] for row in rows})
    expected = all(
        (
            by_category_split[(category, "train")],
            by_category_split[(category, "val")],
            by_category_split[(category, "test")],
        )
        == (3, 1, 2)
        for category in categories
    )
    counts = {
        category: {
            split: by_category_split[(category, split)]
            for split in ("train", "val", "test")
        }
        for category in categories
    }
    audit.check(
        expected,
        name="stratified_3_1_2_split",
        detail=json.dumps(counts, sort_keys=True),
    )


def audit_predictors(
    audit: Audit,
    rows: list[dict[str, str]],
    layers: list[int],
) -> None:
    audit.check(
        finite_rows(
            rows,
            (
                "mse",
                "nmse",
                "mean_cosine",
                "p05_cosine",
                "residual_top10_energy",
                "raw_temporal_spectral_entropy",
                "residual_temporal_spectral_entropy",
                "spectral_entropy_reduction",
                "ops_per_scalar_proxy",
            ),
        ),
        name="predictor_metrics_finite",
        detail=f"checked {len(rows)} predictor rows",
    )
    complete = True
    for layer in layers:
        for split in ("val", "test"):
            observed = {
                row["predictor"]
                for row in rows
                if int(row["layer"]) == layer and row["split"] == split
            }
            complete &= observed == PREDICTORS
    audit.check(
        complete,
        name="predictor_ablation_complete",
        detail=f"expected predictors: {sorted(PREDICTORS)}",
    )
    nontrivial_entropy = any(
        as_float(row, "residual_temporal_spectral_entropy") > 0.0
        for row in rows
    )
    audit.check(
        nontrivial_entropy,
        name="temporal_entropy_not_degenerate",
        severity="minor",
        detail="at least one residual time-series entropy must be nonzero",
    )


def audit_vq(
    audit: Audit,
    rows: list[dict[str, str]],
    layers: list[int],
) -> None:
    audit.check(
        finite_rows(rows, NUMERIC_METRICS),
        name="vq_metrics_finite",
        detail=f"checked {len(rows)} VQ/scalar rows",
    )
    methods = {row["method"] for row in rows}
    audit.check(
        {"raw_pq", "residual_pq", "scalar_quant"} <= methods,
        name="vq_baselines_complete",
        detail=f"observed methods: {sorted(methods)}",
    )
    int4_layers = {
        int(row["layer"])
        for row in rows
        if row["split"] == "test"
        and row["method"] == "scalar_quant"
        and row["codec"] == "int4"
    }
    audit.check(
        int4_layers == set(layers),
        name="int4_baseline_present",
        detail=f"INT4 test layers: {sorted(int4_layers)}",
    )
    nominal_points = {
        round(as_float(row, "nominal_bps"), 3)
        for row in rows
        if row["split"] == "test" and row["method"] == "raw_pq"
    }
    audit.check(
        {0.5, 1.0, 1.5, 2.0, 4.0} <= nominal_points,
        name="multi_bit_pq_sweep_present",
        detail=f"raw PQ nominal points: {sorted(nominal_points)}",
    )
    pq_rows = [row for row in rows if row["method"] != "scalar_quant"]
    complete_accounting = all(
        as_float(row, "codebook_static_bits") > 0.0
        and as_float(row, "effective_bps")
        >= as_float(row, "payload_bps") - 1e-12
        for row in pq_rows
    )
    audit.check(
        complete_accounting,
        name="vq_static_and_metadata_accounted",
        detail="effective_bps must include nonzero codebook static bits",
    )
    hashes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in pq_rows:
        hashes[(row["method"], row["codec"], row["layer"])].add(
            row["codebook_sha256"]
        )
    stable = all(
        len(values) == 1 and "" not in values for values in hashes.values()
    )
    audit.check(
        stable,
        name="codebook_hash_stable_across_val_test",
        detail=f"checked {len(hashes)} method/codec/layer codebooks",
    )


def audit_controllers(
    audit: Audit,
    rows: list[dict[str, str]],
    combined: list[dict[str, str]],
    *,
    layers: list[int],
    torch_enabled: bool,
) -> None:
    learned_expected = {"threshold", "decision_tree"}
    if torch_enabled:
        learned_expected |= {"mlp", "dlgn"}
    observed = {row["controller"] for row in rows}
    audit.check(
        learned_expected <= observed,
        name="controller_ablation_complete",
        detail=f"observed learned controllers: {sorted(observed)}",
    )
    budgets = {
        round(as_float(row, "budget_bps"), 2)
        for row in rows
    }
    audit.check(
        {0.5, 1.0, 1.58, 2.0, 4.0} <= budgets,
        name="controller_budget_sweep_complete",
        detail=f"observed budgets: {sorted(budgets)}",
    )
    if torch_enabled:
        dlgn_rows = [row for row in rows if row["controller"] == "dlgn"]
        hardened = bool(dlgn_rows) and all(
            row.get("val_hard_accuracy", "") != ""
            and row.get("val_soft_accuracy", "") != ""
            and row.get("val_discretization_gap", "") != ""
            and row.get("val_soft_hard_agreement", "") != ""
            for row in dlgn_rows
        )
        audit.check(
            hardened,
            name="dlgn_hard_metrics_reported",
            detail=f"checked {len(dlgn_rows)} hardened DLGN rows",
        )
        mlp_rows = [row for row in rows if row["controller"] == "mlp"]
        mlp_accounting = bool(mlp_rows) and all(
            as_float(row, "controller_static_bits")
            == 32.0
            * (
                as_float(row, "parameter_count")
                + as_float(row, "normalizer_parameter_count")
            )
            for row in mlp_rows
        )
        audit.check(
            mlp_accounting,
            name="mlp_normalizer_state_accounted",
            detail=f"checked {len(mlp_rows)} MLP rows",
        )
    tree_rows = [row for row in rows if row["controller"] == "decision_tree"]
    tree_accounting = bool(tree_rows) and all(
        as_float(row, "controller_static_bits")
        == (
            ((int(as_float(row, "tree_nodes")) - 1) // 2) * 19
            + (((int(as_float(row, "tree_nodes")) - 1) // 2) + 1) * 2
            + int(as_float(row, "tree_nodes"))
        )
        for row in tree_rows
    )
    audit.check(
        tree_accounting,
        name="tree_topology_bits_accounted",
        detail=f"checked {len(tree_rows)} decision-tree rows",
    )
    combined_controllers = {row["controller"] for row in combined}
    audit.check(
        FIXED_POLICIES <= combined_controllers,
        name="fixed_policy_baselines_present",
        detail=f"observed policies: {sorted(combined_controllers)}",
    )
    rate_sums_ok = True
    accounting_ok = True
    exact_effective_accounting = True
    for row in combined:
        rate_sum = sum(
            as_float(row, key)
            for key in (
                "reuse_rate",
                "predict_rate",
                "innovation_rate",
                "refresh_rate",
            )
        )
        rate_sums_ok &= abs(rate_sum - 1.0) < 1e-6
        accounting_ok &= (
            as_float(row, "effective_bps")
            >= as_float(row, "payload_bps") - 1e-12
        )
        scalar_count = as_float(row, "evaluated_scalars")
        expected_effective = as_float(row, "payload_bps") + (
            as_float(row, "controller_static_bits")
            + as_float(row, "codebook_static_bits")
        ) / scalar_count
        exact_effective_accounting &= (
            abs(as_float(row, "effective_bps") - expected_effective) < 1e-9
        )
    audit.check(
        rate_sums_ok,
        name="combined_action_rates_sum_to_one",
        detail=f"checked {len(combined)} combined rows",
    )
    audit.check(
        accounting_ok,
        name="combined_static_and_action_bits_accounted",
        detail="effective_bps must be no smaller than stream payload_bps",
    )
    audit.check(
        exact_effective_accounting,
        name="combined_effective_bps_recomputes",
        detail="effective_bps exactly amortizes codebook and controller state",
    )
    encoder_rates_ok = all(
        abs(
            (
                as_float(row, "innovation_rate")
                + as_float(row, "refresh_rate")
            )
            + (
                as_float(row, "reuse_rate")
                + as_float(row, "predict_rate")
            )
            - 1.0
        )
        < 1e-6
        for row in combined
    )
    audit.check(
        encoder_rates_ok,
        name="encoder_required_rate_contract",
        detail=(
            "innovation+refresh requires current hidden state; "
            "reuse+predict is the encoder-skip fraction"
        ),
    )
    covered_layers = {int(row["layer"]) for row in combined}
    audit.check(
        covered_layers == set(layers),
        name="combined_layers_complete",
        detail=f"combined layers: {sorted(covered_layers)}",
    )


def audit_source_and_claims(
    audit: Audit,
    source_root: Path,
    result_summary: str,
    analysis_report: str,
) -> None:
    evaluate_source = (
        source_root / "streaming_hybrid_state" / "evaluate.py"
    ).read_text(encoding="utf-8")
    core_source = (
        source_root / "streaming_hybrid_state" / "core.py"
    ).read_text(encoding="utf-8")
    causal_features = (
        "rgb_summary_features(clip.frames_rgb)" in evaluate_source
        and "controller.predict(features[1:])" in evaluate_source
        and "changed_features[:4]" in (
            source_root
            / "streaming_hybrid_state"
            / "tests"
            / "test_core.py"
        ).read_text(encoding="utf-8")
    )
    audit.check(
        causal_features,
        name="controller_uses_causal_rgb_features",
        detail="source and regression test use current/previous RGB summaries",
    )
    train_only_codebook = (
        'runs=splits["train"]' in evaluate_source
        and "collect_group_samples(" in evaluate_source
    )
    audit.check(
        train_only_codebook,
        name="codebook_fit_uses_train_split",
        detail="codebook samples are selected from train clips",
    )
    fp16_consistent = (
        "centroids.astype(np.float16).astype(np.float32)" in core_source
        and "scales.astype(np.float16).astype(np.float32)" in core_source
        and "threshold=float(np.float16(threshold))" in core_source
        and ".to(torch.float16)" in evaluate_source
    )
    audit.check(
        fp16_consistent,
        name="fp16_cost_matches_parameter_precision",
        detail=(
            "codebooks, scalar scales, tree thresholds, and DLGN thresholds "
            "are numerically rounded to the precision used in bit accounting"
        ),
    )
    controller_description_complete = (
        "topology_bits = tree.node_count()" in evaluate_source
        and "normalizer_params = int(mean.numel() + std.numel())"
        in evaluate_source
        and "static_bits=(mlp_params + normalizer_params) * 32"
        in evaluate_source
    )
    audit.check(
        controller_description_complete,
        name="controller_static_description_complete",
        detail="tree topology and MLP normalization state are counted",
    )
    open_loop = (
        "prediction = predict_next(decoded, method, params)" in core_source
    )
    audit.check(
        open_loop,
        name="combined_codec_is_open_loop",
        detail="predictor history is reconstructed state, not target state",
    )
    all_claims = result_summary + "\n" + analysis_report
    boundary_phrases = (
        "does not claim end-to-end Video-LLM accuracy" in result_summary
        and "No ViT skip" in result_summary
        and "effective_bps" in result_summary
        and "encoder_required_rate" in analysis_report
        and "Conditional visual compute" in analysis_report
    )
    audit.check(
        boundary_phrases,
        name="claim_boundary_explicit",
        detail="summary separates representation quality from task/PPA claims",
    )
    prohibited = (
        "PPA improved",
        "area reduced",
        "power reduced",
        "timing improved",
        "end-to-end accuracy improved",
    )
    unsupported = [phrase for phrase in prohibited if phrase in all_claims]
    audit.check(
        not unsupported,
        name="no_unsupported_positive_claim",
        detail=f"unsupported phrases: {unsupported}",
    )


def audit_derived_verdicts(
    audit: Audit,
    verdicts: dict[str, Any],
    candidates: list[dict[str, str]],
    layers: list[int],
) -> None:
    expected_keys = {
        "predictor",
        "residual_vq",
        "logic_controller",
        "combined_memory",
        "conditional_compute",
        "end_to_end_task",
    }
    audit.check(
        expected_keys <= set(verdicts),
        name="component_verdicts_complete",
        detail=f"observed verdict keys: {sorted(verdicts)}",
    )
    memory_layers = {
        int(row["layer"])
        for row in candidates
        if row["memory_representation_candidate"] == "True"
    }
    compute_layers = {
        int(row["layer"])
        for row in candidates
        if row["conditional_compute_candidate"] == "True"
    }

    def expected(observed_layers: set[int]) -> str:
        if observed_layers == set(layers):
            return "Positive"
        if observed_layers:
            return "Mixed"
        return "Negative"

    memory_consistent = (
        verdicts["combined_memory"]["verdict"] == expected(memory_layers)
        and set(verdicts["combined_memory"]["qualified_layers"])
        == memory_layers
    )
    compute_consistent = (
        verdicts["conditional_compute"]["verdict"]
        == expected(compute_layers)
        and set(verdicts["conditional_compute"]["qualified_layers"])
        == compute_layers
    )
    audit.check(
        memory_consistent,
        name="memory_verdict_matches_candidate_rows",
        detail=(
            f"candidate layers={sorted(memory_layers)}, "
            f"verdict={verdicts['combined_memory']['verdict']}"
        ),
    )
    audit.check(
        compute_consistent,
        name="compute_verdict_matches_encoder_gate",
        detail=(
            f"candidate layers={sorted(compute_layers)}, "
            f"verdict={verdicts['conditional_compute']['verdict']}"
        ),
    )
    encoder_contract = all(
        abs(
            as_float(row, "encoder_required_rate")
            - (
                as_float(row, "innovation_rate")
                + as_float(row, "refresh_rate")
            )
        )
        < 1e-6
        and abs(
            as_float(row, "encoder_skip_rate")
            - (
                as_float(row, "reuse_rate")
                + as_float(row, "predict_rate")
            )
        )
        < 1e-6
        for row in candidates
    )
    audit.check(
        encoder_contract,
        name="derived_encoder_rates_exact",
        detail=f"checked {len(candidates)} learned policy rows",
    )


def render_report(audit: Audit) -> str:
    overall = "PASS" if not audit.major_failures else "FAIL"
    lines = [
        "# Streaming Hybrid State V0 Review Report",
        "",
        f"## Overall: {overall}",
        "",
        "This report is produced by an independent result-audit script. "
        "The reviewer does not train models or select operating points.",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Detail |",
        "|---|---|---|---|",
    ]
    for check in audit.checks:
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {check['name']} | {check['status']} | "
            f"{check['severity']} | {detail} |"
        )
    lines.extend(["", "## Major Issues", ""])
    if audit.major_failures:
        lines.extend(
            f"- {check['name']}: {check['detail']}"
            for check in audit.major_failures
        )
    else:
        lines.append("- None.")
    lines.extend(["", "## Minor Issues", ""])
    if audit.minor_failures:
        lines.extend(
            f"- {check['name']}: {check['detail']}"
            for check in audit.minor_failures
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- PASS means the probe is internally consistent and auditable.",
            "- It does not imply end-to-end Video-LLM quality, encoder speedup, "
            "or hardware PPA improvement.",
            "- Component and combined scientific verdicts must still follow "
            "the measured held-out rows and stated kill criteria.",
            "",
            "## Recommended Next Step",
            "",
            "Promote only components that beat their matched simple baseline. "
            "Any promising combined point should next be tested on a task-level "
            "streaming benchmark before RTL or PPA claims are added.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    audit = Audit()
    missing = [
        name for name in REQUIRED_FILES if not (args.results_dir / name).exists()
    ]
    audit.check(
        not missing,
        name="required_outputs_present",
        detail=f"missing files: {missing}",
    )
    if missing:
        report = render_report(audit)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / "review_report.md").write_text(
            report,
            encoding="utf-8",
        )
        return 1

    split_rows = read_csv(args.results_dir / "split_manifest.csv")
    predictor_rows = read_csv(args.results_dir / "predictor_results.csv")
    vq_rows = read_csv(args.results_dir / "vq_results.csv")
    controller_rows = read_csv(args.results_dir / "controller_results.csv")
    combined_rows = read_csv(args.results_dir / "combined_results.csv")
    summary = json.loads(
        (args.results_dir / "summary.json").read_text(encoding="utf-8")
    )
    result_summary = (
        args.results_dir / "RESULT_SUMMARY.md"
    ).read_text(encoding="utf-8")
    analysis_report = (
        args.results_dir / "STREAMING_HYBRID_STATE_V0_ANALYSIS.md"
    ).read_text(encoding="utf-8")
    verdicts = json.loads(
        (args.results_dir / "component_verdicts.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_rows = read_csv(
        args.results_dir / "combined_candidate_points.csv"
    )
    layers = [int(layer) for layer in summary["layers"]]

    audit_split(audit, split_rows)
    audit_predictors(audit, predictor_rows, layers)
    audit_vq(audit, vq_rows, layers)
    audit_controllers(
        audit,
        controller_rows,
        combined_rows,
        layers=layers,
        torch_enabled=bool(summary["torch_controllers"]),
    )
    audit_derived_verdicts(audit, verdicts, candidate_rows, layers)
    audit_source_and_claims(
        audit,
        args.source_root,
        result_summary,
        analysis_report,
    )

    findings = {
        "overall": "PASS" if not audit.major_failures else "FAIL",
        "major_issue_count": len(audit.major_failures),
        "minor_issue_count": len(audit.minor_failures),
        "checks": audit.checks,
    }
    (args.results_dir / "review_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.results_dir / "review_report.md").write_text(
        render_report(audit),
        encoding="utf-8",
    )
    print(json.dumps(findings, indent=2, sort_keys=True))
    return 0 if not audit.major_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
