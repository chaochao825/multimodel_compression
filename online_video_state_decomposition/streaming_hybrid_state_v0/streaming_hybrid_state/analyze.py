from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


BUDGETS = (0.50, 1.00, 1.58, 2.00, 4.00)
SIMPLE_PREDICTORS = {
    "previous",
    "ema_025",
    "ema_050",
    "ema_075",
    "linear",
}
FOURIER_PREDICTORS = {"fourier_h4_k1", "fourier_h8_k2"}
LEARNED_CONTROLLERS = {
    "threshold",
    "decision_tree",
    "mlp",
    "dlgn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive held-out component and combined verdicts."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def vq_key(row: dict[str, str]) -> tuple[str, str, float]:
    return (
        row["method"],
        row["codec"],
        round(number(row, "selected_fraction"), 8),
    )


def selected_vq_test_rows(
    rows: Sequence[dict[str, str]],
    *,
    layer: int,
    method: str,
    budget: float,
) -> dict[str, str] | None:
    validation = [
        row
        for row in rows
        if int(row["layer"]) == layer
        and row["split"] == "val"
        and row["method"] == method
        and number(row, "nominal_bps") <= budget + 1e-9
    ]
    if not validation:
        return None
    selected = max(
        validation,
        key=lambda row: (
            number(row, "mean_cosine"),
            -number(row, "payload_bps"),
            -number(row, "effective_bps"),
        ),
    )
    key = vq_key(selected)
    return next(
        row
        for row in rows
        if int(row["layer"]) == layer
        and row["split"] == "test"
        and vq_key(row) == key
    )


def predictor_analysis(
    rows: Sequence[dict[str, str]],
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    wins = 0
    for layer in layers:
        validation = [
            row
            for row in rows
            if int(row["layer"]) == layer and row["split"] == "val"
        ]
        best_simple_val = min(
            (row for row in validation if row["predictor"] in SIMPLE_PREDICTORS),
            key=lambda row: (number(row, "nmse"), row["predictor"]),
        )
        best_fourier_val = min(
            (row for row in validation if row["predictor"] in FOURIER_PREDICTORS),
            key=lambda row: (number(row, "nmse"), row["predictor"]),
        )

        def test_row(name: str) -> dict[str, str]:
            return next(
                row
                for row in rows
                if int(row["layer"]) == layer
                and row["split"] == "test"
                and row["predictor"] == name
            )

        simple = test_row(best_simple_val["predictor"])
        fourier = test_row(best_fourier_val["predictor"])
        fourier_win = number(fourier, "nmse") < number(simple, "nmse")
        wins += int(fourier_win)
        output.append(
            {
                "layer": layer,
                "best_simple": simple["predictor"],
                "simple_test_nmse": number(simple, "nmse"),
                "simple_test_cosine": number(simple, "mean_cosine"),
                "simple_ops_per_scalar": number(
                    simple,
                    "ops_per_scalar_proxy",
                ),
                "best_fourier": fourier["predictor"],
                "fourier_test_nmse": number(fourier, "nmse"),
                "fourier_test_cosine": number(fourier, "mean_cosine"),
                "fourier_ops_per_scalar": number(
                    fourier,
                    "ops_per_scalar_proxy",
                ),
                "fourier_nmse_change_vs_simple": (
                    number(fourier, "nmse") / number(simple, "nmse") - 1.0
                ),
                "fourier_win": fourier_win,
                "simple_raw_spectral_entropy": number(
                    simple,
                    "raw_temporal_spectral_entropy",
                ),
                "simple_residual_spectral_entropy": number(
                    simple,
                    "residual_temporal_spectral_entropy",
                ),
                "simple_spectral_entropy_reduction": number(
                    simple,
                    "spectral_entropy_reduction",
                ),
            }
        )
    verdict = (
        "Positive"
        if wins == len(layers)
        else "Mixed"
        if wins
        else "Negative"
    )
    return output, {
        "verdict": verdict,
        "fourier_wins": wins,
        "layer_count": len(layers),
        "rule": (
            "Fourier is positive only when validation-selected Fourier "
            "beats the validation-selected simple causal predictor on test NMSE."
        ),
    }


def vq_analysis(
    rows: Sequence[dict[str, str]],
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rate_rows = []
    for layer in layers:
        for budget in BUDGETS:
            for method in ("raw_pq", "residual_pq", "scalar_quant"):
                selected = selected_vq_test_rows(
                    rows,
                    layer=layer,
                    method=method,
                    budget=budget,
                )
                if selected is None:
                    continue
                rate_rows.append(
                    {
                        "layer": layer,
                        "nominal_budget_bps": budget,
                        "method": method,
                        "codec": selected["codec"],
                        "selected_fraction": number(
                            selected,
                            "selected_fraction",
                        ),
                        "nominal_bps": number(selected, "nominal_bps"),
                        "payload_bps": number(selected, "payload_bps"),
                        "effective_bps": number(selected, "effective_bps"),
                        "mean_cosine": number(selected, "mean_cosine"),
                        "p05_cosine": number(selected, "p05_cosine"),
                        "nmse": number(selected, "nmse"),
                        "index_entropy_bps": selected["index_entropy_bps"],
                    }
                )

    entropy_rows = []
    reductions = []
    for layer in layers:
        raw_lookup = {
            row["codec"]: row
            for row in rows
            if int(row["layer"]) == layer
            and row["split"] == "test"
            and row["method"] == "raw_pq"
        }
        residual_full = [
            row
            for row in rows
            if int(row["layer"]) == layer
            and row["split"] == "test"
            and row["method"] == "residual_pq"
            and abs(number(row, "selected_fraction") - 1.0) < 1e-9
        ]
        for residual in residual_full:
            raw = raw_lookup[residual["codec"]]
            raw_entropy = number(raw, "index_entropy_bps")
            residual_entropy = number(residual, "index_entropy_bps")
            reduction = (
                1.0 - residual_entropy / raw_entropy
                if raw_entropy > 0.0
                else 0.0
            )
            reductions.append(reduction)
            entropy_rows.append(
                {
                    "layer": layer,
                    "codec": residual["codec"],
                    "raw_index_entropy_bps": raw_entropy,
                    "residual_index_entropy_bps": residual_entropy,
                    "entropy_reduction": reduction,
                    "raw_payload_bps": number(raw, "payload_bps"),
                    "residual_payload_bps": number(
                        residual,
                        "payload_bps",
                    ),
                    "raw_cosine": number(raw, "mean_cosine"),
                    "residual_cosine": number(
                        residual,
                        "mean_cosine",
                    ),
                }
            )
    over_thirty = sum(reduction > 0.30 for reduction in reductions)
    verdict = (
        "Positive"
        if reductions and over_thirty == len(reductions)
        else "Mixed"
        if over_thirty
        else "Negative"
    )
    return rate_rows, entropy_rows, {
        "verdict": verdict,
        "entropy_reduction_over_30pct": over_thirty,
        "matched_rows": len(reductions),
        "rule": (
            "Residual VQ is retained when matched full-PQ index entropy falls "
            "by more than 30%, or when it improves held-out distortion at a "
            "comparable reported stream rate. This probe does not test task accuracy."
        ),
    }


def controller_analysis(
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, float], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["layer"]), number(row, "budget_bps"))][
            row["controller"]
        ] = row
    output = []
    near_mlp = 0
    comparisons = 0
    for (layer, budget), local in sorted(grouped.items()):
        if "dlgn" not in local or "mlp" not in local:
            continue
        dlgn = local["dlgn"]
        mlp = local["mlp"]
        gap = number(dlgn, "val_discretization_gap")
        near = (
            number(dlgn, "balanced_accuracy")
            >= number(mlp, "balanced_accuracy") - 0.02
            and number(dlgn, "controller_static_bits")
            < number(mlp, "controller_static_bits")
            and abs(gap) <= 0.02
        )
        comparisons += 1
        near_mlp += int(near)
        output.append(
            {
                "layer": layer,
                "budget_bps": budget,
                "mlp_accuracy": number(mlp, "accuracy"),
                "mlp_balanced_accuracy": number(
                    mlp,
                    "balanced_accuracy",
                ),
                "mlp_static_bits": number(mlp, "controller_static_bits"),
                "dlgn_accuracy": number(dlgn, "accuracy"),
                "dlgn_balanced_accuracy": number(
                    dlgn,
                    "balanced_accuracy",
                ),
                "dlgn_static_bits": number(
                    dlgn,
                    "controller_static_bits",
                ),
                "dlgn_gate_count": dlgn.get("gate_count", ""),
                "dlgn_val_discretization_gap": gap,
                "dlgn_near_mlp_pareto": near,
            }
        )
    ratio = near_mlp / comparisons if comparisons else 0.0
    verdict = (
        "Positive"
        if comparisons and ratio >= 0.8
        else "Mixed"
        if near_mlp
        else "Negative"
    )
    return output, {
        "verdict": verdict,
        "dlgn_near_mlp_rows": near_mlp,
        "comparison_rows": comparisons,
        "rule": (
            "DLGN enters the provisional Pareto set only when test balanced "
            "accuracy is within 0.02 of MLP, its static description is smaller, "
            "and validation soft-hard gap is at most 0.02."
        ),
    }


def combined_analysis(
    rows: Sequence[dict[str, str]],
    vq_rows: Sequence[dict[str, str]],
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    int4_baseline: dict[tuple[int, float], dict[str, str]] = {}
    for row in rows:
        if row["controller"] == "always_int4_refresh":
            int4_baseline[(int(row["layer"]), number(row, "budget_bps"))] = row
    candidates = []
    memory_qualified_layers = set()
    compute_qualified_layers = set()
    max_predict_rate = 0.0
    for row in rows:
        controller = row["controller"]
        if controller not in LEARNED_CONTROLLERS:
            continue
        layer = int(row["layer"])
        budget = number(row, "budget_bps")
        baseline = int4_baseline[(layer, budget)]
        encoder_required_rate = (
            number(row, "innovation_rate") + number(row, "refresh_rate")
        )
        encoder_skip_rate = (
            number(row, "reuse_rate") + number(row, "predict_rate")
        )
        max_predict_rate = max(max_predict_rate, number(row, "predict_rate"))
        standalone_points = [
            baseline_row
            for baseline_row in vq_rows
            if baseline_row["split"] == "test"
            and int(baseline_row["layer"]) == layer
        ]
        dominating_points = [
            baseline_row
            for baseline_row in standalone_points
            if number(baseline_row, "effective_bps")
            <= number(row, "effective_bps") + 1e-12
            and number(baseline_row, "mean_cosine")
            >= number(row, "mean_cosine") - 1e-12
            and (
                number(baseline_row, "effective_bps")
                < number(row, "effective_bps") - 1e-12
                or number(baseline_row, "mean_cosine")
                > number(row, "mean_cosine") + 1e-12
            )
        ]
        rate_quality_dominated = bool(dominating_points)
        best_dominator = (
            min(
                dominating_points,
                key=lambda baseline_row: (
                    number(baseline_row, "effective_bps"),
                    -number(baseline_row, "mean_cosine"),
                ),
            )
            if dominating_points
            else None
        )
        memory_qualified = (
            number(row, "mean_cosine") >= 0.95
            and number(row, "effective_bps")
            < number(baseline, "effective_bps")
            and not rate_quality_dominated
        )
        compute_qualified = (
            memory_qualified and encoder_required_rate < 0.30
        )
        if memory_qualified:
            memory_qualified_layers.add(layer)
        if compute_qualified:
            compute_qualified_layers.add(layer)
        candidates.append(
            {
                "layer": layer,
                "budget_bps": budget,
                "controller": controller,
                "predictor": row["predictor"],
                "innovation_codec": row["innovation_codec"],
                "innovation_fraction": number(
                    row,
                    "innovation_fraction",
                ),
                "mean_cosine": number(row, "mean_cosine"),
                "p05_cosine": number(row, "p05_cosine"),
                "effective_bps": number(row, "effective_bps"),
                "int4_effective_bps": number(
                    baseline,
                    "effective_bps",
                ),
                "reuse_rate": number(row, "reuse_rate"),
                "predict_rate": number(row, "predict_rate"),
                "innovation_rate": number(row, "innovation_rate"),
                "refresh_rate": number(row, "refresh_rate"),
                "encoder_required_rate": encoder_required_rate,
                "encoder_skip_rate": encoder_skip_rate,
                "state_update_cost_proxy": number(
                    row,
                    "action_cost_proxy",
                ),
                "standalone_rate_quality_dominated": (
                    rate_quality_dominated
                ),
                "best_dominator": (
                    (
                        f"{best_dominator['method']}:"
                        f"{best_dominator['codec']}@"
                        f"{number(best_dominator, 'selected_fraction'):.2f}"
                    )
                    if best_dominator is not None
                    else ""
                ),
                "memory_representation_candidate": memory_qualified,
                "conditional_compute_candidate": compute_qualified,
            }
        )
    memory_verdict = (
        "Positive"
        if memory_qualified_layers == set(layers)
        else "Mixed"
        if memory_qualified_layers
        else "Negative"
    )
    compute_verdict = (
        "Positive"
        if compute_qualified_layers == set(layers)
        else "Mixed"
        if compute_qualified_layers
        else "Negative"
    )
    return candidates, {
        "verdict": memory_verdict,
        "qualified_layers": sorted(memory_qualified_layers),
        "layer_count": len(layers),
        "max_predict_rate": max_predict_rate,
        "rule": (
            "A memory-representation candidate needs mean cosine >=0.95 and "
            "lower effective bps than always-INT4 refresh on the same held-out "
            "layer, and it must not be rate-quality dominated by standalone "
            "raw PQ, residual PQ, or scalar quantization. It is not "
            "automatically a compute-saving point."
        ),
    }, {
        "verdict": compute_verdict,
        "qualified_layers": sorted(compute_qualified_layers),
        "layer_count": len(layers),
        "rule": (
            "A conditional-compute candidate additionally needs "
            "encoder_required_rate = innovation_rate + refresh_rate <0.30. "
            "Innovation coding requires the current hidden state and therefore "
            "does not skip the visual encoder."
        ),
    }


def markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def render_report(
    *,
    predictor_rows: Sequence[dict[str, Any]],
    predictor_verdict: dict[str, Any],
    rate_rows: Sequence[dict[str, Any]],
    entropy_rows: Sequence[dict[str, Any]],
    vq_verdict: dict[str, Any],
    controller_rows: Sequence[dict[str, Any]],
    controller_verdict: dict[str, Any],
    combined_rows: Sequence[dict[str, Any]],
    combined_verdict: dict[str, Any],
    compute_verdict: dict[str, Any],
) -> str:
    lines = [
        "# Streaming Hybrid State V0 Analysis",
        "",
        "## Executive Summary",
        "",
        f"- Spectral/Fourier predictor: **{predictor_verdict['verdict']}**.",
        f"- Prediction-residual VQ: **{vq_verdict['verdict']}**.",
        f"- Hardened logic controller: **{controller_verdict['verdict']}**.",
        f"- Open-loop memory representation: **{combined_verdict['verdict']}**.",
        f"- Conditional visual compute: **{compute_verdict['verdict']}**.",
        "- End-to-end streaming Video-LLM competitiveness: **UNVALIDATED**.",
        "- No encoder speedup, task-accuracy, or hardware PPA claim is made.",
        "",
        "## 1. Predictor Ablation",
        "",
    ]
    lines += markdown_table(
        (
            "Layer",
            "Best simple",
            "Simple NMSE",
            "Best Fourier",
            "Fourier NMSE",
            "Fourier change",
            "Raw->residual entropy",
            "Winner",
        ),
        (
            (
                row["layer"],
                row["best_simple"],
                f"{row['simple_test_nmse']:.4f}",
                row["best_fourier"],
                f"{row['fourier_test_nmse']:.4f}",
                f"{100.0 * row['fourier_nmse_change_vs_simple']:+.1f}%",
                (
                    f"{100.0 * row['simple_spectral_entropy_reduction']:+.1f}%"
                ),
                "Fourier" if row["fourier_win"] else "simple",
            )
            for row in predictor_rows
        ),
    )
    lines.extend(
        [
            "",
            predictor_verdict["rule"],
            "",
            "## 2. Multi-Bit Rate-Quality Comparison",
            "",
        ]
    )
    lines += markdown_table(
        (
            "Layer",
            "Nominal budget",
            "Method",
            "Point",
            "Actual bps",
            "Cosine",
            "NMSE",
        ),
        (
            (
                row["layer"],
                f"{row['nominal_budget_bps']:.2f}",
                row["method"],
                (
                    f"{row['codec']}@{row['selected_fraction']:.2f}"
                    if row["method"] == "residual_pq"
                    else row["codec"]
                ),
                f"{row['effective_bps']:.3f}",
                f"{row['mean_cosine']:.4f}",
                f"{row['nmse']:.4f}",
            )
            for row in rate_rows
        ),
    )
    lines.extend(["", "### Residual Entropy", ""])
    lines += markdown_table(
        (
            "Layer",
            "Codec",
            "Raw H bps",
            "Residual H bps",
            "Reduction",
            "Raw cosine",
            "Residual cosine",
        ),
        (
            (
                row["layer"],
                row["codec"],
                f"{row['raw_index_entropy_bps']:.3f}",
                f"{row['residual_index_entropy_bps']:.3f}",
                f"{100.0 * row['entropy_reduction']:.1f}%",
                f"{row['raw_cosine']:.4f}",
                f"{row['residual_cosine']:.4f}",
            )
            for row in entropy_rows
        ),
    )
    lines.extend(
        [
            "",
            vq_verdict["rule"],
            "",
            "## 3. Logic Controller",
            "",
        ]
    )
    if controller_rows:
        lines += markdown_table(
            (
                "Layer",
                "Budget",
                "MLP bAcc",
                "DLGN bAcc",
                "Soft-hard gap",
                "DLGN near MLP",
            ),
            (
                (
                    row["layer"],
                    f"{row['budget_bps']:.2f}",
                    f"{row['mlp_balanced_accuracy']:.3f}",
                    f"{row['dlgn_balanced_accuracy']:.3f}",
                    f"{row['dlgn_val_discretization_gap']:+.3f}",
                    row["dlgn_near_mlp_pareto"],
                )
                for row in controller_rows
            ),
        )
    else:
        lines.append("Torch controllers were not evaluated.")
    lines.extend(
        [
            "",
            controller_verdict["rule"],
            "",
            "## 4. Combined Open-Loop Policy",
            "",
        ]
    )
    qualified = [
        row
        for row in combined_rows
        if row["memory_representation_candidate"]
    ]
    if qualified:
        lines += markdown_table(
            (
                "Layer",
                "Budget",
                "Controller",
                "Cosine",
                "Effective bps",
                "Refresh",
                "Innovation",
                "Encoder required",
                "Encoder skip",
                "State-update proxy",
                "RQ dominated",
                "Compute candidate",
            ),
            (
                (
                    row["layer"],
                    f"{row['budget_bps']:.2f}",
                    row["controller"],
                    f"{row['mean_cosine']:.4f}",
                    f"{row['effective_bps']:.3f}",
                    f"{row['refresh_rate']:.3f}",
                    f"{row['innovation_rate']:.3f}",
                    f"{row['encoder_required_rate']:.3f}",
                    f"{row['encoder_skip_rate']:.3f}",
                    f"{row['state_update_cost_proxy']:.3f}",
                    row["standalone_rate_quality_dominated"],
                    row["conditional_compute_candidate"],
                )
                for row in qualified
            ),
        )
    else:
        lines.append("No learned policy passed the representation-level gate.")
    lines.extend(
        [
            "",
            combined_verdict["rule"],
            "",
            compute_verdict["rule"],
            "",
            (
                "The learned policies selected the cheap predictor action at "
                f"a maximum rate of {combined_verdict['max_predict_rate']:.3f}. "
                "A zero value means the nominal four-way controller collapsed "
                "to reuse/innovation/refresh on this corpus."
            ),
            "",
            "## 5. Scientific Verdict",
            "",
            "The components are not automatically stronger when stacked. "
            "A weak Fourier result removes Fourier from the preferred path; "
            "a useful residual code or controller can still be retained "
            "independently. The combined result is a bounded-state latent "
            "codec/controller probe, not a direct comparison with published "
            "Video-LLM task accuracy or measured latency.",
            "",
            "The next experiment should connect the qualified 2-bit/4-bit "
            "policy "
            "to a task-level streaming benchmark and first reduce the true "
            "encoder-required rate. RTL work is justified only after that gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summary = json.loads(
        (args.results_dir / "summary.json").read_text(encoding="utf-8")
    )
    layers = [int(layer) for layer in summary["layers"]]
    predictor_input = read_csv(args.results_dir / "predictor_results.csv")
    vq_input = read_csv(args.results_dir / "vq_results.csv")
    controller_input = read_csv(
        args.results_dir / "controller_results.csv"
    )
    combined_input = read_csv(args.results_dir / "combined_results.csv")

    predictor_rows, predictor_verdict = predictor_analysis(
        predictor_input,
        layers,
    )
    rate_rows, entropy_rows, vq_verdict = vq_analysis(vq_input, layers)
    controller_rows, controller_verdict = controller_analysis(
        controller_input
    )
    combined_rows, combined_verdict, compute_verdict = combined_analysis(
        combined_input,
        vq_input,
        layers,
    )

    verdicts = {
        "predictor": predictor_verdict,
        "residual_vq": vq_verdict,
        "logic_controller": controller_verdict,
        "combined_memory": combined_verdict,
        "conditional_compute": compute_verdict,
        "end_to_end_task": {
            "verdict": "UNVALIDATED",
            "rule": "Requires a streaming Video-LLM task benchmark.",
        },
    }
    write_csv(args.results_dir / "predictor_comparison.csv", predictor_rows)
    write_csv(args.results_dir / "rate_quality_points.csv", rate_rows)
    write_csv(args.results_dir / "residual_entropy_comparison.csv", entropy_rows)
    write_csv(args.results_dir / "logic_controller_comparison.csv", controller_rows)
    write_csv(args.results_dir / "combined_candidate_points.csv", combined_rows)
    (args.results_dir / "component_verdicts.json").write_text(
        json.dumps(verdicts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.results_dir / "STREAMING_HYBRID_STATE_V0_ANALYSIS.md").write_text(
        render_report(
            predictor_rows=predictor_rows,
            predictor_verdict=predictor_verdict,
            rate_rows=rate_rows,
            entropy_rows=entropy_rows,
            vq_verdict=vq_verdict,
            controller_rows=controller_rows,
            controller_verdict=controller_verdict,
            combined_rows=combined_rows,
            combined_verdict=combined_verdict,
            compute_verdict=compute_verdict,
        ),
        encoding="utf-8",
    )
    print(json.dumps(verdicts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
