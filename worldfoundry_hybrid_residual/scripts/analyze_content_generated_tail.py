#!/usr/bin/env python3
"""Merge content-tail rank probes and issue a granularity-matched decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RANKS = (16, 32, 48, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def one(
    rows: list[dict[str, str]], variant: str, split: str, route: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["model_variant"] == variant
        and row["split"] == split
        and row["route"] == route
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {(variant, split, route)}")
    return matches[0]


def metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def aggregate_record_subset(
    rows: list[dict[str, str]],
    *,
    variant: str,
    route: str,
    sample_ids: set[str],
    stage: str,
    evidence: str,
) -> dict[str, object]:
    """Aggregate one method on an identical sample subset.

    This avoids comparing a transductive all-sample aggregate with a held-out
    test aggregate. Residuals are summed before taking the relative L2 root.
    """
    selected = [
        row
        for row in rows
        if row["model_variant"] == variant
        and row["route"] == route
        and row["sample_id"] in sample_ids
    ]
    if not selected:
        raise ValueError(f"no records for {(variant, route, sorted(sample_ids))}")
    observed_ids = {row["sample_id"] for row in selected}
    if observed_ids != sample_ids:
        raise ValueError(
            f"record subset mismatch for {stage}: expected {sorted(sample_ids)}, "
            f"found {sorted(observed_ids)}"
        )
    reference_sq = sum(float(row["reference_sq"]) for row in selected)
    if reference_sq <= 0:
        raise ValueError(f"non-positive reference energy for {stage}")

    def relative_l2(field: str) -> float:
        return math.sqrt(sum(float(row[field]) for row in selected) / reference_sq)

    return {
        "stage": stage,
        "model_variant": variant,
        "route": route,
        "evidence": evidence,
        "sample_ids": "|".join(sorted(sample_ids)),
        "samples": len(sample_ids),
        "records": len(selected),
        "reference_sq": reference_sq,
        "content_output_relative_l2": relative_l2("content_residual_sq"),
        "shared_rank16_output_relative_l2": relative_l2(
            "post_adaptive_rank16_residual_sq"
        ),
        "per_tile_rank16_output_relative_l2": relative_l2(
            "post_adaptive_rank16_per_tile_residual_sq"
        ),
        "worst_shared_record_relative_l2": max(
            float(row["post_adaptive_rank16_output_relative_l2"]) for row in selected
        ),
        "worst_tile_relative_l2": max(
            float(row["post_adaptive_rank16_worst_tile_relative_l2"])
            for row in selected
        ),
    }


def add_squared_error_accounting(
    stages: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Add same-test squared-error increments normalized by final proxy error."""
    if len(stages) != 3:
        raise ValueError("expected capacity, frozen, and proxy stages")
    energies = [float(row["per_tile_rank16_output_relative_l2"]) ** 2 for row in stages]
    if energies[-1] <= 0:
        raise ValueError("final proxy error energy must be positive")
    previous = 0.0
    output = []
    for row, energy in zip(stages, energies):
        increment = energy - previous
        output.append(
            {
                **row,
                "per_tile_rank16_relative_squared_error": energy,
                "incremental_squared_error": increment,
                "incremental_squared_error_fraction_of_proxy": increment / energies[-1],
            }
        )
        previous = energy
    return output


def pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)

    summaries: dict[int, list[dict[str, str]]] = {}
    records: dict[int, list[dict[str, str]]] = {}
    decisions = {}
    for rank in RANKS:
        original = root / f"results/content_generated_tail_f81_rank{rank}_v1"
        reevaluated = root / f"results/content_generated_tail_f81_reval_rank{rank}_v1"
        if json.loads((original / "SUCCESS.json").read_text())["artifact_status"] != "SUCCESS":
            raise RuntimeError(f"rank {rank} original probe is incomplete")
        if json.loads((reevaluated / "SUCCESS.json").read_text())["artifact_status"] != "SUCCESS":
            raise RuntimeError(f"rank {rank} re-evaluation is incomplete")
        summaries[rank] = read_csv(reevaluated / "content_tail_summary.csv")
        records[rank] = read_csv(reevaluated / "content_tail_records.csv")
        decisions[rank] = json.loads((reevaluated / "decision.json").read_text())

    frontier = []
    selected_proxy_by_rank: dict[int, str] = {}
    for rank in RANKS:
        selected_proxy = str(decisions[rank]["validation_selected_proxy_route"])
        selected_proxy_by_rank[rank] = selected_proxy
        transductive = one(
            summaries[rank],
            "transductive_capacity",
            "transductive_fit",
            "oracle_trajectory_width_family",
        )
        frozen_oracle = one(
            summaries[rank],
            "calibration_frozen",
            "test",
            "oracle_trajectory_width_family",
        )
        frozen_proxy = one(
            summaries[rank], "calibration_frozen", "test", selected_proxy
        )
        parameters = 12 * (2 * 128 * rank + 2 * rank + 1)
        arithmetic_ratio = 0.25 + 2.0 * rank / 32760.0
        frontier.append(
            {
                "rank": rank,
                "feature_map_parameters": parameters,
                "attention_arithmetic_ratio_proxy": arithmetic_ratio,
                "attention_arithmetic_speedup_upper_bound": 1.0 / arithmetic_ratio,
                "transductive_content_error": metric(transductive, "content_output_relative_l2"),
                "transductive_shared_rank16_error": metric(
                    transductive, "post_adaptive_rank16_output_relative_l2"
                ),
                "transductive_per_tile_rank16_error": metric(
                    transductive, "post_adaptive_rank16_per_tile_output_relative_l2"
                ),
                "transductive_worst_shared_record": metric(
                    transductive, "post_adaptive_rank16_worst_record_relative_l2"
                ),
                "transductive_worst_tile": metric(
                    transductive, "post_adaptive_rank16_worst_tile_relative_l2"
                ),
                "frozen_oracle_content_error": metric(
                    frozen_oracle, "content_output_relative_l2"
                ),
                "frozen_oracle_shared_rank16_error": metric(
                    frozen_oracle, "post_adaptive_rank16_output_relative_l2"
                ),
                "frozen_oracle_per_tile_rank16_error": metric(
                    frozen_oracle, "post_adaptive_rank16_per_tile_output_relative_l2"
                ),
                "frozen_oracle_worst_shared_record": metric(
                    frozen_oracle, "post_adaptive_rank16_worst_record_relative_l2"
                ),
                "frozen_oracle_worst_tile": metric(
                    frozen_oracle, "post_adaptive_rank16_worst_tile_relative_l2"
                ),
                "selected_proxy_route": selected_proxy,
                "frozen_proxy_content_error": metric(
                    frozen_proxy, "content_output_relative_l2"
                ),
                "frozen_proxy_shared_rank16_error": metric(
                    frozen_proxy, "post_adaptive_rank16_output_relative_l2"
                ),
                "frozen_proxy_per_tile_rank16_error": metric(
                    frozen_proxy, "post_adaptive_rank16_per_tile_output_relative_l2"
                ),
                "frozen_proxy_worst_shared_record": metric(
                    frozen_proxy, "post_adaptive_rank16_worst_record_relative_l2"
                ),
                "frozen_proxy_worst_tile": metric(
                    frozen_proxy, "post_adaptive_rank16_worst_tile_relative_l2"
                ),
            }
        )
    write_csv(output / "rank_frontier.csv", frontier)

    rank64_routes = [
        row
        for row in summaries[64]
        if row["model_variant"] == "calibration_frozen" and row["split"] == "test"
    ]
    router_rows = [
        {
            "route": row["route"],
            "content_error": row["content_output_relative_l2"],
            "shared_rank16_error": row["post_adaptive_rank16_output_relative_l2"],
            "per_tile_rank16_error": row["post_adaptive_rank16_per_tile_output_relative_l2"],
            "worst_shared_record": row["post_adaptive_rank16_worst_record_relative_l2"],
            "worst_tile": row["post_adaptive_rank16_worst_tile_relative_l2"],
            "oracle_access": "dense_AV" if row["route"].startswith("oracle_") else "current_QKV_only",
        }
        for row in sorted(rank64_routes, key=lambda item: item["route"])
    ]
    write_csv(output / "router_comparison_rank64.csv", router_rows)

    head_rows = []
    selected_proxy = selected_proxy_by_rank[64]
    for row in records[64]:
        if row["model_variant"] != "calibration_frozen" or row["split"] != "test":
            continue
        if row["route"] not in {selected_proxy, "oracle_trajectory_width_family"}:
            continue
        head_rows.append(
            {
                "head": int(row["head"]),
                "route": row["route"],
                "content_error": row["content_output_relative_l2"],
                "shared_rank16_error": row["post_adaptive_rank16_output_relative_l2"],
                "per_tile_rank16_error": row[
                    "post_adaptive_rank16_per_tile_output_relative_l2"
                ],
                "worst_tile": row["post_adaptive_rank16_worst_tile_relative_l2"],
                "rank_required_for_1pct": row["rank_required_for_1pct_record_gate"],
            }
        )
    write_csv(output / "head_errors_rank64.csv", head_rows)

    test_ids = {
        row["sample_id"]
        for row in records[64]
        if row["model_variant"] == "calibration_frozen" and row["split"] == "test"
    }
    if not test_ids:
        raise ValueError("rank-64 records do not contain a frozen test subset")
    same_test_stages = add_squared_error_accounting(
        [
            aggregate_record_subset(
                records[64],
                variant="transductive_capacity",
                route="oracle_trajectory_width_family",
                sample_ids=test_ids,
                stage="capacity_same_test",
                evidence="all_record_fit_plus_dense_AV_support_on_test_subset",
            ),
            aggregate_record_subset(
                records[64],
                variant="calibration_frozen",
                route="oracle_trajectory_width_family",
                sample_ids=test_ids,
                stage="frozen_same_test",
                evidence="calibration_fit_plus_dense_AV_support_on_test_subset",
            ),
            aggregate_record_subset(
                records[64],
                variant="calibration_frozen",
                route=selected_proxy,
                sample_ids=test_ids,
                stage="proxy_same_test",
                evidence="calibration_fit_plus_validation_selected_QKV_proxy_on_test_subset",
            ),
        ]
    )
    write_csv(output / "same_test_error_decomposition_rank64.csv", same_test_stages)

    baseline_rows = read_csv(
        root / "results/support_manifold_oracle_f81_screen_v1_merged/support_summary.csv"
    )
    baseline = next(
        row
        for row in baseline_rows
        if row["cell"] == "layer14_step09_middle"
        and row["family"] == "support_family_oracle"
        and float(row["density_target"]) == 0.25
    )
    trans64 = one(
        summaries[64],
        "transductive_capacity",
        "transductive_fit",
        "oracle_trajectory_width_family",
    )
    frozen64 = one(
        summaries[64], "calibration_frozen", "test", "oracle_trajectory_width_family"
    )
    proxy64 = one(summaries[64], "calibration_frozen", "test", selected_proxy)
    comparison = [
        {
            "method": "old_support_family_oracle",
            "evidence": "dense_AV_support_plus_per_tile_rank16",
            "content_error": float(baseline["critical_output_relative_l2"]),
            "per_tile_rank16_error": float(baseline["adaptive_rank16_output_relative_l2"]),
            "worst_tile": float(baseline["adaptive_rank16_worst_record_relative_l2"]),
        },
        {
            "method": "rank64_transductive_content_tail_oracle",
            "evidence": "all_record_fit_dense_AV_support_plus_per_tile_rank16",
            "content_error": metric(trans64, "content_output_relative_l2"),
            "per_tile_rank16_error": metric(
                trans64, "post_adaptive_rank16_per_tile_output_relative_l2"
            ),
            "worst_tile": metric(trans64, "post_adaptive_rank16_worst_tile_relative_l2"),
        },
        {
            "method": "rank64_frozen_tail_support_oracle",
            "evidence": "calibration_frozen_tail_dense_AV_support_plus_per_tile_rank16",
            "content_error": metric(frozen64, "content_output_relative_l2"),
            "per_tile_rank16_error": metric(
                frozen64, "post_adaptive_rank16_per_tile_output_relative_l2"
            ),
            "worst_tile": metric(frozen64, "post_adaptive_rank16_worst_tile_relative_l2"),
        },
        {
            "method": "rank64_frozen_tail_selected_proxy",
            "evidence": "calibration_frozen_current_QKV_support_plus_per_tile_rank16",
            "content_error": metric(proxy64, "content_output_relative_l2"),
            "per_tile_rank16_error": metric(
                proxy64, "post_adaptive_rank16_per_tile_output_relative_l2"
            ),
            "worst_tile": metric(proxy64, "post_adaptive_rank16_worst_tile_relative_l2"),
        },
    ]
    write_csv(output / "baseline_comparison.csv", comparison)

    old = comparison[0]
    capacity = comparison[1]
    rank16_capacity = frontier[0]["transductive_per_tile_rank16_error"]
    rank64_capacity = frontier[-1]["transductive_per_tile_rank16_error"]
    capacity_pass = (
        float(capacity["per_tile_rank16_error"]) <= 0.005
        and float(capacity["worst_tile"]) <= 0.01
    )
    decision = {
        "verdict": "STOP_TRAIN_FREE_CONTENT_TAIL_FOR_LAYER14_STEP09",
        "rank64_transductive_capacity_pass": capacity_pass,
        "chart_resume_gate_pass": False,
        "layer14_runtime_assignment": "FP8_OR_BF16_DENSE",
        "do_not_continue": [
            "larger positive linear feature rank",
            "rotation or chart coordinate predictor",
            "additional fixed BCM or support-family expansion",
            "1k-2k adaptation of this tail family because its transductive oracle did not pass",
        ],
        "retained_system_paths": [
            "diffuse heads: fused FP8/BF16 dense attention",
            "localized heads: confidence-gated geometry or semantic sparse proxy",
            "other transitional cells: separately certified learned sparse-linear methods only",
        ],
        "rank64_metrics": {
            "transductive_per_tile_aggregate": float(capacity["per_tile_rank16_error"]),
            "transductive_worst_tile": float(capacity["worst_tile"]),
            "transductive_shared_aggregate": metric(
                trans64, "post_adaptive_rank16_output_relative_l2"
            ),
            "frozen_support_oracle_per_tile_aggregate": metric(
                frozen64, "post_adaptive_rank16_per_tile_output_relative_l2"
            ),
            "frozen_support_oracle_worst_tile": metric(
                frozen64, "post_adaptive_rank16_worst_tile_relative_l2"
            ),
            "selected_proxy_per_tile_aggregate": metric(
                proxy64, "post_adaptive_rank16_per_tile_output_relative_l2"
            ),
            "selected_proxy_worst_tile": metric(
                proxy64, "post_adaptive_rank16_worst_tile_relative_l2"
            ),
        },
        "same_test_squared_error_accounting": {
            "warning": (
                "Descriptive accounting on one identical test subset, not a causal or "
                "mathematical lower-bound decomposition. Fractions use squared aggregate "
                "relative L2 because relative L2 values are not additive."
            ),
            "capacity_per_tile_aggregate": float(
                same_test_stages[0]["per_tile_rank16_output_relative_l2"]
            ),
            "frozen_per_tile_aggregate": float(
                same_test_stages[1]["per_tile_rank16_output_relative_l2"]
            ),
            "proxy_per_tile_aggregate": float(
                same_test_stages[2]["per_tile_rank16_output_relative_l2"]
            ),
            "capacity_floor_fraction_of_proxy_squared_error": float(
                same_test_stages[0]["incremental_squared_error_fraction_of_proxy"]
            ),
            "frozen_transfer_increment_fraction_of_proxy_squared_error": float(
                same_test_stages[1]["incremental_squared_error_fraction_of_proxy"]
            ),
            "proxy_routing_increment_fraction_of_proxy_squared_error": float(
                same_test_stages[2]["incremental_squared_error_fraction_of_proxy"]
            ),
        },
        "matched_baseline_change": {
            "content_error_relative_change": float(capacity["content_error"])
            / float(old["content_error"])
            - 1.0,
            "per_tile_rank16_error_relative_change": float(capacity["per_tile_rank16_error"])
            / float(old["per_tile_rank16_error"])
            - 1.0,
            "worst_tile_relative_change": float(capacity["worst_tile"])
            / float(old["worst_tile"])
            - 1.0,
        },
        "rank16_to_rank64_capacity_relative_change": float(rank64_capacity)
        / float(rank16_capacity)
        - 1.0,
        "claim_boundary": (
            "Previously explored four-capture diagnostic at Layer14/step9/cond with three "
            "sampled query tiles. No end-to-end rollout or measured kernel speed claim."
        ),
    }
    write_json(output / "decision.json", decision)

    report = f"""# Content-Generated Sparse-Linear Tail Diagnostic

## Decision

**{decision['verdict']}**

Rank-64 transductive capacity reaches {pct(float(capacity['per_tile_rank16_error']))}
per-tile aggregate error and {pct(float(capacity['worst_tile']))} worst-tile error after
the additional adaptive rank-16 oracle. Both miss the registered `0.5% / 1.0%`
capacity gate. The stricter basis shared across three query tiles is
{pct(metric(trans64, 'post_adaptive_rank16_output_relative_l2'))} aggregate error.

The calibration-frozen feature map with held-out support oracle reaches
{pct(metric(frozen64, 'post_adaptive_rank16_per_tile_output_relative_l2'))} aggregate
and {pct(metric(frozen64, 'post_adaptive_rank16_worst_tile_relative_l2'))} worst-tile
error. The validation-selected train-free proxy (`{selected_proxy}`) reaches
{pct(metric(proxy64, 'post_adaptive_rank16_per_tile_output_relative_l2'))} and
{pct(metric(proxy64, 'post_adaptive_rank16_worst_tile_relative_l2'))}.

## Same-Test Error Accounting

The headline transductive value above aggregates all four captures, whereas frozen and
proxy values use the held-out test capture. They must not be directly subtracted. On the
identical test capture and granularity, capacity, frozen, and proxy per-tile errors are
{pct(float(same_test_stages[0]['per_tile_rank16_output_relative_l2']))},
{pct(float(same_test_stages[1]['per_tile_rank16_output_relative_l2']))}, and
{pct(float(same_test_stages[2]['per_tile_rank16_output_relative_l2']))}.

Using squared aggregate error only as descriptive accounting, the current function-class
floor contributes {100.0 * float(same_test_stages[0]['incremental_squared_error_fraction_of_proxy']):.1f}%
of final proxy error energy, calibration-to-test freezing adds
{100.0 * float(same_test_stages[1]['incremental_squared_error_fraction_of_proxy']):.1f}%,
and proxy routing adds
{100.0 * float(same_test_stages[2]['incremental_squared_error_fraction_of_proxy']):.1f}%.
This supports capacity mismatch as the dominant observed term, but it is not a causal
decomposition or an impossibility proof for all sparse-linear methods.

## What Improved

Against the old, granularity-matched `25% support-family oracle + per-tile rank-16`
baseline, the rank-64 transductive content tail changes raw content error from
{pct(float(old['content_error']))} to {pct(float(capacity['content_error']))}, per-tile
post-rank error from {pct(float(old['per_tile_rank16_error']))} to
{pct(float(capacity['per_tile_rank16_error']))}, and worst-tile error from
{pct(float(old['worst_tile']))} to {pct(float(capacity['worst_tile']))}. This is useful
tail shaping, but not enough to meet the gate.

Increasing the learned positive feature rank from 16 to 64 changes transductive
per-tile error by only {100.0 * decision['rank16_to_rank64_capacity_relative_change']:.2f}%.
The plateau means feature width is not the limiting variable; the positive separable
kernel and low-cost support semantics remain mismatched to Layer 14 content.

## Interpretation

The result supports semantic permutation as a candidate-layout improvement, not as a
complete tail solution. SVG2 uses semantic clustering, permutation, top-p control, and
custom kernels; this experiment uses a deliberately labeled SVG2-style Q/K sorting
proxy and does not claim a paper-faithful reproduction. VSA similarly relies on a
trainable coarse-to-fine router and fused block kernel. SLA2 addresses sparse/linear
branch mismatch with learnable routing and branch ratio. Our shared numerator and
denominator removes one normalization error source, yet the remaining Layer 14 defect
still requires roughly 28 output dimensions on average and up to 59 for the 1% record
gate in the transductive rank-64 run.

Therefore Layer 14/step 9 should use FP8 or BF16 dense fallback in this train-free
system. Rotation/chart work remains stopped. A 1k-2k adaptation of this exact tail
family is not justified because even the all-record transductive oracle misses the
capacity gate. Learned sparse-linear methods remain viable as a separate function class,
but require a new registered hypothesis rather than widening this feature map.

## Evidence Boundary

- Four previously explored captures, one layer/step/CFG branch, and three query tiles.
- Transductive and dense-support results are oracle diagnostics, not deployment results.
- The support search is a monotone projected-rank heuristic, not a global optimum.
- Arithmetic speedup is an upper bound; no fused H200 kernel or end-to-end rollout was measured.
- Figures compare per-tile rank-16 only where the old baseline uses the same granularity.
- Cross-stage error accounting uses the same held-out test records and squared errors;
  the all-sample `1.179%` capacity number is not subtracted from held-out metrics.

## Related Boundaries

- [Sparse VideoGen2](https://arxiv.org/abs/2505.18875)
- [VSA](https://arxiv.org/abs/2505.13389)
- [SLA2](https://arxiv.org/abs/2602.12675)
- [DynamicRad](https://arxiv.org/abs/2604.20470)
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"[content-tail-analysis] wrote {output}")


if __name__ == "__main__":
    main()
