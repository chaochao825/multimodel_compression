from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = Path(__file__).resolve().parents[5] / "experiments" / "probes"
sys.path.insert(0, str(PROBE_DIR))

from reader_quotient_cmrq_stage_b import summarize_progressive_fallback  # noqa: E402


DENSE_STATE_BYTES = 22_478_848
COMPRESSED_STATE_BYTES = 2_860_032
STAGE_B_FOLDS = (
    "cmrq_stage_b_loo0",
    "cmrq_stage_b_loo1",
    "cmrq_stage_b_merged48_v2",
)
STAGE_C_FOLDS = (
    "cmrq_stage_c_mix_loo0",
    "cmrq_stage_c_mix_loo1",
    "cmrq_stage_c_mix_loo2",
)
PROGRESSIVE_FOLDS = (
    "cmrq_progressive_loo0",
    "cmrq_progressive_loo1",
    "cmrq_progressive_loo2",
)
STATIC_METHOD_SOURCES = {
    "pooled3_pca_r456": STAGE_B_FOLDS,
    "vsi_pca_train96_r456": STAGE_B_FOLDS,
    "cmrq_risk_atoms32_r456": STAGE_B_FOLDS,
    "feature_only_null_atoms32_r456": STAGE_B_FOLDS,
    "random_null_atoms32_r456": STAGE_B_FOLDS,
    "permuted_risk_null_atoms32_r456": STAGE_B_FOLDS,
    "cmrq_mix_g32_w0p3_r456": STAGE_C_FOLDS,
    "permuted_mix_g32_w0p3_r456": STAGE_C_FOLDS,
}
PAIRWISE_COMPARISONS = (
    ("cmrq_risk_atoms32_r456", "pooled3_pca_r456"),
    ("cmrq_risk_atoms32_r456", "vsi_pca_train96_r456"),
    ("cmrq_risk_atoms32_r456", "feature_only_null_atoms32_r456"),
    ("cmrq_risk_atoms32_r456", "random_null_atoms32_r456"),
    ("cmrq_risk_atoms32_r456", "permuted_risk_null_atoms32_r456"),
    ("cmrq_mix_g32_w0p3_r456", "pooled3_pca_r456"),
    ("cmrq_mix_g32_w0p3_r456", "vsi_pca_train96_r456"),
    ("cmrq_mix_g32_w0p3_r456", "cmrq_risk_atoms32_r456"),
    ("cmrq_mix_g32_w0p3_r456", "permuted_mix_g32_w0p3_r456"),
)
FALLBACK_THRESHOLDS = (0.0, 0.125, 0.25, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=ROOT / "analysis" / "onevision_reader_quotient_stage_a_20260830",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--risk-artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_rows(
    root: Path,
    directories: tuple[str, ...],
    method: str,
) -> list[dict[str, str]]:
    rows = []
    for fold, directory in enumerate(directories):
        for row in read_csv(root / directory / "sample_metrics.csv"):
            if row["method"] == method:
                rows.append({**row, "fold": str(fold)})
    if len(rows) != 72:
        raise ValueError(f"expected 72 rows for {method}, observed {len(rows)}")
    return rows


def aggregate_static_methods(
    root: Path,
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]]]:
    by_method = {
        method: method_rows(root, directories, method)
        for method, directories in STATIC_METHOD_SOURCES.items()
    }
    output = []
    for method, rows in by_method.items():
        kl = np.asarray([float(row["candidate_kl"]) for row in rows])
        l2 = np.asarray([float(row["feature_relative_l2"]) for row in rows])
        output.append(
            {
                "method": method,
                "sample_count": len(rows),
                "candidate_kl_mean": float(kl.mean()),
                "candidate_kl_p95": float(np.quantile(kl, 0.95)),
                "feature_relative_l2_mean": float(l2.mean()),
                "agreement": float(
                    np.mean([int(row["prediction_match"]) for row in rows])
                ),
                "mismatch_count": sum(
                    1 - int(row["prediction_match"]) for row in rows
                ),
                "harmful_count": sum(int(row["harmful"]) for row in rows),
                "beneficial_count": sum(int(row["beneficial"]) for row in rows),
            }
        )
    return output, by_method


def paired_bootstrap(
    by_method: dict[str, list[dict[str, str]]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, object]]:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    generator = np.random.default_rng(seed)
    output = []
    for primary, comparator in PAIRWISE_COMPARISONS:
        primary_rows = {
            (row["fold"], row["sample_id"]): float(row["candidate_kl"])
            for row in by_method[primary]
        }
        comparator_rows = {
            (row["fold"], row["sample_id"]): float(row["candidate_kl"])
            for row in by_method[comparator]
        }
        if primary_rows.keys() != comparator_rows.keys():
            raise ValueError(f"paired sample mismatch: {primary} vs {comparator}")
        keys = sorted(primary_rows)
        primary_values = np.asarray([primary_rows[key] for key in keys])
        comparator_values = np.asarray([comparator_rows[key] for key in keys])
        delta = primary_values - comparator_values
        indices = generator.integers(0, len(keys), size=(draws, len(keys)))
        bootstrap_delta = delta[indices].mean(axis=1)
        output.append(
            {
                "primary": primary,
                "comparator": comparator,
                "sample_count": len(keys),
                "primary_kl_mean": float(primary_values.mean()),
                "comparator_kl_mean": float(comparator_values.mean()),
                "relative_improvement": float(
                    1.0 - primary_values.mean() / comparator_values.mean()
                ),
                "paired_delta_mean": float(delta.mean()),
                "paired_delta_ci_low": float(np.quantile(bootstrap_delta, 0.025)),
                "paired_delta_ci_high": float(np.quantile(bootstrap_delta, 0.975)),
            }
        )
    return output


def aggregate_progressive(root: Path) -> list[dict[str, object]]:
    by_method = defaultdict(list)
    for fold, directory in enumerate(PROGRESSIVE_FOLDS):
        for row in read_csv(root / directory / "sample_metrics.csv"):
            by_method[row["method"]].append({**row, "fold": str(fold)})
    output = []
    for method, rows in sorted(by_method.items()):
        if len(rows) != 72:
            raise ValueError(f"expected 72 progressive rows for {method}")
        for threshold in FALLBACK_THRESHOLDS:
            summary = summarize_progressive_fallback(
                rows,
                margin_threshold=threshold,
                compressed_state_bytes=COMPRESSED_STATE_BYTES,
                dense_state_bytes=DENSE_STATE_BYTES,
            )
            output.append({"method": method, **summary})
    return output


def feature_risk_tradeoff(root: Path) -> list[dict[str, object]]:
    summary = json.loads((root / "cmrq_stage_b_v2" / "summary.json").read_text())
    methods = (
        "pooled3_pca_r456",
        "risk_only_r456",
        "cmrq_risk_atoms16_r456",
        "cmrq_risk_atoms32_r456",
        "cmrq_risk_atoms64_r456",
        "cmrq_risk_atoms96_r456",
    )
    return [
        {
            "method": method,
            "pooled_feature_capture": summary["method_summaries"][method][
                "pooled_feature_capture"
            ],
            "reader_risk_capture": summary["method_summaries"][method][
                "reader_risk_capture"
            ],
            "candidate_kl_mean": summary["method_summaries"][method][
                "candidate_kl_mean"
            ],
        }
        for method in methods
    ]


def parse_risk_artifacts(values: list[str]) -> dict[str, Path]:
    artifacts = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("risk artifacts must use LABEL=PATH")
        artifacts[label] = Path(path)
    if artifacts and len(artifacts) < 2:
        raise ValueError("risk stability requires at least two artifacts")
    return artifacts


def risk_stability(artifacts: dict[str, Path]) -> list[dict[str, object]]:
    if not artifacts:
        return []
    import torch

    payloads = {
        label: torch.load(path, map_location="cpu", weights_only=True)
        for label, path in artifacts.items()
    }
    output = []
    ranks = (16, 32, 64, 96, 456)
    for left_index, left in enumerate(payloads):
        for right in list(payloads)[left_index + 1 :]:
            left_matrix = payloads[left]["risk_matrix"].float()
            right_matrix = payloads[right]["risk_matrix"].float()
            left_basis = payloads[left]["risk_basis"].float()
            right_basis = payloads[right]["risk_basis"].float()
            matrix_cosine = float(
                torch.sum(left_matrix * right_matrix)
                / (
                    torch.linalg.vector_norm(left_matrix)
                    * torch.linalg.vector_norm(right_matrix)
                )
            )
            row: dict[str, object] = {
                "left": left,
                "right": right,
                "matrix_cosine": matrix_cosine,
            }
            cross_gram = left_basis.transpose(0, 1) @ right_basis
            left_capture_right = torch.diagonal(
                left_basis.transpose(0, 1) @ right_matrix @ left_basis
            ).cumsum(0)
            right_capture_left = torch.diagonal(
                right_basis.transpose(0, 1) @ left_matrix @ right_basis
            ).cumsum(0)
            for rank in ranks:
                row[f"overlap_r{rank}"] = float(
                    cross_gram[:rank, :rank].square().sum()
                    / rank
                )
                row[f"left_r{rank}_capture_right"] = float(
                    left_capture_right[rank - 1] / torch.trace(right_matrix)
                )
                row[f"right_r{rank}_capture_left"] = float(
                    right_capture_left[rank - 1] / torch.trace(left_matrix)
                )
            output.append(row)
    return output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.analysis_root / "cmrq_analysis"
    static_rows, by_method = aggregate_static_methods(args.analysis_root)
    pairwise_rows = paired_bootstrap(
        by_method,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    progressive_rows = aggregate_progressive(args.analysis_root)
    tradeoff_rows = feature_risk_tradeoff(args.analysis_root)
    stability_rows = risk_stability(parse_risk_artifacts(args.risk_artifact))
    write_csv(output_dir / "cmrq_crossfit_methods.csv", static_rows)
    write_csv(output_dir / "cmrq_pairwise_bootstrap.csv", pairwise_rows)
    write_csv(output_dir / "progressive_fallback_curve.csv", progressive_rows)
    write_csv(output_dir / "feature_risk_tradeoff.csv", tradeoff_rows)
    if stability_rows:
        write_csv(output_dir / "reader_risk_stability.csv", stability_rows)
    summary_by_method = {row["method"]: row for row in static_rows}
    progressive_zero = {
        row["method"]: row
        for row in progressive_rows
        if float(row["margin_threshold"]) == 0.0
    }
    summary = {
        "calibration_only": True,
        "selection_or_formal_used": False,
        "sample_count": 72,
        "rank": 456,
        "compressed_state_bytes": COMPRESSED_STATE_BYTES,
        "dense_state_bytes": DENSE_STATE_BYTES,
        "static": {
            "hard_risk32": summary_by_method["cmrq_risk_atoms32_r456"],
            "boundary_mix_w0p3": summary_by_method["cmrq_mix_g32_w0p3_r456"],
        },
        "progressive_margin_zero": {
            "hard_risk32": progressive_zero["cmrq_risk_atoms32_r456"],
            "boundary_mix_w0p3": progressive_zero[
                "cmrq_mix_g32_w0p3_r456"
            ],
        },
        "decision": "calibration_conditional_go_to_fresh_selection",
        "claim_boundary": (
            "No formal generalization, task-accuracy, reader-compute, or latency claim."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cmrq_crossfit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
