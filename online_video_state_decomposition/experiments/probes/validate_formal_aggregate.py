from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


EXPECTED_CATEGORIES = (
    "camera_motion",
    "high_change",
    "object_motion",
    "scene_cut",
    "static",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-runs", type=int, default=30)
    parser.add_argument("--expected-per-category", type=int, default=6)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row}")
    return value


def numeric_values_are_finite(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        for key, raw in row.items():
            if raw in ("", None):
                continue
            if key in {
                "category",
                "alignment",
                "method",
                "mode",
            }:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if not math.isfinite(value):
                return False
    return True


def keyed(
    rows: list[dict[str, str]],
    *keys: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(reference, 1e-12)


def make_decision_rows(
    rank_rows: list[dict[str, str]],
    subspace_rows: list[dict[str, str]],
    residual_rows: list[dict[str, str]],
    transport_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    rank_lookup = keyed(
        rank_rows,
        "category",
        "layer",
        "alignment",
        "rank",
    )
    subspace_lookup = keyed(
        subspace_rows,
        "category",
        "layer",
        "rank",
        "mode",
    )
    residual_lookup = keyed(
        residual_rows,
        "category",
        "layer",
        "method",
        "fraction",
    )
    transport_lookup = keyed(
        transport_rows,
        "category",
        "layer",
        "method",
    )
    layers = sorted({int(row["layer"]) for row in rank_rows})

    for category in EXPECTED_CATEGORIES:
        for layer in layers:
            layer_text = str(layer)
            rank = rank_lookup[
                (
                    category,
                    layer_text,
                    "history_feature_subspace_token_normalized",
                    "32",
                )
            ]
            state_energy = to_float(rank, "energy_ratio_mean")
            output.append(
                {
                    "category": category,
                    "layer": layer,
                    "metric": "state_rank32_energy",
                    "value": state_energy,
                    "threshold": 0.70,
                    "eligible": True,
                    "gate_pass": state_energy >= 0.70,
                }
            )

            subspace = subspace_lookup[
                (
                    category,
                    layer_text,
                    "32",
                    "history_centered",
                )
            ]
            output.append(
                {
                    "category": category,
                    "layer": layer,
                    "metric": "causal_rank32_projection_error",
                    "value": to_float(
                        subspace,
                        "relative_projection_error_mean",
                    ),
                    "threshold": "",
                    "eligible": True,
                    "gate_pass": "",
                }
            )

            residual = residual_lookup[
                (category, layer_text, "identity", "0.1")
            ]
            residual_energy = to_float(residual, "energy_ratio_mean")
            residual_recall = to_float(
                residual,
                "pixel_change_recall_mean",
            )
            output.extend(
                [
                    {
                        "category": category,
                        "layer": layer,
                        "metric": "residual_top10_energy",
                        "value": residual_energy,
                        "threshold": 0.70,
                        "eligible": True,
                        "gate_pass": residual_energy >= 0.70,
                    },
                    {
                        "category": category,
                        "layer": layer,
                        "metric": "residual_top10_change_recall",
                        "value": residual_recall,
                        "threshold": 0.80,
                        "eligible": True,
                        "gate_pass": residual_recall >= 0.80,
                    },
                    {
                        "category": category,
                        "layer": layer,
                        "metric": "residual_joint_gate",
                        "value": min(residual_energy, residual_recall),
                        "threshold": "",
                        "eligible": True,
                        "gate_pass": (
                            residual_energy >= 0.70
                            and residual_recall >= 0.80
                        ),
                    },
                ]
            )

            methods = {
                method: transport_lookup[(category, layer_text, method)]
                for method in (
                    "identity",
                    "optical_flow_warp",
                    "local_bttb_causal",
                    "local_bccb_causal",
                )
            }
            errors = {
                method: to_float(row, "stable_centered_error_mean")
                for method, row in methods.items()
            }
            identity_eligible = errors["identity"] >= 0.05
            bttb_eligible = errors["local_bttb_causal"] >= 0.05
            for method, metric in (
                ("optical_flow_warp", "flow_gain_vs_identity"),
                ("local_bttb_causal", "bttb_gain_vs_identity"),
                ("local_bccb_causal", "bccb_gain_vs_identity"),
            ):
                gain = relative_gain(errors["identity"], errors[method])
                output.append(
                    {
                        "category": category,
                        "layer": layer,
                        "metric": metric,
                        "value": gain,
                        "threshold": 0.10,
                        "eligible": identity_eligible,
                        "gate_pass": identity_eligible and gain >= 0.10,
                    }
                )
            incremental = relative_gain(
                errors["local_bttb_causal"],
                errors["local_bccb_causal"],
            )
            output.append(
                {
                    "category": category,
                    "layer": layer,
                    "metric": "bccb_increment_vs_bttb",
                    "value": incremental,
                    "threshold": 0.0,
                    "eligible": bttb_eligible,
                    "gate_pass": bttb_eligible and incremental > 0.0,
                }
            )
    return output


def summarize_metric(
    rows: list[dict[str, object]],
    metric: str,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["metric"] == metric and bool(row["eligible"])
    ]
    values = [float(row["value"]) for row in selected]
    passes = [
        row
        for row in selected
        if isinstance(row["gate_pass"], bool) and row["gate_pass"]
    ]
    return {
        "eligible_cells": len(selected),
        "pass_cells": len(passes),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> int:
    args = parse_args()
    aggregate_dir = args.aggregate_dir.resolve()
    summary_path = aggregate_dir / "aggregate_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    category_counts = Counter(
        str(run["category"]) for run in summary["runs"]
    )

    csv_paths = {
        "rank": aggregate_dir / "rank_summary_by_category.csv",
        "subspace": (
            aggregate_dir / "causal_subspace_summary_by_category.csv"
        ),
        "residual": aggregate_dir / "residual_summary_by_category.csv",
        "transport": aggregate_dir / "transport_summary_by_category.csv",
    }
    tables = {name: read_csv(path) for name, path in csv_paths.items()}
    decision_rows = make_decision_rows(
        tables["rank"],
        tables["subspace"],
        tables["residual"],
        tables["transport"],
    )
    decision_path = aggregate_dir / "formal_probe_decision_metrics.csv"
    write_csv(decision_path, decision_rows)

    required_figures = [
        f"{args.prefix}_temporal_spectrum.png",
        f"{args.prefix}_state_spectrum.png",
        f"{args.prefix}_state_spectrum.pdf",
        f"{args.prefix}_transport_error.png",
        f"{args.prefix}_causal_subspace.png",
        f"{args.prefix}_residual_concentration.png",
        f"{args.prefix}_category_probe_summary.png",
        f"{args.prefix}_category_probe_summary.pdf",
    ]
    checks = {
        "run_count": len(summary["runs"]) == args.expected_runs,
        "category_counts": category_counts
        == Counter(
            {
                category: args.expected_per_category
                for category in EXPECTED_CATEGORIES
            }
        ),
        "all_tables_nonempty": all(tables.values()),
        "all_numeric_values_finite": all(
            numeric_values_are_finite(rows) for rows in tables.values()
        ),
        "figures_present_and_nonempty": all(
            (aggregate_dir / name).exists()
            and (aggregate_dir / name).stat().st_size > 0
            for name in required_figures
        ),
    }
    metric_names = sorted({str(row["metric"]) for row in decision_rows})
    validation = {
        "aggregate_dir": str(aggregate_dir),
        "checks": checks,
        "valid": all(checks.values()),
        "run_count": len(summary["runs"]),
        "category_counts": dict(sorted(category_counts.items())),
        "table_rows": {
            name: len(rows) for name, rows in tables.items()
        },
        "required_figures": required_figures,
        "decision_summary": {
            metric: summarize_metric(decision_rows, metric)
            for metric in metric_names
        },
    }
    validation_path = aggregate_dir / "formal_probe_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
