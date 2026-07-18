from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from aggregate_mvbench_llava import load_rows


DEFAULT_SELECTOR = "learned_recent_query_topk"
DEFAULT_VARIANTS = {
    "Full state": "full",
    "Low-rank only": "pca_r256_s0",
    "Fixed sparse s4": "pca_r256_s4",
    "Spatial grid 2x2": "pca_r256_grid2x2",
    "Routed grid/sparse": "pca_r256_route_grid2_s4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--routed-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selector", default=DEFAULT_SELECTOR)
    parser.add_argument("--expected-samples", type=int, default=200)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def filter_selector(
    rows: list[dict[str, object]], selector: str
) -> list[dict[str, object]]:
    return [row for row in rows if str(row["selection_policy"]) == selector]


def index_rows(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    indexed = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["memory_variant"]))
        if key in indexed:
            raise ValueError(f"duplicate row for sample/variant {key}")
        indexed[key] = row
    return indexed


def mcnemar_exact_p(better: int, worse: int) -> float:
    discordant = better + worse
    if discordant == 0:
        return 1.0
    lower = min(better, worse)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize_method(
    label: str,
    variant: str,
    rows: list[dict[str, object]],
    full_by_sample: dict[str, dict[str, object]],
) -> dict[str, object]:
    selected = [row for row in rows if str(row["memory_variant"]) == variant]
    if not selected:
        raise ValueError(f"variant {variant} is unavailable")
    selected_by_sample = {str(row["sample_id"]): row for row in selected}
    if set(selected_by_sample) != set(full_by_sample):
        missing = sorted(set(full_by_sample) - set(selected_by_sample))
        extra = sorted(set(selected_by_sample) - set(full_by_sample))
        raise ValueError(
            f"sample mismatch for {variant}: missing={missing[:5]} extra={extra[:5]}"
        )
    correct = sum(int(row["correct"]) for row in selected)
    better = 0
    worse = 0
    prediction_matches = 0
    for sample_id, row in selected_by_sample.items():
        full = full_by_sample[sample_id]
        better += int(not int(full["correct"]) and int(row["correct"]))
        worse += int(int(full["correct"]) and not int(row["correct"]))
        prediction_matches += int(
            int(full["predicted_index"]) == int(row["predicted_index"])
        )
    samples = len(selected)
    state_bytes = sum(
        int(row["selection_state_proxy_bytes"]) for row in selected
    ) / samples
    codec_bytes = int(selected[0].get("codec_parameter_bytes", 0))
    cold_start_bytes = state_bytes + (codec_bytes if variant != "full" else 0)
    pool_frames = sum(len(row["pool_frame_indices"]) for row in selected)
    grid_frames = sum(int(row.get("grid_mode_frames", 0)) for row in selected)
    sparse_frames = sum(int(row.get("sparse_mode_frames", 0)) for row in selected)
    return {
        "label": label,
        "memory_variant": variant,
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples,
        "accuracy_delta_vs_full": (better - worse) / samples,
        "better_samples": better,
        "worse_samples": worse,
        "mcnemar_exact_p": mcnemar_exact_p(better, worse),
        "prediction_agreement_rate": prediction_matches / samples,
        "steady_state_bytes": state_bytes,
        "steady_state_mib": state_bytes / 2**20,
        "cold_start_bytes": cold_start_bytes,
        "cold_start_mib": cold_start_bytes / 2**20,
        "mean_selected_error": sum(
            float(row["selected_reconstruction_relative_error"])
            for row in selected
        )
        / samples,
        "grid_frame_rate": grid_frames / pool_frames if pool_frames else 0.0,
        "sparse_frame_rate": sparse_frames / pool_frames if pool_frames else 0.0,
    }


def sample_table(
    variants: dict[str, dict[str, dict[str, object]]],
    sample_ids: list[str],
) -> list[dict[str, object]]:
    output = []
    for sample_id in sample_ids:
        full = variants["Full state"][sample_id]
        routed = variants["Routed grid/sparse"][sample_id]
        output.append(
            {
                "sample_id": sample_id,
                "task": full["task"],
                "full_correct": int(full["correct"]),
                "fixed_correct": int(
                    variants["Fixed sparse s4"][sample_id]["correct"]
                ),
                "grid_correct": int(
                    variants["Spatial grid 2x2"][sample_id]["correct"]
                ),
                "routed_correct": int(routed["correct"]),
                "full_prediction": int(full["predicted_index"]),
                "fixed_prediction": int(
                    variants["Fixed sparse s4"][sample_id]["predicted_index"]
                ),
                "grid_prediction": int(
                    variants["Spatial grid 2x2"][sample_id]["predicted_index"]
                ),
                "routed_prediction": int(routed["predicted_index"]),
                "fixed_selected_error": float(
                    variants["Fixed sparse s4"][sample_id][
                        "selected_reconstruction_relative_error"
                    ]
                ),
                "grid_selected_error": float(
                    variants["Spatial grid 2x2"][sample_id][
                        "selected_reconstruction_relative_error"
                    ]
                ),
                "routed_selected_error": float(
                    routed["selected_reconstruction_relative_error"]
                ),
                "routed_grid_frames": int(routed.get("grid_mode_frames", 0)),
                "routed_sparse_frames": int(routed.get("sparse_mode_frames", 0)),
            }
        )
    return output


def routing_by_task(
    routed_by_sample: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in routed_by_sample.values():
        grouped[str(row["task"])].append(row)
    output = []
    for task in sorted(grouped):
        values = grouped[task]
        pool_frames = sum(len(row["pool_frame_indices"]) for row in values)
        grid_frames = sum(int(row.get("grid_mode_frames", 0)) for row in values)
        sparse_frames = sum(
            int(row.get("sparse_mode_frames", 0)) for row in values
        )
        output.append(
            {
                "task": task,
                "samples": len(values),
                "accuracy": sum(int(row["correct"]) for row in values)
                / len(values),
                "grid_frame_rate": grid_frames / pool_frames,
                "sparse_frame_rate": sparse_frames / pool_frames,
                "all_grid_samples": sum(
                    int(int(row.get("grid_mode_frames", 0)) == len(row["pool_frame_indices"]))
                    for row in values
                ),
                "all_sparse_samples": sum(
                    int(int(row.get("sparse_mode_frames", 0)) == len(row["pool_frame_indices"]))
                    for row in values
                ),
                "mixed_samples": sum(
                    int(
                        int(row.get("grid_mode_frames", 0)) > 0
                        and int(row.get("sparse_mode_frames", 0)) > 0
                    )
                    for row in values
                ),
            }
        )
    return output


def write_report(
    path: Path,
    summaries: list[dict[str, object]],
    tasks: list[dict[str, object]],
) -> None:
    by_label = {str(row["label"]): row for row in summaries}
    full = by_label["Full state"]
    fixed = by_label["Fixed sparse s4"]
    grid = by_label["Spatial grid 2x2"]
    routed = by_label["Routed grid/sparse"]
    lines = [
        "# Spatial Routing Exploratory Analysis",
        "",
        "This is a post-hoc mechanism-selection analysis on a reused 200-sample set. "
        "It is not an independent confirmation result.",
        "",
        "## Overall",
        "",
        "| Method | Correct | Accuracy | Better / worse vs full | Steady state | Cold start | Mean error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['correct']} / {row['samples']} | "
            f"{100 * float(row['accuracy']):.1f}% | "
            f"{row['better_samples']} / {row['worse_samples']} | "
            f"{float(row['steady_state_mib']):.3f} MiB | "
            f"{float(row['cold_start_mib']):.3f} MiB | "
            f"{100 * float(row['mean_selected_error']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Full state reaches {full['correct']}/{full['samples']}; fixed-s4 and "
            f"grid reach {fixed['correct']} and {grid['correct']} respectively.",
            f"- Routed memory reaches {routed['correct']}/{routed['samples']} with "
            f"{routed['better_samples']} better and {routed['worse_samples']} worse "
            "samples relative to full state.",
            f"- The routed writer uses grid mode on "
            f"{100 * float(routed['grid_frame_rate']):.1f}% of frames and sparse mode "
            f"on {100 * float(routed['sparse_frame_rate']):.1f}%.",
            f"- The amortized steady-state ratio is "
            f"{float(full['steady_state_bytes']) / float(routed['steady_state_bytes']):.2f}x; "
            f"including the shared codec at cold start it is "
            f"{float(full['cold_start_bytes']) / float(routed['cold_start_bytes']):.2f}x.",
            "- Reconstruction error is diagnostic only: the fixed and grid methods "
            "can have similar aggregate accuracy while failing on complementary evidence.",
            "",
            "## Routing By Task",
            "",
            "| Task | Accuracy | Grid frames | Sparse frames | All-grid / all-sparse / mixed samples |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in tasks:
        lines.append(
            f"| {row['task']} | {100 * float(row['accuracy']):.1f}% | "
            f"{100 * float(row['grid_frame_rate']):.1f}% | "
            f"{100 * float(row['sparse_frame_rate']):.1f}% | "
            f"{row['all_grid_samples']} / {row['all_sparse_samples']} / {row['mixed_samples']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "The route was designed after inspecting this same sample set. A paper claim "
            "requires a frozen implementation and a fresh paired reserve set, together "
            "with a matched-budget recent-only baseline and cold-start/runtime accounting.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    reference_rows, reference_fingerprints = load_rows(args.reference_run_dir)
    routed_rows, routed_fingerprints = load_rows(args.routed_run_dir)
    reference_rows = filter_selector(reference_rows, args.selector)
    routed_rows = filter_selector(routed_rows, args.selector)
    reference_index = index_rows(reference_rows)
    routed_index = index_rows(routed_rows)

    full_by_sample = {
        sample_id: row
        for (sample_id, variant), row in reference_index.items()
        if variant == "full"
    }
    if len(full_by_sample) != args.expected_samples:
        raise ValueError(
            f"expected {args.expected_samples} full samples, got {len(full_by_sample)}"
        )
    combined_rows = reference_rows + routed_rows
    summaries = []
    for label, variant in DEFAULT_VARIANTS.items():
        source = routed_rows if variant in {"pca_r256_s0", "pca_r256_route_grid2_s4"} else reference_rows
        summaries.append(
            summarize_method(label, variant, source, full_by_sample)
        )

    variants: dict[str, dict[str, dict[str, object]]] = {}
    for label, variant in DEFAULT_VARIANTS.items():
        index = routed_index if variant in {"pca_r256_s0", "pca_r256_route_grid2_s4"} else reference_index
        variants[label] = {
            sample_id: row
            for (sample_id, row_variant), row in index.items()
            if row_variant == variant
        }
    sample_rows = sample_table(variants, sorted(full_by_sample))
    task_rows = routing_by_task(variants["Routed grid/sparse"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "method_summary.csv", summaries)
    write_csv(args.out_dir / "sample_outcomes.csv", sample_rows)
    write_csv(args.out_dir / "routing_by_task.csv", task_rows)
    metadata = {
        "selector": args.selector,
        "expected_samples": args.expected_samples,
        "reference_run_dir": str(args.reference_run_dir.resolve()),
        "routed_run_dir": str(args.routed_run_dir.resolve()),
        "reference_fingerprints": reference_fingerprints,
        "routed_fingerprints": routed_fingerprints,
        "combined_rows": len(combined_rows),
    }
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.out_dir / "RESULTS_ANALYSIS.md", summaries, task_rows)
    print(args.out_dir / "RESULTS_ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
