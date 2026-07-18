from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fit-summary", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=200)
    parser.add_argument(
        "--expected-policies",
        default="exact_recent,learned_recent_query_topk",
    )
    parser.add_argument(
        "--expected-variants",
        default="full,pca_r256_s0,pca_r256_s4",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_rows(
    run_dir: Path,
) -> tuple[list[dict[str, object]], set[str], int]:
    rows = []
    fingerprints = set()
    checkpoint_paths = sorted(
        (run_dir / "checkpoints").glob("*.json")
    )
    for path in checkpoint_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fingerprints.add(str(payload["configuration_fingerprint"]))
        rows.extend(payload["rows"])
    return rows, fingerprints, len(checkpoint_paths)


def split_sample_ids(
    manifest: dict[str, object],
    split: str,
) -> set[str]:
    output = set()
    for task, values in manifest[split].items():
        for value in values:
            if isinstance(value, str):
                output.add(value)
            elif isinstance(value, int):
                output.add(f"{task}_{value:04d}")
            else:
                output.add(
                    str(value.get("sample_id", value.get("id")))
                )
    return output


def selection_split_audit(
    selection_ids: set[str],
    split_manifest: dict[str, object],
    expected_samples: int,
) -> tuple[dict[str, bool], dict[str, int]]:
    calibration_ids = split_sample_ids(
        split_manifest,
        "calibration",
    )
    original_evaluation_ids = split_sample_ids(
        split_manifest,
        "evaluation",
    )
    reserve_ids = split_sample_ids(split_manifest, "reserve")
    prior_ids = split_sample_ids(
        split_manifest,
        "excluded_prior_formal",
    )
    checks = {
        "selection_manifest_has_expected_samples": (
            len(selection_ids) == expected_samples
        ),
        "selection_manifest_is_subset_of_reserve": (
            selection_ids <= reserve_ids
        ),
        "selection_disjoint_from_calibration": not (
            selection_ids & calibration_ids
        ),
        "selection_disjoint_from_original_evaluation": not (
            selection_ids & original_evaluation_ids
        ),
        "selection_disjoint_from_prior_formal": not (
            selection_ids & prior_ids
        ),
    }
    intersections = {
        "selection_calibration": len(
            selection_ids & calibration_ids
        ),
        "selection_original_evaluation": len(
            selection_ids & original_evaluation_ids
        ),
        "selection_prior_formal": len(selection_ids & prior_ids),
        "selection_outside_reserve": len(selection_ids - reserve_ids),
    }
    return checks, intersections


def expected_feature_payload_bytes(
    variant: str,
    *,
    source_frames: int,
    source_tokens: int,
    hidden_size: int,
    rank: int,
    dense_feature_bytes: int,
) -> int:
    """Return logical tensor payload bytes, excluding archive metadata."""

    if variant == "full":
        return dense_feature_bytes
    latent_bytes = source_frames * source_tokens * rank * 2
    routed = re.fullmatch(r"pca_r\d+_route_grid(\d+)_s(\d+)", variant)
    if routed:
        grid_size = int(routed.group(1))
        vectors = int(routed.group(2))
        if vectors != grid_size**2:
            raise ValueError("routed grid and vector counts disagree")
        value_bytes = source_frames * vectors * hidden_size * 2
        index_slot_bytes = source_frames * vectors
        route_mask_bytes = source_frames
        return latent_bytes + value_bytes + index_slot_bytes + route_mask_bytes
    grid = re.fullmatch(r"pca_r\d+_grid(\d+)x(\d+)", variant)
    if grid:
        rows, columns = int(grid.group(1)), int(grid.group(2))
        return latent_bytes + source_frames * rows * columns * hidden_size * 2
    pooled_sparse = re.fullmatch(r"pca_r\d+_mean1_s(\d+)", variant)
    if pooled_sparse:
        sparse = int(pooled_sparse.group(1))
        vectors = 1 + sparse
        return (
            latent_bytes
            + source_frames * vectors * hidden_size * 2
            + source_frames * sparse * 2
        )
    adaptive = re.fullmatch(r"pca_r\d+_(?:global|temporal)_k(\d+)", variant)
    if adaptive:
        budget = int(adaptive.group(1))
        return latent_bytes + budget * hidden_size * 2 + budget * 2
    fixed = re.fullmatch(r"pca_r\d+_s(\d+)", variant)
    if fixed:
        sparse = int(fixed.group(1))
        return (
            latent_bytes
            + source_frames * sparse * hidden_size * 2
            + source_frames * sparse * 2
        )
    raise ValueError(f"cannot parse compressed memory variant: {variant}")


def failure_count(run_dir: Path) -> int:
    count = 0
    for path in sorted(run_dir.glob("failures_shard_*.json")):
        count += len(json.loads(path.read_text(encoding="utf-8")))
    return count


def finite_row_metrics(row: dict[str, object]) -> bool:
    keys = (
        "decode_seconds",
        "preprocess_seconds",
        "vision_encode_seconds",
        "feature_cache_write_seconds",
        "compression_seconds",
        "reconstruction_seconds",
        "inference_seconds",
        "pool_reconstruction_relative_error",
        "selected_reconstruction_relative_error",
        "feature_state_compression_ratio",
        "total_state_compression_ratio",
    )
    return all(math.isfinite(float(row[key])) for key in keys)


def validate(args: argparse.Namespace) -> dict[str, object]:
    policies = csv_values(args.expected_policies)
    variants = csv_values(args.expected_variants)
    expected_pairs = {
        (policy, variant)
        for policy in policies
        for variant in variants
    }
    rows, fingerprints, checkpoint_count = load_checkpoint_rows(
        args.run_dir
    )
    selection_manifest = json.loads(
        args.selection_manifest.read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8")
    )
    fit = json.loads(args.fit_summary.read_text(encoding="utf-8"))
    configuration = json.loads(
        (args.run_dir / "configuration.json").read_text(
            encoding="utf-8"
        )
    )
    selection_ids = set(selection_manifest["samples"])
    split_checks, split_intersections = selection_split_audit(
        selection_ids,
        split_manifest,
        args.expected_samples,
    )
    observed_ids = {str(row["sample_id"]) for row in rows}

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    complete_sample_layout = True
    frame_selection_matches = True
    immutable_sample_fields = True
    for sample_id, sample_rows in grouped.items():
        observed_pairs = {
            (
                str(row["selection_policy"]),
                str(row["memory_variant"]),
            )
            for row in sample_rows
        }
        complete_sample_layout &= observed_pairs == expected_pairs
        answers = {
            (
                str(row["question"]),
                str(row["answer"]),
                int(row["answer_index"]),
                tuple(row["candidates"]),
            )
            for row in sample_rows
        }
        immutable_sample_fields &= len(answers) == 1
        if sample_id not in selection_manifest["samples"]:
            frame_selection_matches = False
            continue
        expected_policy_frames = selection_manifest["samples"][
            sample_id
        ]["policies"]
        for row in sample_rows:
            policy = str(row["selection_policy"])
            frame_selection_matches &= [
                int(value) for value in row["frame_indices"]
            ] == [
                int(value) for value in expected_policy_frames[policy]
            ]

    source_frames, source_tokens, hidden_size = [
        int(value) for value in fit["source_feature_shape"]
    ]
    rank = int(fit["rank"])
    selector_bytes = {
        int(row["provisioned_selector_state_bytes"]) for row in rows
    }
    dense_bytes = {
        int(row["dense_feature_cache_bytes"]) for row in rows
    }
    selector_bytes_value = next(iter(selector_bytes), 0)
    dense_bytes_value = next(iter(dense_bytes), 0)
    variant_state_bytes: dict[str, list[int]] = {}
    expected_state_bytes = {}
    for variant in variants:
        feature_bytes = expected_feature_payload_bytes(
            variant,
            source_frames=source_frames,
            source_tokens=source_tokens,
            hidden_size=hidden_size,
            rank=rank,
            dense_feature_bytes=dense_bytes_value,
        )
        expected_state_bytes[variant] = (
            feature_bytes + selector_bytes_value
        )
        variant_state_bytes[variant] = sorted(
            {
                int(row["selection_state_proxy_bytes"])
                for row in rows
                if row["memory_variant"] == variant
            }
        )

    reference_rows, _, _ = load_checkpoint_rows(args.reference_run)
    reference_lookup = {
        (str(row["sample_id"]), str(row["policy"])): row
        for row in reference_rows
        if str(row["policy"]) in policies
    }
    full_lookup = {
        (str(row["sample_id"]), str(row["selection_policy"])): row
        for row in rows
        if row["memory_variant"] == "full"
    }
    paired_keys = sorted(set(reference_lookup) & set(full_lookup))
    prediction_matches = sum(
        str(reference_lookup[key]["predicted_index"])
        == str(full_lookup[key]["predicted_index"])
        for key in paired_keys
    )
    correctness_matches = sum(
        int(reference_lookup[key]["correct"])
        == int(full_lookup[key]["correct"])
        for key in paired_keys
    )
    prediction_mismatches = [
        {
            "sample_id": key[0],
            "policy": key[1],
            "reference_prediction": reference_lookup[key][
                "predicted_index"
            ],
            "current_prediction": full_lookup[key]["predicted_index"],
        }
        for key in paired_keys
        if str(reference_lookup[key]["predicted_index"])
        != str(full_lookup[key]["predicted_index"])
    ]
    reference_pair_count = len(paired_keys)
    prediction_agreement_rate = (
        prediction_matches / reference_pair_count
        if reference_pair_count
        else 0.0
    )
    correctness_agreement_rate = (
        correctness_matches / reference_pair_count
        if reference_pair_count
        else 0.0
    )

    task_counts = Counter(
        str(next(iter(values))["task"])
        for values in grouped.values()
    )
    configured_tasks = [
        str(value) for value in configuration["tasks"]
    ]
    expected_task_count = (
        args.expected_samples // len(configured_tasks)
        if configured_tasks
        else 0
    )
    expected_task_counts = {
        task: expected_task_count for task in configured_tasks
    }
    expected_rows = (
        args.expected_samples * len(policies) * len(variants)
    )
    checks = {
        "expected_checkpoint_count": (
            checkpoint_count == args.expected_samples
        ),
        "expected_prediction_rows": len(rows) == expected_rows,
        "one_configuration_fingerprint": len(fingerprints) == 1,
        "no_recorded_failures": failure_count(args.run_dir) == 0,
        "observed_ids_match_selection_manifest": (
            observed_ids == selection_ids
        ),
        "selection_manifest_declares_evaluation": (
            selection_manifest.get("split") == "evaluation"
        ),
        **split_checks,
        "balanced_task_sample_counts": (
            dict(task_counts) == expected_task_counts
        ),
        "complete_sample_policy_variant_layout": (
            complete_sample_layout
        ),
        "selection_frames_match_frozen_manifest": (
            frame_selection_matches
        ),
        "sample_fields_immutable_across_variants": (
            immutable_sample_fields
        ),
        "all_rows_parsed": all(int(row["parsed"]) == 1 for row in rows),
        "all_numeric_metrics_finite": all(
            finite_row_metrics(row) for row in rows
        ),
        "online_bounded_flag_set": all(
            int(row["selection_online_bounded"]) == 1 for row in rows
        ),
        "visual_cache_counted": all(
            int(row["visual_evidence_cache_counted"]) == 1
            for row in rows
        ),
        "no_raw_frame_replay_at_read": all(
            int(row["raw_frame_replay_at_read"]) == 0 for row in rows
        ),
        "matched_provisioned_state": all(
            int(row["matched_provisioned_state"]) == 1 for row in rows
        ),
        "one_selector_state_budget": len(selector_bytes) == 1,
        "one_dense_cache_budget": len(dense_bytes) == 1,
        "variant_state_bytes_match_formula": all(
            variant_state_bytes[variant]
            == [expected_state_bytes[variant]]
            for variant in variants
        ),
        "variant_tensor_payload_bytes_match_formula": all(
            variant_state_bytes[variant]
            == [expected_state_bytes[variant]]
            for variant in variants
        ),
        "codec_rank_matches_fit": all(
            int(row["codec_rank"]) == rank for row in rows
        ),
        "codec_parameter_bytes_match_fit": all(
            int(row["codec_parameter_bytes"])
            == int(fit["model_parameter_bytes"])
            for row in rows
        ),
        "codec_hash_matches_fit": (
            str(configuration["codec_sha256"])
            == str(fit["codec_sha256"])
        ),
        "selection_manifest_hash_matches_configuration": (
            str(configuration["selection_manifest_sha256"])
            == sha256(args.selection_manifest)
        ),
        "reference_pair_coverage": (
            len(paired_keys) == args.expected_samples * len(policies)
        ),
        "reference_full_prediction_agreement_at_least_99pct": (
            prediction_agreement_rate >= 0.99
        ),
        "reference_full_correctness_agreement_at_least_99_5pct": (
            correctness_agreement_rate >= 0.995
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "checkpoints": checkpoint_count,
            "prediction_rows": len(rows),
            "sample_ids": len(observed_ids),
            "configuration_fingerprints": len(fingerprints),
            "failures": failure_count(args.run_dir),
            "reference_pairs": reference_pair_count,
            "reference_prediction_matches": prediction_matches,
            "reference_correctness_matches": correctness_matches,
        },
        "task_sample_counts": dict(sorted(task_counts.items())),
        "state_accounting": {
            "scope": "logical_tensor_payload_bytes_excluding_archive_metadata",
            "selector_state_bytes": selector_bytes_value,
            "dense_feature_cache_bytes": dense_bytes_value,
            "codec_parameter_bytes": int(fit["model_parameter_bytes"]),
            "codec_rank": rank,
            "observed_total_state_bytes": variant_state_bytes,
            "expected_total_state_bytes": expected_state_bytes,
        },
        "split_intersections": split_intersections,
        "reference_agreement": {
            "prediction_rate": prediction_agreement_rate,
            "correctness_rate": correctness_agreement_rate,
            "prediction_mismatches": prediction_mismatches[:20],
        },
        "configuration_fingerprints": sorted(fingerprints),
        "codec_sha256": str(fit["codec_sha256"]),
        "selection_manifest_sha256": sha256(args.selection_manifest),
    }


def main() -> int:
    args = parse_args()
    result = validate(args)
    out = args.out or args.run_dir / "aggregate" / "full_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
