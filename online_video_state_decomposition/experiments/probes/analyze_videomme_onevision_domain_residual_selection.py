from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from videomme_onevision_domain_residual import validate_manifest
from videomme_onevision_domain_residual_protocol import (
    CANDIDATES,
    SOURCE_CANDIDATE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def adaptation_rank(candidate_id: str) -> int:
    if candidate_id.startswith("residual_swap_r"):
        return int(candidate_id.rsplit("r", maxsplit=1)[1])
    if candidate_id == "target_pca_r456":
        return 456
    return 0


def summarize_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    reference = np.asarray(
        [int(row["reference_candidate_correct"]) for row in rows], dtype=np.int64
    )
    candidate = np.asarray([int(row["candidate_correct"]) for row in rows], dtype=np.int64)
    duration_kl = {}
    for duration in ("short", "medium", "long"):
        selected = [row for row in rows if row["duration"] == duration]
        duration_kl[duration] = float(
            np.mean([float(row["candidate_kl"]) for row in selected])
        )
    return {
        "candidate": str(rows[0]["candidate"]),
        "samples": len(rows),
        "reference_correct": int(np.sum(reference)),
        "candidate_correct": int(np.sum(candidate)),
        "reference_accuracy": float(np.mean(reference)),
        "candidate_accuracy": float(np.mean(candidate)),
        "prediction_mismatches": sum(1 - int(row["prediction_match"]) for row in rows),
        "prediction_agreement": float(
            np.mean([int(row["prediction_match"]) for row in rows])
        ),
        "harmful_flips": sum(int(row["harmful_flip"]) for row in rows),
        "beneficial_flips": sum(int(row["beneficial_flip"]) for row in rows),
        "candidate_kl_mean": float(
            np.mean([float(row["candidate_kl"]) for row in rows])
        ),
        "candidate_kl_p95": float(
            np.quantile([float(row["candidate_kl"]) for row in rows], 0.95)
        ),
        "feature_relative_l2_mean": float(
            np.mean([float(row["feature_relative_l2"]) for row in rows])
        ),
        "max_state_bytes": max(int(row["native_feature_state_bytes"]) for row in rows),
        "min_compression_ratio": min(
            float(row["state_compression_ratio"]) for row in rows
        ),
        "max_injection_abs": max(float(row["injection_max_abs"]) for row in rows),
        "duration_kl": duration_kl,
    }


def classify_candidate(
    candidate: dict[str, object],
    source: dict[str, object],
) -> str:
    required_mismatch_reduction = max(
        2,
        math.ceil(0.20 * int(source["prediction_mismatches"])),
    )
    safety = (
        int(candidate["harmful_flips"]) <= int(source["harmful_flips"])
        and int(candidate["candidate_correct"]) >= int(source["candidate_correct"])
        and all(
            float(candidate["duration_kl"][duration])
            <= 1.10 * float(source["duration_kl"][duration])
            for duration in ("short", "medium", "long")
        )
        and int(candidate["max_state_bytes"]) == int(source["max_state_bytes"])
        and float(candidate["max_injection_abs"]) <= 1e-3
    )
    capacity = (
        safety
        and float(candidate["candidate_kl_mean"])
        <= 0.70 * float(source["candidate_kl_mean"])
        and float(candidate["candidate_kl_p95"])
        <= 0.80 * float(source["candidate_kl_p95"])
        and float(candidate["feature_relative_l2_mean"])
        <= 0.90 * float(source["feature_relative_l2_mean"])
    )
    mismatch = int(candidate["prediction_mismatches"]) <= (
        int(source["prediction_mismatches"]) - required_mismatch_reduction
    )
    if capacity and mismatch:
        return "GO"
    return "CAPACITY_ONLY" if capacity else "NO_GO"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    flattened = []
    for row in rows:
        flattened.append(
            {
                **{key: value for key, value in row.items() if key != "duration_kl"},
                **{
                    f"{duration}_kl": row["duration_kl"][duration]
                    for duration in ("short", "medium", "long")
                },
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    configuration = json.loads(
        (args.run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    if configuration["protocol_id"] != manifest["protocol_id"]:
        raise ValueError("run protocol differs from manifest")
    if configuration["role"] != "selection":
        raise ValueError("selection analyzer received a non-selection run")
    expected_ids = [
        f"videomme_{entry['question_id']}" for entry in manifest["roles"]["selection"]
    ]
    if configuration["sample_ids"] != expected_ids:
        raise ValueError("run sample order differs from the frozen selection role")

    failures = []
    for path in sorted(args.run_dir.glob("failures_shard_*.json")):
        failures.extend(json.loads(path.read_text(encoding="utf-8")))
    checkpoints = sorted((args.run_dir / "checkpoints").glob("*.json"))
    rows = []
    observed_ids = []
    for path in checkpoints:
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed_ids.append(str(payload["metadata"]["sample_id"]))
        rows.extend(payload["rows"])
    if failures:
        raise ValueError(f"selection run contains {len(failures)} failures")
    if set(observed_ids) != set(expected_ids) or len(observed_ids) != len(expected_ids):
        raise ValueError("selection checkpoints are incomplete or unexpected")
    counts = Counter(str(row["candidate"]) for row in rows)
    expected_per_candidate = len(expected_ids)
    if counts != Counter({candidate: expected_per_candidate for candidate in CANDIDATES}):
        raise ValueError("candidate row counts differ from the frozen protocol")
    numeric_fields = (
        "candidate_kl",
        "feature_relative_l2",
        "injection_max_abs",
        "state_compression_ratio",
    )
    if any(
        not math.isfinite(float(row[field]))
        for row in rows
        for field in numeric_fields
    ):
        raise ValueError("selection run contains non-finite metrics")

    summaries = []
    for candidate_id in CANDIDATES:
        candidate_rows = [row for row in rows if row["candidate"] == candidate_id]
        summaries.append(summarize_candidate(candidate_rows))
    by_candidate = {row["candidate"]: row for row in summaries}
    source = by_candidate[SOURCE_CANDIDATE]
    decisions = {}
    for candidate_id in CANDIDATES:
        if candidate_id == SOURCE_CANDIDATE:
            decisions[candidate_id] = "BASELINE"
        else:
            decisions[candidate_id] = classify_candidate(
                by_candidate[candidate_id],
                source,
            )
        by_candidate[candidate_id]["decision"] = decisions[candidate_id]
        by_candidate[candidate_id]["adaptation_rank"] = adaptation_rank(candidate_id)
        by_candidate[candidate_id]["kl_ratio_to_source"] = (
            float(by_candidate[candidate_id]["candidate_kl_mean"])
            / float(source["candidate_kl_mean"])
        )
        by_candidate[candidate_id]["p95_ratio_to_source"] = (
            float(by_candidate[candidate_id]["candidate_kl_p95"])
            / float(source["candidate_kl_p95"])
        )
        by_candidate[candidate_id]["feature_l2_ratio_to_source"] = (
            float(by_candidate[candidate_id]["feature_relative_l2_mean"])
            / float(source["feature_relative_l2_mean"])
        )

    go_candidates = [
        row for row in summaries if row["decision"] == "GO"
    ]
    capacity_candidates = [
        row for row in summaries if row["decision"] == "CAPACITY_ONLY"
    ]
    if go_candidates:
        decision = "GO"
        selected = min(
            go_candidates,
            key=lambda row: (
                float(row["candidate_kl_mean"]),
                int(row["adaptation_rank"]),
                str(row["candidate"]),
            ),
        )
    elif capacity_candidates:
        decision = "CAPACITY_ONLY"
        selected = min(
            capacity_candidates,
            key=lambda row: (
                float(row["candidate_kl_mean"]),
                int(row["adaptation_rank"]),
                str(row["candidate"]),
            ),
        )
    else:
        decision = "NO_GO"
        selected = min(
            [row for row in summaries if row["candidate"] != SOURCE_CANDIDATE],
            key=lambda row: float(row["candidate_kl_mean"]),
        )

    summary = {
        "decision": decision,
        "selected_candidate": selected["candidate"],
        "formal_authorized": decision == "GO",
        "samples": len(expected_ids),
        "source": source,
        "candidates": summaries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "candidate_summary.csv", summaries)
    report = [
        "# Video-MME OneVision Same-Rank Domain-Residual Selection",
        "",
        f"Decision: **{decision}**",
        "",
        f"Selected candidate: `{selected['candidate']}`",
        "",
        "| Candidate | Decision | KL ratio | P95 ratio | L2 ratio | Mismatch | Harmful | Correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            f"| {row['candidate']} | {row['decision']} | "
            f"{row['kl_ratio_to_source']:.3f} | {row['p95_ratio_to_source']:.3f} | "
            f"{row['feature_l2_ratio_to_source']:.3f} | "
            f"{row['prediction_mismatches']} | {row['harmful_flips']} | "
            f"{row['candidate_correct']} |"
        )
    (args.out_dir / "RESULTS_ANALYSIS.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
