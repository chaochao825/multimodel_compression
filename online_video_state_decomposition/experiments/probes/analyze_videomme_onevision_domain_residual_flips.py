from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from videomme_onevision_domain_residual_protocol import (
    CANDIDATES,
    SOURCE_CANDIDATE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def summarize_flip_sets(
    rows: list[dict[str, object]],
    *,
    candidates: tuple[str, ...] = CANDIDATES,
    source_candidate: str = SOURCE_CANDIDATE,
) -> dict[str, dict[str, object]]:
    by_candidate: dict[str, dict[str, dict[str, object]]] = {
        candidate: {} for candidate in candidates
    }
    for row in rows:
        candidate = str(row["candidate"])
        sample_id = str(row["sample_id"])
        if candidate not in by_candidate:
            raise ValueError(f"unexpected candidate {candidate}")
        if sample_id in by_candidate[candidate]:
            raise ValueError(f"duplicate row for {candidate} and {sample_id}")
        by_candidate[candidate][sample_id] = row
    sample_sets = {candidate: set(values) for candidate, values in by_candidate.items()}
    if any(sample_set != sample_sets[source_candidate] for sample_set in sample_sets.values()):
        raise ValueError("candidate sample sets differ")

    source_rows = by_candidate[source_candidate]
    source_mismatches = {
        sample_id
        for sample_id, row in source_rows.items()
        if int(row["prediction_match"]) == 0
    }
    summary = {}
    for candidate in candidates:
        current = by_candidate[candidate]
        mismatches = {
            sample_id
            for sample_id, row in current.items()
            if int(row["prediction_match"]) == 0
        }
        harmful = {
            sample_id
            for sample_id, row in current.items()
            if int(row["harmful_flip"]) == 1
        }
        beneficial = {
            sample_id
            for sample_id, row in current.items()
            if int(row["beneficial_flip"]) == 1
        }
        summary[candidate] = {
            "candidate": candidate,
            "samples": len(current),
            "prediction_mismatches": len(mismatches),
            "harmful_flips": len(harmful),
            "beneficial_flips": len(beneficial),
            "overlap_with_source": len(mismatches & source_mismatches),
            "fixed_source_mismatches": len(source_mismatches - mismatches),
            "new_mismatches": len(mismatches - source_mismatches),
            "kl_better_than_source": sum(
                float(row["candidate_kl"])
                < float(source_rows[sample_id]["candidate_kl"])
                for sample_id, row in current.items()
            ),
            "mismatch_sample_ids": sorted(mismatches),
            "fixed_source_sample_ids": sorted(source_mismatches - mismatches),
            "new_mismatch_sample_ids": sorted(mismatches - source_mismatches),
            "mismatch_domains": dict(
                sorted(Counter(str(current[sample_id]["domain"]) for sample_id in mismatches).items())
            ),
            "mismatch_tasks": dict(
                sorted(Counter(str(current[sample_id]["task_type"]) for sample_id in mismatches).items())
            ),
            "mismatch_durations": dict(
                sorted(Counter(str(current[sample_id]["duration"]) for sample_id in mismatches).items())
            ),
        }
    return summary


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted((args.run_dir / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    summary = summarize_flip_sets(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "flip_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flat_rows = []
    for candidate in CANDIDATES:
        row = summary[candidate]
        flat_rows.append(
            {
                key: value
                for key, value in row.items()
                if not key.endswith("_sample_ids")
                and not key.startswith("mismatch_domain")
                and not key.startswith("mismatch_task")
                and not key.startswith("mismatch_duration")
            }
        )
    with (args.out_dir / "flip_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
