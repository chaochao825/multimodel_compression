from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation-per-task", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--primary-policy",
        default="recent_pool_query_topk",
    )
    parser.add_argument(
        "--analysis-stage",
        default="posthoc_reserve_confirmation",
    )
    return parser.parse_args()


def stable_task_seed(seed: int, task: str) -> int:
    payload = f"{seed}|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_confirmation(
    source: dict[str, object],
    *,
    evaluation_per_task: int,
    seed: int,
) -> dict[str, dict[str, list[int]]]:
    calibration: dict[str, list[int]] = {}
    evaluation: dict[str, list[int]] = {}
    reserve: dict[str, list[int]] = {}
    for task in source["tasks"]:
        values = np.asarray(source["reserve"][task], dtype=np.int64)
        if len(values) < evaluation_per_task:
            raise ValueError(
                f"task {task} has only {len(values)} reserve samples"
            )
        rng = np.random.default_rng(stable_task_seed(seed, str(task)))
        rng.shuffle(values)
        calibration[str(task)] = []
        evaluation[str(task)] = sorted(
            int(value) for value in values[:evaluation_per_task]
        )
        reserve[str(task)] = sorted(
            int(value) for value in values[evaluation_per_task:]
        )
    return {
        "calibration": calibration,
        "evaluation": evaluation,
        "reserve": reserve,
    }


def validate_confirmation(
    source: dict[str, object],
    confirmation: dict[str, dict[str, list[int]]],
) -> None:
    for task in source["tasks"]:
        source_reserve = set(source["reserve"][task])
        evaluation = set(confirmation["evaluation"][task])
        remaining = set(confirmation["reserve"][task])
        if evaluation & remaining:
            raise ValueError(f"confirmation overlap for task {task}")
        if evaluation | remaining != source_reserve:
            raise ValueError(f"reserve membership changed for task {task}")
        prior = set(source["calibration"][task]) | set(
            source["evaluation"][task]
        )
        if evaluation & prior:
            raise ValueError(f"prior sample leaked for task {task}")


def build_payload(
    source: dict[str, object],
    *,
    source_path: Path,
    evaluation_per_task: int,
    seed: int,
    primary_policy: str,
    analysis_stage: str,
) -> dict[str, object]:
    if not analysis_stage.strip():
        raise ValueError("analysis_stage must be non-empty")
    confirmation = build_confirmation(
        source,
        evaluation_per_task=evaluation_per_task,
        seed=seed,
    )
    validate_confirmation(source, confirmation)
    source_record: dict[str, object] = {
        "name": source_path.name,
        "sha256": file_sha256(source_path),
    }
    if source.get("analysis_stage") is not None:
        source_record["analysis_stage"] = source["analysis_stage"]
    if source.get("source_split") is not None:
        source_record["parent"] = source["source_split"]
    return {
        "format_version": 1,
        "analysis_stage": analysis_stage,
        "seed": seed,
        "tasks": source["tasks"],
        "task_sizes": source["task_sizes"],
        "calibration_per_task": 0,
        "evaluation_per_task": evaluation_per_task,
        "primary_policy": primary_policy,
        "source_split": source_record,
        **confirmation,
    }


def main() -> int:
    args = parse_args()
    source = json.loads(
        args.source_split.read_text(encoding="utf-8")
    )
    payload = build_payload(
        source,
        source_path=args.source_split,
        evaluation_per_task=args.evaluation_per_task,
        seed=args.seed,
        primary_policy=args.primary_policy,
        analysis_stage=args.analysis_stage,
    )
    serialized = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
