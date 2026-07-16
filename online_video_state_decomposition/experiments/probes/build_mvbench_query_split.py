from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mvbench_utils import parse_csv_list


DEFAULT_TASKS = (
    "object_existence",
    "state_change",
    "scene_transition",
    "action_sequence",
    "moving_direction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--exclude-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--calibration-per-task", type=int, default=20)
    parser.add_argument("--evaluation-per-task", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def stable_task_seed(seed: int, task: str) -> int:
    payload = f"{seed}|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_excluded_indices(
    path: Path,
    *,
    tasks: list[str],
) -> dict[str, set[int]]:
    output = {task: set() for task in tasks}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            task = str(row["task"])
            if task not in output:
                continue
            output[task].add(int(row["sample_index"]))
    return output


def build_split(
    *,
    task_sizes: dict[str, int],
    excluded: dict[str, set[int]],
    calibration_per_task: int,
    evaluation_per_task: int,
    seed: int,
) -> dict[str, dict[str, list[int]]]:
    calibration: dict[str, list[int]] = {}
    evaluation: dict[str, list[int]] = {}
    reserve: dict[str, list[int]] = {}
    for task, task_size in task_sizes.items():
        remaining = np.asarray(
            [
                index
                for index in range(task_size)
                if index not in excluded[task]
            ],
            dtype=np.int64,
        )
        required = calibration_per_task + evaluation_per_task
        if len(remaining) < required:
            raise ValueError(
                f"task {task} has {len(remaining)} available records, "
                f"but {required} are required"
            )
        rng = np.random.default_rng(stable_task_seed(seed, task))
        rng.shuffle(remaining)
        calibration[task] = sorted(
            int(value) for value in remaining[:calibration_per_task]
        )
        evaluation[task] = sorted(
            int(value)
            for value in remaining[
                calibration_per_task : calibration_per_task
                + evaluation_per_task
            ]
        )
        reserve[task] = sorted(
            int(value) for value in remaining[required:]
        )
    return {
        "calibration": calibration,
        "evaluation": evaluation,
        "reserve": reserve,
    }


def validate_split(
    split: dict[str, dict[str, list[int]]],
    excluded: dict[str, set[int]],
) -> None:
    for task in excluded:
        calibration = set(split["calibration"][task])
        evaluation = set(split["evaluation"][task])
        reserve = set(split["reserve"][task])
        if calibration & evaluation or calibration & reserve or evaluation & reserve:
            raise ValueError(f"split overlap detected for task {task}")
        if (calibration | evaluation | reserve) & excluded[task]:
            raise ValueError(f"excluded sample leaked into split for task {task}")


def main() -> int:
    args = parse_args()
    tasks = parse_csv_list(args.tasks)
    manifest = json.loads(
        (args.dataset_root / "test.json").read_text(encoding="utf-8")
    )
    task_sizes = {
        task: len(manifest["meta"][task])
        for task in tasks
    }
    excluded = load_excluded_indices(
        args.exclude_predictions,
        tasks=tasks,
    )
    split = build_split(
        task_sizes=task_sizes,
        excluded=excluded,
        calibration_per_task=args.calibration_per_task,
        evaluation_per_task=args.evaluation_per_task,
        seed=args.seed,
    )
    validate_split(split, excluded)
    payload = {
        "format_version": 1,
        "seed": args.seed,
        "tasks": tasks,
        "task_sizes": task_sizes,
        "calibration_per_task": args.calibration_per_task,
        "evaluation_per_task": args.evaluation_per_task,
        "excluded_source": {
            "name": args.exclude_predictions.name,
            "sha256": file_sha256(args.exclude_predictions),
        },
        "excluded_prior_formal": {
            task: sorted(excluded[task])
            for task in tasks
        },
        **split,
    }
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
