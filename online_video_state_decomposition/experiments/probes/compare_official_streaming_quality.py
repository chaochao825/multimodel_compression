from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair official OASIS and CausalMem answers by question ID"
    )
    parser.add_argument("--oasis-output", type=Path, required=True)
    parser.add_argument("--oasis-preflight", type=Path, required=True)
    parser.add_argument("--causalmem-predictions", type=Path, required=True)
    parser.add_argument("--causalmem-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def parse_correct(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{label} must be a boolean or True/False string: {value!r}")


def load_oasis_answers(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_object(path)
    videos = payload.get("results")
    if not isinstance(videos, list) or not videos:
        raise ValueError("OASIS output has no non-empty results list")
    answers: dict[str, dict[str, Any]] = {}
    for video in videos:
        if not isinstance(video, dict) or not isinstance(video.get("breakpoint"), list):
            raise ValueError("OASIS result contains an invalid video record")
        for question in video["breakpoint"]:
            if not isinstance(question, dict):
                raise ValueError("OASIS breakpoint must be an object")
            question_id = question.get("question_id")
            task = question.get("task")
            ground_truth = question.get("gt")
            if not all(
                isinstance(value, str) and value
                for value in (question_id, task, ground_truth)
            ):
                raise ValueError("OASIS answer is missing question_id, task, or gt")
            if question_id in answers:
                raise ValueError(f"duplicate OASIS question_id: {question_id}")
            answers[question_id] = {
                "task": task,
                "ground_truth": ground_truth,
                "correct": parse_correct(
                    question.get("correct"), label=f"OASIS {question_id} correct"
                ),
            }
    return answers


def load_causalmem_answers(path: Path) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid CausalMem JSONL line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"CausalMem line {line_number} is not an object")
        question_id = row.get("id")
        task = row.get("task")
        ground_truth = row.get("answer_id")
        if not all(
            isinstance(value, str) and value
            for value in (question_id, task, ground_truth)
        ):
            raise ValueError(
                f"CausalMem line {line_number} is missing id, task, or answer_id"
            )
        if question_id in answers:
            raise ValueError(f"duplicate CausalMem question_id: {question_id}")
        answers[question_id] = {
            "task": task,
            "ground_truth": ground_truth,
            "correct": parse_correct(
                row.get("acc"), label=f"CausalMem {question_id} acc"
            ),
        }
    if not answers:
        raise ValueError("CausalMem predictions are empty")
    return answers


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def compare_answers(
    oasis: dict[str, dict[str, Any]],
    causalmem: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    oasis_ids = set(oasis)
    causalmem_ids = set(causalmem)
    if oasis_ids != causalmem_ids:
        missing_oasis = sorted(causalmem_ids - oasis_ids)
        missing_causalmem = sorted(oasis_ids - causalmem_ids)
        raise ValueError(
            "question ID sets differ: "
            f"missing_oasis={missing_oasis[:5]}, "
            f"missing_causalmem={missing_causalmem[:5]}"
        )

    outcome_counts = defaultdict(int)
    task_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    question_rows = []
    for question_id in sorted(oasis):
        oasis_row = oasis[question_id]
        causalmem_row = causalmem[question_id]
        for field in ("task", "ground_truth"):
            if oasis_row[field] != causalmem_row[field]:
                raise ValueError(
                    f"{field} mismatch for {question_id}: "
                    f"{oasis_row[field]!r} != {causalmem_row[field]!r}"
                )
        oasis_correct = oasis_row["correct"]
        causalmem_correct = causalmem_row["correct"]
        if oasis_correct and causalmem_correct:
            outcome = "both_correct"
        elif oasis_correct:
            outcome = "oasis_only_correct"
        elif causalmem_correct:
            outcome = "causalmem_only_correct"
        else:
            outcome = "both_wrong"
        outcome_counts[outcome] += 1
        task_counts[oasis_row["task"]][outcome] += 1
        question_rows.append(
            {
                "question_id": question_id,
                "task": oasis_row["task"],
                "ground_truth": oasis_row["ground_truth"],
                "oasis_correct": oasis_correct,
                "causalmem_correct": causalmem_correct,
                "outcome": outcome,
            }
        )

    task_rows = []
    task_total = len(task_counts)
    for task, counts in sorted(task_counts.items()):
        total = sum(counts.values())
        oasis_correct = counts["both_correct"] + counts["oasis_only_correct"]
        causalmem_correct = counts["both_correct"] + counts["causalmem_only_correct"]
        p_value = exact_mcnemar_p(
            counts["oasis_only_correct"], counts["causalmem_only_correct"]
        )
        task_rows.append(
            {
                "task": task,
                "questions": total,
                "oasis_correct": oasis_correct,
                "causalmem_correct": causalmem_correct,
                "oasis_accuracy": oasis_correct / total,
                "causalmem_accuracy": causalmem_correct / total,
                "delta_accuracy_pp": 100.0
                * (oasis_correct - causalmem_correct)
                / total,
                "oasis_only_correct": counts["oasis_only_correct"],
                "causalmem_only_correct": counts["causalmem_only_correct"],
                "mcnemar_exact_p": p_value,
                "mcnemar_bonferroni_p": min(1.0, p_value * task_total),
            }
        )

    total = len(question_rows)
    oasis_correct = sum(row["oasis_correct"] for row in question_rows)
    causalmem_correct = sum(row["causalmem_correct"] for row in question_rows)
    summary = {
        "format_version": 1,
        "comparison_scope": "paired benchmark-system diagnostic, not a memory-module ablation",
        "questions": total,
        "oasis_correct": oasis_correct,
        "causalmem_correct": causalmem_correct,
        "oasis_accuracy": oasis_correct / total,
        "causalmem_accuracy": causalmem_correct / total,
        "delta_accuracy_pp": 100.0 * (oasis_correct - causalmem_correct) / total,
        "both_correct": outcome_counts["both_correct"],
        "oasis_only_correct": outcome_counts["oasis_only_correct"],
        "causalmem_only_correct": outcome_counts["causalmem_only_correct"],
        "both_wrong": outcome_counts["both_wrong"],
        "mcnemar_exact_p": exact_mcnemar_p(
            outcome_counts["oasis_only_correct"],
            outcome_counts["causalmem_only_correct"],
        ),
        "task_count": task_total,
    }
    return summary, question_rows, task_rows


def model_record(payload: dict[str, Any], key: str) -> dict[str, Any]:
    model = payload.get(key)
    if not isinstance(model, dict):
        raise ValueError(f"manifest has no {key} model object")
    path = model.get("path")
    architectures = model.get("architectures")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{key} model path is missing")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError(f"{key} model architectures are missing")
    return {
        "path": path,
        "name": Path(path).name,
        "architectures": architectures,
        "model_type": model.get("model_type"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_task_deltas(task_rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(task_rows, key=lambda row: row["delta_accuracy_pp"])
    labels = [f"{row['task']} (n={row['questions']})" for row in rows]
    values = [row["delta_accuracy_pp"] for row in rows]
    colors = [
        "#bd5b39" if value < 0 else "#167c70" if value > 0 else "#6d7478"
        for value in values
    ]
    fig, axis = plt.subplots(figsize=(10.5, 6.4))
    axis.barh(labels, values, color=colors)
    axis.axvline(0.0, color="#263238", linewidth=1.0)
    span = max(values) - min(values)
    margin = max(5.0, 0.08 * span)
    axis.set_xlim(min(0.0, min(values)) - margin, max(0.0, max(values)) + margin)
    for index, value in enumerate(values):
        offset = 1.2 if value >= 0 else -1.2
        axis.text(
            value + offset,
            index,
            f"{value:+.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )
    axis.set_xlabel("Accuracy difference (OASIS - CausalMem), percentage points")
    axis.set_title("Paired task differences on StreamingBench RTU 1-50")
    axis.grid(axis="x", alpha=0.2)
    fig.text(
        0.5,
        0.01,
        "Diagnostic system comparison: official runs use different VLM backbones; small-n tasks are unstable.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    paths = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"paired_task_accuracy_delta.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def write_analysis(
    path: Path,
    summary: dict[str, Any],
    task_rows: list[dict[str, Any]],
) -> None:
    eligible = [row for row in task_rows if row["questions"] >= 10]
    positive = max(eligible, key=lambda row: row["delta_accuracy_pp"])
    negative = min(eligible, key=lambda row: row["delta_accuracy_pp"])
    text = f"""# Paired Official Quality Analysis

OASIS scored {summary["oasis_correct"]}/{summary["questions"]} ({100 * summary["oasis_accuracy"]:.1f}%) and CausalMem scored {summary["causalmem_correct"]}/{summary["questions"]} ({100 * summary["causalmem_accuracy"]:.1f}%). The difference is {summary["delta_accuracy_pp"]:+.1f} percentage points, or {summary["oasis_correct"] - summary["causalmem_correct"]:+d} questions.

The paired outcomes are {summary["both_correct"]} both correct, {summary["oasis_only_correct"]} OASIS-only correct, {summary["causalmem_only_correct"]} CausalMem-only correct, and {summary["both_wrong"]} both wrong. The exact McNemar p-value is {summary["mcnemar_exact_p"]:.3f}, so the overall difference is not statistically distinguishable in this 250-question run.

Among task groups with at least 10 questions, the largest positive difference is {positive["task"]} ({positive["delta_accuracy_pp"]:+.1f} points; uncorrected p={positive["mcnemar_exact_p"]:.3f}, Bonferroni p={positive["mcnemar_bonferroni_p"]:.3f}). The largest negative difference is {negative["task"]} ({negative["delta_accuracy_pp"]:+.1f} points; uncorrected p={negative["mcnemar_exact_p"]:.3f}).

This is a benchmark-system comparison, not a controlled memory-module ablation: OASIS uses Qwen3-VL-8B-Instruct while CausalMem uses LLaVA-OneVision-Qwen2-7B. The result therefore does not establish that either memory mechanism is superior. A defensible method claim requires a shared backbone, identical frame sampling, and matched memory/token budgets.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    oasis = load_oasis_answers(args.oasis_output)
    causalmem = load_causalmem_answers(args.causalmem_predictions)
    summary, question_rows, task_rows = compare_answers(oasis, causalmem)
    oasis_preflight = load_object(args.oasis_preflight)
    causalmem_manifest = load_object(args.causalmem_manifest)
    summary.update(
        {
            "oasis_model": model_record(oasis_preflight, "mllm"),
            "causalmem_model": model_record(causalmem_manifest, "model"),
            "sources": {
                "oasis_output": {
                    "path": str(args.oasis_output.resolve()),
                    "sha256": sha256_file(args.oasis_output),
                },
                "oasis_preflight": {
                    "path": str(args.oasis_preflight.resolve()),
                    "sha256": sha256_file(args.oasis_preflight),
                },
                "causalmem_predictions": {
                    "path": str(args.causalmem_predictions.resolve()),
                    "sha256": sha256_file(args.causalmem_predictions),
                },
                "causalmem_manifest": {
                    "path": str(args.causalmem_manifest.resolve()),
                    "sha256": sha256_file(args.causalmem_manifest),
                },
            },
            "cautions": [
                "The official runs use different VLM backbones and are not a method-isolated ablation.",
                "Per-task p-values are exploratory and use a Bonferroni correction across reported task groups.",
                "Tasks with very small question counts should not be interpreted from percentage differences alone.",
            ],
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "paired_question_outcomes.csv", question_rows)
    write_csv(args.out_dir / "paired_task_summary.csv", task_rows)
    summary["plots"] = plot_task_deltas(task_rows, args.out_dir)
    (args.out_dir / "paired_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_analysis(args.out_dir / "PAIRED_QUALITY_ANALYSIS.md", summary, task_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
