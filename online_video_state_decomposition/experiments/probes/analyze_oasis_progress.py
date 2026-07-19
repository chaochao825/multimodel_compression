from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_oasis_streamingbench import summarize_official_output


VIDEO_FIELDS = (
    "video_index",
    "video_label",
    "questions_completed",
    "questions_scored",
    "correct",
    "cumulative_questions_completed",
    "cumulative_questions_scored",
    "cumulative_correct",
    "cumulative_accuracy_on_scored",
    "cumulative_wilson_low",
    "cumulative_wilson_high",
    "video_completion_fraction",
    "question_completion_fraction",
)

TASK_FIELDS = (
    "task",
    "questions_completed",
    "questions_scored",
    "correct",
    "accuracy_on_scored",
    "wilson_low",
    "wilson_high",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and visualize an in-progress audited OASIS evaluation. "
            "Partial accuracy is diagnostic and never formal-comparison eligible."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--observed-at",
        help="Optional deterministic UTC timestamp for tests or archived snapshots",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _parse_timestamp(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def wilson_interval(correct: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if isinstance(correct, bool) or isinstance(total, bool):
        raise ValueError("Wilson counts must be integers")
    if not isinstance(correct, int) or not isinstance(total, int):
        raise ValueError("Wilson counts must be integers")
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError(f"invalid Wilson counts: correct={correct}, total={total}")
    proportion = correct / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _validate_preflight(preflight: dict[str, Any], *, metadata_path: Path) -> tuple[str, datetime]:
    if preflight.get("method") != "OASIS":
        raise ValueError("preflight is not an audited OASIS run")
    fingerprint = preflight.get("run_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("preflight has no valid OASIS run fingerprint")
    metadata = preflight.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
        raise ValueError("preflight does not bind an OASIS metadata path")
    if Path(metadata["path"]).resolve() != metadata_path.resolve():
        raise ValueError("preflight metadata path does not match --metadata")
    created_at = _parse_timestamp(preflight.get("created_at"), label="preflight created_at")
    return fingerprint, created_at


def _is_error(breakpoint: dict[str, Any]) -> bool:
    return "error" in breakpoint or breakpoint.get("prediction") == "Error"


def _video_rows(
    payload: dict[str, Any],
    *,
    expected_videos: int,
    expected_questions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_completed = 0
    cumulative_scored = 0
    cumulative_correct = 0
    for index, result in enumerate(payload["results"], start=1):
        breakpoints = result["breakpoint"]
        scored = [row for row in breakpoints if not _is_error(row)]
        correct = sum(int(row["correct"]) for row in scored)
        cumulative_completed += len(breakpoints)
        cumulative_scored += len(scored)
        cumulative_correct += correct
        low, high = wilson_interval(cumulative_correct, cumulative_scored)
        info = result.get("info", {})
        video_path = info.get("video_path", f"video_{index}")
        label = Path(video_path).parent.name or Path(video_path).stem
        rows.append(
            {
                "video_index": index,
                "video_label": label,
                "questions_completed": len(breakpoints),
                "questions_scored": len(scored),
                "correct": correct,
                "cumulative_questions_completed": cumulative_completed,
                "cumulative_questions_scored": cumulative_scored,
                "cumulative_correct": cumulative_correct,
                "cumulative_accuracy_on_scored": cumulative_correct / cumulative_scored,
                "cumulative_wilson_low": low,
                "cumulative_wilson_high": high,
                "video_completion_fraction": index / expected_videos,
                "question_completion_fraction": cumulative_completed / expected_questions,
            }
        )
    return rows


def _task_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    completed: Counter[str] = Counter()
    scored: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for result in payload["results"]:
        for breakpoint in result["breakpoint"]:
            task = breakpoint["task"]
            completed[task] += 1
            if not _is_error(breakpoint):
                scored[task] += 1
                correct[task] += int(breakpoint["correct"])
    rows = []
    for task in sorted(completed):
        low, high = wilson_interval(correct[task], scored[task])
        rows.append(
            {
                "task": task,
                "questions_completed": completed[task],
                "questions_scored": scored[task],
                "correct": correct[task],
                "accuracy_on_scored": correct[task] / scored[task],
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_progress(
    *,
    summary: dict[str, Any],
    video_rows: list[dict[str, Any]],
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    blue = "#0072B2"
    orange = "#D55E00"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    fractions = [summary["video_completion_fraction"], summary["question_completion_fraction"]]
    labels = ["Videos", "Questions"]
    colors = [blue, orange]
    bars = axes[0].barh(labels, fractions, color=colors, edgecolor="black", linewidth=0.7)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Formal-run completion fraction")
    axes[0].grid(axis="x", alpha=0.25, linewidth=0.6)
    counts = [
        f"{summary['completed_videos']}/{summary['expected_videos']}",
        f"{summary['completed_questions']}/{summary['expected_questions']}",
    ]
    for bar, fraction, count in zip(bars, fractions, counts, strict=True):
        axes[0].text(
            min(fraction + 0.025, 0.96),
            bar.get_y() + bar.get_height() / 2.0,
            count,
            va="center",
            ha="left" if fraction < 0.90 else "right",
        )

    x = [row["video_index"] for row in video_rows]
    accuracy = [row["cumulative_accuracy_on_scored"] for row in video_rows]
    low = [row["cumulative_wilson_low"] for row in video_rows]
    high = [row["cumulative_wilson_high"] for row in video_rows]
    axes[1].plot(
        x,
        accuracy,
        color=blue,
        marker="o",
        linewidth=1.8,
        label="Completed-question accuracy",
    )
    axes[1].fill_between(x, low, high, color=blue, alpha=0.18, label="95% Wilson interval")
    axes[1].set_xlim(0.5, max(summary["expected_videos"] + 0.5, 1.5))
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("Completed videos (count)")
    axes[1].set_ylabel("Accuracy on completed questions")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False, loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze_progress(
    *,
    output_path: Path,
    metadata_path: Path,
    preflight_path: Path,
    out_dir: Path,
    observed_at: str | None = None,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    metadata_path = metadata_path.resolve()
    preflight_path = preflight_path.resolve()
    output = _load_json_object(output_path)
    preflight = _load_json_object(preflight_path)
    fingerprint, created_at = _validate_preflight(preflight, metadata_path=metadata_path)
    metrics = summarize_official_output(output_path, metadata_path=metadata_path)

    observed = (
        _parse_timestamp(observed_at, label="observed_at")
        if observed_at is not None
        else datetime.now(timezone.utc)
    )
    elapsed_seconds = (observed - created_at).total_seconds()
    if elapsed_seconds < 0.0:
        raise ValueError("observed_at precedes the audited preflight")

    video_rows = _video_rows(
        output,
        expected_videos=metrics["expected_videos"],
        expected_questions=metrics["expected_questions"],
    )
    task_rows = _task_rows(output)
    last = video_rows[-1]
    if last["cumulative_questions_completed"] != metrics["completed_questions"]:
        raise ValueError("derived completed-question count disagrees with audited output")
    if last["cumulative_questions_scored"] != metrics["scored_questions"]:
        raise ValueError("derived scored-question count disagrees with audited output")
    if last["cumulative_correct"] != metrics["correct"]:
        raise ValueError("derived correct count disagrees with audited output")

    low, high = wilson_interval(metrics["correct"], metrics["scored_questions"])
    complete = metrics["complete"]
    completion_fraction = metrics["completed_videos"] / metrics["expected_videos"]
    seconds_per_completed_video = elapsed_seconds / metrics["completed_videos"]
    estimated_total_seconds = seconds_per_completed_video * metrics["expected_videos"]
    estimated_remaining_seconds = max(0.0, estimated_total_seconds - elapsed_seconds)
    scope = (
        "formal_50x5"
        if metrics["expected_videos"] == 50 and metrics["expected_questions"] == 250
        else f"subset_{metrics['expected_videos']}x{metrics['expected_questions']}"
    )
    summary = {
        "format_version": 1,
        "method": "OASIS",
        "benchmark": "StreamingBench Real-Time Visual Understanding",
        "scope": scope,
        "status": "complete_monitor" if complete else "partial_monitor",
        "evidence_tier": (
            "official_model_level_complete_monitor"
            if complete
            else "official_model_level_partial_monitor"
        ),
        "formal_comparison_eligible": False,
        "run_fingerprint": fingerprint,
        "output_path": str(output_path),
        "output_sha256": metrics["output_sha256"],
        "preflight_path": str(preflight_path),
        "preflight_created_at": created_at.isoformat(),
        "observed_at": observed.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "completed_videos": metrics["completed_videos"],
        "expected_videos": metrics["expected_videos"],
        "video_completion_fraction": completion_fraction,
        "completed_questions": metrics["completed_questions"],
        "scored_questions": metrics["scored_questions"],
        "expected_questions": metrics["expected_questions"],
        "question_completion_fraction": metrics["completed_questions"]
        / metrics["expected_questions"],
        "correct": metrics["correct"],
        "accuracy_on_completed_scored_questions": metrics["accuracy_on_scored"],
        "accuracy_wilson_95_low": low,
        "accuracy_wilson_95_high": high,
        "formal_quality_accuracy": metrics["accuracy"],
        "errors": metrics["errors"],
        "safe_resume_prefix": metrics["safe_resume_prefix"],
        "seconds_per_completed_video_linear": seconds_per_completed_video,
        "estimated_total_seconds_linear": estimated_total_seconds,
        "estimated_remaining_seconds_linear": estimated_remaining_seconds,
        "task_metrics_partial": task_rows,
        "cautions": [
            "Accuracy and task metrics use completed questions only and are not a formal result while the run is partial.",
            "The Wilson interval quantifies binomial uncertainty but does not correct completion-order bias.",
            "The linear time estimate includes initialization and assumes future videos cost the same as completed videos.",
            "Use the strict official-result aggregator, not this monitor artifact, after result.json is complete.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "progress_by_video.csv", VIDEO_FIELDS, video_rows)
    _write_csv(out_dir / "progress_by_task.csv", TASK_FIELDS, task_rows)
    (out_dir / "progress_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_progress(summary=summary, video_rows=video_rows, output_stem=out_dir / "oasis_progress")
    return summary


def main() -> int:
    args = parse_args()
    summary = analyze_progress(
        output_path=args.output,
        metadata_path=args.metadata,
        preflight_path=args.preflight,
        out_dir=args.out_dir.resolve(),
        observed_at=args.observed_at,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
