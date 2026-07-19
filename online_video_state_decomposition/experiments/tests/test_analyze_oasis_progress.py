from __future__ import annotations

import csv
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import analyze_oasis_progress as progress  # noqa: E402


def _metadata(video_count: int = 3) -> list[dict]:
    rows = []
    for video_index in range(1, video_count + 1):
        breakpoints = []
        for question_index in range(1, 3):
            breakpoints.append(
                {
                    "question": f"question {video_index}-{question_index}",
                    "answer": "A",
                    "options": ["A", "B"],
                    "gt": "A",
                    "type": "multiple-choice",
                    "time": float(question_index),
                    "task": "action" if question_index == 1 else "scene",
                    "question_id": f"sample_{video_index}_{question_index}",
                }
            )
        rows.append(
            {
                "info": {
                    "video_path": (
                        "StreamingBench/Real-Time_Visual_Understanding/"
                        f"sample_{video_index}/video.mp4"
                    ),
                    "dataset": "StreamingBench",
                },
                "breakpoint": breakpoints,
            }
        )
    return rows


def _output(metadata: list[dict], completed: int) -> dict:
    results = copy.deepcopy(metadata[:completed])
    correct = 0
    scored = 0
    for video_index, item in enumerate(results):
        for question_index, breakpoint in enumerate(item["breakpoint"]):
            is_correct = (video_index + question_index) % 3 != 0
            breakpoint["response"] = "<answer>A</answer>"
            breakpoint["prediction"] = "A" if is_correct else "B"
            breakpoint["correct"] = is_correct
            scored += 1
            correct += int(is_correct)
    return {
        "results": results,
        "total_videos": completed,
        "mc_total": scored,
        "mc_correct": correct,
        "mc_accuracy": correct / scored,
    }


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _preflight(metadata_path: Path) -> dict:
    return {
        "method": "OASIS",
        "run_fingerprint": "a" * 64,
        "created_at": "2026-07-19T00:00:00+00:00",
        "metadata": {"path": str(metadata_path.resolve())},
    }


class AnalyzeOasisProgressTests(unittest.TestCase):
    def test_partial_snapshot_is_validated_but_not_comparison_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_payload = _metadata()
            metadata = _write(root / "metadata.json", metadata_payload)
            output = _write(root / "output.json", _output(metadata_payload, completed=2))
            preflight = _write(root / "preflight.json", _preflight(metadata))
            out_dir = root / "analysis"

            summary = progress.analyze_progress(
                output_path=output,
                metadata_path=metadata,
                preflight_path=preflight,
                out_dir=out_dir,
                observed_at="2026-07-19T00:20:00+00:00",
            )

            self.assertEqual(summary["status"], "partial_monitor")
            self.assertFalse(summary["formal_comparison_eligible"])
            self.assertIsNone(summary["formal_quality_accuracy"])
            self.assertEqual(summary["completed_videos"], 2)
            self.assertEqual(summary["expected_videos"], 3)
            self.assertEqual(summary["completed_questions"], 4)
            self.assertEqual(summary["correct"], 3)
            self.assertAlmostEqual(summary["estimated_remaining_seconds_linear"], 600.0)
            self.assertLess(summary["accuracy_wilson_95_low"], 0.75)
            self.assertGreater(summary["accuracy_wilson_95_high"], 0.75)

            for name in (
                "progress_summary.json",
                "progress_by_video.csv",
                "progress_by_task.csv",
                "oasis_progress.png",
                "oasis_progress.pdf",
            ):
                self.assertGreater((out_dir / name).stat().st_size, 0, name)
            with (out_dir / "progress_by_video.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["cumulative_questions_scored"], "4")

    def test_complete_snapshot_still_delegates_formal_use_to_aggregator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_payload = _metadata(video_count=1)
            metadata = _write(root / "metadata.json", metadata_payload)
            output = _write(root / "output.json", _output(metadata_payload, completed=1))
            preflight = _write(root / "preflight.json", _preflight(metadata))

            summary = progress.analyze_progress(
                output_path=output,
                metadata_path=metadata,
                preflight_path=preflight,
                out_dir=root / "analysis",
                observed_at="2026-07-19T00:01:00+00:00",
            )

            self.assertEqual(summary["status"], "complete_monitor")
            self.assertFalse(summary["formal_comparison_eligible"])
            self.assertEqual(summary["formal_quality_accuracy"], 0.5)
            self.assertFalse(summary["safe_resume_prefix"])

    def test_rejects_output_counter_mismatch_and_bad_observation_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_payload = _metadata()
            metadata = _write(root / "metadata.json", metadata_payload)
            bad_payload = _output(metadata_payload, completed=1)
            bad_payload["mc_correct"] += 1
            output = _write(root / "output.json", bad_payload)
            preflight = _write(root / "preflight.json", _preflight(metadata))
            with self.assertRaisesRegex(ValueError, "mc_correct is inconsistent"):
                progress.analyze_progress(
                    output_path=output,
                    metadata_path=metadata,
                    preflight_path=preflight,
                    out_dir=root / "bad-output",
                    observed_at="2026-07-19T00:01:00+00:00",
                )

            output = _write(root / "output.json", _output(metadata_payload, completed=1))
            with self.assertRaisesRegex(ValueError, "precedes"):
                progress.analyze_progress(
                    output_path=output,
                    metadata_path=metadata,
                    preflight_path=preflight,
                    out_dir=root / "bad-time",
                    observed_at="2026-07-18T23:59:59+00:00",
                )

    def test_wilson_interval_rejects_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            progress.wilson_interval(1, 0)
        with self.assertRaises(ValueError):
            progress.wilson_interval(2, 1)


if __name__ == "__main__":
    unittest.main()
