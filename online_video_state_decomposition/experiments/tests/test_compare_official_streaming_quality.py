from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import compare_official_streaming_quality as comparison  # noqa: E402


def _write_oasis(path: Path, rows: list[dict]) -> Path:
    payload = {"results": [{"info": {"video_id": "v1"}, "breakpoint": rows}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_causalmem(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


class CompareOfficialStreamingQualityTests(unittest.TestCase):
    def test_pairs_answers_and_computes_exact_mcnemar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oasis_path = _write_oasis(
                root / "oasis.json",
                [
                    {"question_id": "q1", "task": "Action", "gt": "A", "correct": True},
                    {"question_id": "q2", "task": "Action", "gt": "B", "correct": True},
                    {
                        "question_id": "q3",
                        "task": "Object",
                        "gt": "C",
                        "correct": False,
                    },
                    {
                        "question_id": "q4",
                        "task": "Object",
                        "gt": "D",
                        "correct": False,
                    },
                ],
            )
            causalmem_path = _write_causalmem(
                root / "causalmem.jsonl",
                [
                    {"id": "q1", "task": "Action", "answer_id": "A", "acc": "True"},
                    {"id": "q2", "task": "Action", "answer_id": "B", "acc": "False"},
                    {"id": "q3", "task": "Object", "answer_id": "C", "acc": "True"},
                    {"id": "q4", "task": "Object", "answer_id": "D", "acc": "False"},
                ],
            )
            summary, questions, tasks = comparison.compare_answers(
                comparison.load_oasis_answers(oasis_path),
                comparison.load_causalmem_answers(causalmem_path),
            )
            self.assertEqual(summary["questions"], 4)
            self.assertEqual(summary["both_correct"], 1)
            self.assertEqual(summary["oasis_only_correct"], 1)
            self.assertEqual(summary["causalmem_only_correct"], 1)
            self.assertEqual(summary["both_wrong"], 1)
            self.assertEqual(summary["mcnemar_exact_p"], 1.0)
            self.assertEqual(len(questions), 4)
            self.assertEqual(len(tasks), 2)

    def test_rejects_mismatched_question_sets(self) -> None:
        oasis = {"q1": {"task": "Action", "ground_truth": "A", "correct": True}}
        causalmem = {"q2": {"task": "Action", "ground_truth": "A", "correct": True}}
        with self.assertRaisesRegex(ValueError, "question ID sets differ"):
            comparison.compare_answers(oasis, causalmem)


if __name__ == "__main__":
    unittest.main()
