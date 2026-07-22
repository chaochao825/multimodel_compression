from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import build_streaming_evidence_matrix as evidence_matrix  # noqa: E402
import collect_streaming_runtime_status as runtime_status  # noqa: E402


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _queue(run_dir: Path, status: str, *, pid_name: str = "pid") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "queue_status").write_text(status + "\n", encoding="utf-8")
    (run_dir / pid_name).write_text(str(os.getpid()) + "\n", encoding="utf-8")
    (run_dir / "idle_samples.log").write_text(
        "2026-07-19T00:00:00Z memory_mib=8192 utilization=0\n",
        encoding="utf-8",
    )


def _breakpoint(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "options": ["A", "B"],
        "gt": "A",
        "answer": "A",
        "type": "multiple_choice",
        "time": 1,
        "task": "Event Understanding",
    }


def _oasis_assets(root: Path) -> tuple[Path, Path]:
    metadata = [
        {"info": {"video_id": "v1"}, "breakpoint": [_breakpoint("q1")]},
        {"info": {"video_id": "v2"}, "breakpoint": [_breakpoint("q2")]},
    ]
    metadata_path = _write_json(root / "metadata.json", metadata)
    observed = {
        **_breakpoint("q1"),
        "response": "A",
        "prediction": "A",
        "correct": True,
    }
    output = {
        "results": [{"info": {"video_id": "v1"}, "breakpoint": [observed]}],
        "total_videos": 1,
        "mc_total": 1,
        "mc_correct": 1,
        "mc_accuracy": 1.0,
    }
    output_path = _write_json(root / "metadata_output.json", output)
    return metadata_path, output_path


def _streamingtom_preflight(method: str) -> dict:
    spec = runtime_status.STREAMINGTOM_SPECS[method]
    return {
        "format_version": 2,
        "evidence_tier": "official_core_gpu_microbenchmark",
        "method": method,
        "source": {
            "name": "streamingtom",
            "commit": runtime_status.STREAMINGTOM_COMMIT,
            "code_clean": True,
        },
        "frames": spec["frames"],
        "layers": 28,
        "warmup": 20,
        "repeat": 200,
        "dtype": "float16",
    }


class CollectStreamingRuntimeStatusTests(unittest.TestCase):
    @mock.patch.object(runtime_status, "_pid_alive", return_value=True)
    def test_collects_live_queues_and_valid_oasis_prefix(
        self, _pid_alive: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oasis = root / "oasis"
            stc = root / "stc"
            causalmem = root / "causalmem"
            streamingtom = root / "streamingtom"
            _queue(oasis, "launching", pid_name="queue_pid")
            metadata, output = _oasis_assets(oasis)
            _queue(stc, "waiting_for_idle")
            _queue(causalmem, "waiting_for_dependency")
            _queue(streamingtom, "waiting_for_idle")
            preflight = root / "preflight"
            for method in runtime_status.STREAMINGTOM_SPECS:
                _write_json(
                    preflight / f"{method}.json",
                    _streamingtom_preflight(method),
                )

            payload = runtime_status.collect_status(
                oasis_run=oasis,
                oasis_metadata=metadata,
                stc_run=stc,
                causalmem_run=causalmem,
                streamingtom_run=streamingtom,
                streamingtom_preflight_dir=preflight,
                observed_at="2026-07-19T00:00:00Z",
            )

            records = {
                (row["method_id"], row["stage"]): row for row in payload["records"]
            }
            self.assertEqual(
                records[("oasis", "official_quality")]["status"], "RUNNING"
            )
            self.assertIn(
                "1/2 videos", records[("oasis", "official_quality")]["detail"]
            )
            self.assertIn(
                "diagnostic only", records[("oasis", "official_quality")]["detail"]
            )
            self.assertEqual(
                records[("oasis", "official_quality")]["source_path"],
                str(output.resolve()),
            )
            self.assertEqual(records[("stc", "official_latency")]["status"], "QUEUED")
            self.assertEqual(
                records[("causalmem", "official_quality")]["status"], "QUEUED"
            )
            self.assertEqual(
                records[("streamingtom", "official_latency")]["status"], "QUEUED"
            )
            self.assertEqual(
                records[("streamingtom", "runtime_preflight")]["status"], "PASS"
            )
            self.assertEqual(payload["validation_errors"], [])
            self.assertEqual(
                payload["diagnostics"]["oasis"]["progress"]["completed_questions"],
                1,
            )
            self.assertEqual(_pid_alive.call_count, 4)

            runtime_path = _write_json(root / "runtime_status.json", payload)
            matrix = [
                {
                    "method_id": method_id,
                    "stage": stage,
                    "status": "OPEN",
                    "detail": "",
                    "source_path": "",
                }
                for method_id, stage in records
            ]
            updated, audit = evidence_matrix.apply_runtime_status(matrix, runtime_path)
            self.assertEqual(len(updated), 5)
            self.assertEqual(audit["record_count"], 5)

    @mock.patch.object(runtime_status, "_pid_alive", return_value=False)
    def test_marks_nonterminal_queue_with_dead_pid_failed(
        self, _pid_alive: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "streamingtom"
            _queue(run_dir, "running_ctr")
            payload = runtime_status.collect_status(
                oasis_run=None,
                oasis_metadata=None,
                stc_run=None,
                causalmem_run=None,
                streamingtom_run=run_dir,
                streamingtom_preflight_dir=None,
                observed_at="2026-07-19T00:00:00+00:00",
            )
            record = payload["records"][0]
            self.assertEqual(record["status"], "FAIL")
            self.assertIn("is not alive", record["detail"])
            self.assertEqual(len(payload["validation_errors"]), 1)
            _pid_alive.assert_called_once()

    def test_complete_status_requires_strict_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "stc"
            _queue(run_dir, "complete")
            payload = runtime_status.collect_status(
                oasis_run=None,
                oasis_metadata=None,
                stc_run=run_dir,
                causalmem_run=None,
                streamingtom_run=None,
                streamingtom_preflight_dir=None,
                observed_at="2026-07-19T00:00:00Z",
            )
            record = payload["records"][0]
            self.assertEqual(record["status"], "FAIL")
            self.assertIn("artifact validation", record["detail"])

    @mock.patch.object(runtime_status, "_pid_alive", return_value=True)
    def test_supports_actual_queue_status_vocabulary(
        self, pid_alive: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validator = mock.Mock(return_value="strict result passed")
            expected = {
                "completed": "PASS",
                "waiting_for_flash_attn": "QUEUED",
                "failed_dependency": "FAIL",
            }
            for status_value, expected_status in expected.items():
                run_dir = root / status_value
                _queue(run_dir, status_value)
                status, _, _, _, _ = runtime_status._queue_snapshot(
                    run_dir,
                    pid_candidates=("pid",),
                    complete_validator=validator,
                )
                self.assertEqual(status, expected_status)
            validator.assert_called_once_with()
            pid_alive.assert_called_once_with(os.getpid())

    def test_rejects_missing_metadata_or_naive_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "oasis"
            with self.assertRaisesRegex(ValueError, "oasis-metadata"):
                runtime_status.collect_status(
                    oasis_run=run_dir,
                    oasis_metadata=None,
                    stc_run=None,
                    causalmem_run=None,
                    streamingtom_run=None,
                    streamingtom_preflight_dir=None,
                    observed_at=None,
                )
            with self.assertRaisesRegex(ValueError, "UTC offset"):
                runtime_status.collect_status(
                    oasis_run=None,
                    oasis_metadata=None,
                    stc_run=run_dir,
                    causalmem_run=None,
                    streamingtom_run=None,
                    streamingtom_preflight_dir=None,
                    observed_at="2026-07-19T00:00:00",
                )


if __name__ == "__main__":
    unittest.main()
