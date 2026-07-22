from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import aggregate_official_streaming_results as aggregate  # noqa: E402


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _monitor(peak: int) -> dict:
    return {"gpu_peak_process_mib_sampled": peak}


def _causalmem() -> dict:
    return {
        "format_version": 2,
        "evidence_tier": "official_model_level_streamingbench_rt_1_50",
        "run_fingerprint": "causal-fingerprint",
        "method": "causal_mem",
        "returncode": 0,
        "success": True,
        "wall_seconds": 120.0,
        "gpu_monitor": _monitor(24_000),
        "quality": {
            "parse_errors": 0,
            "invalid_records": [],
            "duplicate_ids": [],
            "unexpected_ids": [],
            "missing_question_ids": [],
            "completed_questions": 250,
            "expected_questions": 250,
            "correct": 200,
            "accuracy": 0.8,
        },
        "latency_scope": {"per_sample_p50_p95_p99_available": False},
    }


def _stage(scale: float) -> dict:
    return {
        "count": 3,
        "min": 1.0 * scale,
        "p50": 2.0 * scale,
        "p95": 3.0 * scale,
        "p99": 3.0 * scale,
        "mean": 2.0 * scale,
        "std": 0.5 * scale,
        "max": 3.0 * scale,
    }


def _stc(mode: str) -> dict:
    return {
        "format_version": 1,
        "status": "complete",
        "run_fingerprint": f"stc-{mode}-fingerprint",
        "mode": mode,
        "derived": {
            "vit_encode_ms": _stage(1.0),
            "llm_prefill_ms": _stage(2.0),
            "instrumented_stage_sum_ms": _stage(3.0),
            "peak_mem_gb_official": 18.5,
        },
    }


def _oasis(*, expected: int, correct: int) -> dict:
    return {
        "format_version": 1,
        "status": "complete",
        "run_fingerprint": f"oasis-{expected}-fingerprint",
        "metrics": {
            "complete": True,
            "errors": [],
            "scored_questions": expected,
            "expected_questions": expected,
            "correct": correct,
            "accuracy": correct / expected,
            "scored_coverage": 1.0,
        },
        "run_record": {
            "run_fingerprint": f"oasis-{expected}-fingerprint",
            "elapsed_wall_seconds": 300.0,
            "gpu_monitor": _monitor(30_000),
        },
    }


def _distribution(scale: float) -> dict:
    return {
        "count": 200.0,
        "min": 1.0 * scale,
        "p50": 2.0 * scale,
        "p95": 3.0 * scale,
        "p99": 4.0 * scale,
        "mean": 2.5 * scale,
        "max": 5.0 * scale,
    }


def _streamingtom(method: str) -> dict:
    frames = {
        "streamingtom_ctr": 64,
        "streamingtom_oqm_write": 64,
        "streamingtom_oqm_select": 256,
    }[method]
    return {
        "format_version": 2,
        "evidence_tier": "official_core_gpu_microbenchmark",
        "method": method,
        "source": {
            "name": "streamingtom",
            "commit": aggregate.STREAMINGTOM_COMMIT,
            "code_clean": True,
        },
        "frames": frames,
        "layers": 28,
        "warmup": 20,
        "repeat": 200,
        "dtype": "float16",
        "started_at_utc": "2026-07-19T00:00:00+00:00",
        "finished_at_utc": "2026-07-19T00:01:00+00:00",
        "quality_gate": {"passed": True},
        "wall_ms": _distribution(1.1),
        "cuda_event_ms": _distribution(1.0),
        "peak_allocated_mib": _distribution(20.0),
        "peak_reserved_mib": _distribution(25.0),
        "peak_allocated_delta_mib": _distribution(4.0),
        "peak_reserved_delta_mib": _distribution(5.0),
        "tail_latency_protocol": {
            "quantile_method": "higher",
            "global_cuda_synchronize_per_iteration": True,
            "input_preparation_timed": False,
            "model_loading_timed": False,
        },
    }


class AggregateOfficialStreamingResultsTests(unittest.TestCase):
    def test_aggregate_separates_formal_smoke_and_stage_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            causalmem = _write(root / "causalmem" / "metrics.json", _causalmem())
            stc_rekv = _write(root / "stc_rekv" / "result.json", _stc("rekv"))
            stc_stc = _write(
                root / "stc_stc" / "result.json", _stc("rekv_stc")
            )
            oasis_formal = _write(
                root / "oasis_formal" / "result.json", _oasis(expected=250, correct=190)
            )
            oasis_smoke = _write(
                root / "oasis_smoke" / "result.json", _oasis(expected=5, correct=4)
            )
            streamingtom = [
                _write(
                    root / method / "summary.json",
                    _streamingtom(method),
                )
                for method in aggregate.STREAMINGTOM_SPECS
            ]
            out = root / "aggregate"

            summary = aggregate.aggregate_results(
                causalmem_metrics=[causalmem],
                stc_results=[stc_rekv, stc_stc],
                oasis_results=[oasis_formal, oasis_smoke],
                streamingtom_summaries=streamingtom,
                out_dir=out,
            )

            self.assertEqual(summary["run_count"], 8)
            self.assertEqual(summary["formal_quality_run_count"], 2)
            self.assertEqual(summary["smoke_quality_run_count"], 1)
            self.assertEqual(summary["stc_stage_row_count"], 6)
            self.assertEqual(summary["streamingtom_core_run_count"], 3)
            self.assertEqual(summary["streamingtom_core_row_count"], 6)
            self.assertTrue(summary["streamingtom_core_complete_set"])
            self.assertEqual(
                {row["method"] for row in summary["quality_formal"]},
                {"CausalMem", "OASIS"},
            )
            self.assertEqual(summary["quality_smoke"][0]["method"], "OASIS")
            self.assertTrue(
                all(row["method"] == "STC ReKV" for row in summary["stc_stage_latency"])
            )
            self.assertEqual(
                {row["mode"] for row in summary["stc_stage_latency"]},
                {"rekv", "stc"},
            )
            for name in (
                "aggregation_summary.json",
                "official_runs.csv",
                "official_quality_formal.csv",
                "official_quality_smoke.csv",
                "official_stc_stage_latency.csv",
                "official_quality_formal.png",
                "official_quality_formal.pdf",
                "official_quality_smoke.png",
                "official_quality_smoke.pdf",
                "official_stc_stage_latency.png",
                "official_stc_stage_latency.pdf",
                "official_streamingtom_core_latency.csv",
                "official_streamingtom_core_latency.png",
                "official_streamingtom_core_latency.pdf",
            ):
                self.assertGreater((out / name).stat().st_size, 0, name)

            with (out / "official_runs.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                runs = list(csv.DictReader(handle))
            causal_row = next(row for row in runs if row["method"] == "CausalMem")
            stc_row = next(row for row in runs if row["method"] == "STC ReKV")
            streamingtom_row = next(
                row for row in runs if row["method"] == "StreamingTOM"
            )
            self.assertEqual(causal_row["tail_latency_available"], "False")
            self.assertEqual(stc_row["tail_latency_available"], "False")
            self.assertEqual(stc_row["stage_latency_available"], "True")
            self.assertEqual(stc_row["quality_accuracy"], "")
            self.assertEqual(streamingtom_row["role"], "core_latency")
            self.assertEqual(streamingtom_row["tail_latency_available"], "False")

            with (out / "official_streamingtom_core_latency.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                core_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["timing_basis"] for row in core_rows},
                {"cuda_event", "synchronized_host_wall"},
            )
            self.assertEqual({int(row["count"]) for row in core_rows}, {200})

    def test_rejects_incomplete_or_failed_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            causalmem_payload = _causalmem()
            causalmem_payload["quality"]["missing_question_ids"] = ["missing"]
            causalmem = _write(root / "causalmem.json", causalmem_payload)
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                aggregate.parse_causalmem(causalmem)

            oasis_payload = _oasis(expected=5, correct=4)
            oasis_payload["metrics"]["errors"] = [{"question_id": "q1"}]
            oasis = _write(root / "oasis.json", oasis_payload)
            with self.assertRaisesRegex(ValueError, "failed questions"):
                aggregate.parse_oasis(oasis)

            streamingtom_payload = _streamingtom("streamingtom_ctr")
            streamingtom_payload["quality_gate"]["passed"] = False
            streamingtom = _write(root / "streamingtom.json", streamingtom_payload)
            with self.assertRaisesRegex(ValueError, "quality gate failed"):
                aggregate.parse_streamingtom(streamingtom)

    def test_rejects_streamingtom_protocol_or_source_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _streamingtom("streamingtom_oqm_select")
            payload["frames"] = 64
            path = _write(root / "frames.json", payload)
            with self.assertRaisesRegex(ValueError, "formal protocol mismatch"):
                aggregate.parse_streamingtom(path)

            payload = _streamingtom("streamingtom_oqm_select")
            payload["source"]["commit"] = "wrong"
            path = _write(root / "source.json", payload)
            with self.assertRaisesRegex(ValueError, "source audit failed"):
                aggregate.parse_streamingtom(path)

            payload = _streamingtom("streamingtom_oqm_select")
            payload["cuda_event_ms"]["count"] = 199.0
            path = _write(root / "count.json", payload)
            with self.assertRaisesRegex(ValueError, "must match repeat"):
                aggregate.parse_streamingtom(path)

    def test_requires_at_least_one_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one"):
                aggregate.aggregate_results(
                    causalmem_metrics=[],
                    stc_results=[],
                    oasis_results=[],
                    streamingtom_summaries=[],
                    out_dir=Path(directory),
                )


if __name__ == "__main__":
    unittest.main()
