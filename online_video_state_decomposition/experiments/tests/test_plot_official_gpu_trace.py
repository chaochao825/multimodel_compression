from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import plot_official_gpu_trace as gpu_trace  # noqa: E402


def _write_samples(path: Path, rows: list[tuple[float, int, int, int]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(gpu_trace.EXPECTED_FIELDS)
        writer.writerows(rows)
    return path


def _write_result(path: Path) -> Path:
    payload = {
        "status": "complete",
        "run_fingerprint": "fixture-fingerprint",
        "run_record": {
            "run_fingerprint": "fixture-fingerprint",
            "gpu_monitor": {
                "sample_count": 3,
                "gpu_peak_process_mib_sampled": 100,
                "gpu_peak_total_mib_sampled": 120,
                "gpu_peak_utilization_percent_sampled": 98,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class PlotOfficialGpuTraceTests(unittest.TestCase):
    def test_analyze_trace_writes_validated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = _write_samples(
                root / "samples.csv",
                [(10.0, 10, 0, 0), (11.0, 89, 50, 90), (12.5, 120, 98, 100)],
            )
            result = _write_result(root / "result.json")
            out = root / "out"

            summary = gpu_trace.analyze_trace(
                samples_path=samples,
                result_path=result,
                out_dir=out,
                stem="trace",
            )

            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["sampled_duration_seconds"], 2.5)
            self.assertEqual(summary["peak_process_memory_mib"], 100.0)
            self.assertEqual(summary["peak_gpu_memory_used_mib"], 120.0)
            self.assertEqual(summary["peak_gpu_utilization_percent"], 98.0)
            self.assertEqual(summary["run_fingerprint"], "fixture-fingerprint")
            for name in ("trace.csv", "trace.png", "trace.pdf", "trace_summary.json"):
                self.assertGreater((out / name).stat().st_size, 0, name)

            with (out / "trace.csv").open(encoding="utf-8", newline="") as handle:
                normalized = list(csv.DictReader(handle))
            self.assertEqual(normalized[0]["elapsed_seconds"], "0.0")
            self.assertEqual(normalized[-1]["elapsed_seconds"], "2.5")

    def test_rejects_non_monotonic_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            samples = _write_samples(
                Path(directory) / "samples.csv",
                [(10.0, 10, 0, 0), (10.0, 20, 10, 10)],
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                gpu_trace.load_samples(samples)

    def test_rejects_result_peak_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = _write_samples(
                root / "samples.csv",
                [(10.0, 10, 0, 0), (11.0, 100, 50, 90), (12.5, 120, 98, 99)],
            )
            result = _write_result(root / "result.json")
            with self.assertRaisesRegex(ValueError, "peak_process"):
                gpu_trace.analyze_trace(
                    samples_path=samples,
                    result_path=result,
                    out_dir=root / "out",
                    stem="trace",
                )


if __name__ == "__main__":
    unittest.main()
