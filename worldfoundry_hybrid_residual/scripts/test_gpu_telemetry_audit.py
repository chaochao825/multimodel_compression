#!/usr/bin/env python3
"""Unit tests for GPU telemetry exclusivity auditing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audit_gpu_telemetry import audit_snapshots, read_snapshots


class GpuTelemetryAuditTests(unittest.TestCase):
    def test_foreign_overlap_on_used_gpu_invalidates_timing(self) -> None:
        text = """\
timestamp=2026-07-26T15:00:00+08:00
2, NVIDIA H200 NVL, 100, 143000, 1, 0
GPU-used, 101, /project/.venv/bin/python, 100
GPU-other, 999, /foreign/python, 200
timestamp=2026-07-26T15:00:05+08:00
GPU-used, 101, /project/.venv/bin/python, 100
GPU-used, 202, /foreign/python, 200
timestamp=2026-07-26T15:00:10+08:00
GPU-used, 101, /project/.venv/bin/python, 100
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.log"
            path.write_text(text, encoding="utf-8")
            result = audit_snapshots(
                read_snapshots(path), {"GPU-used"}, ("/project/.venv/bin/python",)
            )
        self.assertFalse(result["timing_valid"])
        self.assertEqual(result["snapshots_with_foreign_process"], 1)
        self.assertAlmostEqual(result["foreign_snapshot_fraction"], 1.0 / 3.0)
        self.assertEqual(result["estimated_foreign_overlap_seconds"], 5.0)
        self.assertEqual(result["foreign_processes"][0]["pid"], 202)

    def test_foreign_process_on_unused_gpu_is_ignored(self) -> None:
        text = """\
timestamp=2026-07-26T15:00:00+08:00
GPU-used, 101, /project/.venv/bin/python, 100
GPU-other, 202, /foreign/python, 200
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.log"
            path.write_text(text, encoding="utf-8")
            result = audit_snapshots(
                read_snapshots(path), {"GPU-used"}, ("/project/.venv/bin/python",)
            )
        self.assertTrue(result["timing_valid"])
        self.assertEqual(result["snapshots_with_foreign_process"], 0)


if __name__ == "__main__":
    unittest.main()
