from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENTS_ROOT.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import build_streaming_evidence_matrix as matrix  # noqa: E402


def _entry(entries: list[dict], evidence_id: str) -> dict:
    matches = [entry for entry in entries if entry["evidence_id"] == evidence_id]
    if len(matches) != 1:
        raise AssertionError(f"expected one evidence entry for {evidence_id}: {matches}")
    return matches[0]


class BuildStreamingEvidenceMatrixTests(unittest.TestCase):
    def test_repository_evidence_preserves_pass_fail_boundaries(self) -> None:
        evidence = matrix.build_evidence(PROJECT_ROOT)
        formal = _entry(evidence, "ours_query_selector_formal_200")
        self.assertEqual(formal["gate"]["status"], "FAIL")
        self.assertEqual(formal["metrics"][0]["value"], -0.005)

        native = _entry(evidence, "ours_native_query_memory_200")
        self.assertEqual(native["gate"]["status"], "OPEN")
        self.assertEqual(native["metrics"][0]["value"], 0.51)

        spectral = _entry(evidence, "ours_dual_spectral_controlled")
        self.assertEqual(spectral["gate"]["status"], "FAIL")
        self.assertAlmostEqual(spectral["metrics"][1]["value"], 0.0035714285714285713)

        codec = _entry(evidence, "ours_fixed_rank256_sparse4_200")
        self.assertEqual(codec["gate"]["status"], "FAIL")
        self.assertAlmostEqual(codec["metrics"][2]["value"], 0.023498470434950736)

        routed = _entry(evidence, "ours_routed_codec_posthoc_200")
        self.assertEqual(routed["gate"]["status"], "OPEN")
        self.assertAlmostEqual(routed["metrics"][1]["value"], 0.014867039231272083)

        independent = _entry(evidence, "ours_routed_codec_independent_300")
        self.assertEqual(independent["gate"]["status"], "PASS")
        self.assertTrue(independent["gate"]["valid_for_positive_claim"])
        self.assertAlmostEqual(
            independent["metrics"][3]["value"],
            0.015714554891583965,
        )

        selector = _entry(evidence, "ours_query_memory_independent_300")
        self.assertEqual(selector["gate"]["status"], "OPEN")
        self.assertEqual(selector["metrics"][0]["value"], 0.02)
        self.assertEqual(selector["metrics"][1]["value"], 0.109375)

        bccb = _entry(evidence, "ours_bccb_transport_formal30")
        self.assertEqual(bccb["gate"]["status"], "FAIL")
        self.assertLess(abs(bccb["metrics"][2]["value"]), 0.001)

        oasis = _entry(evidence, "oasis_official_smoke_1x5")
        self.assertEqual(oasis["gate"]["status"], "SMOKE_ONLY")
        self.assertFalse(oasis["gate"]["valid_for_positive_claim"])

        causalmem = _entry(evidence, "causalmem_official_quality_50x5")
        self.assertEqual(causalmem["gate"]["status"], "PASS")
        self.assertEqual(causalmem["metrics"][0]["value"], 0.824)

        stc = _entry(evidence, "stc_official_rekv_stage_pair_20")
        self.assertEqual(stc["gate"]["status"], "PASS")
        self.assertAlmostEqual(
            stc["metrics"][2]["value"],
            0.27647474450764886,
        )

    def test_runtime_overrides_are_strict_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "runtime.json"
            status_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "observed_at": "2026-07-19T04:20:00Z",
                        "records": [
                            {
                                "method_id": "oasis",
                                "stage": "official_quality",
                                "status": "RUNNING",
                                "detail": "3/50 complete",
                                "source_path": "/runtime/oasis",
                            },
                            {
                                "method_id": "streamingtom",
                                "stage": "official_latency",
                                "status": "QUEUED",
                                "detail": "waiting for GPU2",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows, audit = matrix.apply_runtime_status(
                matrix.base_completion_matrix(), status_path
            )
            index = {(row["method_id"], row["stage"]): row for row in rows}
            self.assertEqual(index[("oasis", "official_quality")]["status"], "RUNNING")
            self.assertEqual(index[("streamingtom", "official_latency")]["status"], "QUEUED")
            self.assertEqual(audit["record_count"], 2)
            self.assertEqual(len(audit["sha256"]), 64)

            bad_path = root / "bad.json"
            bad_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "observed_at": "2026-07-19T04:20:00Z",
                        "records": [
                            {
                                "method_id": "unknown",
                                "stage": "official_quality",
                                "status": "PASS",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown runtime status target"):
                matrix.apply_runtime_status(matrix.base_completion_matrix(), bad_path)

    def test_build_writes_json_csv_report_and_vector_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "matrix"
            summary = matrix.build_matrix(repo_root=PROJECT_ROOT, out_dir=out_dir)
            self.assertGreaterEqual(summary["evidence_count"], 17)
            self.assertEqual(summary["completion_cell_count"], 64)
            self.assertEqual(summary["status_counts"].get("FAIL", 0), 0)
            for name in (
                "evidence_matrix.json",
                "evidence_metrics.csv",
                "completion_matrix.csv",
                "streaming_evidence_completion_matrix.png",
                "streaming_evidence_completion_matrix.pdf",
                "EVIDENCE_MATRIX_ANALYSIS.md",
            ):
                self.assertGreater((out_dir / name).stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()
