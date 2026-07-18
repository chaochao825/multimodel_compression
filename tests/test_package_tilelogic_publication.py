from __future__ import annotations

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.package_tilelogic_publication import (
    CORRECTED_RATE_SENTINELS,
    FINAL_RELEASE_PASS_MARKER,
    REQUIRED_FILES,
    SOURCE_REVIEW_PASS_MARKER,
    _validate_public_provenance,
    _validate_publication_source,
    main,
    sanitize_json,
    sanitize_string,
    sanitize_text,
)


class TileLogicPublicationTest(unittest.TestCase):
    def _write_publication_fixture(self, run_dir: Path) -> None:
        for relative in REQUIRED_FILES:
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text("{}\n", encoding="utf-8")
            elif path.suffix == ".csv":
                path.write_text("value\n1\n", encoding="utf-8")
            else:
                path.write_text("# Evidence\n", encoding="utf-8")
        (run_dir / "rate_precision_correction_validation.json").write_text(
            json.dumps(
                {
                    "records": 360,
                    "compared_variants": 360 * 23,
                    "non_rate_semantics_identical": True,
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
        source_hash = "a" * 64
        (run_dir / "training/training_summary.json").write_text(
            json.dumps({"cache_manifest_sha256": source_hash}),
            encoding="utf-8",
        )
        (run_dir / "cache_provenance_backfill.json").write_text(
            json.dumps(
                {
                    "records": 600,
                    "payload_tensors_unchanged": True,
                    "training_summary_sha256": source_hash,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "cache/cache_summary.json").write_text(
            json.dumps(
                {
                    "cache_manifest_sha256": source_hash,
                    "split_manifest_sha256": source_hash,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "feature_eval/feature_eval_summary.json").write_text(
            json.dumps(
                {
                    "cache_manifest_sha256": source_hash,
                    "training_summary_sha256": source_hash,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "quality/quality_environment.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": source_hash,
                    "cache_manifest_sha256": source_hash,
                    "feature_eval_summary_sha256": source_hash,
                }
            ),
            encoding="utf-8",
        )
        log_path = run_dir / "latency/gpu_co_residency_during_run.log"
        (run_dir / "latency/latency_environment.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": source_hash,
                    "gpu_co_residency_log": {
                        "file": log_path.name,
                        "bytes": log_path.stat().st_size,
                        "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "analysis/TILELOGIC_RVQ_FINAL_REPORT.md").write_text(
            "# Report\n\n## 8. Evidence Boundaries\n",
            encoding="utf-8",
        )
        (run_dir / "analysis/independent_review_report.md").write_text(
            f"# Review\n\n{SOURCE_REVIEW_PASS_MARKER}\n\n"
            f"{FINAL_RELEASE_PASS_MARKER}\n\n"
            "[Machine audit](result_audit_report.md)\n",
            encoding="utf-8",
        )
        audit_checks = [
            {"check": f"check_{index}", "status": "PASS"} for index in range(22)
        ]
        (run_dir / "analysis/result_audit_findings.json").write_text(
            json.dumps(
                {"overall": "PASS", "major_failures": 0, "checks": audit_checks}
            ),
            encoding="utf-8",
        )
        (run_dir / "analysis/decision_summary.json").write_text(
            json.dumps(
                {
                    "questions": [
                        {"id": 1, "status": "FAIL", "question": "Frozen question"}
                    ],
                    "aggregate_positive_claim_allowed": False,
                }
            ),
            encoding="utf-8",
        )
        with (run_dir / "analysis/method_points.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "method",
                    "retention_rate",
                    "effective_bits_per_original_value",
                ),
            )
            writer.writeheader()
            for (method, rate), value in CORRECTED_RATE_SENTINELS.items():
                writer.writerow(
                    {
                        "method": method,
                        "retention_rate": rate,
                        "effective_bits_per_original_value": value,
                    }
                )
        figure = run_dir / "analysis/figures/test.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_bytes(b"publication-test-image")

    def test_sanitize_json_redacts_model_and_private_paths(self) -> None:
        payload = {
            "model_dir": "/home/user/cache/model-snapshot",
            "cache_dir": "/home/user/work/run/cache",
            "nested": ["report at /data6/private/result.csv"],
        }
        sanitized = sanitize_json(payload)
        self.assertEqual(
            sanitized["model_dir"], "external://models/model-snapshot"
        )
        self.assertEqual(sanitized["cache_dir"], "external://private/cache")
        self.assertEqual(
            sanitized["nested"], ["report at external://private/result.csv"]
        )

    def test_sanitize_string_leaves_relative_paths_unchanged(self) -> None:
        self.assertEqual(
            sanitize_string("analysis/decision_summary.json"),
            "analysis/decision_summary.json",
        )

    def test_sanitize_text_strips_trailing_space_and_preserves_final_newline(self) -> None:
        self.assertEqual(sanitize_text("alpha  \n beta\t\n"), "alpha\n beta\n")

    def test_end_to_end_package_validates_required_evidence_hashes_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            output_dir = root / "public"
            self._write_publication_fixture(run_dir)
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "package_tilelogic_publication.py",
                        "--run-dir",
                        str(run_dir),
                        "--output-dir",
                        str(output_dir),
                        "--final-release",
                    ],
                ),
                redirect_stdout(io.StringIO()),
            ):
                main()
            manifest = json.loads(
                (output_dir / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["private_path_scan_pass"])
            self.assertTrue(manifest["binary_private_marker_scan_pass"])
            self.assertTrue(manifest["markdown_link_scan_pass"])
            self.assertTrue(manifest["source_public_provenance_scan_pass"])
            self.assertTrue(manifest["release_ready"])
            self.assertEqual(manifest["review_stage"], "final")
            indexed = {item["path"]: item for item in manifest["files"]}
            for relative in REQUIRED_FILES:
                self.assertIn(relative, indexed)
            for relative, item in indexed.items():
                path = output_dir / relative
                self.assertEqual(path.stat().st_size, item["bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            _validate_public_provenance(output_dir)

    def test_source_validation_rejects_pre_correction_rate_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            self._write_publication_fixture(run_dir)
            path = run_dir / "analysis/method_points.csv"
            text = path.read_text(encoding="utf-8").replace(
                "0.501953125", "0.5009765625"
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "rate sentinels"):
                _validate_publication_source(run_dir)

    def test_final_release_requires_bundle_review_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            self._write_publication_fixture(run_dir)
            report = run_dir / "analysis/independent_review_report.md"
            report.write_text(
                report.read_text(encoding="utf-8").replace(
                    FINAL_RELEASE_PASS_MARKER, "Final release verdict: **PENDING**"
                ),
                encoding="utf-8",
            )
            _validate_publication_source(run_dir)
            with self.assertRaisesRegex(RuntimeError, "generated bundle"):
                _validate_publication_source(run_dir, final_release=True)

    def test_public_provenance_rejects_stale_embedded_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            output_dir = root / "public"
            self._write_publication_fixture(run_dir)
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "package_tilelogic_publication.py",
                        "--run-dir",
                        str(run_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
                redirect_stdout(io.StringIO()),
            ):
                main()
            environment_path = output_dir / "latency/latency_environment.json"
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment["gpu_co_residency_log"]["public_sha256"] = "0" * 64
            environment_path.write_text(json.dumps(environment), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "public provenance"):
                _validate_public_provenance(output_dir)


if __name__ == "__main__":
    unittest.main()
