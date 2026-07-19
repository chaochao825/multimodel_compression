from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import materialize_oasis_dataset_links as materializer  # noqa: E402


def _write_manifest(root: Path, source: Path) -> Path:
    manifest = {
        "format_version": 1,
        "formal_contract": False,
        "subset_metadata_sha256": "metadata-fixture",
        "video_count": 1,
        "videos": [
            {
                "video_id": 1,
                "relative_path": (
                    "StreamingBench/Real-Time_Visual_Understanding/"
                    "sample_1/video.mp4"
                ),
                "source_path": str(source),
                "size_bytes": source.stat().st_size,
                "crc32": materializer.crc32_file(source),
            }
        ],
    }
    path = root / "mapping.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@unittest.skipIf(os.name == "nt" and not hasattr(os, "symlink"), "no symlink support")
class MaterializeOasisDatasetLinksTests(unittest.TestCase):
    def test_materializes_and_reuses_exact_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"video fixture")
            manifest = _write_manifest(root, source)
            dataset_root = root / "dataset"
            output = root / "materialized.json"

            first = materializer.materialize_links(
                mapping_manifest=manifest,
                dataset_root=dataset_root,
                output_manifest=output,
            )
            self.assertEqual(first["created_links"], 1)
            target = dataset_root / first["links"][0]["relative_path"]
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), source.resolve())

            second = materializer.materialize_links(
                mapping_manifest=manifest,
                dataset_root=dataset_root,
                output_manifest=output,
            )
            self.assertEqual(second["created_links"], 0)
            self.assertEqual(second["reused_links"], 1)

    def test_refuses_conflicting_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"video fixture")
            manifest = _write_manifest(root, source)
            dataset_root = root / "dataset"
            target = dataset_root / (
                "StreamingBench/Real-Time_Visual_Understanding/sample_1/video.mp4"
            )
            target.parent.mkdir(parents=True)
            target.write_bytes(b"different")
            with self.assertRaisesRegex(FileExistsError, "different content"):
                materializer.materialize_links(
                    mapping_manifest=manifest,
                    dataset_root=dataset_root,
                    output_manifest=root / "out.json",
                )

    def test_rejects_unsafe_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"video fixture")
            manifest_path = _write_manifest(root, source)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["videos"][0]["relative_path"] = "../escape.mp4"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe relative path"):
                materializer.build_link_plan(manifest_path, root / "dataset")


if __name__ == "__main__":
    unittest.main()
