from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_wan_rcm_vae_cudagraph as cudagraph  # noqa: E402


class DecisionTest(unittest.TestCase):
    def test_component_requires_exactness_memory_and_both_speed_guards(self) -> None:
        self.assertEqual(
            cudagraph.component_decision(True, True, 1.12, 1.05, 1.12, 1.05),
            (True, "advance"),
        )
        self.assertEqual(
            cudagraph.component_decision(False, True, 2.0, 2.0, 1.12, 1.05),
            (False, "exactness-null"),
        )
        self.assertEqual(
            cudagraph.component_decision(True, False, 2.0, 2.0, 1.12, 1.05),
            (False, "invalid-memory"),
        )
        self.assertEqual(
            cudagraph.component_decision(True, True, 1.119, 1.05, 1.12, 1.05),
            (False, "performance-null"),
        )

    def test_request_uses_absolute_incumbent_threshold(self) -> None:
        self.assertEqual(
            cudagraph.request_decision(True, True, 9.1790428875, 9.1790428875),
            (True, "pass"),
        )
        self.assertEqual(
            cudagraph.request_decision(True, True, 9.18, 9.1790428875),
            (False, "speed-boundary"),
        )


class PrerequisiteTest(unittest.TestCase):
    def write_manifest(self, root: Path, stage: str, advance: bool) -> Path:
        path = root / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "experiment_id": "EXP-055",
                    "gate_id": "G-034",
                    "stage": stage,
                    "result": {"advance": advance},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_only_an_advancing_matching_stage_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_manifest(root, "f17-screen", True)
            self.assertTrue(
                cudagraph.validate_prerequisite(path, "f17-screen")["result"][
                    "advance"
                ]
            )
            path = self.write_manifest(root, "f17-screen", False)
            with self.assertRaises(ValueError):
                cudagraph.validate_prerequisite(path, "f17-screen")


class FrozenConfigTest(unittest.TestCase):
    def test_checked_in_config_matches_registered_identity(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "wan_rcm_vae_cudagraph_exp055_v1.json"
        )
        config, _base = cudagraph.load_configs(config_path)
        self.assertEqual(config["f17_replay_order"], [0, 1, 2, 1, 0])
        self.assertEqual(config["max_request_seconds"], 9.1790428875)


if __name__ == "__main__":
    unittest.main()
