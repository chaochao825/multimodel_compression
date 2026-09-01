from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_wan_rcm_attention_atlas as runner  # noqa: E402


class AttentionAtlasConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "wan_rcm_onpolicy_attention_exp054_v1.json"
        )

    def test_frozen_config_loads_with_disjoint_splits(self) -> None:
        config, base = runner.load_configs(self.path)
        self.assertEqual(config["experiment_id"], "EXP-054")
        self.assertEqual(base["methods"]["rcm4"]["num_steps"], 4)
        self.assertEqual(len(runner._identity_ids(config, "calibration")), 4)
        self.assertEqual(len(runner._identity_ids(config, "evaluation")), 4)

    def test_operator_mutation_is_rejected(self) -> None:
        config = json.loads(self.path.read_text(encoding="utf-8"))
        config["operator"]["smooth_k"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "operator contract"):
                runner.load_configs(path)

    def test_identity_overlap_is_rejected(self) -> None:
        config = json.loads(self.path.read_text(encoding="utf-8"))
        config["atlas_identities"][4]["identity"] = config["atlas_identities"][0][
            "identity"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                runner.load_configs(path)


if __name__ == "__main__":
    unittest.main()
