from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_artifacts import (
    SplitProtocol,
    canonical_json,
    object_sha256,
    require_fresh_output_dir,
)


class ExperimentArtifactTests(unittest.TestCase):
    def test_protocol_rejects_cross_split_duplicate(self) -> None:
        protocol = SplitProtocol(
            name="bad",
            calibration=("a",),
            validation=("b",),
            test=("a",),
            claim_boundary="test",
        )
        with self.assertRaisesRegex(ValueError, "more than once"):
            protocol.validate()

    def test_protocol_requires_exact_observed_coverage(self) -> None:
        protocol = SplitProtocol(
            name="strict",
            calibration=("a",),
            validation=("b",),
            test=("c",),
            claim_boundary="test",
        )
        protocol.validate()
        protocol.assert_exact_coverage(("a", "b", "c"))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            protocol.assert_exact_coverage(("a", "b", "c", "d"))

    def test_config_hash_is_order_independent(self) -> None:
        left = {"b": [2, 3], "a": 1}
        right = {"a": 1, "b": [2, 3]}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(object_sha256(left), object_sha256(right))

    def test_nonempty_output_directory_is_rejected(self) -> None:
        path = Mock()
        path.exists.return_value = True
        path.iterdir.return_value = iter((Mock(),))
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            require_fresh_output_dir(path)
        path.mkdir.assert_not_called()

    def test_empty_output_directory_is_accepted(self) -> None:
        path = Mock()
        path.exists.return_value = True
        path.iterdir.return_value = iter(())
        require_fresh_output_dir(path)
        path.mkdir.assert_called_once_with(parents=True, exist_ok=True)


if __name__ == "__main__":
    unittest.main()
