from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_artifacts import SplitProtocol
from select_nystrom_sparse_tail import (
    DEPLOYABLE_METHODS,
    SelectionThresholds,
    frozen_calibration_role_heads,
    select_for_protocol,
    validate_rectangular_sweep,
)


def make_row(
    sample_id: str,
    method: str,
    residual_sq: float,
    work: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "sampling_step": 9,
        "branch": "cond",
        "layer": 14,
        "head": 3,
        "head_role_diagnostic_only": "transitional",
        "method": method,
        "landmark_mode": "segment",
        "landmarks": 64 if method == "nystrom_signed" else 32,
        "pinv_rtol": 1e-4,
        "density": 0.0 if method == "nystrom_signed" else 0.125,
        "deployable_candidate": True,
        "residual_sq": residual_sq,
        "reference_sq": 1.0,
        "output_relative_l2": residual_sq**0.5,
        "projected_attention_work_ratio": work,
        "arithmetic_speedup_upper_bound": 1.0 / work,
    }


class NystromSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = SplitProtocol(
            name="leakage_test",
            calibration=("cal",),
            validation=("val",),
            test=("test",),
            claim_boundary="unit test",
        )
        self.thresholds = SelectionThresholds(0.10, 0.10, 1.5, 0.5)
        self.rows = []
        for sample_id in self.protocol.sample_ids:
            self.rows.append(make_row(sample_id, "nystrom_signed", 0.0049, 0.4))
            residual = 0.0064 if sample_id != "test" else 0.0001
            self.rows.append(
                make_row(sample_id, "proxy_mass_nystrom_mixture", residual, 0.3)
            )

    def test_test_mutation_cannot_change_validation_selection(self) -> None:
        chosen_before, _, _ = select_for_protocol(
            self.rows, self.protocol, DEPLOYABLE_METHODS, self.thresholds
        )
        mutated = copy.deepcopy(self.rows)
        for row in mutated:
            if row["sample_id"] == "test":
                row["residual_sq"] = 100.0 if row["method"] == chosen_before.method else 0.0
                row["output_relative_l2"] = float(row["residual_sq"]) ** 0.5
        chosen_after, _, _ = select_for_protocol(
            mutated, self.protocol, DEPLOYABLE_METHODS, self.thresholds
        )
        self.assertEqual(chosen_before, chosen_after)

    def test_non_rectangular_sweep_is_rejected(self) -> None:
        incomplete = self.rows[:-1]
        with self.assertRaisesRegex(ValueError, "non-rectangular sweep"):
            validate_rectangular_sweep(incomplete)

    def test_test_roles_cannot_change_frozen_head_map(self) -> None:
        selected_before, _ = frozen_calibration_role_heads(
            self.rows, self.protocol, "transitional"
        )
        mutated = copy.deepcopy(self.rows)
        for row in mutated:
            if row["sample_id"] == "test":
                row["head_role_diagnostic_only"] = "diffuse"
        selected_after, _ = frozen_calibration_role_heads(
            mutated, self.protocol, "transitional"
        )
        self.assertEqual(selected_before, selected_after)


if __name__ == "__main__":
    unittest.main()
