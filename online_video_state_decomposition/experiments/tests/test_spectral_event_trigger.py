from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from probes.spectral_event_trigger import (  # noqa: E402
    SCENARIOS,
    generate_controlled_sequence,
    trace_controlled_sequence,
)


class SpectralEventTriggerTest(unittest.TestCase):
    def _sequence(self, scenario: str, *, noise_std: float = 0.0):
        return generate_controlled_sequence(
            scenario=scenario,
            seed=123,
            frames=24,
            height=6,
            width=6,
            hidden_dim=32,
            base_rank=4,
            event_frame=12,
            event_block_size=2,
            event_amplitude=8.0,
            noise_std=noise_std,
        )

    def _trace(self, scenario: str, *, noise_std: float = 0.0):
        return trace_controlled_sequence(
            self._sequence(scenario, noise_std=noise_std),
            total_rank_budget=8,
            storage_bits=16,
            fast_beta=0.8,
            slow_beta=0.98,
            fast_learning_rate=0.05,
            slow_learning_rate=0.005,
            causal_activity_beta=0.9,
            causal_max_new_directions=2,
        )

    def test_all_scenarios_have_expected_shape_and_events(self) -> None:
        for scenario in SCENARIOS:
            sequence = self._sequence(scenario)
            self.assertEqual(sequence.values.shape, (24, 36, 32))
            self.assertTrue(np.isfinite(sequence.values).all())
            if scenario in {
                "static",
                "camera_slow",
                "camera_fast",
                "lighting_drift",
                "periodic_motion",
            }:
                self.assertEqual(sequence.event_onsets, ())
            else:
                self.assertEqual(sequence.event_onsets, (12,))

    def test_covariance_trigger_is_invariant_to_cyclic_camera_motion(self) -> None:
        static = self._trace("static")
        camera = self._trace("camera_fast")
        for method in ("single_oja_residual", "dual_spectral"):
            static_scores = [
                float(row["residual_component"])
                for row in static
                if row["method"] == method and int(row["frame"]) >= 3
            ]
            camera_scores = [
                float(row["residual_component"])
                for row in camera
                if row["method"] == method and int(row["frame"]) >= 3
            ]
            np.testing.assert_allclose(static_scores, camera_scores, atol=1e-10)
        frame_delta = [
            float(row["raw_score"])
            for row in camera
            if row["method"] == "frame_delta" and int(row["frame"]) >= 3
        ]
        self.assertGreater(float(np.mean(frame_delta)), 0.1)

    def test_scene_cut_produces_large_slow_subspace_residual(self) -> None:
        rows = [
            row
            for row in self._trace("scene_cut")
            if row["method"] == "dual_spectral"
        ]
        before = [float(row["residual_component"]) for row in rows[4:12]]
        at_cut = float(rows[12]["residual_component"])
        self.assertGreater(at_cut, max(before) + 0.25)

    def test_disappearance_changes_fast_slow_spectrum(self) -> None:
        rows = [
            row
            for row in self._trace("object_disappear")
            if row["method"] == "dual_spectral"
        ]
        before = [float(row["spectrum_component"]) for row in rows[6:11]]
        after = [float(row["spectrum_component"]) for row in rows[12:15]]
        self.assertGreater(max(after), float(np.median(before)))

    def test_dual_and_single_basis_payloads_use_matched_rank_budget(self) -> None:
        rows = self._trace("static")
        first = {
            str(row["method"]): row
            for row in rows
            if int(row["frame"]) == 0
        }
        self.assertEqual(
            first["dual_spectral"]["state_bytes"],
            first["single_oja_residual"]["state_bytes"],
        )
        self.assertEqual(
            first["dual_spectral"]["state_bytes"],
            first["causalmem_residual_proxy"]["state_bytes"],
        )


if __name__ == "__main__":
    unittest.main()
