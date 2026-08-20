#!/usr/bin/env python3
"""High-value synthetic tests for EXP-045 causality and geometry."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from analyze_wan_current_input_observability import (
    aggregate_output_relative_l2,
    build_gate_tables,
)
from current_input_observability_core import (
    CellTrajectory,
    ScalarAR2Statistics,
    nonperiodic_shift,
    oracle_recovery_fraction,
    relative_l2_terms,
    rollout_predict,
)
from wan_current_input_observability_runtime import (
    WanCurrentInputObservabilityRecorder,
)


def make_trajectory(block_input: torch.Tensor, residual: torch.Tensor) -> CellTrajectory:
    steps, tokens, channels = block_input.shape
    return CellTrajectory(
        block_input=block_input,
        residual=residual,
        adaln=torch.zeros(steps, 6, channels),
        qk_sketch=block_input[..., : min(4, channels)].clone(),
        thw=(1, 1, tokens),
    )


class ObservabilityCoreTests(unittest.TestCase):
    def test_output_error_aggregation_uses_output_denominator(self) -> None:
        rows = [
            {"output_relative_l2": 0.1, "output_target_sq": 1.0},
            {"output_relative_l2": 0.2, "output_target_sq": 3.0},
        ]
        expected = ((0.1**2 + 3.0 * 0.2**2) / 4.0) ** 0.5
        self.assertAlmostEqual(aggregate_output_relative_l2(rows), expected)

    def test_nonperiodic_shift_does_not_wrap(self) -> None:
        values = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        shifted, mask = nonperiodic_shift(values, (1, 1, 4), (0, 0, 1))
        torch.testing.assert_close(
            shifted[:, 0], torch.tensor([0.0, 0.0, 1.0, 2.0])
        )
        torch.testing.assert_close(mask[:, 0], torch.tensor([0.0, 1.0, 1.0, 1.0]))

    def test_open_loop_never_reads_intermediate_exact_residual(self) -> None:
        generator = torch.Generator().manual_seed(7)
        block_input = torch.randn(6, 12, 5, generator=generator).cumsum(dim=0)
        residual = torch.randn(6, 12, 5, generator=generator).cumsum(dim=0)
        original = make_trajectory(block_input, residual)
        changed_residual = residual.clone()
        changed_residual[4:] = changed_residual[4:] + 1000
        changed = make_trajectory(block_input, changed_residual)
        for method in ("taylor1", "diagonal", "broyden2"):
            expected = rollout_predict(
                original, method=method, target_step=5, horizon=2
            ).prediction
            observed = rollout_predict(
                changed, method=method, target_step=5, horizon=2
            ).prediction
            torch.testing.assert_close(observed, expected)

    def test_broyden_recovers_a_secant_span(self) -> None:
        generator = torch.Generator().manual_seed(11)
        base = torch.randn(20, 6, generator=generator)
        drifts = [torch.randn(20, 6, generator=generator) for _ in range(4)]
        drifts.append(0.35 * drifts[1] + 0.65 * drifts[3])
        matrix = torch.randn(6, 6, generator=generator) / 4
        inputs = [base]
        residuals = [torch.randn(20, 6, generator=generator)]
        for drift in drifts:
            inputs.append(inputs[-1] + drift)
            residuals.append(residuals[-1] + drift @ matrix.T)
        trajectory = make_trajectory(torch.stack(inputs), torch.stack(residuals))
        result = rollout_predict(
            trajectory, method="broyden4", target_step=5, horizon=1, ridge=1e-7
        )
        relative, _, _ = relative_l2_terms(result.prediction, trajectory.residual[5])
        self.assertLess(relative, 1e-4)
        self.assertEqual(result.effective_secants, 4)

    def test_transport_operates_on_tokens_not_hidden_channels(self) -> None:
        generator = torch.Generator().manual_seed(13)
        initial = torch.randn(8, 4, generator=generator)
        residuals = [initial]
        for _ in range(5):
            shifted, _ = nonperiodic_shift(residuals[-1], (1, 1, 8), (0, 0, 1))
            residuals.append(shifted)
        block_input = torch.zeros(6, 8, 4)
        trajectory = CellTrajectory(
            block_input=block_input,
            residual=torch.stack(residuals),
            adaln=torch.zeros(6, 6, 4),
            qk_sketch=torch.zeros(6, 8, 2),
            thw=(1, 1, 8),
        )
        result = rollout_predict(
            trajectory,
            method="transport1_history",
            target_step=5,
            horizon=1,
            ridge=1e-6,
        )
        relative, _, _ = relative_l2_terms(result.prediction, trajectory.residual[5])
        self.assertLess(relative, 1e-3)
        self.assertEqual(result.shifts, ((0, 0, 1),))

    def test_oracle_recovery_uses_linear_risk_gap(self) -> None:
        self.assertAlmostEqual(oracle_recovery_fraction(0.20, 0.14, 0.08), 0.5)
        self.assertEqual(
            oracle_recovery_fraction(0.20, 0.14, 0.20), float("-inf")
        )

    def test_calibrated_scalar_ar2_recovers_known_coefficients(self) -> None:
        generator = torch.Generator().manual_seed(17)
        lag2 = torch.randn(6, 10, generator=generator)
        lag1 = torch.randn(6, 10, generator=generator)
        target = 1.25 * lag1 - 0.35 * lag2
        statistics = ScalarAR2Statistics()
        statistics.update(target, lag1, lag2)
        fitted = statistics.fit(1e-8)
        self.assertAlmostEqual(fitted.lag1, 1.25, places=5)
        self.assertAlmostEqual(fitted.lag2, -0.35, places=5)

    def test_gate_rejects_one_step_gain_with_open_loop_instability(self) -> None:
        config = {
            "target_steps": [4, 6],
            "blocks": list(range(20, 30)),
            "branches": [0, 1],
            "methods": ["ar2", "good", "unstable"],
            "gate_candidate_methods": ["good", "unstable"],
            "gate": {
                "minimum_risk_reduction": 2.0,
                "minimum_passing_layers_per_step": 6,
                "minimum_oracle_recovery": 0.5,
                "maximum_branch_harm_ratio": 1.1,
                "maximum_open_loop_risk_ratio": 1.0,
            },
        }
        rows: list[dict[str, object]] = []
        risks = {
            "ar2": {1: 0.20, 2: 0.20, 3: 0.20},
            "good": {1: 0.08, 2: 0.18, 3: 0.19},
            "unstable": {1: 0.08, 2: 0.21, 3: 0.22},
            "oracle_transport75_token_ls": {1: 0.04},
        }
        for sample_index in range(4):
            for block in range(20, 30):
                for target_step in (4, 6):
                    for branch in (0, 1):
                        for method, horizon_risks in risks.items():
                            for horizon, risk in horizon_risks.items():
                                rows.append(
                                    {
                                        "sample_id": f"selection_{sample_index}",
                                        "method": method,
                                        "block": block,
                                        "target_step": target_step,
                                        "branch": branch,
                                        "horizon": horizon,
                                        "error_sq": risk * risk,
                                        "target_sq": 1.0,
                                        "output_relative_l2": risk,
                                        "output_target_sq": 1.0,
                                        "total_runtime_macs": 0,
                                        "observable_macs": 0,
                                    }
                                )
        _, method_rows, decision, best_method = build_gate_tables(
            rows, config, "selection"
        )
        by_method = {str(row["method"]): row for row in method_rows}
        self.assertTrue(by_method["good"]["g024_pass"])
        self.assertFalse(by_method["unstable"]["g024_pass"])
        self.assertFalse(by_method["unstable"]["open_loop_pass"])
        self.assertEqual(decision, "PASS")
        self.assertEqual(best_method, "good")


class FakeSelfAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int],
        rope_cache: object,
    ) -> torch.Tensor:
        del grid_size, rope_cache
        return 0.1 * x


class FakeWanBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.modulation = nn.Parameter(torch.zeros(1, 6, width))
        self.norm1 = nn.Identity()
        self.self_attn = FakeSelfAttention(width)

    def forward(
        self,
        x: torch.Tensor,
        *,
        e: torch.Tensor,
        grid_sizes: tuple[int, int, int],
        rope_cache: object,
    ) -> torch.Tensor:
        modulation = (self.modulation + e).chunk(6, dim=1)
        attention_input = self.norm1(x) * (1 + modulation[1]) + modulation[0]
        return x + self.self_attn(attention_input, grid_sizes, rope_cache) * modulation[2]


class FakeWanModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeWanBlock(width)])

    def forward(
        self,
        x: torch.Tensor,
        *,
        e: torch.Tensor,
        grid_sizes: tuple[int, int, int],
        rope_cache: object,
    ) -> torch.Tensor:
        return self.blocks[0](
            x, e=e, grid_sizes=grid_sizes, rope_cache=rope_cache
        )


class ObservabilityRuntimeTests(unittest.TestCase):
    def test_recorder_preserves_dense_output_and_captures_full_fields(self) -> None:
        model = FakeWanModel(width=4)
        original_model = model.forward
        original_block = model.blocks[0].forward
        recorder = WanCurrentInputObservabilityRecorder(
            model,
            sampling_steps=2,
            capture_steps=(0, 1),
            blocks=(0,),
            branches=(0, 1),
            qk_rows=2,
        )
        recorder.begin_run("unit", (2.0, 1.0), {"split": "synthetic"})
        try:
            for step in range(2):
                for branch in range(2):
                    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
                    x = x + step + branch
                    e = torch.ones(1, 6, 4)
                    expected = original_block(
                        x, e=e, grid_sizes=(1, 2, 2), rope_cache=None
                    )
                    observed = model(
                        x, e=e, grid_sizes=(1, 2, 2), rope_cache=None
                    )
                    torch.testing.assert_close(observed, expected)
            summary = recorder.end_run()
            trajectory = recorder.cell_trajectory(
                0, 0, device=torch.device("cpu")
            )
        finally:
            recorder.restore()
        self.assertEqual(summary["record_count"], 4)
        self.assertEqual(trajectory.block_input.shape, (2, 4, 4))
        self.assertEqual(trajectory.adaln.shape, (2, 6, 4))
        self.assertEqual(trajectory.qk_sketch.shape, (2, 4, 4))
        self.assertEqual(model.forward, original_model)
        self.assertEqual(model.blocks[0].forward, original_block)


if __name__ == "__main__":
    unittest.main()
