from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rcm_attention_atlas_core as core  # noqa: E402

if importlib.util.find_spec("torch") is not None:
    import torch
else:
    torch = None


def metric(
    identity: str,
    split: str,
    step: int,
    layer: int,
    value: float,
) -> core.CellMetric:
    return core.CellMetric(
        identity=identity,
        split=split,
        step=step,
        layer=layer,
        aggregate=value,
        worst_head=value,
        worst_query_tile=value,
    )


def grid(identity: str, split: str, value: float) -> list[core.CellMetric]:
    return [
        metric(identity, split, step, layer, value)
        for step in range(core.CELL_STEPS)
        for layer in range(core.CELL_LAYERS)
    ]


@unittest.skipIf(torch is None, "PyTorch numerical metric test runs in the rCM environment")
class ErrorMetricTest(unittest.TestCase):
    def test_exact_and_scaled_outputs(self) -> None:
        reference = torch.ones(1, 5, 2, 3)
        exact = core.output_error_metrics(reference, reference, query_tile_size=4)
        self.assertEqual(tuple(float(value) for value in exact), (0.0, 0.0, 0.0))

        candidate = reference * 1.1
        scaled = core.output_error_metrics(reference, candidate, query_tile_size=4)
        for value in scaled:
            self.assertAlmostEqual(float(value), 0.1, places=6)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            core.output_error_metrics(
                torch.ones(1, 4, 2, 3), torch.ones(1, 5, 2, 3)
            )


class AtlasTest(unittest.TestCase):
    def test_calibration_only_selection_and_clean_transfer(self) -> None:
        result = core.freeze_and_evaluate_atlas(
            grid("cal", "calibration", 0.005),
            grid("eval", "evaluation", 0.009),
            ["cal"],
            ["eval"],
            core.ErrorThresholds(0.008, 0.016, 0.016),
            core.ErrorThresholds(0.01, 0.02, 0.02),
            minimum_selected_cells=87,
        )
        self.assertEqual(result["selected_cell_count"], core.CELL_COUNT)
        self.assertEqual(result["false_safe_count"], 0)
        self.assertTrue(result["passes_transfer_and_count"])

    def test_evaluation_failure_is_not_used_to_refit_the_atlas(self) -> None:
        evaluation = grid("eval", "evaluation", 0.009)
        evaluation[0] = metric("eval", "evaluation", 0, 0, 0.03)
        result = core.freeze_and_evaluate_atlas(
            grid("cal", "calibration", 0.005),
            evaluation,
            ["cal"],
            ["eval"],
            core.ErrorThresholds(0.008, 0.016, 0.016),
            core.ErrorThresholds(0.01, 0.02, 0.02),
            minimum_selected_cells=87,
        )
        self.assertEqual(result["selected_cell_count"], core.CELL_COUNT)
        self.assertEqual(result["false_safe_count"], 1)
        self.assertFalse(result["passes_transfer_and_count"])

    def test_duplicate_grid_cell_is_rejected(self) -> None:
        records = grid("cal", "calibration", 0.005)
        records.append(records[0])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            core.validate_record_grid(records, ["cal"], "calibration")


class ProjectionTest(unittest.TestCase):
    def test_exp052_full_coverage_ceiling(self) -> None:
        seconds, speedup = core.projected_request(
            9.637995031895116,
            3.20536530460231,
            0.5388086760322437,
            1.0,
            1.5906,
        )
        self.assertAlmostEqual(seconds, 8.996719637449413)
        self.assertAlmostEqual(speedup, 1.0712788016396948)

    def test_zero_coverage_preserves_the_baseline(self) -> None:
        seconds, speedup = core.projected_request(
            9.637995031895116,
            3.20536530460231,
            0.5388086760322437,
            0.0,
            1.586376845918112,
        )
        self.assertEqual(seconds, 9.637995031895116)
        self.assertEqual(speedup, 1.0)


class PatchTest(unittest.TestCase):
    def make_network(self) -> SimpleNamespace:
        blocks = []
        for layer in range(core.CELL_LAYERS):
            def attention(q: int, _k: int, _v: int, *, bias: int = layer) -> int:
                return q + bias

            local = SimpleNamespace(attn=attention)
            blocks.append(
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn_op=SimpleNamespace(local_attn=local)
                    )
                )
            )
        return SimpleNamespace(blocks=blocks)

    def test_patch_dispatches_layers_and_restores_originals(self) -> None:
        network = self.make_network()
        originals = [block.self_attn.attn_op.local_attn.attn for block in network.blocks]
        seen: list[int] = []

        def dispatcher(layer: int, reference: object, q: int, k: int, v: int) -> int:
            seen.append(layer)
            return reference(q, k, v)

        with core.WanSelfAttentionPatch(network, dispatcher):
            output = network.blocks[7].self_attn.attn_op.local_attn.attn(
                1, 2, 3
            )
            self.assertEqual(int(output), 8)
            self.assertEqual(seen, [7])
        for block, original in zip(network.blocks, originals):
            self.assertIs(block.self_attn.attn_op.local_attn.attn, original)


if __name__ == "__main__":
    unittest.main()
