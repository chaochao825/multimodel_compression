import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from capture_longlive_causal_qkv import (
    CaptureController,
    install_flash_attention_3_output_compatibility,
)


class CaptureControllerTests(unittest.TestCase):
    def test_fa3_tuple_compatibility_preserves_output_tensor(self):
        output = torch.randn(2, 3)
        auxiliary = torch.randn(2)
        module = SimpleNamespace(
            flash_attn_varlen_func=lambda *args, **kwargs: (output, auxiliary)
        )
        self.assertTrue(install_flash_attention_3_output_compatibility(module))
        self.assertIs(module.flash_attn_varlen_func(), output)
        # Installation is idempotent and must not wrap an already wrapped call.
        self.assertTrue(install_flash_attention_3_output_compatibility(module))
        self.assertIs(module.flash_attn_varlen_func(), output)

    def setUp(self):
        self.protocol = {
            "protocol_id": "capture-unit",
            "capture": {
                "layer_indices": [2],
                "current_start_frames": [3],
                "denoising_call_indices": [1, 3],
                "frame_seq_len": 4,
                "sink_frames": 1,
                "query_tile_size": 2,
                "query_tiles_per_record": 1,
                "heads": [0, 2],
            },
        }
        self.prompt = {
            "id": "calib",
            "split": "calibration",
            "seed": 7,
            "text": "unit prompt",
        }

    @staticmethod
    def module_arguments(current_start):
        return {
            "current_start": current_start,
            "grid_sizes": torch.tensor([[2, 1, 4]]),
            "freqs": torch.ones(16, 1, dtype=torch.complex64),
        }

    def test_selected_call_saves_expected_query_tiles_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = CaptureController(self.protocol, Path(directory))
            controller.set_denoising_timesteps([901, 701, 401, 101])
            controller.set_prompt(self.prompt)
            query = torch.arange(1 * 8 * 3 * 2).reshape(1, 8, 3, 2).float()
            key = torch.randn(1, 16, 3, 2)
            value = torch.randn(1, 16, 3, 2)
            output = query + 100

            with controller.module_call(2, self.module_arguments(12)):
                controller.maybe_save(query, key, value, output)

            self.assertEqual(len(controller.saved_paths), 1)
            payload = torch.load(controller.saved_paths[0], weights_only=False)
            self.assertEqual(payload["query"].shape, (4, 2, 2))
            self.assertEqual(payload["key"].shape, (16, 2, 2))
            self.assertEqual(payload["metadata"]["current_start_frame"], 3)
            self.assertEqual(payload["metadata"]["denoising_call_index"], 1)
            self.assertEqual(payload["metadata"]["denoising_timestep"], 901)
            self.assertEqual(payload["metadata"]["head_indices"], [0, 2])
            self.assertEqual(payload["metadata"]["query_frame_ids"], [3, 4])
            self.assertEqual(payload["metadata"]["key_frame_ids"], [0, 2, 3, 4])
            self.assertEqual(payload["rope_freqs"].dtype, torch.complex64)
            expected_indices = torch.tensor([1, 2, 5, 6])
            expected = query[0].index_select(0, expected_indices)[:, [0, 2]]
            torch.testing.assert_close(payload["query"].float(), expected)
            torch.testing.assert_close(payload["dense_output"].float(), expected + 100)

    def test_unselected_layer_start_and_occurrence_do_not_save(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = CaptureController(self.protocol, Path(directory))
            controller.set_prompt(self.prompt)
            query = torch.zeros(1, 4, 3, 2)
            key = torch.zeros(1, 16, 3, 2)
            value = torch.zeros_like(key)
            for layer, current_start in ((1, 12), (2, 8), (2, 12), (2, 12)):
                with controller.module_call(layer, self.module_arguments(current_start)):
                    controller.maybe_save(query, key, value, query)
            # The first selected (layer=2, start=3) call is occurrence one.
            self.assertEqual(len(controller.saved_paths), 1)

    def test_non_frame_aligned_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = CaptureController(self.protocol, Path(directory))
            with self.assertRaisesRegex(ValueError, "not aligned"):
                with controller.module_call(2, self.module_arguments(13)):
                    pass


if __name__ == "__main__":
    unittest.main()
