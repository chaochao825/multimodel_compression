from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_wan_rcm_baseline as baseline  # noqa: E402


class WanRcmBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "wan_rcm_baseline_exp047_v1.json"
        )
        cls.config = baseline.load_config(config_path)

    def test_frozen_method_order_and_prompt_count(self) -> None:
        self.assertEqual(tuple(self.config["methods"]), baseline.METHODS)
        self.assertEqual(len(baseline.load_formal_prompts(self.config)), 4)

    def test_smoke_identity_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen engineering identity"):
            baseline.resolve_run_spec(
                self.config, "rcm4", "f17-smoke", 0, None, Path("out")
            )

    def test_formal_identity_must_be_registered(self) -> None:
        with self.assertRaisesRegex(ValueError, "formal seed"):
            baseline.resolve_run_spec(
                self.config, "teacher20", "formal", 0, 123, Path("out")
            )
        spec = baseline.resolve_run_spec(
            self.config,
            "teacher20",
            "formal",
            3,
            self.config["generation"]["formal_seeds"][1],
            Path("out"),
        )
        self.assertEqual(spec.prompt_index, 3)
        self.assertEqual(spec.num_frames, 81)

    def test_state_dict_prefix_normalization(self) -> None:
        normalized = baseline.normalize_state_dict_keys(
            {"net.blocks.0.weight": 1, "head.weight": 2}
        )
        self.assertEqual(
            normalized, {"blocks.0.weight": 1, "head.weight": 2}
        )

    def test_only_registered_rcm_training_metadata_is_removed(self) -> None:
        state = {key: index for index, key in enumerate(baseline.RCM_TRAINING_METADATA_KEYS)}
        state["blocks.0.weight"] = 9
        cleaned = baseline.remove_rcm_training_metadata(state)
        self.assertEqual(cleaned, {"blocks.0.weight": 9})

    def test_partial_rcm_training_metadata_is_accepted(self) -> None:
        state = {
            baseline.RCM_TRAINING_METADATA_KEYS[1]: 1,
            "blocks.0.weight": 9,
        }
        cleaned = baseline.remove_rcm_training_metadata(state)
        self.assertEqual(cleaned, {"blocks.0.weight": 9})

    def test_prefixed_rcm_training_metadata_is_removed_after_normalization(self) -> None:
        state = {
            f"net.{baseline.RCM_TRAINING_METADATA_KEYS[0]}": 1,
            "net.blocks.0.weight": 9,
        }
        normalized = baseline.normalize_state_dict_keys(state)
        cleaned = baseline.remove_rcm_training_metadata(normalized)
        self.assertEqual(cleaned, {"blocks.0.weight": 9})

    def test_timing_summary_uses_median(self) -> None:
        fields = (
            "text_seconds",
            "denoiser_seconds",
            "vae_seconds",
            "serialization_seconds",
            "warm_e2e_seconds",
            "denoiser_seconds_per_step",
            "denoiser_seconds_per_forward",
            "peak_allocated_mib",
            "peak_reserved_mib",
        )
        rows = [{field: value for field in fields} for value in (1.0, 9.0, 3.0)]
        summary = baseline.timing_summary(rows)
        self.assertTrue(all(value == 3.0 for value in summary.values()))

    def test_checkpoint_identity_must_match(self) -> None:
        config = json.loads(json.dumps(self.config))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"registered checkpoint")
            config["remote"]["checkpoint"] = str(checkpoint)
            config["source"]["checkpoint_sha256"] = baseline.file_sha256(checkpoint)
            path, identity = baseline.verify_checkpoint(config, "rcm4")
            self.assertEqual(path, checkpoint.resolve())
            self.assertEqual(identity, config["source"]["checkpoint_sha256"])
            config["source"]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                baseline.verify_checkpoint(config, "rcm4")

    def test_offline_model_cache_is_explicit(self) -> None:
        config = json.loads(json.dumps(self.config))
        with tempfile.TemporaryDirectory() as directory:
            config["remote"]["hf_home"] = directory
            with patch.dict(baseline.os.environ):
                cache = baseline.configure_offline_model_cache(config)
                self.assertEqual(cache, Path(directory).resolve())
                self.assertEqual(baseline.os.environ["HF_HOME"], str(cache))
                self.assertEqual(baseline.os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(baseline.os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_flash_attention_tuple_output_is_normalized(self) -> None:
        module = types.SimpleNamespace(
            FLASH_ATTN_3_AVAILABLE=True,
            flash_attn_func=lambda *_args, **_kwargs: ("output", "softmax_lse"),
        )
        installed = baseline.install_flash_attention_output_compat(module)
        self.assertTrue(installed)
        self.assertEqual(module.flash_attn_func("q", "k", "v"), "output")

    def test_flash_attention_compat_is_inactive_without_fa3(self) -> None:
        def original(*_args, **_kwargs):
            return "output"

        module = types.SimpleNamespace(
            FLASH_ATTN_3_AVAILABLE=False,
            flash_attn_func=original,
        )
        installed = baseline.install_flash_attention_output_compat(module)
        self.assertFalse(installed)
        self.assertIs(module.flash_attn_func, original)

    def test_config_identity_rejects_wrong_gate(self) -> None:
        bad = json.loads(json.dumps(self.config))
        bad["gate_id"] = "G-wrong"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen EXP-047"):
                baseline.load_config(path)


if __name__ == "__main__":
    unittest.main()
