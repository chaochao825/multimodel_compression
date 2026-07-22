from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "probes"))

import run_stc_rekv_official as runner  # noqa: E402


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _write_model(model: Path) -> None:
    model.mkdir()
    config = {
        "architectures": [runner.EXPECTED_ARCHITECTURE],
        "model_type": runner.EXPECTED_MODEL_TYPE,
        "vision_config": {"model_type": runner.EXPECTED_VISION_MODEL_TYPE},
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for path in runner.REQUIRED_MODEL_PATHS:
        target = model / path
        if target.exists():
            continue
        target.write_text("{}", encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"fixture shard")
    index = {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}
    (model / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def _official_payload(mode: str, repeats: int = 4) -> dict:
    vit = [10.0, 11.0, 12.0, 13.0]
    llm = [20.0, 22.0, 24.0, 26.0]

    def stage(values: list[float]) -> dict:
        return {
            "min": min(values),
            "median": 11.5 if values is vit else 23.0,
            "mean": sum(values) / len(values),
            "std": 1.118033988749895 if values is vit else 2.23606797749979,
            "max": max(values),
            "samples": values,
        }

    return {
        "label": mode,
        "model": "llava_ov_7b",
        "num_frames": 64,
        "repeats": repeats,
        "config": runner.MODE_CONFIGS[mode].copy(),
        "vit_encode_ms": stage(vit),
        "llm_prefill_ms": stage(llm),
        "peak_mem_gb": 18.25,
    }


class STCReKVOfficialTest(unittest.TestCase):
    def test_executable_path_preserves_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "python"
            try:
                link.symlink_to(Path(sys.executable))
            except OSError as error:
                self.skipTest(f"symlinks are unavailable: {error}")
            observed = runner.absolute_executable_path(link)
            self.assertEqual(observed, Path(os.path.abspath(link)))
            self.assertNotEqual(observed, link.resolve())

    def test_environment_preserves_isolated_triton_cache(self) -> None:
        cache = "/workspace/.cache/triton-stc"
        with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": cache}):
            environment = runner.build_environment(
                source_root=Path("/source"),
                model_path=Path("/model"),
                mode="rekv",
                gpu_index=2,
            )
        self.assertEqual(environment["TRITON_CACHE_DIR"], cache)

    def test_mode_environment_matches_upstream_run_script(self) -> None:
        self.assertEqual(
            runner.mode_environment("rekv"),
            {
                "STC_PATCH_VISION": "0",
                "STC_TOKEN_PER_FRAME": "196",
                "STC_UPDATE_TOKEN_RATIO": "1.0",
                "STC_CACHE_INTERVAL": "4",
            },
        )
        self.assertEqual(runner.mode_environment("rekv_stc")["STC_PATCH_VISION"], "1")
        self.assertEqual(runner.mode_environment("rekv_stc")["STC_TOKEN_PER_FRAME"], "64")

    def test_checkout_rejects_non_cache_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "STC"
            checkout.mkdir()
            _git("init", cwd=checkout)
            _git("config", "user.email", "benchmark@example.invalid", cwd=checkout)
            _git("config", "user.name", "Benchmark Test", cwd=checkout)
            for relative in runner.REQUIRED_SOURCE_PATHS:
                path = checkout / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("VALUE = 1\n", encoding="utf-8")
            _git("add", ".", cwd=checkout)
            _git("commit", "-m", "fixture", cwd=checkout)
            commit = _git("rev-parse", "HEAD", cwd=checkout)
            cache = checkout / "stc/__pycache__/config.cpython-310.pyc"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(b"cache")
            self.assertTrue(runner.validate_checkout(checkout, commit)["code_clean"])

            (checkout / "stc/config.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-cache changes"):
                runner.validate_checkout(checkout, commit)

    def test_model_requires_exact_indexed_shard_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            _write_model(model)
            validated = runner.validate_model(model)
            self.assertEqual(len(validated["shards"]), 1)

            (model / "unexpected.safetensors").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "does not match index"):
                runner.validate_model(model)

    def test_result_validation_recomputes_observed_quantiles(self) -> None:
        payload = _official_payload("rekv_stc")
        derived = runner.validate_official_result(
            payload, mode="rekv_stc", num_frames=64, repeats=4
        )
        self.assertEqual(derived["vit_encode_ms"]["p95"], 13.0)
        self.assertEqual(derived["llm_prefill_ms"]["p50"], 24.0)
        self.assertEqual(derived["instrumented_stage_sum_ms"]["p99"], 39.0)

    def test_result_validation_rejects_mode_mismatch(self) -> None:
        payload = _official_payload("rekv_stc")
        payload["config"]["token_per_frame"] = 196
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            runner.validate_official_result(
                payload, mode="rekv_stc", num_frames=64, repeats=4
            )

    def test_fingerprint_is_order_independent(self) -> None:
        self.assertEqual(
            runner.stable_fingerprint({"a": 1, "b": 2}),
            runner.stable_fingerprint({"b": 2, "a": 1}),
        )


if __name__ == "__main__":
    unittest.main()
