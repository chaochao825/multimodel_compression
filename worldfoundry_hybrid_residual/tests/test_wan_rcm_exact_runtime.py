from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_wan_rcm_exact_runtime as exact_runtime  # noqa: E402


class FakeTensor:
    def __init__(self, value: str) -> None:
        self.value = value

    def to(self, **_kwargs: object) -> "FakeTensor":
        return self


class FakeTorch:
    bfloat16 = "bfloat16"

    class cuda:
        @staticmethod
        def synchronize(_device: object) -> None:
            return None


class TextEncoderPolicyTest(unittest.TestCase):
    def make_runtime(self) -> tuple[dict[str, object], list[str], list[str]]:
        calls: list[str] = []
        clears: list[str] = []

        def get_umt5(*, checkpoint_path: str, prompts: str) -> FakeTensor:
            self.assertEqual(checkpoint_path, "text.pt")
            calls.append(prompts)
            return FakeTensor(prompts)

        runtime: dict[str, object] = {
            "get_umt5": get_umt5,
            "clear_umt5": lambda: clears.append("clear"),
            "repeat": lambda tensor, _pattern, **_kwargs: tensor,
            "torch": FakeTorch(),
        }
        return runtime, calls, clears

    def test_resident_policy_never_caches_positive_prompts(self) -> None:
        runtime, calls, _clears = self.make_runtime()
        policy = exact_runtime.TextEncoderPolicy(
            runtime, "text.pt", "cuda:0", resident=True, cache_negative=False
        )
        policy.encode_condition("same prompt", need_negative=False)
        policy.encode_condition("same prompt", need_negative=False)
        self.assertEqual(calls, ["same prompt", "same prompt"])
        self.assertEqual(policy.stats()["positive_cache_hits"], 0)

    def test_resident_policy_reuses_only_fixed_negative(self) -> None:
        runtime, calls, _clears = self.make_runtime()
        policy = exact_runtime.TextEncoderPolicy(
            runtime, "text.pt", "cuda:0", resident=True, cache_negative=True
        )
        policy.encode_condition("prompt one", need_negative=True)
        policy.encode_condition("prompt two", need_negative=True)
        self.assertEqual(
            calls,
            ["prompt one", exact_runtime.baseline.NEGATIVE_PROMPT, "prompt two"],
        )
        self.assertEqual(policy.stats()["negative_model_calls"], 1)
        self.assertEqual(policy.stats()["negative_cache_hits"], 1)

    def test_reload_policy_clears_after_every_request(self) -> None:
        runtime, calls, clears = self.make_runtime()
        policy = exact_runtime.TextEncoderPolicy(
            runtime, "text.pt", "cuda:0", resident=False, cache_negative=False
        )
        policy.encode_condition("prompt one", need_negative=False)
        policy.encode_condition("prompt two", need_negative=False)
        self.assertEqual(calls, ["prompt one", "prompt two"])
        self.assertEqual(clears, ["clear", "clear"])


class ExactRuntimeConfigTest(unittest.TestCase):
    def test_positive_prompt_cache_is_rejected(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "wan_rcm_exact_runtime_exp052_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["positive_prompt_cache"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbids positive-prompt"):
                exact_runtime.load_configs(path)


if __name__ == "__main__":
    unittest.main()
