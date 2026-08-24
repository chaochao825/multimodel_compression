import importlib.util
import sys
import unittest
from pathlib import Path

PROBES = Path(__file__).resolve().parents[1] / "probes"
sys.path.insert(0, str(PROBES))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for the GPU diagnostic")
class NativeConfidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global torch, first_token_diagnostics
        import torch

        from mvbench_llava_native_confidence import first_token_diagnostics

    def test_first_token_margin_and_entropy_are_finite(self) -> None:
        result = first_token_diagnostics(torch.tensor([0.0, 2.0, 1.0]))
        self.assertEqual(result["native_first_token_id"], 1)
        self.assertAlmostEqual(result["native_first_token_margin"], 1.0)
        self.assertGreater(result["native_first_token_entropy"], 0.0)

    def test_rejects_nonfinite_logits(self) -> None:
        with self.assertRaises(ValueError):
            first_token_diagnostics(torch.tensor([0.0, float("nan")]))


if __name__ == "__main__":
    unittest.main()
