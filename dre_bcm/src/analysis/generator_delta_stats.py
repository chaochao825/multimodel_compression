import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import empirical_entropy, quantize_symmetric, summarize_tensor


def collect_generators(state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
    base = []
    delta = []
    for key, value in state_dict.items():
        if "base_generator" in key:
            base.append(value.float())
        elif "delta_generator" in key:
            delta.append(value.float())
    return {"base": base, "delta": delta}


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="results/peft")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    state_dict = checkpoint.get("state_dict", checkpoint)
    tensors = collect_generators(state_dict)
    base = torch.cat([tensor.reshape(-1) for tensor in tensors["base"]], dim=0) if tensors["base"] else torch.empty(0)
    delta = torch.cat([tensor.reshape(-1) for tensor in tensors["delta"]], dim=0) if tensors["delta"] else torch.empty(0)

    thresholds = [1e-3, 1e-2, 5e-2]
    stats = {
        "generator_entropy": empirical_entropy(base),
        "delta_entropy": empirical_entropy(delta),
        "generator_summary": summarize_tensor(base) if base.numel() else {},
        "delta_summary": summarize_tensor(delta) if delta.numel() else {},
        "threshold_sparsity": {
            str(threshold): float((delta.abs() < threshold).float().mean().item()) if delta.numel() else 0.0
            for threshold in thresholds
        },
        "quantization_mse": {
            f"int{bits}": mse(delta, quantize_symmetric(delta, bits)) if delta.numel() else 0.0
            for bits in [8, 4, 2]
        },
        "estimated_compressed_bits": {
            "fp16_base_int4_delta": int(base.numel() * 16 + delta.numel() * 4),
            "entropy_delta": float(base.numel() * 16 + delta.numel() * empirical_entropy(delta)),
        },
    }

    output_dir = Path(args.output_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generator_delta_stats.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
