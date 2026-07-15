import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from src.analysis.fit_matrix import evaluate_fit, load_weight_file
from src.utils.metrics import energy_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-file", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    weight_file = Path(args.weight_file)
    target, metadata = load_weight_file(weight_file)
    target = target.to(args.device)

    approx, metrics, _ = evaluate_fit(
        method="bcm_only",
        target=target,
        block_size=args.block_size,
        rank=0,
        num_bcm_basis=1,
        sparse_ratio=0.0,
        delta_mode="horizontal",
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    residual = target - approx.to(target.device)
    singular_values = torch.linalg.svdvals(residual).cpu()

    rows: List[Dict] = []
    for rank in args.ranks:
        rows.append(
            {
                "rank": rank,
                "energy_ratio": energy_ratio(singular_values, rank),
                "block_size": args.block_size,
                "weight_file": str(weight_file),
                "layer_key": metadata.get("key"),
                "bcm_relative_fro_error": metrics["relative_fro_error"],
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"residual_spectrum_{weight_file.stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
