import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from src.analysis.activation_error import estimate_activation_error
from src.analysis.param_flop_counter import estimate_method_cost
from src.methods.multi_bcm import MultiBCMLinear
from src.methods.sparse_delta import SparseDeltaLinear
from src.modules.block_circulant_linear import BlockCirculantLinear
from src.modules.low_rank_linear import LowRankResidual
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import relative_fro_error, spectral_error
from src.utils.seed import set_seed


class BCMPlusLowRankApprox(nn.Module):
    def __init__(self, in_features: int, out_features: int, block_size: int, rank: int) -> None:
        super().__init__()
        self.bcm = BlockCirculantLinear(in_features, out_features, block_size, bias=False)
        self.lowrank = LowRankResidual(in_features, out_features, rank=rank, alpha=max(rank, 1))

    def dense_weight(self) -> torch.Tensor:
        return self.bcm.dense_weight() + self.lowrank.dense_weight().to(self.bcm.device, self.bcm.dtype)

    def regularization_loss(self) -> torch.Tensor:
        return self.bcm.regularization_loss()


class BCMPlusSparseApprox(nn.Module):
    def __init__(self, in_features: int, out_features: int, block_size: int, sparse_ratio: float) -> None:
        super().__init__()
        self.bcm = BlockCirculantLinear(in_features, out_features, block_size, bias=False)
        self.sparse = SparseDeltaLinear(in_features, out_features, sparse_ratio=sparse_ratio)

    def dense_weight(self) -> torch.Tensor:
        return self.bcm.dense_weight() + self.sparse.dense_weight().to(self.bcm.device, self.bcm.dtype)

    def regularization_loss(self) -> torch.Tensor:
        return self.bcm.regularization_loss()


def load_weight_file(path: Path) -> Tuple[torch.Tensor, Dict]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "weight" in payload:
        metadata = {k: v for k, v in payload.items() if k != "weight"}
        return payload["weight"].float(), metadata
    if torch.is_tensor(payload):
        return payload.float(), {"key": path.stem}
    raise ValueError(f"unsupported payload format: {path}")


def iter_weight_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.glob("*.pt"))


def truncated_svd(target: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros_like(target)
    q = min(rank, min(target.shape))
    u, s, v = torch.svd_lowrank(target, q=q, niter=2)
    return (u[:, :q] * s[:q]) @ v[:, :q].t()


def optimize_module(
    module: nn.Module,
    target: torch.Tensor,
    steps: int = 300,
    learning_rate: float = 3e-2,
) -> Tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
    module = module.to(target.device)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = {}
    best_weight = None

    for _ in range(steps):
        optimizer.zero_grad()
        approx = module.dense_weight()
        loss = torch.mean((approx - target) ** 2)
        if hasattr(module, "regularization_loss"):
            loss = loss + module.regularization_loss()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        if loss_value < best_loss:
            best_loss = loss_value
            best_weight = approx.detach().cpu()
            best_state = {key: value.detach().cpu() for key, value in module.state_dict().items()}

    return best_weight, best_loss, best_state


def build_module(
    method: str,
    target: torch.Tensor,
    block_size: int,
    rank: int,
    num_bcm_basis: int,
    sparse_ratio: float,
    delta_mode: str,
) -> nn.Module:
    out_features, in_features = target.shape
    if method == "bcm_only":
        return BlockCirculantLinear(in_features, out_features, block_size, bias=False)
    if method == "bcm_plus_lowrank":
        return BCMPlusLowRankApprox(in_features, out_features, block_size, rank)
    if method == "multi_bcm":
        return MultiBCMLinear(in_features, out_features, block_size, num_bases=num_bcm_basis)
    if method == "bcm_plus_sparse_delta":
        return BCMPlusSparseApprox(in_features, out_features, block_size, sparse_ratio)
    if method == "generator_delta_bcm":
        return BlockCirculantLinear(
            in_features,
            out_features,
            block_size,
            bias=False,
            use_generator_delta=True,
            delta_mode=delta_mode,
            l1_lambda=1e-4,
        )
    raise ValueError(f"unsupported method: {method}")


def evaluate_fit(
    method: str,
    target: torch.Tensor,
    block_size: int,
    rank: int,
    num_bcm_basis: int,
    sparse_ratio: float,
    delta_mode: str,
    steps: int,
    learning_rate: float,
) -> Tuple[torch.Tensor, Dict, Dict[str, torch.Tensor]]:
    if method == "lowrank_svd":
        approx = truncated_svd(target, rank).to(target.device)
        costs = estimate_method_cost(method, target.shape[1], target.shape[0], block_size=block_size, rank=rank)
        metrics = asdict(costs)
        metrics.update(
            {
                "method": method,
                "block_size": block_size,
                "rank": rank,
                "num_bcm_basis": num_bcm_basis,
                "sparse_ratio": sparse_ratio,
                "delta_mode": delta_mode,
                "objective": 0.0,
                "relative_fro_error": relative_fro_error(target, approx),
                "spectral_error": spectral_error(target, approx),
                "activation_error": estimate_activation_error(target, approx),
            }
        )
        return approx, metrics, {}

    module = build_module(method, target, block_size, rank, num_bcm_basis, sparse_ratio, delta_mode)
    approx, objective, state_dict = optimize_module(module, target, steps=steps, learning_rate=learning_rate)
    approx_device = approx.to(target.device)
    costs = estimate_method_cost(
        method,
        target.shape[1],
        target.shape[0],
        block_size=block_size,
        rank=rank,
        num_bcm_basis=num_bcm_basis,
        sparse_ratio=sparse_ratio,
        use_generator_delta=method == "generator_delta_bcm",
    )
    metrics = asdict(costs)
    metrics.update(
        {
            "method": method,
            "block_size": block_size,
            "rank": rank,
            "num_bcm_basis": num_bcm_basis,
            "sparse_ratio": sparse_ratio,
            "delta_mode": delta_mode,
            "objective": objective,
            "relative_fro_error": relative_fro_error(target, approx_device),
            "spectral_error": spectral_error(target, approx_device),
            "activation_error": estimate_activation_error(target, approx_device),
        }
    )
    return approx_device, metrics, state_dict


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lowrank_svd", "bcm_only", "bcm_plus_lowrank", "generator_delta_bcm"],
    )
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--ranks", nargs="+", type=int, default=[0, 4, 8, 16])
    parser.add_argument("--num-bcm-basis", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--sparse-ratios", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--delta-mode", default="horizontal")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    input_dir = Path(args.input_dir)
    output_root = Path("results/matrix_fit") / input_dir.name
    summary_rows: List[Dict] = []

    for weight_path in iter_weight_files(input_dir):
        target, metadata = load_weight_file(weight_path)
        target = target.to(args.device)
        layer_name = metadata.get("key", weight_path.stem).replace(".", "__")
        layer_output_dir = output_root / layer_name
        layer_rows: List[Dict] = []

        for method in args.methods:
            for block_size in args.block_sizes:
                if target.shape[0] % block_size != 0 or target.shape[1] % block_size != 0:
                    continue
                rank_candidates = args.ranks if method in {"lowrank_svd", "bcm_plus_lowrank"} else [0]
                basis_candidates = args.num_bcm_basis if method == "multi_bcm" else [1]
                sparse_candidates = args.sparse_ratios if method == "bcm_plus_sparse_delta" else [0.0]

                for rank in rank_candidates:
                    for num_bases in basis_candidates:
                        for sparse_ratio in sparse_candidates:
                            approx, metrics, state_dict = evaluate_fit(
                                method=method,
                                target=target,
                                block_size=block_size,
                                rank=rank,
                                num_bcm_basis=num_bases,
                                sparse_ratio=sparse_ratio,
                                delta_mode=args.delta_mode,
                                steps=args.steps,
                                learning_rate=args.learning_rate,
                            )
                            row = {
                                "weight_file": str(weight_path),
                                "layer_key": metadata.get("key"),
                                **metrics,
                            }
                            layer_rows.append(row)
                            summary_rows.append(row)

                            checkpoint_name = f"{method}__bs{block_size}__r{rank}__basis{num_bases}.pt"
                            save_checkpoint(
                                str(layer_output_dir / checkpoint_name),
                                state_dict=state_dict,
                                metrics=row,
                                approx_weight=approx.cpu(),
                                target_weight=target.cpu(),
                            )

        write_csv(layer_output_dir / "metrics.csv", layer_rows)

    write_csv(output_root / "summary_metrics.csv", summary_rows)
    with (output_root / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)


if __name__ == "__main__":
    main()
