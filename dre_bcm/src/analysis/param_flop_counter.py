import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass
class CostReport:
    method: str
    trainable_params: int
    stored_params: int
    estimated_bits_fp16: int
    estimated_bits_int8: int
    estimated_bits_int4_delta: int
    bcm_direct_ops: int
    bcm_fft_ops: int
    lowrank_ops: int
    total_ops: int
    activation_memory: int
    extra_latency_proxy: float


def bcm_param_count(in_features: int, out_features: int, block_size: int) -> int:
    return (in_features // block_size) * (out_features // block_size) * block_size


def lowrank_param_count(in_features: int, out_features: int, rank: int) -> int:
    return rank * (in_features + out_features)


def fourier_param_count(freq_rows: int, freq_cols: int) -> int:
    return 2 * freq_rows * freq_cols


def sparse_param_count(in_features: int, out_features: int, sparse_ratio: float) -> int:
    return max(1, int(in_features * out_features * sparse_ratio))


def estimate_method_cost(
    method: str,
    in_features: int,
    out_features: int,
    block_size: int = 32,
    rank: int = 0,
    num_bcm_basis: int = 1,
    sparse_ratio: float = 0.05,
    use_generator_delta: bool = False,
    freq_rows: int = 16,
    freq_cols: int = 16,
) -> CostReport:
    bcm_params = bcm_param_count(in_features, out_features, block_size)
    lowrank_params = lowrank_param_count(in_features, out_features, rank)
    direct_ops = (out_features // block_size) * (in_features // block_size) * (block_size ** 2)
    fft_ops_per_block = int(8 * block_size * max(1, math.log2(block_size)))
    fft_ops = (out_features // block_size) * (in_features // block_size) * fft_ops_per_block
    lowrank_ops = rank * (in_features + out_features)
    activation_memory = in_features + out_features

    if method == "bcm_only":
        trainable_params = bcm_params
        stored_params = bcm_params * (2 if use_generator_delta else 1)
        total_ops = direct_ops
        latency_proxy = direct_ops / max(1, out_features * in_features)
    elif method == "generator_delta_bcm":
        trainable_params = bcm_params * 2
        stored_params = bcm_params * 2
        total_ops = direct_ops
        latency_proxy = direct_ops / max(1, out_features * in_features)
    elif method == "bcm_plus_lowrank":
        trainable_params = bcm_params + lowrank_params
        stored_params = bcm_params + lowrank_params
        total_ops = direct_ops + lowrank_ops
        latency_proxy = total_ops / max(1, out_features * in_features)
    elif method == "lowrank_svd":
        trainable_params = lowrank_params
        stored_params = lowrank_params
        total_ops = lowrank_ops
        latency_proxy = lowrank_ops / max(1, out_features * in_features)
    elif method == "multi_bcm":
        trainable_params = num_bcm_basis * bcm_params
        stored_params = trainable_params
        total_ops = num_bcm_basis * direct_ops
        latency_proxy = total_ops / max(1, out_features * in_features)
    elif method == "bcm_plus_sparse_delta":
        sparse_params = sparse_param_count(in_features, out_features, sparse_ratio)
        trainable_params = bcm_params + sparse_params
        stored_params = trainable_params
        total_ops = direct_ops + sparse_params
        latency_proxy = total_ops / max(1, out_features * in_features)
    elif method == "fourierft":
        freq_params = fourier_param_count(freq_rows, freq_cols)
        trainable_params = freq_params
        stored_params = freq_params
        total_ops = out_features * in_features
        latency_proxy = 1.0
    else:
        trainable_params = 0
        stored_params = 0
        total_ops = 0
        latency_proxy = 0.0

    estimated_bits_fp16 = stored_params * 16
    estimated_bits_int8 = stored_params * 8
    if method in {"generator_delta_bcm", "bcm_only"} and use_generator_delta:
        estimated_bits_int4_delta = bcm_params * 16 + bcm_params * 4
    elif method == "generator_delta_bcm":
        estimated_bits_int4_delta = bcm_params * 16 + bcm_params * 4
    else:
        estimated_bits_int4_delta = stored_params * 4

    return CostReport(
        method=method,
        trainable_params=trainable_params,
        stored_params=stored_params,
        estimated_bits_fp16=estimated_bits_fp16,
        estimated_bits_int8=estimated_bits_int8,
        estimated_bits_int4_delta=estimated_bits_int4_delta,
        bcm_direct_ops=direct_ops if "bcm" in method else 0,
        bcm_fft_ops=fft_ops if "bcm" in method else 0,
        lowrank_ops=lowrank_ops if "lowrank" in method or method == "lowrank_svd" else 0,
        total_ops=total_ops,
        activation_memory=activation_memory,
        extra_latency_proxy=latency_proxy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--in-features", type=int, required=True)
    parser.add_argument("--out-features", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num-bcm-basis", type=int, default=1)
    parser.add_argument("--sparse-ratio", type=float, default=0.05)
    parser.add_argument("--use-generator-delta", action="store_true")
    args = parser.parse_args()

    report = estimate_method_cost(
        method=args.method,
        in_features=args.in_features,
        out_features=args.out_features,
        block_size=args.block_size,
        rank=args.rank,
        num_bcm_basis=args.num_bcm_basis,
        sparse_ratio=args.sparse_ratio,
        use_generator_delta=args.use_generator_delta,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
