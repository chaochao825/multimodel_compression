#!/usr/bin/env python3
"""Derive source-bound local and end-to-end ceilings for Wan FFN fusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--profile-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--cfg-branches", type=int, default=2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def amdahl_speedup(share: float, local_speedup: float) -> float:
    if not 0.0 <= share < 1.0:
        raise ValueError(f"share must be in [0, 1), received {share}")
    if local_speedup <= 0.0:
        raise ValueError("local speedup must be positive")
    if math.isinf(local_speedup):
        return 1.0 / (1.0 - share)
    return 1.0 / (1.0 - share + share / local_speedup)


def benchmark_value(
    rows: list[dict[str, str]], token_rows: int, operation: str, width: int
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if int(row["rows"]) == token_rows
        and int(row["width"]) == width
        and row["operation"] == operation
        and row["status"] == "ok"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one benchmark row for rows={token_rows}, width={width}, "
            f"operation={operation}; found {len(matches)}"
        )
    return matches[0]


def profile_shares(
    rows: list[dict[str, str]], case: str
) -> tuple[float, dict[str, float]]:
    selected = [row for row in rows if row["case"] == case]
    if not selected:
        raise ValueError(f"profile has no rows for {case}")
    total_ms = sum(float(row["per_denoise_step_ms"]) for row in selected)
    shares = {
        row["component"]: float(row["per_denoise_step_ms"]) / total_ms
        for row in selected
    }
    return total_ms, shares


def analyze_cases(
    benchmark_rows: list[dict[str, str]],
    profile_rows: list[dict[str, str]],
    *,
    layers: int,
    cfg_branches: int,
) -> list[dict[str, object]]:
    if layers <= 0 or cfg_branches <= 0:
        raise ValueError("layers and CFG branches must be positive")
    cases = {7800: "F17", 32760: "F81"}
    hidden_width = 1536
    ffn_width = 8960
    calls_per_step = layers * cfg_branches
    output: list[dict[str, object]] = []

    for token_rows, case in cases.items():
        full = benchmark_value(
            benchmark_rows, token_rows, "ffn_full_eager", ffn_width
        )
        up_no_bias = benchmark_value(
            benchmark_rows, token_rows, "ffn_up_linear_no_bias", ffn_width
        )
        up_eager = benchmark_value(
            benchmark_rows, token_rows, "ffn_up_linear_bias_gelu_eager", ffn_width
        )
        up_triton = benchmark_value(
            benchmark_rows, token_rows, "ffn_up_linear_triton_bias_gelu", ffn_width
        )
        intermediate_copy = benchmark_value(
            benchmark_rows, token_rows, "copy", ffn_width
        )

        full_ms = float(full["latency_ms_median"])
        up_no_bias_ms = float(up_no_bias["latency_ms_median"])
        up_eager_ms = float(up_eager["latency_ms_median"])
        up_triton_ms = float(up_triton["latency_ms_median"])
        copy_ms = float(intermediate_copy["latency_ms_median"])
        epilogue_overhead_ms = up_eager_ms - up_no_bias_ms
        if min(full_ms - epilogue_overhead_ms, full_ms - copy_ms) <= 0.0:
            raise ValueError(f"invalid fusion ceiling inputs for {case}")

        ideal_epilogue_local = full_ms / (full_ms - epilogue_overhead_ms)
        ideal_traffic_local = full_ms / (full_ms - copy_ms)
        triton_projected_full_ms = full_ms - up_eager_ms + up_triton_ms
        triton_projected_local = full_ms / triton_projected_full_ms
        total_step_ms, shares = profile_shares(profile_rows, case)
        estimated_ffn_ms = calls_per_step * full_ms
        estimated_ffn_share = estimated_ffn_ms / total_step_ms
        if not 0.0 <= estimated_ffn_share < 1.0:
            raise ValueError(
                f"estimated FFN share for {case} is invalid: {estimated_ffn_share}"
            )
        elementwise_share = shares["elementwise_memory"]
        triton_relative_l2 = float(up_triton.get("relative_l2_vs_torch") or math.nan)
        triton_max_abs = float(up_triton.get("max_abs_vs_torch") or math.nan)
        triton_exact = triton_relative_l2 == 0.0 and triton_max_abs == 0.0

        output.append(
            {
                "case": case,
                "token_rows": token_rows,
                "hidden_width": hidden_width,
                "ffn_width": ffn_width,
                "ffn_calls_per_denoise_step": calls_per_step,
                "profile_step_ms": total_step_ms,
                "full_ffn_ms": full_ms,
                "up_bias_gelu_ms": up_eager_ms,
                "up_no_bias_ms": up_no_bias_ms,
                "epilogue_overhead_ms": epilogue_overhead_ms,
                "intermediate_copy_ms": copy_ms,
                "triton_up_bias_gelu_ms": up_triton_ms,
                "triton_relative_l2": triton_relative_l2,
                "triton_max_abs": triton_max_abs,
                "triton_bitwise_exact_proxy": triton_exact,
                "ideal_epilogue_local_speedup": ideal_epilogue_local,
                "ideal_intermediate_traffic_local_speedup": ideal_traffic_local,
                "standalone_triton_projected_local_speedup": triton_projected_local,
                "estimated_ffn_share": estimated_ffn_share,
                "profile_elementwise_share": elementwise_share,
                "profile_linear_share": shares["linear_gemm"],
                "profile_self_attention_share": shares["self_attention_core"],
                "ideal_epilogue_e2e_speedup": amdahl_speedup(
                    estimated_ffn_share, ideal_epilogue_local
                ),
                "ideal_intermediate_traffic_e2e_speedup": amdahl_speedup(
                    estimated_ffn_share, ideal_traffic_local
                ),
                "standalone_triton_projected_e2e_speedup": amdahl_speedup(
                    estimated_ffn_share, triton_projected_local
                ),
                "remove_entire_ffn_e2e_ceiling": amdahl_speedup(
                    estimated_ffn_share, math.inf
                ),
                "remove_all_elementwise_e2e_ceiling": amdahl_speedup(
                    elementwise_share, math.inf
                ),
                "standalone_triton_decision": (
                    "GO"
                    if triton_exact and triton_projected_local > 1.0
                    else "NO-GO"
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    rows = analyze_cases(
        read_csv(args.benchmark_csv),
        read_csv(args.profile_csv),
        layers=args.layers,
        cfg_branches=args.cfg_branches,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ffn_fusion_ceiling.csv", rows)
    payload = {
        "scope": "source-bound Wan FFN fusion ceilings from measured H200 kernels",
        "assumptions": {
            "layers": args.layers,
            "cfg_branches": args.cfg_branches,
            "ffn_calls_per_denoise_step": args.layers * args.cfg_branches,
            "ideal_epilogue": "removes all measured up-projection bias/GELU overhead",
            "ideal_intermediate_traffic": "removes one measured BF16 copy of the FFN hidden activation",
            "remove_entire_ffn": "non-deployable upper bound",
            "remove_all_elementwise": "non-deployable profiler-category upper bound",
        },
        "rows": rows,
        "interpretation": (
            "Standalone pointwise replacement is rejected unless it is exact and "
            "reduces projected complete-FFN latency. Larger gains require whole-block "
            "fusion or an algorithm that removes work, not another post-GEMM launch."
        ),
    }
    write_json(args.output_dir / "ffn_fusion_ceiling.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
