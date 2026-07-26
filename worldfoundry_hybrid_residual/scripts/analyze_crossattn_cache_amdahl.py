#!/usr/bin/env python3
"""Estimate the arithmetic coverage of exact text cross-attention K/V caching."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_cases(text: str) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for item in text.split(","):
        name, separator, tokens = item.strip().partition(":")
        if not separator or not name or int(tokens) <= 0:
            raise argparse.ArgumentTypeError("cases must use NAME:POSITIVE_TOKENS")
        cases.append((name, int(tokens)))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("F17:7800,F81:32760"))
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--ffn-dim", type=int, default=8960)
    parser.add_argument("--text-tokens", type=int, default=512)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--sampling-steps", type=int, default=20)
    return parser.parse_args()


def case_metrics(
    name: str,
    video_tokens: int,
    dim: int,
    ffn_dim: int,
    text_tokens: int,
    layers: int,
    steps: int,
) -> dict[str, object]:
    # A multiply-add is counted as two FLOPs.
    per_layer = {
        "self_attention_qkvo": 8 * video_tokens * dim * dim,
        "self_attention_qk_pv": 4 * video_tokens * video_tokens * dim,
        "cross_attention_qo": 4 * video_tokens * dim * dim,
        "cross_attention_text_kv": 4 * text_tokens * dim * dim,
        "cross_attention_qk_pv": 4 * video_tokens * text_tokens * dim,
        "ffn_up_down": 4 * video_tokens * dim * ffn_dim,
    }
    per_step = {key: value * layers for key, value in per_layer.items()}
    total_per_step = sum(per_step.values())
    cacheable_fraction = per_step["cross_attention_text_kv"] / total_per_step
    reusable_fraction = (steps - 1) / steps
    eliminated_fraction = cacheable_fraction * reusable_fraction
    ideal_flop_speedup = 1.0 / (1.0 - eliminated_fraction)
    return {
        "case": name,
        "video_tokens": video_tokens,
        "text_tokens": text_tokens,
        "layers": layers,
        "sampling_steps": steps,
        **{f"{key}_flops_per_step": value for key, value in per_step.items()},
        "total_flops_per_step": total_per_step,
        "text_kv_cacheable_flop_fraction": cacheable_fraction,
        "text_kv_eliminated_flop_fraction": eliminated_fraction,
        "ideal_flop_speedup": ideal_flop_speedup,
    }


def main() -> None:
    args = parse_args()
    if min(args.dim, args.ffn_dim, args.text_tokens, args.layers, args.sampling_steps) <= 0:
        raise ValueError("model dimensions and sampling steps must be positive")
    rows = [
        case_metrics(
            name,
            tokens,
            args.dim,
            args.ffn_dim,
            args.text_tokens,
            args.layers,
            args.sampling_steps,
        )
        for name, tokens in args.cases
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "crossattn_cache_amdahl_proxy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = [
        ("self_attention_qk_pv", "Self-attention QK/PV", "#174a5b"),
        ("ffn_up_down", "FFN up/down", "#d27732"),
        ("self_attention_qkvo", "Self-attention QKVO", "#4f7d8c"),
        ("cross_attention_qo", "Cross-attention Q/O", "#76a6a0"),
        ("cross_attention_qk_pv", "Cross-attention QK/PV", "#d9a75d"),
        ("cross_attention_text_kv", "Cacheable text K/V", "#a83232"),
    ]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.5))
    names = [str(row["case"]) for row in rows]
    bottom = [0.0] * len(rows)
    for key, label, color in components:
        values = [
            100.0 * float(row[f"{key}_flops_per_step"]) / float(row["total_flops_per_step"])
            for row in rows
        ]
        left.bar(names, values, bottom=bottom, label=label, color=color)
        bottom = [old + value for old, value in zip(bottom, values)]
    left.set_ylim(0.0, 100.0)
    left.set_ylabel("Arithmetic share per denoiser call (%)")
    left.set_title("(a) Wan block FLOP composition proxy")
    left.legend(fontsize=7.6, loc="upper right")

    speedups = [float(row["ideal_flop_speedup"]) for row in rows]
    bars = right.bar(names, speedups, color="#a83232", width=0.55)
    right.axhline(1.0, color="#303030", linewidth=1.0)
    right.set_ylim(1.0, max(1.01, max(speedups) * 1.004))
    right.set_ylabel("Ideal speedup if saved FLOPs cost zero")
    right.set_title("(b) Exact text K/V cache Amdahl ceiling")
    for bar, row, speedup in zip(bars, rows, speedups):
        saved = 100.0 * float(row["text_kv_eliminated_flop_fraction"])
        right.text(
            bar.get_x() + bar.get_width() / 2,
            speedup,
            f"{speedup:.4f}x\n({saved:.3f}% FLOPs)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.suptitle("Why exact text cross-attention K/V caching is a systems hygiene optimization")
    figure.text(
        0.5,
        -0.02,
        "FLOPs are a structural proxy, not an H200 latency prediction; kernel inefficiency may change the measured share.",
        ha="center",
        fontsize=8.3,
        color="#4a4a4a",
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"crossattn_cache_amdahl_proxy.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)

    summary = {
        "model": {
            "dim": args.dim,
            "ffn_dim": args.ffn_dim,
            "text_tokens": args.text_tokens,
            "layers": args.layers,
            "sampling_steps": args.sampling_steps,
        },
        "rows": rows,
        "scope_warning": (
            "This is a dense arithmetic proxy. It excludes normalization, modulation, "
            "VAE, scheduler, launch overhead, memory effects, and actual H200 kernel timing."
        ),
    }
    (args.output_dir / "crossattn_cache_amdahl_proxy.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
