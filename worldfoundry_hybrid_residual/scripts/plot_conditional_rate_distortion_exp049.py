#!/usr/bin/env python3
"""Plot the EXP-049 exposed local conditional rate-distortion screen."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "self_attn": "#0B6E75",
    "ffn": "#D97732",
    "whole_block": "#334E68",
}
LABELS = {
    "self_attn": "Self-attention",
    "ffn": "FFN",
    "whole_block": "Whole block",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                **row,
                "selection_threshold": float(row["selection_threshold"]),
                "selected_fraction": float(row["selected_fraction"]),
                "deployable_risk": float(row["deployable_risk"]),
                "deployable_worst": float(row["deployable_worst"]),
                "target_visible_risk": float(row["target_visible_risk"]),
                "zero_renderer_e2e_speedup_ceiling": float(
                    row["zero_renderer_e2e_speedup_ceiling"]
                ),
            }
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    args = parse_args()
    rows = read_rows(args.frontier)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    for target in ("self_attn", "ffn", "whole_block"):
        target_rows = sorted(
            (row for row in rows if row["target"] == target),
            key=lambda row: float(row["selection_threshold"]),
        )
        color = COLORS[target]
        axes[0].plot(
            [float(row["selected_fraction"]) * 100 for row in target_rows],
            [float(row["deployable_risk"]) * 100 for row in target_rows],
            marker="o",
            linewidth=1.8,
            color=color,
            label=LABELS[target],
        )
        axes[1].plot(
            [float(row["zero_renderer_e2e_speedup_ceiling"]) for row in target_rows],
            [float(row["deployable_risk"]) * 100 for row in target_rows],
            marker="o",
            linewidth=1.8,
            color=color,
            label=LABELS[target],
        )
    axes[0].axhline(1.0, color="#B42318", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Selection-frozen calls (%)")
    axes[0].set_ylabel("Held-out additive block-output risk (%)")
    axes[0].set_title("Local quality versus certified coverage")
    axes[0].grid(alpha=0.22)
    axes[1].axhline(1.0, color="#B42318", linestyle="--", linewidth=1)
    axes[1].axvline(1.2, color="#B42318", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Optimistic end-to-end ceiling (zero renderer cost)")
    axes[1].set_ylabel("Held-out additive block-output risk (%)")
    axes[1].set_title("Stage-0 screen, not H200 timing")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, loc="best")
    figure.suptitle(
        "EXP-049 target-separated conditional interface screen",
        fontsize=12,
        fontweight="bold",
    )
    for extension in ("png", "pdf"):
        figure.savefig(
            args.out_dir / f"conditional_rate_distortion_screen.{extension}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
