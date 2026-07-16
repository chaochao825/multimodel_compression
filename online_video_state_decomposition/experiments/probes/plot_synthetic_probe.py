from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    summary = json.loads(
        (result_dir / "synthetic_summary.json").read_text(encoding="utf-8")
    )["summary"]
    transport = read_csv(result_dir / "synthetic_transport.csv")
    concentration = read_csv(
        result_dir / "synthetic_residual_concentration.csv"
    )

    methods = [
        "identity",
        "estimated_shift",
        "local_bttb_causal",
        "local_bccb_causal",
        "global_bccb_causal",
        "oracle_shift",
    ]
    stable_errors = [
        float(summary["method_summary"]["transport"][method]["stable_error"])
        for method in methods
    ]
    event_errors = [
        float(summary["method_summary"]["event"][method]["event_error"])
        for method in methods
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    x = np.arange(len(methods))
    width = 0.38
    axes[0].bar(
        x - width / 2,
        stable_errors,
        width,
        label="stable",
        color="#1f6f8b",
    )
    axes[0].bar(
        x + width / 2,
        event_errors,
        width,
        label="event",
        color="#e07a5f",
    )
    axes[0].set_xticks(x, methods, rotation=35, ha="right")
    axes[0].set_ylabel("Relative activation error")
    axes[0].set_title("Transport prediction")
    axes[0].legend(frameon=False)

    rank_rows = json.loads(
        (result_dir / "synthetic_summary.json").read_text(encoding="utf-8")
    )["rank_metrics"]
    ranks = sorted(
        int(rank)
        for rank in rank_rows[0]["aligned_temporal"]["energy_at_rank"]
    )
    aligned = [
        np.mean(
            [
                row["aligned_temporal"]["energy_at_rank"][str(rank)]
                for row in rank_rows
            ]
        )
        for rank in ranks
    ]
    unaligned = [
        np.mean(
            [
                row["unaligned_temporal"]["energy_at_rank"][str(rank)]
                for row in rank_rows
            ]
        )
        for rank in ranks
    ]
    axes[1].plot(ranks, aligned, marker="o", color="#2a9d8f", label="aligned")
    axes[1].plot(
        ranks,
        unaligned,
        marker="s",
        color="#6c757d",
        label="unaligned",
    )
    axes[1].axhline(0.70, color="#d1495b", linestyle="--", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Explained energy")
    axes[1].set_title("Temporal spectrum")
    axes[1].legend(frameon=False)

    event_rows = [
        row
        for row in concentration
        if row["is_event"] == "1"
        and row["is_scene_cut"] == "0"
    ]
    fractions = sorted({float(row["fraction"]) for row in event_rows})
    energy = [
        np.mean(
            [
                float(row["energy_ratio"])
                for row in event_rows
                if abs(float(row["fraction"]) - fraction) < 1e-9
            ]
        )
        for fraction in fractions
    ]
    recall = [
        np.mean(
            [
                float(row["event_recall"])
                for row in event_rows
                if abs(float(row["fraction"]) - fraction) < 1e-9
            ]
        )
        for fraction in fractions
    ]
    axes[2].plot(
        np.asarray(fractions) * 100,
        energy,
        marker="o",
        color="#e76f51",
        label="residual energy",
    )
    axes[2].plot(
        np.asarray(fractions) * 100,
        recall,
        marker="s",
        color="#264653",
        label="event recall",
    )
    axes[2].set_ylim(0, 1.03)
    axes[2].set_xlabel("Selected blocks (%)")
    axes[2].set_ylabel("Ratio")
    axes[2].set_title("Sparse event concentration")
    axes[2].legend(frameon=False)

    fig.suptitle("Controlled latent-dynamics probe", fontsize=14, y=1.02)
    fig.tight_layout()
    output = result_dir / "synthetic_probe_overview.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
