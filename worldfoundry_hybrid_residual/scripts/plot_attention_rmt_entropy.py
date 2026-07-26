#!/usr/bin/env python3
"""Plot attention support and Q/K random-matrix spectra."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heads = read_csv(args.input_dir / "attention_rmt_entropy_heads.csv")
    eigen = read_csv(args.input_dir / "attention_rmt_eigenvalues.csv")
    if not heads or not eigen:
        raise ValueError("RMT probe CSVs are empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_ids = [int(row["head"]) for row in heads]
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))

    actual_entropy = [100.0 * float(row["actual_normalized_entropy_mean"]) for row in heads]
    gaussian_entropy = [100.0 * float(row["gaussian_normalized_entropy_mean"]) for row in heads]
    axes[0].plot(head_ids, actual_entropy, marker="o", label="Wan attention")
    axes[0].plot(head_ids, gaussian_entropy, marker="s", label="matched Gaussian")
    axes[0].set_title("(a) Normalized row entropy")
    axes[0].set_ylabel("Entropy / log(N) (%)")
    axes[0].legend(fontsize=8)

    support = [
        100.0 * float(row["actual_participation_support_fraction_mean"])
        for row in heads
    ]
    geometry = [100.0 * float(row["geometry_mass_mean"]) for row in heads]
    top1024 = [100.0 * float(row["actual_top1024_mass_mean"]) for row in heads]
    axes[1].plot(head_ids, support, marker="o", label="participation support / N")
    axes[1].plot(head_ids, geometry, marker="s", label="temporal-PM2 geometry mass")
    axes[1].plot(head_ids, top1024, marker="^", label="top-1024 mass")
    axes[1].set_title("(b) Support concentration")
    axes[1].set_ylabel("Fraction (%)")
    axes[1].legend(fontsize=7.5)

    by_source: dict[str, list[list[float]]] = defaultdict(list)
    mp_upper: dict[str, float] = {}
    by_head_source: dict[tuple[int, str], list[tuple[int, float]]] = defaultdict(list)
    for row in eigen:
        source = row["source"]
        by_head_source[(int(row["head"]), source)].append(
            (int(row["ascending_index"]), float(row["eigenvalue"]))
        )
        mp_upper[source] = float(row["mp_upper"])
    for (_, source), values in sorted(by_head_source.items()):
        by_source[source].append([value for _, value in sorted(values)])
    for source, spectra in sorted(by_source.items()):
        mean_spectrum = [sum(values) / len(values) for values in zip(*spectra)]
        axes[2].plot(mean_spectrum, label=f"{source.upper()} mean spectrum")
        axes[2].axhline(
            mp_upper[source], linestyle=":", linewidth=1.1, label=f"{source.upper()} MP upper"
        )
    axes[2].set_yscale("log")
    axes[2].set_title("(c) Standardized covariance spectrum")
    axes[2].set_ylabel("Eigenvalue (log scale)")
    axes[2].set_xlabel("Ascending eigenvalue index")
    axes[2].legend(fontsize=7.3)

    for axis in axes[:2]:
        axis.set_xlabel("Attention head")
        axis.set_xticks(head_ids)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("F81 attention: token support and channel anisotropy", y=1.02)
    figure.text(
        0.5,
        -0.02,
        "Single replay screen; MP outliers diagnose channel covariance, not a deployable sparse token mask.",
        ha="center",
        fontsize=8.5,
        color="#4a4a4a",
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"attention_rmt_entropy.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
