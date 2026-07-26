#!/usr/bin/env python3
"""Visualize held-out joint quantization and residual-shaping results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = (
    "ptq_only",
    "ptq_weight_svd",
    "ptq_activation_lr",
    "joint_shaped_ptq_lr_block_sparse",
    "joint_residual_shaped_ptq_lr",
    "joint_residual_shaped_ptq_lr_block_sparse",
)
METHOD_LABELS = {
    "ptq_only": "PTQ",
    "ptq_weight_svd": "PTQ +\nweight SVD",
    "ptq_activation_lr": "PTQ +\nactivation LR",
    "joint_shaped_ptq_lr_block_sparse": "post-hoc LR\n+ sparse",
    "joint_residual_shaped_ptq_lr": "residual-shaped\nPTQ + LR",
    "joint_residual_shaped_ptq_lr_block_sparse": "residual-shaped\nPTQ + LR + S",
}
METHOD_COLORS = {
    "ptq_only": "#999999",
    "ptq_weight_svd": "#CC6677",
    "ptq_activation_lr": "#44AA99",
    "joint_shaped_ptq_lr_block_sparse": "#AA4499",
    "joint_residual_shaped_ptq_lr": "#4477AA",
    "joint_residual_shaped_ptq_lr_block_sparse": "#EE7733",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("joint-shaping CSV is empty")
    return rows


def best_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        method = row["method"]
        if method not in METHOD_ORDER:
            continue
        key = (int(row["block"]), method)
        current = selected.get(key)
        if current is None or float(row["test_relative_l2"]) < float(
            current["test_relative_l2"]
        ):
            selected[key] = row

    output: list[dict[str, object]] = []
    for (block, method), row in sorted(
        selected.items(), key=lambda item: (item[0][0], METHOD_ORDER.index(item[0][1]))
    ):
        output.append(
            {
                "block": block,
                "method": method,
                "rank": int(row.get("rank", 0) or 0),
                "clip": float(row.get("clip", 0.0) or 0.0),
                "sparse_ratio": float(row.get("sparse_ratio", 0.0) or 0.0),
                "validation_relative_l2": float(
                    row.get("validation_relative_l2", 0.0) or 0.0
                ),
                "test_relative_l2": float(row["test_relative_l2"]),
                "defect_energy_captured_test": float(
                    row.get("defect_energy_captured_test", 0.0) or 0.0
                ),
                "subspace_overlap": float(row.get("subspace_overlap", 0.0) or 0.0),
                "estimated_stored_bits": int(row.get("estimated_stored_bits", 0) or 0),
                "estimated_extra_ops": int(row.get("estimated_extra_ops", 0) or 0),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
    blocks = sorted({int(row["block"]) for row in rows})

    width = 0.36
    x_values = list(range(len(METHOD_ORDER)))
    for block_index, block in enumerate(blocks):
        lookup = {
            str(row["method"]): row for row in rows if int(row["block"]) == block
        }
        errors = [
            100.0 * float(lookup[method]["test_relative_l2"])
            if method in lookup
            else float("nan")
            for method in METHOD_ORDER
        ]
        offset = (block_index - (len(blocks) - 1) / 2.0) * width
        axes[0, 0].bar(
            [value + offset for value in x_values],
            errors,
            width=width,
            label=f"block {block}",
            color=("#7F8FA6", "#E58B58")[block_index % 2],
            edgecolor="#222222",
            linewidth=0.4,
        )
    axes[0, 0].axhline(2.0, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[0, 0].set_xticks(
        x_values, [METHOD_LABELS[method] for method in METHOD_ORDER], rotation=18, ha="right"
    )
    axes[0, 0].set_ylabel("Held-out output relative L2 (%)")
    axes[0, 0].set_title("(a) Best held-out error per method")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.2)

    valid_rows = [
        row
        for row in rows
        if float(row["validation_relative_l2"]) > 0.0
        and str(row["method"]) != "ptq_only"
    ]
    for row in valid_rows:
        method = str(row["method"])
        marker = "o" if int(row["block"]) == blocks[0] else "^"
        axes[0, 1].scatter(
            100.0 * float(row["validation_relative_l2"]),
            100.0 * float(row["test_relative_l2"]),
            color=METHOD_COLORS[method],
            marker=marker,
            s=65,
            edgecolor="#222222",
            linewidth=0.45,
        )
    limit = max(
        100.0 * float(row["test_relative_l2"]) for row in valid_rows
    ) * 1.08
    axes[0, 1].plot([0.0, limit], [0.0, limit], color="#666666", linestyle=":")
    axes[0, 1].set_xlim(0.0, limit)
    axes[0, 1].set_ylim(0.0, limit)
    axes[0, 1].set_xlabel("Validation output relative L2 (%)")
    axes[0, 1].set_ylabel("Held-out output relative L2 (%)")
    axes[0, 1].set_title("(b) Calibration-selection generalization gap")
    axes[0, 1].grid(alpha=0.2)

    lr_rows = [
        row
        for row in rows
        if int(row["rank"]) > 0 and "weight_svd" not in str(row["method"])
    ]
    for row in lr_rows:
        method = str(row["method"])
        marker = "o" if int(row["block"]) == blocks[0] else "^"
        axes[1, 0].scatter(
            float(row["subspace_overlap"]),
            100.0 * float(row["defect_energy_captured_test"]),
            color=METHOD_COLORS[method],
            marker=marker,
            s=70,
            edgecolor="#222222",
            linewidth=0.45,
        )
    axes[1, 0].axvline(0.7, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[1, 0].axhline(70.0, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xlim(0.0, 0.8)
    axes[1, 0].set_ylim(0.0, 100.0)
    axes[1, 0].set_xlabel("Cross-seed correction-subspace overlap")
    axes[1, 0].set_ylabel("Held-out defect energy captured (%)")
    axes[1, 0].set_title("(c) Defect shapeability and transfer gates")
    axes[1, 0].grid(alpha=0.2)

    for row in rows:
        method = str(row["method"])
        marker = "o" if int(row["block"]) == blocks[0] else "^"
        axes[1, 1].scatter(
            float(row["estimated_stored_bits"]) / 8.0 / 2**20,
            100.0 * float(row["test_relative_l2"]),
            color=METHOD_COLORS[method],
            marker=marker,
            s=64,
            edgecolor="#222222",
            linewidth=0.45,
        )
    axes[1, 1].axhline(2.0, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[1, 1].set_xlabel("Estimated adapter-aware storage (MiB)")
    axes[1, 1].set_ylabel("Held-out output relative L2 (%)")
    axes[1, 1].set_title("(d) Error-storage trade-off")
    axes[1, 1].grid(alpha=0.2)

    for method in METHOD_ORDER:
        axes[1, 1].scatter(
            [], [], color=METHOD_COLORS[method], label=METHOD_LABELS[method].replace("\n", " ")
        )
    axes[1, 1].scatter([], [], color="#777777", marker="o", label=f"block {blocks[0]}")
    if len(blocks) > 1:
        axes[1, 1].scatter([], [], color="#777777", marker="^", label=f"block {blocks[1]}")
    axes[1, 1].legend(frameon=False, fontsize=6.7, ncol=2, loc="upper right")

    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"joint_quant_lr_shaping.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = best_rows(read_rows(args.input))
    if not rows:
        raise ValueError("no supported joint-shaping methods found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "joint_quant_lr_shaping_plot_data.csv", rows)
    plot(rows, args.output_dir)


if __name__ == "__main__":
    main()
