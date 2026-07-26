#!/usr/bin/env python3
"""Plot numerical and system gates for the content-structured attention probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "centroid": "#0072B2",
    "diag_gaussian": "#D55E00",
    "oracle_nonnegative": "#009E73",
    "ridge_nonnegative": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--displacement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attention-share", type=float, default=0.5388)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def plot_block_pareto(
    axis: plt.Axes, rows: list[dict[str, str]], plot_data: list[dict[str, object]]
) -> None:
    selected = [row for row in rows if row["split"] == "test"]
    for (method, router), group in _group(selected, "method", "router").items():
        ordered = sorted(group, key=lambda row: float(row["attention_work_ratio_mean"]))
        x = [float(row["attention_work_ratio_mean"]) for row in ordered]
        y = [100 * float(row["aggregate_output_relative_l2"]) for row in ordered]
        axis.scatter(
            x,
            y,
            s=30,
            marker="o" if router == "moment" else "x",
            color=COLORS[method],
            alpha=0.85,
            label=f"{method}, {router}",
        )
        for row, x_value, y_value in zip(ordered, x, y):
            plot_data.append(
                {
                    "panel": "block_pareto",
                    "method": method,
                    "router": router,
                    "density": row["density"],
                    "tail_group_size": row["tail_group_size"],
                    "work_ratio": x_value,
                    "error_percent": y_value,
                }
            )
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1, label="1% target")
    axis.axvline(0.5, color="#777777", linestyle=":", linewidth=1)
    axis.set_xlabel("Arithmetic attention work / dense")
    axis.set_ylabel("Aggregate output relative L2 (%)")
    axis.set_yscale("log")
    axis.text(-0.13, 1.05, "a", transform=axis.transAxes, fontweight="bold", fontsize=12)
    axis.legend(fontsize=7, ncol=2)


def plot_router_gap(
    axis: plt.Axes, rows: list[dict[str, str]], plot_data: list[dict[str, object]]
) -> None:
    lookup = {
        (row["method"], row["density"], row["tail_group_size"], row["router"]): row
        for row in rows
        if row["split"] == "test"
    }
    points = []
    configurations = sorted(
        {(method, density, tail) for method, density, tail, _ in lookup}
    )
    for method, density, tail in configurations:
        moment = lookup.get((method, density, tail, "moment"))
        oracle = lookup.get((method, density, tail, "oracle_mass"))
        if not moment or not oracle:
            continue
        oracle_error = 100 * float(oracle["aggregate_output_relative_l2"])
        moment_error = 100 * float(moment["aggregate_output_relative_l2"])
        points.append((method, density, tail, oracle_error, moment_error))
    for method in sorted({point[0] for point in points}):
        subset = [point for point in points if point[0] == method]
        axis.scatter(
            [point[3] for point in subset],
            [point[4] for point in subset],
            color=COLORS[method],
            s=35,
            label=method,
        )
        for point in subset:
            plot_data.append(
                {
                    "panel": "router_gap",
                    "method": point[0],
                    "density": point[1],
                    "tail_group_size": point[2],
                    "oracle_error_percent": point[3],
                    "moment_error_percent": point[4],
                }
            )
    maximum = max([max(point[3], point[4]) for point in points], default=1.0)
    axis.plot([0, maximum], [0, maximum], color="#555555", linestyle="--", linewidth=1)
    axis.set_xlabel("Oracle-router error (%)")
    axis.set_ylabel("Deployable moment-router error (%)")
    axis.text(-0.13, 1.05, "b", transform=axis.transAxes, fontweight="bold", fontsize=12)
    axis.legend(fontsize=8)


def plot_displacement(
    axis: plt.Axes, rows: list[dict[str, str]], plot_data: list[dict[str, object]]
) -> None:
    selected = [
        row
        for row in rows
        if row["split"] == "test"
        and row["head_role"] == "localized"
        and row["prediction"] in ("oracle_nonnegative", "ridge_nonnegative")
    ]
    for prediction, group in _group(selected, "prediction").items():
        ordered = sorted(group, key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in ordered]
        errors = [100 * float(row["aggregate_full_output_relative_l2"]) for row in ordered]
        worst = [100 * float(row["record_error_max"]) for row in ordered]
        axis.plot(
            ranks,
            errors,
            marker="o",
            color=COLORS[prediction],
            label=prediction.replace("_nonnegative", ""),
        )
        axis.fill_between(ranks, errors, worst, color=COLORS[prediction], alpha=0.12)
        for row, rank, error, maximum in zip(ordered, ranks, errors, worst):
            plot_data.append(
                {
                    "panel": "displacement",
                    "prediction": prediction,
                    "rank": rank,
                    "aggregate_error_percent": error,
                    "worst_record_error_percent": maximum,
                    "local_mass_mean": row["local_mass_mean"],
                }
            )
    axis.axhline(2.0, color="#333333", linestyle="--", linewidth=1, label="2% record gate")
    axis.set_xlabel("Displacement expert rank")
    axis.set_ylabel("Localized full-output error (%)")
    axis.text(-0.13, 1.05, "c", transform=axis.transAxes, fontweight="bold", fontsize=12)
    axis.legend(fontsize=8)


def plot_amdahl(
    axis: plt.Axes, attention_share: float, plot_data: list[dict[str, object]]
) -> None:
    local_speedups = [1.0 + index * 0.05 for index in range(61)]
    denoiser_speedups = [
        1.0 / ((1.0 - attention_share) + attention_share / speedup)
        for speedup in local_speedups
    ]
    axis.plot(local_speedups, denoiser_speedups, color="#E69F00", linewidth=2)
    for local in (1.15, 1.5, 2.0):
        end_to_end = 1.0 / ((1.0 - attention_share) + attention_share / local)
        axis.scatter([local], [end_to_end], color="#222222", s=25, zorder=3)
        axis.annotate(f"{local:.2g}x -> {end_to_end:.3f}x", (local, end_to_end), xytext=(5, 5), textcoords="offset points", fontsize=7)
    for local, end_to_end in zip(local_speedups, denoiser_speedups):
        plot_data.append(
            {
                "panel": "amdahl",
                "attention_share": attention_share,
                "local_attention_speedup": local,
                "denoiser_speedup_upper_bound": end_to_end,
            }
        )
    axis.set_xlabel("Local attention kernel speedup")
    axis.set_ylabel("Denoiser speedup upper bound")
    axis.text(-0.13, 1.05, "d", transform=axis.transAxes, fontweight="bold", fontsize=12)


def _group(
    rows: list[dict[str, str]], *fields: str
) -> dict[object, list[dict[str, str]]]:
    output: dict[object, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key_tuple = tuple(row[field] for field in fields)
        key: object = key_tuple[0] if len(key_tuple) == 1 else key_tuple
        output[key].append(row)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    block_rows = read_csv(args.block_dir / "block_moment_marginal_summary.csv")
    displacement_rows = read_csv(
        args.displacement_dir / "local_displacement_mixture_roles.csv"
    )
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    plot_data: list[dict[str, object]] = []
    plot_block_pareto(axes[0, 0], block_rows, plot_data)
    plot_router_gap(axes[0, 1], block_rows, plot_data)
    plot_displacement(axes[1, 0], displacement_rows, plot_data)
    plot_amdahl(axes[1, 1], args.attention_share, plot_data)
    for extension in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"content_structure_gates.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    write_csv(args.output_dir / "content_structure_gates_data.csv", plot_data)
    manifest = {
        "block_decision": json.loads((args.block_dir / "decision.json").read_text()),
        "displacement_decision": json.loads(
            (args.displacement_dir / "decision.json").read_text()
        ),
        "attention_share": args.attention_share,
    }
    (args.output_dir / "content_structure_gates_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
