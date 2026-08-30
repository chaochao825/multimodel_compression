from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ("positioned_equal_mass", "positioned_group_mass")
GEOMETRIES = ("flat_contiguous_4", "spatial_2x2")
EXPECTED_ROWS = 24 * len(MODES) * len(GEOMETRIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty plot data")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def one_sided_sign_pvalue(wins: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    return sum(math.comb(non_ties, count) for count in range(wins, non_ties + 1)) / (
        2**non_ties
    )


def bootstrap_paired(
    flat: np.ndarray,
    spatial: np.ndarray,
    *,
    seed: int,
    draws: int = 20_000,
) -> dict[str, float]:
    if flat.shape != spatial.shape or flat.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and equally sized")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, flat.size, size=(draws, flat.size))
    sampled_flat = flat[indices].mean(axis=1)
    sampled_spatial = spatial[indices].mean(axis=1)
    sampled_delta = sampled_spatial - sampled_flat
    sampled_ratio = sampled_spatial / sampled_flat
    return {
        "mean_kl_delta": float((spatial - flat).mean()),
        "mean_kl_delta_ci_low": float(np.quantile(sampled_delta, 0.025)),
        "mean_kl_delta_ci_high": float(np.quantile(sampled_delta, 0.975)),
        "mean_kl_ratio": float(spatial.mean() / flat.mean()),
        "mean_kl_ratio_ci_low": float(np.quantile(sampled_ratio, 0.025)),
        "mean_kl_ratio_ci_high": float(np.quantile(sampled_ratio, 0.975)),
    }


def validate_and_summarize(
    recorded: dict[str, object], rows: list[dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    sample_ids = {row["sample_id"] for row in rows}
    if len(sample_ids) != 24:
        raise ValueError(f"expected 24 samples, found {len(sample_ids)}")
    if float(recorded["maximum_flat_kl_repeat_error"]) != 0.0:
        raise ValueError("flat baseline did not exactly repeat M1")
    if int(recorded["flat_prediction_repeat_mismatches"]) != 0:
        raise ValueError("flat baseline predictions did not exactly repeat M1")

    aggregate_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    summary = recorded["summaries"]
    comparisons = recorded["comparisons"]
    for mode_index, mode in enumerate(MODES):
        by_geometry: dict[str, dict[str, dict[str, str]]] = {}
        for geometry in GEOMETRIES:
            selected = [
                row
                for row in rows
                if row["mode"] == mode and row["geometry"] == geometry
            ]
            if len(selected) != 24:
                raise ValueError(f"expected 24 rows for {mode}/{geometry}")
            lookup = {row["sample_id"]: row for row in selected}
            if set(lookup) != sample_ids:
                raise ValueError(f"sample identity mismatch for {mode}/{geometry}")
            by_geometry[geometry] = lookup
            kl = np.asarray([float(row["candidate_kl"]) for row in selected])
            recomputed = {
                "agreement": sum(int(row["prediction_match"]) for row in selected) / 24,
                "mismatch_count": sum(
                    not int(row["prediction_match"]) for row in selected
                ),
                "harmful_count": sum(int(row["harmful"]) for row in selected),
                "candidate_kl_mean": float(kl.mean()),
                "candidate_kl_p95": float(np.quantile(kl, 0.95)),
                "token_retention": float(selected[0]["token_retention"]),
            }
            for key, value in recomputed.items():
                if not np.isclose(
                    float(summary[mode][geometry][key]), float(value), atol=1e-12
                ):
                    raise ValueError(
                        f"summary mismatch for {mode}/{geometry}/{key}: "
                        f"{summary[mode][geometry][key]} != {value}"
                    )
            aggregate_rows.append({"mode": mode, "geometry": geometry, **recomputed})

        flat_rows = by_geometry[GEOMETRIES[0]]
        spatial_rows = by_geometry[GEOMETRIES[1]]
        ordered_ids = sorted(
            sample_ids,
            key=lambda sample_id: int(flat_rows[sample_id]["sample_position"]),
        )
        flat_kl = np.asarray(
            [float(flat_rows[sample_id]["candidate_kl"]) for sample_id in ordered_ids]
        )
        spatial_kl = np.asarray(
            [
                float(spatial_rows[sample_id]["candidate_kl"])
                for sample_id in ordered_ids
            ]
        )
        for sample_id, flat_value, spatial_value in zip(
            ordered_ids, flat_kl, spatial_kl
        ):
            flat_row = flat_rows[sample_id]
            spatial_row = spatial_rows[sample_id]
            paired_rows.append(
                {
                    "mode": mode,
                    "sample_id": sample_id,
                    "sample_position": int(flat_row["sample_position"]),
                    "flat_candidate_kl": flat_value,
                    "spatial_candidate_kl": spatial_value,
                    "spatial_minus_flat_kl": spatial_value - flat_value,
                    "spatial_kl_win": int(spatial_value < flat_value),
                    "flat_prediction_match": int(flat_row["prediction_match"]),
                    "spatial_prediction_match": int(spatial_row["prediction_match"]),
                    "flat_harmful": int(flat_row["harmful"]),
                    "spatial_harmful": int(spatial_row["harmful"]),
                }
            )

        statistics = bootstrap_paired(flat_kl, spatial_kl, seed=20260830 + mode_index)
        wins = int(np.sum(spatial_kl < flat_kl))
        losses = int(np.sum(spatial_kl > flat_kl))
        ties = int(flat_kl.size - wins - losses)
        comparison = comparisons[mode]
        for key in ("mean_kl_ratio",):
            if not np.isclose(
                float(comparison[key]), float(statistics[key]), atol=1e-12
            ):
                raise ValueError(f"comparison mismatch for {mode}/{key}")
        aggregate_rows.append(
            {
                "mode": mode,
                "geometry": "paired_spatial_vs_flat",
                **statistics,
                "paired_kl_wins": wins,
                "paired_kl_losses": losses,
                "paired_kl_ties": ties,
                "one_sided_sign_pvalue": one_sided_sign_pvalue(wins, wins + losses),
                "prediction_match_wins": int(comparison["prediction_match_wins"]),
                "prediction_match_losses": int(comparison["prediction_match_losses"]),
                "mismatch_reduction": int(comparison["mismatch_reduction"]),
                "harmful_delta": int(comparison["harmful_delta"]),
            }
        )
    return aggregate_rows, paired_rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.analysis_dir / "true_2x2_geometry_exposed_v1"
    recorded = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(result_dir / "true_2x2_geometry_rows.csv")
    aggregate_rows, paired_rows = validate_and_summarize(recorded, rows)
    write_csv(args.output_dir / "true_2x2_geometry_summary.csv", aggregate_rows)
    write_csv(args.output_dir / "true_2x2_geometry_paired.csv", paired_rows)

    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    panel_a, panel_b, panel_c, panel_d = axes.flatten()
    colors = {"positioned_equal_mass": "#0072B2", "positioned_group_mass": "#E69F00"}
    mode_labels = {
        "positioned_equal_mass": "Equal mass",
        "positioned_group_mass": "Group mass",
    }
    geometry_labels = {
        "flat_contiguous_4": "Flat contiguous-4",
        "spatial_2x2": "Spatial 2x2",
    }

    x = np.arange(len(MODES), dtype=np.float64)
    width = 0.34
    for geometry_index, geometry in enumerate(GEOMETRIES):
        selected = [
            next(
                row
                for row in aggregate_rows
                if row["mode"] == mode and row["geometry"] == geometry
            )
            for mode in MODES
        ]
        offset = (geometry_index - 0.5) * width
        bars = panel_a.bar(
            x + offset,
            [float(row["candidate_kl_mean"]) for row in selected],
            width=width,
            color=("#9CA3AF" if geometry_index == 0 else "#009E73"),
            alpha=0.9,
            label=geometry_labels[geometry],
        )
        panel_a.scatter(
            x + offset,
            [float(row["candidate_kl_p95"]) for row in selected],
            marker="D",
            s=24,
            color="#111827",
            zorder=3,
        )
        for bar in bars:
            panel_a.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
            )
    panel_a.set_xticks(x, labels=[mode_labels[mode] for mode in MODES])
    panel_a.set_ylabel("Candidate KL (bar mean, diamond P95)")
    panel_a.set_yscale("log")
    panel_a.legend(frameon=False, loc="upper left")
    panel_a.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)

    for mode in MODES:
        selected = [row for row in paired_rows if row["mode"] == mode]
        flat = np.asarray([float(row["flat_candidate_kl"]) for row in selected])
        spatial = np.asarray([float(row["spatial_candidate_kl"]) for row in selected])
        panel_b.scatter(
            flat,
            spatial,
            s=28,
            color=colors[mode],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
            label=mode_labels[mode],
        )
    limits = (1e-5, 0.55)
    panel_b.plot(limits, limits, color="#222222", linestyle=":", linewidth=1)
    panel_b.set_xscale("log")
    panel_b.set_yscale("log")
    panel_b.set_xlim(limits)
    panel_b.set_ylim(limits)
    panel_b.set_xlabel("Flat contiguous-4 KL")
    panel_b.set_ylabel("Spatial 2x2 KL")
    panel_b.legend(frameon=False, loc="upper left")
    panel_b.grid(color="#D1D5DB", linewidth=0.5, alpha=0.55)

    for mode_index, mode in enumerate(MODES):
        selected = sorted(
            (row for row in paired_rows if row["mode"] == mode),
            key=lambda row: float(row["spatial_minus_flat_kl"]),
        )
        values = np.asarray([float(row["spatial_minus_flat_kl"]) for row in selected])
        panel_c.plot(
            np.arange(1, values.size + 1),
            values,
            color=colors[mode],
            marker=("o" if mode_index == 0 else "s"),
            markersize=3.2,
            linewidth=1.2,
            label=mode_labels[mode],
        )
    panel_c.axhline(0, color="#222222", linestyle=":", linewidth=1)
    panel_c.set_xlabel("Exposed samples sorted within each mode")
    panel_c.set_ylabel("Spatial 2x2 KL - flat KL")
    panel_c.legend(frameon=False, loc="upper left")
    panel_c.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)

    paired_statistics = [
        next(
            row
            for row in aggregate_rows
            if row["mode"] == mode and row["geometry"] == "paired_spatial_vs_flat"
        )
        for mode in MODES
    ]
    ratios = np.asarray([float(row["mean_kl_ratio"]) for row in paired_statistics])
    lower = ratios - np.asarray(
        [float(row["mean_kl_ratio_ci_low"]) for row in paired_statistics]
    )
    upper = (
        np.asarray([float(row["mean_kl_ratio_ci_high"]) for row in paired_statistics])
        - ratios
    )
    panel_d.errorbar(
        x,
        ratios,
        yerr=np.vstack((lower, upper)),
        fmt="o",
        color="#111827",
        markerfacecolor="#56B4E9",
        markersize=7,
        linewidth=1.4,
        capsize=4,
    )
    panel_d.axhline(
        0.8, color="#009E73", linestyle="--", linewidth=1, label="Strict ratio gate"
    )
    panel_d.axhline(1.0, color="#222222", linestyle=":", linewidth=1, label="No change")
    panel_d.axhline(
        1.05, color="#D55E00", linestyle="-.", linewidth=1, label="Decision-only guard"
    )
    panel_d.set_xticks(x, labels=[mode_labels[mode] for mode in MODES])
    panel_d.set_ylabel("Mean KL ratio: spatial 2x2 / flat (95% bootstrap CI)")
    panel_d.set_ylim(0.35, 2.3)
    panel_d.legend(frameon=False, loc="upper left")
    panel_d.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)

    for label, axis in zip(("a", "b", "c", "d"), axes.flatten()):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    output_stem = args.output_dir / "true_2x2_geometry_control"
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)
    svg_path = output_stem.with_suffix(".svg")
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": recorded["decision"],
                "paired_statistics": paired_statistics,
                "row_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
