"""Bind the three Butterfly-lifting gates into publication-ready figures."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "ar_video_butterfly_lifting_20260805"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(result: str) -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(ROOT / "results" / result / "summary.csv")
        if row["scope"] == "held_out" and row["correction"] == "adaptive_rank16"
    ]


def method_metadata(name: str) -> tuple[str, str]:
    labels = {
        "canonical_identity_d0p10": ("Identity lifting 10%", "identity"),
        "canonical_identity_d0p19": ("Identity lifting 19%", "identity"),
        "canonical_identity_d0p20": ("Identity lifting 20%", "identity"),
        "canonical_shared_shift_d0p05": ("Global shift 5%", "global"),
        "canonical_shared_shift_d0p10": ("Global shift 10%", "global"),
        "canonical_shared_shift_d0p20": ("Global shift 20%", "global"),
        "canonical_per_head_shift_d0p20": ("Per-head shift 20%", "global"),
        "postrope_shared_shift_d0p20": ("Post-RoPE shift 20%", "ablation"),
        "canonical_global_shared_d0p19": ("Global shift 19%", "global"),
        "canonical_window_fixed_d0p10": ("Fixed windows 10%", "window"),
        "canonical_window_fixed_d0p19": ("Fixed windows 19%", "window"),
        "canonical_window_staggered_d0p10": ("Staggered windows 10%", "window"),
        "canonical_window_staggered_d0p19": ("Staggered windows 19%", "window"),
        "kv_detail_energy": ("K/V-energy details 19%", "energy"),
        "adaptive_tail_singleton_oracle": ("Singleton AV selector 19%", "oracle"),
    }
    return labels[name]


def build_frontier() -> list[dict[str, object]]:
    rows = []
    seen = set()
    for result in (
        "butterfly_lifting_primary_full_20260805a",
        "windowed_lifting_primary_full_20260805a",
        "lifting_detail_oracle_primary_full_20260805a",
    ):
        for row in summary_rows(result):
            name = row["method"]
            if name in seen:
                continue
            seen.add(name)
            label, family = method_metadata(name)
            rows.append(
                {
                    "method": name,
                    "label": label,
                    "family": family,
                    "cache_compression_ratio": float(
                        row["minimum_cache_compression_ratio"]
                    ),
                    "aggregate_av_l2_percent": 100
                    * float(row["aggregate_relative_av_l2"]),
                    "worst_head_av_l2_percent": 100
                    * float(row["worst_head_relative_av_l2"]),
                }
            )
    return rows


def aggregate_group(group: list[dict[str, str]]) -> tuple[float, float]:
    aggregate = math.sqrt(
        sum(float(row["numerator_sq"]) for row in group)
        / sum(float(row["denominator_sq"]) for row in group)
    )
    worst = max(float(row["relative_av_l2"]) for row in group)
    return 100 * aggregate, 100 * worst


def build_layer_rows() -> list[dict[str, object]]:
    metrics = [
        row
        for row in read_csv(
            ROOT
            / "results"
            / "lifting_detail_oracle_primary_full_20260805a"
            / "metrics.csv"
        )
        if row["split"] != "calibration" and row["correction"] == "adaptive_rank16"
    ]
    groups: defaultdict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        groups[(row["method"], int(row["layer"]))].append(row)
    output = []
    for (method, layer), group in sorted(groups.items()):
        aggregate, worst = aggregate_group(group)
        label, _ = method_metadata(method)
        output.append(
            {
                "method": method,
                "label": label,
                "layer": layer,
                "aggregate_av_l2_percent": aggregate,
                "worst_head_av_l2_percent": worst,
            }
        )
    return output


def build_capture_rows() -> list[dict[str, object]]:
    result = ROOT / "results" / "lifting_detail_oracle_primary_full_20260805a"
    metrics = [
        row
        for row in read_csv(result / "metrics.csv")
        if row["correction"] == "adaptive_rank16"
    ]
    groups: defaultdict[tuple[str, int, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        groups[
            (
                row["prompt_id"],
                int(row["layer"]),
                int(row["current_start_frame"]),
                int(row["denoising_call_index"]),
                row["method"],
            )
        ].append(row)
    debug = json.loads((result / "debug.json").read_text(encoding="utf-8"))
    overlap = {tuple(item["capture"]): float(item["energy_oracle_jaccard"]) for item in debug}
    output = []
    identities = sorted({key[:4] for key in groups})
    for identity in identities:
        energy, _ = aggregate_group(groups[identity + ("kv_detail_energy",)])
        oracle, _ = aggregate_group(
            groups[identity + ("adaptive_tail_singleton_oracle",)]
        )
        output.append(
            {
                "prompt_id": identity[0],
                "layer": identity[1],
                "current_start_frame": identity[2],
                "denoising_call_index": identity[3],
                "energy_oracle_jaccard": overlap[identity],
                "energy_av_l2_percent": energy,
                "oracle_av_l2_percent": oracle,
                "relative_improvement_percent": 100 * (1 - oracle / energy),
            }
        )
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frontier = build_frontier()
    layers = build_layer_rows()
    captures = build_capture_rows()
    write_csv(OUTPUT / "quality_storage_frontier.csv", frontier)
    write_csv(OUTPUT / "layer_comparison.csv", layers)
    write_csv(OUTPUT / "selector_overlap.csv", captures)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "identity": "#7A7A7A",
        "global": "#2F6B8A",
        "ablation": "#AA7C39",
        "window": "#D1495B",
        "energy": "#4C956C",
        "oracle": "#111111",
    }
    markers = {
        "identity": "o",
        "global": "s",
        "ablation": "X",
        "window": "D",
        "energy": "^",
        "oracle": "*",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))

    axis = axes[0, 0]
    for row in frontier:
        axis.scatter(
            row["cache_compression_ratio"],
            row["aggregate_av_l2_percent"],
            color=colors[row["family"]],
            marker=markers[row["family"]],
            s=65 if row["family"] != "oracle" else 110,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    for row in frontier:
        if row["method"] in {
            "canonical_identity_d0p10",
            "canonical_shared_shift_d0p20",
            "postrope_shared_shift_d0p20",
            "adaptive_tail_singleton_oracle",
        }:
            axis.annotate(
                row["label"],
                (row["cache_compression_ratio"], row["aggregate_av_l2_percent"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
    axis.axvline(1.5, color="#777777", linestyle="--", linewidth=1)
    axis.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Logical BF16 cache compression estimate (x)")
    axis.set_ylabel("Adaptive rank-16 aggregate AV error (%)")
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.text(-0.12, 1.03, "A", transform=axis.transAxes, fontweight="bold", fontsize=12)

    selected_names = [
        "canonical_global_shared_d0p19",
        "canonical_window_fixed_d0p19",
        "canonical_window_staggered_d0p19",
        "adaptive_tail_singleton_oracle",
    ]
    selected = [{row["method"]: row for row in frontier}[name] for name in selected_names]
    axis = axes[0, 1]
    positions = list(range(len(selected)))
    width = 0.36
    axis.bar(
        [position - width / 2 for position in positions],
        [row["aggregate_av_l2_percent"] for row in selected],
        width,
        color="#2F6B8A",
        label="Aggregate",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [row["worst_head_av_l2_percent"] for row in selected],
        width,
        color="#D1495B",
        label="Worst head",
    )
    axis.axhline(0.5, color="#2F6B8A", linestyle=":", linewidth=1)
    axis.axhline(1.0, color="#D1495B", linestyle=":", linewidth=1)
    axis.set_yscale("log")
    axis.set_xticks(
        positions,
        ["Global", "Fixed\nwindow", "Staggered\nwindow", "Singleton\nselector"],
    )
    axis.set_ylabel("AV error (%)")
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.text(-0.12, 1.03, "B", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[1, 0]
    layer_ids = sorted({int(row["layer"]) for row in layers})
    for method, offset, color, label in (
        ("kv_detail_energy", -0.18, "#4C956C", "K/V energy"),
        ("adaptive_tail_singleton_oracle", 0.18, "#111111", "Singleton AV selector"),
    ):
        values = [
            next(
                row["aggregate_av_l2_percent"]
                for row in layers
                if row["method"] == method and row["layer"] == layer
            )
            for layer in layer_ids
        ]
        axis.bar(
            [index + offset for index in range(len(layer_ids))],
            values,
            0.36,
            color=color,
            label=label,
        )
    axis.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    axis.set_xticks(range(len(layer_ids)), [f"Layer {layer}" for layer in layer_ids])
    axis.set_ylabel("Adaptive rank-16 aggregate AV error (%)")
    axis.set_yscale("log")
    axis.legend(frameon=False)
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.text(-0.12, 1.03, "C", transform=axis.transAxes, fontweight="bold", fontsize=12)

    axis = axes[1, 1]
    for layer, color, marker in ((14, "#D1495B", "o"), (29, "#2F6B8A", "s")):
        selected_capture = [row for row in captures if row["layer"] == layer]
        axis.scatter(
            [row["energy_oracle_jaccard"] for row in selected_capture],
            [row["relative_improvement_percent"] for row in selected_capture],
            color=color,
            marker=marker,
            s=60,
            label=f"Layer {layer}",
        )
    axis.axhline(20, color="#777777", linestyle=":", linewidth=1)
    axis.set_xlabel("Energy/singleton selected-block Jaccard")
    axis.set_ylabel("Oracle improvement over energy (%)")
    axis.set_xlim(0, 1)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    axis.text(-0.12, 1.03, "D", transform=axis.transAxes, fontweight="bold", fontsize=12)

    figure.tight_layout()
    figure.savefig(OUTPUT / "ar_video_butterfly_lifting_20260805.png", dpi=300)
    figure.savefig(OUTPUT / "ar_video_butterfly_lifting_20260805.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
