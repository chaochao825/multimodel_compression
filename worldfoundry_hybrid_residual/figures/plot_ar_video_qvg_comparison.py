"""Plot the frozen LongLive QuantVideoGen comparison and calibration attribution."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


COLORS = {
    "ink": "#17222B",
    "blue": "#1B6CA8",
    "teal": "#138A72",
    "amber": "#D9822B",
    "red": "#C84A3A",
    "slate": "#71808C",
    "paper": "#F7F3EA",
    "grid": "#D8D1C4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qvg-result", type=Path, default=Path("results/qvg_all_full_20260805a")
    )
    parser.add_argument(
        "--temporal-result", type=Path, default=Path("results/20260805_full_v1")
    )
    parser.add_argument(
        "--int2-attribution",
        type=Path,
        default=Path("results/qvg_attribution_calibration_20260805a"),
    )
    parser.add_argument(
        "--int2-asym-attribution",
        type=Path,
        default=Path("results/qvg_attribution_int2_asym_calibration_20260805a"),
    )
    parser.add_argument(
        "--int4-attribution",
        type=Path,
        default=Path("results/qvg_attribution_int4_calibration_20260805a"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures/qvg_ar_video_20260805")
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def qvg_pareto(result: Path) -> list[dict]:
    labels = {
        "rtn_int2_b64_all": "RTN INT2",
        "qvg_int2_s1_b64_all": "QVG INT2",
        "qvg_int2_s1_b64_sink_recent": "QVG INT2 + exact 6F",
        "qvg_int4_s1_b64_all": "QVG INT4",
        "qvg_pro_int2_s4_b16_all": "QVG-Pro INT2",
    }
    rows = []
    for item in load_json(result / "summary.json")["summaries"]:
        if item["scope"] != "held_out":
            continue
        rows.append(
            {
                "method": item["method"],
                "label": labels[item["method"]],
                "cache_compression": item["mean_cache_compression_ratio"],
                "aggregate_av_error": item["aggregate_relative_av_l2"],
                "worst_head_av_error": item["worst_head_relative_av_l2"],
            }
        )
    return sorted(rows, key=lambda item: item["cache_compression"])


def temporal_pareto(result: Path) -> list[dict]:
    rows = []
    for item in load_json(result / "summary.json")["summaries"]:
        if (
            item["scope"] == "held_out"
            and item["correction"] == "adaptive_rank_oracle"
            and int(item["rank"]) == 16
            and item["method"].startswith("phasealigned_recency_")
        ):
            rows.append(
                {
                    "method": item["method"],
                    "arithmetic_reduction": item["mean_arithmetic_reduction"],
                    "aggregate_av_error": item["aggregate_relative_av_l2"],
                    "worst_head_av_error": item["worst_head_relative_av_l2"],
                }
            )
    return sorted(rows, key=lambda item: item["arithmetic_reduction"])


def attribution_rows(paths: list[tuple[str, Path]]) -> list[dict]:
    rows = []
    for method, path in paths:
        for item in load_json(path / "summary.json")["variant_summaries"]:
            if item["variant"] in {"qvg_k_only", "qvg_v_only", "qvg_both"}:
                rows.append(
                    {
                        "residual_format": method,
                        "variant": item["variant"].removeprefix("qvg_"),
                        "aggregate_av_error": item["aggregate_relative_av_l2"],
                        "worst_head_av_error": item["worst_head_relative_av_l2"],
                    }
                )
    return rows


def frame_region_rows(path: Path) -> list[dict]:
    grouped: dict[tuple[int, str], dict[str, float]] = defaultdict(
        lambda: {"attention_mass": 0.0, "value_leverage": 0.0, "head_frames": 0.0}
    )
    with (path / "frame_statistics.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            offset = int(row["frame_offset"])
            region = "sink" if offset < 3 else ("recent" if offset >= 9 else "middle")
            entry = grouped[(int(row["layer"]), region)]
            entry["attention_mass"] += float(row["attention_mass"])
            entry["value_leverage"] += float(row["value_leverage_fraction"])
            entry["head_frames"] += 1
    region_frames = {"sink": 3, "middle": 6, "recent": 3}
    rows = []
    for (layer, region), item in sorted(grouped.items()):
        heads = item["head_frames"] / region_frames[region]
        rows.append(
            {
                "layer": layer,
                "region": region,
                "attention_mass": item["attention_mass"] / heads,
                "value_leverage": item["value_leverage"] / heads,
            }
        )
    return rows


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["paper"])
    axis.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(COLORS["slate"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qvg = qvg_pareto(args.qvg_result)
    temporal = temporal_pareto(args.temporal_result)
    attribution = attribution_rows(
        [
            ("INT2 symmetric", args.int2_attribution),
            ("INT2 asymmetric", args.int2_asym_attribution),
            ("INT4 symmetric", args.int4_attribution),
        ]
    )
    regions = frame_region_rows(args.int2_attribution)
    write_csv(args.output_dir / "qvg_heldout_pareto.csv", qvg)
    write_csv(args.output_dir / "temporal_summary_heldout_pareto.csv", temporal)
    write_csv(args.output_dir / "qvg_calibration_kv_attribution.csv", attribution)
    write_csv(args.output_dir / "qvg_calibration_frame_regions.csv", regions)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    figure.patch.set_facecolor("#FCFAF5")

    axis = axes[0, 0]
    style_axis(axis)
    for index, row in enumerate(qvg):
        color = [COLORS["amber"], COLORS["teal"], COLORS["red"], COLORS["blue"], COLORS["slate"]][index]
        axis.scatter(
            row["cache_compression"],
            row["aggregate_av_error"],
            s=90,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        axis.annotate(
            row["label"],
            (row["cache_compression"], row["aggregate_av_error"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
        )
    axis.axhline(0.01, color=COLORS["red"], linestyle="--", linewidth=1.2, label="1% aggregate gate")
    axis.set_yscale("log")
    axis.set_xlabel("Logical KV-cache compression (x)")
    axis.set_ylabel("Held-out aggregate AV relative L2")
    axis.set_title("A. QuantVideoGen: memory Pareto, no speed claim")
    axis.legend(loc="lower right", frameon=False, fontsize=8)

    axis = axes[0, 1]
    style_axis(axis)
    axis.scatter(
        [row["arithmetic_reduction"] for row in temporal],
        [row["aggregate_av_error"] for row in temporal],
        color=COLORS["slate"],
        alpha=0.55,
        s=32,
        label="rank-16 adaptive oracle",
    )
    highlights = {
        "phasealigned_recency_g1_event_0p05": (COLORS["amber"], "Primary: 1 group"),
        "phasealigned_recency_g4_event_0p1": (COLORS["teal"], "Best error: 4 groups"),
    }
    for row in temporal:
        if row["method"] not in highlights:
            continue
        color, label = highlights[row["method"]]
        axis.scatter(row["arithmetic_reduction"], row["aggregate_av_error"], s=95, color=color, edgecolor="white", linewidth=1.2, label=label)
    axis.axhline(0.01, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axis.set_xlabel("Ideal attention arithmetic reduction (x)")
    axis.set_ylabel("Held-out aggregate AV relative L2")
    axis.set_title("B. Prior temporal summary: quality costs compute gain")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    style_axis(axis)
    formats = ["INT2 symmetric", "INT2 asymmetric", "INT4 symmetric"]
    variants = ["k_only", "v_only", "both"]
    variant_labels = {"k_only": "K only", "v_only": "V only", "both": "K + V"}
    variant_colors = {"k_only": COLORS["blue"], "v_only": COLORS["amber"], "both": COLORS["red"]}
    width = 0.23
    for offset, variant in enumerate(variants):
        values = [
            next(
                row["aggregate_av_error"]
                for row in attribution
                if row["residual_format"] == method and row["variant"] == variant
            )
            for method in formats
        ]
        x = [index + (offset - 1) * width for index in range(len(formats))]
        axis.bar(x, values, width=width, color=variant_colors[variant], label=variant_labels[variant])
    axis.axhline(0.01, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axis.set_xticks(range(len(formats)), formats)
    axis.set_ylabel("Calibration aggregate AV relative L2")
    axis.set_title("C. Error attribution: V residual is dominant")
    axis.legend(frameon=False, ncol=3, fontsize=8)

    axis = axes[1, 1]
    style_axis(axis)
    region_order = ["sink", "middle", "recent"]
    region_colors = {"sink": COLORS["blue"], "middle": COLORS["slate"], "recent": COLORS["teal"]}
    layers = [0, 14, 29]
    positions = []
    labels = []
    for layer_index, layer in enumerate(layers):
        for metric_index, metric in enumerate(["attention_mass", "value_leverage"]):
            position = layer_index * 2.7 + metric_index * 0.9
            positions.append(position)
            labels.append("Mass" if metric_index == 0 else "V-leverage")
            bottom = 0.0
            for region in region_order:
                value = next(
                    row[metric]
                    for row in regions
                    if row["layer"] == layer and row["region"] == region
                )
                axis.bar(position, value, bottom=bottom, width=0.72, color=region_colors[region], label=region if layer_index == 0 and metric_index == 0 else None)
                bottom += value
    axis.set_xticks(positions, labels)
    for layer_index, layer in enumerate(layers):
        axis.text(layer_index * 2.7 + 0.45, -0.17, f"Layer {layer}", ha="center", transform=axis.get_xaxis_transform(), fontweight="bold")
    axis.set_ylim(0, 1.02)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Fraction of total")
    axis.set_title("D. Static sink/recent misses layer-specific history")
    axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")

    figure.suptitle(
        "LongLive-1.3B causal KV compression: what QuantVideoGen changes, and what still fails",
        fontsize=15,
        fontweight="bold",
    )
    output = args.output_dir / "qvg_ar_video_comparison_20260805"
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
