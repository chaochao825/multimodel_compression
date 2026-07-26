#!/usr/bin/env python3
"""Assemble the strict World Foundry phase-2 evidence into one dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path)
    parser.add_argument("--ffn-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pending(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.5,
        0.5,
        f"{label}\npending",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=13,
        color="#667085",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)


def finite(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def format_number(value: object, digits: int = 3) -> str:
    number = finite(value)
    return "pending" if not math.isfinite(number) else f"{number:.{digits}f}"


def main() -> None:
    args = parse_args()
    strict_dir = args.strict_dir.resolve()
    geometry_dir = args.geometry_dir.resolve() if args.geometry_dir else None
    ffn_dir = args.ffn_dir.resolve() if args.ffn_dir else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dir = strict_dir / "cfg_f81_matrix"
    cache_dir = strict_dir / "crossattn_cache_f17"
    cfg_rows = read_csv_if_exists(cfg_dir / "cfg_parallel_paired_metrics.csv")
    cache_rows = read_csv_if_exists(cache_dir / "crossattn_cache_paired_metrics.csv")
    geometry_rows = (
        read_csv_if_exists(geometry_dir / "geometry_generalization_cells.csv")
        if geometry_dir
        else []
    )
    ffn_rows = (
        read_csv_if_exists(ffn_dir / "wan_ffn_exact_summary.csv") if ffn_dir else []
    )
    cfg_summary = read_json_if_exists(cfg_dir / "cfg_parallel_summary.json")
    cache_summary = read_json_if_exists(cache_dir / "crossattn_cache_summary.json")
    geometry_summary = (
        read_json_if_exists(geometry_dir / "geometry_generalization_summary.json")
        if geometry_dir
        else None
    )
    ffn_summary = (
        read_json_if_exists(ffn_dir / "wan_ffn_exact_summary.json") if ffn_dir else None
    )

    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.4), constrained_layout=True)
    figure.patch.set_facecolor("#F7F4ED")
    for axis in axes.flat:
        axis.set_facecolor("#FFFCF5")
    points: list[dict[str, object]] = []

    axis = axes[0, 0]
    if cfg_rows:
        values = [finite(row["speedup"]) for row in cfg_rows]
        axis.scatter(range(len(values)), values, color="#0E7490", s=36, alpha=0.8)
        axis.axhline(1.0, color="#B42318", linestyle="--", linewidth=1.2)
        axis.set_xlabel("Paired prompt/seed/repeat")
        axis.set_ylabel("Sequential / CFG-parallel time")
        axis.set_title(
            "F81 exact two-H200 CFG\n"
            f"mean={format_number((cfg_summary or {}).get('speedup_mean'))}x, "
            f"95% CI=[{format_number((cfg_summary or {}).get('speedup_bootstrap_95ci_low'))}, "
            f"{format_number((cfg_summary or {}).get('speedup_bootstrap_95ci_high'))}]"
        )
        for index, (row, value) in enumerate(zip(cfg_rows, values)):
            points.append(
                {
                    "panel": "cfg_parallel",
                    "series": row.get("source_run", "cfg"),
                    "sample": index,
                    "x": index,
                    "y": value,
                    "prompt_index": row.get("prompt_index", ""),
                    "seed": row.get("seed", ""),
                    "repeat": row.get("repeat", ""),
                }
            )
    else:
        pending(axis, "F81 exact CFG")

    axis = axes[0, 1]
    if cache_rows:
        values = [finite(row["speedup"]) for row in cache_rows]
        axis.scatter(range(len(values)), values, color="#B54708", s=42, alpha=0.85)
        axis.axhline(1.0, color="#B42318", linestyle="--", linewidth=1.2)
        axis.set_xlabel("Paired prompt/repeat")
        axis.set_ylabel("Baseline / exact-cache time")
        axis.set_title(
            "F17 exact text K/V cache\n"
            f"mean={format_number((cache_summary or {}).get('speedup_mean'))}x, "
            f"memory delta={format_number((cache_summary or {}).get('peak_memory_delta_mib_mean'), 1)} MiB"
        )
        for index, (row, value) in enumerate(zip(cache_rows, values)):
            points.append(
                {
                    "panel": "crossattn_cache",
                    "series": "exact_kv_cache",
                    "sample": index,
                    "x": index,
                    "y": value,
                    "prompt_index": row.get("prompt_index", ""),
                    "seed": row.get("seed", ""),
                    "repeat": row.get("repeat", ""),
                }
            )
    else:
        pending(axis, "F17 exact K/V cache")

    axis = axes[1, 0]
    if geometry_rows:
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in geometry_rows:
            groups[(row["split"], row["mask"])].append(row)
        masks = sorted({mask for _, mask in groups})
        palette = plt.get_cmap("tab10")
        colors = {mask: palette(index % 10) for index, mask in enumerate(masks)}
        markers = {"validation": "o", "test": "^"}
        for (split, mask), rows in sorted(groups.items()):
            xs = [finite(row["effective_execution_density"]) for row in rows]
            ys = [finite(row["output_relative_l2_energy_proxy"]) for row in rows]
            axis.scatter(
                xs,
                ys,
                color=colors[mask],
                marker=markers.get(split, "s"),
                s=28,
                alpha=0.72,
                label=f"{split}: {mask}",
            )
            for index, (x_value, y_value) in enumerate(zip(xs, ys)):
                points.append(
                    {
                        "panel": "geometry_generalization",
                        "series": f"{split}:{mask}",
                        "sample": index,
                        "x": x_value,
                        "y": y_value,
                        "layer": rows[index].get("layer", ""),
                        "sampling_step": rows[index].get("sampling_step", ""),
                        "branch": rows[index].get("branch", ""),
                    }
                )
        axis.axhline(0.02, color="#B42318", linestyle="--", linewidth=1.2)
        axis.axvline(0.125, color="#B54708", linestyle="--", linewidth=1.2)
        axis.set_xlabel("Effective 64x64 execution density")
        axis.set_ylabel("Pre-o relative L2 energy proxy")
        axis.set_title(
            "F81 calibration-frozen geometry policy\n"
            f"independent test gate={str((geometry_summary or {}).get('independent_test_gate_passed', False)).lower()}"
        )
        axis.legend(fontsize=6.5, ncol=2)
    else:
        pending(axis, "F81 geometry generalization")

    axis = axes[1, 1]
    candidate_rows = [
        row
        for row in ffn_rows
        if row.get("path") != "eager" and row.get("status") == "ok"
    ]
    if candidate_rows:
        paths = sorted({row["path"] for row in candidate_rows})
        cases = sorted({row["case"] for row in candidate_rows})
        width = 0.34
        for case_index, case in enumerate(cases):
            medians = []
            for path_index, path in enumerate(paths):
                selected = [
                    finite(row["median_speedup"])
                    for row in candidate_rows
                    if row["case"] == case and row["path"] == path
                ]
                median = sorted(selected)[len(selected) // 2] if selected else math.nan
                medians.append(median)
                for layer_index, value in enumerate(selected):
                    points.append(
                        {
                            "panel": "ffn_exact",
                            "series": f"{case}:{path}",
                            "sample": layer_index,
                            "x": path_index,
                            "y": value,
                        }
                    )
            offsets = [index + (case_index - (len(cases) - 1) / 2) * width for index in range(len(paths))]
            axis.bar(offsets, medians, width=width, label=case, alpha=0.82)
        axis.axhline(1.0, color="#B42318", linestyle="--", linewidth=1.2)
        axis.axhline(1.10, color="#027A48", linestyle=":", linewidth=1.2)
        axis.set_xticks(range(len(paths)), paths, rotation=22, ha="right", fontsize=8)
        axis.set_ylabel("Median speedup vs eager full FFN")
        axis.set_title("Complete Wan FFN exact paths")
        axis.legend(fontsize=8)
    else:
        pending(axis, "F17/F81 full FFN exact paths")

    for axis in axes.flat:
        if axis.axison:
            axis.grid(alpha=0.18)
    figure.suptitle(
        "World Foundry Phase-2: exact systems, geometry sparsity, and kernel gates",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(args.output_dir / "phase2_decision_dashboard.png", dpi=190)
    figure.savefig(args.output_dir / "phase2_decision_dashboard.pdf")
    plt.close(figure)
    write_csv(args.output_dir / "phase2_dashboard_points.csv", points)

    pending_stages = [
        name
        for name, payload in (
            ("cfg_parallel", cfg_summary),
            ("crossattn_cache", cache_summary),
            ("geometry_generalization", geometry_summary),
            ("ffn_exact", ffn_summary),
        )
        if payload is None
    ]
    decision = {
        "pending_stages": pending_stages,
        "cfg_parallel": cfg_summary,
        "crossattn_cache": cache_summary,
        "geometry_generalization": geometry_summary,
        "ffn_exact": ffn_summary,
        "evidence_boundary": (
            "Geometry values are sampled pre-output-projection proxies; only an "
            "end-to-end fused-kernel generation can establish video speed/quality."
        ),
    }
    (args.output_dir / "phase2_decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cfg_exact = (cfg_summary or {}).get("exact_gate_passed", "pending")
    cache_exact = (cache_summary or {}).get("exact_gate_passed", "pending")
    geometry_gate = (geometry_summary or {}).get(
        "independent_test_gate_passed", "pending"
    )
    ffn_go = [
        row.get("path")
        for row in (ffn_summary or {}).get("paths", [])
        if row.get("decision") == "GO"
    ]
    report = f"""# World Foundry Phase-2 结果摘要

## 状态

- Pending stages: `{', '.join(pending_stages) if pending_stages else 'none'}`
- F81 exact CFG gate: `{cfg_exact}`; mean speedup: `{format_number((cfg_summary or {}).get('speedup_mean'))}x`.
- F17 exact K/V cache gate: `{cache_exact}`; mean speedup: `{format_number((cache_summary or {}).get('speedup_mean'))}x`.
- Geometry independent-test proxy gate: `{geometry_gate}`.
- Complete FFN exact-path GO candidates: `{', '.join(str(item) for item in ffn_go) if ffn_go else 'none or pending'}`.

## 证据边界

Geometry 结果是 sampled pre-output-projection energy proxy，不等于端到端视频质量或 fused-kernel 加速。只有通过 held-out policy、真实 H200 kernel 和多 prompt/seed 视频指标三重门槛后，才能形成最终加速结论。
"""
    (args.output_dir / "phase2_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"pending_stages": pending_stages}, indent=2), flush=True)


if __name__ == "__main__":
    main()
