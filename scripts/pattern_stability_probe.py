#!/usr/bin/env python3
"""Probe stability and mechanism-relevant attention pattern metrics.

This script is intentionally diagnostic. It does not run any model forward; it
aggregates existing attention-space probes and representative exported matrices
to quantify which patterns are stable across sampled maps and which are
input/layer/head dependent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "remote_logs"
FIG_DIR = ROOT / "figures"


METHODS = [
    "grid_cyclic_bccb",
    "flat_block_circulant",
    "permuted_flat_block_circulant",
    "monarch_like_mask_proxy",
]


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


def load_json(name: str) -> dict:
    return json.loads((LOG_DIR / name).read_text(encoding="utf-8"))


def best_rows_by_map(rows: Iterable[dict], method: str, metric: str = "relative_fro_error") -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for row in rows:
        if row.get("matrix_kind") != "attention" or row.get("method") != method:
            continue
        map_id = str(row["map_id"])
        old = best.get(map_id)
        if old is None or float(row[metric]) < float(old[metric]):
            best[map_id] = row
    return best


def summarize_method_rows(rows: List[dict], family_label: str) -> List[dict]:
    out: List[dict] = []
    per_method = {method: best_rows_by_map(rows, method) for method in METHODS}
    map_ids = sorted(set().union(*(set(v) for v in per_method.values())))
    winners: Dict[str, str] = {}
    for map_id in map_ids:
        candidates = [
            (method, float(per_method[method][map_id]["relative_fro_error"]))
            for method in METHODS
            if map_id in per_method[method]
        ]
        if candidates:
            winners[map_id] = min(candidates, key=lambda item: item[1])[0]

    for method, by_map in per_method.items():
        vals = [float(row["relative_fro_error"]) for row in by_map.values()]
        outs = [
            float(row["output_relative_error"])
            for row in by_map.values()
            if row.get("output_relative_error") is not None
        ]
        out.append(
            {
                "scope": "attention_replacement",
                "family": family_label,
                "method": method,
                "maps": len(vals),
                "mean_matrix_error": mean(vals),
                "std_matrix_error": std(vals),
                "p10_matrix_error": quantile(vals, 0.10),
                "p50_matrix_error": quantile(vals, 0.50),
                "p90_matrix_error": quantile(vals, 0.90),
                "frac_matrix_error_lt_0p5": mean([float(v < 0.5) for v in vals]),
                "mean_output_error": mean(outs),
                "frac_output_error_lt_0p2": mean([float(v < 0.2) for v in outs]),
                "winner_rate": mean([float(winners.get(map_id) == method) for map_id in map_ids]),
            }
        )
    return out


def summarize_layer_grid(rows: List[dict], family_label: str) -> List[dict]:
    grid_rows = list(best_rows_by_map(rows, "grid_cyclic_bccb").values())
    out = []
    for layer in sorted({int(r["layer"]) for r in grid_rows}):
        vals = [float(r["relative_fro_error"]) for r in grid_rows if int(r["layer"]) == layer]
        outs = [
            float(r["output_relative_error"])
            for r in grid_rows
            if int(r["layer"]) == layer and r.get("output_relative_error") is not None
        ]
        out.append(
            {
                "scope": "layer_grid_bccb_stability",
                "family": family_label,
                "layer": layer,
                "maps": len(vals),
                "mean_grid_error": mean(vals),
                "std_grid_error": std(vals),
                "frac_grid_error_lt_0p6": mean([float(v < 0.6) for v in vals]),
                "mean_grid_output_error": mean(outs),
            }
        )
    return out


def summarize_wan() -> List[dict]:
    files = [
        "wan_bccb_high_noise_layers0_8_20_39_heads0_10_20_30.json",
        "wan_bccb_low_noise_layers0_8_heads0_10_20_30.json",
    ]
    records = []
    for file_name in files:
        data = load_json(file_name)
        for record in data["records"]:
            records.append(
                {
                    "branch": data["branch"],
                    "layer": int(record["layer"]),
                    "head": int(record["head"]),
                    "attention_r2": float(record["attention"]["cyclic_r2"]),
                    "attention_error": float(record["attention"]["relative_fro_error"]),
                    "logits_r2": float(record["logits"]["cyclic_r2"]),
                    "logits_error": float(record["logits"]["relative_fro_error"]),
                }
            )
    out: List[dict] = []
    for branch in sorted({r["branch"] for r in records}):
        vals = [r["attention_r2"] for r in records if r["branch"] == branch]
        errs = [r["attention_error"] for r in records if r["branch"] == branch]
        out.append(
            {
                "scope": "wan_branch_cyclic_stability",
                "branch": branch,
                "records": len(vals),
                "mean_attention_r2": mean(vals),
                "std_attention_r2": std(vals),
                "frac_attention_r2_ge_0p7": mean([float(v >= 0.7) for v in vals]),
                "frac_attention_r2_lt_0p2": mean([float(v < 0.2) for v in vals]),
                "mean_attention_error": mean(errs),
            }
        )
    common = {}
    for r in records:
        common.setdefault((r["layer"], r["head"]), {})[r["branch"]] = r
    paired = [v for v in common.values() if "high_noise" in v and "low_noise" in v]
    if paired:
        hi = np.array([p["high_noise"]["attention_r2"] for p in paired], dtype=np.float64)
        lo = np.array([p["low_noise"]["attention_r2"] for p in paired], dtype=np.float64)
        corr = float(np.corrcoef(hi, lo)[0, 1]) if len(paired) > 1 else float("nan")
        out.append(
            {
                "scope": "wan_high_low_pair_stability",
                "paired_layer_heads": len(paired),
                "mean_abs_r2_delta": float(np.mean(np.abs(hi - lo))),
                "corr_high_low_r2": corr,
                "same_side_0p7_rate": mean([float((h >= 0.7) == (l >= 0.7)) for h, l in zip(hi, lo)]),
            }
        )
    for r in sorted(records, key=lambda item: (item["branch"], item["layer"], item["head"])):
        out.append({"scope": "wan_record", **r})
    return out


def offset_mask(grid_shape: Sequence[int], radius: int) -> np.ndarray:
    coords = np.array(list(np.ndindex(tuple(grid_shape))), dtype=np.int64)
    shape = np.array(grid_shape, dtype=np.int64)
    diff = (coords[None, :, :] - coords[:, None, :]) % shape
    wrapped = np.minimum(diff, shape - diff)
    return wrapped.max(axis=-1) <= radius


def gini(values: np.ndarray) -> float:
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or float(x.sum()) == 0.0:
        return 0.0
    idx = np.arange(1, x.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x) / (x.size * np.sum(x))) - (x.size + 1.0) / x.size)


def attention_matrix_metrics(attn: np.ndarray, grid_shape: Sequence[int]) -> dict:
    a = attn.astype(np.float64)
    n = a.shape[0]
    col_mass = a.sum(axis=0)
    entropy = -np.sum(np.clip(a, 1e-12, None) * np.log(np.clip(a, 1e-12, None)), axis=1)
    s = np.linalg.svd(a, compute_uv=False)
    p = s / max(float(s.sum()), 1e-12)
    eff_rank = float(np.exp(-np.sum(p * np.log(np.clip(p, 1e-12, None)))))
    local1 = float((a * offset_mask(grid_shape, 1)).sum() / max(float(a.sum()), 1e-12))
    local2 = float((a * offset_mask(grid_shape, 2)).sum() / max(float(a.sum()), 1e-12))
    sorted_rows = np.sort(a, axis=1)[:, ::-1]
    return {
        "n": n,
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "row_argmax_unique_fraction": float(np.unique(a.argmax(axis=1)).size / n),
        "col_mass_gini": gini(col_mass),
        "top1_col_mass_fraction": float(col_mass.max() / max(float(a.sum()), 1e-12)),
        "top2_col_mass_fraction": float(np.sort(col_mass)[-2:].sum() / max(float(a.sum()), 1e-12)),
        "top4_col_mass_fraction": float(np.sort(col_mass)[-min(4, n) :].sum() / max(float(a.sum()), 1e-12)),
        "local_radius1_mass": local1,
        "local_radius2_mass": local2,
        "top1_per_row_mass_mean": float(sorted_rows[:, :1].sum(axis=1).mean()),
        "top2_per_row_mass_mean": float(sorted_rows[:, :2].sum(axis=1).mean()),
        "top4_per_row_mass_mean": float(sorted_rows[:, :4].sum(axis=1).mean()),
        "effective_rank": eff_rank,
        "effective_rank_fraction": eff_rank / n,
    }


def summarize_examples() -> List[dict]:
    sources = [
        (
            "structured_attention_visual_vit_examples_hybrid_20260704.json",
            "structured_attention_visual_vit_examples_hybrid_20260704.npz",
            "vit",
        ),
        (
            "structured_attention_visual_qwen_examples_hybrid_20260704.json",
            "structured_attention_visual_qwen_examples_hybrid_20260704.npz",
            "qwen3vl_visual",
        ),
    ]
    out: List[dict] = []
    for json_name, npz_name, family in sources:
        meta = load_json(json_name)
        arrays = np.load(LOG_DIR / npz_name)
        for item in meta["items"]:
            key = item["key"]
            attn = arrays[f"{key}_attention"]
            metrics = attention_matrix_metrics(attn, item["grid_shape"])
            hybrid = item["metrics"].get("hybrid_balanced", {})
            proxy = item["metrics"].get("monarch_like_mask_proxy", {})
            out.append(
                {
                    "scope": "representative_matrix_pattern",
                    "family": family,
                    "label": item["label"],
                    "map_id": item["map_id"],
                    "grid_shape": "x".join(str(v) for v in item["grid_shape"]),
                    **metrics,
                    "proxy_matrix_error": float(proxy.get("relative_fro_error", float("nan"))),
                    "proxy_output_error": float(proxy.get("output_relative_error", float("nan"))),
                    "hybrid_matrix_error": float(hybrid.get("relative_fro_error", float("nan"))),
                    "hybrid_output_error": float(hybrid.get("output_relative_error", float("nan"))),
                    "hybrid_sink_mass": float(hybrid.get("sink_mass", float("nan"))),
                    "hybrid_local_mass": float(hybrid.get("local_mass", float("nan"))),
                    "hybrid_global_svd_mass": float(hybrid.get("global_svd_mass", float("nan"))),
                    "hybrid_sparse_mass": float(hybrid.get("sparse_mass", float("nan"))),
                }
            )
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: List[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    examples = [r for r in rows if r.get("scope") == "representative_matrix_pattern"]
    labels = [r["label"] for r in examples]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)

    x = np.arange(len(examples))
    axes[0].bar(x - 0.18, [r["top2_col_mass_fraction"] for r in examples], width=0.36, label="top-2 column mass")
    axes[0].bar(x + 0.18, [r["row_argmax_unique_fraction"] for r in examples], width=0.36, label="argmax unique frac")
    axes[0].set_title("Sink / collapse")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].bar(x - 0.18, [r["local_radius1_mass"] for r in examples], width=0.36, label="radius-1 local")
    axes[1].bar(x + 0.18, [r["top4_per_row_mass_mean"] for r in examples], width=0.36, label="row top-4 mass")
    axes[1].set_title("Local vs sparse")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, fontsize=7)

    axes[2].bar(x - 0.18, [r["proxy_output_error"] for r in examples], width=0.36, label="proxy A@V err")
    axes[2].bar(x + 0.18, [r["hybrid_output_error"] for r in examples], width=0.36, label="hybrid A@V err")
    axes[2].set_title("Output replacement")
    axes[2].set_ylim(0, max(0.75, max(r["proxy_output_error"] for r in examples) + 0.05))
    axes[2].legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    fig.text(0.01, 0.96, "(k)", weight="bold")
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_path.with_suffix(suffix), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default=str(LOG_DIR / "pattern_stability_probe_20260707.json"))
    parser.add_argument("--out-csv", default=str(LOG_DIR / "pattern_stability_probe_20260707.csv"))
    parser.add_argument("--out-fig", default=str(FIG_DIR / "fig11_attention_pattern_stability.png"))
    args = parser.parse_args()

    qwen = load_json("structured_attention_probe_qwen_20260703.json")["rows"]
    vit = load_json("structured_attention_probe_vit_20260703.json")["rows"]
    rows: List[dict] = []
    rows.extend(summarize_method_rows(qwen, "qwen3vl_visual"))
    rows.extend(summarize_method_rows(vit, "vit"))
    rows.extend(summarize_layer_grid(qwen, "qwen3vl_visual"))
    rows.extend(summarize_layer_grid(vit, "vit"))
    rows.extend(summarize_wan())
    rows.extend(summarize_examples())

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_csv, rows)
    make_figure(rows, Path(args.out_fig))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {Path(args.out_fig).with_suffix('.png')}")
    print(f"Wrote {Path(args.out_fig).with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
