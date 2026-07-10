#!/usr/bin/env python3
"""Create paper-style visualizations for the video circulant probe."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "remote_logs"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILES = [
    "qwen3vl_visual_circulant_probe.json",
    "qwen3vl_visual_circulant_probe_224_vp4.json",
    "qwen3vl_visual_circulant_probe_224_hpro.json",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def load_probe_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for file_name in RESULT_FILES:
        data = json.loads((LOG_DIR / file_name).read_text(encoding="utf-8"))
        video = Path(data["video"]).stem
        for row in data["rows"]:
            item = dict(row)
            item["video"] = video
            item["grid"] = "x".join(str(v) for v in data["grid_thw"][0])
            rows.append(item)
    return rows


def grouped_mean(rows: list[dict[str, float]], key: str, value: str) -> tuple[list[int], list[float], list[float]]:
    keys = sorted({int(row[key]) for row in rows})
    means, stds = [], []
    for k in keys:
        vals = [float(row[value]) for row in rows if int(row[key]) == k]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return keys, means, stds


def figure_layer_summary(rows: list[dict[str, float]]) -> None:
    layers, r2_mean, r2_std = grouped_mean(rows, "layer", "circulant_r2")
    _, err_mean, err_std = grouped_mean(rows, "layer", "relative_fro_error")

    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(5.9, 2.5))
    ax2 = ax.twinx()
    bars = ax.bar(x - 0.18, r2_mean, width=0.34, yerr=r2_std, capsize=2.5, color="#4C78A8", label="Cyclic R2")
    line = ax2.errorbar(x + 0.18, err_mean, yerr=err_std, fmt="o-", ms=4, lw=1.4, capsize=2.5, color="#F58518", label="Relative error")

    ax.set_xticks(x)
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_xlabel("Qwen3-VL visual layer")
    ax.set_ylabel("Cyclic/BCCB R2")
    ax2.set_ylabel("Relative Frobenius error")
    ax.set_ylim(0, max(r2_mean) + max(r2_std) + 0.08)
    ax2.set_ylim(0.65, 1.02)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    handles = [bars, line.lines[0]]
    labels = ["Cyclic R2", "Relative error"]
    ax.legend(handles, labels, loc="upper right", frameon=False)
    fig.text(0.02, 0.96, "(a)", weight="bold")
    save(fig, "fig1_qwen_layer_cyclic_fit")


def figure_head_heatmap(rows: list[dict[str, float]]) -> None:
    layers = sorted({int(r["layer"]) for r in rows})
    heads = sorted({int(r["head"]) for r in rows})
    r2 = np.full((len(layers), len(heads)), np.nan)
    err = np.full_like(r2, np.nan)
    for i, layer in enumerate(layers):
        for j, head in enumerate(heads):
            vals = [r for r in rows if int(r["layer"]) == layer and int(r["head"]) == head]
            if vals:
                r2[i, j] = np.mean([float(v["circulant_r2"]) for v in vals])
                err[i, j] = np.mean([float(v["relative_fro_error"]) for v in vals])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35), constrained_layout=True)
    im0 = axes[0].imshow(r2, aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.42, float(np.nanmax(r2))))
    im1 = axes[1].imshow(err, aspect="auto", cmap="viridis_r", vmin=0.70, vmax=1.0)
    for ax, title in zip(axes, ["Cyclic/BCCB R2", "Relative error"]):
        ax.set_xticks(np.arange(len(heads)))
        ax.set_xticklabels([str(h) for h in heads])
        ax.set_yticks(np.arange(len(layers)))
        ax.set_yticklabels([str(l) for l in layers])
        ax.set_xlabel("Head")
        ax.set_title(title, pad=4)
    axes[0].set_ylabel("Layer")
    for i in range(len(layers)):
        for j in range(len(heads)):
            if not np.isnan(r2[i, j]):
                axes[0].text(j, i, f"{r2[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
                axes[1].text(j, i, f"{err[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    fig.text(
        0.5,
        -0.02,
        "Cells H0-H3 use 3 clips x 2 temporal slices; H4-H7 use the two 224px clips x 2 slices.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    fig.text(0.01, 0.96, "(b)", weight="bold")
    save(fig, "fig2_qwen_head_heatmaps")


def figure_attention_examples() -> None:
    npz = np.load(LOG_DIR / "qwen3vl_attention_visual_examples.npz")
    meta = json.loads((LOG_DIR / "qwen3vl_attention_visual_examples.json").read_text(encoding="utf-8"))
    items = meta["items"]
    cols = ["attention", "projection", "residual", "kernel"]
    col_labels = ["Attention A", "Nearest BCCB P(A)", "|A - P(A)|", "Cyclic kernel"]

    fig, axes = plt.subplots(len(items), len(cols), figsize=(7.6, 7.2), constrained_layout=True)
    for i, item in enumerate(items):
        key = item["key"]
        attn = npz[f"{key}_attention"]
        proj = npz[f"{key}_projection"]
        residual = npz[f"{key}_residual"]
        kernel = npz[f"{key}_kernel"]
        mats = [attn, proj, residual, kernel]
        vmax = np.percentile(attn, 99.4)
        for j, mat in enumerate(mats):
            ax = axes[i, j]
            if j < 2:
                im = ax.imshow(mat, cmap="magma", vmin=0, vmax=vmax)
            elif j == 2:
                im = ax.imshow(mat, cmap="gray_r", vmin=0, vmax=np.percentile(residual, 99.2))
            else:
                im = ax.imshow(mat, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_labels[j], pad=5)
            if j == 0:
                label = f"L{item['layer']} H{item['head']} F{item['frame']}\nR2={item['circulant_r2']:.2f}, err={item['relative_fro_error']:.2f}"
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=42)
    fig.text(0.01, 0.985, "(c)", weight="bold")
    save(fig, "fig3_qwen_attention_bccb_examples")


def figure_wan_rowmax_context() -> None:
    context = json.loads((LOG_DIR / "wan_rowmax_context.json").read_text(encoding="utf-8"))["metrics"]
    labels = ["old preset\ncoverage", "new preset\ncoverage", "dynamic-slot\nreduction"]
    values = [
        context["old_coverage_percent"],
        context["new_coverage_percent"],
        context["dynamic_reduction_among_previously_dynamic_percent"],
    ]
    colors = ["#9ECAE9", "#31A354", "#756BB1"]

    fig, ax = plt.subplots(figsize=(4.7, 2.5))
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.58)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Coverage / reduction (%)")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.0,
        -0.34,
        "Wan2.2 rowmax presets show QK-scale regularity, not a direct BCCB/circulant proof.",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )
    fig.text(0.02, 0.96, "(d)", weight="bold")
    save(fig, "fig4_wan_rowmax_context")


def load_wan_bccb_rows() -> list[dict[str, float | int | str]]:
    files = [
        "wan_bccb_high_noise_layers0_8_20_39_heads0_10_20_30.json",
        "wan_bccb_low_noise_layers0_8_heads0_10_20_30.json",
    ]
    rows: list[dict[str, float | int | str]] = []
    for file_name in files:
        data = json.loads((LOG_DIR / file_name).read_text(encoding="utf-8"))
        for record in data["records"]:
            rows.append(
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
    return rows


def _metric_grid(rows: list[dict[str, float | int | str]], branch: str, metric: str) -> tuple[list[int], list[int], np.ndarray]:
    subset = [row for row in rows if row["branch"] == branch]
    layers = sorted({int(row["layer"]) for row in subset})
    heads = sorted({int(row["head"]) for row in subset})
    grid = np.full((len(layers), len(heads)), np.nan)
    for i, layer in enumerate(layers):
        for j, head in enumerate(heads):
            vals = [float(row[metric]) for row in subset if int(row["layer"]) == layer and int(row["head"]) == head]
            if vals:
                grid[i, j] = float(np.mean(vals))
    return layers, heads, grid


def figure_wan_direct_bccb_probe() -> None:
    rows = load_wan_bccb_rows()
    panels = [
        ("high_noise", "attention_r2", "High-noise attention R2", "magma", 0.0, 1.0),
        ("high_noise", "attention_error", "High-noise relative error", "viridis_r", 0.0, 0.75),
        ("low_noise", "attention_r2", "Low-noise attention R2", "magma", 0.0, 1.0),
        ("low_noise", "attention_error", "Low-noise relative error", "viridis_r", 0.0, 0.75),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2), constrained_layout=True)
    for ax, (branch, metric, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        layers, heads, grid = _metric_grid(rows, branch, metric)
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, pad=4)
        ax.set_xticks(np.arange(len(heads)))
        ax.set_xticklabels([str(h) for h in heads])
        ax.set_yticks(np.arange(len(layers)))
        ax.set_yticklabels([str(layer) for layer in layers])
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        for i in range(len(layers)):
            for j in range(len(heads)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    high = [row for row in rows if row["branch"] == "high_noise"]
    low = [row for row in rows if row["branch"] == "low_noise"]
    fig.text(
        0.5,
        -0.02,
        "Wan2.2 direct Q/K probe: true DiT weights, blockwise forward, synthetic latent/text context, grid 2x30x52; "
        f"mean R2 high={np.mean([r['attention_r2'] for r in high]):.2f}, low={np.mean([r['attention_r2'] for r in low]):.2f}.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(e)", weight="bold", fontsize=9)
    save(fig, "fig5_wan_direct_bccb_probe")


def figure_structured_matrix_fit() -> None:
    data = json.loads((LOG_DIR / "structured_matrix_probe_vit_qwen_20260702.json").read_text(encoding="utf-8"))
    summary = data["summary"]
    method_order = ["block_circulant", "permuted_block_circulant", "monarch_like_proxy"]
    method_labels = {
        "block_circulant": "BCM",
        "permuted_block_circulant": "BCM + fixed perm.",
        "monarch_like_proxy": "Monarch-like proxy",
    }
    family_labels = {"vit": "ViT", "qwen3vl_visual": "Qwen3-VL visual"}
    colors = {
        "block_circulant": "#4C78A8",
        "permuted_block_circulant": "#F58518",
        "monarch_like_proxy": "#54A24B",
    }

    best = {}
    for row in summary:
        key = (row["family"], row["method"])
        old = best.get(key)
        if old is None or row["mean_best_relative_fro_error"] < old["mean_best_relative_fro_error"]:
            best[key] = row

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9), constrained_layout=True)
    ax = axes[0]
    families = ["vit", "qwen3vl_visual"]
    x = np.arange(len(families))
    width = 0.24
    for offset, method in zip([-width, 0, width], method_order):
        vals = [best[(family, method)]["mean_best_relative_fro_error"] for family in families]
        labels = [
            f"b={best[(family, method)]['block_size']}, x{best[(family, method)]['mean_compression_ratio']:.1f}"
            for family in families
        ]
        bars = ax.bar(x + offset, vals, width=width, color=colors[method], label=method_labels[method])
        for bar, label in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([family_labels[f] for f in families])
    ax.set_ylabel("Best mean relative error")
    ax.set_ylim(0.0, 1.01)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    markers = {"vit": "o", "qwen3vl_visual": "s"}
    for row in summary:
        ax.scatter(
            row["mean_compression_ratio"],
            row["mean_best_relative_fro_error"],
            s=42,
            marker=markers[row["family"]],
            color=colors[row["method"]],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.92,
        )
        ax.text(
            row["mean_compression_ratio"] * 1.02,
            row["mean_best_relative_fro_error"],
            f"b{row['block_size']}",
            fontsize=6.5,
            va="center",
            color="#333333",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Compression ratio vs dense")
    ax.set_ylabel("Mean relative error")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, color="#e6e6e6", lw=0.6, which="both")
    ax.text(
        0.98,
        0.04,
        "Proxy keeps two fixed block-diagonal layouts;\nnot a trained Monarch factorization.",
        transform=ax.transAxes,
        fontsize=7,
        color="#444444",
        ha="right",
        va="bottom",
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)
    fig.text(0.01, 0.985, "(f)", weight="bold", fontsize=9)
    save(fig, "fig6_structured_matrix_fit")


def figure_structured_attention_fit() -> None:
    files = [
        "structured_attention_probe_vit_20260703.json",
        "structured_attention_probe_qwen_20260703.json",
    ]
    summary = []
    for file_name in files:
        data = json.loads((LOG_DIR / file_name).read_text(encoding="utf-8"))
        summary.extend(row for row in data["summary"] if row["matrix_kind"] == "attention")

    method_order = [
        "grid_cyclic_bccb",
        "flat_block_circulant",
        "permuted_flat_block_circulant",
        "monarch_like_mask_proxy",
    ]
    method_labels = {
        "grid_cyclic_bccb": "Grid BCCB",
        "flat_block_circulant": "Flat BCM",
        "permuted_flat_block_circulant": "Flat BCM + perm.",
        "monarch_like_mask_proxy": "Monarch-like proxy",
    }
    colors = {
        "grid_cyclic_bccb": "#4C78A8",
        "flat_block_circulant": "#F58518",
        "permuted_flat_block_circulant": "#B279A2",
        "monarch_like_mask_proxy": "#54A24B",
    }
    groups = [
        ("vit", "patch_patch_resoftmax_exact_layer0", "ViT L0"),
        ("vit", "patch_patch_resoftmax_attention_only_rollout", "ViT rollout"),
        ("qwen3vl_visual", "per_temporal_slice_spatial_attention", "Qwen3-VL"),
    ]

    def get_row(family: str, scope: str, method: str) -> dict:
        matches = [row for row in summary if row["family"] == family and row["scope"] == scope and row["method"] == method]
        if not matches:
            raise KeyError((family, scope, method))
        return matches[0]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), constrained_layout=True)
    x = np.arange(len(groups))
    width = 0.18
    for idx, method in enumerate(method_order):
        offset = (idx - 1.5) * width
        fro_vals = [get_row(f, s, method)["mean_best_relative_fro_error"] for f, s, _label in groups]
        out_vals = [get_row(f, s, method)["mean_best_output_relative_error"] for f, s, _label in groups]
        axes[0].bar(x + offset, fro_vals, width=width, color=colors[method], label=method_labels[method])
        axes[1].bar(x + offset, out_vals, width=width, color=colors[method], label=method_labels[method])

    for ax, ylabel in zip(axes, ["Attention matrix relative error", "Replacement output error in A@V"]):
        ax.set_xticks(x)
        ax.set_xticklabels([label for _f, _s, label in groups])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    fig.text(
        0.5,
        -0.02,
        "ViT rollout uses dense attention-only state after layer 0. Qwen3-VL is true visual forward, per temporal slice. "
        "Proxy is row-renormalized masked attention, not a trained Monarch factorization.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False)
    fig.text(0.01, 0.985, "(g)", weight="bold", fontsize=9)
    save(fig, "fig7_structured_attention_replacement")


def figure_attention_matrix_failure_modes() -> None:
    sources = [
        (
            json.loads((LOG_DIR / "structured_attention_visual_vit_examples_20260703.json").read_text(encoding="utf-8")),
            np.load(LOG_DIR / "structured_attention_visual_vit_examples_20260703.npz"),
        ),
        (
            json.loads((LOG_DIR / "structured_attention_visual_qwen_examples_20260703.json").read_text(encoding="utf-8")),
            np.load(LOG_DIR / "structured_attention_visual_qwen_examples_20260703.npz"),
        ),
    ]
    examples = []
    for meta, arrays in sources:
        for item in meta["items"]:
            examples.append((item, arrays))

    cols = [
        ("attention", "Attention A"),
        ("grid_cyclic_bccb", "Grid BCCB"),
        ("grid_cyclic_bccb_residual", "|A - Grid|"),
        ("permuted_flat_block_circulant", "Best flat BCM+perm"),
        ("monarch_like_mask_proxy", "Monarch-like proxy"),
        ("argmax", "Row argmax"),
    ]
    fig, axes = plt.subplots(len(examples), len(cols), figsize=(9.0, 6.9), constrained_layout=True)
    for row_idx, (item, arrays) in enumerate(examples):
        key = item["key"]
        attn = arrays[f"{key}_attention"]
        vmax = max(float(np.percentile(attn, 99.4)), 1e-6)
        for col_idx, (suffix, title) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            if suffix == "argmax":
                argmax = attn.argmax(axis=-1)
                entropy = -np.sum(np.clip(attn, 1e-12, None) * np.log(np.clip(attn, 1e-12, None)), axis=-1)
                sc = ax.scatter(np.arange(attn.shape[0]), argmax, c=entropy, s=5, cmap="viridis")
                ax.plot([0, attn.shape[0] - 1], [0, attn.shape[1] - 1], color="#aaaaaa", lw=0.6, ls="--")
                ax.set_xlim(-1, attn.shape[0])
                ax.set_ylim(attn.shape[1], -1)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                mat = arrays[f"{key}_{suffix}"] if suffix != "attention" else attn
                if "residual" in suffix:
                    ax.imshow(mat, cmap="gray_r", vmin=0, vmax=max(float(np.percentile(mat, 99.4)), 1e-6))
                else:
                    ax.imshow(mat, cmap="magma", vmin=0, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, pad=5, fontsize=8)
            if col_idx == 0:
                metrics = item["metrics"]
                label = (
                    f"{item['label']}\n"
                    f"Grid err={metrics['grid_cyclic_bccb']['relative_fro_error']:.2f}, "
                    f"out={metrics['grid_cyclic_bccb']['output_relative_error']:.2f}\n"
                    f"Perm err={metrics['permuted_flat_block_circulant']['relative_fro_error']:.2f}, "
                    f"Proxy err={metrics['monarch_like_mask_proxy']['relative_fro_error']:.2f}"
                )
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=54, fontsize=7)
    fig.text(
        0.5,
        -0.01,
        "If grid-BCCB held, rows would look like cyclic shifts and row-argmax points would form offset diagonals. "
        "Actual maps show sinks, sparse rows, and content-dependent row patterns.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.99, "(h)", weight="bold", fontsize=9)
    save(fig, "fig8_attention_matrix_failure_modes")


def figure_hybrid_attention_decomposition() -> None:
    meta = json.loads((LOG_DIR / "hybrid_attention_decomposition_20260704.json").read_text(encoding="utf-8"))
    arrays = np.load(LOG_DIR / "hybrid_attention_decomposition_20260704.npz")
    items = meta["items"]
    cols = [
        ("attention", "Attention A"),
        ("sink_global_svd", "Sink + global SVD"),
        ("local_cyclic", "Local cyclic"),
        ("sparse_routing", "Sparse routing"),
        ("hybrid", "Hybrid sum"),
        ("residual", "|A - Hybrid|"),
    ]

    fig, axes = plt.subplots(len(items), len(cols), figsize=(9.2, 6.9), constrained_layout=True)
    for row_idx, item in enumerate(items):
        key = item["key"]
        attn = arrays[f"{key}_attention"]
        vmax = max(float(np.percentile(attn, 99.4)), 1e-6)
        for col_idx, (suffix, title) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            mat = arrays[f"{key}_{suffix}"]
            if suffix == "residual":
                ax.imshow(mat, cmap="gray_r", vmin=0, vmax=max(float(np.percentile(mat, 99.4)), 1e-6))
            else:
                ax.imshow(mat, cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, pad=5, fontsize=8)
            if col_idx == 0:
                balanced = next(row for row in item["hybrid_configs"] if row["name"] == "hybrid_balanced")
                baselines = item["baseline_relative_fro_error"]
                label = (
                    f"{item['label']}\n"
                    f"Grid={baselines['grid_cyclic_bccb']:.2f}, Perm={baselines['permuted_flat_block_circulant']:.2f}\n"
                    f"Proxy={baselines['monarch_like_mask_proxy']:.2f}, Hybrid={balanced['relative_fro_error']:.2f}\n"
                    f"budget {balanced['nominal_budget_ratio']:.1f}x, s/r/k={balanced['sink_k']}/{balanced['rank']}/{balanced['sparse_k']}"
                )
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=60, fontsize=7)
    fig.text(
        0.5,
        -0.01,
        "Hybrid is an oracle diagnostic: sink columns and sparse routes are selected from observed A. "
        "The SVD-derived global component is clipped, so budget ratios are not deployable compression claims.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.99, "(i)", weight="bold", fontsize=9)
    save(fig, "fig9_hybrid_attention_decomposition")


def figure_hybrid_attention_tradeoff() -> None:
    data = json.loads((LOG_DIR / "hybrid_attention_decomposition_20260704.json").read_text(encoding="utf-8"))
    items = data["items"]
    method_labels = [
        ("grid_cyclic_bccb", "Grid BCCB", "#4C78A8"),
        ("permuted_flat_block_circulant", "Flat BCM+perm", "#B279A2"),
        ("monarch_like_mask_proxy", "Monarch-like proxy", "#54A24B"),
        ("hybrid_balanced", "Hybrid balanced", "#E45756"),
        ("hybrid_plus", "Hybrid plus", "#72B7B2"),
    ]
    x = np.arange(len(items))
    width = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), constrained_layout=True)
    for idx, (method, label, color) in enumerate(method_labels):
        vals = []
        comps = []
        for item in items:
            if method.startswith("hybrid_"):
                row = next(cfg for cfg in item["hybrid_configs"] if cfg["name"] == method)
                vals.append(row["relative_fro_error"])
                comps.append(row["nominal_budget_ratio"])
            else:
                vals.append(item["baseline_relative_fro_error"][method])
                comps.append(np.nan)
        axes[0].bar(x + (idx - 2) * width, vals, width=width, label=label, color=color)
        if method.startswith("hybrid_"):
            axes[1].scatter(comps, vals, s=46, label=label, color=color, edgecolor="white", linewidth=0.5)
            for comp, val, item in zip(comps, vals, items):
                axes[1].text(comp * 1.015, val, item["label"].replace(" ", "\n", 1), fontsize=6, va="center")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([item["label"].replace(" ", "\n", 1) for item in items])
    axes[0].set_ylabel("Attention matrix relative error")
    axes[0].set_ylim(0, 1.30)
    axes[0].grid(axis="y", color="#e6e6e6", lw=0.6)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Hybrid nominal budget ratio vs dense")
    axes[1].set_ylabel("Relative error")
    axes[1].set_ylim(0, 0.45)
    axes[1].grid(True, color="#e6e6e6", lw=0.6, which="both")
    axes[1].text(
        0.98,
        0.96,
        "Lower error uses oracle sink/top-k choices;\nbudget ratio is not deployable compression.",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False)
    fig.text(0.01, 0.985, "(j)", weight="bold", fontsize=9)
    save(fig, "fig10_hybrid_attention_tradeoff")


def figure_attention_pattern_full_sweep() -> None:
    sources = [
        ("ViT", json.loads((LOG_DIR / "attention_pattern_full_vit_20260707.json").read_text(encoding="utf-8"))),
        ("Qwen3-VL", json.loads((LOG_DIR / "attention_pattern_full_qwen_20260707.json").read_text(encoding="utf-8"))),
    ]

    layer_rows = {}
    for label, data in sources:
        rows = data["rows"]
        layers = sorted({int(row["layer"]) for row in rows})
        layer_rows[label] = []
        for layer in layers:
            subset = [row for row in rows if int(row["layer"]) == layer]
            layer_rows[label].append(
                {
                    "layer": layer,
                    "top2": float(np.mean([row["top2_col_mass_fraction"] for row in subset])),
                    "arguniq": float(np.mean([row["row_argmax_unique_fraction"] for row in subset])),
                    "local1": float(np.mean([row["local_radius1_mass"] for row in subset])),
                    "top4row": float(np.mean([row["top4_per_row_mass_mean"] for row in subset])),
                    "union_v": float(np.mean([row["union_sink_local_top4_output_error"] for row in subset])),
                    "union_rand": float(np.mean([row["union_random_v_output_error"] for row in subset])),
                }
            )

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    colors = {"ViT": "#4C78A8", "Qwen3-VL": "#F58518"}
    markers = {"ViT": "o", "Qwen3-VL": "s"}

    for label, points in layer_rows.items():
        xs = [p["layer"] for p in points]
        axes[0].plot(xs, [p["top2"] for p in points], marker=markers[label], lw=1.6, color=colors[label], label=f"{label} top-2")
        axes[0].plot(xs, [p["arguniq"] for p in points], marker=markers[label], lw=1.2, color=colors[label], ls="--", label=f"{label} argmax")
        axes[1].plot(xs, [p["local1"] for p in points], marker=markers[label], lw=1.6, color=colors[label], label=f"{label} local r1")
        axes[1].plot(xs, [p["top4row"] for p in points], marker=markers[label], lw=1.2, color=colors[label], ls="--", label=f"{label} row top-4")
        axes[2].plot(xs, [p["union_v"] for p in points], marker=markers[label], lw=1.6, color=colors[label], label=f"{label} true V")
        axes[2].plot(xs, [p["union_rand"] for p in points], marker=markers[label], lw=1.2, color=colors[label], ls="--", label=f"{label} random V")

    axes[0].set_title("Sink / collapse by layer")
    axes[0].set_ylabel("Mean fraction")
    axes[1].set_title("Local vs row-sparse mass")
    axes[2].set_title("Union-mask output error")
    axes[2].set_ylabel("Relative error")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylim(0, 0.72)
    axes[1].set_ylim(0, 0.92)
    axes[2].set_ylim(0, 1.9)
    fig.text(
        0.5,
        -0.04,
        "Union mask = oracle sink-2 columns + local radius-1 + row top-4. "
        "Random-V stress exposes value-subspace dependence.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(l)", weight="bold", fontsize=9)
    save(fig, "fig12_attention_pattern_full_sweep")


def figure_attention_component_intervention() -> None:
    data = json.loads((LOG_DIR / "attention_component_intervention_20260707.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    labels = [row["label"].replace(" ", "\n", 1) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), constrained_layout=True)

    error_methods = [
        ("grid_bccb_error", "Grid BCCB", "#4C78A8"),
        ("monarch_proxy_error", "Monarch proxy", "#54A24B"),
        ("full_hybrid_error", "Hybrid full", "#E45756"),
        ("no_sink_global_error", "No sink/global", "#B279A2"),
        ("no_local_cyclic_error", "No local", "#F58518"),
        ("no_sparse_routing_error", "No sparse", "#72B7B2"),
    ]
    width = 0.12
    for idx, (key, label, color) in enumerate(error_methods):
        vals = [float(row[key]) for row in rows]
        axes[0].bar(x + (idx - 2.5) * width, vals, width=width, label=label, color=color)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Relative Frobenius error")
    axes[0].set_title("Component ablation on representative A")
    axes[0].set_ylim(0, 1.58)
    axes[0].grid(axis="y", color="#e6e6e6", lw=0.6)

    mass_keys = [
        ("sink_global_mass", "Sink/global", "#B279A2"),
        ("local_cyclic_mass", "Local cyclic", "#F58518"),
        ("sparse_routing_mass", "Sparse routing", "#72B7B2"),
    ]
    bottom = np.zeros(len(rows))
    for key, label, color in mass_keys:
        vals = np.array([float(row[key]) for row in rows])
        axes[1].bar(x, vals, bottom=bottom, label=label, color=color, width=0.55)
        bottom += vals
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Component mass / attention mass")
    axes[1].set_title("Oracle decomposition mass")
    axes[1].set_ylim(0, 1.12)
    axes[1].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[1].text(
        0.98,
        0.96,
        "Sink/global includes selected sink columns\nand clipped low-rank global SVD.",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )

    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    fig.legend(handles0 + handles1, labels0 + labels1, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=5, frameon=False)
    fig.text(
        0.5,
        -0.04,
        "Matrix-level diagnostic only: components are selected from observed dense attention A, so this is not a deployable router or task-loss ablation.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(m)", weight="bold", fontsize=9)
    save(fig, "fig13_attention_component_intervention")


def figure_value_subspace_stress() -> None:
    sources = [
        ("ViT", json.loads((LOG_DIR / "attention_pattern_vstress_vit_20260707.json").read_text(encoding="utf-8"))),
        ("Qwen3-VL", json.loads((LOG_DIR / "attention_pattern_vstress_qwen_20260707.json").read_text(encoding="utf-8"))),
    ]
    stress_keys = [
        ("union_sink_local_top4_output_error", "True V", "#4C78A8"),
        ("union_permuted_v_output_error", "Permuted V", "#F58518"),
        ("union_orthogonalized_v_output_error", "Orthogonalized V", "#54A24B"),
        ("union_random_v_output_error", "Random V", "#B279A2"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0), constrained_layout=True)
    x = np.arange(len(stress_keys))
    for family_idx, (label, data) in enumerate(sources):
        rows = data["rows"]
        vals = [float(np.mean([row[key] for row in rows])) for key, _name, _color in stress_keys]
        colors = [color for _key, _name, color in stress_keys]
        axes[0].bar(x + (family_idx - 0.5) * 0.32, vals, width=0.30, label=label, color=colors, alpha=0.65 + 0.25 * family_idx)
        for xi, val in zip(x + (family_idx - 0.5) * 0.32, vals):
            axes[0].text(xi, val + 0.035, f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

        layers = sorted({int(row["layer"]) for row in rows})
        for key, name, color in stress_keys:
            y = [float(np.mean([row[key] for row in rows if int(row["layer"]) == layer])) for layer in layers]
            axes[1].plot(layers, y, marker="o" if label == "ViT" else "s", lw=1.3, color=color, ls="-" if label == "ViT" else "--", label=f"{label} {name}")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name for _key, name, _color in stress_keys])
    axes[0].set_ylabel("Union-mask A@V relative error")
    axes[0].set_title("Aggregate value-subspace stress")
    axes[0].set_ylim(0, 1.85)
    axes[0].grid(axis="y", color="#e6e6e6", lw=0.6)
    handles = [
        mpatches.Patch(facecolor="#888888", alpha=0.65, label="ViT"),
        mpatches.Patch(facecolor="#888888", alpha=0.90, label="Qwen3-VL"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="upper left")

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Layer-wise stress")
    axes[1].set_ylim(0, 2.05)
    axes[1].grid(True, color="#e6e6e6", lw=0.6)
    axes[1].legend(frameon=False, fontsize=5.8, ncol=2)
    fig.text(
        0.5,
        -0.04,
        "Union mask = oracle sink-2 + radius-1 local + row top-4. "
        "Permuted V breaks token-value alignment; orthogonalized V removes feature covariance; random V changes the value subspace.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(n)", weight="bold", fontsize=9)
    save(fig, "fig14_value_subspace_stress")


def figure_head_output_intervention() -> None:
    sources = [
        ("ViT", json.loads((LOG_DIR / "attention_head_intervention_vit_20260707.json").read_text(encoding="utf-8"))),
        ("Qwen3-VL", json.loads((LOG_DIR / "attention_head_intervention_qwen_20260707.json").read_text(encoding="utf-8"))),
    ]
    components = [
        ("sink2", "Sink-2", "sink2_output_error", "drop_sink2_output_error", "sink2_raw_component_norm_ratio", "#4C78A8"),
        ("local1", "Local r1", "local1_output_error", "drop_local1_output_error", "local1_raw_component_norm_ratio", "#F58518"),
        ("top4", "Row top-4", "row_top4_output_error", "drop_row_top4_output_error", "row_top4_raw_component_norm_ratio", "#54A24B"),
        ("union", "Union", "union_sink_local_top4_output_error", "drop_union_sink_local_top4_output_error", "union_raw_component_norm_ratio", "#B279A2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), constrained_layout=True)
    x = np.arange(len(components))
    width = 0.32
    panels = [
        ("Keep-only A@V error", 2, "keep"),
        ("Drop + renorm A@V error", 3, "drop"),
        ("Raw component norm / base", 4, "ratio"),
    ]
    for ax, (title, idx, _kind) in zip(axes, panels):
        for family_idx, (family_label, data) in enumerate(sources):
            rows = data["rows"]
            vals = [float(np.mean([row[component[idx]] for row in rows])) for component in components]
            bars = ax.bar(x + (family_idx - 0.5) * width, vals, width=width, label=family_label, color=[c[5] for c in components], alpha=0.65 + 0.25 * family_idx)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.035, f"{val:.2f}", ha="center", va="bottom", fontsize=6.3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([component[1] for component in components], rotation=15)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[0].set_ylabel("Relative error")
    axes[1].set_ylabel("Relative error")
    axes[2].set_ylabel("Norm ratio")
    axes[0].set_ylim(0, 2.2)
    axes[1].set_ylim(0, 1.7)
    axes[2].set_ylim(0, 1.45)
    handles = [
        mpatches.Patch(facecolor="#888888", alpha=0.65, label="ViT"),
        mpatches.Patch(facecolor="#888888", alpha=0.90, label="Qwen3-VL"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
    fig.text(
        0.5,
        -0.04,
        "Head-output intervention only: masks are selected from observed A. "
        "Keep-only tests sufficiency; drop+renorm tests whether remaining attention can compensate.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(o)", weight="bold", fontsize=9)
    save(fig, "fig15_head_output_intervention")


def figure_wan_delta_perturbation() -> None:
    files = [
        ("High noise", "wan_bccb_high_noise_delta_perturb_smallgrid_layers0_8_20_39_heads0_10_20_30.json"),
        ("Low noise", "wan_bccb_low_noise_delta_perturb_smallgrid_layers0_8_heads0_10_20_30.json"),
    ]
    rows = []
    for branch_label, file_name in files:
        data = json.loads((LOG_DIR / file_name).read_text(encoding="utf-8"))
        rows.append(
            {
                "branch": branch_label,
                "variant": "true F-H-W",
                "attention_r2": float(data["mean_attention_cyclic_r2"]),
                "attention_error": float(data["mean_attention_relative_fro_error"]),
            }
        )
        for variant, value in data["delta_perturbation_mean_attention_cyclic_r2"].items():
            rows.append(
                {
                    "branch": branch_label,
                    "variant": variant.replace("_", "\n"),
                    "attention_r2": float(value),
                    "attention_error": float(data["delta_perturbation_mean_attention_relative_fro_error"][variant]),
                }
            )

    variants = ["true F-H-W", "axis\nhfw", "axis\nfwh", "axis\nwhf", "reverse\ncoord", "random\ncoord"]
    branches = ["High noise", "Low noise"]
    colors = {"High noise": "#4C78A8", "Low noise": "#F58518"}
    x = np.arange(len(variants))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), constrained_layout=True)
    for branch_idx, branch in enumerate(branches):
        subset = [row for row in rows if row["branch"] == branch]
        r2_vals = [next(row["attention_r2"] for row in subset if row["variant"] == variant) for variant in variants]
        err_vals = [next(row["attention_error"] for row in subset if row["variant"] == variant) for variant in variants]
        offset = (branch_idx - 0.5) * width
        axes[0].bar(x + offset, r2_vals, width=width, label=branch, color=colors[branch], alpha=0.78)
        axes[1].bar(x + offset, err_vals, width=width, label=branch, color=colors[branch], alpha=0.78)
        for ax, vals in zip(axes, [r2_vals, err_vals]):
            for xi, val in zip(x + offset, vals):
                ax.text(xi, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=6.3)

    axes[0].set_title("Wan 3D cyclic R2 under coordinate perturbation")
    axes[0].set_ylabel("Mean attention R2")
    axes[0].set_ylim(0, 0.82)
    axes[1].set_title("Relative error")
    axes[1].set_ylabel("Mean relative Frobenius error")
    max_err = max(row["attention_error"] for row in rows)
    axes[1].set_ylim(0, max(0.72, max_err * 1.16))
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.legend(frameon=False, fontsize=7)
    fig.text(
        0.5,
        -0.04,
        "Small-grid Wan probe, patch grid 2x15x26. Perturbations reinterpret or shuffle token coordinates after the same RoPE-applied Q/K capture. "
        "A large drop from true F-H-W supports geometry-aligned 3D relative-offset structure.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(p)", weight="bold", fontsize=9)
    save(fig, "fig16_wan_delta_perturbation")


def figure_hybrid_transfer_probe() -> None:
    data = json.loads((LOG_DIR / "hybrid_transfer_probe_20260708.json").read_text(encoding="utf-8"))
    summary_by_scope = {row["scope"]: row for row in data["summary"]}
    scopes = ["all_same_grid_pairs", "same_family_pairs", "cross_family_pairs"]
    labels = ["All same-grid", "Same family", "Cross family"]
    error_keys = [
        ("target_oracle_hybrid_error", "Target\noracle\nhybrid", "#4C78A8"),
        ("target_support_mask_error", "Target\nsupport\nonly", "#72B7B2"),
        ("source_support_transfer_error", "Source\nsupport\ntransfer", "#F58518"),
        ("source_hybrid_template_error", "Source\nhybrid\ntemplate", "#E45756"),
    ]
    jaccard_keys = [
        ("support_jaccard", "Union support", "#72B7B2"),
        ("sink_jaccard", "Sink cols", "#4C78A8"),
        ("sparse_route_jaccard", "Sparse routes", "#E45756"),
    ]
    x = np.arange(len(scopes))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), constrained_layout=True)

    width = 0.18
    for idx, (key, label, color) in enumerate(error_keys):
        vals = [float(summary_by_scope[scope][f"mean_{key}"]) for scope in scopes]
        offset = (idx - 1.5) * width
        axes[0].bar(x + offset, vals, width=width, color=color, alpha=0.82, label=label.replace("\n", " "))
        for xi, val in zip(x + offset, vals):
            axes[0].text(xi, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=6.1)
    axes[0].set_title("Oracle hybrid does not transfer as a fixed template")
    axes[0].set_ylabel("Relative Frobenius error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 2.75)
    axes[0].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[0].legend(frameon=False, fontsize=6.4, ncol=2)

    width = 0.22
    for idx, (key, label, color) in enumerate(jaccard_keys):
        vals = [float(summary_by_scope[scope][f"mean_{key}"]) for scope in scopes]
        offset = (idx - 1.0) * width
        axes[1].bar(x + offset, vals, width=width, color=color, alpha=0.82, label=label)
        for xi, val in zip(x + offset, vals):
            axes[1].text(xi, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=6.3)
    axes[1].set_title("Sink and sparse routes are target-specific")
    axes[1].set_ylabel("Source/target Jaccard")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 0.72)
    axes[1].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[1].legend(frameon=False, fontsize=7)

    fig.text(
        0.5,
        -0.05,
        "Transfer probe over saved representative 8x8 matrices. Source support/template uses another map; "
        "target support is a semi-oracle support-only baseline. Low oracle error but high transfer error indicates "
        "content- and layer-specific routing rather than a reusable fixed BCCB/Monarch template.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(q)", weight="bold", fontsize=9)
    save(fig, "fig17_hybrid_transfer_probe")


def figure_wan_noise_branch_stability() -> None:
    data = json.loads((LOG_DIR / "wan_noise_branch_stability_20260708.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    summary = data["summary"][0]
    labels = [f"L{int(row['layer'])}\nH{int(row['head'])}" for row in rows]
    high_r2 = np.array([float(row["high_attention_r2"]) for row in rows])
    low_r2 = np.array([float(row["low_attention_r2"]) for row in rows])
    high_drop = np.array([float(row["high_random_coord_r2_drop"]) for row in rows])
    low_drop = np.array([float(row["low_random_coord_r2_drop"]) for row in rows])
    high_axis_drop = np.array([float(row["high_axis_mean_r2_drop"]) for row in rows])
    low_axis_drop = np.array([float(row["low_axis_mean_r2_drop"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), constrained_layout=True)
    axes[0].scatter(high_r2, low_r2, s=34, color="#4C78A8", alpha=0.86)
    axes[0].plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    for label, x_val, y_val in zip(labels, high_r2, low_r2):
        axes[0].text(x_val + 0.012, y_val + 0.012, label.replace("\n", " "), fontsize=5.8)
    axes[0].set_title("High/low noise head stability")
    axes[0].set_xlabel("High-noise attention R2")
    axes[0].set_ylabel("Low-noise attention R2")
    axes[0].set_xlim(-0.03, 0.98)
    axes[0].set_ylim(-0.03, 0.98)
    axes[0].grid(True, color="#e6e6e6", lw=0.6)
    axes[0].text(
        0.02,
        0.93,
        f"Pearson={float(summary['pearson_high_low_attention_r2']):.2f}",
        fontsize=7,
        transform=axes[0].transAxes,
    )

    x = np.arange(len(rows))
    width = 0.36
    axes[1].bar(x - width / 2, high_r2, width=width, color="#4C78A8", alpha=0.78, label="High noise")
    axes[1].bar(x + width / 2, low_r2, width=width, color="#F58518", alpha=0.78, label="Low noise")
    axes[1].set_title("Per-head cyclic R2")
    axes[1].set_ylabel("Attention R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[1].legend(frameon=False, fontsize=7)

    drop_labels = ["Random\ncoord", "Axis\nmean"]
    drop_vals = [
        [float(high_drop.mean()), float(high_axis_drop.mean())],
        [float(low_drop.mean()), float(low_axis_drop.mean())],
    ]
    dx = np.arange(len(drop_labels))
    axes[2].bar(dx - width / 2, drop_vals[0], width=width, color="#4C78A8", alpha=0.78, label="High noise")
    axes[2].bar(dx + width / 2, drop_vals[1], width=width, color="#F58518", alpha=0.78, label="Low noise")
    for branch_idx, vals in enumerate(drop_vals):
        offset = (branch_idx - 0.5) * width
        for xi, val in zip(dx + offset, vals):
            axes[2].text(xi, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)
    axes[2].set_title("Coordinate perturbation drop")
    axes[2].set_ylabel("R2 drop from true F-H-W")
    axes[2].set_xticks(dx)
    axes[2].set_xticklabels(drop_labels)
    axes[2].set_ylim(0, 0.62)
    axes[2].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[2].legend(frameon=False, fontsize=7)

    fig.text(
        0.5,
        -0.05,
        "Overlap: layers 0/8, heads 0/10/20/30, patch grid 2x15x26. "
        "The cyclic signal is head-dependent and stronger at low noise, but random-coordinate destruction persists across noise branches.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(r)", weight="bold", fontsize=9)
    save(fig, "fig18_wan_noise_branch_stability")


def figure_sink_noop_correlation() -> None:
    data = json.loads((LOG_DIR / "sink_noop_correlation_20260708.json").read_text(encoding="utf-8"))
    corr = data["correlations"]
    quartiles = data["quartiles"]
    families = ["vit", "qwen3vl_visual"]
    family_labels = {"vit": "ViT", "qwen3vl_visual": "Qwen3-VL"}
    colors = {"vit": "#4C78A8", "qwen3vl_visual": "#F58518"}
    corr_pairs = [
        ("sink_vs_entropy", "sink vs\nentropy"),
        ("sink_vs_drop_sink_error", "sink vs\ndrop-sink"),
        ("sink_vs_sink_component_norm", "sink vs\nsink output"),
        ("true_v_vs_random_v_union_error", "true vs random\nunion error"),
    ]
    corr_lookup = {(row["family"], row["pair"]): float(row["pearson"]) for row in corr}
    quart_lookup = {(row["family"], row["bucket"]): row for row in quartiles}

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), constrained_layout=True)
    x = np.arange(len(corr_pairs))
    width = 0.34
    for idx, family in enumerate(families):
        vals = [corr_lookup[(family, pair)] for pair, _label in corr_pairs]
        offset = (idx - 0.5) * width
        axes[0].bar(x + offset, vals, width=width, color=colors[family], alpha=0.8, label=family_labels[family])
        for xi, val in zip(x + offset, vals):
            axes[0].text(xi, val + (0.035 if val >= 0 else -0.08), f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=6.3)
    axes[0].axhline(0, color="#777777", lw=0.7)
    axes[0].set_title("Sink-strength correlations")
    axes[0].set_ylabel("Pearson correlation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for _pair, label in corr_pairs])
    axes[0].set_ylim(-1.08, 1.08)
    axes[0].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[0].legend(frameon=False, fontsize=7)

    metrics = [
        ("mean_entropy_mean", "Entropy"),
        ("mean_drop_sink2_output_error", "Drop sink"),
        ("mean_drop_union_sink_local_top4_output_error", "Drop union"),
    ]
    x2 = np.arange(len(metrics))
    ratio_max = 0.0
    for idx, family in enumerate(families):
        low = quart_lookup[(family, "low_sink_q1")]
        high = quart_lookup[(family, "high_sink_q4")]
        vals = [float(high[key]) / max(float(low[key]), 1e-12) for key, _label in metrics]
        ratio_max = max(ratio_max, max(vals))
        offset = (idx - 0.5) * width
        axes[1].bar(x2 + offset, vals, width=width, color=colors[family], alpha=0.8, label=family_labels[family])
        for xi, val in zip(x2 + offset, vals):
            axes[1].text(xi, val + 0.06, f"{val:.1f}x", ha="center", va="bottom", fontsize=6.3)
    axes[1].axhline(1, color="#777777", lw=0.7, ls="--")
    axes[1].set_title("High-sink / low-sink quartile ratio")
    axes[1].set_ylabel("Ratio")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([label for _key, label in metrics])
    axes[1].set_ylim(0, max(8.0, ratio_max * 1.16))
    axes[1].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[1].legend(frameon=False, fontsize=7)

    for family in families:
        path = LOG_DIR / f"attention_head_intervention_{'qwen' if family == 'qwen3vl_visual' else 'vit'}_20260707.json"
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        sink = np.array([float(row["top2_col_mass_fraction"]) for row in rows])
        drop = np.array([float(row["drop_sink2_output_error"]) for row in rows])
        axes[2].scatter(sink, drop, s=14, alpha=0.55, color=colors[family], label=family_labels[family], edgecolors="none")
    axes[2].set_title("Dropping sink hurts high-sink heads")
    axes[2].set_xlabel("Top-2 column mass")
    axes[2].set_ylabel("Drop-sink output error")
    axes[2].set_xlim(0, 1.02)
    axes[2].set_ylim(0, 1.65)
    axes[2].grid(True, color="#e6e6e6", lw=0.6)
    axes[2].legend(frameon=False, fontsize=7)

    fig.text(
        0.5,
        -0.05,
        "Correlation probe over saved head-output intervention logs. High sink mass aligns with low entropy and high drop-sink error, "
        "so sinks are functional routes in these heads; Qwen's stronger true/random-V coupling indicates additional value-subspace and dynamic-routing effects.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(s)", weight="bold", fontsize=9)
    save(fig, "fig19_sink_noop_correlation")


def figure_vit_sctm_route_causal() -> None:
    data = json.loads((LOG_DIR / "vit_sctm_route_causal_20260708.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    route_stats = data["route_stats"]
    variants = [row["variant"] for row in rows if row["variant"] != "baseline"]
    lookup = {row["variant"]: row for row in rows}
    label_map = {
        "drop_top1": "drop\ntop-1",
        "drop_top2": "drop\ntop-2",
        "drop_tail1": "drop\nweakest",
        "drop_random1": "drop\nrandom",
        "zero_all": "zero\nroutes",
    }
    labels = []
    for name in variants:
        label = label_map.get(name, name.replace("_", "\n"))
        repeats = int(lookup[name].get("random_repeats", 1))
        if name == "drop_random1" and repeats > 1:
            label = f"{label}\nx{repeats}"
        labels.append(label)
    loss_delta = np.array([float(lookup[name]["loss_delta_vs_baseline"]) for name in variants])
    loss_delta_yerr = np.array([float(lookup[name].get("loss_delta_std_vs_baseline", 0.0)) for name in variants])
    acc_delta = np.array([float(lookup[name]["acc_delta_vs_baseline"]) for name in variants])
    logit_delta = np.array([float(lookup[name]["mean_logit_l2_delta"]) for name in variants])
    prob_drop = np.array([float(lookup[name]["mean_baseline_pred_prob_drop"]) for name in variants])
    flip = np.array([float(lookup[name]["pred_flip_rate"]) for name in variants])

    blocks = [int(row["block"]) for row in route_stats]
    top_patch_mass = np.array([float(row["mean_top_patch_mass"]) for row in route_stats])
    entropy_norm = np.array([float(row["mean_route_entropy_norm"]) for row in route_stats])
    top1_weight = np.array([float(row["mean_top1_selected_weight"]) for row in route_stats])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    x = np.arange(len(variants))
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518", "#54A24B"][: len(variants)]
    bars = axes[0].bar(x, loss_delta, yerr=loss_delta_yerr, capsize=2.5, color=colors, alpha=0.82)
    axes[0].axhline(0, color="#777777", lw=0.7)
    axes[0].set_title("Task loss after route intervention")
    axes[0].set_ylabel("CE loss delta vs baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", color="#e6e6e6", lw=0.6)
    for bar, val in zip(bars, loss_delta):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=6.2)

    width = 0.25
    axes[1].bar(x - width, logit_delta, width=width, color="#4C78A8", alpha=0.82, label="logit L2")
    axes[1].bar(x, prob_drop, width=width, color="#E45756", alpha=0.82, label="baseline-pred prob drop")
    axes[1].bar(x + width, flip, width=width, color="#72B7B2", alpha=0.82, label="prediction flip")
    axes[1].set_title("Logit/probability perturbation")
    axes[1].set_ylabel("Mean per sample")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[1].legend(frameon=False, fontsize=6.7)

    xb = np.arange(len(blocks))
    axes[2].bar(xb - width, top_patch_mass, width=width, color="#756BB1", alpha=0.82, label="top patch mass")
    axes[2].bar(xb, 1.0 - entropy_norm, width=width, color="#F58518", alpha=0.82, label="1 - route entropy")
    axes[2].bar(xb + width, top1_weight, width=width, color="#54A24B", alpha=0.82, label="top-1 route weight")
    axes[2].set_title("Baseline SCTM route concentration")
    axes[2].set_ylabel("Fraction / normalized score")
    axes[2].set_xticks(xb)
    axes[2].set_xticklabels([str(block) for block in blocks])
    axes[2].set_xlabel("Block")
    axes[2].set_ylim(0, max(0.42, float(max(top_patch_mass.max(), (1.0 - entropy_norm).max(), top1_weight.max())) * 1.18))
    axes[2].grid(axis="y", color="#e6e6e6", lw=0.6)
    axes[2].legend(frameon=False, fontsize=6.5)

    baseline = next(row for row in rows if row["variant"] == "baseline")
    summary = data["summary"]
    random_std = float(summary.get("drop_random1_loss_delta_std", 0.0))
    random_repeats = int(summary.get("drop_random1_repeats", 1))
    fig.text(
        0.5,
        -0.06,
        f"Actual saved ViT/SCTM forward path on {int(baseline['samples'])} CIFAR-10 samples. "
        f"Baseline acc={float(baseline['acc']):.3f}, loss={float(baseline['loss']):.3f}. "
        f"Random selected-route control is mean over {random_repeats} seeds (loss-delta std={random_std:.3f}); "
        f"top-1 minus random mean loss delta={float(summary['drop_top1_minus_random1_loss_delta']):.3f}. "
        "This is a task-level intervention on selected SCTM routes, not a dense-attention matrix proxy.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.text(0.01, 0.985, "(t)", weight="bold", fontsize=9)
    save(fig, "fig20_vit_sctm_route_causal")


def main() -> int:
    setup_style()
    rows = load_probe_rows()
    figure_layer_summary(rows)
    figure_head_heatmap(rows)
    figure_attention_examples()
    figure_wan_rowmax_context()
    figure_wan_direct_bccb_probe()
    figure_structured_matrix_fit()
    figure_structured_attention_fit()
    figure_attention_matrix_failure_modes()
    figure_hybrid_attention_decomposition()
    figure_hybrid_attention_tradeoff()
    figure_attention_pattern_full_sweep()
    figure_attention_component_intervention()
    figure_value_subspace_stress()
    figure_head_output_intervention()
    figure_wan_delta_perturbation()
    figure_hybrid_transfer_probe()
    figure_wan_noise_branch_stability()
    figure_sink_noop_correlation()
    figure_vit_sctm_route_causal()
    print(f"Wrote figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
