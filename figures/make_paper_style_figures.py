#!/usr/bin/env python3
"""Create paper-style visualizations for the video circulant probe."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
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
    print(f"Wrote figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
