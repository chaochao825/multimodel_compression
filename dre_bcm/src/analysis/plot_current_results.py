import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "trainable_params",
    "stored_params",
    "estimated_bits_fp16",
    "estimated_bits_int8",
    "estimated_bits_int4_delta",
    "bcm_direct_ops",
    "bcm_fft_ops",
    "lowrank_ops",
    "total_ops",
    "activation_memory",
    "extra_latency_proxy",
    "block_size",
    "rank",
    "num_bcm_basis",
    "sparse_ratio",
    "objective",
    "relative_fro_error",
    "spectral_error",
    "activation_error",
]


def infer_model(run_name: str, layer_key: str) -> str:
    text = f"{run_name} {layer_key}".lower()
    if "bitnet" in text:
        return "BitNet"
    if "llama" in text:
        return "LLaMA"
    if "mistral" in text:
        return "Mistral"
    if "qwen2_5" in text or "qwen2.5" in text:
        return "Qwen2.5"
    if "qwen3" in text:
        return "Qwen3"
    if "bert" in text:
        return "BERT"
    return "Other"


def short_run_label(run_name: str) -> str:
    match = re.match(r"(.+?)_generator_delta_bcm__bs(\d+)__r(\d+)__basis(\d+)", run_name)
    if match:
        base, block_size, rank, basis = match.groups()
        base = base.replace("_retry", "")
        return f"{base} bs{block_size} r{rank} b{basis}"

    label = run_name.replace("_retry", "")
    label = label.replace("_qproj", " q")
    label = label.replace("_downproj", " down")
    label = label.replace("_query", " query")
    label = label.replace("qwen2_5", "qwen2.5")
    label = label.replace("_", " ")
    return label


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def read_summary_files(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("summary_metrics.csv")):
        if path.name != "summary_metrics.csv":
            continue
        df = pd.read_csv(path)
        df["run_name"] = path.parent.name
        df["summary_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = ensure_numeric(merged)
    merged["model_family"] = [
        infer_model(run_name, layer_key)
        for run_name, layer_key in zip(merged["run_name"], merged.get("layer_key", pd.Series(dtype=str)).fillna(""))
    ]
    return merged


def read_best_metrics(root: Path) -> pd.DataFrame:
    summary_path = root / "best_metrics_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df["best_source"] = str(summary_path)
        df = ensure_numeric(df)
        df["run_name"] = df["result_dir"].fillna("").map(lambda value: Path(str(value)).name)
        df["model_family"] = [
            infer_model(run_name, layer_key)
            for run_name, layer_key in zip(df["run_name"], df.get("layer_key", pd.Series(dtype=str)).fillna(""))
        ]
        return df

    rows = []
    for path in sorted(root.rglob("best_metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["run_name"] = path.parent.name
        payload["best_source"] = str(path)
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = ensure_numeric(df)
    df["model_family"] = [
        infer_model(run_name, layer_key)
        for run_name, layer_key in zip(df["run_name"], df.get("layer_key", pd.Series(dtype=str)).fillna(""))
    ]
    return df


def read_generator_stats(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("generator_delta_stats.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "run_name": path.parent.name,
                "generator_entropy": payload.get("generator_entropy"),
                "delta_entropy": payload.get("delta_entropy"),
                "delta_std": payload.get("delta_summary", {}).get("std"),
                "threshold_0.01": payload.get("threshold_sparsity", {}).get("0.01"),
                "int4_mse": payload.get("quantization_mse", {}).get("int4"),
                "compressed_bits_entropy_delta": payload.get("estimated_compressed_bits", {}).get("entropy_delta"),
                "compressed_bits_fp16_base_int4_delta": payload.get("estimated_compressed_bits", {}).get(
                    "fp16_base_int4_delta"
                ),
                "source_file": str(path),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return ensure_numeric(df)


def read_residual_spectra(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("residual_spectrum_*.csv")):
        if path.name == "residual_spectrum_by_layer.csv":
            continue
        if "current_overview" in path.parts:
            continue
        df = pd.read_csv(path)
        parent = path.parent.name
        stem = path.stem.replace("residual_spectrum_", "")
        df["spectrum_name"] = f"{parent}:{stem}"
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return ensure_numeric(merged)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def finalize_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rotate_xticks(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", rotation=25)


def plot_matrix_error_vs_params(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df.dropna(subset=["trainable_params", "relative_fro_error", "method"]).copy()
    if subset.empty:
        return
    save_csv(
        subset[
            ["run_name", "model_family", "method", "trainable_params", "relative_fro_error", "stored_params", "rank", "block_size"]
        ],
        output_dir / "matrix_fit_error_vs_params.csv",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method, group in subset.groupby("method"):
        ax.scatter(group["trainable_params"], group["relative_fro_error"], label=method, alpha=0.8, s=36)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Matrix Fit Error vs Trainable Parameters")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    finalize_plot(fig, output_dir / "matrix_fit_error_vs_params.png")


def plot_matrix_error_vs_bits(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df.dropna(subset=["estimated_bits_int4_delta", "relative_fro_error", "method"]).copy()
    if subset.empty:
        return
    save_csv(
        subset[
            ["run_name", "model_family", "method", "estimated_bits_int4_delta", "relative_fro_error", "rank", "block_size"]
        ],
        output_dir / "matrix_fit_error_vs_bits.csv",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method, group in subset.groupby("method"):
        ax.scatter(group["estimated_bits_int4_delta"], group["relative_fro_error"], label=method, alpha=0.8, s=36)
    ax.set_xscale("log")
    ax.set_xlabel("Estimated Bits (int4-delta proxy)")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Matrix Fit Error vs Estimated Bits")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    finalize_plot(fig, output_dir / "matrix_fit_error_vs_bits.png")


def plot_best_error_bar(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df.dropna(subset=["relative_fro_error"]).copy().sort_values("relative_fro_error")
    if subset.empty:
        return
    subset["plot_label"] = subset["run_name"].map(short_run_label)
    save_csv(
        subset[["run_name", "model_family", "method", "relative_fro_error", "rank", "block_size"]],
        output_dir / "best_relative_fro_by_run.csv",
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {"bcm_plus_lowrank": "#1f77b4", "lowrank_svd": "#ff7f0e", "bcm_only": "#2ca02c", "generator_delta_bcm": "#d62728"}
    ax.bar(
        subset["plot_label"],
        subset["relative_fro_error"],
        color=[colors.get(method, "#7f7f7f") for method in subset["method"]],
    )
    for idx, row in subset.reset_index(drop=True).iterrows():
        ax.text(idx, row["relative_fro_error"] + 0.01, row["method"], ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_ylabel("Best Relative Frobenius Error")
    ax.set_title("Best Current Result by Run")
    rotate_xticks(ax)
    ax.grid(True, axis="y", alpha=0.25)
    finalize_plot(fig, output_dir / "best_relative_fro_by_run.png")


def plot_residual_spectrum(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df.dropna(subset=["rank", "energy_ratio"]).copy()
    if subset.empty:
        return
    save_csv(
        subset[["spectrum_name", "rank", "energy_ratio", "block_size", "layer_key", "bcm_relative_fro_error"]],
        output_dir / "residual_spectrum_by_layer.csv",
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for spectrum_name, group in subset.groupby("spectrum_name"):
        group = group.sort_values("rank")
        ax.plot(group["rank"], group["energy_ratio"], marker="o", label=spectrum_name)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Explained Residual Energy")
    ax.set_title("Residual Spectrum by Layer")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    finalize_plot(fig, output_dir / "residual_spectrum_by_layer.png")


def plot_block_size_ablation(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df[df["method"].isin(["bcm_only", "bcm_plus_lowrank", "generator_delta_bcm", "lowrank_svd"])].copy()
    subset = subset.dropna(subset=["block_size", "relative_fro_error"])
    if subset.empty:
        return
    subset = (
        subset.groupby(["run_name", "method", "block_size"], as_index=False)["relative_fro_error"]
        .min()
        .sort_values(["run_name", "method", "block_size"])
    )
    save_csv(subset, output_dir / "block_size_ablation.csv")
    run_names = sorted(subset["run_name"].unique())
    ncols = 2
    nrows = (len(run_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 3.7 * nrows), squeeze=False)
    method_colors = {
        "bcm_only": "#2ca02c",
        "bcm_plus_lowrank": "#1f77b4",
        "generator_delta_bcm": "#d62728",
        "lowrank_svd": "#ff7f0e",
    }
    for ax, run_name in zip(axes.flatten(), run_names):
        run_group = subset[subset["run_name"] == run_name]
        for method, method_group in run_group.groupby("method"):
            ax.plot(
                method_group["block_size"],
                method_group["relative_fro_error"],
                marker="o",
                label=method,
                color=method_colors.get(method),
            )
        ax.set_title(short_run_label(run_name), fontsize=10)
        ax.set_xlabel("Block Size")
        ax.set_ylabel("Best Rel. Fro Error")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes.flatten()[len(run_names) :]:
        ax.axis("off")
    fig.suptitle("Block Size Ablation", fontsize=14)
    finalize_plot(fig, output_dir / "block_size_ablation.png")


def plot_rank_ablation(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df[df["method"].isin(["lowrank_svd", "bcm_plus_lowrank"])].copy()
    subset = subset.dropna(subset=["rank", "relative_fro_error"])
    if subset.empty:
        return
    subset = (
        subset.groupby(["run_name", "method", "rank"], as_index=False)["relative_fro_error"]
        .min()
        .sort_values(["run_name", "method", "rank"])
    )
    save_csv(subset, output_dir / "rank_ablation.csv")
    run_names = sorted(subset["run_name"].unique())
    ncols = 2
    nrows = (len(run_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 3.7 * nrows), squeeze=False)
    method_colors = {
        "bcm_plus_lowrank": "#1f77b4",
        "lowrank_svd": "#ff7f0e",
    }
    for ax, run_name in zip(axes.flatten(), run_names):
        run_group = subset[subset["run_name"] == run_name]
        for method, method_group in run_group.groupby("method"):
            ax.plot(
                method_group["rank"],
                method_group["relative_fro_error"],
                marker="o",
                label=method,
                color=method_colors.get(method),
            )
        ax.set_title(short_run_label(run_name), fontsize=10)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Best Rel. Fro Error")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes.flatten()[len(run_names) :]:
        ax.axis("off")
    fig.suptitle("Rank Ablation", fontsize=14)
    finalize_plot(fig, output_dir / "rank_ablation.png")


def plot_generator_delta_entropy(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df.dropna(subset=["generator_entropy", "delta_entropy"]).copy()
    if subset.empty:
        return
    subset["plot_label"] = subset["run_name"].map(short_run_label)
    save_csv(subset, output_dir / "generator_delta_entropy.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    positions = range(len(subset))
    ax.bar([pos - 0.18 for pos in positions], subset["generator_entropy"], width=0.36, label="generator_entropy")
    ax.bar([pos + 0.18 for pos in positions], subset["delta_entropy"], width=0.36, label="delta_entropy")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(subset["plot_label"])
    rotate_xticks(ax)
    ax.set_ylabel("Empirical Entropy")
    ax.set_title("Generator vs Delta Entropy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    finalize_plot(fig, output_dir / "generator_delta_entropy.png")


def plot_latency_proxy(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df.dropna(subset=["extra_latency_proxy"]).copy().sort_values("extra_latency_proxy")
    if subset.empty:
        return
    subset["plot_label"] = subset["run_name"].map(short_run_label)
    save_csv(
        subset[["run_name", "model_family", "method", "extra_latency_proxy", "total_ops", "stored_params"]],
        output_dir / "method_latency_proxy.csv",
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {"bcm_plus_lowrank": "#1f77b4", "lowrank_svd": "#ff7f0e", "bcm_only": "#2ca02c", "generator_delta_bcm": "#d62728"}
    ax.bar(
        subset["plot_label"],
        subset["extra_latency_proxy"],
        color=[colors.get(method, "#7f7f7f") for method in subset["method"]],
    )
    ax.set_ylabel("Extra Latency Proxy")
    ax.set_title("Method Latency Proxy")
    rotate_xticks(ax)
    ax.grid(True, axis="y", alpha=0.25)
    finalize_plot(fig, output_dir / "method_latency_proxy.png")


def build_plots(matrix_root: Path, peft_root: Path, plots_root: Path, output_dir: Path) -> None:
    all_summary = read_summary_files(matrix_root)
    best_summary = read_best_metrics(matrix_root)
    delta_stats = read_generator_stats(peft_root)
    residual_spectra = read_residual_spectra(plots_root)

    if not all_summary.empty:
        plot_matrix_error_vs_params(all_summary, output_dir)
        plot_matrix_error_vs_bits(all_summary, output_dir)
        plot_block_size_ablation(all_summary, output_dir)
        plot_rank_ablation(all_summary, output_dir)

    if not best_summary.empty:
        plot_best_error_bar(best_summary, output_dir)
        plot_latency_proxy(best_summary, output_dir)

    if not residual_spectra.empty:
        plot_residual_spectrum(residual_spectra, output_dir)

    if not delta_stats.empty:
        plot_generator_delta_entropy(delta_stats, output_dir)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", default="results/matrix_fit")
    parser.add_argument("--peft-root", default="results/peft")
    parser.add_argument("--plots-root", default="results/plots")
    parser.add_argument("--output-dir", default="results/plots/current_overview")
    args = parser.parse_args(list(argv) if argv is not None else None)

    build_plots(
        matrix_root=Path(args.matrix_root),
        peft_root=Path(args.peft_root),
        plots_root=Path(args.plots_root),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
