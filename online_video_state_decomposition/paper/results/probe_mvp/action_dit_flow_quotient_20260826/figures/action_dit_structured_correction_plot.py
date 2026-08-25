from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "action_dit_structured_correction_20260826"
OUTPUT_ROOT = ROOT / "figures"
RUNS = {
    "train-0": "pusht_train0_w4_20260826_v1",
    "train-1": "pusht_train1_w4_20260826_v1",
    "train-2": "pusht_train2_w4_20260826_v1",
}
METHODS = [
    "bucket_mean",
    "channel_affine",
    "circulant_r2",
    "toeplitz_r2",
    "reduced_rank_r4",
    "toeplitz_r2_rank4",
    "dense_ridge_ceiling",
]
LABELS = {
    "bucket_mean": "Bucket mean",
    "channel_affine": "Channel affine",
    "circulant_r2": "Circular r2",
    "toeplitz_r2": "Toeplitz r2",
    "reduced_rank_r4": "Low-rank r4",
    "toeplitz_r2_rank4": "Toeplitz + r4",
    "dense_ridge_ceiling": "Dense ridge",
}
COLORS = ["#0072B2", "#E69F00", "#009E73"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teacher_frames = []
    rollout_frames = []
    bucket_frames = []
    for run_label, run_dir in RUNS.items():
        path = RESULT_ROOT / run_dir
        teacher = pd.read_csv(path / "teacher_forced_metrics.csv")
        teacher["run"] = run_label
        teacher_frames.append(teacher)

        rollout = pd.read_csv(path / "rollout_metrics.csv")
        rollout["run"] = run_label
        baseline = rollout.loc[rollout["method"] == "w4_plain"].iloc[0]
        rollout["mean_improvement_pct"] = 100.0 * (
            1.0 - rollout["action_relative_l2"] / baseline["action_relative_l2"]
        )
        rollout["p95_improvement_pct"] = 100.0 * (
            1.0
            - rollout["p95_sample_relative_l2"]
            / baseline["p95_sample_relative_l2"]
        )
        rollout_frames.append(rollout)

        bucket = pd.read_csv(path / "per_bucket_metrics.csv")
        bucket["run"] = run_label
        baseline_by_bucket = (
            bucket.loc[bucket["method"] == "w4_plain"]
            .set_index("bucket")["denoiser_output_relative_l2"]
        )
        bucket["teacher_improvement_pct"] = [
            100.0 * (1.0 - value / baseline_by_bucket.loc[index])
            for index, value in zip(
                bucket["bucket"], bucket["denoiser_output_relative_l2"]
            )
        ]
        bucket_frames.append(bucket)
    return (
        pd.concat(teacher_frames, ignore_index=True),
        pd.concat(rollout_frames, ignore_index=True),
        pd.concat(bucket_frames, ignore_index=True),
    )


def heatmap(
    axis,
    frame: pd.DataFrame,
    value: str,
    panel: str,
    color_limit: float,
) -> None:
    matrix = (
        frame.loc[frame["method"].isin(METHODS)]
        .pivot(index="method", columns="run", values=value)
        .reindex(METHODS)
        .reindex(columns=RUNS)
    )
    image = axis.imshow(
        matrix.to_numpy(),
        cmap="RdBu",
        vmin=-color_limit,
        vmax=color_limit,
        aspect="auto",
    )
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), [LABELS[item] for item in matrix.index])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value_at_cell = matrix.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value_at_cell:.1f}",
                ha="center",
                va="center",
                color="white" if abs(value_at_cell) > 0.55 * color_limit else "black",
                fontsize=8,
            )
    axis.text(-0.17, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    axis.set_xlabel("independent training checkpoint")
    return image


def plot_evidence(
    teacher: pd.DataFrame, rollout: pd.DataFrame, bucket: pd.DataFrame
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)

    deployable = teacher.loc[teacher["method"].isin(METHODS[:-1])].copy()
    summary = (
        deployable.groupby("method")["captured_defect_energy"]
        .agg(["mean", "min", "max"])
        .reindex(METHODS[:-1])
    )
    positions = np.arange(len(summary))
    axes[0, 0].bar(
        positions,
        100.0 * summary["mean"],
        color="#56B4E9",
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0, 0].errorbar(
        positions,
        100.0 * summary["mean"],
        yerr=np.vstack(
            [
                100.0 * (summary["mean"] - summary["min"]),
                100.0 * (summary["max"] - summary["mean"]),
            ]
        ),
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_xticks(
        positions,
        [LABELS[item] for item in summary.index],
        rotation=24,
        ha="right",
    )
    axes[0, 0].set_ylabel("held-out W4 defect energy removed (%)")
    axes[0, 0].text(-0.12, 1.04, "a", transform=axes[0, 0].transAxes, fontweight="bold")

    mean_image = heatmap(
        axes[0, 1], rollout, "mean_improvement_pct", "b", color_limit=32.0
    )
    axes[0, 1].set_ylabel("complete-sampling correction")
    colorbar = figure.colorbar(mean_image, ax=axes[0, 1], shrink=0.82)
    colorbar.set_label("mean action-error improvement (%)")

    p95_image = heatmap(
        axes[1, 0], rollout, "p95_improvement_pct", "c", color_limit=32.0
    )
    axes[1, 0].set_ylabel("complete-sampling correction")
    colorbar = figure.colorbar(p95_image, ax=axes[1, 0], shrink=0.82)
    colorbar.set_label("P95 action-error improvement (%)")

    selected = bucket.loc[
        bucket["method"].isin(["circulant_r2", "toeplitz_r2", "toeplitz_r2_rank4"])
    ]
    for method, color in zip(
        ["circulant_r2", "toeplitz_r2", "toeplitz_r2_rank4"], COLORS
    ):
        method_rows = selected.loc[selected["method"] == method]
        stats = method_rows.groupby("bucket")["teacher_improvement_pct"].agg(
            ["mean", "min", "max"]
        )
        axes[1, 1].plot(
            stats.index,
            stats["mean"],
            marker="o",
            color=color,
            label=LABELS[method],
        )
        axes[1, 1].fill_between(
            stats.index,
            stats["min"],
            stats["max"],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("denoising-step bucket (early to late)")
    axes[1, 1].set_ylabel("teacher-forced error improvement (%)")
    axes[1, 1].legend(frameon=False, ncol=1)
    axes[1, 1].text(-0.12, 1.04, "d", transform=axes[1, 1].transAxes, fontweight="bold")

    summary.reset_index().to_csv(
        OUTPUT_ROOT / "action_dit_teacher_aggregate.csv", index=False
    )
    rollout.to_csv(OUTPUT_ROOT / "action_dit_rollout_improvement.csv", index=False)
    bucket.to_csv(OUTPUT_ROOT / "action_dit_bucket_improvement.csv", index=False)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUTPUT_ROOT / f"action_dit_structured_correction.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_workload_profile() -> None:
    profile = pd.read_json(
        RESULT_ROOT / RUNS["train-0"] / "model_profile.json", typ="series"
    )
    attention_score = float(profile["attention_score_mac_fraction"])
    selected_ffn = float(profile["selected_ffn_mac_fraction"])
    other = 1.0 - attention_score - selected_ffn
    bf16 = float(profile["bf16_parameter_bytes"])
    selective_w4 = float(profile["selective_w4_parameter_bytes"])

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    left = 0.0
    for label, value, color in [
        ("attention score", attention_score, "#CC79A7"),
        ("selected FFN", selected_ffn, "#0072B2"),
        ("other projections", other, "#BDBDBD"),
    ]:
        axes[0].barh([0], [100.0 * value], left=[100.0 * left], color=color, label=label)
        left += value
    axes[0].set_xlim(0, 100)
    axes[0].set_yticks([])
    axes[0].set_xlabel("estimated denoiser MAC share (%)")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=3)
    axes[0].text(-0.08, 1.08, "a", transform=axes[0].transAxes, fontweight="bold")

    storage = np.array([bf16, selective_w4]) / (1024.0 * 1024.0)
    axes[1].bar([0, 1], storage, color=["#999999", "#009E73"], width=0.62)
    axes[1].set_xticks([0, 1], ["BF16", "selective W4"])
    axes[1].set_ylabel("parameter storage (MiB)")
    axes[1].text(-0.12, 1.08, "b", transform=axes[1].transAxes, fontweight="bold")

    pd.DataFrame(
        {
            "component": ["attention_score", "selected_ffn", "other"],
            "mac_fraction": [attention_score, selected_ffn, other],
        }
    ).to_csv(OUTPUT_ROOT / "action_dit_workload_profile.csv", index=False)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUTPUT_ROOT / f"action_dit_workload_profile.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    configure_style()
    teacher, rollout, bucket = load_tables()
    plot_evidence(teacher, rollout, bucket)
    plot_workload_profile()


if __name__ == "__main__":
    main()
