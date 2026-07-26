#!/usr/bin/env python3
"""Build a source-bound DiT acceleration frontier from existing Wan evidence.

The analysis intentionally separates exact parallelism, approximate kernels,
step reduction, and parallel-in-time methods.  This prevents a distributional
"exactness" claim from being confused with paired trajectory fidelity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "ink": "#17222b",
    "paper": "#f4f0e6",
    "grid": "#c8c1b2",
    "blue": "#16697a",
    "green": "#4f772d",
    "gold": "#d08c32",
    "red": "#b33a3a",
    "slate": "#64727d",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "acceleration_frontier_v1",
    )
    parser.add_argument(
        "--verification-benchmark",
        type=Path,
        help="Optional full-Wan H200 batch-verification CSV produced by benchmark_wan_target_batch.py.",
    )
    parser.add_argument(
        "--cfg-summary",
        type=Path,
        help="Optional paired CFG summary JSON produced by summarize_cfg_parallel.py.",
    )
    return parser.parse_args()


def amdahl_speedup(fraction: float, component_speedup: float) -> float:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    if component_speedup <= 0.0:
        raise ValueError("component_speedup must be positive")
    return 1.0 / ((1.0 - fraction) + fraction / component_speedup)


def fit_step_cost(nfe: pd.DataFrame) -> tuple[float, float, float]:
    """Return fixed seconds, per-step seconds, and R^2 for measured generation."""

    steps = nfe["sampling_steps"].to_numpy(dtype=float)
    seconds = nfe["seconds_including_text_and_vae"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(steps, seconds, deg=1)
    prediction = intercept + slope * steps
    residual = np.square(seconds - prediction).sum()
    total = np.square(seconds - seconds.mean()).sum()
    r2 = 1.0 - residual / total if total > 0 else 1.0
    return float(intercept), float(slope), float(r2)


def cfg_parallel_speedup(
    fixed_seconds: float,
    per_step_seconds: float,
    steps: int,
    communication_fraction: float = 0.0,
) -> float:
    """Build an optimistic two-way CFG envelope from a sequential step fit.

    ``communication_fraction`` is relative to the original per-step time.  The
    envelope keeps text/VAE/fixed work serial and assumes the entire fitted
    step slope consists of two equal model branches.  Scheduler and Python
    serial work make this optimistic until branch-level timing is measured.
    """

    baseline = fixed_seconds + steps * per_step_seconds
    parallel_step = per_step_seconds * (0.5 + communication_fraction)
    return baseline / (fixed_seconds + steps * parallel_step)


def expected_speculative_progress(accept_probability: float, block: int) -> float:
    """Expected progress after verification with exact correction on rejection.

    A rejected candidate still advances the chain by one residual sample.  If
    all candidates are accepted, the chain advances by ``block`` steps.  The
    independent acceptance model is only an economics approximation.
    """

    if not 0.0 <= accept_probability <= 1.0:
        raise ValueError("accept_probability must lie in [0, 1]")
    if block <= 0:
        raise ValueError("block must be positive")
    return sum(accept_probability**index for index in range(block))


def speculative_speedup(
    accept_probability: float,
    block: int,
    verification_ratio: float,
    draft_ratio: float = 0.0,
) -> float:
    """Hardware-aware speedup model for one speculative block.

    Ratios are relative to one target-model evaluation.  With zero draft cost
    this is an upper-bound screen; nonzero draft cost tightens it.
    """

    cost = verification_ratio + block * draft_ratio
    if cost <= 0.0:
        raise ValueError("speculative block cost must be positive")
    return expected_speculative_progress(accept_probability, block) / cost


def picard_speedup(
    sequential_steps: int,
    window: int,
    devices: int,
    iterations_per_window: float,
) -> float:
    """Ideal parallel-in-time speedup before communication and CFG costs."""

    if min(sequential_steps, window, devices) <= 0 or iterations_per_window <= 0:
        raise ValueError("Picard parameters must be positive")
    windows = math.ceil(sequential_steps / window)
    waves_per_iteration = math.ceil(window / devices)
    parallel_rounds = windows * waves_per_iteration * iterations_per_window
    return sequential_steps / parallel_rounds


def read_inputs(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "profile": pd.read_csv(
            root
            / "results"
            / "attention_lowrank_audit_v1"
            / "data"
            / "profile_component_shares.csv"
        ),
        "attention_f17": pd.read_csv(
            root
            / "results"
            / "attention_lowrank_audit_v1"
            / "raw"
            / "attention_f17.csv"
        ),
        "attention_f81": pd.read_csv(
            root
            / "results"
            / "attention_lowrank_audit_v1"
            / "raw"
            / "attention_f81.csv"
        ),
        "nfe": pd.read_csv(root / "figures" / "nfe_v1" / "nfe_sweep_summary.csv"),
        "mp": pd.read_csv(
            root
            / "results"
            / "entropy_structure_audit_v1"
            / "raw"
            / "weight_mp_outliers_wan_ffn6_v1"
            / "weight_mp_summary.csv"
        ),
        "mixed": pd.read_csv(
            root
            / "results"
            / "entropy_structure_audit_v1"
            / "raw"
            / "weight_mp_outliers_wan_ffn6_v1"
            / "weight_mixed_bit.csv"
        ),
        "activation_mixed": pd.read_csv(
            root
            / "results"
            / "entropy_structure_audit_v1"
            / "raw"
            / "weight_split_ffn_activation_v1"
            / "weight_split_activation_metrics.csv"
        ),
        "defect": pd.read_csv(
            root
            / "results"
            / "tri_mode_oracle_v1"
            / "activation_defect"
            / "defect_spectrum_summary.csv"
        ),
    }


def attention_row(frame: str, data: pd.DataFrame, profile: pd.DataFrame) -> list[dict[str, object]]:
    frame_profile = profile[profile["case"] == frame]
    fraction = float(
        frame_profile.loc[
            frame_profile["component"] == "self_attention_core", "share_percent"
        ].iloc[0]
        / 100.0
    )
    valid = data[(data["status"] == "ok") & data["method"].isin(["fa3_bf16", "fa3_fp8", "sage_sm90_smooth"])]
    baseline = float(valid.loc[valid["method"] == "fa3_bf16", "milliseconds"].iloc[0])
    rows: list[dict[str, object]] = []
    for _, item in valid.iterrows():
        kernel_speed = baseline / float(item["milliseconds"])
        rows.append(
            {
                "case": frame,
                "method": item["method"],
                "self_attention_share": fraction,
                "attention_kernel_speedup": kernel_speed,
                "denoiser_amdahl_speedup": amdahl_speedup(fraction, kernel_speed),
                "attention_output_rel_l2": float(item["output_rel_l2"]),
            }
        )
    for hypothetical in (1.5, 2.0, 3.0, 4.0):
        rows.append(
            {
                "case": frame,
                "method": f"fused_sparse_{hypothetical:g}x_attention",
                "self_attention_share": fraction,
                "attention_kernel_speedup": hypothetical,
                "denoiser_amdahl_speedup": amdahl_speedup(fraction, hypothetical),
                "attention_output_rel_l2": np.nan,
            }
        )
    return rows


def build_frontier(
    data: dict[str, pd.DataFrame], verification: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    fixed, per_step, r2 = fit_step_cost(data["nfe"])
    cfg_rows = []
    for communication in (0.0, 0.025, 0.05, 0.10):
        cfg_rows.append(
            {
                "steps": 20,
                "fixed_seconds": fixed,
                "per_step_seconds": per_step,
                "fit_r2": r2,
                "communication_fraction_of_sequential_step": communication,
                "optimistic_envelope_speedup": cfg_parallel_speedup(
                    fixed, per_step, 20, communication
                ),
            }
        )
    cfg = pd.DataFrame(cfg_rows)

    attention = pd.DataFrame(
        attention_row("F17", data["attention_f17"], data["profile"])
        + attention_row("F81", data["attention_f81"], data["profile"])
    )

    probabilities = np.linspace(0.0, 1.0, 101)
    speculative_rows: list[dict[str, object]] = []
    measured_ratios: dict[tuple[str, int], float] = {}
    if verification is not None and not verification.empty:
        ok = verification[
            (verification["status"] == "ok")
            & (verification["operation"] == "full_wan_model")
        ]
        for case, group in ok.groupby("case"):
            base = group[group["batch"] == 1]
            if base.empty:
                continue
            base_ms = float(base["latency_ms_median"].iloc[0])
            for _, row in group.iterrows():
                measured_ratios[(str(case), int(row["batch"]))] = float(
                    row["latency_ms_median"]
                ) / base_ms

    cases = sorted({case for case, _ in measured_ratios}) or ["F17"]
    for case in cases:
        for block in (2, 4, 8):
            ratio = float(block)
            source = "linear_full_model_assumption"
            if (case, block) in measured_ratios:
                ratio = measured_ratios[(case, block)]
                source = f"measured_{case}_full_wan_model_H200"
            for probability in probabilities:
                speculative_rows.append(
                    {
                        "case": case,
                        "block": block,
                        "accept_probability": probability,
                        "verification_ratio_vs_batch1": ratio,
                        "draft_ratio": 0.0,
                        "speedup_upper_bound": speculative_speedup(
                            probability, block, ratio
                        ),
                        "ratio_source": source,
                        "verification_scope": "full Wan target forward only",
                    }
                )
    speculative = pd.DataFrame(speculative_rows)

    picard_rows = []
    for devices in (1, 2, 4, 8, 20):
        for window in (2, 4, 10, 20):
            for iterations in (1, 2, 3, 4, 6, 8):
                picard_rows.append(
                    {
                        "sequential_steps": 20,
                        "devices": devices,
                        "window": window,
                        "iterations_per_window": iterations,
                        "ideal_speedup": picard_speedup(
                            20, window, devices, iterations
                        ),
                    }
                )
    picard = pd.DataFrame(picard_rows)

    original_mp = data["mp"][data["mp"]["variant"] == "original_centered"].copy()
    int4 = data["mixed"][data["mixed"]["method"] == "int4_groupwise"][
        ["weight_key", "gaussian_output_relative_l2"]
    ].rename(columns={"gaussian_output_relative_l2": "int4_gaussian_error"})
    r16 = data["mixed"][
        (data["mixed"]["method"] == "spectral_fp16_plus_int4")
        & (data["mixed"]["rank"] == 16)
    ][["weight_key", "gaussian_output_relative_l2"]].rename(
        columns={"gaussian_output_relative_l2": "r16_int4_gaussian_error"}
    )
    rmt = original_mp.merge(int4, on="weight_key").merge(r16, on="weight_key")
    rmt["r16_absolute_rescue"] = (
        rmt["int4_gaussian_error"] - rmt["r16_int4_gaussian_error"]
    )
    rmt["r16_relative_rescue"] = rmt["r16_absolute_rescue"] / rmt[
        "int4_gaussian_error"
    ]

    activation = data["activation_mixed"]
    base = activation[activation["method"] == "int4_groupwise"][
        ["weight_key", "signal", "activation_output_relative_l2"]
    ].rename(columns={"activation_output_relative_l2": "int4_error"})
    spectral = activation[
        (activation["method"] == "spectral_fp16_plus_int4")
        & (activation["rank"] == 16)
    ][["weight_key", "signal", "activation_output_relative_l2"]].rename(
        columns={"activation_output_relative_l2": "r16_int4_error"}
    )
    activation_rescue = base.merge(spectral, on=["weight_key", "signal"])
    activation_rescue["absolute_rescue"] = (
        activation_rescue["int4_error"] - activation_rescue["r16_int4_error"]
    )
    activation_rescue["relative_rescue"] = activation_rescue[
        "absolute_rescue"
    ] / activation_rescue["int4_error"]

    return {
        "cfg": cfg,
        "attention": attention,
        "speculative": speculative,
        "picard": picard,
        "rmt": rmt,
        "activation_rescue": activation_rescue,
    }


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["paper"])
    axis.grid(True, color=COLORS["grid"], alpha=0.45, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_dashboard(
    data: dict[str, pd.DataFrame], frontier: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(17.0, 13.0))
    fig.patch.set_facecolor(COLORS["paper"])
    for axis in axes.flat:
        style_axis(axis)

    profile = data["profile"].pivot(index="case", columns="component", values="share_percent")
    order = [
        "self_attention_core",
        "linear_gemm",
        "elementwise_memory",
        "normalization",
        "cross_attention_core",
        "other",
    ]
    bottom = np.zeros(len(profile))
    for component, color in zip(
        order,
        [COLORS["blue"], COLORS["gold"], COLORS["slate"], "#8f6f56", COLORS["green"], "#a9a39a"],
    ):
        values = profile[component].to_numpy()
        axes[0, 0].bar(profile.index, values, bottom=bottom, label=component.replace("_", " "), color=color)
        bottom += values
    axes[0, 0].set_title("A  Runtime bottleneck shifts with video length", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("profile share (%)")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    profile.reset_index().to_csv(output_dir / "panel_a_runtime_profile.csv", index=False)

    cfg = frontier["cfg"]
    axes[0, 1].plot(
        cfg["communication_fraction_of_sequential_step"] * 100,
        cfg["optimistic_envelope_speedup"],
        marker="o",
        color=COLORS["green"],
        linewidth=2.2,
    )
    axes[0, 1].axhline(1.0, color=COLORS["ink"], linewidth=0.9)
    if "measured_f17_speedup" in cfg and cfg["measured_f17_speedup"].notna().any():
        measured = float(cfg["measured_f17_speedup"].dropna().iloc[0])
        axes[0, 1].axhline(
            measured,
            color=COLORS["red"],
            linestyle="--",
            linewidth=1.5,
            label=f"measured F17: {measured:.3f}x",
        )
        axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].set_title("B  Exact CFG optimistic envelope", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("communication cost / sequential step (%)")
    axes[0, 1].set_ylabel("predicted end-to-end speedup")
    cfg.to_csv(output_dir / "panel_b_cfg_parallel.csv", index=False)

    nfe = data["nfe"].sort_values("sampling_steps")
    scatter = axes[0, 2].scatter(
        nfe["speedup_vs_20step"], nfe["frame_ssim_mean"],
        c=nfe["sampling_steps"], cmap="viridis", s=75, edgecolor="white", linewidth=0.7,
    )
    axes[0, 2].axhline(0.98, color=COLORS["red"], linestyle="--", label="strict SSIM gate")
    axes[0, 2].set_title("C  Naive NFE reduction fails fidelity", loc="left", fontweight="bold")
    axes[0, 2].set_xlabel("measured end-to-end speedup")
    axes[0, 2].set_ylabel("paired frame SSIM")
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.colorbar(scatter, ax=axes[0, 2], label="sampling steps", fraction=0.05)
    nfe.to_csv(output_dir / "panel_c_nfe_quality.csv", index=False)

    speculative = frontier["speculative"]
    for (case, block), group in speculative.groupby(["case", "block"]):
        axes[1, 0].plot(
            group["accept_probability"], group["speedup_upper_bound"],
            label=f"{case}, draft block {block}", linewidth=2,
        )
    axes[1, 0].axhline(1.0, color=COLORS["ink"], linewidth=0.9)
    axes[1, 0].set_title("D  Speculation needs sublinear verification", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("per-step acceptance probability")
    axes[1, 0].set_ylabel("zero-draft-cost speedup upper bound")
    axes[1, 0].legend(frameon=False, fontsize=8)
    speculative.to_csv(output_dir / "panel_d_speculative_economics.csv", index=False)

    picard = frontier["picard"]
    view = picard[(picard["window"] == 20) & (picard["devices"].isin([2, 4, 8, 20]))]
    for devices, group in view.groupby("devices"):
        axes[1, 1].plot(
            group["iterations_per_window"], group["ideal_speedup"],
            marker="o", label=f"{devices} GPUs",
        )
    axes[1, 1].axhline(1.0, color=COLORS["ink"], linewidth=0.9)
    axes[1, 1].set_title("E  Picard full-window hardware bound", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("fixed-point iterations")
    axes[1, 1].set_ylabel("ideal speedup, T=20")
    axes[1, 1].legend(frameon=False, fontsize=8)
    picard.to_csv(output_dir / "panel_e_picard_hardware_bound.csv", index=False)

    attention = frontier["attention"]
    measured = attention[~attention["method"].str.startswith("fused_sparse")]
    x = np.arange(len(measured))
    axes[1, 2].bar(
        x, measured["denoiser_amdahl_speedup"],
        color=[COLORS["blue"] if case == "F81" else COLORS["gold"] for case in measured["case"]],
    )
    axes[1, 2].set_xticks(x, [f"{c}\n{m.replace('fa3_', '')}" for c, m in zip(measured["case"], measured["method"])], rotation=25)
    axes[1, 2].axhline(1.0, color=COLORS["ink"], linewidth=0.9)
    axes[1, 2].set_title("F  Kernel gain is Amdahl-limited", loc="left", fontweight="bold")
    axes[1, 2].set_ylabel("denoiser speedup")
    attention.to_csv(output_dir / "panel_f_attention_amdahl.csv", index=False)

    rmt = frontier["rmt"]
    axes[2, 0].scatter(
        rmt["above_mp_upper_5pct"], rmt["r16_relative_rescue"] * 100,
        s=65, color=COLORS["blue"], edgecolor="white",
    )
    for _, row in rmt.iterrows():
        label = row["weight_key"].replace("blocks.", "b").replace(".ffn.", "/f")
        axes[2, 0].annotate(label, (row["above_mp_upper_5pct"], row["r16_relative_rescue"] * 100), fontsize=6)
    axes[2, 0].set_title("G  MP spikes are signal, not a speed guarantee", loc="left", fontweight="bold")
    axes[2, 0].set_xlabel("eigenvalues above MP edge +5%")
    axes[2, 0].set_ylabel("rank-16 rescue of INT4 error (%)")
    rmt.to_csv(output_dir / "panel_g_rmt_function.csv", index=False)

    rescue = frontier["activation_rescue"].sort_values("int4_error")
    axes[2, 1].scatter(
        rescue["int4_error"] * 100,
        rescue["r16_int4_error"] * 100,
        c=rescue["relative_rescue"] * 100,
        cmap="YlGnBu",
        s=70,
        edgecolor="white",
    )
    maximum = float(max(rescue["int4_error"].max(), rescue["r16_int4_error"].max()) * 100)
    axes[2, 1].plot([0, maximum], [0, maximum], linestyle="--", color=COLORS["ink"], linewidth=0.9)
    axes[2, 1].set_title("H  Spectral tail helps but remains lossy", loc="left", fontweight="bold")
    axes[2, 1].set_xlabel("group-INT4 activation output error (%)")
    axes[2, 1].set_ylabel("rank-16 + INT4 error (%)")
    rescue.to_csv(output_dir / "panel_h_activation_rescue.csv", index=False)

    decision = pd.DataFrame(
        [
            ("2-GPU CFG branch parallel", 5, 5, "RUN"),
            ("F81 fused sparse attention", 4, 4, "RUN"),
            ("F17 whole-block fusion", 5, 3, "RUN"),
            ("parallel-in-time on 2 GPUs", 5, 1, "STOP unless oracle wins"),
            ("continuous exact speculation", 3, 1, "distribution-exact only"),
            ("RMT-guided mixed precision", 3, 2, "CALIBRATION"),
            ("uniform low-rank / BCM", 1, 1, "STOP"),
        ],
        columns=["direction", "fidelity_score", "speed_potential_score", "decision"],
    )
    decision["score_source"] = "qualitative prior; not a measured metric"
    colors = decision["decision"].map(
        {"RUN": COLORS["green"], "CALIBRATION": COLORS["gold"], "STOP": COLORS["red"], "STOP unless oracle wins": COLORS["red"], "distribution-exact only": COLORS["slate"]}
    )
    axes[2, 2].scatter(
        decision["speed_potential_score"], decision["fidelity_score"],
        s=110, c=colors, edgecolor="white",
    )
    label_offsets = {
        "continuous exact speculation": (5, -11),
        "RMT-guided mixed precision": (5, 7),
        "parallel-in-time on 2 GPUs": (5, 5),
        "uniform low-rank / BCM": (5, -10),
    }
    for _, row in decision.iterrows():
        axes[2, 2].annotate(
            row["direction"],
            (row["speed_potential_score"], row["fidelity_score"]),
            xytext=label_offsets.get(row["direction"], (5, 4)),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[2, 2].set_xlim(0.5, 5.7)
    axes[2, 2].set_ylim(0.5, 5.7)
    axes[2, 2].set_title("I  Qualitative go / no-go prior", loc="left", fontweight="bold")
    axes[2, 2].set_xlabel("credible speed potential")
    axes[2, 2].set_ylabel("paired-fidelity confidence")
    decision.to_csv(output_dir / "panel_i_decision_map.csv", index=False)

    fig.suptitle(
        "Wan2.1 / H200 acceleration frontier under near-lossless constraints",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.1, w_pad=1.5)
    fig.savefig(output_dir / "acceleration_frontier_dashboard.png", dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(output_dir / "acceleration_frontier_dashboard.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_inputs(args.root)
    verification = None
    if args.verification_benchmark is not None and args.verification_benchmark.exists():
        verification = pd.read_csv(args.verification_benchmark)
    frontier = build_frontier(data, verification)
    if args.cfg_summary is not None and args.cfg_summary.exists():
        summary = json.loads(args.cfg_summary.read_text(encoding="utf-8"))
        frontier["cfg"]["measured_f17_speedup"] = float(summary["speedup_mean"])
        frontier["cfg"]["measured_f17_pairs"] = int(summary["pairs"])
        frontier["cfg"]["measured_f17_latent_relative_l2_max"] = float(
            summary["latent_relative_l2_max"]
        )
        frontier["cfg"]["measured_f17_frame_ssim_min"] = float(
            summary["frame_ssim_min"]
        )
    for name, frame in frontier.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False, lineterminator="\n")
    plot_dashboard(data, frontier, args.output_dir)
    print(f"[frontier] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
