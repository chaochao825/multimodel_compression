# World Foundry Hybrid Residual Probe

This directory contains the implementation and evidence for a training-free
Wan2.1-T2V-1.3B acceleration probe on NVIDIA H200 NVL:

```text
W_hat = Q_fp8(W) + U V^T + P_Omega S
```

The runtime combines a World Foundry FP8 main path, a BF16 low-rank residual,
optional static contiguous output-row blocks, step-aware dense refresh, and
TeaCache. The experiment replaces all 30 Wan FFN `up/down` pairs while keeping
FA3 BF16 self-attention and dense cross-attention fixed.

## Main Finding

The implementation is functional, but the original rank-8 plus row-block
residual is not a useful H200 Pareto point.

| Method | Mean SSIM vs dense | Min SSIM | Paired speedup |
| --- | ---: | ---: | ---: |
| Dense + TeaCache 0.08 | 0.95512 | 0.93286 | 1.038x to 1.052x |
| Full FFN FP8 | 0.79663 | 0.64155 | 1.017x to 1.030x |
| FP8 middle-1 | 0.95418 | 0.92918 | 0.998x |
| Rank-8 + row sparse middle-1 | 0.94827 | 0.89837 | 0.990x |
| Rank-16 middle-1 | 0.95428 | 0.93005 | 0.994x |
| Rank-16 + TeaCache 0.08 | 0.94830 | 0.91570 | 1.035x |

The main run covers 4 prompts, 2 seeds, 8 methods, and 64 successful F17
videos. A second 48-video run validates the rank-16 revision. Rank-16 improves
over rank-8 plus sparse by `0.00601` SSIM on average, but is statistically tied
with FP8 middle-1 and remains dominated by dense plus TeaCache.

The pre-registered non-cache quality gate was SSIM `>= 0.98`; neither residual
configuration passed, so the hybrid F81 confirmation was intentionally not
run. Existing F81 TeaCache evidence is retained in the reports for context.

## Trajectory-Budgeted Tri-Mode Follow-up

The follow-up reframes quantization and caching as mutually exclusive actions
at each block, diffusion step, and CFG branch:

```text
D = BF16 dense anchor
Q = FP8 FFN recompute
C = residual cache reuse or first-order forecast
```

The H200 audit covers 384 generated videos or dense-trajectory audits. No
tested Q/C granularity clears the preregistered `SSIM >= 0.98` gate. Serial
global anchors reach `1.051x / 0.67689 SSIM` for all-Q and
`3.843x / 0.20267 SSIM` for all-C. The best single forecast action reaches
`0.975997` on the screening sample; across 2 prompts x 2 seeds its mean/minimum
SSIM is `0.97351/0.97085`, with only `1.005x` geometric-mean speedup (below the
video-level timing noise floor).

Activation-defect spectra also reject a uniform rank-8/16 correction: global
rank-16 explained energy is `34.2%` for Q and `76.1%` for first-order cache
forecast. Only the probed late block 24 is strongly low-dimensional (`91.4%`),
so any retained correction should be layer-specific and fused rather than a
global eager residual path. See
[`results/tri_mode_oracle_v1/TRI_MODE_ORACLE_REPORT.zh-CN.md`](results/tri_mode_oracle_v1/TRI_MODE_ORACLE_REPORT.zh-CN.md)
and the source-bound dashboard in the same directory.

## Layout

- `scripts/worldfoundry_hybrid_residual.py`: switchable dense/FP8/hybrid linear.
- `scripts/trajectory_budget_runtime.py`: mutually exclusive D/Q/C block runtime.
- `scripts/generate_wan_tri_mode_oracle.py`: paired tri-mode rollout runner.
- `scripts/activation_defect_runtime.py`: dense-preserving Q/C defect recorder.
- `scripts/search_tri_mode_oracle.py`: measured-cost conservative schedule search.
- `scripts/generate_wan_hybrid_residual.py`: paired Wan generation runner.
- `scripts/summarize_hybrid_residual.py`: prompt/seed paired video analysis.
- `scripts/run_ffn_*.sh`: H200 pilot, schedule, component, and multi-seed runs.
- `scripts/generate_wan_h200_v4.py`: shared Wan/TeaCache/attention utilities.
- `scripts/compare_paired_videos.py`: decoded video SSIM/PSNR metrics.
- `figures/`: plotting scripts and prior decomposition/system figures.
- `results/h200_live/hybrid_worldfoundry_report.md`: consolidated final report.
- `results/h200_live/figures/`: publication PNG/PDF plus source-bound CSV files.
- `results/tri_mode_oracle_v1/`: compact tri-mode report, CSV evidence, and figures.
- `results/`: prior matrix, activation, H200, NFE, and TeaCache evidence.

Generated MP4 files, model weights, external repositories, checkpoints, and
machine caches are deliberately excluded. Manifests preserve prompts, seeds,
versions, checkpoint hashes, and exact experiment arguments.

## Reproduce Analysis

Install analysis dependencies in an isolated environment:

```bash
python -m pip install -r requirements-analysis.txt
python figures/worldfoundry_hybrid_results_plot.py
python scripts/plot_tri_mode_evidence_dashboard.py \
  --results-root results/tri_mode_oracle_v1 \
  --out-dir results/tri_mode_oracle_v1
```

The contact-sheet script additionally expects the selected MP4 files at the
paths documented in the final report; those videos are not committed.

## Reproduce H200 Generation

The shell runners record the original lab paths. For another machine, update
`BASE_ROOT`, `PROBE_ROOT`, checkpoint paths, and `PYTHONPATH`, then run for
example:

```bash
bash scripts/run_ffn_hybrid_f17_multiseed_v2.sh
bash scripts/run_ffn_residual_component_probe_v1.sh lr16 3 16 16
bash scripts/run_ffn_rank16_f17_multiseed_v1.sh
```

Required external runtime components are Wan2.1-T2V-1.3B, World Foundry,
FA3 for Hopper, and a CUDA/PyTorch build with FP8 scaled matrix multiplication.
The code retains a dense fallback for paired switching, so measured peak memory
is not a deployment-minimal footprint.

## Interpretation

Top-energy row blocks are conditionally optimal after fixing the low-rank term
under a Frobenius objective, but finite rank plus finite row coverage is not a
complete representation of arbitrary matrices. More importantly, Frobenius
weight error is not aligned with diffusion trajectory sensitivity. A dense
refresh stops adding operator error but does not reset an already-diverged
latent state.

The measured action set does not support a training-free `>=1.2x` turbo under
strict dense-relative SSIM 0.98. The remaining defensible directions are exact
system optimization (fused FP8, CUDA Graph, allocation and synchronization
cleanup), NFE/solver changes, or low-cost trajectory-aware adaptation. Static
row blocks and uniform eager low-rank correction should remain stopped unless a
native fused kernel and new defect evidence change their quality-speed bound.
