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

## F81 Attention And FFN Spectral Audit

The next-stage audit keeps `sparse high-rank critical attention + low-rank
marginal tail + cache-aware refresh` as the F81 research direction, while
making the current evidence boundary explicit: the measured compression is a
post-softmax representation oracle, not yet a deployable sparse attention
kernel. Moving from token top-k to contiguous GPU tiles creates a measurable
quality gap, so coarse routing, shared normalization, and a fused H200 kernel
are required before claiming end-to-end acceleration.

For FFN, static row/column/2D FFT sparsification and a hidden-channel BCM main
path are stopped. Across sampled Wan and Llama weights, low-frequency energy
matches the scalar budget and shuffled/Gaussian controls, while the nearest
circulant projection captures only the random-subspace expectation. On H200,
the input and hidden FP32 rFFT round trips alone cost more than `2x` the full
BF16 FFN latency. F17 therefore moves to whole-segment pointwise/kernel fusion;
a standalone post-GEMM Triton bias/GELU kernel is not sufficient. See
[`results/ffn_attention_audit_v1/FFN_ATTENTION_AUDIT_20260726.zh-CN.md`](results/ffn_attention_audit_v1/FFN_ATTENTION_AUDIT_20260726.zh-CN.md)
and its source-bound dashboard and CSV evidence.

## Function-Aware Entropy And THW Structure Audit

The follow-up separates parameter entropy, activation structure, and operator
fidelity instead of treating hidden-channel FFT failure as evidence that Wan is
incompressible. Six Wan FFN matrices have 124-153 eigenvalues above the
Marchenko-Pastur upper edge, while entry-shuffled and matched-Gaussian controls
have none. These spectral outliers are functionally relevant on held-out FFN
activations, but a BF16 rank-16 tail only reduces group-INT4 local output error
from `7.533%` to `6.891%`, so it does not make training-free INT4 a main path.

The correct geometric axis is the video token grid. At 12.5% true THW
low-frequency density, F81 Q/K retain `91.94%/94.59%` centered energy, but K
low-pass still produces `44.77%` attention-output relative error. Q low-pass
reaches `85.77%` exact top-128 recall and is retained only as a coarse-router
feature. An eager H200 cost gate rejects online FFT: F81 Q/QK round trips cost
`20.97%/41.86%` of FA3 BF16 attention, while spatial pool2 feature extraction
costs only `0.76%/1.42%` before routing. See
[`results/entropy_structure_audit_v1/ENTROPY_STRUCTURE_AUDIT_20260726.zh-CN.md`](results/entropy_structure_audit_v1/ENTROPY_STRUCTURE_AUDIT_20260726.zh-CN.md)
and its nine-panel source-bound dashboard.

## Strict-Fidelity DiT Acceleration Frontier

The latest audit separates same-trajectory exact optimization from
target-distribution-exact speculative sampling and perceptual approximations.
For the available two-H200 setup, exact CFG branch parallelism is the first
priority, followed by fused F81 spatial/temporal sparse attention.  Global
low-rank, static row sparsity, coarse NFE reduction, and two-device Picard
parallelism fail either the fidelity gate or the hardware-economics gate.  See
[`results/acceleration_frontier_v1/DIT_ACCELERATION_FRONTIER_20260726.zh-CN.md`](results/acceleration_frontier_v1/DIT_ACCELERATION_FRONTIER_20260726.zh-CN.md)
and the source-bound dashboard in the same directory.

The completed F17 20-step paired run measures `1.7743x` mean end-to-end
speedup from exact two-H200 CFG branch parallelism, with zero final-latent and
decoded-pixel difference. Full-Wan speculative verification is nearly linear:
batch-2 costs `1.952x` at F17 and `1.990x` at F81. Runtime-defect top-16
subspace overlap is only `0.220-0.356`, rejecting one global low-rank
correction. Raw H200 CSV/JSON evidence and the defect RMT dashboard live in
`results/acceleration_frontier_h200_v1/`.

## Layout

- `scripts/worldfoundry_hybrid_residual.py`: switchable dense/FP8/hybrid linear.
- `scripts/trajectory_budget_runtime.py`: mutually exclusive D/Q/C block runtime.
- `scripts/generate_wan_tri_mode_oracle.py`: paired tri-mode rollout runner.
- `scripts/activation_defect_runtime.py`: dense-preserving Q/C defect recorder.
- `scripts/probe_ffn_spectral.py`: Wan/Llama FFN spectral and circulant controls.
- `scripts/benchmark_h200_ffn_transforms.py`: H200 FFT and pointwise fusion probes.
- `scripts/probe_attention_block_tail_oracle.py`: F81 token-to-tile tail oracle.
- `scripts/probe_token_thw_spectrum.py`: true T x H x W activation spectra and controls.
- `scripts/probe_spectral_qk_router.py`: sampled-query softmax and support-recall gate.
- `scripts/probe_weight_mp_outliers.py`: MP bulk/spike and mixed-bit controls.
- `scripts/probe_wan_ffn_activation_structure.py`: cross-step/CFG FFN recorder.
- `scripts/analyze_ffn_activation_quantization.py`: held-out activation quantization audit.
- `scripts/plot_entropy_structure_audit.py`: source-bound nine-panel dashboard.
- `scripts/analyze_acceleration_frontier.py`: CFG, NFE, speculation, Picard, attention, and RMT frontier.
- `scripts/benchmark_h200_speculative_batch.py`: H200 target batch-verification economics.
- `scripts/benchmark_wan_target_batch.py`: full-denoiser batch verification cost.
- `scripts/probe_defect_rmt.py`: runtime-defect MP/null and cross-run subspace stability.
- `scripts/plot_defect_rmt.py`: source-bound runtime-defect eigenspectrum dashboard.
- `scripts/generate_wan_cfg_parallel.py`: exact two-H200 CFG branch executor.
- `scripts/summarize_cfg_parallel.py`: paired video fidelity and speed summary.
- `scripts/run_acceleration_frontier_h200_v1.sh`: idle-gated H200 experiment sequence.
- `scripts/plot_ffn_attention_audit.py`: source-bound decision dashboard.
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
- `results/ffn_attention_audit_v1/`: FFN/attention audit, raw CSVs, and dashboard.
- `results/entropy_structure_audit_v1/`: THW, MP, activation, function, and H200 audit.
- `results/acceleration_frontier_v1/`: strict-fidelity theory, CSV evidence, and dashboard.
- `results/acceleration_frontier_h200_v1/`: measured target batching, defect RMT, and exact CFG evidence.
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
python scripts/plot_entropy_structure_audit.py \
  --raw-dir results/entropy_structure_audit_v1/raw \
  --output-dir results/entropy_structure_audit_v1/figures
python scripts/analyze_acceleration_frontier.py \
  --verification-benchmark results/acceleration_frontier_h200_v1/full_model_batch/wan_target_batch_benchmark.csv \
  --cfg-summary results/acceleration_frontier_h200_v1/cfg_f17/cfg_parallel_summary.json
python scripts/plot_defect_rmt.py \
  --summary results/acceleration_frontier_h200_v1/defect_rmt/defect_rmt_summary.csv \
  --eigenvalues results/acceleration_frontier_h200_v1/defect_rmt/defect_rmt_eigenvalues.csv \
  --out-dir results/acceleration_frontier_h200_v1/defect_rmt
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
