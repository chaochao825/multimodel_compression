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

The follow-up two-seed screen separates stable head roles from unstable
correction bases. All 12 layer-0/step-0 heads preserve their
localized/transitional/diffuse class across independent seeds, with entropy
and temporal-tile mass correlations above `0.99998`; this is a provisional GO
for studying a head-role-aware adaptive router, not for a fixed token mask.
See [`results/ATTENTION_HEAD_CLASS_STABILITY_F81_20260726.zh-CN.md`](results/ATTENTION_HEAD_CLASS_STABILITY_F81_20260726.zh-CN.md).

For FFN, static row/column/2D FFT sparsification and a hidden-channel BCM main
path are stopped. Across sampled Wan and Llama weights, low-frequency energy
matches the scalar budget and shuffled/Gaussian controls, while the nearest
circulant projection captures only the random-subspace expectation. On H200,
the input and hidden FP32 rFFT round trips alone cost more than `2x` the full
BF16 FFN latency. F17 therefore moves to whole-segment pointwise/kernel fusion;
a standalone post-GEMM Triton bias/GELU kernel is not sufficient. See
[`results/ffn_attention_audit_v1/FFN_ATTENTION_AUDIT_20260726.zh-CN.md`](results/ffn_attention_audit_v1/FFN_ATTENTION_AUDIT_20260726.zh-CN.md)
and its source-bound dashboard and CSV evidence.

The measured H200 fusion ceiling further separates local FFN wins from
end-to-end value. Even removing all measured up-projection bias/GELU overhead
would yield only `1.038x` for F17 and `1.016x` for F81 under the profiled
runtime shares; the existing standalone Triton path is both approximate and
slower when projected onto the complete FFN. See
[`results/FFN_FUSION_CEILING_H200_20260726.zh-CN.md`](results/FFN_FUSION_CEILING_H200_20260726.zh-CN.md).

The exclusive-H200 exact-path gate confirms that generic compilation is not a
substitute for fusion. All three `torch.compile` modes are slower and alter the
BF16 trajectory; static-address CUDA Graph replay is bitwise exact but reaches
only `1.022x` harmonic-mean steady speedup and fails 40-call amortization.
Standalone FFN compile/graph is therefore stopped; only whole-segment graphing
or a real GEMM epilogue remains as an F17 systems candidate. See
[`results/FFN_EXACT_PATHS_H200_20260726.zh-CN.md`](results/FFN_EXACT_PATHS_H200_20260726.zh-CN.md).

## Multi-Block BCM And Joint Error Shaping

A calibration-frozen F81 probe rejects increasingly local fixed BCM attention
tables. Global BCCB, query-block multi-BCM, and hierarchical coarse/tile/local
BCM reach `57.20%`, `54.91%`, and `50.41%` mean held-out attention-output L2;
the hierarchical model uses `36.7x` more parameters per head and still misses
the relaxed `5%` gate by an order of magnitude. It helps selected heads but
does not fix dynamic support or content-dependent value directions.

Joint `Q(W-beta*L-S)+L+S` INT4 shaping is mechanistically real but also fails
the deployment gate. Rank-16 plus 2% blocks raises held-out defect energy from
`25.0%` to `71.1%` at block 0 and from `52.0%` to `89.1%` at block 24, yet
output errors remain `2.555%/8.014%`. Rank-64 plus 5% blocks only reaches
`2.265%/7.112%` while adding about `1.74B` estimated operations. A validation
sweep over `beta={0,.25,.5,.75,1}` does not close the gap. See
[`results/JOINT_ERROR_SHAPING_MULTIBLOCK_BCM_20260726.zh-CN.md`](results/JOINT_ERROR_SHAPING_MULTIBLOCK_BCM_20260726.zh-CN.md).

## Dynamic Sparse Critical And Conditional Tail

The output-aware F81 follow-up finds a real but still oracle-only
representation opportunity. At 12.5% `64x64` key-block density, a
sample-adaptive rank-16 output tail reaches `0.629%` aggregate relative L2 and
`1.84%` worst-head error across all 12 heads. Freezing the calibration mask
barely changes the aggregate error (`0.630%`), while freezing the calibration
basis raises it to `2.68-2.76%` and leaves diffuse head 9 above `10%`.

More router complexity does not close this transfer gap. Tail-aware
alternation improves only `0.07-0.34%` relative, and splitting a static basis
into 2/4/8/16 position banks bottoms out at `2.82%` aggregate error while the
worst head remains above `12%`. The main path is therefore narrowed to a
head-role-aware executor with content-conditioned linear/low-rank marginal
features and dense fallback for Gaussian-like diffuse heads. See
[`results/DYNAMIC_SPARSE_LOWRANK_F81_20260726.zh-CN.md`](results/DYNAMIC_SPARSE_LOWRANK_F81_20260726.zh-CN.md).

A registered follow-up rejects the first train-free content-generated tail.
Segment Nyström/landmark candidates are selected on validation only and then
evaluated under seed, prompt, and prompt-seed-combination holdouts. The best
full-pilot post-hoc diagnostic has `22.684%` aggregate and `57.062%`
worst-record output error despite a `3.821x` arithmetic upper bound; it is not
a frozen test estimate. Validation-frozen test errors remain
`20.804%-23.218%` aggregate, and calibration-frozen transitional heads still
have about `20%` test error. Input captures are
SHA256-pinned and the holdouts are explicitly labeled within-run because the
same captures appeared in earlier exploratory probes. This stops the fixed
landmark family, not learned sparse-linear tails. See
[`results/CONTENT_STRUCTURE_ATTENTION_F81_20260726.zh-CN.md`](results/CONTENT_STRUCTURE_ATTENTION_F81_20260726.zh-CN.md)
and [`docs/nystrom_sparse_tail_protocol.md`](docs/nystrom_sparse_tail_protocol.md).

The final bounded train-free screen tests value-aware K/V/THW coresets,
critical-removed order-1/4 polynomial tails, and rank-4/8/16 K/V covariance
moments under strict shared normalization. Even a per-sample/head post-hoc
envelope reaches only `4.952%-6.719%` aggregate error and
`10.134%-21.938%` worst-record error, versus the registered `0.5%/1%`
oracle gate. The tail score range remains `18.523` after 25% exact critical
selection, and full-covariance Gaussian moments are worse than centroids.
This closes the remaining train-free tail family without authorizing an H200
kernel or rollout; it does not reject the previously observed adaptive
rank-16 output witness. See
[`results/TRAINFREE_RESIDUAL_TAIL_ORACLE_F81_20260728.zh-CN.md`](results/TRAINFREE_RESIDUAL_TAIL_ORACLE_F81_20260728.zh-CN.md).

A registered restricted-rotation oracle then tests whether the moving rank-16
tail can be generated from a source basis with at most 16 Givens,
Householder, orthogonal block-circulant, DCD, or Butterfly factors. Only the
layer-0/step-0 capacity control passes the adaptive `0.5%/1%` pre-gate;
layer 14 fails at steps 0/9/19 even with a fresh per-record SVD. Householder-16
matches the adaptive ceiling but emits 2,048 dynamic scalars, exactly the
payload of a `128x16` basis. The best <=512-scalar candidate, Butterfly-8,
improves to `0.285%` aggregate and `1.082%` worst error after a 600-step,
four-restart refinement and still misses the worst-case gate. Post-hoc dense
fallback for heads 7/9 leaves a `2.66x` ideal layer-0 attention-work upper
bound, but no universal rotation gate or H200 kernel is authorized. See
[`results/RESTRICTED_ROTATION_ORACLE_F81_20260728.zh-CN.md`](results/RESTRICTED_ROTATION_ORACLE_F81_20260728.zh-CN.md).

The completed 72-cell prompt/seed/step/CFG factorial sharpens that executor.
CFG branches have `1.0` class agreement and localized-head Jaccard in all 36
pairs, while step comparisons pass only `52.8%` of the stability gates. Layer
0 is stable across every tested factor; layer 14 requires dynamic content/step
conditioning; layer 29 is sample-stable but step-dependent. This supports
sharing operator classes across CFG, not activations, masks, scales, or frozen
correction bases. See
[`results/ATTENTION_HEAD_FACTORIAL_F81_20260726.zh-CN.md`](results/ATTENTION_HEAD_FACTORIAL_F81_20260726.zh-CN.md).

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
- `scripts/probe_joint_quant_lr_shaping.py`: activation-shaped INT4 plus low-rank/block-sparse probe.
- `scripts/probe_multiblock_bcm_attention.py`: frozen global, query-block, and hierarchical BCM attention probe.
- `scripts/plot_joint_quant_lr_shaping.py`: joint-shaping gate and generalization dashboard.
- `scripts/plot_multiblock_bcm_attention.py`: BCM parameter, head, and wrap-leakage dashboard.
- `scripts/probe_dynamic_sparse_lowrank_oracle.py`: output-aware block-sparse and cross-seed tail transfer probe.
- `scripts/probe_tail_aware_sparse_router.py`: alternating routing on the low-rank-unrepairable residual.
- `scripts/probe_conditional_defect_basis_bank.py`: frozen position-bucketed defect basis-bank audit.
- `scripts/plot_dynamic_sparse_lowrank_oracle.py`: dynamic/static transfer and normalization dashboard.
- `scripts/plot_tail_aware_sparse_router.py`: fixed-objective versus alternating router comparison.
- `scripts/plot_conditional_defect_basis_bank.py`: basis-bank transfer and overfit dashboard.
- `scripts/probe_nystrom_sparse_tail.py`: associative Nyström/landmark capacity probe with capture integrity checks.
- `scripts/select_nystrom_sparse_tail.py`: validation-only selection and frozen-head transfer evaluation.
- `scripts/plot_nystrom_sparse_tail.py`: source-bound validation/test plots without test-driven selection.
- `scripts/analyze_nystrom_sparse_tail_failure.py`: capacity, conditioning, role-transfer, and router-error diagnosis.
- `scripts/trainfree_tail_oracle_core.py`: shared-normalization coreset, polynomial, and covariance kernels.
- `scripts/probe_trainfree_tail_oracles.py`: registered stronger train-free residual-tail capacity screen.
- `scripts/analyze_trainfree_tail_oracles.py`: source-bound capacity and per-head failure dashboards.
- `scripts/restricted_rotation_oracle_core.py`: Givens, Householder, orthogonal BCM, DCD, Butterfly, and cost-model primitives.
- `scripts/probe_restricted_rotation_oracle.py`: SHA256-bound layer/step/head/tile post-hoc rotation-capacity screen.
- `scripts/analyze_restricted_rotation_oracle.py`: cross-holdout, payload, and universal-head-fallback analysis.
- `scripts/experiment_artifacts.py`: atomic run state, provenance, hashing, and rectangular-sweep utilities.
- `scripts/generate_wan_cfg_parallel.py`: exact two-H200 CFG branch executor.
- `scripts/summarize_cfg_parallel.py`: paired video fidelity and speed summary.
- `scripts/run_phase2_head_role_factorial_v1.sh`: resumable F81 prompt/seed/step/CFG head-role capture.
- `scripts/summarize_attention_head_factorial.py`: factor-separated router-stability gates and dashboard.
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
- `results/joint_quant_lr_shaping_cpu_v2/`: rank-8/16 joint-shaping evidence and dashboard.
- `results/joint_quant_lr_shaping_capacity_cpu_v1/`: rank-32/64 capacity ceiling.
- `results/joint_quant_lr_shaping_beta_cpu_v1/`: validation-selected shaping-strength sweep.
- `results/multiblock_bcm_attention_f81_cpu_v1/`: F81 multi-block BCM held-out evidence.
- `results/dynamic_sparse_lowrank_oracle_f81_full_v1/`: output-aware dynamic/static transfer evidence.
- `results/tail_aware_sparse_router_f81_full_v1/`: alternating tail-aware router evidence.
- `results/conditional_defect_basis_bank_f81_full_v1/`: cross-seed position basis-bank evidence.
- `results/attention_head_factorial_f81_v1/summary/`: compact 72-cell factor-separated evidence.
- `results/nystrom_sparse_tail_f81_pilot_v1/`: registered F81 numerical sweep and hashed input manifest.
- `results/nystrom_sparse_tail_f81_pilot_selection_v1/`: validation-frozen all-head holdout decisions.
- `results/nystrom_sparse_tail_f81_pilot_transitional_selection_v1/`: calibration-frozen transitional-head decisions.
- `results/nystrom_sparse_tail_f81_pilot_failure_analysis_v1/`: six-panel failure analysis and source CSVs.
- `results/trainfree_tail_oracle_f81_registered_v1/`: SHA256-pinned registered train-free tail evidence.
- `results/trainfree_tail_oracle_f81_registered_analysis_v1/`: source CSVs and two failure dashboards.
- `results/restricted_rotation_oracle_f81_registered_v1/`: registered raw rotation records and control-closure evidence.
- `results/restricted_rotation_oracle_f81_registered_analysis_v2/`: cross-holdout and head-fallback source CSVs/figures.
- `results/restricted_rotation_oracle_f81_refinement_v1/`: 600-step, four-restart BCM-8/Butterfly-8 convergence audit.
- `results/restricted_rotation_oracle_f81_refinement_analysis_v2/`: refinement frontier and fallback upper bounds.
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
python scripts/plot_joint_quant_lr_shaping.py \
  --input results/joint_quant_lr_shaping_cpu_v2/joint_quant_lr_shaping.csv \
  --output-dir results/joint_quant_lr_shaping_cpu_v2
python scripts/plot_multiblock_bcm_attention.py \
  --heldout results/multiblock_bcm_attention_f81_cpu_v1/multiblock_bcm_attention_heldout.csv \
  --summary results/multiblock_bcm_attention_f81_cpu_v1/multiblock_bcm_attention_summary.csv \
  --output-dir results/multiblock_bcm_attention_f81_cpu_v1
python scripts/plot_dynamic_sparse_lowrank_oracle.py \
  --summary results/dynamic_sparse_lowrank_oracle_f81_full_v1/dynamic_sparse_lowrank_summary.csv \
  --output-dir results/dynamic_sparse_lowrank_oracle_f81_full_v1 --rank 16
python scripts/plot_tail_aware_sparse_router.py \
  --baseline-summary results/dynamic_sparse_lowrank_oracle_f81_full_v1/dynamic_sparse_lowrank_summary.csv \
  --tail-aware-summary results/tail_aware_sparse_router_f81_full_v1/tail_aware_sparse_router_summary.csv \
  --output-dir results/tail_aware_sparse_router_f81_full_v1 --rank 16
python scripts/plot_conditional_defect_basis_bank.py \
  --summary results/conditional_defect_basis_bank_f81_full_v1/conditional_basis_bank_summary.csv \
  --output-dir results/conditional_defect_basis_bank_f81_full_v1
python scripts/analyze_trainfree_tail_oracles.py \
  --probe-dir results/trainfree_tail_oracle_f81_registered_v1 \
  --output-dir results/trainfree_tail_oracle_f81_registered_analysis_v1
python scripts/analyze_restricted_rotation_oracle.py \
  --input-dir results/restricted_rotation_oracle_f81_registered_v1 \
  --output-dir results/restricted_rotation_oracle_f81_registered_analysis_v2
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
