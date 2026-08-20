# PLAN-027: Audit the accurate low-precision attention kernel frontier

- Status: completed
- Owner: researcher and Agent
- Gate: fully exposed external-baseline diagnostic
- Claims: none
- Candidate line: paper-faithful accurate low-precision attention baseline
- Lane: explore
- Resource cap: one implementation, one repair, and one exclusive H200 pass under one hour

## Decision to unlock

Determine whether an already installed, paper-faithful fine-grained attention
kernel can satisfy the local fidelity and H200 speed conditions that tensorwise
FA3 FP8 and exact gauge equalization missed. This is a baseline audit, not a new
method claim.

## Frozen data and baselines

- Reuse the immutable EXP-040 Layer-14/step-9/conditional F81 Q/K/V captures.
- `s00/s01` remain calibration-labelled only for continuity; no method in this
  plan fits data. Evaluate `s02/s03` held-out on the same 16 deterministic
  64-query tiles and all 12 heads as DIAG-026.
- Use `s02` full `32760 x 12 x 128` Q/K/V for latency, with fixed alternating
  order, two warmups, and five measured repetitions per method.
- Reuse DIAG-026 FA3 BF16/FP8 quality metrics and historical F81 latency as
  cross-checks; do not reinterpret them as new evidence.

## Registered methods

1. `fa3_bf16`: installed FlashAttention-3 BF16 reference worker.
2. `fa3_fp8_tensorwise`: installed tensorwise E4M3 wrapper.
3. `sage_thread_smooth`: SageAttention 2.2.0 SM90, per-thread INT8 Q/K,
   smooth-K enabled, per-channel FP8 V, and two-level `fp32+fp32` PV.
4. `sage_warp_smooth`: identical except per-warp Q/K quantization.
5. `sage_thread_no_smooth`: per-thread ablation with smooth-K disabled.

No threshold, method, granularity, accumulation, smoothing, input, or repetition
sweep is allowed. `sage_thread_smooth` is the historical production point; it
is rerun only on the fixed DIAG-026 cell and actual captured latency shape.

## Evaluation and fairness

- Quality reference is the same FP32 tiled dense attention used by DIAG-026.
- Report aggregate, worst head, worst tile, identity, and nonfinite counts.
- Measure complete Python-call latency including each method's quantization and
  smoothing, excluding model projections and output projection equally.
- Synchronize CUDA events, alternate method order, report every repetition,
  median and interquartile range, and compare with the existing historical row.
- Record package/source/code/config/data hashes, GPU, clocks if available,
  memory, and execution order.
- Local error cannot imply rollout quality. Kernel latency cannot imply
  whole-attention or end-to-end speed without Amdahl and integration overhead.

## Outcome map

- `paper-kernel-pass`: one Sage method reaches held-out `1%` aggregate and `2%`
  worst-head/worst-tile, plus measured `>=1.4x` versus FA3 BF16. Authorize a
  fresh multi-cell atlas before rollout.
- `accuracy-only-boundary`: one Sage method passes `1% / 2%` but remains below
  `1.4x`. Retain as a quality baseline; optimize/fuse before model integration.
- `speed-only-boundary`: one Sage method reaches `>=1.4x` but misses fidelity.
  Do not rollout; use only as a low-cost QAT teacher/student worker candidate.
- `kernel-frontier-null`: no method passes either useful joint boundary. Move
  directly to training-aware quantization/sparsity or exact system work.
- Standard invalid, contradictory, adverse, and engineering-failure outcomes
  retain precedence.

## Stop rules

Stop after one valid run or the one allowed implementation repair. Do not
install a different SageAttention build, tune thresholds, change the capture,
run Wan, add sparse routing, or start QAT in this plan. A fresh package build,
multi-cell atlas, rollout, or trained operator requires a new prospective Gate.

## Frozen implementation

- Config SHA-256:
  `3874ee8dbf3789009f8effdb681ba43b2fe235fa94931a9583245dc4142c96b2`.
- Core SHA-256:
  `668df9dc73a059368eba7d8f9c550773672ee5c64666df051b59eea3a5e45d8d`.
- Probe SHA-256:
  `7788297e157785244ff458248946e1a276debe7f6edf8be0f27b21eac58e4c8b`.
- Test SHA-256:
  `120cce00cea30fda4aea475067a64de83acd39a12a7dfcaba8c29b6a20d4d7af`.
- Installed SageAttention core SHA-256:
  `e475182fc55d6a683499e1eb4d8886fc06e050ed938d8563b090773eede2a99e`.
- Installed FA3 wrapper SHA-256:
  `53813a192b9d64b49cf72179aadbce39c67c70c00cae2331ed273887c6a58af9`.
- Capture-index SHA-256:
  `7d12d9ad9fa19379417e943d0962222bfdc59a5df4a91b5f12a625a2e7fda947`.
- DIAG-026 manifest SHA-256:
  `444e15d4e95656ea8e24c15e1907dfdc6dd21ebffbebd96b3a616601f327b942`.
- Historical attention-table SHA-256:
  `e96907679d57e35a16148d231999e1283b82494bc93f72ae9cda58409f24cd6d`.
- Local and remote implementation hashes match. Four pure-core tests and an
  actual H200 SM90 smoke for all three frozen Sage variants passed before the
  formal run. The smoke is an API/layout check and carries no quality evidence.

## Closure

Write a kernel-frontier report and visualization, update `STATUS.md`, and keep
claim/candidate/experiment/RDR registries unchanged because all data are fully
exposed and the methods are external baselines.

Closed 2026-08-12 with outcome `paper-kernel-pass` under the frozen aggregate
held-out gate.

- `sage_thread_smooth` reached `0.9824%` aggregate, `1.9204%` worst-head,
  `1.0629%` worst-tile error, and `1.5890x` measured full-attention speed.
- `sage_warp_smooth` was fastest at `1.6283x` but missed quality at `1.0370% /
  2.0722%`; `sage_thread_no_smooth` reached `0.9998% / 2.0439%` and also
  missed the worst-head guard.
- Five alternating-order wall-time repetitions had `0.10%--0.45%` coefficient
  of variation; all outputs were finite and GPU3 was released.
- The pass is marginal: `s02` aggregate was `1.0116%` and `s03` worst-head was
  `2.0210%`. The registered gate aggregates held-out identities, so the formal
  result is valid but cannot be described as identity-robust.
- Decision SHA-256:
  `f62f2d7a6a1f23feed1c31dc528d59516a0a5c92bda6ee052d7234ba99f4ef6d`.
- Manifest SHA-256:
  `56e2f09857b846a425172098c2e9711318f6ef47ae8fceda087166a8a6850bc2`.
- Report SHA-256:
  `eb0210c288b7b328dce29264fd1136373f0e599e016c2c95b0eae138f14c323d`.
- Evidence:
  `worldfoundry_hybrid_residual/results/accurate_low_precision_kernel_f81_l14s9_diag027_v1/`.
- Report:
  `worldfoundry_hybrid_residual/results/WAN_ACCURATE_LOW_PRECISION_KERNEL_DIAG027_20260812.zh-CN.md`.
- Required action: run one fixed-method adjacent-step robustness diagnostic
  before spending compute on fresh multi-layer captures or rollout.
