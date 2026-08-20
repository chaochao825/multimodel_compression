# PLAN-028: Replicate the accurate kernel across adjacent denoising steps

- Status: completed
- Owner: researcher and Agent
- Gate: fully exposed temporal robustness diagnostic
- Claims: none
- Candidate line: Sage per-thread smooth-K dense attention backbone
- Lane: explore
- Resource cap: one implementation, one repair, and one exclusive H200 pass under 30 minutes

## Decision to unlock

Determine whether the marginal DIAG-027 pass is stable across adjacent
denoising steps or is an isolated Layer-14/step-9 cell. This cheap replication
must precede any fresh multi-layer capture, rollout, or QAT campaign.

## Frozen data and method

- Reuse immutable EXP-040 post-RoPE Q/K/V captures only.
- Cell family: Wan2.1-T2V-1.3B F81, Layer 14, conditional branch, sampling
  steps `7, 8, 9, 10`.
- Evaluate only held-out-labelled `s02/s03`, with the same 16 deterministic
  64-query tiles and all 12 heads as DIAG-027.
- Run exactly two workers: FA3 BF16 as a numerical reference cross-check and
  the installed SageAttention 2.2.0 SM90 `per_thread`, `smooth_k=True`,
  `pv_accum_dtype=fp32+fp32` candidate. No alternative granularity, smoothing,
  rank, scale, support, or threshold is allowed.
- Quality reference remains FP32 tiled dense attention. Reuse DIAG-027's
  `1.5890x` full-shape latency only as provenance; do not create a new speed
  claim from this quality replication.

## Stronger robustness gate

For every step and identity separately, require aggregate relative L2 at most
`1%`, worst-head at most `2%`, and worst-tile at most `2%`. Also report the
combined two-identity result at each step and every head-level error.

## Outcome map

- `adjacent-step-robust`: all four combined-step gates and at least six of
  eight step-identity gates pass. Authorize a fresh layer × step × identity
  atlas with the same fixed worker.
- `adjacent-step-boundary`: two or three combined-step gates pass, or at least
  four of eight step-identity gates pass. Do not rollout; use the map to define
  one low-cost QAT gate for boundary cells.
- `isolated-cell-pass`: at most one combined-step gate and fewer than four
  step-identity gates pass. Reclassify DIAG-027 as a narrow cell witness and
  prioritize training-aware quantization or exact BF16 fallback.
- Standard invalid, contradictory, adverse, and engineering-failure outcomes
  retain precedence.

## Stop rules

Stop after one valid run or the one allowed engineering repair. Do not capture
new data, fit any parameter, change the worker, measure rollout, or tune on
step/identity outcomes. All identities are exposed, so this plan cannot update
a protected claim or establish deployment transfer.

## Frozen implementation

- Config SHA-256:
  `9f2f85696379dbbbb10c086e93875c1bcef0c5f9d01686990353d00a46391649`.
- Core SHA-256:
  `e140969a8cbef7d4e4c2e0dd983dabc7875d72b4a780424bee5c89c1690378b0`.
- Probe SHA-256:
  `26455bc4ef27d352b6cf522485188480125f880496f493c7acec92476cd15ed5`.
- Test SHA-256:
  `e5c93c6ca723e443bd66d42bd8d0c424db0bad947d88685cff68f9efb21fd2df`.
- Capture-index SHA-256:
  `7d12d9ad9fa19379417e943d0962222bfdc59a5df4a91b5f12a625a2e7fda947`.
- Capture-manifest SHA-256:
  `7f8ad990afd2642677cf9e94a118fe4fe4f227a0ca18a2dbd38bc767e7758aba`.
- DIAG-027 decision SHA-256:
  `f62f2d7a6a1f23feed1c31dc528d59516a0a5c92bda6ee052d7234ba99f4ef6d`.
- Installed SageAttention source hashes remain those frozen by PLAN-027.
  Local and remote implementation hashes match; three core tests passed.

## Closure

Write a temporal robustness report and heatmap, update `STATUS.md`, and keep all
claim/candidate/experiment/RDR registries unchanged.

Closed 2026-08-12 with outcome `adjacent-step-boundary`.

- Combined steps 7/8/9 passed and step 10 failed: `3/4` coverage.
- The stronger step × identity gate passed only `4/8`: both identities passed
  at steps 7/8, while both failed at steps 9/10 for different tail metrics.
- Combined aggregate error increased smoothly from `0.9451%` to `1.0039%`;
  head 4 remained the worst head at every step (`1.8646%--1.9872%`).
- FA3 BF16 guards remained at about `0.168%`; all outputs were finite and the
  formal H200 run exited zero after 9.15 seconds.
- Decision SHA-256:
  `cf6daea633aec4d41c68459a442d45f0707898f4566cc3bde2b6466f863a9c64`.
- Manifest SHA-256:
  `f9ef367b327e0bf0604539a43ee2865553f9c9bc83432b54ddc7e4da67cef550`.
- Report SHA-256:
  `952a8fca5c4b42436c1b320f196f2ace24f5b9d954d2b52c25e0cfd7ca059cda`.
- Report:
  `worldfoundry_hybrid_residual/results/WAN_ADJACENT_STEP_ACCURATE_KERNEL_DIAG028_20260812.zh-CN.md`.
- Registered action remains one low-cost QAT gate. A new post-hoc observation
  additionally motivates a bounded implementation diagnostic: replacing only
  exposed worst head 4 with BF16 passes all eight recorded identity cells at an
  optimistic `1.515x` attention upper bound. It is not part of this outcome.
