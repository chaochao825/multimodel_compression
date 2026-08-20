# PLAN-029: Test one-head BF16 precision-island implementation feasibility

- Status: completed
- Owner: researcher and Agent
- Gate: fully exposed implementation-feasibility diagnostic
- Claims: none
- Candidate line: static one-head precision island plus accurate low precision
- Lane: explore
- Resource cap: one implementation, one repair, and one exclusive H200 pass under one hour

## Decision to unlock

Determine whether the post-hoc DIAG-028 quality witness can retain useful H200
speed once the precision split is actually executed. This precedes both QAT and
fresh multi-layer capture because split-kernel overhead may erase its arithmetic
advantage.

## Frozen candidate and data

- High-risk head is fixed to head `4`; no head selection or count sweep.
- Static permutation is fixed to `[0,1,2,3,5,6,7,8,9,10,11,4]`, placing 11
  Sage heads and one BF16 head in contiguous groups. In deployment this channel
  permutation can be folded into Q/K/V and O projections.
- Low-precision group uses the unchanged SageAttention 2.2.0 SM90
  `per_thread`, `smooth_k=True`, `pv_accum_dtype=fp32+fp32` worker. The final
  head uses FA3 BF16.
- Quality uses the same eight exposed Layer-14, steps 7--10, `s02/s03` cells,
  16 query tiles, 12 heads, and FP32 reference as DIAG-028. Actual grouped
  workers must be called; residual-table substitution is not sufficient.
- Latency uses full step-10 `s02` Q/K/V. Permuted contiguous inputs are
  materialized before timing to model projection folding. Timed work includes
  both attention calls and concatenating their outputs in permuted order.
- Compare `fa3_bf16`, full 12-head `sage_thread_smooth`, and
  `head4_bf16_precision_island`; two warmups and five alternating-order wall
  and CUDA-event repetitions. No O projection or end-to-end claim is allowed.

## Gates and outcome map

- Quality requires all `8/8` step x identity cells to satisfy aggregate
  `<=1%`, worst-head `<=2%`, and worst-tile `<=2%`.
- Speed requires measured complete split-path speedup `>=1.4x` versus full
  FA3 BF16.
- `precision-island-pass`: both gates pass. Authorize a fresh stratified
  layer x step x identity atlas before any rollout.
- `quality-only-boundary`: quality passes but speed fails. Stop the Python/two
  kernel path; continue only with a prospectively scoped fused heterogeneous
  kernel.
- `speed-only-boundary`: speed passes but actual quality fails. Prefer the
  registered low-cost QAT action from PLAN-028.
- `precision-island-null`: both fail. Reject the precision-island line and
  proceed to low-cost QAT or exact BF16 fallback.
- Standard invalid, contradictory, adverse, and engineering-failure outcomes
  retain precedence.

## Stop rules

Stop after one valid run or the one allowed engineering repair. Do not tune
heads, reorderings, thresholds, worker settings, captures, or repetition count.
Do not write a custom CUDA kernel, capture new data, run Wan, or start QAT in
this plan. The head choice and all identities are exposed, so a pass establishes
implementation feasibility only.

## Frozen implementation

- Config SHA-256: `c5996c44af7dce8783b43119618b02ce67a8df0c33a8b2d656a8d961af9bc7fc`.
- Decision-core SHA-256: `f3bfec5316a21223b2191151e0227db42593a05ccbe1d8d62780102c2b026e1d`.
- Core-test SHA-256: `7181909113dabef14061e796cb0b2c6064d464783ed8be461918f66945bd6901`.
- Probe SHA-256: `35ea094b798f89498763f4182d8091a439c69b48dd9a289d0feae50987b23b0a`.
- SageAttention core SHA-256: `e475182fc55d6a683499e1eb4d8886fc06e050ed938d8563b090773eede2a99e`.
- FA3 wrapper SHA-256: `53813a192b9d64b49cf72179aadbce39c67c70c00cae2331ed273887c6a58af9`.
- Capture-index SHA-256: `7d12d9ad9fa19379417e943d0962222bfdc59a5df4a91b5f12a625a2e7fda947`.
- Capture-manifest SHA-256: `7f8ad990afd2642677cf9e94a118fe4fe4f227a0ca18a2dbd38bc767e7758aba`.
- DIAG-028 decision SHA-256: `cf6daea633aec4d41c68459a442d45f0707898f4566cc3bde2b6466f863a9c64`.
- DIAG-028 manifest SHA-256: `f9ef367b327e0bf0604539a43ee2865553f9c9bc83432b54ddc7e4da67cef550`.
- Local and remote pure tests passed `3/3`; the H200 smoke test verified
  finite `[1,64,11,128]` Sage and `[1,64,1,128]` FA3 outputs before joining.

## Closure

Write a joint quality/latency report and visualization, update `STATUS.md`, and
leave all protected registries unchanged.

Closed 2026-08-12 with outcome `precision-island-pass`.

- Actual grouped workers passed all `8/8` identity cells. The worst observed
  aggregate/head/tile errors were `0.8453% / 1.3543% / 0.9501%`.
- Full-sequence H200 wall latency was `9.617 ms`, versus `14.360 ms` for FA3
  BF16 and `9.050 ms` for all-head Sage. The split path therefore reached
  `1.4931x`, above the frozen `1.4x` gate.
- The five latency repetitions had `0.49%` CV for the precision island and all
  outputs were finite.
- Decision SHA-256:
  `c4ced5815034b80fca64f28e61799deb22d9a982cc7c2b42963d5402dca641a9`.
- Manifest SHA-256:
  `422c47dbdd235e6e3e226aff7aa8a0e2bcbaa8c7a9292cc02b13dab3e054a27c`.
- Report SHA-256:
  `e0f68a6c4d0d766476954ff42ed62335426cae7144b8768e3c7d482b0017693c`.
- Visualization PNG SHA-256:
  `05bc11de8fa9475135ca3560049d0c453b1bf5a9c21387c7e480cb45c8f00121`.
- Evidence:
  `worldfoundry_hybrid_residual/results/head4_precision_island_f81_l14_diag029_v1/`.
- Report:
  `worldfoundry_hybrid_residual/results/WAN_HEAD4_PRECISION_ISLAND_DIAG029_20260812.zh-CN.md`.
- Required action: open a fresh calibration/evaluation-separated, stratified
  layer x step x identity atlas before rollout, QAT, or a fused kernel claim.
