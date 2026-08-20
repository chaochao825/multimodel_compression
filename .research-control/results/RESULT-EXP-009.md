# RESULT-EXP-009: Signed value-aware latent-pooling ceiling

- Status: complete
- Validity: valid exploratory boundary
- Date: 2026-08-11
- Gate: G-009
- Claim: C-009
- Candidate: L-009

## Registered outcome

`rank16-generation-boundary`. Validation selected
`signed_kv16_softmaxdiff`. On the independent test identity:

- sparse-only base: 12.511% aggregate / 34.766% worst-head error;
- adaptive rank 16: 2.120% / 3.872%;
- selected latent rank 16: 2.448% / 4.467%, recovering 99.02% of adaptive
  captured energy at an optimistic 65.62% dense-work lower bound;
- adaptive rank 32: 0.915% / 1.509%;
- selected latent rank 32: 1.148% / 1.880%, recovering 99.69% of adaptive
  captured energy at 106.25% optimistic dense work.

The rank-16 claim is refuted because its adaptive lower bound already misses
1% / 2%. Rank 32 nearly reaches the absolute quality gate but is not an
acceleration candidate under the registered cost accounting.

## Supports

- Signed value-aware softmax-difference pooling generates a substantially more
  relevant output subspace than deterministic covariance/moment bases.
- Basis generation is no longer the dominant rank-16 gap: the selected family
  recovers 99.02% of adaptive captured energy and has 0.851 mean subspace
  overlap.
- The deployment-relevant bottleneck is the intrinsic rank left by the current
  pooled-Q/K support, not merely latent-query optimization.

## Does not support

- It does not establish a transferable latent-query or coefficient generator;
  both are target-exposed.
- It does not establish measured speed, rollout quality, or Wan-wide coverage.
- The family comparison is not a complete K-only/K+V by signed/unsigned
  factorial, so V-sketch causality remains unresolved.

## Decision

Do not add rank or optimize the same support further. Park direct generator and
kernel work. The one justified revival is a hardware-regular support-rank
co-design Gate that minimizes post-rank-16 defect energy at equal or lower
density. If its adaptive rank-16 oracle cannot reach 0.5% / 1%, stop this
sparse-plus-low-rank target.

## Integrity

- Six EXP-009 unit tests and five EXP-008 regression tests passed remotely.
- Synthetic optimizer guard reduced generated-subspace residual to 0.8%--3.1%
  of initialization.
- All generated-basis minimum singular ratios exceeded the registered `1e-6`
  guard; test minimum was 0.00762 at rank 32.
- `SUCCESS.json` hashes match downloaded `decision.json` and `manifest.json`.
- Runtime was 108.30 seconds on exclusive physical H200 GPU2; no latency
  benchmark was performed.

## Evidence

- `worldfoundry_hybrid_residual/results/signed_value_latent_pool_f81_l14s9_exp009_v1/decision.json`
- `worldfoundry_hybrid_residual/results/signed_value_latent_pool_f81_l14s9_exp009_v1/manifest.json`
- `worldfoundry_hybrid_residual/results/signed_value_latent_pool_f81_l14s9_exp009_v1/latent_summary.csv`
- `worldfoundry_hybrid_residual/figures/signed_value_latent_pool_exp009_20260811/signed_value_latent_pool_exp009_diagnostics.png`
