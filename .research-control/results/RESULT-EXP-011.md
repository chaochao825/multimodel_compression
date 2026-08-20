# RESULT-EXP-011: Bounded additive block innovation

- Status: complete
- Validity: valid exploratory rank boundary
- Date: 2026-08-12
- Gate: G-011
- Claim: C-011
- Candidate: L-011

## Registered outcome

`rank-boundary`. Validation selected atom-energy at 2.5% and 5% innovation,
and exact mass at 10%. On the independent test identity, additive rank 16
reached 1.715% / 3.407%, 1.514% / 3.147%, and 1.249% / 2.917% respectively.
It therefore missed both the strict 0.5% / 1% gate and the 1% / 2% deployment
boundary.

Additive rank 32 reached 0.731% / 1.328% at only 2.5% innovation and improved
to 0.509% / 1.116% at 10%. This establishes a posterior rank boundary, not an
executable acceleration method.

## Supports

- Exact high-rank block atoms reduce the rank tail, but the residual contains a
  broad rank-17--32 shoulder rather than only a few outlier directions.
- Validation selector choices transfer to the test identity at all budgets.
- A post-hoc validation-frozen mixed-rank diagnostic reaches 0.922% / 1.265%
  with average rank 22.67 at 5% innovation, showing real head heterogeneity.

## Does not support

- It does not support uniform rank 16 plus at most 10% block innovation.
- It does not support exact additive atoms as superior to ordinary support
  expansion; expanded renormalization was consistently better with the same
  blocks.
- It does not establish a selector, latent generator, measured latency,
  rollout quality, or model-wide coverage.

## Decision

Park `L-011` as a uniform-rank train-free candidate. Do not add another subset
heuristic, fixed structure, or rank-16 rotation. The one justified successor is
a fresh-data, QKV-frozen heterogeneous sparse-linear adaptation Gate that must
shift the optimistic work below 60% dense while preserving 1% / 2% local
quality. Until that succeeds, this cell remains a dense FP8/BF16 fallback.

## Integrity

- Six EXP-011 tests passed remotely; local/remote source and config hashes
  match.
- The exact atom identity had maximum relative reconstruction error 5.36e-6.
- All 72 greedy traces were monotone and all outputs finite.
- `SUCCESS.json`, decision, manifest, protocol, probe, core, and test hashes
  match.
- Runtime was 116.38 seconds on exclusive physical H200 GPU3; no latency
  benchmark was performed.

## Evidence

- `worldfoundry_hybrid_residual/results/bounded_block_innovation_f81_l14s9_exp011_v1/decision.json`
- `worldfoundry_hybrid_residual/results/bounded_block_innovation_f81_l14s9_exp011_v1/manifest.json`
- `worldfoundry_hybrid_residual/results/bounded_block_innovation_f81_l14s9_exp011_v1/innovation_summary.csv`
- `worldfoundry_hybrid_residual/figures/bounded_block_innovation_exp011_20260812/bounded_block_innovation_exp011_diagnostics.png`
