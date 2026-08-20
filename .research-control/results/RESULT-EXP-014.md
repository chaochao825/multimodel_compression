# RESULT-EXP-014: Shared centered-latent tail amortization

- Status: complete
- Validity: valid exploratory function-class pass
- Date: 2026-08-12
- Gate: G-014
- Claim: C-014
- Candidate: L-014

## Registered outcome

`pass`. Validation selected a centered K/V latent rank-64 basis shared by eight
query tiles at 48.39% optimistic dense work. The frozen `(M=8, rank=64)` test
configuration reached 0.977% aggregate and 1.958% worst-head relative L2 while
recovering 99.818% of shared-adaptive captured defect energy.

## Supports

- Output-channel basis construction can be amortized across a finite query
  region even when the shared rank is higher than a per-tile rank.
- The centered `r+1` value-centroid contrast family closely tracks the adaptive
  shared subspace; ordinary covariance/PCA failure was a basis-family mismatch.
- Sharing has a finite sweet spot: rank 64 passed at `M=8`, while `M=16` failed
  the validation 1% / 2% quality gate despite lower arithmetic work.

## Does not support

- It does not establish a Q/K/V-only predictor, target-free method, measured
  H200 speedup, rollout quality, model-wide coverage, or paper-level result.
- Test latent queries and coefficients still read the dense test defect; only
  tile count and rank were frozen from validation.
- The 48.39% work omits support selection, coefficient prediction,
  normalization, memory traffic, QR, launch, and fusion overhead.

## Decision

Close `L-014` as a positive function-class witness. The next bounded Gate may
replace target-exposed basis generation with one runtime dense anchor tile per
eight-tile group and test anchor-to-query residual lifting. Only if the anchor
basis transfers should a target-free online coefficient predictor be tested.

## Integrity

- Six new centered-core tests and fourteen inherited mechanism tests passed
  remotely before the run.
- Local/remote probe, core, test, and config hashes match.
- Decision/manifest hashes, finite values, exact support, dense reconstruction,
  centered zero-sum, adaptive lower-bound, and basis-conditioning guards pass.
- Runtime was 56.15 seconds on exclusive physical H200 GPU3; no latency
  benchmark was performed.

## Evidence

- `worldfoundry_hybrid_residual/results/shared_centered_latent_tail_f81_l14s9_exp014_v1/decision.json`
- `worldfoundry_hybrid_residual/results/shared_centered_latent_tail_f81_l14s9_exp014_v1/shared_tail_candidates.csv`
- `worldfoundry_hybrid_residual/results/WAN_SHARED_CENTERED_LATENT_TAIL_EXP014_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/shared_centered_latent_tail_exp014_20260812/shared_centered_latent_tail_exp014_diagnostics.png`
