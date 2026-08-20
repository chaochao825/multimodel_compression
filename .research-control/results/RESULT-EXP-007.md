# RESULT-EXP-007: Signed numerator tail with separate positive partition

- Status: complete
- Validity: valid exploratory null
- Date: 2026-08-11
- Gate: G-007
- Claim: C-007
- Candidate: L-007

## Registered outcome

`null`: the selected calibration-frozen architecture was
`signed_n32_z32_o8`. On the one-shot test identity its joint aggregate / worst
head local AV relative L2 was 10.969% / 24.395%. With the exact partition it
was 17.242% / 34.856%. The target-exposed all-capture transductive exact-Z
capacity diagnostic was 15.819% / 37.598%, far above both the 0.5% / 1% oracle
gate and the 1% / 2% deployment gate.

All evaluated partitions and outputs were positive and finite. The registered
arithmetic ratio was 0.2532, but this is only a MAC proxy and no H200 latency was
measured. Runtime was 76.55 seconds on an idle H200 NVL.

## Mechanism evidence

- Validation joint error was nearly rank-invariant: 10.47%--10.69% across
  numerator ranks 16, 32, and 64.
- The rank-8 output adapter helped the exact-Z calibration objective, but the
  best objective-equivalent error remained about 17.2%.
- On test, joint normalization reduced aggregate error from the exact-Z
  numerator diagnostic's 17.24% to 10.97%. This is error compensation, not an
  accurate numerator: the exact-N denominator-only diagnostic was 28.17%.
- Joint output improved over the registered 25% sparse-only baseline
  (12.51% aggregate / 34.77% worst head), but it remained an order of magnitude
  outside the gate.
- Head 6 was the largest test failure at 24.40% joint error. Every head missed
  the 2% guard.

## Supports

- The EXP-006 numerator bottleneck is not rescued by replacing a positive
  feature map with a standalone signed separable Q/K feature map.
- Increasing this signed feature width is not the next high-information action
  under the registered optimizer and data.
- A learned denominator can mask part of a numerator error, so joint output
  alone is insufficient evidence that either branch is accurate.

## Does not support

- It does not refute all learned sparse-linear attention, all signed control
  variates, or all content-generated output bases.
- It does not test a positive bulk plus signed residual construction.
- It does not establish rollout quality, H200 kernel speed, or Wan-wide
  behavior beyond one layer-step-branch cell and three query tiles.
- The transductive fit is a target-exposed capacity diagnostic and cannot be
  interpreted as held-out transfer.

## Decision

Stop the standalone signed separable numerator family on the registered cell.
The most informative revival condition is a residualized construction whose
positive branch first models the stable bulk and whose signed, value-aware
branch only predicts the remaining numerator defect. Before training that
construction, a new Gate should test whether the known adaptive output defect
basis is recoverable from current V covariance or K/V cross-moments without
held-out fitting.

## Evidence

- `worldfoundry_hybrid_residual/results/signed_numerator_tail_f81_l14s9_exp007_v1/decision.json`
- `worldfoundry_hybrid_residual/results/signed_numerator_tail_f81_l14s9_exp007_v1/manifest.json`
- `worldfoundry_hybrid_residual/results/signed_numerator_tail_f81_l14s9_exp007_v1/signed_tail_summary.csv`
- `worldfoundry_hybrid_residual/figures/signed_numerator_tail_exp007_20260811/signed_numerator_tail_exp007_diagnostics.png`
