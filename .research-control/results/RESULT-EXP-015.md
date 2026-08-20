# RESULT-EXP-015: Dense-anchor residual lifting

- Status: complete
- Validity: valid exploratory null
- Date: 2026-08-12
- Gate: G-015
- Claim: C-015
- Candidate: L-015

## Registered outcome

`basis-null`. At rank 64, the validation target-exposed oracle-best anchor
reached 2.545% aggregate and 6.600% worst-head relative L2; Q-medoid reached
2.713% and 7.434%. Both miss 1% / 2%, despite 43.32% optimistic work.

The corresponding adaptive non-anchor floor was only 0.711% aggregate for the
oracle anchor. The failure is therefore anchor-basis transfer, not insufficient
rank, anchor-only coefficient prediction, or arithmetic cost. Stage B was not
executed because no predictor restricted to the failed basis can beat its
oracle projection floor.

## Supports

- The M8 residual remains adaptively rank-64 compressible after one exact
  anchor tile; its validation adaptive floor is below the registered gate.
- A contiguous 64-query anchor is not a subspace embedding for the other seven
  tiles. The oracle projection/adaptive-floor L2 ratio is about 3.58 at rank 64.
- Q-medoid routing is secondary: oracle-best improves validation aggregate by
  only 0.168 percentage points and test by 0.018 points.

## Does not support

- It does not support single-anchor runtime generation of the EXP-014 shared
  basis or further anchor-only mean/nearest/kernel-ridge development.
- It does not refute content-specific shared low rank in general; EXP-014 still
  provides the group-level function-class witness.
- It does not establish measured H200 speed, rollout quality, full-model
  coverage, or a deployable support selector.

## Decision

Park `L-015`. Do not add ranks, tune anchor routing, or execute coefficient
predictors on this basis. A distinct Gate may keep the same 64 dense-row budget
but distribute it across multiple regular micro-anchors to test leverage-aware
subspace coverage before any learned adapter.

## Integrity

- Fourteen new and inherited tests passed in the formal run.
- Local/remote probe, core, test, and config hashes match.
- Decision/manifest hashes, finite-value checks, exact support reconstruction,
  shared-head Q-medoid, and adaptive lower-bound guards pass.
- Runtime was 10.11 seconds on physical H200 GPU3 after three idle checks; no
  latency benchmark was performed.

## Evidence

- `worldfoundry_hybrid_residual/results/dense_anchor_residual_lifting_f81_l14s9_exp015_v1/decision.json`
- `worldfoundry_hybrid_residual/results/dense_anchor_residual_lifting_f81_l14s9_exp015_v1/stage_a_candidates.csv`
- `worldfoundry_hybrid_residual/results/WAN_DENSE_ANCHOR_RESIDUAL_LIFTING_EXP015_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/dense_anchor_residual_lifting_exp015_20260812/dense_anchor_residual_lifting_exp015_diagnostics.png`
