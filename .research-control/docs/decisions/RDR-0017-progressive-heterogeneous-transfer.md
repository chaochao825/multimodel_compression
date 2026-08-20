# RDR-0017: Authorize progressive heterogeneous function-class transfer

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to continue theory-driven,
  high-information experiments after failures
- Supersedes: none

## Context

EXP-014 shows a content-specific M8/rank64 representation witness, while
EXP-015/016 show that one tile or equal-cost 4x16 rows cannot observe it. The
formal EXP-016 oracle reached only 2.013% / 4.557% despite a 0.737% adaptive
floor, so more equal-budget selectors or fixed bases are stopped.

A separate development-only screen changed both the information and execution
class: a calibration basis from two identities, 64/96/112 current rows, and
per-head dense fallback. Uniform execution remained null, but a legal
validation-exposed heterogeneous allocation reached 0.856% / 1.926% at 57.90%
optimistic work. Its coefficients still read the target and test was unopened.

## Decision

Authorize one function-class transfer Gate. Validation may select one action
per head from Q-k-center progressive stages `(4, 6, 7)` crossed with the frozen
prior/innovation splits `(32,64)`, `(48,48)`, `(64,32)`, plus dense fallback.
The selected head actions are frozen before the test tensor is loaded.

Non-anchor coefficients remain target-exposed optimal projections on both
splits. This deliberately tests only whether the per-head action class transfers
across identity. It does not test a predictor and cannot support a speed claim.

## Consequences

- A frozen test pass at 1% / 2% and mean work at most two-thirds permits one
  fresh Gate for target-free coefficient prediction and online certification.
- A validation pass followed by test failure localizes the bottleneck to static
  head-action transfer and requires an online certificate before any predictor.
- A validation null or frozen transfer boundary parks same-step train-free
  residual completion on this cell; do not add more rank, anchors, BCM, BCCB,
  Butterfly, or post-hoc selectors.
- No rollout, fused kernel, learned adapter, or measured H200 speed claim is
  authorized.

## Pre-run implementation lock

- Config: `d3fabe3b3bbaf0d8`
- Probe: `470d6e39d1f89501`
- Core: `87c26894d1dcd7e3`
- Test: `50afbd0f0577d34e`
- Thirty-four new and inherited remote tests pass. The formal launcher must
  verify these hashes before loading any capture.
