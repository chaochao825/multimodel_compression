# RDR-0008: Authorize positive-residual basis provenance analysis

- Status: accepted
- Date: 2026-08-11
- Decider: researcher through the request to continue high-information theory
  and method probes after failures rather than abandon the direction
- Supersedes: none

## Context

EXP-007 refuted a standalone signed Q/K-separable numerator on the registered
Wan F81 cell. Feature rank 16--64 plateaued, and even a target-exposed
all-capture exact-partition fit remained at 15.819% aggregate / 37.598% worst
head. This makes another signed-only training sweep low value.

The earlier positive sparse-linear branch had lower raw error, and its residual
showed a much stronger adaptive output-rank witness. The unresolved mechanism
is therefore not whether a post-hoc SVD can fit a defect, but whether the needed
output basis can be generated from the current V covariance or K/V cross-moment
without inspecting the held-out defect.

## Decision

Authorize one offline `side_probe` on the same four captures. Freeze the
existing calibration-frozen positive rank-64 checkpoint and the same 25%
contiguous-64 Q/K proxy support. Compare adaptive defect SVD, a
calibration-frozen defect basis, V-PCA, selected-V PCA, K-transpose-V, and
pooled-Q-weighted V-PCA at ranks 8/16/32.

Held-out defect may determine projection coefficients only in clearly labelled
oracle diagnostics. It may not determine a content basis, family selection, or
support. One family is selected on validation and evaluated once on test.

This decision permits one analysis script, tests, one execution, one
pre-outcome engineering repair, and one report. It does not authorize a new
tail training run, QKV tuning, new captures, rollout, kernel work, or a measured
speed claim.

## Consequences

- A rank-16 QKV-only content-basis pass permits a new coefficient-generator
  Gate for a positive-bulk plus signed residual construction.
- An adaptive-only pass identifies basis generation as the bottleneck; it does
  not justify a deployable low-rank branch.
- Failure of adaptive rank 16 stops low-rank residuals for this exact positive
  bulk/support target.
- Failure of all content bases through rank 32 parks deterministic V/KV moment
  generation; future work must use a learned nonseparable branch, stronger
  sparse routing, or dense execution.
