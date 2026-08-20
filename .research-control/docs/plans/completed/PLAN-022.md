# PLAN-022: Decompose sign-equivariant temporal coefficient dynamics

- Status: completed
- Owner: researcher and Agent
- Gate: pre-gate diagnostic only
- Claims: none; all identities are exposed
- Candidate line: bounded follow-up to the `L-020` mechanism witness
- Lane: explore
- Resource cap: one implementation, one repair, and one H200 analysis pass under one hour

## Decision to unlock

Determine whether the failed two-scalar coefficient model omitted cheap
heterogeneity along the singular-direction axis, the token/content axis, or
both. Decide whether any low-payload causal family merits fresh prospective
captures or whether train-free temporal coefficient transport should stop.

## Context and source authority

DIAG-021 showed an immediate-previous target-projection floor of 0.821% / 1.877%
but a best causal error of 2.537% / 6.039%. Cross-fitted and transductive scalar
fits differed by less than one percent relatively, and hard heads attributed
86%--94% of their error energy to coefficient prediction rather than basis
projection. Reuse the immutable EXP-040 captures, EXP-041 action table, and
DIAG-021 support/basis semantics.

## Non-goals

- No claim, fresh-data transfer statement, rollout, kernel, or speed result.
- No rank, support, basis, dense-fallback, BCM/BCCB/Butterfly, or threshold
  changes.
- Do not use unconstrained full coefficient matrices or per-element target
  payload as a deployable candidate.

## Milestones

1. Measure target-exposed per-rank diagonal and per-row two-scalar capacity.
2. Evaluate LOIO per-rank diagonal and aligned-position row maps.
3. Evaluate a sign-equivariant row gate using only source coefficients and
   sparse-delta norms, cosine, and norm ratio.
4. Attribute remaining error to direction-only, row-only, interaction, and
   source-basis projection components.
5. Close with a precise stop or fresh-data recommendation.

## Validation

- Preserve immediate-previous source mode and all EXP-041 frozen actions.
- Every cross-fitted parameter excludes the evaluated identity.
- Report per identity, target, head, and pooled aggregate/worst errors.
- A low-payload family merits a prospective Gate only if at least 3/4 identities
  pass 1% / 2%, its target-exposed version also passes, and estimated predictor
  arithmetic is below 5% of skipped dense attention.
- Per-row target oracle is diagnosis only and cannot authorize a Gate.

## Pre-run implementation lock

- Screen config: `c50d5dff9d206aeb`
- Probe: `5c7539e4367bf8c3`
- Core: `83e89fe5f5baf196`
- Core test: `1d78a31b0476559c`
- Probe test: `6850c09e4b636425`
- EXP-041 protocol/decision: `f5050fba6be93007` / `b7cbe3dff3cced5a`
- DIAG-021 decision: `6f42eec056bc09ca`
- Fourteen remote tests pass before data access.
- The one repair allowance was consumed after the first execution reached no
  result write: double-precision fitted coefficients are now explicitly cast
  to the locked FP32 basis dtype at reconstruction, matching EXP-041/DIAG-021.
  The failed partial directory was preserved under remote `trash/`.

## Stop and escalation rules

Stop after one valid run or one repair. If only a target-exposed row oracle
passes, move to a small learned content gate rather than claiming train-free
success. If every structured family fails capacity, park train-free temporal
coefficient transport and prioritize learned sparse-linear or dense FP8 paths.

## Closure

Write a bound diagnostic report, update `STATUS.md`, and keep exactly one active
plan. Do not alter claim, candidate, or experiment registries for exposed-data
analysis.

Closed as `equivariant-capacity-null`. The best causal rank-plus-row gate was
2.422% / 5.781% with 0/4 identities passing. Even target-exposed per-row
two-vector factors reached only 1.907% / 4.673%, so finer scaling of source and
sparse-delta coefficients cannot reach the projection floor. Train-free
coefficient transport is parked; the next plan introduces new current-Q channel
directions under low-cost calibration-only adaptation.
