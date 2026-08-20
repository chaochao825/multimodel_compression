# PLAN-021: Screen causal coefficient transport and anchor amortization

- Status: completed
- Owner: researcher and Agent
- Gate: pre-gate diagnostic only
- Claims: none; all four identities are exposed
- Candidate line: follow-up diagnostic for `L-020`
- Lane: explore
- Resource cap: one implementation, one repair, and one H200 analysis pass under one hour

## Decision to unlock

Decide whether a fresh prospective Gate for target-free temporal residual
transport is justified, or whether the positive EXP-041 frontier collapses once
current dense coefficients and one-step source-refresh costs are removed.

## Context and source authority

Use only the immutable EXP-040 steps 7--10 Layer-14 conditional captures and
the frozen EXP-041 action table. EXP-041 is already fully exposed and may only
support a mechanism screen, not a transfer or population claim.

## Non-goals

- No new paper claim, prospective generalization claim, rollout, kernel, or
  measured-speed claim.
- No target dense output may enter a deployable predictor feature.
- Do not change support density, rank actions, source basis family, thresholds,
  or the fixed BCM/BCCB/Butterfly no-go boundary.

## Milestones

1. Compare oracle target coefficients with causal raw reuse, calibration-only
   affine transport, and sparse-output-delta innovation.
2. Cross-fit every calibration parameter leave-one-identity-out and report all
   identities rather than selecting a favorable split.
3. Compare immediate-previous and fixed-anchor horizons for targets 8/9/10.
4. Convert local work to refresh-aware arithmetic ceilings for horizons 1--3.
5. Close with a go/no-go recommendation for a fresh prospective Gate.

## Validation

- Unit-test coefficient regression, held-out exclusion, error aggregation, and
  refresh-aware cost equations.
- Report aggregate and worst head-step relative L2 for every method and horizon.
- A prospective Gate is worth opening only if a causal cross-fitted method
  reaches at most 1% / 2% on at least three of four identities, does not use
  target dense coefficients, and retains at least a 1.2x refresh-aware
  arithmetic ceiling before omitted runtime overhead.
- Oracle results are diagnostic ceilings and cannot satisfy the causal gate.

## Pre-run implementation lock

- Screen config: `ade516a803f0f3ac`
- Probe: `97561e15114bdc8a`
- Core: `900e72ef4e5b93a6`
- Core test: `fd988f3a4ca1d971`
- Probe test: `73bfffcbec7929f3`
- EXP-041 source protocol: `f5050fba6be93007`
- EXP-041 decision: `b7cbe3dff3cced5a`
- Seven remote tests pass. The one allowed repair was consumed by relaxing a
  synthetic FP32 coefficient assertion from nine to seven decimal places; no
  method, formula, feature, threshold, or data access changed.

## Stop and escalation rules

Stop after one valid outcome or one implementation repair. If causal methods
miss the quality boundary, park train-free coefficient transport and route the
next effort to learned sparse-linear adaptation or dense FP8. If they pass, do
not reuse the exposed identities as a transfer test; register fresh captures
and a separate claim.

## Closure

Write an exposed-data diagnostic report and update `STATUS.md`. Do not add or
upgrade a claim, candidate, or experiment registry row unless a fresh
prospective Gate is subsequently authorized.

Closed as `causal-coefficient-null`. The best causal method, LOIO source plus
sparse-delta scaling, reached 2.537% aggregate and 6.039% worst head-step error
with 0/4 identities passing. Its transductive counterpart was nearly identical,
so scalar function-class capacity rather than split transfer dominated. Fixed
step-7 anchoring also failed its target-projection floor at 1.031% / 2.636%.
No claim or registry row was added.
