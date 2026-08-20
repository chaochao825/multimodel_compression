# PLAN-023: Screen Q-conditioned channel-state innovation

- Status: completed
- Owner: researcher and Agent
- Gate: pre-gate diagnostic only
- Claims: none; all identities are exposed
- Candidate line: low-cost learned innovation after parking train-free transport
- Lane: explore
- Resource cap: one implementation, one repair, and one H200 analysis pass under one hour

## Decision to unlock

Determine whether stable head-channel maps using current query and sparse-output
observables can recover the missing temporal defect directions at low arithmetic
cost, and therefore justify fresh prospective low-cost adaptation captures.

## Context and source authority

DIAG-021/022 showed that the previous-step basis is useful under current target
projection, but source coefficients plus sparse-output delta cannot generate
current coefficients even with target-exposed per-row gates. This plan changes
the function class rather than gate granularity:

\[
\hat D_t=D_sA+(S_t-S_s)B+Q_tC+(Q_t-Q_s)D.
\]

All maps act in stable per-head channel coordinates. Reuse immutable EXP-040
captures, EXP-041 support and dense-fallback heads, and immediate-previous pairs.

## Non-goals

- No train-free, transfer, deployment, rollout, kernel, latency, or novelty
  claim from exposed identities.
- No support, head action, source horizon, threshold, BCM/BCCB/Butterfly, or
  target-projection coefficient changes.
- No nonlinear network, hyperparameter sweep, or post-outcome feature addition.

## Milestones

1. Compare nested ridge families: source defect; source plus sparse delta; source
   plus sparse delta, current Q, and Q delta.
2. Report target-exposed same-identity fit, LOIO channel-map transfer, and
   per-row feature-span oracle for each family.
3. Attribute gains to current Q versus sparse innovation and identify hard heads.
4. Estimate static parameters and arithmetic relative to dense attention.
5. Decide whether to collect fresh identities for a low-rank-map adaptation Gate.

## Validation

- Use one fixed trace-relative ridge value and training-only block RMS scaling.
- Every LOIO map and scaler excludes the evaluated identity.
- Report per identity, target, head, aggregate, and worst relative L2.
- A family merits a fresh Gate only if LOIO passes 1% / 2% on at least 3/4
  identities, same-identity capacity passes, and estimated arithmetic is below
  5% of dense attention.
- Per-row oracle can only authorize a learned nonlinear/content predictor, not
  the registered linear channel map.

## Stop and escalation rules

Stop after one valid outcome or one repair. If the four-block feature family
fails even per-row span capacity, terminate temporal residual prediction. If
capacity passes but LOIO fails, do not add features post hoc; recommend a small
trained innovation map on fresh data. If LOIO passes, preregister fresh captures
before any rollout or kernel work.

## Implementation lock

- Config SHA256: `0392c4ee464b8b92c1696e54d810e4255bbae3fc73b4fdf20452dbea85d6438f`
- Core SHA256: `e063832252203b3f227a4cb1d2288d056d27c26d1ff373af0f4ca302d4ec51e7`
- Probe SHA256: `f01a45f1f492ca9eca11586801f535ec9489e372d7ee37ceb9a02b7ce66c6a18`
- Core-test SHA256: `53d4e4a3a12f66256c5cc878fd1cec9ee046b96c390ebdff9da11179c44d6972`
- Probe-test SHA256: `dbdc233c95fe222c909dc55e52b2dded2cef0c5f94fe013973f61fc9ce340855`
- Source-protocol SHA256: `f5050fba6be930071840d398f394bd244b3f1c14fad20e5b27da6b3bba6b1a23`
- EXP-041 decision SHA256: `b7cbe3dff3cced5a03182a25fbe4a6070b87e2af0b6428372b667cede15b23b7`
- DIAG-022 decision SHA256: `d5c2d8f6e6daebe694aac2f734932528f11e05d5664596e67b689dcd4c783602`
- Validation before lock: 7 tests passed on the remote experiment environment.

## Closure

Write a bound report and update `STATUS.md`. Keep claim/candidate/experiment
registries unchanged because the current identities are fully exposed.

Closed as `channel-state-capacity-null`. The richest row-span oracle reached
1.787% aggregate / 4.508% worst-head-step error with 0/4 identities passing.
Its same-identity full channel map reached 1.000% / 2.684% and 2/4 passing,
whereas LOIO transfer regressed to 2.987% / 7.349% and 0/4 passing. Current Q
therefore contains sample-specific fitting signal but does not make the defect
causally transportable in the registered stable-channel function class.
Temporal residual prediction is terminated under this protocol. The next
bounded diagnostic changes the attention worker itself and audits the official
SLA operator family before any fresh-data adaptation or rollout.
