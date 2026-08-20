# PLAN-017: Test progressive heterogeneous transfer before prediction

- Status: completed
- Owner: researcher and Agent
- Gate: G-017
- Experiment: EXP-038
- Claim: C-017
- Candidate: L-017
- Lane: explore
- Resource cap: four existing captures, at most one H200-hour, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether validation-frozen per-head progressive actions transfer to an
untouched identity when coefficients are still oracle projections.

## Milestones

1. Freeze staged data access, action family, allocation objective, quality/work
   gates, outcome map, and tie rules.
2. Implement exact per-head allocation and frozen-action evaluation with tests
   for leakage, dense fallback, cost, and adaptive lower bounds.
3. Lock hashes, run once on exclusive H200, and visualize validation-to-test
   head actions and quality/work margins.
4. Close G-017 before any target-free coefficient predictor or online risk
   certificate.

## Outcome

Completed as `transfer-boundary`. Validation passed at 0.856% / 1.926% and
57.90% optimistic work, but frozen test aggregate error reached 1.097% despite
a 1.713% worst-head error. Static head actions are parked; aggregate-risk-aware
online certification is the only bounded continuation.

## Stop rules

- Do not add stages, basis splits, selectors, ranks, support, or thresholds
  after validation or test outcomes are visible.
- Stop after one terminal outcome or the one allowed pre-outcome repair.
- A null or transfer boundary parks static same-step residual completion on
  this cell.
