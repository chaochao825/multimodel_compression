# PLAN-016: Test distributed observation before learned adaptation

- Status: completed
- Owner: researcher and Agent
- Gate: G-016
- Claim: C-016
- Candidate: L-016
- Lane: explore
- Resource cap: two existing captures, at most one H200-hour, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether equal-cost distributed residual observations repair the
single-anchor subspace-coverage failure without using target information in the
deployed selector.

## Milestones

1. Freeze microblock geometry, legal selectors, target-exposed oracle, ranks,
   cost, quality Gates, and validation-to-test selection.
2. Test distinct-parent constraints, selector target isolation, exact selected
   rows, adaptive lower bounds, and equal-row cost.
3. Run once on physical H200 and visualize rank, per-head, selector, and
   historical basis gaps.
4. Close G-016 before any coefficient predictor or learned selector.

## Outcome

Completed as `distributed-null`. The target-exposed rank-64 oracle reached
2.013% / 4.557% on validation, so no test tensor or coefficient predictor was
opened.

## Stop rules

- Do not add microblock sizes, selectors, ranks, or row budgets after outcomes
  are visible.
- Stop after one terminal outcome or the one allowed pre-outcome repair.
- A distributed null closes this equal-cost train-free residual-basis route on
  the current cell.
