# PLAN-013: Shape support for tail cost, not attention mass

- Status: completed
- Owner: researcher and Agent
- Gate: G-013
- Claim: C-013
- Candidate: L-013
- Lane: explore
- Resource cap: two existing captures, at most one H200-hour, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether rank-conditioned regular support can create enough residual-
rank margin to justify fresh-data sparse-linear adaptation.

## Milestones

1. Freeze the 35% support, ranks, swap search, cost model, allocation rule,
   quality gates, and oracle boundary.
2. Implement and test fixed-budget rank-conditioned swaps and monotone traces.
3. Run validation selection and one frozen test execution, then compare with
   unswapped support and the registered posterior test diagnostic.
4. Close G-013 before any learned router, rollout, or kernel.

## Stop rules

- Do not add initializers, patterns, ranks, rounds, or thresholds after output.
- Dense targets are oracle-only and cannot support deployment or speed claims.
- Stop after one valid outcome or one allowed pre-outcome repair.
