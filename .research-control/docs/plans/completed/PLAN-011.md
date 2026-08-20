# PLAN-011: Test a bounded high-rank innovation branch

- Status: completed
- Owner: researcher and Agent
- Gate: G-011
- Claim: C-011
- Candidate: L-011
- Lane: explore
- Resource cap: two existing captures, at most one H200-hour, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether the residual left by a rank-16 tail consists of a small
number of hardware-regular high-rank block innovations, rather than a diffuse
bulk that requires dense fallback.

## Milestones

1. Freeze the exact atom identity, budgets, selectors, comparators, gates, and
   oracle labels.
2. Implement and test block-atom reconstruction, exact budgets, and monotone
   residual-tail greedy selection.
3. Select one selector per budget on validation and execute once on test.
4. Visualize quality versus innovation budget and additive versus expanded
   normalization.
5. Close G-011 before any predictor, training, rollout, or kernel work.

## Stop rules

- Do not add a budget, selector, coefficient, rank, or target after outcomes.
- Held-out dense output may guide atoms only under the explicit oracle label.
- Stop after one valid outcome or one allowed pre-outcome repair.
- Do not train a selector or latent basis inside this Gate.
