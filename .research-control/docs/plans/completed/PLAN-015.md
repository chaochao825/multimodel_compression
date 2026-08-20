# PLAN-015: Replace target-exposed basis with a dense anchor

- Status: completed
- Owner: researcher and Agent
- Gate: G-015
- Claim: C-015
- Candidate: L-015
- Lane: explore
- Resource cap: two existing captures, at most one H200-hour, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether the EXP-014 shared basis can be obtained from one legal dense
anchor tile and whether its coefficients can be predicted online without dense
non-anchor targets.

## Milestones

1. Freeze anchor selectors, rank, coefficient families, hyperparameters, cost,
   and validation-to-test selection.
2. Test anchor selection, basis transfer, kernel/ridge algebra, data-access
   guards, and method-specific work accounting.
3. Run Stage A and Stage B once on physical H200 with bounded monitoring.
4. Visualize basis versus coefficient error and close G-015 before any learned
   adapter, rollout, or kernel work.

## Stop rules

- Do not add features, kernels, lambdas, temperatures, ranks, or anchor methods
  after outcomes are visible.
- Stop after one terminal outcome or the one allowed pre-outcome repair.
- A coefficient-boundary permits only a fresh-data low-cost adapter proposal;
  it does not permit post-hoc predictor tuning on these identities.
