# PLAN-012: Test the heterogeneous-rank speed ridge

- Status: active
- Owner: researcher and Agent
- Gate: G-012
- Claim: C-012
- Candidate: L-012
- Lane: explore
- Resource cap: two existing captures, at most one H200-hour, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether intermediate per-head ranks can preserve the EXP-011 quality
boundary below 60% optimistic dense work, which is the necessary condition for
training a small frozen-QKV sparse-linear generator.

## Milestones

1. Freeze rank actions, support families, allocation optimizer, cost model,
   quality gates, and oracle labels.
2. Implement and test exact Pareto pruning, cap compliance, deterministic tie
   breaking, and frozen validation-to-test allocation.
3. Run once on validation and test, then visualize quality versus work and the
   selected per-head rank map.
4. Close G-012 before any fresh-data training, rollout, or kernel work.

## Stop rules

- Do not add ranks, costs, support families, or thresholds after outcomes.
- Dense targets and SVD are oracle-only and cannot support a speed claim.
- Stop after one valid outcome or one allowed pre-outcome repair.
- Do not train a router, latent basis, or Q/K adapter inside this Gate.
