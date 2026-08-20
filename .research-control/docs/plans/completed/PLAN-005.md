# PLAN-005: Compare attention-sampling mechanisms before engineering

- Status: completed
- Owner: researcher and Agent
- Gate: G-006
- Claim: C-006
- Candidate: L-006
- Lane: explore
- Resource cap: one pass over the two held-out F81 captures, no H200 requirement,
  one pre-outcome repair, and no model generation

## Decision to unlock

Determine whether CMAQ failed because of proposal granularity, sampling design,
or joint partition estimation, and whether any same-budget mechanism has a
credible numerical ceiling before kernel work.

## Milestones

1. Freeze peer-mechanism mappings, split, group sizes, budgets, estimators, and
   outcome map.
2. Unit-test row/group masks, ratio factorials, and fixed-work stratified HT.
3. Run the registered held-out numerical comparison exactly once.
4. Report numerical error, proposal granularity, duplicate/union inflation, and
   scope-limited literature comparison.
5. Close G-006 without starting training, rollout, or kernel work.

## Stop rules

- Stop after the first valid terminal outcome or one allowed pre-outcome repair.
- Do not add a method, density, group size, or estimator after reading outcomes.
- Do not infer GPU speed from arithmetic block work.
