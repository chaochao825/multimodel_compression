# PLAN-058: Build the conditional rate-distortion frontier

- Status: complete
- Owner: researcher and Agent
- Gate: G-028
- Claims: C-028
- Candidate line: L-028
- Lane: explore
- Resource cap: exposed EXP-003 artifacts, then one isolated H200 for at most
  eight GPU-hours and 35 GiB only after promotion

## Decision to unlock

Decide whether self-attention, FFN, or whole-block current-state interfaces can
support a faithful post-hoc acceleration path, and which target should receive
any later training or kernel investment.

## Milestones

1. Implement and test one evaluator shared by all three module targets.
2. Produce the exposed-data local capacity/deployability screen and raw CSV.
3. Promote only locally passing targets to fresh full-token suffix intervention.
4. Measure complete candidate and exact-boundary H200 costs.
5. Plot the conditional rate-distortion frontier, apply G-028, and record one
   bounded outcome.

## Fairness and leakage controls

- All targets share splits, histories, horizons, coefficient grids, and metrics.
- Calibration-only artifacts are frozen before any held-out target is read.
- Target-visible rows are labeled capacity ceilings and never select a
  deployable action.
- Stage 0 exposed rows are screening evidence and cannot pass the Gate.
- Exact module interfaces are charged only when they are not already available
  at the replacement boundary.

## Stop and escalation

Do not capture suffixes for a locally failed target. Do not time a numerically
invalid candidate. Do not train a router or develop a custom kernel unless one
target passes every G-028 condition.

## Closure

Stage 0 completed with 38,880 finite rows and no promoted target. The closest
quality point covered only 0.741% of self-attention calls, while every relaxed
policy remained below the 1.2x zero-renderer Amdahl ceiling or failed quality.
G-028 closed at its registered early stop without suffix capture or H200 timing.
