# PLAN-060: Resolve true support-state capacity

- Status: complete
- Owner: researcher and Agent
- Gate: G-030
- Claim: C-029
- Candidate line: L-029
- Lane: explore side probe
- Resource cap: repaired exposed capture and one A800; no H200

## Steps

1. Test the shared control-variate training objective and hard support budget.
2. Train mass-support and residual-support states from the same checkpoint and
   batch schedule.
3. Evaluate the primary four arms and complete 2x2 factorial diagnostics.
4. Apply G-030 before any router, task, later split, or latency work.
5. Close the post-hoc capacity line on a miss or propose one deployable-router
   Gate on a pass.

## Stop rule

Do not increase width, density, steps, support irregularity, or data after
seeing development outcomes. Preserve all failed attempts and use at most two
implementation repairs.

## Closure

EXP-051 completed one valid G-030 outcome. The joint residual arm reached
4.281%/13.642%/21.254% visual mean/P95/worst and was 66.536% worse in risk
than mass-support-trained state plus mass support; the paired interval remained
entirely adverse. Runtime replay, exact budget, finite checkpoints, and
all-page dense recovery passed. Close the registered post-hoc capacity family
before router, task, sealed confirmation, or H200 work. Park L-029 without
changing the L-026 Wan/rCM mainline.
