# PLAN-010: Shape the rank-16 residual with regular support

- Status: completed
- Owner: researcher and Agent
- Gate: G-010
- Claim: C-010
- Candidate: L-010
- Lane: explore
- Resource cap: two existing held-out captures, at most one H200-hour, one valid
  execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether support selection, rather than basis generation, can reduce
the deployment-relevant rank-16 lower bound under regular block constraints.

## Milestones

1. Freeze densities, support families, influence score, swap search, ranks,
   selection, gates, and oracle labels.
2. Implement stable block numerator/partition decomposition and exact-budget
   support evaluation.
3. Test gradient-guided swap against exhaustive tiny instances and invariants.
4. Select one swap initialization per density on validation and evaluate once
   on test.
5. Visualize error by density/method, swap gain, and rank spectrum; close G-010.

## Stop rules

- Do not add a density, support shape, initialization, candidate count, round,
  or target after reading outcomes.
- Held-out dense AV may guide support only under the explicit oracle label.
- Stop after one valid outcome or one allowed pre-outcome repair.
- Do not train a router or latent basis inside this Gate.
