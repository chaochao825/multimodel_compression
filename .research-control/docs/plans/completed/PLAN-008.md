# PLAN-008: Trace the positive-tail residual basis

- Status: completed
- Owner: researcher and Agent
- Gate: G-008
- Claim: C-008
- Candidate: L-008
- Lane: explore
- Resource cap: existing four captures and checkpoint, at most one H200-hour,
  one valid execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether the residualized low-rank opportunity has a basis available
from current Q/K/V content, or whether prior adaptive SVD gains depend on
unavailable posterior defect information.

## Milestones

1. Freeze positive checkpoint, support, split, basis families, ranks, selection,
   gates, and leakage labels.
2. Implement and test orthonormal basis construction and projection accounting.
3. Select one content basis on validation and evaluate it once on test against
   adaptive, frozen, and random controls.
4. Visualize rank/error, captured energy, head tails, and basis overlap.
5. Close G-008 before training a residual coefficient generator.

## Stop rules

- Do not add a basis, rank, weight rule, or support after reading outcomes.
- Held-out defect may only enter registered oracle coefficients and adaptive
  SVD; it may not construct a content basis.
- Stop after one valid terminal outcome or one allowed pre-outcome repair.
- Do not infer H200 speed from moment MACs.
