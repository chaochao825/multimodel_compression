# PLAN-001: Full-model selective-execution atlas

- Status: completed; G-003 closed as null
- Owner: researcher and Agent
- Gate: G-003
- Claims: C-003
- Candidate line: L-003
- Lane: explore
- Resource cap: four F81 dense captures in two H200 waves and one frozen
  analysis; one pre-outcome engineering retry

## Decision outcome

The calibration-frozen policy and held-out post-hoc oracle both had insufficient
compute coverage. No rollout gate was opened.

## Milestones

1. Freeze protocol, identities, code semantics, thresholds, and outcome mapping.
2. Add a compatibility-preserving multi-sample capture entry and deterministic
   selector tests.
3. Capture all 30 blocks, 20 steps, and both CFG branches for four identities.
4. Run the evaluator once and classify capacity, transfer, quality, and work.
5. Close G-003. Open rollout only under the preregistered pass outcome.

## Non-goals

- No learned controller, new structured basis, or threshold search.
- No claim that local sampled-output error proves final-video quality.
- No H200 speed claim from profile shares or Amdahl estimates.
- No held-out fitting or retrospective sample removal.

## Stop rule

Stop on the first terminal outcome in EXP-003. Preserve all valid, null,
adverse, invalid, and repaired artifacts.
