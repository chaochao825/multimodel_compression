# PLAN-003: Test block-control-variate compressibility

- Status: active
- Owner: researcher and Agent
- Gate: G-005
- Claims: C-005
- Candidate line: L-005
- Lane: explore
- Resource cap: one pass over four existing F81 captures, at most two H200
  GPU-hours, and one pre-outcome engineering repair

## Decision to unlock

Decide whether CMAQ has a useful numerical ceiling and a transfer-stable frozen
block proposal before any proposal network, certificate, rollout, or kernel is
considered.

## Frozen scope

- Wan2.1-T2V-1.3B F81, Layer 14, sampling step 9, conditional branch.
- Four existing Q/K/V captures and three stratified 64-query tiles per capture.
- Two calibration identities and two untouched held-out identities.
- Block size 64 and draw densities 12.5%, 25%, 37.5%, and 50%.
- Dense reference is legal only for evaluation, oracle proposal, and
  calibration-only proposal construction.

## Non-goals

- No new model capture, training, rollout, attention kernel, or H200 speed claim.
- No test-dependent proposal, threshold, density, moment order, or random seed.
- No claim that local AV error proves final-video fidelity.
- No attempt to repair a null by adding low rank, BCM, Butterfly, or extra
  support families.

## Milestones

1. Freeze protocol, config, split, estimator equations, and outcome map.
2. Implement and unit-test block moments, joint correction scores, and
   with-replacement control-variate estimation.
3. Validate capture identity and dense-reference parity before reading outcomes.
4. Run one frozen numerical pass and classify G-005.
5. Produce machine-readable artifacts, a concise Chinese report, and close the
   Gate without automatically changing the mainline.

## Stop rules

Stop after the first valid terminal outcome, after one allowed engineering
repair, or when the resource cap is reached. Oracle failure at the 50% ceiling
ends the direction. Any proposal learner, direct defect sentinel, new capture,
rollout, or kernel needs a new researcher decision.
