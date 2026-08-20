# PLAN-049: EXP-042 midpoint-RK2 teacher convergence Gate

- Status: complete; neither midpoint grid produced an eligible teacher
- Owner: researcher and agent
- Gate: G-021 teacher engineering sub-gate
- Claims: C-021, unchanged and untested
- Candidate: L-021
- Lane: explore
- Resource scope: one H200 with contamination sentinel, development identities
  only, shifted and uniform midpoint RK2 at 100 and 200 intervals

## Decision question

Can a second-order teacher approximate the same guided Wan flow endpoint within
the frozen `0.25%/0.5%` convergence rule without brute-force Euler NFE?

## Authorized work

1. Implement and unit-test explicit midpoint RK2 arithmetic independently of
   Wan runtime integration.
2. Capture exactly two development identities for each grid and resolution.
3. Save only five FP32 states and four FP32 start velocities per identity.
4. Independently verify finite values, indices, manifests, hashes, and pair
   errors, then apply RDR-0025's deterministic selection rule.

## Prohibited work

- No calibration, validation, test, target fitting, operator changes, RK4,
  adaptive solver, extra grid, VAE decode, QKV, or rollout.
- No relaxation of `0.25%/0.5%` after observing results.

## Exit mapping

- At least one eligible grid: freeze selected solver/grid/resolution and open a
  prospective calibration/validation plan.
- Neither grid eligible: close EXP-042 as teacher engineering failure; C-021
  remains unknown rather than refuted.
- Contamination/schema/hash failure: quarantine and repair the same run only.

## Completion outcome

- All four registered captures completed on one H200 without contamination:
  shifted and uniform grids at 100 and 200 explicit-midpoint intervals.
- The final audit passed for eight payloads and 72 FP32 finite tensors. Each
  identity retained exactly five states and four start velocities; strict tensor
  payload is `603,870,592` bytes with no QKV or VAE decode.
- Shifted midpoint 100/200 endpoint error is `21.1244%` aggregate and `29.0273%`
  worst. Uniform is `36.6326%` aggregate and `42.0954%` worst. Both are far
  above the `0.5%` refined guard despite approximately 24--27% smaller pair gaps
  than Euler.
- RDR-0025 therefore maps the result to teacher engineering failure. No
  calibration, validation, test, curvature fitting, or scientific method arm
  was opened. C-021 remains unknown.
- Evidence is summarized in
  `worldfoundry_hybrid_residual/results/WAN_MOTION_CONJUGATED_FLOW_CURVATURE_EXP042_TEACHER_DIAGNOSTIC_20260813.zh-CN.md`.
