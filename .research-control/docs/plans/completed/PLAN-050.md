# PLAN-050: Decide the post-EXP-042 teacher path

- Status: active; awaiting protected researcher decision
- Owner: researcher and agent
- Gate: G-021 postmortem decision
- Claims: C-021, unchanged and untested
- Candidate line: L-021
- Lane: explore
- Resource cap: zero GPU execution and no new payload until a new Gate is accepted

## Decision to unlock

Choose whether to open a new prospective teacher-definition Gate or park L-021.
The evidence supports neither silently continuing EXP-042 nor refuting C-021.

## Context and source authority

- RDR-0025 prospectively stops when neither midpoint grid meets the frozen
  teacher guard; that outcome occurred.
- EXP-042 is complete with failure class `engineering-failure`.
- Shifted midpoint is `21.1244%/29.0273%` aggregate/worst and uniform is
  `36.6326%/42.0954%`; all scientific method gates remain unevaluated.
- Canonical report:
  `worldfoundry_hybrid_residual/results/WAN_MOTION_CONJUGATED_FLOW_CURVATURE_EXP042_TEACHER_DIAGNOSTIC_20260813.zh-CN.md`.

## Options requiring researcher judgment

1. Open a new bounded diagnostic that first measures fractional-versus-integer
   timestep vector-field discontinuity, then freezes a production-compatible
   float-time teacher only if the discontinuity explains the failed formal order.
2. Park L-021 with revival conditioned on external or new evidence for a
   numerically coherent finite-horizon Wan teacher.

## Non-goals

- Do not append RK4, adaptive integration, more NFE, a new sigma grid, or relaxed
  thresholds to EXP-042.
- Do not access calibration, validation, or test identities.
- Do not change C-021, the method family, or the mainline without an accepted RDR.

## Validation

The plan closes only when the researcher selects one option and the resulting
RDR, status, candidate state, and next Gate are internally consistent under
strict Research Control Plane validation.

## Stop and escalation rules

No experiment may run under this plan. Escalate every protocol or portfolio
change to the researcher.
