# PLAN-047: EXP-042 development teacher-convergence capture

- Status: complete; engineering failure isolated before method evaluation
- Owner: researcher and agent
- Gate: G-021
- Claims: C-021
- Candidate: L-021
- Lane: explore
- Resource scope: two H200s, no fixed GPU-hour ceiling, development identities
  only, Euler-100 and Euler-200, tensor storage below 1 GiB for this stage

## Decision question

Does the explicit shifted-sigma teacher produce schema-valid trajectories, and
is Euler-100 sufficiently converged relative to Euler-200 under the frozen rule?

## Authorized work

1. Capture the two development identities with Euler-100 and Euler-200 using
   exact two-rank CFG branch parallelism.
2. Store only five FP32 states and four FP32 guided velocities per identity on
   `/opt/data` with atomic writes and SHA-256 manifests.
3. Compare terminal latents. Use Euler-100 only when pooled and worst relative
   L2 are both at most 0.25%; otherwise use Euler-200 when both are at most 0.5%.
4. Log and repair reproducible engineering failures without changing the solver,
   identities, tensors, thresholds, or method family.

## Prohibited work

- Do not capture calibration, validation, or test identities.
- Do not load QKV/block activations, decode VAE output, train a curvature model,
  or inspect a downstream quality endpoint.

## Exit mapping

- Valid teacher selection: freeze resolved config/capture hashes, close this
  plan, and open calibration/validation execution.
- Error above 0.5%: engineering failure; diagnose teacher discretization before
  any method claim.
- Schema/hash/solver mismatch: invalid; repair and rerun development only.

## Completion outcome

- Euler-100 and Euler-200 both produced two schema-valid development payloads.
  Every payload contains exactly five FP32 latent states and four FP32 guided
  velocities with finite values; initial latents and first guided velocities
  match exactly across resolutions.
- Payload size is `75,483,506` or `75,483,630` bytes per identity. The four
  payloads total `301,934,272` bytes, and all manifest SHA-256 values match.
- Euler-100 versus Euler-200 endpoint relative L2 is `29.0653%` aggregate and
  `40.0219%` worst, so the frozen `0.5%` engineering threshold fails by a wide
  margin. No curvature arm, calibration identity, validation identity, or test
  identity was opened.
- The error grows monotonically from identical initial conditions. On the
  worst identity the five matched-state errors are `0%`, `1.9232%`, `6.8505%`,
  `19.3775%`, and `40.0219%`. This rules out seed mismatch and points to teacher
  time-grid discretization or vector-field stiffness.
- The shifted-base grid has a final Euler step of about `0.04808` at 100 steps,
  despite a nominal base-grid interval of `0.01`. PLAN-048 tests a true uniform
  flow-sigma grid using development identities only.
