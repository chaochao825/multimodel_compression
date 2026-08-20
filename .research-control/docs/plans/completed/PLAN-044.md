# PLAN-044: Prospective motion-conjugated flow-curvature Gate design

- Status: complete
- Owner: researcher and agent
- Gate: none; this plan authorizes no execution
- Claims: C-021
- Candidate: L-021
- Lane: explore
- Resource cap: zero GPU-hours, zero fresh captures, zero training, zero rollout,
  zero model or environment changes, and zero held-out payload loading

## Decision question

Can the existing immutable Wan trajectory artifacts support a leakage-safe,
equal-budget test of whether current-observable motion alignment makes
finite-interval flow curvature materially more structured and transferable?

## Frozen conceptual contract

1. Denoising time, physical video time, and transformer depth are distinct.
2. The target is the finite-interval curvature remainder after the exact current
   instantaneous-velocity term, not an old block feature or attention defect.
3. Compare identical payloads in raw-noisy, clean-prediction, and motion-aligned
   coordinates. Coordinate choice is the treatment; structure is held fixed.
4. The deployment warp may use only current observable predictions and frozen
   calibration parameters. Target video, endpoint, future teacher state, and
   held-out residual are prohibited.
5. Use nonperiodic boundaries by default. BCCB is only a periodic negative
   control or a numerical embedding whose boundary error is separately reported.
6. Report target-exposed capacity, frozen transfer, composition defect, endpoint
   error, arithmetic, memory, and measured cost as separate quantities.

## Required protocol draft

- Freeze model, checkpoint, sampler, scheduler, prompt, seed, interval, CFG,
  precision, trajectory, and code identities.
- Freeze disjoint development, calibration, validation, and untouched test
  identities before computing a test metric.
- Define the exact curvature target and numerically stable small-interval rule.
- Define an observable warp and a target-exposed oracle warp as different arms.
- Compare zero-curvature Euler, same-budget unaligned structure, clean-coordinate
  structure, motion-conjugated structure, and same-budget unstructured capacity.
- Freeze parameter/MAC/payload budgets and one common evaluator.
- Register pass, null, adverse, boundary, leakage-invalid, and engineering-fail
  mappings before opening EXP-042/G-021.

## Prospective stop/go boundary

- Capacity continuation: target-exposed motion-conjugated structure must reach
  aggregate/worst 4-NFE latent endpoint relative L2 at most 0.5%/1%.
- Deployable continuation: frozen current-observable alignment must reduce the
  held-out risk-weighted curvature error by at least 25% versus the identical
  unaligned structure and reach endpoint relative L2 at most 1%/2%.
- Cost continuation: adapter arithmetic must be at most 10% of one Wan NFE and
  coordinate construction must have a bounded fused implementation path.
- Any miss parks the structured branch. Do not add ranks, experts, BCM variants,
  sparse residuals, cache, fallback, or relaxed identities after seeing results.

## Exit mapping

- Existing artifacts sufficient: complete PLAN-044 and present a frozen
  EXP-042/G-021 protocol for researcher acceptance; do not execute it.
- Existing artifacts insufficient: complete PLAN-044 with the minimum fresh
  capture identities, tensors, hashes, and H200 budget; request acceptance.
- Related-work overlap removes the distinct claim: narrow or park L-021 through
  a new protected decision rather than silently changing the endpoint.
- Any compute or held-out access under PLAN-044: invalid and non-evidentiary.

## Completion outcome

- Existing artifacts are insufficient. They contain final latents and 72 F81
  full-QKV UniPC captures, but no identity-indexed per-step latent state paired
  with its guided model velocity.
- UniPC history makes adjacent captured states unsuitable as a Markov
  finite-horizon flow-map target. The minimum valid fresh source is an explicit
  normalized-sigma Euler teacher that saves five coarse states and four guided
  velocities per identity; no QKV or block activations are required.
- The proposed leakage-safe protocol, 22-identity split, machine-readable
  configuration, artifact audit, mathematical utilities, and seven unit tests
  now exist. EXP-042/G-021 remains proposed and must not execute without
  explicit researcher acceptance and final capture/evaluator hashes.
- PLAN-044 used zero GPU-hours, loaded no tensor payload, changed no remote
  environment, and made no scientific outcome claim.
