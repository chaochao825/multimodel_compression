# PLAN-048: EXP-042 flow-sigma teacher-grid diagnostic

- Status: complete; Euler teacher rejected before method evaluation
- Owner: researcher and agent
- Gate: G-021
- Claims: C-021, unchanged and untested
- Candidate: L-021
- Lane: explore
- Resource scope: two H200s, development identities only, Euler-100 and
  Euler-200 on a uniform flow-sigma grid; one Euler-400 refinement for the
  worst identity on each of the shifted and uniform grids

## Decision question

Was PLAN-047's failure caused by applying explicit Euler to a strongly
nonuniform shifted-base grid rather than to the actual flow-sigma coordinate?

## Authorized work

1. Reuse the frozen capture code and run the same two development identities
   with `shift=1`, giving a uniform sigma grid and unchanged Euler arithmetic.
2. Store exactly the registered five states and four guided velocities per
   identity under a diagnostic namespace on `/opt/data`.
3. Verify hashes, schema, finite values, exact initial-state equality, and the
   same `0.25%/0.5%` convergence decision.
4. Because uniform 100/200 exceeds `0.5%`, run shifted `dev01_turntable` and
   uniform `dev00_cloth` at Euler-400, then measure their 200/400 contraction
   without loading any other identity or output.

## Prohibited work

- No calibration, validation, or test access.
- No curvature fitting, rank/expert sweep, VAE decode, QKV capture, rollout, or
  quality/speed claim.
- No post-result change to the method endpoint or scientific thresholds.

## Exit mapping

- Uniform 100/200 at most `0.25%`: propose uniform-sigma Euler-100 as teacher.
- Uniform 100/200 in `(0.25%, 0.5%]`: propose uniform-sigma Euler-200 as teacher.
- Above `0.5%` with clear 200/400 contraction: quantify required resolution and
  propose a prospective high-order or adaptive teacher protocol.
- Above `0.5%` without contraction: close EXP-042 as a teacher engineering
  failure; do not evaluate the method.

## Execution note

- The shifted Euler-400 refinement completed before an unrelated Wan2.2 job
  entered GPU2. Its `200 -> 400` endpoint difference is `24.8149%` for
  `dev01_turntable`, versus `40.0219%` for `100 -> 200`.
- The first uniform Euler-400 attempt was stopped as soon as the unrelated job
  was detected at about 121 GB on GPU2. It produced no tensor payload. Its logs
  were moved intact to
  `/opt/data/wangmeiqi/trash/20260812-220500-exp042-uniform400-contaminated/`.
- The launcher now rechecks both H200s before every capture and resumes from the
  verified shifted result instead of recomputing it.
- RDR-0024 additionally permits a one-H200 sequential-CFG path only after a
  complete shifted Euler-100 trajectory matches the two-H200 reference at
  worst tensor relative L2 `<=1e-6`. This is an execution-equivalence gate, not
  a new teacher or method arm.

## Completion outcome

- Sequential CFG passed more strongly than required: all five saved states and
  all four saved guided velocities are bitwise identical to the two-H200
  reference (`worst relative L2 = 0`). Communication and branch ordering are
  therefore excluded as causes of teacher nonconvergence.
- Shifted-grid `200 -> 400` endpoint relative L2 is `24.8149%` on its frozen
  worst development identity. Relative to `40.0219%` at `100 -> 200`, the
  contraction ratio is `0.6200` and empirical order is `0.6896`.
- Uniform-grid `200 -> 400` endpoint relative L2 is `23.1600%` on its frozen
  worst development identity. Relative to `55.4407%` at `100 -> 200`, the
  contraction ratio is `0.4177` and empirical order is `1.2593`.
- Both are more than 46 times above the `0.5%` teacher guard. Under the more
  optimistic uniform empirical order, a naive Euler pair would require roughly
  `4.2k vs 8.4k` steps; shifted-grid extrapolation is substantially worse.
- All payloads, manifests, hashes, schemas, finite-value checks, and split
  guards pass. No calibration, validation, test, curvature target, or method
  arm was opened. RDR-0025/PLAN-049 replace Euler with one bounded explicit
  midpoint RK2 teacher diagnostic on the same ODE coordinate.
