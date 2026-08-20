# PLAN-046: Freeze EXP-042 capture and evaluator identities

- Status: complete
- Owner: researcher and agent
- Gate: G-021 accepted, but GPU execution remains locked until this plan closes
- Claims: C-021
- Candidate: L-021
- Lane: explore
- Resource scope: no fixed GPU-hour ceiling after implementation freeze; this
  plan itself authorizes only local/CPU tests and read-only remote diagnostics

## Decision question

Can a minimal explicit-Euler Wan capture and common curvature evaluator be made
schema-correct, numerically exact, identity-safe, and reproducible before any
decision-bearing GPU execution?

## Required implementation

1. Capture only states at `0,25,50,75,100` and guided velocities at
   `0,25,50,75` on the frozen shifted-sigma Euler teacher.
2. Keep model forward BF16 and latent arithmetic/payload FP32.
3. Write atomic per-identity artifacts, manifests, hashes, runtime/software
   identities, and explicit split labels without loading unopened test data.
4. Implement motion estimation, nonperiodic warping, equal-budget operator arms,
   curvature/endpoint/composition metrics, and staged test unlocking.
5. Add unit, synthetic-recovery, schema, leakage, boundary, and dry-run tests.

## Exit mapping

- Valid implementation: freeze hashes, register EXP-042 as running, close this
  plan, and open one staged execution plan beginning with development only.
- Engineering failure: repair without changing the scientific contract and
  repeat local validation.
- Contract mismatch: revise protocol prospectively before any GPU execution.

## Completion outcome

- Twelve pure logic/schema tests and four torch operator tests pass locally or
  in the canonical remote environment; all capture/evaluator imports and Python
  compilation pass remotely.
- Local and remote hashes match for every frozen source/config file. Wan model
  and solver hashes match the prior artifact audit.
- The deployed remote project is a source copy rather than a Git checkout, so
  provenance binds the local Git HEAD plus remote file hashes explicitly.
- Implementation identities are frozen in
  `results/motion_conjugated_flow_curvature_exp042_implementation_freeze.json`.
  No model forward or tensor payload was produced under PLAN-046.
