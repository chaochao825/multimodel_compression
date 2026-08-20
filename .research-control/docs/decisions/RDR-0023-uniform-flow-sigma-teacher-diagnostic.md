# RDR-0023: Diagnose EXP-042 in the actual flow-sigma coordinate

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit instruction to pursue an
  accuracy-first validation without a fixed resource ceiling
- Scope: development-only engineering diagnosis; no method or claim update

## Context

The frozen shifted-base Euler-100/Euler-200 control completed with valid,
deterministic artifacts but failed convergence at `29.0653%` aggregate and
`40.0219%` worst endpoint relative L2. Initial states and first velocities are
identical, while the error grows along the trajectory. This is an engineering
failure before any curvature-method observation.

The current grid is uniform before the Wan shift transform. In the ODE
coordinate used by the update, its 100-step final interval is approximately
`0.04808`, not `0.01`. The production shift is useful for allocating a small
number of sampler evaluations, but it need not be a suitable high-accuracy
teacher integration grid.

## Decision

Authorize one development-only diagnostic that keeps the model, prompts,
seeds, CFG arithmetic, explicit Euler update, BF16 model execution, FP32 state
update, payload schema, and convergence thresholds fixed, while making sigma
itself uniform. This is implemented by the existing frozen capture code with
`shift=1`, so no model/evaluator source changes are required.

Run Euler-100 and Euler-200 for both development identities. If the original
teacher rule passes, record the result and propose a prospective protocol
amendment before opening calibration. If it still fails, a single Euler-400
worst-identity diagnostic may be used to distinguish first-order contraction
from nonconvergence; it cannot select a deployable method or unlock later data.

The uniform 100/200 run subsequently failed at `50.4400%` aggregate and
`55.4407%` worst, with exact initial-state equality. Because uniform spacing
made the result worse, the diagnosis must separate convergence order from grid
allocation. Authorize one 200/400 comparison on the worst identity of each
already-observed grid: shifted `dev01_turntable` and uniform `dev00_cloth`.
This is still a two-point development engineering diagnostic, not a sweep.

## Guards

- Do not access calibration, validation, or test identities.
- Do not change the method family, operator budget, target, or scientific gate.
- Do not call the diagnostic a positive curvature result.
- Keep QKV, block activations, and VAE decode disabled.
- Stop after a valid teacher-grid diagnosis and record all adverse results.

## Consequences

- PLAN-047 closes with an engineering failure rather than a scientific null.
- PLAN-048 becomes the sole active plan.
- EXP-042 remains running but curvature training remains locked.
