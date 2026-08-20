# RDR-0025: Replace the nonconverged Euler teacher with bounded midpoint RK2

- Status: accepted as an accuracy-first engineering repair
- Date: 2026-08-12
- Decider: researcher through the instruction to explore and validate without a
  fixed resource ceiling, combined with RDR-0022 repair authority
- Scope: development teacher construction only; claim C-021 remains untested

## Context

Euler is deterministic and schema-correct but unusable as a high-accuracy Wan
flow teacher at the registered resolutions. Shifted and uniform grids both
contract from `200 -> 400`, yet remain at `24.8149%` and `23.1600%` endpoint
relative L2. Continuing Euler by brute force would consume thousands to tens of
thousands of NFE and would confound numerical integration error with the
curvature representation question.

The scientific endpoint is the same pretrained guided Wan flow ODE. Replacing
its numerical teacher with explicit midpoint RK2 changes the approximation
order, not the endpoint, model, CFG formula, motion hypothesis, operator family,
identity split, or gate threshold.

## Decision

Authorize one bounded development-only midpoint RK2 diagnostic:

1. Evaluate both previously registered time grids: Wan shifted-base sigma and
   uniform flow sigma.
2. For each grid, capture both development identities at 100 and 200 RK2
   intervals, retaining the same five states and four start-of-interval guided
   velocities.
3. Each RK2 interval uses the exact explicit midpoint update:

       k1 = v(z_j, sigma_j)
       z_mid = z_j - 0.5 * delta * k1
       k_mid = v(z_mid, 0.5 * (sigma_j + sigma_{j+1}))
       z_{j+1} = z_j - delta * k_mid

4. A grid selects RK2-100 only when aggregate and worst 100/200 endpoint error
   are at most `0.25%`; it selects RK2-200 when both are at most `0.5%`.
5. If both grids are eligible, choose lower worst error, then lower aggregate
   error, then shifted grid as a deterministic tie-break. Freeze that teacher
   before calibration. If neither is eligible, classify EXP-042 as a teacher
   engineering failure and stop without adding RK4, adaptive solvers, or more
   NFE in this Gate.

## Execution and leakage guards

- Use single-H200 sequential CFG only because RDR-0024 demonstrated bitwise
  equivalence for all stored tensors; retain the contamination sentinel.
- Model BF16, latent FP32, FA3 BF16, integer model timestep semantics, CFG 5.0,
  prompts, seeds, payload, and threshold remain unchanged.
- No calibration, validation, test, curvature fitting, QKV, VAE decode, or
  downstream quality endpoint is authorized.
- Midpoint results cannot be interpreted as evidence for C-021; they only
  decide whether a valid teacher exists.

## Consequences

- PLAN-048 closes with Euler rejected as a teacher implementation.
- PLAN-049 is the sole active plan.
- A successful RK2 teacher prospectively amends only the teacher solver field;
  every scientific arm and gate remains frozen.
