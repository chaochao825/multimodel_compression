# RDR-0024: Allow sequential CFG only after full-trajectory equivalence

- Status: accepted as an engineering repair under RDR-0022
- Date: 2026-08-12
- Decider: agent under the researcher's accepted accuracy-first engineering
  repair authority; no scientific endpoint or method-family change
- Scope: development teacher diagnostics only

## Context

EXP-042 normally evaluates conditional and unconditional Wan branches on two
H200s in parallel. After the valid shifted Euler-400 result, unrelated jobs
occupied the H200 pair for an extended period. The branch computations have no
cross-branch dependency before the fixed CFG affine combination, so evaluating
them sequentially on one H200 should be numerically equivalent, but that must
be demonstrated rather than assumed.

## Decision

Authorize a single-H200 sequential CFG implementation for development teacher
diagnostics under a strict equivalence gate:

1. Run the complete shifted Euler-100 trajectory for `dev00_cloth` with the
   frozen model, seed, prompts, FA3 BF16, FP32 state update, schedule, and CFG
   formula.
2. Compare every one of its five states and four guided velocities against the
   existing uncontaminated two-H200 payload.
3. Sequential execution is equivalent only if the worst tensor relative L2 is
   at most `1e-6`; hashes, metadata, indices, and finite values must also pass.
4. Only after equivalence passes may it produce the pending uniform Euler-400
   development payload. A failure restores the two-H200 waiting path and does
   not update any scientific claim.

## Guards

- No calibration, validation, test, curvature fitting, VAE decode, or QKV.
- Do not replace FA3, precision, schedule, CFG scale, or payload schema.
- Stop the local process if another large process enters its physical GPU.
- Record sequential provenance explicitly; do not relabel it as two-rank data.

## Consequences

- This reduces resource coupling without weakening numerical comparability.
- PLAN-048 remains the sole active plan and its original outcome map is
  unchanged.
