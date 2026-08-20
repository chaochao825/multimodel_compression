# PLAN-053: Execute the current-input denoising observability Gate

- Status: complete
- Owner: researcher
- Gate: G-024
- Claim: C-024
- Candidate: L-024
- Lane: explore
- Resource scope: one locked H200, four calibration and four selection identities,
  at most four GPU-hours before the no-training decision

## Objective

Determine whether cheap current observables recover the late-layer dynamic
coordinate that past-residual-only predictors missed, while preserving exact
history causality and testing one-, two-, and three-step open-loop stability.

## Execution order

1. Freeze fresh identity splits, observables, arithmetic accounting, methods,
   oracle recovery formula, and G-024 thresholds.
2. Implement capture equivalence and synthetic recovery/leakage tests.
3. Run one small smoke identity without opening the final split.
4. Run calibration for implementation checks, then selection for G-024.
5. Generate per-layer/step/branch/horizon tables and visualizations.
6. Train no router and open no final identity unless every no-training Gate passes.

## Stop rule

Stop after one valid G-024 decision, any dense-equivalence failure, information
leakage, foreign GPU overlap, four GPU-hours, or resource exhaustion. A horizon-1
pass with horizon-2/3 instability is a Gate failure, not permission to add another
predictor family.

## Decision effect

If G-024 passes, freeze one width-32 router using calibration and selection only,
then request the final-split stage. If it fails, close low-cost current-input
observability for these cells without extrapolating to physical-time transport,
video sparsity, or train-native state models.

## Outcome

G-024 returned `FAIL`. The complete eight-identity run exited 0 with no final
identity opened. L-024 is parked, C-024 is refuted within scope, and no router or
latency stage is authorized. The program now requires explicit successor
selection rather than another variation inside this function class.
