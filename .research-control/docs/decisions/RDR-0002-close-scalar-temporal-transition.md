# RDR-0002: Close the scalar denoising-step transition line

- Status: accepted
- Date: 2026-08-07
- Decider: researcher through the preaccepted EXP-000 stop rule
- Supersedes: none

## Context

`EXP-000` prospectively tested the exact Gate selected in `RDR-0001`. The
per-sample scalar AR(3)+innovation oracle failed before rollout or kernel work.

## Options

1. Relax the 1%/2% quality threshold after seeing the result.
2. Increase scalar order or tune stages on held-out identities.
3. Close the scalar family and preserve the negative evidence.

## Evidence

F81 horizon-2 oracle coverage was 0/192 at the frozen quality guard, with 8.344%
full-scope aggregate error. The calibration-only candidate was 8.589%, so
coefficient transfer was not the dominant gap. Only block 0 showed a small
one-step feasible region, which is insufficient for the speed target.

## Decision

Select option 3 according to the stop rule. Do not run skipped-block rollout or
H200 kernels for this family.

## Consequences

- `C-000` is refuted within its registered Wan/F81 scope.
- `L-000` is decided and stopped.
- Offline low temporal rank remains a diagnostic observation, not evidence that
  a causal predictor exists.
- A content-conditioned/channel-subspace state requires a new researcher-approved
  claim, because it changes the function class and overlaps recent prior work.
