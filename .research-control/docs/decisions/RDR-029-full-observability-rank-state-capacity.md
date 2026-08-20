# RDR-029: Test full-observability rank-state capacity before state training

- Status: accepted
- Date: 2026-08-20
- Decider: researcher through the explicit request to combine the prior profile,
  assess the sufficient-state intuition, and execute the bounded next test
- Supersedes: RDR-0028 only as the current mainline selection; EXP-045 and all
  prior nulls remain immutable

## Context

EXP-045 showed that current block input exposes a real one-step Jacobian signal,
but no complete low-cost observer covered the late-layer block population or
remained stable in open loop. Its target-visible 75-shift oracle retained
5.53%--6.86% block-output error, so router training was correctly stopped.

That oracle mixed two questions: whether the missing whole-block defect has a
compact state representation, and whether exact history/current input can infer
the coordinates of that state. The next experiment must separate these before
training a recurrent student.

## Options

1. Increase DPLR rank, secants, Q/K rows, or shift experts under EXP-045.
2. Train a recurrent hidden-state student immediately.
3. First test a target-visible adaptive rank-state ceiling for the complete
   whole-block defect, then authorize observability/transition training only if
   the representation ceiling passes.

## Decision

Select option 3. Register C-025, L-025, EXP-046, G-025, and PLAN-054. The tested
state is a low-rank factorization of the whole-block defect after the frozen
exact-history current-input diagonal renderer. It is a capacity oracle: both
factors may use the target defect and cannot be described as deployable.

Ranks, cells, identities, randomized range-finder settings, arithmetic model,
and thresholds are frozen before selection. Rank 64 is the decision point;
rank 96 is diagnostic and cannot rescue a failed rank-64 Gate.

## Consequences

- L-025 becomes the sole mainline; L-024 is parked as a valid bounded null.
- Four fresh selection identities may be opened after a calibration smoke. Four
  calibration identities are reserved for a later observer, and four final
  identities remain locked.
- A pass only establishes a compact target-visible representation ceiling. It
  authorizes a separate current-input state-coordinate observer Gate, not a
  recurrent student, rollout, kernel, quality, or speed claim.
- A failure closes rank-64 whole-block state prediction under this renderer. It
  does not refute full few-step students, same-step sparse attention, physical
  video-time state, or a training-native architecture with a different state.
