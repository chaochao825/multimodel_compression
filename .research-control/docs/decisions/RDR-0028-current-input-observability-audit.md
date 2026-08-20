# RDR-0028: Accept the current-input denoising observability audit

- Status: accepted
- Date: 2026-08-20
- Decider: researcher through the explicit acceptance of the fresh
  current-input observability audit with four additional scientific constraints
- Supersedes: PLAN-053 successor selection only; all previous null results remain
  immutable

## Context

The externally recorded EXP-004 closed predictors conditioned only on past block
residuals. Its strongest causal method improved scalar AR(2) by only 1.001x, while
a target-visible token temporal/transport oracle improved Layer 24 by 5.271x.
That gap supports one bounded observability question: whether the missing dynamic
coordinate is cheaply visible in the current block input, current timestep
conditioning, or a preregistered lightweight Q/K sketch.

The external proposal used the identifier RDR-0009. The canonical repository
already owns RDR-0009 and EXP-005, so this accepted decision is mapped to
RDR-0028, C-024, L-024, EXP-045, and G-024 without changing its scientific
meaning.

## Options

1. Stop post-hoc denoising-time residual prediction after the past-only null.
2. Reuse the exposed EXP-004 endpoints to train or select a new predictor.
3. Run a fresh, identity-disjoint current-input observability audit before any
   router training or latency claim.

## Decision

Select option 3. EXP-045 tests denoising-time observability, not physical state
prediction from one generated second to the next. It uses fresh F17 trajectories,
Wan2.1-T2V-1.3B, layers 20--29, sampler steps 4 and 6, and both CFG branches.

The causal predictor may use only exact residuals from blocks that were actually
executed, current and previous block inputs, current AdaLN conditioning, and a
preregistered cheap Q/K sketch. A sketch requiring selected Q/K projection rows
must include that projection in its arithmetic and measured runtime cost.

Multisecant/Broyden state may be updated only from consecutive exact block
executions. Evaluation must include one-, two-, and three-step open-loop skips;
predicted intermediate residuals never become exact secant observations.
Transport operators act on the F x H x W token lattice with nonperiodic
boundaries, never on the arbitrary hidden-channel ordering.

Oracle recovery is defined linearly in risk:

`(R_AR2 - R_method) / (R_AR2 - R_oracle)`.

Gain ratios or target-visible oracle coordinates cannot select a runnable method.

## Consequences

- L-024 becomes the sole mainline and EXP-045 is the only active execution Gate.
- Four calibration and four selection identities may be opened first. The four
  final identities remain unopened unless the no-training Gate passes and a
  width-32 router is frozen using calibration plus selection only.
- A method must achieve at least 2x risk reduction versus matched AR(2) in at
  least 6 of 10 contiguous layers at both target steps, recover at least half of
  the oracle gap, and avoid greater than 10% harm to either CFG branch.
- Open-loop horizons 2 and 3 must not be worse than their matched AR(2) baseline
  in aggregate; otherwise a one-step pass is treated as unstable and the Gate
  closes.
- A no-training failure means only that these low-cost current observables are
  insufficient for the tested late-layer denoising coordinate. It does not
  refute video sparsity, physical-time transport, or train-native state models.
- No rollout, VAE decode, router training, or H200 speed claim is permitted before
  G-024 passes.
