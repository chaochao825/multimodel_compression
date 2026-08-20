# RDR-0004: Reuse structure as a bounded risk scout

- Status: accepted
- Date: 2026-08-11
- Decider: researcher through the instruction to revisit the original intuition,
  incorporate recent DiT evidence, and pursue the simplest promising direction
- Supersedes: none

## Context

Fixed BCM/BCCB, token Butterfly, frozen low-rank tails, train-free content tails,
and scalar temporal predictors failed their registered transfer or cost gates.
Those failures reject high-dimensional output replacement under the tested
function classes; they do not test whether inexpensive structure statistics can
predict a scalar approximation risk. Existing deployable centroid/moment rows
also expose a narrow opportunity: every fixed configuration has an optimistic
attention ceiling below 1.5x, while a target-leaking choice among nine existing
configurations and dense fallback is near that boundary.

Recent video-DiT systems obtain speed from current-content routing, aligned
tiles, exact fallback, training or calibration, and fused kernels. SPADE's SICS
is especially relevant evidence that within-block Q/K structure can summarize
attention behavior without constructing the full attention matrix. This also
means that a generic SICS router is not a novel contribution by itself.

## Options

1. Continue increasing BCM, Butterfly, low-rank, or temporal predictor capacity.
2. Start a learned sparse-attention or fused-kernel program immediately.
3. Run one calibration-frozen screen asking whether Q/K/V structure can select
   among existing precompiled approximate actions without reconstructing AV.

## Decision

Select option 3 as one bounded side probe. The structured module may emit only
a scalar risk score. It may use post-RoPE Q/K/V norms, value leverage, SICS,
axis-neighbor cosine, and finite-difference spectral proxies, plus static
layer/step/head identity and the existing deployable moment-mass proxy. It may
not inspect dense attention probabilities, AV outputs, held-out errors, or an
adaptive low-rank basis at inference-selection time.

The action set is frozen to dense fallback and the nine existing
centroid/moment configurations. The split is frozen to one training identity,
one validation identity, and two untouched test identities. This decision does
not change the stopped temporal-transition mainline and does not authorize a
kernel, rollout, or speed claim.

## Consequences

- A positive arithmetic screen can open a fresh-data timing/rollout protocol;
  it cannot establish H200 acceleration or endpoint fidelity.
- If the target-leaking multi-action envelope is below 1.5x, stop before model
  fitting. If the envelope passes but the frozen scout fails, stop the
  train-free scout and require a new researcher decision for learned routing.
- Fixed structured-output expansion remains parked. BCM/Toeplitz/Fourier
  statistics survive only as interpretable risk features unless new evidence
  establishes an aligned output operator.
