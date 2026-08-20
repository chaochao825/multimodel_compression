# RDR-0001: Test denoising-step state before further structured operators

- Status: accepted
- Date: 2026-08-07
- Decider: researcher
- Supersedes: none

## Context

Fixed BCM/BCCB, fixed Toeplitz lifting, static frozen low-rank corrections, and
cross-frame HSS state have failed their strict local transfer tests. Those
negative results concern spatial/operator structure or autoregressive memory;
they do not establish whether adjacent denoising steps have a cheap causal
state transition. Existing TeaCache and tri-mode results also show that reuse
or schedule search alone is insufficient under strict quality.

## Options

1. Continue enlarging static spatial/operator structures.
2. Stop all train-free acceleration work.
3. Run one bounded denoising-step Gate comparing reuse, Taylor, L2P-style
   causal fitting, stagewise AR state, shallow current-input innovation, and a
   clearly labelled per-sample oracle.

## Evidence

- Static LongLive far-state rank-16 event model: 26.78% aggregate and 49.28%
  worst held-out AV error.
- Fixed future-risk Toeplitz lifting worsened the Layer-14 adaptive-rank result
  from 4.42% to 5.37-5.61%.
- Strict existing Wan TeaCache pilot: mean paired frame SSIM 0.9621.
- Existing tri-mode train-free operator/schedule oracle: 1.0x at SSIM 0.98.

These are scope-limited negative results, not evidence against a denoising-step
transition fitted only on calibration trajectories.

## Decision

Adopt option 3 as the sole active mainline. Freeze a representation/transfer
screen first; do not implement rollout control or kernels until it passes.

## Consequences

- Fixed structured-operator expansion remains parked.
- Held-out oracle coefficients are diagnostic only.
- A GO opens a new untouched rollout/H200 protocol; it does not itself support
  final-video quality or speed claims.
- Revisit static structured operators only if a new residual-shaping mechanism
  changes their previously failed adaptive bound.
