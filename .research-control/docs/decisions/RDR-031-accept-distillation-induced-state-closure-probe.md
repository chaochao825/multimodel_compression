# RDR-031: Accept a distillation-induced state-closure side probe

- Status: accepted
- Date: 2026-08-27
- Decider: researcher
- Supersedes: none

## Context

EXP-045 and EXP-046 closed the post-hoc teacher-state families registered there:
cheap current observables covered only three late layers, and even target-visible
rank-96 correction passed no whole-block capacity cell. These results do not
answer whether few-step distillation changes the hidden dynamics themselves.

EXP-047 remains the accepted deployment-baseline mainline. The researcher has
separately authorized restarting the scientific analysis and explicitly allowed
the currently compute-idle H200 on server 236. The new work must not modify the
frozen EXP-047 methods, identities, thresholds, or outputs.

## Options

1. Open a bounded side probe comparing teacher and rCM weights on both native4
   and rCM4 on-policy latent trajectories at fixed low-rate state payload.
2. Resume larger teacher-only post-hoc rank, BCM, Butterfly, or observer sweeps.
3. Train a new state architecture before establishing that rCM induces a more
   closed low-rate dynamics.

## Evidence

- EXP-045 found a useful current-input diagonal field in only L21/L24/L25 and
  failed the registered breadth and open-loop Gate.
- EXP-046 found rank-64 aggregate/worst output errors of 4.833%/11.093% and
  rank-96 errors of 4.345%/10.210%, with 0/60 passing cells.
- The official rCM checkpoint changes both weights and its on-policy trajectory.
  A two-by-two weight/trajectory cross is required to separate these effects.
- H200 GPU2 on server 236 has a stopped foreign process holding memory but zero
  compute. This permits a numerical experiment but invalidates timing claims.

## Decision

Accept option 1 as L-027 / EXP-048 / G-027, a side probe under PLAN-057. The
probe fits rank-16/32/64 calibration-only token states and stagewise first-order
innovation dynamics, then evaluates held-out H1/H2/H3 rollout for all four
`weight x input-trajectory` cells. Both model-specific and shared bases are
required so a hidden-gauge change cannot be mislabeled as reduced closure rank.

## Consequences

- L-026 remains the sole mainline and EXP-047 remains frozen and running.
- L-027 is the only active side probe. It may use H200 GPU2 for numerical work,
  but makes no latency, energy, or exclusivity claim.
- Selection identities cannot fit bases, coefficients, ranks, thresholds, or
  fallback rules.
- State-Closure Distillation is authorized only after G-027 establishes broad,
  absolute, open-loop low-rate closure rather than a relative trend alone.
- A null stops this cross-denoising state line and does not refute same-step
  attention bottlenecks, train-native state/render separation, or physical-time
  long-video memory.
