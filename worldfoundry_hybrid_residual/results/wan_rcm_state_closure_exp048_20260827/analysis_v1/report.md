# EXP-048 distillation-induced state-closure result

## Verdict

G-027 outcome: **null/adverse**.

- Capacity-pass layers: `[]` / 10.
- H1-pass layers: `[]` / 10.
- Joint capacity/H1-pass layers: `[]` / 10.
- Weight-improvement-pass layers on both trajectories: `[]` / 10.
- Pooled H2 aggregate/worst: `33.096%` / `54.047%`.
- Pooled H3 aggregate/worst: `35.734%` / `56.884%`.
- Two-lag H1 advantage: `0.137%`.
- Shared-basis H1 penalty: `0.049%`.

## Interpretation

This endpoint tests denoising-time low-rate state closure, not physical-time video memory.
All bases and transition coefficients are calibration-only; the four selection identities
are used only once for fixed evaluation. Whole-block-output error, rather than residual
energy capture, determines every G-027 threshold. The 2x2 cross separates the effect of
rCM weights from the effect of evaluating on the rCM4 latent trajectory.

No latency claim is made because GPU2 retained a stopped co-resident process during the
numerical experiment.
