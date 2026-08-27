# RESULT-EXP-048: Distillation-induced low-rate state closure cross

- Status: complete
- Validity: valid prospective null/adverse result
- Date: 2026-08-27
- Gate: G-027
- Claim: C-027
- Candidate: L-027

## Registered outcome

`null/adverse`. At rank 64, rCM weights on the rCM4 trajectory passed 0/10
capacity layers and 0/10 H1 layers. No layer improved by at least 25% over
teacher weights on both frozen input trajectories. H2/H3 stability also failed.

The four untouched selection identities produced all 22,080 expected finite
rows. No selection value fitted or selected a basis, transition, rank, threshold,
or fallback.

## Principal measurements

| Cell | Rank-64 capacity output L2 | Rank-64 H1 output L2 |
|---|---:|---:|
| teacher weights / native4 trajectory | 10.721% | 20.473% |
| teacher weights / rCM4 trajectory | 12.958% | 22.858% |
| rCM weights / native4 trajectory | 22.569% | 31.753% |
| rCM weights / rCM4 trajectory | 22.460% | 30.843% |

At fixed native4 inputs, rCM weights worsened H1 by 55.10%; at fixed rCM4
inputs they worsened H1 by 34.93%. The rCM4 trajectory reduced the rCM model's
H1 error by only 2.86%, yielding a negative interaction that partially offsets
but does not reverse the adverse weight effect.

For rCM/rCM4, rank-64 H2/H3 aggregate errors were 33.096%/35.734%, with
54.047%/56.884% worst rows. The two-lag H1 advantage was only 0.137%, and the
shared-basis H1 penalty was only 0.049%.

## Mechanism interpretation

The result does not fail because the state is one lag short or because teacher
and rCM use incompatible hidden gauges. Both registered diagnostics passed while
absolute capacity failed by a large margin.

The more direct limitation is residual concentration. On held-out rCM/rCM4,
rank 64 captured 82.627% of residual energy, corresponding to 41.681% residual
L2 and 22.460% whole-block-output L2. Teacher/native4 captured 94.726% and still
had 10.721% output error. The strict 0.5% output endpoint requires essentially
complete residual energy, not an apparently high 80%--95% PCA statistic.

Rank growth also plateaued: rCM/rCM4 capacity moved from 26.288% at rank 16 to
22.460% at rank 64, while H1 moved only from 31.791% to 30.843%.

## Supports

- The 2x2 design identifies an on-policy interaction: the rCM4 trajectory
  slightly compensates the rCM weight effect.
- Two-lag history and a model-specific basis are not the dominant bottlenecks
  for this registered family.
- Whole-block output metrics are necessary; residual energy capture alone would
  substantially overstate usefulness.

## Does not support

- Released four-step distillation does not automatically induce broad rank-64
  late-layer Markov closure.
- The current basis does not justify observer/router training, rank extension,
  renderer kernels, or approximate rollout.
- rCM-on-rCM behavior cannot be attributed solely to simpler rCM4 latents; the
  fixed-trajectory weight effects are adverse.

## Boundary

This result does not evaluate released rCM video quality or speed, which remain
under EXP-047. It also does not refute same-step attention/FFN optimization,
physical-time long-video memory, or a model trained with an explicit persistent
state, innovation, renderer, and multi-step closure objective.

## Decision

Refute C-027 within the registered F17 late-block boundary and park L-027. Stop
the post-hoc cross-denoising state line. Any training-native state architecture
or same-step bottleneck study requires a new accepted decision; EXP-047 remains
the unchanged mainline.

## Integrity

- Engineering smoke attempt02 preserved exact Wan outputs under capture with
  `torch.equal`; two earlier load-only failures were retained.
- Calibration contained exactly indices 0--3 and selection exactly 4--7.
- The official source commit and both checkpoint identities matched the frozen
  config.
- Calibration, fitting, selection, and analysis exited successfully.
- Seventeen relevant remote tests passed.
- The remote root used 2.4 GiB, within the 30 GiB cap.
- No latency claim is made because a stopped foreign process retained memory on
  GPU2 and selection time was dominated by CPU FP64 metric recomputation.

## Evidence

- `worldfoundry_hybrid_residual/results/WAN_RCM_STATE_CLOSURE_EXP048_20260827.zh-CN.md`
- `worldfoundry_hybrid_residual/results/wan_rcm_state_closure_exp048_20260827/selection_v1/cell_metrics.csv`
- `worldfoundry_hybrid_residual/results/wan_rcm_state_closure_exp048_20260827/analysis_v1/summary.json`
- `worldfoundry_hybrid_residual/results/wan_rcm_state_closure_exp048_20260827/analysis_v1/block_gate_summary.csv`
- `worldfoundry_hybrid_residual/results/wan_rcm_state_closure_exp048_20260827/analysis_v1/cross_effects.csv`
- `worldfoundry_hybrid_residual/results/wan_rcm_state_closure_exp048_20260827/analysis_v1/mechanism_sweep.csv`
- `worldfoundry_hybrid_residual/results/wan_rcm_state_closure_exp048_20260827/analysis_v1/h1_cross_by_block.png`
- `worldfoundry_hybrid_residual/results/wan_rcm_state_closure_exp048_20260827/analysis_v1/open_loop_by_block.png`
