# RDR-030: Accept a released few-step H200 baseline after EXP-046

- Status: accepted
- Date: 2026-08-20
- Decider: researcher through the explicit request to continue only when the
  successor has evidence of potential and moves toward the original faithful
  Wan acceleration objective
- Supersedes: none

## Context

EXP-046 returned a valid target-visible capacity null. Rank 64 reached
4.833%/11.093% aggregate/worst block-output L2 and passed 0/60 cells; rank 96
also passed 0/60. The accepted EXP-046 outcome map stops coordinate-observer,
rank-rescue, rollout, and kernel work for L-025.

Public full-observability few-step video students remain positive external
evidence, but they have not been measured on this repository's prompts, seeds,
quality metrics, checkpoint stack, and H200 runtime. Starting a new state
student without that baseline would leave its practical target undefined.

## Options

1. **Released few-step baseline first (recommended).** Select one reproducible
   Wan-compatible released checkpoint, freeze prompts/seeds and quality/latency
   metrics, then measure exact end-to-end H200 Pareto and component profile.
2. **Training-native state immediately.** Register a different shared decoder
   and recurrent transition with fresh identities and own-state rollout loss.
   This has higher cost and lacks a measured incumbent.
3. **System-only continuation.** Stop learned algorithmic state work and retain
   exact dual-H200 CFG, FP8/BF16 dense attention, and separately validated
   sparse-attention candidates.

## Decision

Accept option 1 and register C-026, L-026, EXP-047, G-026, and PLAN-056. The
primary candidate is the official unquantized four-step rCM checkpoint for
Wan2.1-T2V-1.3B. Native 20-step UniPC is the incumbent and the unchanged native
four-step model is the matched NFE control. TurboDiffusion is literature and
future system context only because it combines distillation, sparse attention,
quantization, and custom kernels.

The Gate must distinguish NFE reduction from per-step acceleration, include
text/DiT/VAE timing, VRAM, multi-prompt/multi-seed quality and temporal metrics,
and make no claim that the student validates EXP-046. Paired SSIM/PSNR are
diagnostic because a distilled generator need not preserve the teacher's
sample-wise trajectory; semantic, temporal, and diversity metrics decide the
quality guard.

## Consequences

- L-026 becomes the sole active mainline and EXP-047 becomes the sole active
  experiment Gate.
- EXP-046 final identities remain locked and are not reused automatically.
- No training-native state work begins until the released baseline is measured
  or a later accepted decision supersedes this ordering.
- The official repository and checkpoint are pinned before the first scientific
  output. A one-prompt integration smoke may repair engineering issues but may
  not alter the formal prompts, seeds, methods, thresholds, or metrics.

## Revisit condition

Reject or supersede this proposal if no reproducible Wan-compatible released
checkpoint can be integrated under a bounded effort, or if the research goal
explicitly prioritizes architecture novelty over a deployment baseline.
