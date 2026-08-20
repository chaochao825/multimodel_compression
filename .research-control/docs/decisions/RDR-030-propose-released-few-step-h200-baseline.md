# RDR-030: Propose a released few-step H200 baseline after EXP-046

- Status: proposed
- Date: 2026-08-20
- Decider: researcher
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

## Proposed decision

Accept option 1. The baseline Gate must distinguish NFE reduction from per-step
acceleration, include text/DiT/VAE timing, VRAM, multi-prompt/multi-seed quality
and temporal metrics, and make no claim that the student validates EXP-046.

## Consequences if accepted

- A new candidate, claim, protocol, and bounded H200 Gate may be registered.
- EXP-046 final identities remain locked and are not reused automatically.
- No training-native state work begins until the released baseline is measured
  or a later accepted decision supersedes this ordering.

## Revisit condition

Reject or supersede this proposal if no reproducible Wan-compatible released
checkpoint can be integrated under a bounded effort, or if the research goal
explicitly prioritizes architecture novelty over a deployment baseline.
