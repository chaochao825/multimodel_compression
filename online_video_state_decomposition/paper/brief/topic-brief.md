# Topic Brief

- **Working title**: Probe-First Two-Timescale Memory for Budgeted Online Video Understanding
- **Target**: MLSys main track or CVPR systems/efficient vision track; IEEEtran arXiv preprint during development
- **Assumed length**: 8-10 pages of main text, excluding references and appendix
- **Paper type**: Empirical algorithm-systems paper
- **As-of date**: 2026-07-16
- **Audience**: Researchers in streaming video-language models, efficient vision systems, structured operators, and hardware-aware ML

## Research Question

Can a fixed byte budget preserve both current and delayed online-video evidence
more effectively when it is split between:

1. a small exact recent cache; and
2. a layer-selective causal low-dimensional state for older evidence?

Structured transport and sparse innovation remain falsification probes and
controls, not assumed components of the final method.

## Scope

In scope:

- frozen-encoder probes on aligned hidden states;
- exact-recent versus consolidated-memory budget allocation;
- causal online subspace memory;
- recent-window, reservoir, adaptive-slot, and memory-only controls;
- local BTTB/BCCB, optical flow, and residual concentration as failed or
  optional probe evidence;
- causal memory ablations;
- fixed-memory and fixed-average-compute evaluation;
- measured P50/P95/P99 latency and memory scaling.

Out of scope for the first paper:

- universal replacement of video attention by BCCB;
- a claim that all temporal states or BCM residuals are low rank;
- end-to-end online video generation as the primary task;
- BCCB or sparse-event routing as a primary method contribution;
- ASIC/RTL implementation before a software kernel shows a real latency win;
- using DRE-BCM weight fitting as evidence for temporal-state low rank.

## Existing Evidence

Verified local evidence supports a negative-to-hybrid transition:

- Qwen3-VL visual attention has weak average BCCB fit.
- Zero-shot BCCB/BCM replacement leaves large attention-output error.
- Wan2.2 contains geometry-aligned but head/layer/timestep-dependent cyclic structure.
- Oracle hybrid decomposition is accurate on selected maps, but static support transfer fails.
- Ranked sparse routes are task-functional in one saved ViT/SCTM model.
- DRE-BCM shows a separate parameter-space lesson: structured matrices need corrective residuals, but this does not prove a temporal-memory claim.
- A formal 30-segment CLIP probe finds mean rank-32 token-normalized energy of
  0.766, with 19 of 25 category-layer cells above the 0.70 gate.
- The same formal probe rejects the sparse-event proxy and finds only 0.037%
  mean BCCB improvement over BTTB.
- A matched-byte representation benchmark promotes three exact instant frames
  plus rank-16 Oja memory to the task MVP.

## Closest Prior Families

- Structured attention controls: Circulant Attention, MonarchAttention, VMonarch, MonarchRT, RoPeSLR.
- Bounded streaming memory: Flash-VStream, VideoChat-Online, LiveVLM, StreamingTOM, StateKV, CausalMem.
- Event-centric online understanding: Event-VStream, EventMemAgent.
- State and decomposition antecedents: VideoMamba, online robust PCA, dynamic mode decomposition.

## Core Constraint

Novelty is provisional until the two-timescale state improves
language-conditioned task quality and measured latency under matched total
bytes. Representation retention alone cannot establish semantic usefulness or
a systems advantage.
