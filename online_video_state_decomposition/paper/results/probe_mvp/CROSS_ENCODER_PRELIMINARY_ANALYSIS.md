# Cross-Encoder Preliminary Probe Analysis

Date: 2026-07-17

## Scope and evidence boundary

This comparison uses the same three 16-frame Video-MME development segments
with two independently pretrained visual encoders:

- Qwen3-VL-30B-A3B-FP8 visual blocks 0, 8, 16, and 26.
- LLaVA-1.5 CLIP-ViT-L-336 visual blocks 0, 7, 15, 22, and 23.

The result is cross-encoder evidence, but it is not yet cross-dataset evidence.
There are no semantic event labels, delayed-query task metrics, optical-flow
controls, or matched-byte online-memory baselines in this comparison.

## Gate summary

| Probe | Qwen3-VL | LLaVA-CLIP | Preliminary decision |
|---|---|---|---|
| Rank-32 state spectrum | Selected layers exceed 70%; layer 8 does not; layer 26 is collapsed | Layers 0, 15, 22, and 23 exceed 70%; layer 7 does not | Retain only as a layer-selective candidate |
| Rank-32 causal preservation | 43-59% error at non-collapsed layers | 34-63% error | Offline spectrum alone is insufficient |
| Top-10% residual energy | 33-41% | 43-84%, with 84% only at layer 15 | Energy concentration is not consistently replicated |
| Change/event proxy recall | 14-37% | 12-43% | The 80% event gate fails |
| Local BCCB improvement | At most 5.9% | At most 6.1% | The 10% transport gate fails twice |
| BCCB versus BTTB | Nearly identical | Nearly identical | No specifically cyclic advantage |

## Probe 1: persistent state

After excluding Qwen layer 26 because its centered top-1 component explains
99.2% of spatial variation, rank-32 centered-history energy exceeds 70% at:

- Qwen layers 0 and 16.
- CLIP layers 0, 15, 22, and 23.

This is reproducible evidence that some visual blocks contain a compact feature
subspace. It is not evidence that one fixed rank works across the encoder.

Two controls materially narrow the claim:

1. CLIP layer 15 falls from 88.6% centered-history energy to 68.1% after token
   normalization, showing that feature magnitude contributes much of its
   apparent compactness.
2. At rank 32, non-collapsed causal projection errors remain 43-59% for Qwen
   and 34-63% for CLIP. A compact offline spectrum therefore does not guarantee
   an accurate online state update.

Preliminary decision: **retain a layer-selective low-dimensional state probe,
but do not claim that the full Probe 1 gate has passed**. The next gate must use
matched-byte recent-window, reservoir, and online-subspace baselines on a
delayed-query task.

## Probe 2: sparse innovation

Qwen does not approach the 70% top-10% residual-energy gate. CLIP layer 15 does
reach 84.3%, consistently across the three clips, but:

- centered top-1 energy is 62-71% and effective rank is only 6-11;
- token normalization removes much of the low-rank signal;
- pixel-change proxy recall is only 24.9%;
- local BTTB/BCCB prediction does not improve the proxy recall.

The CLIP layer-15 result is therefore more consistent with magnitude-dominated
or low-dimensional token outliers than with semantically meaningful sparse
events.

Preliminary decision: **the event-sparsity gate fails**. Keep the question open
only for object/region residuals with annotated semantic events and scene cuts.

## Probe 3: structured transport

The best local-BCCB stable-error improvements are:

- 5.9% at Qwen layer 26, which is itself a collapsed representation.
- 6.1% at CLIP layers 22 and 23.

At every layer, local BCCB and zero-padded local BTTB differ by less than about
0.5 percentage points of relative improvement. Pair-fitted global BCCB can
look strong, but causal transfer can be neutral or harmful, including severe
deep-layer failures.

Preliminary decision: **remove BCCB from the title and primary contribution
path now**. Retain BCCB as a negative/control baseline and an optional kernel
only if later motion-stratified data shows a regime where it clearly beats
BTTB and optical flow.

## Revised development priority

1. Build a memory-only MVP around layer-selective online subspace tracking.
2. Compare it against recent window, instant cache, reservoir, and truncated
   SVD at exactly matched retained bytes.
3. Evaluate current-scene and delayed-query quality separately.
4. Expand to 16-32 motion/event-stratified clips and add optical flow.
5. Revisit sparse routing only with semantic event labels.
6. Do not implement the full three-path model unless at least two mechanisms
   pass their causal gates.

## Artifacts

- `cross_encoder/cross_encoder_probe_summary.csv`
- `cross_encoder/cross_encoder_probe_summary.json`
- `cross_encoder/cross_encoder_probe_summary.png`
- `cross_encoder/cross_encoder_probe_summary.pdf`
- `qwen_dev_v2_download/aggregate/`
- `clip_dev_download/aggregate/`
