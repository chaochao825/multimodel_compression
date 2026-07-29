# Content-Generated Sparse-Linear Tail Diagnostic

## Decision

**STOP_TRAIN_FREE_CONTENT_TAIL_FOR_LAYER14_STEP09**

Rank-64 transductive capacity reaches 1.179%
per-tile aggregate error and 3.385% worst-tile error after
the additional adaptive rank-16 oracle. Both miss the registered `0.5% / 1.0%`
capacity gate. The stricter basis shared across three query tiles is
1.851% aggregate error.

The calibration-frozen feature map with held-out support oracle reaches
1.332% aggregate
and 3.234% worst-tile
error. The validation-selected train-free proxy (`proxy_value_aware_semantic64`) reaches
1.488% and
3.632%.

## What Improved

Against the old, granularity-matched `25% support-family oracle + per-tile rank-16`
baseline, the rank-64 transductive content tail changes raw content error from
9.569% to 8.399%, per-tile
post-rank error from 1.584% to
1.179%, and worst-tile error from
4.648% to 3.385%. This is useful
tail shaping, but not enough to meet the gate.

Increasing the learned positive feature rank from 16 to 64 changes transductive
per-tile error by only -0.11%.
The plateau means feature width is not the limiting variable; the positive separable
kernel and low-cost support semantics remain mismatched to Layer 14 content.

## Interpretation

The result supports semantic permutation as a candidate-layout improvement, not as a
complete tail solution. SVG2 uses semantic clustering, permutation, top-p control, and
custom kernels; this experiment uses a deliberately labeled SVG2-style Q/K sorting
proxy and does not claim a paper-faithful reproduction. VSA similarly relies on a
trainable coarse-to-fine router and fused block kernel. SLA2 addresses sparse/linear
branch mismatch with learnable routing and branch ratio. Our shared numerator and
denominator removes one normalization error source, yet the remaining Layer 14 defect
still requires roughly 28 output dimensions on average and up to 59 for the 1% record
gate in the transductive rank-64 run.

Therefore Layer 14/step 9 should use FP8 or BF16 dense fallback in this train-free
system. Rotation/chart work remains stopped. A 1k-2k adaptation of this exact tail
family is not justified because even the all-record transductive oracle misses the
capacity gate. Learned sparse-linear methods remain viable as a separate function class,
but require a new registered hypothesis rather than widening this feature map.

## Evidence Boundary

- Four previously explored captures, one layer/step/CFG branch, and three query tiles.
- Transductive and dense-support results are oracle diagnostics, not deployment results.
- The support search is a monotone projected-rank heuristic, not a global optimum.
- Arithmetic speedup is an upper bound; no fused H200 kernel or end-to-end rollout was measured.
- Figures compare per-tile rank-16 only where the old baseline uses the same granularity.

## Related Boundaries

- [Sparse VideoGen2](https://arxiv.org/abs/2505.18875)
- [VSA](https://arxiv.org/abs/2505.13389)
- [SLA2](https://arxiv.org/abs/2602.12675)
- [DynamicRad](https://arxiv.org/abs/2604.20470)
