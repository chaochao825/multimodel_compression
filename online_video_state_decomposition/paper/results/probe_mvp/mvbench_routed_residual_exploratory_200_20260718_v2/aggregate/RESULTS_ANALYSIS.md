# Compressed Native Feature-Memory Analysis

- Completed checkpoints: 200.
- Configuration fingerprints: 1.

## Variant Summary

| Selector | Memory | Accuracy | Steady-state MiB | Cold-start MiB | Compression | Selected error |
|---|---|---:|---:|---:|---:|---:|
| Learned recent-pool top-k | full | 51.00% | 8.024 | 8.024 | 1.00x | 0.0000 |
| Learned recent-pool top-k | pca_r256_route_grid2_s4 | 51.00% | 1.024 | 3.031 | 7.84x | 0.0513 |
| Learned recent-pool top-k | pca_r256_s0 | 50.00% | 0.524 | 2.531 | 15.32x | 0.0600 |

## Paired Accuracy Versus Full Cache

Non-inferiority margin: 2.0%. The decision uses the one-sided 95% Clopper-Pearson upper bound on full-correct/compressed-wrong outcomes.

| Selector | Memory | Gain | 95% CI | Prediction agreement | Better / worse | Worse upper 95% | Non-inferior |
|---|---|---:|---:|---:|---:|---:|---:|
| Learned recent-pool top-k | pca_r256_route_grid2_s4 | +0.00% | [+0.00%, +0.00%] | 98.50% | 0 / 0 | 1.49% | yes |
| Learned recent-pool top-k | pca_r256_s0 | -1.00% | [-3.00%, +1.00%] | 97.00% | 1 / 3 | 3.83% | no |

## Query-Conditioned Selector Gain at Matched State

| Memory | Candidate versus exact recent | Gain | 95% CI | Better / worse | McNemar p |
|---|---|---:|---:|---:|---:|

## Claim Boundary

- PCA and sparse residual coding are established compression tools. This experiment tests task preservation and systems trade-offs, not mathematical novelty.
- Shared codec parameters and per-stream state are reported separately. Cold-start state includes the shared codec for compressed variants; steady-state state does not amortize it into every stream.
- A lower reconstruction error is not sufficient; promotion requires preserving full-cache LLaVA accuracy.
- The non-inferiority gate is conservative: compressed improvements do not offset full-correct/compressed-wrong events.

## Figures

![Accuracy versus state](accuracy_vs_state.png)

![Reconstruction versus state](reconstruction_vs_state.png)

![Task deltas versus full cache](task_delta_vs_full.png)

![Finite-sample preservation gate](preservation_gate.png)
