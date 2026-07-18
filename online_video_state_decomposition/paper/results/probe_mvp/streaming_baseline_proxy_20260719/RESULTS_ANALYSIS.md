# Streaming Baseline Mechanism-Proxy Comparison

This run replays 200 frozen MVBench CLIP caches from a reused development set. It is a mechanism and accounting comparison, not an official end-to-end reproduction of any external method.

## Overall

| Method | Tier | Accuracy | Evidence | Active KiB | Archive KiB | Detailed KiB | Total KiB | Total bounded |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| Exact recent | `project_native_control` | 50.0% | 8.0 | 12.00 | 0.00 | 0.00 | 12.05 | yes |
| CausalMem proxy | `official_mechanism_feature_proxy` | 49.0% | 8.0 | 24.00 | 0.00 | 0.00 | 24.09 | yes |
| StreamingTOM proxy | `official_mechanism_feature_group_proxy` | 47.5% | 8.0 | 12.00 | 6.00 | 0.00 | 18.25 | no |
| STC proxy | `official_mechanism_feature_group_proxy` | 52.5% | 8.0 | 13.50 | 0.00 | 0.00 | 13.59 | yes |
| SelectStream proxy | `paper_mechanism_feature_proxy_untrained` | 49.0% | 8.0 | 16.73 | 0.00 | 0.00 | 17.22 | yes |
| OASIS proxy | `official_structure_feature_proxy_no_mllm_summaries` | 51.5% | 8.0 | 24.00 | 22.50 | 0.00 | 47.50 | no |
| StateKV proxy | `paper_mechanism_feature_proxy` | 51.5% | 32.0 | 24.00 | 0.00 | 48.00 | 72.41 | no |
| Ours: learned selector (dev-fitted) | `project_native_selector_feature_proxy` | 53.0% | 8.0 | 24.00 | 0.00 | 0.00 | 24.19 | yes |

## Paired Against Exact Recent

| Method | Gain | 95% bootstrap interval | Better / worse | McNemar p |
|---|---:|---:|---:|---:|
| CausalMem proxy | -1.0 pp | [-6.0, +4.0] | 11 / 13 | 0.8388 |
| StreamingTOM proxy | -2.5 pp | [-7.5, +3.0] | 12 / 17 | 0.4583 |
| STC proxy | +2.5 pp | [-2.0, +7.0] | 13 / 8 | 0.3833 |
| SelectStream proxy | -1.0 pp | [-4.5, +2.5] | 6 / 8 | 0.7905 |
| OASIS proxy | +1.5 pp | [-2.0, +5.0] | 8 / 5 | 0.5811 |
| StateKV proxy | +1.5 pp | [-2.5, +5.5] | 10 / 7 | 0.6291 |
| Ours: learned selector (dev-fitted) | +3.0 pp | [+0.5, +6.0] | 7 / 1 | 0.0703 |

## Interpretation Boundary

The strongest total-bounded proxy is **Ours: learned selector (dev-fitted)** at 53.0% on this frozen CLIP proxy. This does not transfer paper-reported quality between different backbones or benchmarks.

- The learned selector was fitted on development evidence from this reused sample pool. Its result is post-hoc and is not an independent generalization estimate.
- CausalMem is reduced from projected token memory to one frame vector per observation; its background-token merge is intentionally omitted because it degenerates at that resolution.
- StreamingTOM includes a 4-bit feature-group archive and bounded active read, but not its real KV kernels or end-to-end TTFT.
- STC uses frame-group reuse and dynamic/uniform pruning proxies; real ViT token reuse and visual-token pruning require the official stack.
- SelectStream lacks public code in this audit and its learned segment encoder, graph attention, calibration, and training are not reproduced.
- OASIS uses vector event centroids instead of MLLM event summaries and intent-driven tool calls.
- StateKV correctly counts its fixed cstate separately from the growing detailed decode cache; its apparent quality is not a matched fixed-total-memory comparison.
- The row for our method evaluates the frozen learned selector only. The routed low-rank/spatial residual codec is analyzed in the separate native LLaVA feature-memory confirmation.

CPU latency plots measure NumPy feature replay only. Official P50/P95/P99 GPU latency remains a separate reproduction gate.
