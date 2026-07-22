# STC Official Stage-Latency Analysis

## Protocol

This matched upstream comparison uses one A800, 64 frames, five warmups, and
20 measured repetitions per mode. ReKV uses 196 visual tokens per frame and a
full update; ReKV+STC uses 64 tokens per frame and an update ratio of 0.25.
Both strict result artifacts, preflights, fingerprints, and GPU traces passed.

## Results

| Instrumented stage | ReKV P50 (ms) | ReKV+STC P50 (ms) | P50 reduction | ReKV mean (ms) | ReKV+STC mean (ms) | Mean reduction |
|---|---:|---:|---:|---:|---:|---:|
| ViT encode | 1681.37 | 527.95 | 68.6% | 1576.18 | 659.61 | 58.2% |
| Visual-token prefill | 7587.77 | 6551.79 | 13.7% | 7692.40 | 6264.16 | 18.6% |
| Per-iteration stage sum | 9761.02 | 7062.34 | 27.6% | 9268.58 | 6923.77 | 25.3% |

The official peak-memory field decreased from 18.42 to 16.46 GB, a 10.6%
reduction. For the summed stages, P95 decreased from 12073.16 to 10995.39 ms,
an 8.9% reduction.

## Interpretation

The median speedup is real within this upstream benchmark, but most of it comes
from ViT encode rather than LLM prefill. The summed-stage coefficient of
variation increased from 15.2% for ReKV to 27.4% for ReKV+STC, and the observed
P95 reduction is much smaller than the P50 reduction. The combined path is
therefore faster on average but more variable relative to its mean in this
20-sample run.

The audited quantile convention is `higher`. With 20 samples, P95 and P99 both
select the maximum observation, so this run is not strong tail-latency evidence.
It also changes caching, token count, and update ratio together, so it cannot
separate the contribution of each STC component.

These measurements cover only ViT encode plus visual-token prefill. They do not
measure task quality, TTFT, decode, request latency, or end-to-end latency, and
they are not a direct comparison against the proposed three-path method.
