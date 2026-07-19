# OASIS Official One-Video Smoke Analysis

## Evidence scope

This is an audited model-level smoke run of the official OASIS evaluator on
StreamingBench Real-Time Visual Understanding sample 1. It is not a formal
method comparison. The run covers one video and five multiple-choice questions;
the formal 50-video, 250-question run remains in progress.

The runner used the pinned OASIS source, local Qwen3-VL-8B-Instruct and
Qwen3-Embedding-0.6B weights, seed `20260719`, 2 fps input sampling, and the
official OASIS memory settings recorded in `preflight.json`. The runtime passed
an actual BF16 FlashAttention 2.8.3 CUDA kernel check on an NVIDIA A800 80GB.

## Validated outcome

| Metric | Observed value | Interpretation |
| --- | ---: | --- |
| Videos completed | 1 / 1 | Full smoke coverage |
| Questions scored | 5 / 5 | Full smoke coverage |
| Correct | 5 | 100% on this five-question subset only |
| Sampled video frames | 376 | 2 fps from 4,707 source frames at 25 fps |
| Whole-run wall time | 309.65 s | Offline `pace=0` evaluator time, not TTFT or request latency |
| Peak evaluator GPU memory | 26,674 MiB (26.05 GiB) | Sampled NVML process memory |
| Peak total GPU memory used | 26,697 MiB (26.07 GiB) | GPU was otherwise effectively idle |
| Peak GPU utilization | 98% | Short generation bursts reached full compute |
| Median GPU utilization | 23% | Decode, CPU work, and model calls alternate |
| Samples at or above 20% utilization | 68.9% | Computed over 383 monitor samples |

The 100% smoke accuracy must not be placed beside formal CausalMem or other
50x5 scores. `quality_smoke.png` is intentionally a separate single-run figure.

## Mechanism observation

The official output shows the event archive operating rather than merely loading
the model. OASIS formed long-term event summaries and accumulated prior QA
history while retaining recent windows. One of five responses invoked
`rag_retrieval`; the other current or near-current questions were answered from
visible recent evidence. This is consistent with the intended role of OASIS as
a slow, on-demand event archive. A single video is insufficient to estimate a
reliable retrieval rate or establish a quality advantage.

The memory trace rises in steps from model residency near 18 GiB to a peak near
26 GiB. This is a process-residency observation, not proof that logical archive
state grows without bound: NVML includes PyTorch's CUDA caching allocator. The
formal run and explicit active/archive byte accounting are needed before making
a state-growth claim.

The utilization trace is bursty, with repeated 68-98% peaks separated by low
utilization intervals. That pattern supports treating OASIS as a quality/archive
baseline rather than a real-time latency baseline. Its whole-run wall time must
not be mixed with STC stage P50/P95/P99 or request SLO measurements.

## Provenance

| Item | Value |
| --- | --- |
| Run fingerprint | `e1d4a0a044fbd0670cd5dc4bc553e82f6bb23839de7bd2ce0e974de0e40567fc` |
| Official output SHA-256 | `2e475dafdab00c85f77415744fe93d73262ed803304d0df4a8703c208968091c` |
| Audited result SHA-256 | `df3f7f63d19f4fae5565ef46a207dec1454e490f1ed78d8399cee615b88e631e` |
| Raw GPU samples SHA-256 | `cb47d01e39333b8b3a2671ae1c36c2452ef052696e5bfdce636c33efdfdfd818` |

`result.json` is the authoritative completion record. `official_output.json`
contains question-level predictions, `gpu_samples.csv` is the raw monitor trace,
and `gpu_trace_summary.json` cross-checks its count and peaks against the result.
The normalized CSV and both PNG/PDF figures are retained for reproducible plots.
