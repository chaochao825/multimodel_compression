# Fixed-Budget Memory Retention: Formal Probe Analysis

## Status and Scope

This report covers a verified representation-level diagnostic on 30 stratified
Video-MME segments, six each from the proxy categories `static`,
`camera_motion`, `object_motion`, `scene_cut`, and `high_change`.

The probe uses the LLaVA CLIP ViT-L/14-336 visual tower, 16 sampled frames per
segment, layers 0/7/15/22/23, and a 4x4 region grid. Region vectors are centered
within each frame and L2-normalized. A delayed region vector is used as a query,
and each fixed-budget memory returns an approximation.

This is evidence about hidden-state retention. It is not language-conditioned
retrieval, online QA accuracy, or long-duration streaming evidence.

All formal runs passed independent artifact validation:

- 30 input segments and 30 per-run checkpoints;
- finite aggregate values;
- non-empty CSV, JSON, PNG, and PDF artifacts;
- 117,000 raw rows for each main comparison;
- 39,000 raw rows for each instant-window allocation ablation.

## Main Result

A compressed state improves delayed hidden-vector retention, but a memory-only
state dilutes current evidence. Reserving an exact instant cache removes that
failure.

At capacity 32, corresponding to about 64 KiB of FP16 state for a 1024-wide
CLIP layer:

| Method | Delay-0 cosine | Mean delay-8 gain vs recent | Delay-8 gain range across layers |
|---|---:|---:|---:|
| Memory-only Oja subspace | 0.82-0.91 | +0.159 | +0.131 to +0.200 |
| Instant + Oja | 1.000 | +0.165 | +0.133 to +0.202 |
| Instant + adaptive slots | 1.000 | +0.148 | weaker than Oja on most dynamic slices |

The matched-byte `instant_oja` state therefore preserves the current frame
exactly while improving the delay-8 reconstruction proxy at every tested layer.

## Category Slices

For capacity 32, the strongest `instant_oja` gains occur on non-static
segments. Representative delay-8 gains versus the recent-window baseline are:

| Category | Layer 15 | Layer 22 | Layer 23 |
|---|---:|---:|---:|
| camera motion | +0.288 | +0.197 | +0.198 |
| high change | +0.206 | +0.160 | +0.158 |
| object motion | +0.178 | +0.141 | +0.140 |
| scene cut | +0.237 | +0.166 | +0.165 |
| static | +0.003 | +0.005 | +0.005 |

This pattern is desirable for a bounded memory: static content is already
retained by a recent cache, while dynamic segments benefit from consolidation.
It does not yet establish semantic event retention.

## Budget Allocation

At capacity 64, approximately 128 KiB of FP16 state, varying the number of
exact instant frames gives the following Oja allocation:

| Instant frames | Long-term rank | Exact-delay range | Mean delay-4 gain | Mean delay-8 gain | Read FLOPs/query token | Update FLOPs/input token |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 48 | delay 0 | +0.018 | +0.126 | 229,376 | 294,912 |
| 2 | 32 | delays 0-1 | +0.022 | +0.126 | 196,608 | 196,608 |
| 3 | 16 | delays 0-2 | +0.029 | +0.126 | 163,840 | 98,304 |

Three instant frames plus rank-16 long-term state is the strongest tested
allocation. It matches the recent baseline through delay 2, improves delay 4
and delay 8, preserves the same delay-8 gain as larger long-term ranks, and has
the lowest Oja read/write proxy among the three allocations.

The result suggests that the next system should not spend all retained bytes on
a larger recurrent subspace. A small exact recent window plus a modest
long-term subspace is a better tested allocation.

## Systems Accounting

At capacity 32, both recent-window and `instant_oja` retain 65,552 bytes
including metadata. The Oja hybrid increases the proxy read cost from 65,536 to
98,304 FLOPs per query token and adds 98,304 update FLOPs per input token.

At capacity 64 with three instant frames, both methods retain 131,088 bytes.
The hybrid uses 163,840 read FLOPs per query token and 98,304 update FLOPs per
input token. Actual GPU latency remains unmeasured and must be reported before
making an efficiency claim.

## Decision

The representation probe promotes the following configuration to the task MVP:

`three-frame instant cache + rank-16 causal subspace memory`

Use layer 22 as the primary LLaVA-compatible interface, keep layer 15 as a
dynamic-content ablation, and compare against:

- capacity-matched recent window;
- uniform reservoir;
- instant-only cache;
- instant plus adaptive slots;
- the same Oja memory without an instant cache.

The next gate is language-conditioned delayed-query accuracy on an online-video
benchmark. The current probe cannot establish semantic usefulness because the
query is the target hidden vector itself and the tested delay spans only a
short sampled segment.

## Artifacts

- `memory_formal_centered_v2_20260717/`: memory-only comparison.
- `memory_hybrid_formal_centered_v1_20260717/`: matched-byte instant-memory comparison.
- `memory_hybrid_instant2_cap64_20260717/`: two-frame allocation.
- `memory_hybrid_instant3_cap64_20260717/`: three-frame allocation.
- `memory_instant_window_ablation_20260717/memory_instant_window_ablation.csv`
- `memory_instant_window_ablation_20260717/memory_instant_window_ablation.png`
- `memory_instant_window_ablation_20260717/memory_instant_window_ablation.pdf`
