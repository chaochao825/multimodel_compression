# Online Video Generation Acceleration Priority Audit

Evidence cutoff: 2026-07-16. Paper results below are author-reported claims
from primary arXiv sources, not results reproduced by this project.

## Decision

For online or interactive video generation, block-circulant plus low-rank
structure should be treated as one component in a hybrid accelerator, not the
dominant architecture. The current research priority is:

1. reduce the number of denoising function evaluations;
2. use a causal frame-wise or chunk-wise streaming architecture;
3. preserve expressive sparse or structured high-rank attention;
4. bound and reuse cross-step and cross-frame state;
5. lower arithmetic and cache precision;
6. schedule requests under latency SLOs;
7. optimize block-circulant or low-rank subpaths only where measurements show
   a favorable kernel regime.

This ordering is consistent with the strongest recent generation systems.

## 1. NFE and Causal Streaming Are First-Order

Few-step distillation changes the number of complete model evaluations and
therefore attacks a larger cost term than replacing one attention operator.

| Work | Primary source | Main consequence |
|---|---|---|
| LongLive | https://arxiv.org/abs/2509.22622 | Frame-level causal AR generation, short-window attention, frame sink, and interactive prompt updates |
| Causal Forcing | https://arxiv.org/abs/2602.02214 | Bridges bidirectional-to-causal distillation with an AR teacher |
| Diagonal Distillation | https://arxiv.org/abs/2603.09488 | Uses asymmetric denoising schedules for streaming AR video |
| Causal Forcing++ | https://arxiv.org/abs/2605.15141 | Extends causal distillation to frame-wise one- or two-step generation |
| CausalCine | https://arxiv.org/abs/2605.12496 | Causal multi-shot generation with bounded content-aware memory |
| TurboDiffusion | https://arxiv.org/abs/2512.16093 | Combines step distillation, attention acceleration, quantization, and systems optimization |

Consequence for this project: a generation paper cannot credibly lead with
BCCB unless NFE and causal rollout are already controlled.

## 2. Cache and Expressive Sparse Attention Are Crowded

Recent work does not assume that video attention is merely low rank or
element-wise sparse. It retains dynamic, periodic, dense, or retrieval-based
paths.

| Work | Primary source | Main consequence |
|---|---|---|
| Fast AR Video Diffusion | https://arxiv.org/abs/2602.01801 | Temporal KV compression plus ANN sparse self- and cross-attention |
| VMonarch | https://arxiv.org/abs/2601.22275 | Dynamic spatiotemporal Monarch structure for Video DiT attention |
| MonarchRT | https://arxiv.org/abs/2602.12271 | Periodic, sparse, and dense mixing with custom kernels for real-time generation |
| Forcing-KV | https://arxiv.org/abs/2605.09681 | Head-specialized static and dynamic KV compression |
| RoPeSLR | https://arxiv.org/abs/2605.20659 | 3D-RoPE-guided sparse plus low-rank attention |
| LongLive-RAG | https://arxiv.org/abs/2606.02553 | Content-addressable retrieval over generated latent history |

Consequence: a block-circulant path must be gated and compared against
Monarch, sparse-low-rank, retrieval, and cache-compression alternatives. Pure
BCCB is not a sufficient novelty claim.

## 3. Low Precision Is Already a Full-System Axis

| Work | Primary source | Main consequence |
|---|---|---|
| DVD-Quant | https://arxiv.org/abs/2505.18663 | Data-free Video DiT PTQ and W4A4 evaluation |
| QuantSparse | https://arxiv.org/abs/2509.23681 | Joint quantization and attention sparsification |
| TurboDiffusion | https://arxiv.org/abs/2512.16093 | W8A8 plus low-bit attention in a combined acceleration stack |
| 6Bit-Diffusion | https://arxiv.org/abs/2603.18742 | Runtime NVFP4/INT8 allocation plus temporal delta caching |
| LongLive-2.0 | https://arxiv.org/abs/2605.18739 | NVFP4 training, inference, KV cache, and streaming VAE decoding |

Consequence: generator-delta coding alone is too narrow. Any low-precision
claim must include real kernels, activations, caches, and end-to-end quality.

## 4. Serving and SLO Scheduling Are Separate Contributions

| Work | Primary source | Main consequence |
|---|---|---|
| GENSERVE | https://arxiv.org/abs/2604.04335 | Step-level preemption, elastic parallelism, batching, and SLO-aware scheduling for mixed T2I/T2V loads |
| FlashDiff | https://arxiv.org/abs/2607.12121 | Region-level execution and affinity-aware scheduling across diffusion requests |

Model FLOPs do not establish serving value. A systems claim needs arrival
traces, queueing, preemption, P50/P95/P99 latency, throughput, SLO attainment,
and quality under load.

## 5. Position of Block-Circulant and Low-Rank Structure

Block-circulant structure remains potentially useful for:

- a gated local spatial path in selected layers or timesteps;
- compact kernel generators;
- compressed memory or router projections;
- frequency-domain background transport;
- low-cost stable-region updates;
- hardware-friendly batched FFT subgraphs.

It should not be assumed to replace full 3D attention globally. The project
already observes weak Qwen3-VL BCCB fit, no meaningful BCCB advantage over
BTTB in temporal transport, and a large gap between correctness-oriented
Python FFT code and a deployable fused kernel.

## 6. Relation to the Understanding-Side Direction

The conceptual decomposition

`structured transport + persistent state + sparse innovation`

is still a useful hypothesis for online video understanding, but the current
task evidence does not validate its existing instantiation:

- BCCB transport failed its formal promotion gate;
- residual-magnitude event routing failed all 25 joint cells;
- unsupervised Oja memory failed the MVBench task-transfer gate.

The next understanding-side method should therefore be:

`exact recent evidence + query-conditioned learned semantic/episodic memory`,

with structured transport and event routing reintroduced only as individually
validated optional paths. This is a stronger evidence-aligned direction than
either pure token pruning or direct low-rankification of a Video DiT.

## Final Recommendation

Continue the online-video-understanding line first, but revise the proposed
three-path method from a fixed structural decomposition into a learnable,
query-conditioned bounded-memory system. Keep block-circulant and low-rank
operators as co-designed acceleration modules after task benefit is
established. Defer a generation-first paper unless the project can integrate
few-step causal generation, bounded cache, low precision, and measured
serving behavior rather than optimizing attention structure in isolation.
