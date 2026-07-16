# Verified Literature Map

Evidence cutoff: 2026-07-17. Every paper below was checked against its primary
arXiv page. Paper-reported results are prior-work claims, not verified results of
this project. The machine-readable companion is `literature-matrix.csv`.

## 1. Fixed-Budget and Selective Streaming Memory

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| VideoStreaming | https://arxiv.org/abs/2405.16009 | Memory-propagated clip encoding and adaptive question-related memory selection | Propagated bounded memory plus query readout is direct prior art |
| Flash-VStream | https://arxiv.org/abs/2406.08085 | Low-capacity context memory plus high-capacity retrieval memory | Compare against an explicit two-tier memory hierarchy |
| VideoChat-Online / OVBench | https://arxiv.org/abs/2501.00584 | Pyramid Memory Bank and online-video benchmark | Use causal online evaluation rather than offline clip QA alone |
| ReKV | https://arxiv.org/abs/2503.00540 | Sliding-window encoding plus host/disk KV offload and query retrieval | Count CPU/disk state and retrieval latency rather than treating offload as free |
| StreamBridge | https://arxiv.org/abs/2505.05467 | Round-decayed memory compression plus a proactive activation model | Separate memory quality from deciding when to respond |
| LiveVLM | https://arxiv.org/abs/2505.15269 | Training-free streaming KV compression and retrieval | Include a KV-centric training-free systems baseline |
| video-SALMONN S | https://arxiv.org/abs/2510.11129 | Fixed-budget test-time-training memory and prompt-dependent readout | Learned fixed state plus query readout is direct competition |
| rLiVS | https://arxiv.org/abs/2510.17364 | LLM-attention visual-token selection and recurrent context | Include a training-free recurrent selector control |
| LiveStar | https://arxiv.org/abs/2511.05299 | Peak-end memory compression, streaming KV, and response-silence decoding | Measure response timing as well as answer correctness |
| StreamKV | https://arxiv.org/abs/2511.07278 | Semantic-segment KV retrieval and layer-adaptive compression | Compare semantic segmentation against raw chronological retention |
| HERMES | https://arxiv.org/abs/2601.14724 | Hierarchical reusable KV memory informed by attention analysis | Test whether hierarchy alone explains long-delay gains |
| WeaveTime | https://arxiv.org/abs/2602.22142 | Order-aware representation with uncertainty-triggered retrieval | Report temporal-order and uncertainty-triggered slices |
| DSCache | https://arxiv.org/abs/2605.01858 | Decoupled cumulative-past and instant caches | Keep a strong recent/instant cache in every memory comparison |
| SAVEMem | https://arxiv.org/abs/2605.07897 | Three-tier semantic memory with query-aware retrieval | Query-conditioned semantic memory is an essential direct baseline |
| MuKV | https://arxiv.org/abs/2605.22269 | Patch/frame/segment multi-grained KV compression and retrieval | Compare against multiscale memory rather than one granularity |
| StateKV | https://arxiv.org/abs/2605.31598 | Fixed-capacity recurrent state for linear-scaling video prefill | Direct comparator for bounded persistent state |
| SelectStream | https://arxiv.org/abs/2606.16353 | Budgeted latent evidence allocation using surprise and priority | Add a strong recent-window baseline because indiscriminate history can hurt |
| CausalMem | https://arxiv.org/abs/2606.25658 | Training-free online semantic basis under a fixed budget | Closest semantic-basis threat to low-dimensional state |
| FOLIO | https://arxiv.org/abs/2607.13298 | Focused entity memory with short-term buffer and evidence cache | Entity-organized semantic memory is the newest closest comparison |

## 1.1 Query-Conditioned Evidence Selection

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| Query-based frame selection | https://arxiv.org/abs/2601.07459 | Submodular mutual-information frame selection on MVBench | Include a full-history query-selection upper bound |
| Question-aware keyframes | https://arxiv.org/abs/2603.14953 | Synthetic selector supervision plus temporal coverage regularization | Use disjoint calibration/evaluation splits and a diversity control |
| QSVideo | https://arxiv.org/abs/2607.04559 | Structured object/action/location relevance, diversity, and temporal alignment | Relevance-only retrieval is insufficient as the sole query baseline |
| ReQuest | https://arxiv.org/abs/2607.01737 | Learned question-aware selector, uncertainty-triggered rethink, adaptive temporal NMS | Report selector training data and keep adaptive compute separate |

Source-code audit on 2026-07-17 pinned CausalMem at commit
`640104b3786125c4918924f9b666ff7fe04d81de`. The released LLaVA path retains
tokens by online-basis reconstruction residual and merges discarded tokens into
per-frame means. The released Qwen2.5-VL path combines residual energy with an
optional recency score. These writers are query-free, so a question-conditioned
readout is complementary but must still compare against CausalMem-style
query-free retention.

## 2. Spatially Persistent and Event-Centric State

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| Tensor Memory | https://arxiv.org/abs/2605.27686 | Fixed-size recurrent 3D spatial memory with gated write/read | Persistent spatial state itself cannot be claimed as new |
| Long-VMNet | https://arxiv.org/abs/2503.13707 | Fixed-size memory of discriminative visual patches | Include a patch-memory control |
| StreamForest | https://arxiv.org/abs/2509.24871 | Persistent event-memory forest and local spatiotemporal window | Compare local window plus event hierarchy as a combined baseline |
| Event-VStream | https://arxiv.org/abs/2601.15655 | Semantic event boundaries and event-triggered persistent memory | Event triggering and persistent event storage are established |
| EventMemAgent | https://arxiv.org/abs/2602.15329 | Short-term event detector plus long-term event archive | Separate memory quality from tool-use and agent confounds |

## 3. Front-End, Back-End, and Hardware Co-Design

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| VideoLLM-MoD | https://arxiv.org/abs/2408.16730 | Layer-wise skipping of most vision-token computation | Compare state reduction against compute-depth routing |
| StreamingTOM | https://arxiv.org/abs/2510.18269 | Pre-LLM token reduction plus 4-bit online memory | Measure both visual prefill and post-LLM cache costs |
| STC | https://arxiv.org/abs/2512.00891 | Repeated-ViT caching and LLM visual-token pruning | Account for encoder recomputation and LLM prefill separately |
| V-Rex | https://arxiv.org/abs/2512.12284 | Algorithm-hardware co-design for dynamic KV retrieval | Do not claim first online-video algorithm-hardware co-design |
| FlashFFTConv | https://arxiv.org/abs/2311.05908 | Fused long-convolution kernels with tensor-core-aware FFT | Asymptotic FFT cost is insufficient; report measured kernels and I/O |

## 4. Structured Operators and Geometry

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| LoRA | https://arxiv.org/abs/2106.09685 | Low-rank additive weight adaptation | A low-rank residual is established PEFT rather than temporal-state novelty |
| FourierFT | https://arxiv.org/abs/2405.03003 | Sparse Fourier coefficients reconstructing adapter delta weights | Frequency-domain parameter adaptation is direct structured prior art |
| C3A | https://arxiv.org/abs/2407.19342 | High-rank circular-convolution adaptation | Circular high-rank adapters predate any BCCB-plus-low-rank parameter claim |
| Circulant Attention | https://arxiv.org/abs/2512.21542 | Nearest-BCCB modeling and FFT computation for ViT attention | BCCB is prior art and our Qwen evidence is a negative boundary |
| Toeplitz Neural Network | https://arxiv.org/abs/2305.04749 | Toeplitz token mixing for long-range modeling | BTTB/Toeplitz is a mandatory structured control |
| CirCNN | https://arxiv.org/abs/1708.08917 | Block-circulant weights and FFT acceleration | Cite the classical BCM compression origin |
| Structured Transforms | https://arxiv.org/abs/1510.01722 | Displacement-structured transforms for compact networks | Structured matrices are not a new primitive |
| Low Displacement Rank | https://arxiv.org/abs/1810.02309 | Learned displacement operators and low-rank generators | Structured-plus-low-rank has a broader theoretical antecedent |
| Kaleidoscope | https://arxiv.org/abs/2012.14966 | Unified learnable low-rank sparse permutation and Fourier maps | Multi-structure composition is established |
| Monarch Mixer | https://arxiv.org/abs/2310.12109 | Hardware-efficient structured sequence mixing | Compare structured transport with a hardware-aware non-BCCB family |
| BCA | https://arxiv.org/abs/2505.00582 | Block-circulant adapters for language-model adaptation | Relevant to parameter compression but not temporal-state evidence |
| VMonarch | https://arxiv.org/abs/2601.22275 | Structured attention for Video DiTs | Generation-side structured attention is already crowded |
| MonarchRT | https://arxiv.org/abs/2602.12271 | Periodic sparse and dense mixing for real-time generation | Closely overlaps any generation-side hybrid-accelerator framing |
| RoPeSLR | https://arxiv.org/abs/2605.20659 | 3D-RoPE sparse-low-rank Video DiT attention | Sparse plus low rank is not a new generation composition |

## 5. Online Video Generation Acceleration

### Sparse high-rank attention

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| Sparse VideoGen | https://arxiv.org/abs/2502.01776 | Training-free spatial/temporal head profiling with custom sparse kernels | A structured path must beat a strong post-hoc sparse-attention baseline |
| VSA | https://arxiv.org/abs/2505.13389 | Trainable tile-level critical-token attention with hardware-aligned kernels | Sparse high-rank attention is a first-order generation axis |

### Few-step and causal generation

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| LongLive | https://arxiv.org/abs/2509.22622 | Frame-level causal AR generation with short-window attention and a frame sink | Establish causal streaming before claiming operator-level acceleration |
| Causal Forcing | https://arxiv.org/abs/2602.02214 | AR-teacher initialization for causal diffusion distillation | Bidirectional-to-causal distillation is a primary quality bottleneck |
| Diagonal Distillation | https://arxiv.org/abs/2603.09488 | Cross-chunk and denoising-step distillation with asymmetric steps | NFE reduction must be compared before attention replacement |
| Causal Forcing++ | https://arxiv.org/abs/2605.15141 | Frame-wise one- or two-step causal generation | One-to-two-step quality and latency define the current frontier |
| CausalCine | https://arxiv.org/abs/2605.12496 | Causal multi-shot generation with content-aware memory routing | Dynamic prompt and shot changes require bounded semantic memory |
| TurboDiffusion | https://arxiv.org/abs/2512.16093 | Step distillation plus attention acceleration and W8A8 | Isolated BCCB cannot represent a complete acceleration stack |

### Cross-step and cross-frame state

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| Fast AR Video Diffusion | https://arxiv.org/abs/2602.01801 | Temporal KV compression and ANN sparse attention | Compare bounded cache and semantic retrieval |
| Forcing-KV | https://arxiv.org/abs/2605.09681 | Static and dynamic head-specific KV compression | Head specialization is a direct cache baseline |
| LongLive-RAG | https://arxiv.org/abs/2606.02553 | Retrieval over self-generated latent history | Recent-window-only generation is not the only bounded-memory design |
| VMonarch | https://arxiv.org/abs/2601.22275 | Dynamic structured Video DiT attention | Structured attention is already crowded |
| MonarchRT | https://arxiv.org/abs/2602.12271 | Periodic sparse and dense mixing with custom kernels | Closest hybrid structured-attention comparator |
| RoPeSLR | https://arxiv.org/abs/2605.20659 | Geometry-aware sparse-low-rank attention | Sparse plus low rank is established |

### Low precision and full-system execution

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| DVD-Quant | https://arxiv.org/abs/2505.18663 | Data-free W4A4 Video DiT PTQ | Include real low-bit kernels and quality |
| QuantSparse | https://arxiv.org/abs/2509.23681 | Joint quantization and sparsification | Quantization noise and sparsity loss interact |
| 6Bit-Diffusion | https://arxiv.org/abs/2603.18742 | Runtime NVFP4/INT8 allocation and temporal delta cache | Precision can be conditioned on temporal stability |
| LongLive-2.0 | https://arxiv.org/abs/2605.18739 | NVFP4 training inference KV cache and streaming VAE | End-to-end infrastructure is part of the contribution |
| GENSERVE | https://arxiv.org/abs/2604.04335 | SLO-aware mixed T2I/T2V scheduling | Report SLO attainment under request traces |
| FlashDiff | https://arxiv.org/abs/2607.12121 | Regional execution and online scheduling | Model FLOPs alone do not establish serving gains |

## 6. State, Subspace, and Low-Rank/Sparse Antecedents

| Work | Primary source | Covered ground | Required consequence |
|---|---|---|---|
| Dynamic Mode Decomposition for Video | https://arxiv.org/abs/1404.7592 | Low-rank dynamics and sparse foreground separation | Video dynamics decomposition predates neural VLMs |
| Online Supervised Subspace Tracking | https://arxiv.org/abs/1509.00137 | Online predictive subspace updates | Compare simple online subspace estimators in Probe 1 |
| Online Robust PCA | https://arxiv.org/abs/1601.07985 | Online low-dimensional plus sparse decomposition | Low-rank plus sparse is a classical theoretical antecedent |
| OLSTEC | https://arxiv.org/abs/1709.10276 | Online low-rank tensor subspace tracking | Tensorized low-rank state is an established control |
| SlotFormer | https://arxiv.org/abs/2210.05861 | Object-centric slot dynamics for video prediction | Region/object pooling is required before declaring temporal rank high |
| VideoMamba | https://arxiv.org/abs/2403.06977 | Linear-complexity state-space video modeling | Recurrent state alone is not new |

## 7. Benchmarks and Failure Slices

| Benchmark | Primary source | What it measures | Use in this project |
|---|---|---|---|
| ProReady-QA / StreamReady | https://arxiv.org/abs/2603.08620 | Correctness under evidence-window-aware early and late timing penalties | Required proactive-response timing slice |
| StreamingBench | https://arxiv.org/abs/2411.03628 | Real-time perception, omni-stream reasoning, and contextual understanding | Primary broad online benchmark candidate |
| OVO-Bench | https://arxiv.org/abs/2501.05510 | Backward tracing, real-time understanding, and forward active response | Primary temporal-order and delayed-query candidate |
| OVBench | https://arxiv.org/abs/2501.00584 | Online video understanding with memory-augmented evaluation | Compatibility benchmark for VideoChat-Online |
| EventHallusion | https://arxiv.org/abs/2409.16597 | Event-level video hallucination | Rare-event and event-order failure slice |
| VidHalluc | https://arxiv.org/abs/2412.03735 | Video hallucination diagnostics | Hallucination robustness slice |
| VERHallu | https://arxiv.org/abs/2601.10010 | Event-relation hallucination | Relation-sensitive sparse-event slice |
| MoHallBench | https://arxiv.org/abs/2607.01117 | Motion hallucination | Motion/transport-specific failure slice |

## Cross-Paper Conclusions That Constrain Our Design

1. Fixed-budget memory is crowded; the contribution cannot be merely a bounded
   memory bank.
2. Semantic entities, events, and multiscale segments increasingly outperform
   raw chronological retention; our state must be compared with these organizations.
3. A recent-window or instant cache is a strong baseline and may beat polluted
   long-history memory on current-scene questions.
4. Tensor Memory already occupies fixed-size persistent spatial state, so our
   novelty must come from probe-first mechanism attribution and budgeted composition.
5. Both front-end repeated vision encoding/prefill and back-end KV growth matter.
6. Query retrieval, routing, archive metadata, host offload, and synchronization
   must be included in memory and latency accounting.
7. V-Rex prevents any first algorithm-hardware-co-design claim.
8. The BCCB path is optional. It survives only if it beats identity, flow,
   convolution/BTTB, and cache controls in a measurable stable-content regime.
9. On the generation side, NFE reduction and causal rollout are first-order;
   cache, precision, and serving must be controlled before attributing gains to
   a structured attention subpath.
10. MonarchRT, VMonarch, and RoPeSLR make a generic structured-plus-sparse or
    structured-plus-low-rank generation claim insufficient.
11. VideoStreaming and video-SALMONN S make bounded learned state plus
    query-conditioned readout an established design family rather than a
    standalone novelty claim.
12. LiveStar and StreamReady require proactive systems to report when an
    answer is emitted, not only whether the final answer is correct.
13. Sparse VideoGen and VSA make sparse high-rank attention a stronger
    generation-side priority than a global BCCB replacement.

## Provisional Defensible Gap

The current gap is not a new decomposition primitive. It is:

1. probe-first validation of latent dynamics before architecture commitment;
2. causal mechanism attribution among local transport, persistent state, and
   event innovation;
3. a common fixed-state and fixed-average-compute contract;
4. explicit recent-window, semantic-memory, spatial-state, and event-memory
   competitors; and
5. measured end-to-end tail latency including router, cache, archive, retrieval,
   and structured-kernel costs.

The original query-free Oja instantiation fails the matched-budget task gate.
The remaining provisional gap is a query-conditioned learned bounded memory
that beats exact recent evidence while preserving spatial fidelity and
accounting for end-to-end systems cost.
