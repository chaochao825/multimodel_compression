# Streaming Baseline Reproduction Audit

Date: 2026-07-19

## Verdict

The six requested baselines are not equally reproducible from public assets.
This audit therefore separates four evidence tiers:

1. `paper mechanism`: the method described by the paper;
2. `official runnable`: an executable public entry point;
3. `source smoke`: a pinned checkout with required paths and valid Python
   syntax; and
4. `feature proxy`: an independent mechanism approximation on the same frozen
   CLIP cache used by this project.

The current completed experiment is tier 4 plus tier 3, augmented by selected
official-module synthetic smokes. It is not an official end-to-end
reproduction. The first 200-sample result is also a reused development
analysis, not an independent confirmation.

| Method | Intended role | Public implementation status | Current status |
|---|---|---|---|
| CausalMem | Closest query-free semantic-memory algorithm | Public, but the documented evaluation tree is incomplete | Pinned source, FOSSCache module smoke, and frame-vector proxy complete |
| StreamingTOM | Real CTR/OQM systems-latency baseline | Public VideoMME-Short path | Pinned source, OQM module smoke, and feature-group proxy complete; GPU run pending |
| STC | Real ViT-cache and prefill-latency baseline | Public model-specific runners and speed tools | Pinned source/core-import smoke and terminal-feature proxy complete; GPU run pending |
| SelectStream | Task-quality target | No discoverable official code as of the audit date | Paper-structure proxy only |
| OASIS | Slow event-archive baseline | Public evaluation path, no public paper-table timing runner | Pinned source, ShortMemory module smoke, and event-centroid proxy complete; GPU run pending |
| StateKV | Recurrent model-state baseline | Official placeholder repository only | Paper-structure proxy only |

## Source Pins And Licensing

The machine-readable manifest is
`experiments/configs/streaming_baseline_sources.json`. The generated source
audit is stored with the selected result bundle.

| Project | Pinned commit | Source-smoke result | Redistribution boundary |
|---|---|---|---|
| [CausalMem](https://github.com/hktk07/CausalMem) | `640104b3786125c4918924f9b666ff7fe04d81de` | Pass | No repository license found; do not vendor its code |
| [StreamingTOM](https://github.com/YIGE24/StreamingTOM) | `6c66b05065692bc3fa4c6ec7fa9cad84d3b0cd75` | Pass | No root license for the new StreamingTOM code; vendored licenses do not cover it |
| [STC](https://github.com/lern-to-write/STC) | `cf53f781d8740df5c07d7924756acc429641ffd0` | Pass | Apache-2.0 is declared in `pyproject.toml`, but no root license file is present and vendored projects differ |
| [OASIS](https://github.com/Solus-sano/OASIS) | `dbd342c79a1b9b03327d4ec5daa87488737db988` | Pass | Root MIT license present |
| [StateKV](https://github.com/ceyzaguirre4/StateKV) | `ba308d107d24d59b265e952e20ee5172d3f7d670` | Placeholder pass | README only, no code or root license |
| [SelectStream](https://arxiv.org/abs/2606.16353) | Not available | Not run | No discoverable official repository |

The smoke audit validates commit identity, clean worktrees, required entry
files, and `compileall`. It writes bytecode to a temporary external cache so
the audited repositories remain clean. Passing this check does not establish
dependency compatibility or numerical equivalence.

## Executable Module Smoke

The machine-readable output is `official_module_smoke.json` in the selected
result bundle. All four requested public implementations with code passed the
targeted smoke, but the checks have deliberately different strengths:

| Method | Official module exercised | Synthetic result | Evidence boundary |
|---|---|---|---|
| CausalMem | `FOSSCache.process_frame` | 121 tokens retained for a requested budget of 120; rank stayed at 4 | Confirms the default `mem2` path is not a strict total-budget implementation |
| StreamingTOM | `OQM` 4-bit write/read | Key/value MSE `8.10e-4`/`9.13e-4`; packed-key payload ratio `4.0x` | Ratio excludes scales, minima, indices, Python objects, and allocator overhead |
| STC | cacher, pruner, and HF-ViT integration imports | Four core modules import with Torch 2.6.0 and Transformers 4.51.0 | Dependency smoke only; no ViT skip/prune numerical or CUDA timing claim |
| OASIS | `ShortMemory.push` plus event modules | Limits 2/4 retain 1/3 frames; one 4-frame event emitted | Confirms bounded windows and an exclusive effective capacity caused by `>= limit` followed by `pop(0)` |

CausalMem, StreamingTOM, and STC use the existing Qwen3 Python 3.10
environment. OASIS uses an isolated Python 3.12.13 environment with Torch
2.5.1+cu124, Transformers 4.57.6, and SentenceTransformers 5.6.0. OASIS's
README specifies CUDA 12.1; the cu124 environment is therefore used only for
CPU/module compatibility evidence, not official GPU latency.

## Paper-To-Code Findings

### CausalMem

[CausalMem](https://arxiv.org/abs/2606.25658) is the closest algorithmic
baseline for a query-free online semantic basis. The paper uses `q=64`, at
most eight new basis candidates per frame, EMA decay `0.9`, and 12k/6k token
budgets. It reports gains of 3.1 points on OVO and 3.2 points on
StreamingBench, more than 20x speedup, and about 82 MB for one hour.

The pinned repository is not README-runnable without repair: several
`modeling_*.py` files are absent, checkpoints are empty, and some scripts use
invalid absolute paths. More importantly, its default `causal_mem` branch
retains a per-evicted-frame mean in `mem2`, whereas the paper describes direct
removal. The resulting `fg_budget=max(100,b-len(mem2))` can exceed the claimed
fixed budget on a long stream. The repository's `only_mem1` path is closer to
the paper's strict single-bank semantics.

Our proxy retains high residual-energy frame vectors and an online basis. It
omits background merging because one vector per frame makes that operation
degenerate. It is therefore a mechanism proxy, not CausalMem quality or
latency reproduction.

### StreamingTOM

[StreamingTOM](https://arxiv.org/abs/2510.18269) combines CTR before the LLM
with a 4-bit online quantized KV memory. The paper reports prefill reduction
from 337.8 ms to 92.8 ms for its stated 64-frame setting, plus 7.3/6.9/28.7 ms
for KV write/retrieval/reconstruction and 0.20 s TTFT.

The official public command covers VideoMME-Short. The implementation
hard-codes 196 visual tokens, 28 layers, and 14 system tokens; environment
variables called defaults by the README are required by the code. A 12,544
token request is rounded by groups to 12,550 visual tokens before prompt
tokens. Most importantly, the quantized archive still grows with stream
length; only the active retrieved KV is bounded.

Our proxy models two-frame compression groups, a 4-bit growing archive, and a
bounded query-conditioned read. It cannot reproduce pre-RoPE layer KV,
attention saliency, 15.7x KV compression, or TTFT.

### STC

[STC](https://arxiv.org/abs/2512.00891) is the relevant real systems baseline
for repeated-ViT caching and visual-token pruning. Its public speed tools use
five warmups, twenty CUDA-event repetitions, and report ViT encode time, LLM
prefill time, and peak memory.

The paper's cacher selects tokens per layer. The current fast path shares the
first-layer selection across layers and adds CUDA Graphs, which the code
itself treats as an accuracy-sensitive approximation. The paper uses
dual-anchor pruning, while the public default is `gaussian`; `dual_anchor`
must be selected explicitly. Paper timing values shown in seconds cannot be
silently equated with the newer tool's millisecond output.

Our terminal-feature proxy can test state and task effects, but it cannot
recover ViT FLOPs. Official STC latency must be measured inside the real ViT
and LLM stack.

### SelectStream

[SelectStream](https://arxiv.org/abs/2606.16353) is the strongest quality
target for exact current evidence, surprise-triggered writes, a bounded
priority graph, relation-aware retrieval, and learned latent evidence.
Default paper budgets are `N/B/M=256/64/8`. The paper contains an unresolved
training-rate conflict: the appendix text gives `2e-4`, while Table 6 gives
`2e-5`.

No public code or model asset was discoverable. Our proxy implements adaptive
segments, bounded priority consolidation, exact recent evidence, and a
query-routed graph, but not the learned segment encoder, GAR, calibration, or
latent-token training. It must remain labeled `untrained structural proxy`.

### OASIS

[OASIS](https://arxiv.org/abs/2604.17052) is the slow event-archive baseline.
Its public implementation contains recent and medium windows, event summaries,
a bounded root set, and two-stage retrieval. The paper reports 6.52 s
end-to-end query latency on an A800, with expensive node generation and root
merge maintenance assumed to overlap asynchronously.

The public code performs summary and merge work synchronously and does not
publish the timing runner used for the paper table. A root budget of four only
bounds root summaries: descendants, parent nodes, and keyframes remain in the
event forest, so complete retained evidence grows with the stream. The code
also fails to write parent pointers after one merge path and uses the original
question rather than generated intent for QA retrieval.

The executable `ShortMemory` smoke also shows that configured limits are
exclusive in steady state: the implementation appends and then pops when
`len >= limit`, so limits 2 and 4 retain only 1 and 3 frames respectively.
Paper-to-code budget comparisons must use effective retained capacity rather
than the raw configuration value.

Our proxy therefore counts bounded active roots separately from a growing
event archive. Visual centroids replace MLLM summaries, embeddings, and tool
calls.

### StateKV

[StateKV](https://arxiv.org/abs/2605.31598) maintains a fixed per-layer
`cstate` for subsequent video prefill and a complete `dstate` for final
decoding. Fixed `B` makes video prefill linear in frame count, but the detailed
cache and final decode context remain linear in the stream. It is not an
end-to-end fixed-memory method.

The author repository currently contains only `README.md` with "Code release
coming soon." Per-KV-head update details and the reported Triton implementation
are unavailable. Our proxy uses attention-like current-frame selection for a
bounded cstate and correctly counts the full growing detailed state separately.

## Unified Proxy Protocol

The completed run replays 200 frozen MVBench CLIP records, each with 32 frame
vectors. All methods see the same chronological stream. The primary evidence
budget is eight vectors, the pool budget is sixteen, and stored floating-point
vectors use FP16 logical bytes. Question-conditioned methods see the question
only during readout. Active state, archive, detailed decode state, metadata,
and shared parameters are reported separately.

The short-stream proxy uses scale-normalized defaults rather than paper
hyperparameters: CausalMem basis rank 8; two-frame StreamingTOM groups; STC
reuse threshold 0.97; SelectStream segment length 2--4 and graph capacity 16;
two-frame OASIS events with 13 active roots; and StateKV cstate capacity 16.
These values make mechanisms observable on 32 vectors and must not be read as
paper-faithful configurations.

This protocol is intentionally weaker than native LLaVA or official model
evaluation. It cannot measure encoder skipping, prefill reduction, fused KV
kernels, generated summaries, or end-to-end answer latency.

## First Results

| Method | Accuracy | Gain vs recent | Total state | Total bounded | Evidence |
|---|---:|---:|---:|:---:|---:|
| Exact recent | 50.0% | - | 12.05 KiB | yes | 8 |
| CausalMem proxy | 49.0% | -1.0 pp | 24.09 KiB | yes | 8 |
| StreamingTOM proxy | 47.5% | -2.5 pp | 18.25 KiB | no | 8 |
| STC proxy | 52.5% | +2.5 pp | 13.59 KiB | yes | 8 |
| SelectStream proxy | 49.0% | -1.0 pp | 17.22 KiB | yes | 8 |
| OASIS proxy | 51.5% | +1.5 pp | 47.50 KiB | no | 8 |
| StateKV proxy | 51.5% | +1.5 pp | 72.41 KiB | no | 32 |
| Ours: learned selector, dev-fitted | 53.0% | +3.0 pp | 24.19 KiB | yes | 8 |

Against exact recent, the dev-fitted selector has a paired bootstrap interval
of `[+0.5,+6.0]` points, but exact McNemar `p=0.0703` from only seven better and
one worse sample. The selector was fitted using development evidence from this
reused sample pool, so this is post-hoc evidence and not an independent
generalization result. All external proxy intervals cross zero.

StateKV's apparent 51.5% is not an equal-evidence result because its decoder
uses all 32 vectors. OASIS and StreamingTOM also retain growing archives. They
must not be described as fixed-total-memory wins.

### Task Localization

The gain heatmap shows that scene transition drives almost every positive
result: gains range from +5 to +15 points on that task. Every method ties at
40% on moving direction. The dev-fitted selector gains +5 points on action
sequence and +10 on scene transition, with zero gain on the other three tasks.

This pattern supports an event-sensitive residual path, but it does not yet
validate long-horizon motion state. A one-vector-per-frame CLIP proxy is too
weak for that claim.

## Positioning Consequence

The evidence supports pursuing online video understanding with:

- exact recent/current visual evidence;
- a bounded semantic or episodic long-term state;
- sparse event residuals for transitions and rare changes; and
- optional low-rank or block-circulant modules inside the writer, router, or
  projection path.

It does not support replacing global attention with BCCB, claiming a new
low-rank primitive, or extrapolating these understanding proxies to online
video generation. For generation, NFE reduction, causal chunking, cache reuse,
low precision, and SLO-aware scheduling remain higher-level bottlenecks; a
block-circulant plus low-rank unit would be one component of a hybrid system.

## Required Next Gates

1. Freeze the proxy implementation and run at least 400 untouched reserve
   samples; report micro/macro accuracy, Wilson intervals, paired bootstrap,
   McNemar, exact agreement, and per-task deltas.
2. Add native `[frame,64,4096]` LLaVA projected-token versions of strict
   CausalMem, CTR, and STC-Pruner under the same 512-token read budget.
3. Run STC and StreamingTOM official speed tools when an A800 is available,
   fixing backbone, frames, precision, prompt, generation length, warmups, and
   repetitions. Report P50/P95/P99 for encoder, prefill, retrieval,
   reconstruction, decode, and end-to-end latency.
4. Extend the now-passing OASIS Python 3.12 module environment to the official
   Qwen3-VL and embedding checkpoints; report foreground latency and
   background GPU-seconds. Do not hide synchronous maintenance behind an
   assumed asynchronous schedule.
5. Keep StateKV and SelectStream labeled paper proxies until executable code
   is public. Do not fill missing architectural details and call the result
   official.

The selected CSV, JSON, and figures are in
`paper/results/probe_mvp/streaming_baseline_proxy_20260719/`.
