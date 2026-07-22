# Streaming Baseline Reproduction Audit

Initial audit: 2026-07-19. Formal-result update: 2026-07-22.

## Verdict

The six requested baselines are not equally reproducible from public assets.
This audit therefore separates six evidence tiers:

1. `paper mechanism`: the method described by the paper;
2. `official runnable`: an executable public entry point;
3. `strict preflight`: pinned source, runtime, assets, protocol, and output
   integrity validated without claiming a result;
4. `source/module smoke`: pinned syntax/import or targeted synthetic execution;
5. `official smoke/partial monitor`: official model execution that is either
   deliberately too small or not yet complete; and
6. `feature proxy`: an independent mechanism approximation on the same frozen
   CLIP cache used by this project.

The proxy table remains tier 6 and is not an official baseline ranking. Since
the initial audit, strict formal 50-video/250-question quality runs have
completed for CausalMem and OASIS, and the pinned StreamingTOM CTR/OQM CUDA
core triplet has completed. STC's matched official ReKV/ReKV+STC model-stage
latency pair also completed after an isolated runtime repair passed a real-model
smoke. SelectStream and StateKV remain paper-only because executable official
code is unavailable.
The first 200-sample proxy result is still a reused development analysis, not
an independent confirmation.

| Method | Intended role | Public implementation status | Current status |
|---|---|---|---|
| CausalMem | Closest query-free semantic-memory algorithm | `causal_mem` evaluator is runnable; stock baseline imports a missing upstream file | Formal 50x5 result passed strict validation: 206/250, 82.4% |
| StreamingTOM | Real CTR/OQM systems-latency baseline | Public VideoMME-Short path | Pinned 64/64/256-frame CTR/OQM core triplet complete; core latency only, not TTFT |
| STC | Real ViT-cache and prefill-latency baseline | Public model-specific runners and speed tools | Matched 64-frame ReKV/ReKV+STC pair passed strict validation; stage latency only |
| SelectStream | Task-quality target | No discoverable official code as of the audit date | Reported quality target; untrained structural proxy only |
| OASIS | Slow event-archive quality baseline | Public evaluation path, no public paper-table timing runner | Formal 50x5 result passed strict validation: 209/250, 83.6% |
| StateKV | Recurrent model-state baseline | Official placeholder repository only | Reference-only; paper-structure proxy cannot be called a reproduction |

## Formal Result Update

The completed OASIS and CausalMem runs cover the same 250 question IDs, but
they do not isolate the memory mechanism. OASIS uses Qwen3-VL-8B-Instruct,
whereas CausalMem uses LLaVA-OneVision-Qwen2-7B. OASIS scored 83.6% and
CausalMem scored 82.4%, a difference of only three questions. The paired
outcomes were 187 both correct, 22 OASIS-only correct, 19 CausalMem-only
correct, and 22 both wrong. Exact McNemar `p=0.755`, so this run does not
distinguish the systems statistically and cannot establish memory-method
superiority. The paired task table and source hashes are in
`official_streaming_formal_20260722/`.

StreamingTOM's pinned official-core measurements are complete. CUDA-event
P50/P95/P99 are 396.50/519.39/553.95 ms for CTR over 64 frames,
37.47/43.13/51.38 ms for OQM write over 64 frames, and
70.19/85.56/123.23 ms for OQM selection over 256 frames. The input scopes
differ and these are repeated core invocations, not additive end-to-end
latency, TTFT, decode time, or task quality.

STC's matched 64-frame, 5-warmup, 20-repeat official pair also completed. For
ReKV versus ReKV+STC, P50 ViT encode fell from 1681.37 to 527.95 ms (68.6%),
P50 visual-token prefill fell from 7587.77 to 6551.79 ms (13.7%), and the
per-iteration sum of those two stages fell from 9761.02 to 7062.34 ms (27.6%).
Mean summed stage time fell 25.3%, while the official peak-memory field fell
from 18.42 to 16.46 GB (10.6%). These are internal upstream-mode comparisons,
not task-quality, TTFT, decode, or end-to-end results. With only 20 measured
samples and the audited `higher` quantile convention, P95 and P99 both select
the maximum sample; tail-latency interpretation is therefore weak. The full
stage table and variability analysis are in
`official_streaming_formal_20260722/STC_STAGE_LATENCY_ANALYSIS.md`.

There is still no official same-backbone, equal-budget comparison against our
three-path proposal. The current project evidence supports individual state,
transport, and accounting probes only; its end-to-end quality and latency
cells remain open.

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

## Strict Official Preflights

CausalMem's official evaluator preflight is stored in
`causalmem_official_preflight_20260719.json`; its detailed contract is
`CAUSALMEM_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md`. It validates the exact
50-video/250-question StreamingBench subset, per-video CRC32 values, all model
shards and hashes, the SigLIP cache, pinned source, runtime versions, resume
integrity, and output fingerprints. The pinned source's stock baseline cannot
run because `llava_arch_baseline.py` is absent; it is not replaced with a local
surrogate. The `causal_mem` formal run completed all 250 questions at 82.4%
accuracy. Its evaluator-process wall time was 9,918.24 s and sampled peak GPU
memory was 42,894 MiB; neither value is a per-request tail-latency metric.

STC's official ReKV preflights are stored in
`stc_rekv_official_rekv_preflight_20260719.json` and
`stc_rekv_official_stc_preflight_20260719.json`; the matched protocol is
`STC_REKV_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md`. Both validate the pinned
source, full 16.06 GB four-shard model, CUDA runtime, official module imports,
exact mode environment, and audited wrapper hash. Passing preflight proves
launch readiness, not latency. The first formal attempt exposed two runtime
compatibility failures: Transformers 4.51 removed per-layer Qwen2 RoPE state,
and a shared stale Triton cache loaded an incompatible `cuda_utils` extension.
No STC source was modified. An isolated Python environment with Transformers
4.46.0 plus a project-local Triton 3.2.0 cache passed a real two-frame model
smoke. The fresh 64-frame, 5-warmup, 20-repeat ReKV/ReKV+STC pair then completed
under the original idle-GPU and lock gates. Both strict result artifacts,
run-record fingerprints, preflights, and GPU traces are present.

StreamingTOM's official-core dry runs pin commit
`6c66b05065692bc3fa4c6ec7fa9cad84d3b0cd75`, Torch 2.5.1,
Transformers 4.53.3, FlashAttention 2.8.0.post2, FP16, 28 layers, 20 warmups,
and 200 measured iterations. CTR and OQM write use 64 frames; OQM selection
uses 256 frames so the upstream top-k branch is exercised. The queue holds one
project GPU lock across all three components and rechecks the 4 GiB and 20%
utilization gate before each launch. These checks establish protocol readiness
only. All three summaries have now passed source, protocol, sample-count, and
finite-quantile validation. Their values are reported only as component-level
core latency because CTR and the two OQM paths do not share one additive input
scope.

OASIS now has an audited data adapter, no-copy dataset materializer, and strict
official evaluator wrapper. The matched contract is
`OASIS_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md`. Static validation passes for
the pinned source, 17.53 GB Qwen3-VL checkpoint, 1.19 GB embedding checkpoint,
and exact 50-video/250-question dataset. The machine-readable evidence is
`oasis_official_static_preflight_20260719.json` (SHA256
`9e7d90cbe8d23c507e74b82a99824939c7afeb144f8cfbb2f6b6ed6fe187ef0a`).
Its runtime field is intentionally null, so this is launch preparation, not a
quality or CUDA result. The separate
`oasis_flash_attn_build_audit_20260719.json` records a successful source build,
CPU-side import, maximum GLIBC requirement 2.14, and only `sm_80` embedded
cubins (SHA256
`061bc3b46b4b049c3071a6349a283500ca62b114ceb2859240dc9aeda645bb80`).
The subsequent BF16 CUDA-kernel check and one-video/five-question official
model smoke both passed. The smoke answered all five questions, took 309.645 s
at offline `pace=0`, and reached a sampled evaluator-process peak of 26,674
MiB. It validates execution, not formal quality or request latency. The formal
50-video/250-question job used a distinct run fingerprint and completed at
83.6% accuracy (209/250). Its whole-run wall time was 54,416.64 s and sampled
evaluator-process peak was 28,478 MiB. This validates official quality only;
the wall time includes offline `pace=0` streaming and synchronous archive work
and is not request latency or the paper's reported 6.52 s query latency.

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

CausalMem uses an isolated Python 3.10 environment. STC now uses a thin
project-local Python 3.10 environment over Torch 2.6.0 with Transformers
4.46.0 and an isolated Triton cache. StreamingTOM uses an isolated Python
3.10/Torch 2.5.1 environment with a verified
source-built `sm_80` FlashAttention import. OASIS uses an isolated Python
3.12.13 environment with Torch 2.5.1+cu124, Transformers 4.57.6, and
SentenceTransformers 5.6.0. Its own FlashAttention 2.8.3 source build and
CPU-side import/ELF audit for `sm_80` pass. Its CUDA BF16 kernel preflight and
one-video model inference smoke also pass. That 1x5 smoke is not promoted to a
formal result, and no OASIS request-latency claim is made from its offline wall
time.

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

The new pinned core harness executes upstream CTR, incremental OQM writes, and
top-k OQM selection with CUDA synchronization and separate CUDA-event and
host-wall clocks. It intentionally excludes model loading and input creation.
Its completed P50/P95/P99 values are component microbenchmarks, not VideoMME
quality, end-to-end Video-LLM latency, or the paper's 0.20 s TTFT.

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
recover ViT FLOPs. The new audited wrapper executes the upstream ReKV speed
benchmark inside the real ViT and LLM stack with separate baseline/STC
processes, 64 frames, five warmups, and twenty repeats. Both preflights, the
compatibility smoke, and the matched pair pass on an idle A800. ReKV uses the
upstream 196-token/frame, full-update mode; ReKV+STC uses the upstream
64-token/frame, 0.25-update mode. The measured stage-time and memory reductions
therefore validate the official combined path, but do not isolate cacher versus
pruner effects and do not establish quality preservation.

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
publish the timing runner used for the paper table. The audited evaluator's
`pace=0` total wall time covers the complete offline stream and synchronous
maintenance; it is not request latency. A root budget of four only bounds root
summaries: descendants, parent nodes, and keyframes remain in the event forest,
so complete retained evidence grows with the stream. The code also fails to
write parent pointers after one merge path and uses the original question
rather than generated intent for QA retrieval.

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

## Evidence Matrix And Gate Findings

`experiments/probes/build_streaming_evidence_matrix.py` now rebuilds a
source-grounded 17-entry evidence ledger, 65-row metric table, and 8-method by
8-stage completion matrix. A `PASS` cell can mean source, preflight, accounting,
or protocol validation; it does not imply that a complete method passed all
end-to-end gates. Runtime states are supplied separately so queued and partial
jobs cannot be mistaken for repository evidence.

- The preregistered 200-sample query-selector confirmation lost 0.5 points
  versus its reference, with a paired interval of `[-4,+3]`; its promotion gate
  failed.
- Native matched-state query memory reached 51.0% versus 47.5% for exact recent
  at 8,413,328 bytes, a +3.5-point paired difference with interval `[0,+7]` and
  `p=0.0923`; it remains open rather than confirmed.
- The rank-32 dual-spectral trigger used 8,320 bytes and reduced false triggers
  from 0.893% to 0.357%, but rare-event recall did not improve; its joint gate
  failed despite a 401.1 us writer P95.
- The fixed rank-256 plus 4-bit codec achieved 7.84x steady-state compression
  and only a 0.5-point observed quality loss, but its one-sided loss bound was
  2.3498%, above the 2% gate. The routed codec passes numerically only on the
  same post-hoc sample and still requires disjoint evaluation.
- Low-rank state evidence remains plausible but incomplete: rank 32 captured
  mean energy 0.766 and passed 19/25 cells at the 0.70 threshold on one encoder
  domain. The sparse-event residual failed its joint gate, with mean top-10
  energy 0.4817 and recall 0.205.
- BCCB improved over identity by 0.09639 on average in the controlled probe,
  but improved by less than 0.001 over the matched BTTB transport and showed no
  cyclic-cost advantage. The current evidence does not justify making BCCB the
  dominant architecture.
- OASIS and CausalMem formal quality now pass, but their different backbones
  prevent a memory-method claim. StreamingTOM official-core latency passes.
  STC matched model-stage latency also passes, while our native end-to-end
  quality and latency cells remain open.

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

1. Add our three-path method to a same-backbone 64-frame harness and report
   quality, state bytes, ViT/prefill stage latency, TTFT, decode, and end-to-end
   latency separately. The completed STC pair supplies a stage-level systems
   reference, not a direct comparison against our method.
2. Run a same-backbone, same-frame-sampling, equal-token/equal-byte comparison
   of exact recent, CausalMem, OASIS-style archive, and our proposed memory.
   The completed cross-backbone 50x5 results are system diagnostics only.
3. Evaluate the routed codec on a disjoint frozen split and add native
   `[frame,64,4096]` projected-token versions of strict CausalMem, CTR, and
   STC-Pruner under one matched read/state-byte budget.
4. Repeat low-rank spectrum tests across layers, encoders, and domains. Redesign
   event routing before claiming a sparse residual, because the current joint
   energy/recall gate failed. Retain BCCB only where a cyclic implementation
   demonstrates incremental benefit over matched BTTB plus a real cost win.
5. Keep StateKV and SelectStream labeled paper proxies until executable code is
   public. Do not fill missing architectural details and call the result
   official.

The proxy bundle is in
`paper/results/probe_mvp/streaming_baseline_proxy_20260719/`. Formal quality,
paired statistics, and official core-latency figures are in
`paper/results/probe_mvp/official_streaming_formal_20260722/`; the current
completion ledger is in
`paper/results/probe_mvp/streaming_evidence_matrix_20260722/`.
