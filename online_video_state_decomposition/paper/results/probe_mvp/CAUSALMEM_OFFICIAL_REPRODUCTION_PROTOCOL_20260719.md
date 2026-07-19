# CausalMem Official Reproduction Protocol

## Evidence target

This protocol upgrades the existing CausalMem mechanism proxy to an official
model-level quality reproduction. It pins the public CausalMem checkout at
`640104b3786125c4918924f9b666ff7fe04d81de` and uses the original
`lmms-lab/llava-onevision-qwen2-7b-ov` checkpoint with architecture
`LlavaQwenForCausalLM`. The Transformers-native
`LlavaOnevisionForConditionalGeneration` checkpoint is incompatible and is
rejected by preflight.

The evaluation subset is the official StreamingBench Real-Time Visual
Understanding archive `Real-Time Visual Understanding_1-50.zip`:

- archive size: `9,080,551,796` bytes;
- archive SHA-256: `39f8e42130424bddfa8c298be882b21fa3e818318e9782e28ef705851c0c82c5`;
- 50 videos, `sample_1` through `sample_50`;
- 250 questions, five per video.

The official metadata contains one `MM:SS` timestamp that the pinned evaluator
would silently skip because it accepts only `HH:MM:SS`. The preparation harness
normalizes only this record and stores the change in the dataset manifest:

```text
Real-Time Visual Understanding_sample_40_3: 08:01 -> 00:08:01
```

No question text, option, answer, or frame requirement is changed.

## Validated preflight

The strengthened full offline preflight passed on 2026-07-19. The current
harness adds content SHA-256 for every indexed model and vision shard, exact
shard-index coverage, CRC32 checks for every extracted video, and the hard
50 x 5 question contract. The machine-readable record is
`causalmem_official_preflight_20260719.json`, with artifact SHA-256
`d3f75f2f88fc9640bf41ce1ac265bf4c34bd3f8d3ea4141820157383fc551e8b`
and run fingerprint
`8c942bbf03144180b5d8f21a4c660f2f0fa74497fc72fcb7609cc02330aa51be`.
Its stderr is empty. These hashes supersede the earlier size-only preflight.

This is setup evidence, not a quality or latency result. The one-question,
long-prefix, and CausalMem model runs remain pending the idle-GPU gate below.

## Official settings

The pinned checkout exposes these two labels, but only one is runnable without
modifying upstream source:

| Run | Upstream behavior |
|---|---|
| `baseline` | **Unavailable at the pinned commit.** `llava_qwen.py` imports a missing `llava/model/llava_arch_baseline.py`. No result is claimed. |
| `causal_mem` | Sample at 0.5 FPS and use the official FOSS cache. |

CausalMem uses budget 12,000, decay 0.9, maximum basis rank 64, maximum eight
new basis vectors, time weight 0.8, update ratio 0.1, and time power 1.0. It
uses one process, one GPU, one chunk, `qwen_1_5`, greedy decoding, and the
pinned upstream evaluator. A stock-model control must come from a separately
pinned, source-complete official implementation and must not be fabricated in
this checkout.

## Measurement boundary

The upstream evaluator clears the FOSS cache for every question and replays the
video prefix from time zero. Therefore its total wall time measures 250 repeated
prefix evaluations, not a persistent 250-query live stream. The harness reports:

- exact-match multiple-choice accuracy and complete 250-ID coverage;
- process wall time and completed questions per second;
- 200 ms sampled evaluator-process and total GPU memory;
- source, model, vision-tower, dataset, runtime, and runner fingerprints;
- whether a run resumed from existing predictions;
- per-attempt GPU samples, preserving earlier attempts during resume.

Resume is allowed only when every existing JSONL row is parseable, ends at a
newline boundary, has `id`, `acc`, `pred`, and `answer_id`, and contains no
duplicate or unexpected ID. Throughput is computed from newly completed
questions in that attempt, not from the cumulative row count.

The upstream evaluator has no per-sample timing. Its wall time must not be used
as P50/P95/P99 latency or directly compared with the StreamingTOM/STC core
kernel tail-latency benchmark. Resumed wall time is also marked incomparable to
an uninterrupted run.

## Execution gates

Run in this order after the dedicated environment and an idle A800 are ready:

1. CPU-only unit tests and `--dry-run` data/model preflight; add
   `--check-runtime` for the isolated CUDA import check.
2. One-question `sample_1_1` model-load and output smoke.
3. Long-prefix `sample_36_5` memory and decode smoke.
4. Full 250-question `causal_mem` run.
5. A source-complete stock-model control only after a separate official pin is
   documented.

A full run passes only when the subprocess exits successfully and all 250
unique question IDs have valid JSONL records with no duplicate or malformed
rows. Decode failures and missing videos cannot be treated as completed
questions.

Exact-ID subsets require `--allow-smoke-subset` and are labeled
`official_model_level_smoke`; they cannot satisfy the formal quality contract.

## Separate latency track

`benchmark_official_streaming_kernels.py` measures pinned official
StreamingTOM CTR/OQM and STC-Pruner calls on CUDA. CTR carries the returned
state across 32-frame batches and must match one-shot output exactly. OQM
appends 32-frame batches and retrieves through the official
`StreamingTOMContext` encode/retrieve processor path. The harness records
synchronized wall time, CUDA Event time, P50/P95/P99 with the `higher` quantile
rule, peak allocator memory, raw samples, and numerical quality gates.

These remain core-module microbenchmarks. In particular, `stc_pruner` does not
execute STC-Cacher, ReKV, ViT encoding, or LLM prefill. The separate official
`STC/speed_benchmark/benchmark_rekv.py` entry point is required for the real
ReKV ViT/prefill baseline. End-to-end StreamingTOM encoder, prefill, TTFT, and
decode measurements remain a later gate.
