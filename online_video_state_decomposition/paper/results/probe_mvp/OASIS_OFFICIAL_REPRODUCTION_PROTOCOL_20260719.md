# OASIS Official StreamingBench Reproduction Protocol

## Scope

This protocol is intended to run the unmodified OASIS unified evaluator on
StreamingBench Real-Time Visual Understanding. OASIS is the slow event-archive
quality baseline. The public evaluator reports no per-query timing and
executes maintenance synchronously with `pace=0`; its total wall clock covers
the full offline evaluation and is not request TTFT, request latency, or SLO
latency.

## Pinned Assets

- Source: OASIS commit `dbd342c79a1b9b03327d4ec5daa87488737db988`.
- MLLM: local Qwen3-VL-8B-Instruct, four indexed safetensor shards totaling
  17,534,339,512 bytes.
- Retrieval encoder: local Qwen3-Embedding-0.6B, one safetensor file totaling
  1,191,586,416 bytes.
- Runtime: isolated Python 3.12 environment with Torch 2.5.1, Transformers
  4.57.6, and SentenceTransformers 5.6.0. FlashAttention 2.8.3 was built from
  source on server 210. CPU-side import passes; the extension has only `sm_80`
  cubins and requires at most GLIBC 2.14. This is not yet a successful CUDA
  BF16 kernel preflight.

The audited wrapper hashes every model weight file, model index, tokenizer and
processor metadata, required upstream source file, runner, and shared GPU
monitor helper. Offline Hugging Face and Transformers modes are mandatory.

## Dataset Contract

The adapter cross-checks the OASIS unified JSON against the validated
StreamingBench CSV for every question field: ID, text, options, answer, task,
and numeric time. A formal run requires ordered samples 1 through 50, exactly
five questions per video, and 250 unique question IDs.

The mapping manifest binds:

- upstream archive SHA256
  `39f8e42130424bddfa8c298be882b21fa3e818318e9782e28ef705851c0c82c5`;
- upstream preparation manifest SHA256
  `fc4c18de257107abd9c519d7b92ea22220041f384913b7961f1e0416e7ccbd7a`;
- the sole timestamp repair, `sample_40_3: 08:01 -> 00:08:01`;
- the exact adapted JSON and CSV hashes; and
- each prepared video's byte count and CRC32.

The official directory layout is materialized with 50 symbolic links. No
video is copied. Existing links are reused only when they resolve to the exact
manifest source; conflicting targets are preserved and rejected.

## Official Configuration

The wrapper refuses paper-comparable runs if any value differs from:

| Parameter | Value |
|---|---:|
| FPS / pace | `2.0` / `0.0` |
| Short memory / now window | `32` / `16` frames |
| Buffer FPS | `1.0` |
| Frames per event node | `16` |
| Tokens per frame | `256` |
| Root count limit | `4` |
| Event / QA retrieval limits | `2` / `1` |
| ASR | `none` |

## Resume And Output Safety

The upstream evaluator resumes from `len(results)` and silently skips missing
videos. The wrapper therefore requires every partial result to be an exact
prefix of the metadata, validates all five breakpoints per completed video,
recomputes correctness and task metrics, and rejects malformed, duplicate,
unexpected, or failed questions. Existing output without a matching preflight
fingerprint is preserved and rejected.

The runner records wall time and sampled GPU memory/utilization. Wall time is
the full offline `pace=0` evaluation duration, including synchronous archive
maintenance, not end-to-end or per-request query latency.

## Commands

```bash
bash experiments/scripts/prepare_oasis_streamingbench.sh

bash experiments/scripts/run_oasis_streamingbench.sh \
  oasis_smoke_1video_v1 smoke 3

bash experiments/scripts/run_oasis_when_idle.sh \
  oasis_smoke_1video_v1 smoke 3
```

These are staged commands, not completed runs. The one-video smoke is waiting
in the safe GPU3 queue; after acquiring the lock, the runner must pass its CUDA
BF16 kernel preflight before it starts model inference. Queue the exact
50-video run only after the smoke has a complete, error-free official output
and measured resource use is compatible with the host.

## Current Gate

The static preflight passes for the exact pinned source, complete local models,
and 50-video/250-question data contract. FlashAttention 2.8.3 has completed its
server-210 source build and CPU-side import/ELF audit. The CUDA BF16 kernel
preflight and one-video model inference have not completed. The static
preflight's runtime field is intentionally null and its fingerprint is not a
launch, quality, or latency result.

The completed machine-readable static evidence is
`oasis_official_static_preflight_20260719.json`. It finished with exit code zero
at `2026-07-19T03:02:50Z`, has SHA256
`9e7d90cbe8d23c507e74b82a99824939c7afeb144f8cfbb2f6b6ed6fe187ef0a`,
and records run fingerprint
`4a4fa2f270f1dda42f0dd01d162f17e52515b551cb718a691fb187260e2b05d5`.
The artifact validates all 50 videos, 250 questions, and a minimum of 180
sampled frames per video without creating the configured output directory.

The separate `oasis_flash_attn_build_audit_20260719.json` has SHA256
`061bc3b46b4b049c3071a6349a283500ca62b114ceb2859240dc9aeda645bb80`.
It records wheel SHA256
`dd9466914b3555a5724ef1168b0760987d68186020830ff333dcd1915f3d242a`,
extension SHA256
`7c64e77a6e8541ccaec1937de1ce217c086a634fd245767771c80bfebdb7e730`,
maximum GLIBC requirement 2.14, and only `sm_80` embedded cubins. Its evidence
scope explicitly excludes CUDA kernel execution, model inference, quality, and
latency.
