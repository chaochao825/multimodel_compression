# Video-MME OneVision PCA-r456 Cross-Domain Replication

Date: 2026-08-29
Status: frozen before model execution; completed as `BOUNDARY`

## Decision question

Does the calibration-only `PCA-r456+s0` codec selected on MVBench preserve the
same frozen LLaVA-OneVision reader on an independent video benchmark, without
refitting the basis, changing rank, or using query-visible support?

This gate tests cross-domain representation transfer. It does not test another
reader, token-count reduction, prefill speedup, or the closed Fisher-support
line.

## Frozen identity and split

- Reader: the same local LLaVA-OneVision-Qwen2-7B checkpoint and first-token
  multiple-choice scoring used in the 2026-08-29 MVBench confirmation.
- Codec: the unchanged MVBench-calibration rank-456 basis from
  `onevision_rank_support_allocation_20_20260825_v1`; no fit or parameter may
  consume Video-MME outputs.
- Benchmark: official Video-MME parquet and the 900 locally available videos.
- Exclude the 30 Video-MME videos used by the July Qwen/CLIP representation
  probes before sampling.
- Select 200 distinct videos from each of `short`, `medium`, and `long` with
  seed `20260829`. Select exactly one of the three questions for each chosen
  video with the same deterministic RNG, for 600 independent video-question
  pairs. The exact split is materialized before any model execution.
- Options must contain exactly A-D and the answer must be one of A-D. Strip the
  existing labels before applying the unchanged OneVision choice prompt.
- Subtitles remain disabled for both Full and PCA paths.

## Frame and state policy

Video-MME questions may depend on the full clip, so both matched paths use 16
uniform frames over the complete video as the persistent feature pool and 8
uniform positions from that pool as the reader input. This differs from the
online-recent MVBench policy but is frozen identically for Full and PCA.

The persistent state remains `[16,196,3584]` BF16 for Full and `[16,196,456]`
FP16 for PCA-r456. Only the codec changes the state; question, frames, prompt,
candidate tokens, reader, and answer metric are paired exactly.

## Engineering smoke

Before the formal run, at most three questions from the 30 excluded historical
videos may be used to validate parquet parsing, frame decoding, option labels,
manual feature injection, and checkpoint writing. Smoke outputs cannot enter
the formal split or tune any threshold.

## Primary endpoints and decision rule

The candidate is `PASS` only if every guard holds:

- all 600 expected questions and 600 unique videos complete exactly once;
- Full candidate accuracy is at least 35%;
- PCA accuracy is no more than 2 percentage points below Full;
- the one-sided 95% Clopper-Pearson upper bound on
  Full-correct/PCA-wrong events is at most 2%;
- candidate prediction agreement is at least 98%;
- no duration group loses more than 5 percentage points;
- tensor payload is at most 2,867,328 bytes with compression at least 7.8x;
- manual feature injection changes first-token logits by at most `1e-3`;
- all metrics are finite and all split/model/codec identities match.

Candidate/vocabulary KL, feature relative L2, per-domain and per-task-type
metrics, paired bootstrap accuracy delta, and component timings are secondary.
They cannot rescue a failed primary guard.

The result is `ADVERSE` if aggregate accuracy loses more than 5 points,
prediction disagreement exceeds 5%, empirical harmful rate exceeds 5%, or any
duration loses more than 10 points. A valid result that is neither `PASS` nor
`ADVERSE` is `BOUNDARY`. Missing/duplicate evidence, reused videos, non-finite
metrics, identity mismatch, or injection mismatch is `INVALID`.

## Resource and consequence

- At most three isolated A800 GPUs and four aggregate GPU-hours; no training,
  generation, subtitles, or new model/data download.
- `PASS` supports cross-domain state-transfer capacity and authorizes a
  separately frozen direct-read/token-count system probe.
- `BOUNDARY` parks rank tuning and permits only a different-reader replication.
- `ADVERSE` rejects the claim that this MVBench PCA basis is domain-general.
