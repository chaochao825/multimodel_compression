# VSI Progressive Exact-Evidence Capacity Gate

Date frozen: 2026-08-30

Role: calibration-only capacity diagnostic

## Decision question

After a fixed rank-456 quotient and regular 7x7 spatial pooling reduce every
frame to 25% of its visual tokens, can one exact full-resolution frame recover
enough answer-relevant evidence to make progressive retrieval a plausible
function class?

This is an oracle capacity diagnostic. It does not test a deployable selector,
selection or formal generalization, task-wide quality, latency, or speedup.

## Frozen identities

- Split protocol: `vsi_onevision_reader_quotient_stage_a_20260830_v1`.
- Evaluation data: the first 24 eligible calibration questions in the frozen
  calibration order.
- Selection and formal roles: unread.
- Reader: frozen LLaVA-OneVision Qwen2-7B.
- Visual input: uniform 8 frames, 14x14 visual tokens per frame.
- Quotient: VSI calibration PCA, rank 456, fitted from all calibration feature
  states without labels or reader outcomes.
- Pooling: deterministic 2x2 mean pooling to 7x7 tokens per frame.
- Exact evidence budget: at most one full 14x14 frame; all other frames remain
  quotient-reconstructed and pooled.

The oracle may inspect each one-frame candidate's exact reader KL only to
measure capacity. It is explicitly non-deployable and cannot be promoted as a
selector.

## Frozen comparisons

1. exact features with 7x7 pooling for every frame;
2. rank-456 quotient features with 7x7 pooling for every frame;
3. one exact frame selected by quotient residual energy;
4. one exact frame selected by minimum exact reader KL oracle.

The full-token variable-input path must reproduce the existing fixed-token
reader logits to maximum absolute error `<= 1e-5` before a sample is valid.

## Decision rule

The function-class result is `GO` only when every condition holds:

- oracle mean candidate KL is at most half the quotient-pooled mean KL;
- oracle top-choice agreement with the full reader is at least 98%;
- oracle harmful flips are zero;
- retained visual-token fraction is at most 35%.

Residual-energy recovery of the oracle improvement is diagnostic only. A `GO`
permits a separately frozen query-conditioned selector probe. A `NO_GO` stops
one-frame progressive retrieval at this pooling granularity; it does not reject
multi-frame, segment-level, or learned evidence allocation.

## Execution repair

The first launch referenced a nonexistent project-relative `DATASET` path and
stopped before loading any record or model output. The rerun changes only the
command-line dataset root to the already registered VSI snapshot used by the
preceding reader probes. Candidate definitions, sample order, and thresholds
remain unchanged; the failed log is preserved.
