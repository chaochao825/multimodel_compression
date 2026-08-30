# VSI Query-Group Margin-Fallback Transfer Gate

Date frozen: 2026-08-30

Role: calibration-fit to fresh calibration-evaluation transfer

## Decision question

Can the deployable part of the preceding capacity result be expressed as a
simple conditional estimator: retrieve regular exact groups using the current
question and compressed quotient state, then fall back to the full visual state
only when the compressed reader margin is insufficient?

## Frozen partition and method

- Fit partition: first 24 eligible calibration questions already used by the
  capacity diagnostics.
- Evaluation partition: next 24 eligible calibration questions, not used by the
  preceding probes.
- Selection and formal roles remain unread.
- Frozen reader, rank-456 quotient, 8 frames, contiguous groups of four, and 98
  exact groups (43.75% visual tokens) are unchanged.
- Group score is the maximum cosine similarity between a quotient group token
  and the mean frozen prompt embedding after the visual placeholder.
- Runtime confidence is the approximate reader's top-1 minus top-2 candidate
  logit margin.

The fallback threshold is deterministically fitted as the largest compressed
margin among fit-partition disagreements. It is applied once to the fresh
evaluation partition without adjustment. Fallback replaces the hybrid reader
result with the registered full-reader result.

## Decision rule

The transfer result is `GO` only when the evaluation partition satisfies all:

- delivered top-choice agreement at least 98%;
- zero remaining harmful flips;
- exact fallback rate at most 15%;
- effective visual-token retention at most 53%;
- delivered task accuracy no more than one percentage point below the full
  reader.

Token reduction is an ideal reader-prefill proxy. A `GO` would permit a formal
untouched evaluation and measured reader latency only after a separate frozen
protocol. A `NO_GO` closes this parameter-free query-score and scalar-margin
controller; it does not reject a shallow learned risk controller.
