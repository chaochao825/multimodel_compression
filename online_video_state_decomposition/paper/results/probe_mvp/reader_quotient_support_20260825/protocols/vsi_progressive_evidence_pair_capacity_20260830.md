# VSI Two-Frame Progressive Evidence Capacity Gate

Date frozen: 2026-08-30

Role: calibration-only capacity and selector diagnostic

## Decision question

The one-frame exact-evidence gate reduced mean KL by 70.5% but failed reader
agreement and harmful-flip guards. Does a fixed budget of two exact frames make
frame-level progressive evidence expressive enough before developing a learned
router?

## Frozen identities and budget

- Reuse the same first 24 eligible calibration questions, frozen reader,
  rank-456 VSI quotient, uniform 8 frames, and 7x7 pooling as the one-frame
  gate.
- Selection and formal roles remain unread.
- Retain 392 pooled tokens plus two exact-frame refinements of 147 tokens each:
  686 of 1568 visual tokens, or 43.75%.
- Enumerate all 28 two-frame pairs only for a non-deployable capacity oracle.

## Frozen selectors

1. top-two quotient residual energy;
2. top-two question-conditioned frame scores, computed by the maximum cosine
   similarity between each quotient-pooled frame token and the mean frozen text
   embedding after the video placeholder;
3. minimum exact candidate-KL pair oracle.

The query selector uses only the compressed visual state, prompt token IDs, and
frozen input embedding table. It does not read exact frame residuals, labels,
reader logits, or held-out outcomes.

## Decision rule

The frame-level function class is `GO` only if the pair oracle satisfies all:

- mean candidate KL at most half the quotient-pooled mean KL;
- full-reader top-choice agreement at least 98%;
- zero harmful flips;
- visual-token retention at most 45%.

Query-score and residual-energy recovery of oracle improvement are diagnostics,
not pass conditions. A `GO` permits a separately frozen learned or calibrated
query selector. A `NO_GO` stops frame-granular exact retrieval and redirects the
next gate to query-conditioned spatial tile/token evidence rather than adding
more exact frames post hoc.
