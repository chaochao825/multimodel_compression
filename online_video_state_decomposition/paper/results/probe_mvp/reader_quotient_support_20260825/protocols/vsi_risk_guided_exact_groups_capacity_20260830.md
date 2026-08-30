# VSI Reader-Risk Guided Exact-Group Capacity Gate

Date frozen: 2026-08-30

Role: calibration-only task-metric capacity diagnostic

## Decision question

Frame-level exact retrieval failed even with an exhaustive two-frame oracle.
At the same 43.75% visual-token budget, does selecting distributed regular
4-token sequence blocks by the frozen reader's per-question margin Jacobian
recover the full-reader decision substantially better than Euclidean residual
energy or a cheap question-similarity score?

This tests whether task-induced risk, rather than raw feature distortion, is the
missing allocation metric. It does not test a deployable risk predictor.

## Frozen identities and representation

- First 24 eligible calibration questions; selection and formal roles unread.
- Frozen LLaVA-OneVision Qwen2-7B, uniform 8 frames, rank-456 VSI quotient.
- Each frame's 196 row-major visual tokens are partitioned into 49 contiguous
  groups of four. A compressed group is represented by one quotient-mean token.
- Refine exactly 98 of 392 groups. Each refinement replaces one mean token with
  its four exact tokens, producing 686 of 1568 tokens (43.75%).
- Full refinement preserves original token order and exactly recovers the full
  feature sequence; every sample must first pass the `1e-5` full-path logit
  equivalence check.

## Frozen selectors

1. largest quotient reconstruction energy per group;
2. largest cosine similarity to the frozen post-video question embedding;
3. largest target-visible normalized adverse margin risk

\[
r_g=\max_c
\frac{[-\langle \nabla_x m_c,\,\delta x_g\rangle]_+}
     {\max(m_c,0.05)}.
\]

The third selector uses full-reader gradients and is a non-deployable capacity
oracle. It is not a candidate for timing or formal evaluation.

## Decision rule

The task-metric function class is `GO` only if the target-gradient selector:

- halves mean candidate KL relative to quotient group means;
- reaches at least 98% full-reader top-choice agreement;
- causes zero harmful flips;
- remains within 45% visual-token retention.

A `GO` permits a new gate for a cheap current-query risk proxy or shallow
reader-derived controller. A `NO_GO` rejects this first-order regular-group
allocation at the registered budget and redirects work toward learned
tokenization or exact sparse tokens rather than more fixed quotient bases.
