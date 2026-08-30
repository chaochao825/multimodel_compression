# VSI Target-Risk Budget Frontier

Date: 2026-08-30

Status: frozen exploratory capacity diagnostic

## Decision question

The previous prospective writer Gate fixed exact support at 98 of 392 four-token
groups. Its target-gradient selector still produced two mismatches and two harmful
cases. Before opening a training-native memory-reader project, determine whether the
same target-visible ordering has a useful near-lossless retention window at a larger
but still compressed support budget.

This diagnostic separates two failure modes:

1. the deployable writer cannot observe task risk; and
2. the registered 98-group budget and first-order independent-group teacher are
   themselves insufficient.

It cannot establish deployability because target gradients are unavailable at
inference time.

## Data boundary

- Reuse only calibration positions 73--96, already exposed by the completed
  risk-observable writer Gate.
- Positions 97--120 remain unread at question, label, gradient, and reader-output
  level.
- Selection and formal roles remain unread.
- Existing calibration feature moments may be reused exactly as in the prior Gate;
  no new basis or hyperparameter is fitted.

The result is exploratory capacity evidence, not a new held-out generalization claim.

## Frozen candidate

For each question, compute the existing normalized first-order adverse group risk

\[
r_g=\max_c
\frac{[-\langle\nabla_{X_g}m_c,\,\bar X_g-X_g\rangle]_+}
{\max(m_c,0.05)}.
\]

Sort all 392 groups once by decreasing \(r_g\). Evaluate exact-prefix support at

\[
k\in\{0,49,98,128,160,196,245,294,343,392\}.
\]

An unselected group contributes one quotient token and a selected group contributes
its four exact tokens. The exact token retention is therefore

\[
\rho(k)=\frac{392+3k}{1568}.
\]

All methods use the same frozen OneVision reader, prompt, frame policy, PCA quotient,
candidate-token evaluator, and group ordering. No fallback, learned scorer, threshold,
or per-question budget is allowed.

## Measurements

For every budget and question record:

- full-reader prediction agreement;
- harmful and helpful flips relative to ground truth;
- candidate-distribution KL, mean and P95;
- compressed top-1 margin;
- captured target-risk mass;
- exact token count and retention.

The \(k=392\) path must reproduce the dense candidate logits within `1e-5`; otherwise
the diagnostic is invalid.

## Frozen interpretation

- `STRONG_CAPACITY_WINDOW`: some `k <= 196` reaches `24/24` agreement, zero harmful
  cases, mean KL at most `0.01`, and P95 KL at most `0.02`.
- `WEAK_CAPACITY_WINDOW`: no strong point exists, but some `k <= 294` reaches `24/24`
  agreement and zero harmful cases.
- `NO_USEFUL_CAPACITY_WINDOW`: neither condition holds.
- `INVALID`: dense equivalence, split identity, finite-value, or artifact checks fail.

A strong window authorizes a separately frozen joint tokenizer-reader Gate on
positions 97--120. A weak window permits method analysis but not a deployment Gate.
No useful window parks near-lossless progressive exact memory at the current quotient,
grouping, and first-order ordering; it does not refute other video-memory interfaces.

## Claim boundary

This diagnostic can identify whether a compressed support interval exists for an
unavailable target-visible teacher. It cannot validate a risk observer, calibrated
certificate, task accuracy, latency, selection/formal transfer, or superiority to
video-memory baselines.
