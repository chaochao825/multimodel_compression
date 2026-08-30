# VSI Measure-Preserving Compaction Diagnostic

Date: 2026-08-30

Status: frozen exploratory mechanism diagnostic

## Question

An unselected four-token group is currently represented by one quotient token. This
changes both its RoPE position and its multiplicity in every softmax denominator. The
position diagnostic recovered all 24 reader decisions at 196 groups but did not meet
the distribution-fidelity boundary and remained non-monotone.

Test whether treating each quotient as a measure atom with mass four improves the
reader path:

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}+\log s\right),
\]

where `s=4` for an unrefined quotient and `s=1` for each exact token. This is the
proportional-attention correction used by Token Merging and is a required baseline,
not a novelty claim.

## Frozen scope

- Reuse only already exposed calibration positions 73--96.
- Positions 97--120, selection, and formal remain unread.
- Reuse target-gradient support ordering and group budgets
  `0,49,98,128,160,196,245,294,343,392`.
- Preserve original full-sequence position IDs with representative offset 1.
- Compare identical compact tokens with all masses one versus quotient mass four.
- No training, fallback, threshold, alternative mass, or support tuning.

The additive key bias is supplied through a prepared causal 4D attention mask at every
Qwen2 layer. Prefix, newline, exact, and text tokens have mass one.

## Validity

- The all-mass-one 4D path must match the existing positioned 2D-mask path within
  `1e-5` on candidate logits.
- Both `k=392` endpoints must match dense candidate logits within `1e-5`.
- Any identity, finite-value, or split failure invalidates the diagnostic.

## Interpretation

- `MASS_FIDELITY_RECOVERY`: a weighted point with `k <= 196` reaches 24/24 agreement,
  zero harmful, mean KL <= 0.01, P95 KL <= 0.02, and at least 20% lower mean KL than
  the equal-mass positioned control.
- `MASS_DECISION_ONLY`: no fidelity point passes, but a weighted point with `k <= 196`
  reaches 24/24 agreement and zero harmful.
- `NO_MASS_RECOVERY`: neither condition holds.

A positive result only motivates a new held-out Gate for measure-aware progressive
refinement. It does not validate a deployable kernel, a risk observer, or novelty over
proportional-attention token merging.
