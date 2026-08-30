# VSI Group-Compaction Geometry Diagnostic

Date: 2026-08-30

Status: frozen exploratory mechanism diagnostic

## Motivation

The target-risk budget frontier on already exposed positions 73--96 was strongly
non-monotone. Four questions changed from a matching prediction back to a mismatch
after more exact groups were restored. At 343 of 392 exact groups, the first-order
teacher reported essentially all risk mass captured, yet two questions still
mismatched.

The teacher and reader path differ structurally. The teacher linearizes a fixed-length
replacement in which every four-token group remains four tokens. The deployed hybrid
path collapses an unselected group to one token, reindexes all following visual and
text RoPE positions, and changes attention multiplicity and normalization.

## Data and frozen budgets

- Reuse only already exposed calibration positions 73--96.
- Positions 97--120, selection, and formal remain unread.
- Reuse the frozen target-gradient ordering and budgets
  `0,49,98,128,160,196,245,294,343,392`.
- No training, threshold selection, fallback, or new support ordering.

## Compared layouts

1. `compact_contiguous`: the existing variable-length path. Unselected groups become
   one quotient token and the model generates contiguous position IDs.
2. `compact_original_position`: identical tokens and sequence length, but each retained
   token receives its original full-sequence position. A quotient token uses offset 1
   of its four-token group; the text suffix keeps its original positions.
3. `fixed_repeated`: every group stays four tokens. Unselected groups repeat the
   quotient mean four times, so token multiplicity and all positions remain fixed.

The representative offset is frozen to 1 before execution. It is not selected from
the result.

## Measurements and validity

Record the same agreement, harmful flips, candidate KL, margin, and risk-mass metrics
for every layout and budget. All three `k=392` endpoints must reproduce dense candidate
logits within `1e-5`.

## Interpretation

- `POSITION_GEOMETRY_RECOVERY`: `compact_original_position` reaches 24/24 agreement,
  zero harmful cases, mean KL <= 0.01, and P95 KL <= 0.02 at `k <= 196`.
- `FIXED_MULTIPLICITY_ONLY`: no positioned compact point passes, but `fixed_repeated`
  passes the same numerical boundary at `k <= 196`.
- `NO_GEOMETRY_RECOVERY`: neither path passes.
- `INVALID`: any dense-equivalence or split guard fails.

Position recovery would justify a fresh compact-memory Gate with explicit original
positions. Fixed-only recovery would show that token multiplicity/attention
normalization, not information loss alone, prevents safe compaction. No recovery would
place the dominant error in first-order interaction or the quotient itself.

This is mechanism evidence on reused data, not held-out generalization or speed.
