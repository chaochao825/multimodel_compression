# Video-MME OneVision Same-Rank Domain-Residual Gate

Date: 2026-08-29
Status: frozen before codec fitting or candidate execution

## Decision question

Did the MVBench `PCA-r456` transfer result remain `BOUNDARY` because rank 456
is intrinsically insufficient, or because a small number of target-domain
directions displaced source-domain directions inside the same rank budget?

This is a development and capacity gate. It does not establish a new method,
cross-reader transfer, token-count reduction, or system speedup. Subspace
alignment and online semantic-basis updates already exist; a positive result
only authorizes a later reader-risk robust-subspace hypothesis.

## Frozen data roles

- Source endpoint: the completed 600-video Video-MME cross-domain split.
- Calibration: 40 source-endpoint videos per duration, 120 total. Only visual
  features may fit means and bases; questions, answers, logits, and prior error
  labels are forbidden during fitting.
- Selection: a disjoint 60 source-endpoint videos per duration, 180 total.
  Candidate choice may use frozen reader metrics only here.
- Formal reserve: 85 videos per duration, 255 total, excluded from the prior
  600-video endpoint and the 30 historical Video-MME diagnostics. It is frozen
  now but cannot be executed unless the selection gate is `GO`.
- Seed: `26082901`. Every role contains one question per unique video.

The calibration and selection identities were observed under the old codec and
therefore cannot support a new confirmation claim. Their only role is method
development. The 255-video reserve remains untouched by OneVision execution.

## Same-rank candidates

All candidates store `[16,196,456]` FP16 coefficients and therefore have the
same `2,860,032`-byte per-video tensor payload:

1. `source_r456`: unchanged MVBench mean and basis.
2. `target_mean_source_r456`: target calibration mean with the source basis.
3. `residual_swap_r{16,32,64,96,128}`: retain the first `456-s` source
   directions, then fill the remaining `s` dimensions with PCA directions of
   the target calibration residual orthogonal to that retained source span.
4. `target_pca_r456`: target-calibration PCA, an equal-rank domain upper bound.

No candidate may consume labels, question text, reader logits, selection
features, or formal features while fitting. Rank, frame policy, reader, prompt,
candidate tokens, and tensor dtype remain fixed.

## Selection gate

The source baseline and every candidate are evaluated in one matched pass. A
candidate is eligible only if all conditions hold relative to `source_r456`:

- mean candidate KL is at most `0.70x`;
- P95 candidate KL is at most `0.80x`;
- feature relative L2 is at most `0.90x`;
- mismatch count falls by at least 20%, with at least two fewer mismatches;
- harmful flips do not increase;
- candidate-correct count does not decrease;
- no duration's mean candidate KL exceeds `1.10x` its source value;
- state bytes, injection identity, sample completeness, and finite checks pass.

If one or more candidates qualify, select the lowest mean-KL candidate, using
lower adaptation rank and then candidate ID only as deterministic ties. This is
`GO` and authorizes the already frozen 255-video formal reserve. A candidate
that passes KL/L2 and safety but not mismatch reduction is `CAPACITY_ONLY`; it
does not authorize formal execution. Otherwise the gate is `NO_GO` and static
same-rank domain adaptation stops.

## Formal consequence

Only a `GO` selection candidate may enter the reserve. Formal `PASS` requires
the previous non-inferiority guards at the same bytes: accuracy loss no more
than 2 points, agreement at least 98%, harmful one-sided 95% bound at most 2%,
and no duration loss greater than 5 points. It must additionally reduce mean KL
by at least 25% and mismatches by at least 20% against `source_r456` on the same
reserve. Any other valid result is `BOUNDARY` or `ADVERSE` under the existing
loss guards.
