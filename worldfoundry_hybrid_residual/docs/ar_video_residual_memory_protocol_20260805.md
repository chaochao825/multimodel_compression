# AR Video Residual Memory Oracle Protocol v1

Status: frozen before LongLive model capture on 2026-08-05. No real-model
capture existed when the arithmetic-gate correction below was made.

## Decision

Determine whether causal frame semantics make a spatially aligned temporal
summary plus sparse event residual produce an attention-output defect that is
both small and rank-16 compressible on held-out LongLive-1.3B captures.

## Candidate and incumbent

- Candidate: exact sink/recent frames + temporal summary representatives with
  log-multiplicity + K/V-residual event tiles + optional low-rank AV defect.
- Preregistered primary candidate: canonical K is inverse-RoPE aligned before
  forming one temporal summary group and re-encoded at the group center, with
  5% regular event tiles and rank-16 AV-defect oracle/frozen-basis diagnostics.
  A direct post-RoPE mean is retained as a required weaker baseline.
- Incumbent: exact LongLive sink and local-window attention.
- Required baselines: drop-old window, one recall-frame mean, recency tree
  without event tiles, and event tiles without low-rank correction.

## Data and leakage guards

- Official LongLive v1.0 commit:
  `e52d9ef6865d843282a6b5e9d46d03b35f88929d`.
- Eight fixed prompt/seed records: two calibration prompts with two seeds each,
  one validation prompt with two seeds, and one test prompt with two seeds.
  This pre-capture amendment separates within-prompt seed stability from
  cross-prompt transfer without changing the candidate or any gate.
- Layer, denoising-call, head, and query-tile coordinates are frozen in the
  JSON config before capture.
- All 12 attention heads are captured at each selected cell because existing
  AR-video work reports strong head-role heterogeneity; query computation is
  bounded with four fixed 64-token tiles per generated frame.
- Dense AV from validation/test may only be used for final metrics and
  explicitly labelled per-record oracle coefficients. It may not select a
  basis, event rule, rank, or threshold.
- Every capture must record source commit, checkpoint hashes, prompt ID, seed,
  layer, call index, query/key lengths, dtype, and frame sequence length.
- The frozen protocol pins both official checkpoint SHA-256 values. Evaluation
  accepts only tensors explicitly listed by a manifest whose protocol,
  runtime-config, source, generator, and LoRA signatures match.

## Endpoints

Primary endpoint: dense-relative AV L2. Report aggregate, record, layer, and
head values. Guards: exact shared normalization, finite values, capture
identity, representative coverage, no duplicated or omitted source token, and
deterministic rerun agreement.

## Outcome mapping

- Pass: the primary candidate's adaptive rank-16 oracle is at most 0.5%
  aggregate and 1% worst, then its calibration-frozen basis is at most 1%
  aggregate and 2% worst, with at least 1.5x arithmetic key reduction.
- Null: valid oracle misses a quality gate. Stop predictor and kernel work.
- Boundary: only selected heads/layers pass. Retain certified heterogeneous
  fallback and narrow the claim.
- Adverse: compressed output is worse than drop-old at equal cost, or requires
  more than 25% event payload. Park this candidate.
- Invalid: identity, split, normalization, or dense-reference guard fails.
  Repair once and rerun without changing the scientific gate.

No end-to-end or H200 speedup claim is allowed from this protocol. A separate
kernel protocol is required after representation and transfer both pass.

## Pre-capture arithmetic correction

LongLive exposes 12 key frames in this setting. Keeping three sink frames and
the current three-frame block already retains 6/12 frames. Dropping all middle
history is therefore exactly a 2x key-count reference; adding any useful
summary makes 2x mathematically impossible. The primary one-group, 5%-event
candidate has an expected reduction of about 1.64x. The representation gate is
therefore 1.5x, while 2x remains the drop-middle baseline. This correction
changes neither captured data nor quality thresholds and prevents an
impossible pass condition.
