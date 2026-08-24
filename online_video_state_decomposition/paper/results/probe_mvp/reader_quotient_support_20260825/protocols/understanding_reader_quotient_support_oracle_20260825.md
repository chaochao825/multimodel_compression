# Reader-Quotient Sparse-Support Oracle Protocol

Date: 2026-08-25
Status: frozen diagnostic

## Decision question

At the exact byte budget of the existing `PCA-r256 + s4` native-feature
codec, does a native-reader Fisher/GGN metric select sparse residual tokens
that preserve the frozen reader distribution better than Euclidean residual
energy?

This is a function-class capacity probe. It is not a deployable writer because
the oracle score observes the full feature tensor and the current question.

## Fixed objects

- Frozen LLaVA-v1.5-7B model and frozen rank-256 codec.
- Frozen MVBench transfer manifest and `exact_recent` evidence frames.
- Sixteen-frame native feature pool, `8x8` pooled tokens per frame.
- Four exact FP16 residual vectors and four int16 positions per frame.
- The same PCA latents, evidence frames, prompts, candidate tokens and model
  forward path for every method.

## Compared methods

1. `pca_only`: no sparse residual, diagnostic lower baseline.
2. `euclidean_s4`: existing top-4 residual-energy support.
3. `fisher_s4`: top-4 support under candidate-distribution diagonal Fisher.
4. `mixed_s4`: equal normalized mixture of Euclidean and Fisher scores.

For candidate log probabilities `l_y` and full-feature candidate
probabilities `p_y`, the per-coordinate diagonal metric is

\[
G_{i,d}=\sum_y p_y\left(\frac{\partial l_y}{\partial X_{i,d}}\right)^2.
\]

The Fisher support score is

\[
s_i^{F}=\sum_d G_{i,d}(X_{i,d}-\hat X_{i,d})^2.
\]

All non-selected feature-pool frames retain the Euclidean support, so every
method has the same complete-state payload. The Fisher choice is applied only
to frames read by the frozen policy.

## Endpoints

Primary endpoint: paired candidate-distribution KL from full native features.

Secondary endpoints:

- full-vocabulary first-token KL;
- candidate top-1 agreement with full native features;
- ground-truth candidate log-probability change;
- feature relative L2 error;
- support overlap with Euclidean selection;
- wall-clock forward/backward diagnostic cost.

## Scope and decision mapping

Run a 5-sample smoke test, then 40 transfer samples (`8` per task) only if all
metrics are finite and the Euclidean reconstruction matches the existing codec
selection. This transfer set is diagnostic-only and cannot become a final
deployment endpoint.

- `GO`: a Fisher or fixed mixed rule reduces aggregate candidate KL by at
  least `25%` versus `euclidean_s4`, does not increase the 95th-percentile KL,
  and does not reduce candidate top-1 agreement by more than `1pp`.
- `WEAK`: aggregate KL improves by `10%` to `25%` with no tail regression.
- `NULL`: aggregate KL improves by less than `10%`.
- `ADVERSE`: aggregate or 95th-percentile KL is worse by more than `10%`, or
  top-1 agreement falls by more than `1pp`.
- `INVALID`: instrumentation changes the full-feature logits, payloads differ,
  gradients are non-finite, or the Euclidean support does not match the codec.

Only `GO` authorizes a calibration-only, query-agnostic Fisher predictor and a
strong-reader replication. `WEAK` may justify one small implementation audit;
`NULL` or `ADVERSE` parks Reader-Quotient Structured Memory.

## Claim boundary

A positive result would show only that native readout geometry can improve
equal-rate sparse support inside this frozen codec. It would not establish
MVBench accuracy, online deployability, TTFT speedup, superiority to token
selectors, or a general theorem for video VLMs.
