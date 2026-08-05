# QuantVideoGen Probe on LongLive-1.3B

Status: frozen on 2026-08-05 before evaluating QuantVideoGen on the existing
LongLive captures. The Q/K/V captures predate this protocol, so this is a
post-capture representation study rather than a capture-level preregistration.

## Decision question

Test whether QuantVideoGen's content-adaptive semantic residual representation
removes the failure mode observed for fixed temporal summaries and static
low-rank tails on LongLive. The primary arm is one-stage, 256-centroid-per-head,
block-64 INT2 quantization of all cached K/V tokens.

QuantVideoGen represents each cached vector as

\[
x_i = c_{\pi_i}^{(1)} + \cdots + c_{\pi_i}^{(S)} + r_i,
\qquad \hat r_i = Q_b(r_i),
\]

where centroids and assignments are generated from the current chunk rather
than shared across prompts. This preserves every token and allows the codebook
to rotate with content. It therefore tests a different hypothesis from the
failed LongLive temporal-summary experiment, which removed tokens and then
asked a static or oracle low-rank output tail to repair the resulting defect.

## Arms

- RTN INT2, block size 64, no semantic smoothing.
- QVG INT2, one semantic residual stage, block size 64. This is primary.
- The same QVG arm with three sink and three recent frames left exact.
- QVG INT4, one stage, block size 64.
- QVG-Pro INT2, four stages, block size 16.

The settings follow the official QVG Self-Forcing launch defaults where
applicable: 256 K/V centroids, two K-means iterations, and one PRQ stage. The
QVG-Pro arm follows the official benchmark's four-stage/block-16 definition.

## RoPE handling

The existing captures store the K passed into attention, after 3D RoPE. QVG
stores pre-RoPE K because its distribution is more quantization-friendly. For
each captured absolute frame and spatial location, this probe removes the exact
captured complex RoPE multiplier, quantizes canonical K, and restores the same
multiplier before computing attention. V is quantized directly. No dense output
is used in this process.

## Endpoints and gates

The numerical endpoints are K reconstruction error, V reconstruction error,
sampled-query dense-relative AV error, per-head worst AV error, and actual bytes
of every packed tensor including centroids, labels, residuals, scales, and exact
chunks. The primary gates are:

- aggregate AV error at most 1%;
- worst-head AV error at most 2%;
- packed K/V compression at least 6x;
- dense-reference parity below 0.5%.

Passing these gates supports only a cache-memory result. The official QVG path
reconstructs full-precision K/V before ordinary attention, so this protocol does
not claim attention FLOP reduction or H200 speedup.

## Follow-up boundary

Only a passing primary result can open a new protocol for query-adaptive
progressive decoding. That follow-up would use centroid-only, partial residual,
or exact-token refinement under one shared softmax and would need a separate
quality, bandwidth, kernel, and end-to-end gate. BCM/BCCB may be tested there as
a compact spatial predictor for assignments or refinement tiles, not as a
replacement for the complete attention matrix.
