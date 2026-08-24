# LLaVA-OneVision Equal-Budget Rank/Support Allocation

Date: 2026-08-25
Status: frozen before scientific execution

## Decision question

Did the preceding OneVision Reader-Quotient probe stop at `BOUNDARY` because
`PCA-r384 + s4` leaves a perturbation too large for a diagonal Fisher metric,
or because diagonal Fisher support is itself an unstable function class?

## Frozen model, samples, and reader

- Reuse the frozen LLaVA-OneVision Qwen2-7B model, processor, prompts, frame
  sampling, and the same 20 allocation-selection samples from the strong-reader
  replication.
- These 20 observed samples may select one allocation but cannot confirm it.
- Fit one PCA-r456 basis on the original 20 calibration samples only. Every
  lower-rank codec is an ordered prefix of that same basis; independent PCA fits
  per rank are forbidden.
- Compute the native candidate Fisher once per sample from the uncompressed
  reference and reuse it for every allocation.
- Support scoring and PCA reconstruction remain FP32. BF16 conversion occurs
  only at frozen-reader injection.

## Frozen allocation family

| Rank | Exact residual tokens/frame | Tensor payload bytes | Difference from maximum |
|---:|---:|---:|---:|
| 384 | 4 | 2,867,328 | 0 |
| 402 | 3 | 2,865,504 | -1,824 |
| 420 | 2 | 2,863,680 | -3,648 |
| 438 | 1 | 2,861,856 | -5,472 |
| 456 | 0 | 2,860,032 | -7,296 |

The maximum spread is `0.2545%`; payloads are not described as byte-identical.
For each nonzero-support allocation compare Euclidean, Fisher, and the already
frozen equal-weight Euclidean/Fisher mixed score. Rank 456 is a PCA-only
endpoint and has no reader-aware support variant.

## Endpoints and selection rule

For every Fisher/Mixed variant, the paired baseline is Euclidean support at the
same rank and residual count. The original nested `euclidean_r384_s4` endpoint
is also the absolute reference across allocations.

A configuration is `GO` only if all conditions hold:

- aggregate candidate-KL reduction versus its paired Euclidean baseline is at
  least `25%`;
- candidate-KL P95 ratio versus the paired baseline is at most `1.0`;
- at least four of five tasks have positive aggregate reduction;
- candidate top-1 agreement with the uncompressed reader does not fall below
  its paired Euclidean baseline;
- absolute aggregate candidate KL is no worse than `euclidean_r384_s4`;
- measured tensor payload does not exceed `2,867,328` bytes.

Positive aggregate reduction that misses a guard is `BOUNDARY`. Material
aggregate or P95 harm is `ADVERSE`; otherwise the result is `NULL`.

## Consequences

- A `GO` only selects an allocation for a fresh confirmation set. It does not
  authorize scorer training or a deployment claim.
- Confirmation uses the five untouched tasks `fine_grained_action`,
  `action_antonym`, `unexpected_action`, `counterfactual_inference`, and
  `action_count`, with the complete allocation and score fixed.
- If no allocation is `GO`, stop diagonal-Fisher support development. Retain
  the Euclidean codec and move effort to encoder/prefill system optimization.
- One fixed reference-margin guard may be tested after this sweep, but no
  coefficient may be tuned on these 20 samples.
