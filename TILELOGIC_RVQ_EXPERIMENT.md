# TileLogic-RVQ Experiment Contract

## Objective

Evaluate whether TileSpec-Ex can be extended from fixed orthogonal transform
coding plus sparse exact residuals into a leakage-free, rate-accounted
TileLogic-RVQ codec:

```text
tile-local DCT + scaled base VQ + Hessian-aware residual VQ
    + variable-depth routing + fixed slots + sparse exact fallback
```

The logic component selects an encoding path. Continuous codebooks reconstruct
vectors. A positive rate-distortion result is not a latency claim, and a
quality run that expands back to 1,280 visual tokens is not compact prefill.

## Data Contract

- Model: Qwen2.5-VL-3B-Instruct with the existing five-image adapter.
- Datasets: GQA, TextVQA, and ChartQA, 200 cached examples per dataset.
- Spatial crop-token rates: 12.5% and 25%; the 256-token thumbnail is unchanged.
- Split each dataset by stable hash rank into 80 calibration and 120 evaluation
  examples. Codebooks, scalar ranges, router thresholds, fixed positions, and
  Fisher/Hessian proxies use calibration examples only.
- Calibration oracle subset: 16 examples per dataset.
- Evaluation oracle subset: 16 disjoint examples per dataset.
- Every cache entry records the source manifest hash, image hash, split, model
  revision, tensor shapes, and dtype. Split overlap is a hard audit failure.

## Method Matrix

The following methods use the same evaluation records and both spatial rates:

1. `tile_lowpass`: floating-point transform baseline.
2. `tile_energy_exception`: floating-point sparse-residual baseline.
3. `tile_risk_exception`: query-aware floating-point baseline.
4. `base_scalar_quant`: per-vector symmetric INT4 base coefficients.
5. `base_vq`: scaled 256-entry base codebook.
6. `base_vq_residual_rvq`: base VQ plus two-stage residual VQ, energy routed.
7. `base_vq_mlp_router`: the same codebooks with an oracle-trained MLP router.
8. `base_vq_logic_router`: a discrete tree distilled from the MLP using
   quantized deployable features.
9. `logic_router_fixed_slots`: calibration-fixed per-tile block positions and
   discrete valid/depth states, with no dynamic position stream.
10. `logic_router_fixed_slots_exact_fallback`: the fixed-slot method with an
    explicitly accounted FP16 residual fallback for selected outliers.

The base codebook is shared across rates. Residual codebooks are trained on
the actual post-base-VQ error with sequential error feedback. The diagonal
empirical-Fisher proxy is normalized and clipped before weighted codebook
training; unweighted distortion is always reported beside it.

## Router Contract

The oracle target is the calibration-only first-order loss proxy:

```text
abs(<dL/dX_block, residual_block>)
```

Deployable block features are energy, query cosine, local variance, residual
RMS, tile ID, normalized row/column, distance from the low-frequency corner,
thumbnail agreement, and the calibration-only curvature prior. The router may
not consume answers, gradients, losses, evaluation labels, or reconstructed
model outputs at inference.

The MLP establishes a routing upper bound. The logic router uses calibration
threshold bits and a bounded-depth binary decision tree. Fixed slots are
chosen once from calibration statistics and are immutable during evaluation.

## Rate Accounting

Report exact stream bits for every sample:

```text
base indices + base scales + residual position/mode/index/scale bits
    + valid masks + padding + exact fallback payload + headers
```

Also report shared model overhead:

```text
base/residual codebooks + scale tables + scalar scales
    + router weights/tree thresholds/leaves + fixed-slot metadata
```

Required denominators are stream bits per original crop vector, stream bits
per original crop scalar, effective bits including codebook overhead amortized
over the 360-example evaluation set, and asymptotic stream-only bits. Codebook
overhead must never be silently excluded from a compression claim.

## Required Metrics

- GQA, TextVQA, and ChartQA answer score.
- Teacher-forced answer NLL.
- Teacher-forced NLL uses the first manifest answer as a fixed verbatim target;
  dataset answer scoring continues to use the complete manifest answer set.
- Feature NMSE and cosine similarity.
- Diagonal-Fisher-weighted distortion and oracle-subset first-order loss proxy.
- Stream/effective bit rate, codebook/metadata overhead, and break-even count.
- Router Spearman correlation, top-k recall, mode/depth usage, and budget error.
- Selector, codebook search, layout/gather, decoder, reconstruction, prefill,
  end-to-end TTFT, and peak allocated/reserved GPU memory.

Latency rows must state whether they include image preprocessing, visual
encoding, native multimodal positions, reconstruction, language prefill, and
first-token generation. Incomplete diagnostics remain `INCONCLUSIVE` for real
latency benefit.

## Decision Questions

1. Does base VQ improve the feature/NLL rate-distortion frontier over INT4
   scalar quantization after full overhead accounting?
2. Does Hessian-aware RVQ improve over base-only VQ and unweighted RVQ?
3. Does an oracle-trained MLP router beat energy and cosine-risk routing?
4. Does the discrete logic router retain the MLP routing benefit?
5. Do fixed slots reduce measured layout/decoder/TTFT cost without an
   unacceptable task-quality regression?
6. Does sparse exact fallback improve the frontier after its FP16 payload is
   charged in full?

Each question is reported independently as `PASS`, `FAIL`, or `INCONCLUSIVE`.
No aggregate positive claim is allowed unless all evidence needed by that
question is present and recomputed by the audit.

## Predeclared Decision Rules

These rules are frozen before the formal feature, quality, and latency runs.
All quality comparisons are paired on the same 360 evaluation examples. Rate
means effective bits per original crop scalar, including shared overhead
amortized over those 360 examples; stream-only rate is secondary context.
Missing required evidence produces `INCONCLUSIVE`, while complete evidence
that misses a threshold produces `FAIL`.

The common quality guardrail is no more than a 0.5 percentage-point answer
score loss on any dataset, no more than a 1% aggregate teacher-NLL increase,
and no more than a 2% aggregate feature-NMSE increase relative to the named
comparator.

1. Base VQ versus INT4 is `PASS` only if the two base-VQ points add a strict
   non-dominated point to both the effective-rate/feature-NMSE and
   effective-rate/teacher-NLL frontiers relative to scalar INT4, while meeting
   the common quality guardrail.
2. Fisher RVQ is `PASS` only if its mean Fisher-weighted distortion is lower
   than both base-only VQ and equal-budget unweighted RVQ at both spatial
   rates, while meeting the feature-NMSE and teacher-NLL guardrails. The added
   effective bits over base-only are reported even when this gate passes.
3. The MLP router is `PASS` only if, on the 48 disjoint evaluation-oracle
   examples, its mean Spearman correlation and top-k recall each exceed the
   stronger of energy and cosine-risk routing by at least 0.02 at both rates,
   while meeting the common quality guardrail against energy routing.
4. Logic-router retention is evaluated only when question 3 passes. It is
   `PASS` only if logic preserves at least 90% of the MLP improvement over the
   stronger heuristic for both oracle metrics at both rates and meets the
   common quality guardrail relative to the MLP.
5. Fixed slots are `PASS` only if paired median `layout_pack`,
   `residual_decode_scatter`, and end-to-end TTFT are all lower than the
   dynamic logic-router path at both rates, and the fixed method meets the
   common quality guardrail. These measurements still expand to 1,280 visual
   tokens and therefore cannot establish native compact-prefill speedup.
6. Exact fallback is `PASS` only if, after charging the complete FP16 payload,
   it adds a non-dominated point to both effective-rate/feature-NMSE and
   effective-rate/teacher-NLL frontiers and improves its paired no-fallback
   feature NMSE by at least 5% or teacher NLL by at least 1%, without violating
   the dataset score guardrail.
