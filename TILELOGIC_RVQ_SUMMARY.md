# TileLogic-RVQ Formal Evaluation Summary

## Method

TileLogic-RVQ extends TileSpec-Ex with a rate-accounted vector-quantization
path:

```text
tile-local DCT
  -> scaled base VQ
  -> sequential residual VQ
  -> calibration-only variable-depth router
  -> fixed-slot structured execution
  -> optional sparse exact FP16 fallback
```

The logic component selects drop/RVQ1/RVQ2/exact modes. It does not replace
the continuous FP16 codebooks or reconstruction datapath. The implementation
compares ten main methods, one unweighted-RVQ ablation, and an uncompressed
reference at 12.5% and 25% crop-token retention. The complete frozen protocol
and decision thresholds are in
[`TILELOGIC_RVQ_EXPERIMENT.md`](TILELOGIC_RVQ_EXPERIMENT.md).

## Formal Protocol

- Model: Qwen2.5-VL-3B-Instruct with the existing five-image adapter.
- Data: GQA, TextVQA, and ChartQA.
- Training: 80 calibration samples per dataset, 240 total.
- Evaluation: 120 disjoint samples per dataset, 360 total.
- Oracle routing subsets: 16 calibration and 16 disjoint evaluation samples
  per dataset.
- Quality matrix: 23 variants on every evaluation sample.
- Rate: exact stream bits plus codebooks, scales, routers, trees, and fixed-slot
  metadata amortized over all 360 evaluation samples.
- Precision: FP32 scalar scales, VQ metric weights, MLP/normalizer state,
  logic leaves, and curvature priors; FP16 codewords, scale tables, logic
  thresholds, and exact fallback values. A lower precision is charged only
  after that value is round-tripped through the declared representation.
- Quality: dataset answer score and teacher-forced NLL. NLL uses the first
  manifest answer verbatim; answer scoring uses all manifest answers.
- Latency: paired dynamic/fixed PyTorch component, prefill, TTFT, and peak GPU
  memory diagnostics. Every quality and timing path expands to 1,280 visual
  tokens before Qwen execution.

Training loaded zero evaluation records, formal evaluation loaded zero
calibration records, and the feature and quality runs use the same 360 sample
keys. The independent machine audit passed all required checks.

## Review-Driven Rate Correction

The sole Review Agent rejected the first publication candidate because several
FP32 values were charged as 16 or 8 bits. The correction changed the logical
rate ledger, added the previously omitted curvature prior, and made the machine
audit derive expected component sizes directly from training artifacts and
verify exact declared-precision round trips for logical payloads. It did not
quantize values after the fact or relabel FP32 execution as a lower-bit
implementation.

The 360-sample feature evaluation was rerun from an empty output directory.
The old and corrected JSONL files were then compared across all 8,280 variants:
all feature, Fisher, router-oracle, mode, and budget fields were identical.
Only the seven allowlisted rate components changed. Cache provenance was also
backfilled for all 600 entries with source-manifest hash, model revision, and
tensor dtype; cached tensor files and their hashes were unchanged.

## Main Results

Base VQ has a much lower fully amortized rate than INT4 at both operating
points, but its feature and quality losses exceed the frozen guardrails:

| Method | Retention | Effective bit/value | Feature NMSE | Teacher NLL | Answer score |
|---|---:|---:|---:|---:|---:|
| INT4 base | 12.5% | 0.501953 | 0.619083 | 0.976434 | 0.641667 |
| Base VQ | 12.5% | 0.011931 | 0.764094 | 1.041761 | 0.627778 |
| INT4 base | 25% | 1.003906 | 0.507636 | 0.956601 | 0.650000 |
| Base VQ | 25% | 0.012663 | 0.728805 | 1.048196 | 0.619444 |

Fisher-weighted RVQ improves over base-only VQ, but is minutely worse than the
unweighted RVQ ablation on Fisher NMSE at both rates (`0.664180` versus
`0.664162`, and `0.623797` versus `0.623766`). The learned MLP router also
does not beat the stronger energy/risk heuristic in both required oracle
metrics. Its top-k recall is `0.015625` and `0.057292`, compared with
`0.052083` and `0.084635` for the stronger heuristic.

Fixed slots reduce paired median TTFT by 23.2% at 12.5% retention and 35.9% at
25%, but layout is not lower at either rate, residual decode is not lower at
25%, and the 25% quality guardrail fails. This timing is diagnostic only:
there is no shorter native visual sequence or fused kernel. The exact fallback
is charged for its full FP16 payload; it improves teacher NLL by 1.78% and
1.19% versus fixed slots, but fails the required rate-distortion/quality
combination.

## Frozen Decisions

| ID | Question | Status | Primary reason |
|---|---|---|---|
| Q1 | Base VQ extends the full-overhead frontier beyond INT4 | FAIL | Feature, NLL, and dataset-score guardrails fail. |
| Q2 | Fisher RVQ beats base-only and unweighted RVQ | FAIL | Fisher distortion is slightly worse than unweighted RVQ. |
| Q3 | MLP routing beats energy/cosine-risk routing | FAIL | Required Spearman/top-k improvements are not met. |
| Q4 | Discrete logic retains the MLP benefit | INCONCLUSIVE | Q3 prerequisite fails. |
| Q5 | Fixed slots lower layout, decoder, and TTFT | FAIL | All components do not improve at both rates; one quality guard fails. |
| Q6 | Fully charged exact fallback improves the frontier | FAIL | Frontier and per-dataset quality requirements are not jointly met. |

No aggregate positive claim is supported.

## Evidence

- [Formal report](remote_logs/tilelogic_rvq_20260718/analysis/TILELOGIC_RVQ_FINAL_REPORT.md)
- [Machine audit](remote_logs/tilelogic_rvq_20260718/analysis/result_audit_report.md)
- [Sole Review Agent report](remote_logs/tilelogic_rvq_20260718/analysis/independent_review_report.md)
- [Machine-readable decisions](remote_logs/tilelogic_rvq_20260718/analysis/decision_summary.json)
- [Rate and quality points](remote_logs/tilelogic_rvq_20260718/analysis/method_points.csv)
- [Exact rate components](remote_logs/tilelogic_rvq_20260718/analysis/rate_components.csv)
- [Latency table](remote_logs/tilelogic_rvq_20260718/analysis/latency_metrics.csv)
- [Sanitized bundle manifest](remote_logs/tilelogic_rvq_20260718/PUBLICATION_MANIFEST.json)

The public bundle excludes model checkpoints, datasets, cached tensors,
trained `.pt` payloads, and per-example questions/answers/predictions. It is a
claim-inspection bundle, not a standalone reproduction of model inference.

## Recommended Next Experiment

Do not introduce an aggregate positive claim. First improve the base VQ
codebook and routing target on calibration data, then rerun the same frozen evaluation.
The next implementation experiment should isolate whether a compact visual
sequence and fused fixed-slot decoder can preserve quality while converting
the observed TTFT diagnostic into a real end-to-end prefill test.
