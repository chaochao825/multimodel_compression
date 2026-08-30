# VSI Progressive CMRQ Selection Gate

Date frozen: 2026-08-30

Role: fresh selection only

## Decision question

Does the calibration-positive Progressive CMRQ mechanism transfer to the
untouched VSI selection role without tuning, while retaining a useful state
transfer ratio and preserving the frozen OneVision reader's multiple-choice
behavior?

This gate evaluates state delivery and reader behavior. It does not establish
formal generalization, token-count reduction, reader-compute reduction, TTFT, or
wall-clock acceleration.

## Frozen identities

- Split protocol: `vsi_onevision_reader_quotient_stage_a_20260830_v1`.
- Fit data: all 120 calibration scene features and the 72 calibration-only
  reader-risk questions.
- Evaluation data: all 49 selection scenes containing a debiased question, one
  question per eligible scene. The other 11 frozen selection scenes have no
  debiased question and are not reader endpoints.
- Formal reserve: 63 scenes; they remain unread in this gate.
- Reader: frozen LLaVA-OneVision Qwen2-7B.
- Frame policy: 16-frame feature pool and uniform 8-frame reader input.
- Rank: 456.
- Random seed: 20260830.

Selection questions, answers, features, gradients, logits, and errors must not be
used to fit or alter a mean, basis, risk atom, permutation, weight, rank, or
threshold.

## Frozen candidates

1. three-domain pooled PCA-r456;
2. VSI calibration PCA-r456;
3. pooled bulk plus 32 reader-risk atoms at rank 456;
4. boundary mix with 32 risk atoms and weight 0.3 at rank 456;
5. same-weight permuted-risk boundary-mix null;
6. progressive boundary mix using compressed top-1 margin `<= 0` as the only
   exact fallback rule.

The final bases are fitted once from all calibration evidence before any
selection feature is read by the evaluator.

## Primary endpoints

- paired candidate-distribution KL for boundary mix versus the permuted null;
- candidate KL mean and P95 versus VSI PCA;
- frozen-reader agreement, harmful flips, and multiple-choice accuracy;
- exact fallback rate;
- conservative and ideal pre-route state-transfer ratios.

The conservative ratio charges both the compressed payload and a full payload
on fallback. No compute speedup is inferred from this byte model.

## Decision rule

The result is `GO` only when every condition holds:

- the paired bootstrap 95% upper bound for `KL(mix) - KL(permuted)` is below 0;
- mix P95 KL is below the permuted null P95;
- mix mean and P95 KL are no worse than VSI calibration PCA;
- progressive delivered agreement is at least 98%;
- remaining harmful flips are zero;
- fallback is at most 15%;
- conservative state-transfer ratio is at least 4x;
- delivered task accuracy is no more than one percentage point below the full
  reader.

Failure of any condition is `NO_GO` for fixed-basis Progressive CMRQ. A `NO_GO`
closes atom-count, risk-weight, and margin-threshold tuning on this split. A
`GO` permits a separate quotient-indexed evidence-retrieval probe; it does not
automatically authorize the formal reserve.

## Pre-evaluation identity repair

The first execution stopped before model evaluation because the original text
incorrectly equated 60 frozen scenes with 60 debiased questions. The split
contains 49 eligible selection scenes. No selection feature, logit, answer, or
error was read by the evaluator before this correction; candidates and decision
thresholds are unchanged.
