# Reader-Quotient Structured Memory probes

This package contains the bounded LLaVA-v1.5 and LLaVA-OneVision experiments
completed through 2026-08-30. It keeps protocols, aggregate/per-sample metrics,
analysis scripts, reports, and publication figures together without model
weights, MVBench videos, native feature tensors, or remote runtime logs.

## Evidence sequence

1. A query-visible diagonal-Fisher support oracle improved equal-byte sparse
   residual placement on LLaVA-v1.5.
2. An old-task static Fisher prior was adverse on disjoint transfer samples.
3. LLaVA-OneVision replicated a positive but unstable capacity signal and was
   classified `BOUNDARY`.
4. A frozen equal-budget allocation sweep also remained `BOUNDARY`; its best
   simple selection-set endpoint moved all sparse bytes into a wider PCA-r456
   bulk state.
5. The frozen PCA-r456 state was then tested on five untouched tasks and 500
   samples. Accuracy changed from 54.2% to 55.2%, the harmful-event upper 95%
   bound was 1.543%, and state payload fell 7.86x. Prediction agreement was
   96.8%, below the frozen 98% gate, so the confirmation remains `BOUNDARY`.
6. Without changing the codec, a cross-domain replication then used 600 unique
   Video-MME videos, balanced across short, medium, and long durations. Accuracy
   changed from 55.17% to 54.17%, state payload again fell 7.86x, and all
   duration losses stayed within two points. Agreement was 95.83% and the
   harmful-event upper 95% bound was 2.603%, so this result is also `BOUNDARY`.
7. A same-rank domain-residual gate then fit visual-only codecs on 120 balanced
   Video-MME calibration videos and evaluated 180 disjoint selection videos.
   Target PCA-r456 reduced mean KL, P95 KL, and feature L2 to
   `0.521x/0.605x/0.852x` of the source codec, but increased prediction
   mismatches from 6 to 8. The result is `CAPACITY_ONLY`; the frozen 255-video
   formal reserve was not run. Residual-swap-r128 reduced mismatches to 4 with
   no harmful flips but did not pass the continuous-risk gates, exposing a
   reader-risk versus feature-energy Pareto split.
8. A fresh VSI calibration audit then equalized the three covariance budgets,
   cross-fit reader-risk over three disjoint folds, and evaluated fixed-rank
   CMRQ candidates. Risk-32 improved mean KL by 25.45% versus pooled PCA and
   boundary mixing improved it by 29.56%, but static candidates did not jointly
   dominate continuous and discrete risk. A compressed-margin-zero exact tier
   removed the only harmful boundary-mix flip with 7/72 fallbacks and retained
   a conservative 4.46x state-transfer ratio. This is a calibration-only
   conditional signal; the frozen 60-scene selection and 63-scene formal sets
   remain untouched.

No result in this package establishes strict strong-reader interchangeability,
a deployable content scorer, TTFT reduction, or end-to-end GPU speedup.

## Layout

- `protocols/`: frozen decision rules used before each scientific run.
- `analysis/`: CSV/JSON summaries and generated Markdown tables.
- `scripts/`: analysis and plotting entry points.
- `tests/`: frozen decision-rule unit tests for the analysis layer.
- `figures/`: PNG/PDF/SVG figures plus bound plotting data where applicable.
- `reports/`: Chinese evidence synthesis and method-boundary reports.
- `metadata/`: calibration summaries for the two OneVision PCA bases.

The latest evidence and next-method boundary are summarized in
`reports/ONEVISION_READER_QUOTIENT_CMRQ_20260830.zh-CN.md`.

The canonical model-side runners and tests live under
`online_video_state_decomposition/experiments/` in this repository.
