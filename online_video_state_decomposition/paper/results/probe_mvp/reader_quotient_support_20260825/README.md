# Reader-Quotient Structured Memory probes

This package contains the bounded LLaVA-v1.5 and LLaVA-OneVision experiments
completed through 2026-08-29. It keeps protocols, aggregate/per-sample metrics,
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

The canonical model-side runners and tests live under
`online_video_state_decomposition/experiments/` in this repository.
