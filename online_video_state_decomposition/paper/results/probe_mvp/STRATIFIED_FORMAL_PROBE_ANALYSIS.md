# Stratified Formal Hidden-State Probe Analysis

## Status and Evidence Boundary

This report covers the completed formal hidden-state probe on 30 stratified
Video-MME segments, with six segments in each proxy category:
`camera_motion`, `high_change`, `object_motion`, `scene_cut`, and `static`.

The probe uses the LLaVA CLIP ViT-L/14-336 visual tower, 16 sampled frames per
segment, and layers 0/7/15/22/23. It evaluates representation spectra, causal
subspace reconstruction, residual concentration, optical-flow transport, local
BTTB transport, and local BCCB transport.

This is representation-level evidence. It is not language-conditioned online
QA accuracy, semantic event detection, long-duration streaming, or measured
GPU latency.

The formal aggregate passed independent validation:

- 30 runs, with exactly six runs per category;
- 2,100 category-rank rows, 350 causal-subspace rows, 375 residual rows, and
  275 transport rows;
- all numeric aggregate values are finite;
- all required CSV, JSON, PNG, and PDF artifacts are present and non-empty.

## Gate Summary

| Probe | Formal result | Decision |
|---|---|---|
| Rank-32 token-normalized state energy | Mean 0.766; 19/25 category-layer cells exceed 0.70 | Retain a layer-selective low-dimensional state candidate |
| Rank-32 causal projection | Mean error 0.393; range 0.226-0.640 | Offline spectrum alone is insufficient |
| Top-10% residual energy | Mean 0.482; 4/25 cells exceed 0.70 | Energy concentration is isolated to layer 15 |
| Top-10% pixel-change recall | Mean 0.205; maximum 0.367 | The event-proxy gate fails |
| Residual joint gate | 0/25 cells pass both energy and recall gates | Remove sparse-event language from the main claim |
| Optical flow versus identity | Mean eligible gain -0.9%; maximum 8.5%; 0/20 pass 10% | Do not promote this flow implementation |
| Local BTTB versus identity | Mean eligible gain 9.61%; 9/20 pass 10% | Retain as an optional motion-regime predictor |
| Local BCCB versus identity | Mean eligible gain 9.64%; 9/20 pass 10% | Same regime as BTTB |
| BCCB incremental gain versus BTTB | Mean 0.037%; range -0.127% to 0.322% | No specifically cyclic advantage |

Transport cells whose identity reference error is below 0.05 are excluded from
relative-gain decisions. This removes numerically unstable static cells from
the transport gate.

## Probe 1: Persistent State

Rank-32 token-normalized energy exceeds the 0.70 gate in 19 of 25
category-layer cells. The layer means are:

| Layer | Rank-32 energy | Rank-32 causal error |
|---:|---:|---:|
| 0 | 0.832 | 0.452 |
| 7 | 0.658 | 0.585 |
| 15 | 0.704 | 0.281 |
| 22 | 0.825 | 0.317 |
| 23 | 0.812 | 0.330 |

Layer 7 fails the spectral gate in every dynamic category, and layer 15 fails
it for `high_change` and `object_motion`. Layers 22 and 23 are the most
consistent deep interfaces. At rank 16, layer 22 still retains 0.768 mean
token-normalized energy, although its causal projection error is 0.367.

Decision: **retain low-dimensional state only as a layer-selective candidate**.
The causal error is too large to treat truncated subspace reconstruction as a
complete state update, but it is compatible with a hybrid that reserves exact
recent evidence.

## Probe 2: Sparse Innovation

Only four category-layer cells exceed the 0.70 top-10% residual-energy gate:

- layer 15 on `camera_motion`: 0.790;
- layer 15 on `high_change`: 0.752;
- layer 15 on `object_motion`: 0.803;
- layer 15 on `scene_cut`: 0.786.

The same cells do not align with the pixel-change proxy. Across all cells, the
maximum top-10% proxy recall is only 0.367, and the mean is 0.205. No cell
passes the joint 0.70 energy and 0.80 recall gate.

Decision: **the current sparse-event mechanism fails**. Layer-15 concentration
is better interpreted as a representation-specific magnitude or low-rank
effect until semantic event annotations and causal router interventions show
otherwise.

## Probe 3: Structured Transport

Local BTTB improves stable centered prediction most clearly on dynamic motion:

- `high_change`: 10.3-13.5% at layers 15/22/23 and 13.2% at layer 0;
- `object_motion`: 11.2-16.9% across all five tested layers;
- `camera_motion`: 2.1-8.5%, below the gate;
- `scene_cut`: 2.8-9.5%, below the gate.

The optical-flow implementation passes no 10% cell and has a mean eligible gain
of -0.9%. It is especially harmful on camera-motion and scene-cut proxies.
This does not establish that optical flow is generally unsuitable; the result
is specific to the sampled-frame cadence, flow estimator, and hidden-token
alignment used here.

Local BCCB tracks BTTB almost exactly. Its mean incremental gain over BTTB is
0.037%, with a maximum of 0.322%. This is far too small to justify periodic
wraparound, FFT machinery, or a circulant contribution claim.

Decision: **retain local BTTB only as an optional motion-regime predictor and
remove BCCB from the title and primary method**.

## Joint Interpretation with the Memory Probe

The matched-byte memory diagnostic on the same 30 segments resolves the main
failure of a memory-only subspace. At about 64 KiB, an instant-plus-Oja state
preserves delay-0 cosine at 1.000 and improves delay-8 cosine by 0.165 over a
recent-window baseline. At about 128 KiB, allocating three exact instant frames
plus rank-16 Oja memory:

- matches the recent baseline through delays 0-2;
- improves mean delay-4 cosine by 0.029;
- improves mean delay-8 cosine by 0.126;
- uses fewer read and update FLOPs than the tested one- and two-frame
  allocations.

The formal probes therefore support a two-timescale representation MVP:

`three-frame exact instant cache + rank-16 causal subspace memory`

Use layer 22 as the primary interface and layer 15 only as a dynamic-content
ablation. BTTB can remain an optional diagnostic branch, but the current
evidence does not justify making it part of the core task model.

## Revised Research Decision

1. Promote the exact-recent plus low-dimensional long-term memory hybrid to the
   language-conditioned task MVP.
2. Keep recent window, instant-only, reservoir, adaptive slots, and memory-only
   Oja as matched-byte baselines.
3. Remove BCCB and sparse-event routing from the primary contribution.
4. Keep local BTTB as a negative/optional control until task-level motion slices
   demonstrate net quality and latency benefit.
5. Require delayed-query accuracy, current-scene accuracy, full retained-byte
   accounting, and measured P50/P95/P99 latency before making a systems claim.

## Limitations

- Categories are proxy strata selected from frame-change statistics, not
  semantic event annotations.
- The visual encoder is frozen and only one encoder has the full 30-segment
  replication; Qwen3-VL currently has a three-segment development probe.
- Queries are target hidden vectors, not language-conditioned questions.
- The stream contains 16 sampled frames, so long-horizon forgetting and archive
  behavior remain untested.
- FLOPs are accounting proxies; no optimized GPU kernel latency has been
  measured.

## Artifacts

- `clip_stratified_formal30_20260717/aggregate_summary.json`
- `clip_stratified_formal30_20260717/formal_probe_validation.json`
- `clip_stratified_formal30_20260717/formal_probe_decision_metrics.csv`
- `clip_stratified_formal30_20260717/clip_stratified_formal30_category_probe_summary.png`
- `clip_stratified_formal30_20260717/clip_stratified_formal30_category_probe_summary.pdf`
- `MEMORY_RETENTION_FORMAL_ANALYSIS.md`
