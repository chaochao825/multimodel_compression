# Qwen3-VL Preliminary Probe Analysis

Date: 2026-07-17

## Evidence scope

This is a development-set diagnostic, not a final hypothesis verdict.

- Model: Qwen3-VL-30B-A3B-FP8 visual tower.
- Layers: 0, 8, 16, and 26, measured before the visual merger.
- Data: three Video-MME clips, one 16-frame segment per clip.
- Hidden grid: 8 temporal positions by 14 by 14 spatial tokens.
- Statistics: centered and token-normalized spectra, causal history-only
  subspace projection, pair-oracle versus causal transport, and residual-block
  concentration.
- Missing evidence: a second encoder, stratified motion/event labels, optical
  flow, delayed-query quality, and matched-byte online-memory baselines.

## Probe 1: low-dimensional persistent state

Rank-32 explained energy averaged across the three clips:

| Layer | Current-frame spatial | Centered history feature | Token-normalized history |
|---:|---:|---:|---:|
| 0 | 82.5% | 81.5% | 81.9% |
| 8 | 68.4% | 66.0% | 65.3% |
| 16 | 78.3% | 75.2% | 70.5% |
| 26 | 99.8% | 99.7% | 98.8% |

History-centered causal projection error:

| Layer | Rank 32 | Rank 64 |
|---:|---:|---:|
| 0 | 43.2% | 28.5% |
| 8 | 59.1% | 46.4% |
| 16 | 51.2% | 40.7% |
| 26 | 5.3% | 4.2% |

Interpretation:

- The low-dimensional-state hypothesis is strongly layer dependent.
- Rank 32 clears the 70% energy threshold at layers 0, 16, and 26, but misses
  it at layer 8.
- Energy captured by an offline spectrum does not imply causal preservation.
  Early and middle layers still have 43-59% centered causal projection error at
  rank 32.
- Layer 26 has an effective rank close to one across all three clips. This may
  be a genuine common-direction collapse, a visual-tower normalization effect,
  or an extraction-location artifact. It must not be used as positive evidence
  without an extraction-point control.

The follow-up hidden-state diagnostic confirms that this is not an aggregation
or plotting error:

| Run | Mean-direction energy | Centered top-1 energy | Centered effective rank | Mean token-pair cosine |
|---|---:|---:|---:|---:|
| fFYN_s020 | 67.3% | 99.0% | 1.10 | 0.939 |
| VP4_s045 | 73.0% | 99.4% | 1.07 | 0.948 |
| HPro_s070 | 72.6% | 99.2% | 1.08 | 0.947 |

Even after removing each frame's mean feature vector, nearly all spatial
variation remains on one direction. This makes layer 26 a poor place to claim
general-purpose compact memory: the representation itself has already
collapsed before any proposed state compressor is applied.

Preliminary status: **mixed, not passed**. The preregistered gate requires two
model/data domains and either causal task preservation or matched-byte
retention superiority.

## Probe 2: sparse event innovation

For identity residuals, the top 10% spatial blocks capture:

| Layer | Residual energy | Pixel-change proxy recall |
|---:|---:|---:|
| 0 | 40.5% | 37.3% |
| 8 | 33.2% | 35.5% |
| 16 | 39.5% | 31.1% |
| 26 | 40.6% | 14.1% |

Local BTTB and BCCB predictors do not make the residual substantially more
concentrated; their top-10% energy remains about 31.7-39.1%.

Interpretation:

- The current residuals are not sparse enough for the preregistered 70%
  top-10% energy gate.
- Pixel change is only a weak event proxy. The result cannot establish semantic
  event recall without annotated events, scene cuts, and object-level labels.
- Concentration varies materially by clip, so aggregate averages hide distinct
  motion and editing regimes.

Preliminary status: **negative on concentration, incomplete on semantics**.

## Probe 3: structured spatial transport

Stable-segment centered prediction error:

| Layer | Identity | Local BTTB causal | Local BCCB causal | BCCB improvement |
|---:|---:|---:|---:|---:|
| 0 | 0.2237 | 0.2248 | 0.2245 | -0.4% |
| 8 | 0.3386 | 0.3392 | 0.3391 | -0.1% |
| 16 | 0.4424 | 0.4342 | 0.4333 | +2.1% |
| 26 | 0.7211 | 0.6814 | 0.6785 | +5.9% |

At layer 26, a pair-fitted global BCCB reaches 0.1276 error, while its causal
transferred version reaches 1.6253. This large oracle-to-causal gap is direct
evidence that pairwise structured fitting can overstate deployable transport
quality.

Interpretation:

- Local BCCB does not reach the preregistered 10% improvement gate.
- BCCB is almost indistinguishable from the zero-padded BTTB control. The
  current evidence therefore supports local convolutional transport, not a
  specifically cyclic mechanism.
- Estimated integer shifts are mostly zero in these segments, so the current
  sample is not a strong translational-motion stress test.

Preliminary status: **gate failed on the three development segments**.

## Current research decision

The evidence supports retaining the overall hybrid-memory question, but not
promoting BCCB as the main mechanism yet.

1. Keep a low-rank state path as a layer-selective candidate, not a universal
   fixed-rank assumption.
2. Treat event routing as unproven until semantic annotations and stronger
   residual predictors are added.
3. Keep BCCB only as an optional transport basis and require it to beat BTTB,
   optical flow, and identity under causal transfer and measured cost.
4. Preserve pair-oracle results only as an upper-bound diagnostic.

## Next required experiments

1. Repeat the same extraction on the LLaVA CLIP vision tower.
2. Expand to 16-32 clips stratified by camera motion, object motion, scene cuts,
   static dialogue, and dense events.
3. Add optical-flow alignment and object/region pooling.
4. Add recent-window, reservoir, online PCA, and fixed-byte state baselines.
5. Validate layer-26 collapse with token variance, cosine similarity, feature
   norm, and pre/post-normalization extraction checks.
6. Replace pixel-change recall with annotated key-event and scene-cut recall.

## Reproducible artifacts

- `qwen_dev_v2_download/aggregate/rank_summary.csv`
- `qwen_dev_v2_download/aggregate/transport_summary.csv`
- `qwen_dev_v2_download/aggregate/causal_subspace_summary.csv`
- `qwen_dev_v2_download/aggregate/residual_summary.csv`
- `qwen_dev_v2_download/aggregate/qwen_state_spectrum.png`
- `qwen_dev_v2_download/aggregate/qwen_transport_error.png`
- `qwen_dev_v2_download/aggregate/qwen_causal_subspace.png`
- `qwen_dev_v2_download/aggregate/qwen_residual_concentration.png`
- `qwen_dev_v2_download/aggregate/qwen_hidden_diagnostics.csv`
- `qwen_dev_v2_download/aggregate/qwen_hidden_diagnostics.json`
- `qwen_dev_v2_download/aggregate/qwen_layer_collapse_diagnostics.png`
