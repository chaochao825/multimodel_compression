# Probe MVP Contract

Frozen on 2026-07-16 after the expanded literature and remote-asset audit.

## Scope

The MVP tests latent dynamics before implementing an end-to-end memory
architecture. Existing per-frame attention BCCB fits are retained only as a
negative spatial-attention boundary.

## Models

1. Primary: Qwen3-VL-30B-A3B-FP8 visual tower on server 210.
2. Cross-encoder: LLaVA-1.5-7B CLIP visual tower on server 210.
3. Deferred: Cambrian-S-7B-LFP if its interface and storage overhead are
   manageable.

The language model remains frozen and is not required for the first activation
probes.

## Data

1. Controlled synthetic sequences with known translation, low-dimensional
   persistent state, sparse events, and scene cuts.
2. Three already verified Video-MME development clips.
3. A stratified 16-32 clip Video-MME probe set.
4. StreamingBench and OVO-Bench after source, license, and split resolution.

## Probe 1: Low-Dimensional Persistent State

Report:

- per-frame spatial stable rank and effective rank;
- history common-feature-subspace energy at ranks 8, 16, 32, and 64;
- temporal-vector energy before and after motion alignment;
- causal projection error using only previous observations;
- current-scene and delayed-query preservation at matched state bytes.

Controls:

- no alignment;
- identity alignment;
- global integer/phase-correlation alignment;
- optical-flow alignment when available;
- region or object pooling;
- truncated SVD;
- online PCA/subspace tracking;
- fixed recent window and uniform reservoir.

MVP gate: on at least two model/data domains, rank at most 32 either explains at
least 70% of the preregistered temporal energy or beats matched-byte
recent-window/reservoir delayed retention without reducing current-scene quality.

## Probe 2: Sparse Event Innovation

Residuals are measured after identity, motion, convolution/BTTB, local BCCB,
and memory predictors. Report top-5/10/20-percent residual energy, block Gini,
event recall, scene-cut stratification, and oracle-versus-causal router gap.

MVP gate: top 10% blocks capture at least 70% of residual energy and at least
80% of annotated key-event blocks on two real data domains.

## Probe 3: Structured Spatial Transport

Compare:

- identity/cache;
- global integer shift or optical-flow warp;
- depthwise local convolution/BTTB;
- global BCCB;
- masked/local BCCB;
- mixture of local structured bases;
- low-rank token map;
- prior-transition kernel transfer.

Cyclic methods must report a zero-padded boundary control. Pair-fitted oracle
operators and causal transferred operators are reported separately.

MVP gate: local BTTB/BCCB reduces stable-segment activation prediction error by
at least 10% versus identity and has lower measured cost than dense mixing. If
BCCB does not beat convolution/BTTB or receives negligible gate allocation, it
is removed from the title and main contribution.

## Fairness and Systems Contract

- Fix total retained GPU, CPU, disk, index, and archive bytes.
- Fix average per-frame compute rather than peak route capacity.
- Keep a recent-window/instant-cache baseline in every comparison.
- Count encoder, prefill, router, retrieval, archive, synchronization, and
  structured-kernel latency.
- Report P50, P95, and P99 at batch size one after synchronized warmup.
- Preserve all failed gates as negative results.
