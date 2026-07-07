# Attention Pattern Mechanism Study

Date: 2026-07-07

This note extends `ANALYSIS_REPORT.md` from "can we replace attention with
circulant/block-structured attention?" to "why do sink/outlier/local/sparse
patterns appear, are they stable, and how should we test whether they are
inductive biases or model defects?"

## Current Answer

The evidence does not support a single universal pattern. The measured
attention behavior is a mixture of:

1. **sink/no-op behavior**: a few columns absorb attention mass when a head
   needs a small or partial residual update;
2. **internal scratch/register behavior**: in vision models, low-information
   or special tokens can become storage locations for internal computation;
3. **local/cyclic geometric bias**: especially plausible in video DiTs with
   latent `F x H x W` grids and 3D RoPE;
4. **dynamic sparse routing**: row-specific content routes that cannot be
   represented by one BCCB kernel or one fixed permutation;
5. **value-subspace effects**: an approximation can have poor matrix fidelity
   but low `A @ V` error for one head because `V` suppresses some differences.

Therefore the next replacement target should not be "all attention becomes
circulant". It should be a gated hybrid:

`sink/global + local-cyclic + sparse routing`

with a learned or calibrated router.

## Literature Mechanism Map

### Attention Sinks

StreamingLLM identifies "attention sink": retaining initial-token KV states can
recover sliding-window streaming performance, and the sink can be strong even
when the initial tokens are not semantically important. The paper also shows
that a dedicated placeholder sink token during pretraining can improve
streaming deployment.

Source: <https://arxiv.org/abs/2309.17453>

Mechanism hypothesis for our setting:

- Softmax rows must sum to one.
- If a head wants "no useful update", it still has to put probability mass
  somewhere.
- Initial/special/register/background tokens become stable mass reservoirs.
- These columns should be input-stable within a head and show low row-argmax
  diversity.

### Outliers from No-op / Partial-update Attention

`Quantizable Transformers: Removing Outliers by Helping Attention Heads Do
Nothing` links strong activation outliers to attention heads attempting no-op
or partial residual updates. The paper argues that pushing softmax inputs to
large values to obtain near-zero attention entries can create outliers, and
tests clipped softmax and gated attention as mitigations.

Source: <https://arxiv.org/abs/2306.12929>

Mechanism hypothesis:

- Outlier logits/activations are not arbitrary numerical accidents.
- They can be a learned workaround for a missing explicit "do nothing" route.
- If this is true, clipped softmax/gated attention/register tokens should
  reduce sink/outlier magnitude without hurting task loss after training or
  light calibration.

### Persistent Outlier Dimensions

`LLM.int8()` shows that large LMs have systematic emergent outlier feature
dimensions that dominate attention and predictive performance, motivating
mixed-precision handling of those dimensions.

Source: <https://arxiv.org/abs/2208.07339>

`BERT Busters` finds high-magnitude LayerNorm scaling/bias outliers that emerge
early and appear consistently in the same dimensional positions across layers;
removing them can significantly degrade loss and downstream performance.

Source: <https://arxiv.org/abs/2105.06990>

Mechanism hypothesis:

- Some outliers are functional circuits, not just defects.
- A pattern is more likely an inductive feature if it is stable across inputs,
  emerges early in training, and causal ablation hurts.
- A pattern is more likely an architectural defect if a cleaner interface
  (registers, gates, clipped softmax) removes it without loss.

### Vision Registers and Background Scratch Tokens

`Vision Transformers Need Registers` reports high-norm tokens in ViT feature
maps, often in low-informative background areas, repurposed for internal
computation. Adding extra register tokens can remove the artifacts and smooth
feature/attention maps.

Source: <https://arxiv.org/abs/2309.16588>

Mechanism hypothesis for ViT/video:

- Background or low-information patches can become implicit registers.
- Strong sink columns in visual attention may be a register/scratch-space
  phenomenon rather than semantic focus.
- Adding explicit registers should shift mass from background/sink patches to
  registers and improve spatial smoothness.

### Induction and Routing Circuits

Induction heads show that attention patterns can be learned algorithmic
circuits supporting in-context learning, not merely artifacts.

Source: <https://arxiv.org/abs/2209.11895>

Mechanism hypothesis:

- Dynamic sparse stripes/routes can be useful content-routing circuits.
- These should be input-dependent and head-specific.
- Causal ablation should hurt only when the relevant content relation is
  present, unlike pure sink/no-op routes.

### Structured Attention Families

Circulant Attention models ViT self-attention as a nearest BCCB matrix and uses
DFT-based computation for `O(N log N)` attention. Its repository frames the
pattern as an inherent efficient paradigm in ViTs.

Source: <https://github.com/LeapLabTHU/Circulant-Attention>

Adaptively Sparse Transformers replace softmax with entmax so heads can learn
context-dependent sparsity; different layers learn different sparsity
preferences.

Source: <https://arxiv.org/abs/1909.00015>

Mechanism hypothesis:

- BCCB/local-cyclic is one geometric component, not the whole story.
- Sparse patterns should be allowed to be context-dependent rather than fixed.
- A useful compression path should separate local/cyclic, sink/global, and
  sparse routes.

### 2025-2026 Mechanism Updates

Several newer papers sharpen the mechanism interpretation:

- `Attention Sinks: A 'Catch, Tag, Release' Mechanism for Embeddings` argues
  that sinks can implement useful residual-stream bookkeeping through a catch,
  tag, and release mechanism.
  Source: <https://arxiv.org/abs/2502.00919>
- `Attention Sinks Are Provably Necessary in Softmax Transformers` proves that
  softmax normalization can force a stable anchor for trigger/default behavior,
  while non-normalized ReLU attention can avoid the sink on the same task.
  Source: <https://arxiv.org/abs/2603.11487>
- `The Spike, the Sparse and the Sink` separates massive activations from
  attention sinks: massive activations behave more like global implicit
  parameters, while sinks modulate local head outputs; pre-norm helps them
  co-occur.
  Source: <https://arxiv.org/abs/2603.05498>
- `A Single Layer to Explain Them All` traces massive activations to a
  consistent emergence layer and reports that weakening the massive activation
  token can also mitigate attention sinks.
  Source: <https://arxiv.org/abs/2605.08504>
- `Attention Sinks and Outliers in Attention Residuals` treats sinks/outliers as
  a routing and quantization-stability issue in AttnResidual architectures,
  again supporting explicit null/sink handling rather than folding everything
  into one geometric kernel.
  Source: <https://arxiv.org/abs/2605.17887>
- `RoPeSLR` proposes 3D RoPE-driven sparse-low-rank attention for video DiTs and
  reports Wan2.1/HunyuanVideo acceleration. This is close to our video
  generation diagnosis: RoPE geometry supplies distance structure, but the
  useful approximation is sparse plus low-rank rather than pure BCCB.
  Source: <https://arxiv.org/abs/2605.20659>

Implication for this project:

- A sink can be a functional circuit, a softmax-normalization workaround, or a
  model-specific artifact. Stability alone is not enough; ablation and
  cleaner-interface tests are needed.
- For video DiTs, the current literature is converging toward hybrid sparse /
  low-rank / geometry-aware attention. That is more compatible with our Wan
  evidence than a universal circulant-attention replacement.

## Existing Experimental Evidence

Generated by:

- `scripts/pattern_stability_probe.py`
- `remote_logs/pattern_stability_probe_20260707.json`
- `remote_logs/pattern_stability_probe_20260707.csv`
- `figures/fig11_attention_pattern_stability.png`
- `scripts/attention_pattern_full_probe.py`
- `remote_logs/attention_pattern_full_vit_20260707.json/csv`
- `remote_logs/attention_pattern_full_qwen_20260707.json/csv`
- `figures/fig12_attention_pattern_full_sweep.png`

### Full-sweep Method Stability

From 96 Qwen3-VL visual maps and 96 ViT maps in the existing 210
attention-space probe:

| Family | Method | Mean matrix error | Winner rate | Mean `A @ V` error | Fraction `A @ V` error < 0.2 |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL visual | grid BCCB | 0.882 | 0.000 | 0.765 | 0.000 |
| Qwen3-VL visual | flat BCM | 0.734 | 0.240 | 0.594 | 0.000 |
| Qwen3-VL visual | fixed-perm BCM | 0.714 | 0.083 | 0.575 | 0.000 |
| Qwen3-VL visual | Monarch-like proxy | 0.521 | 0.677 | 0.470 | 0.208 |
| ViT | grid BCCB | 0.926 | 0.000 | 0.717 | 0.000 |
| ViT | flat BCM | 0.766 | 0.052 | 0.459 | 0.094 |
| ViT | fixed-perm BCM | 0.743 | 0.240 | 0.440 | 0.094 |
| ViT | Monarch-like proxy | 0.619 | 0.708 | 0.267 | 0.438 |

Interpretation:

- Grid-BCCB is not a stable replacement pattern in these ViT/Qwen samples.
- Proxy masks often win matrix error among the tested fixed families, but even
  proxy output error is usually too large for zero-shot replacement.
- The high proxy winner rate supports "sinks/sparse masks matter", not "fixed
  Monarch replacement is solved".

### Layer Stability

Qwen grid-BCCB mean error by layer:

- layer 0: `0.848`
- layer 8: `0.785`
- layer 16: `0.949`
- layer 26: `0.944`

ViT grid-BCCB mean error by layer:

- layer 0: `0.979`
- layer 1: `0.921`
- layer 2: `0.914`
- layer 5: `0.889`

No sampled Qwen/ViT layer has `grid_error < 0.6` on any map in this sweep.
Layer 8 in Qwen is less bad, consistent with a local/cyclic component, but the
effect is not strong enough to be a standalone BCCB mechanism.

### Wan2.2 3D Cyclic Component

Wan direct Q/K probe:

| Branch | Records | Mean attention R2 | Std R2 | Fraction R2 >= 0.7 | Fraction R2 < 0.2 |
|---|---:|---:|---:|---:|---:|
| high noise | 16 | 0.608 | 0.187 | 0.312 | 0.062 |
| low noise | 8 | 0.617 | 0.276 | 0.375 | 0.125 |

Common high/low layer-head pairs:

- paired layer-heads: `8`
- high/low R2 correlation: `0.711`
- mean absolute R2 delta: `0.152`
- same-side-of-0.7 rate: `0.625`

Interpretation:

- Wan has a real 3D cyclic/local component and it is partially stable across
  high/low noise.
- It is not constant: strength varies by layer/head/timestep.
- This supports a gated head/layer/timestep policy, not unconditional
  circulant replacement.

### Representative Matrix Mechanisms

| Example | Top-2 column mass | Row-argmax unique fraction | Radius-1 local mass | Row top-4 mass | Effective-rank fraction | Proxy `A @ V` | Hybrid `A @ V` |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViT L0 H0 | 0.881 | 0.031 | 0.187 | 0.947 | 0.047 | 0.689 | 0.032 |
| ViT L5 H0 | 0.183 | 0.047 | 0.139 | 0.348 | 0.045 | 0.038 | 0.052 |
| Qwen L8 H0 F1 | 0.117 | 0.500 | 0.286 | 0.363 | 0.526 | 0.177 | 0.159 |
| Qwen L26 H0 F0 | 0.315 | 0.143 | 0.144 | 0.620 | 0.167 | 0.508 | 0.073 |

Interpretation:

- **ViT L0 H0** is a strong sink/no-op case: two columns carry 88.1% of mass,
  row argmax diversity is only 3.1%, and effective rank is low.
- **ViT L5 H0** is low-rank/global and value-subspace dominated: proxy has poor
  matrix fidelity but low output error, so `A @ V` alone is insufficient.
- **Qwen L8 H0 F1** is the most dynamic/local case: high argmax diversity,
  high effective rank, and local radius-1 mass 28.6%.
- **Qwen L26 H0 F0** is sparse/sink late-layer routing: top-2 columns carry
  31.5%, row top-4 mass is 62.0%, and hybrid greatly improves output error.

## Mechanism Hypotheses to Test

### H1: Softmax No-op Sink

If a head wants no update or a partial residual update, softmax still requires
probability mass. It pushes mass into stable columns and may create large
logits/activation outliers.

Predictions:

- top-k column mass is stable across inputs for the same head;
- row argmax collapses to a few keys;
- output norm from that head is small or aligned with residual cancellation;
- clipped softmax/gated attention/register tokens reduce the sink without loss
  after training or light calibration.

### H2: Register/Scratch Tokens

Vision/video models may repurpose background, special, or early tokens as
internal scratch space.

Predictions:

- high sink mass occurs in low-information patches or special tokens;
- adding explicit register tokens shifts sink mass to registers;
- feature/attention maps become smoother after adding registers;
- task metrics do not degrade and may improve for dense prediction.

### H3: Geometric Local/Cyclic Bias

For grid-like video latents, RoPE and denoising objectives can induce
translation/offset-like attention components.

Predictions:

- BCCB/local-cyclic R2 is higher in video DiT latent attention than in Qwen
  visual per-frame attention;
- R2 is partly stable by layer/head but changes with noise timestep;
- high-R2 heads tolerate circulant/local replacement better than low-R2 heads;
- offset kernels align with temporal/spatial RoPE axes.

### H4: Dynamic Sparse Semantic Routing

Some stripes and off-diagonal routes are content-dependent circuits.

Predictions:

- row argmax diversity and effective rank are high;
- top routes change with video content/prompt;
- fixed masks underfit, but dynamic top-k or learned routing works;
- causal ablation hurts only for inputs requiring the routed relation.

### H5: Value-subspace Degeneracy

Low `A @ V` error does not always imply a faithful attention approximation.

Predictions:

- replacing `V` with random, permuted, or orthogonalized values breaks some
  low-output-error approximations;
- matrix error and output error diverge most in low effective-rank value
  subspaces;
- task-level evaluation is needed before claiming deployability.

## Proposed Experiments

### Experiment 1: Stability Grid Across Inputs, Layers, Heads, Timesteps

Metrics:

- top-k column mass and column-mass Gini;
- row-argmax unique fraction;
- entropy and effective rank;
- local radius-1/2 mass;
- cyclic/BCCB R2 and relative error;
- sparse top-k row mass;
- `A @ V` error under several approximations.

Runs:

- Qwen3-VL: more Video-MME clips, all sampled visual layers/heads, temporal
  slices.
- Wan2.2: multiple prompts, timesteps, layers, heads, and both high/low noise
  branches.
- ViT: standard DeiT/ViT and current SCTM checkpoint for comparison.

Decision rule:

- stable sink if top-k columns recur across inputs with high overlap;
- stage-specific cyclic if R2 clusters by layer/head/timestep;
- dynamic routing if row-argmax routes vary with input and semantic content.

### Experiment 2: Causal Interventions

Interventions:

- zero or mask top sink columns;
- keep only sink columns;
- add register tokens and see whether sink mass moves;
- clip softmax or gate attention;
- replace `V` with random/permuted/orthogonalized values;
- remove or perturb RoPE dimensions/axes in Wan probe.

Readouts:

- task loss or video quality proxy where available;
- head output norm and residual update norm;
- changes in sink/local/sparse metrics;
- whether cyclic R2 survives RoPE perturbation.

Decision rule:

- if removing a pattern hurts, it is likely functional;
- if adding registers/gates removes it without loss, it is likely an
  architectural workaround;
- if a pattern is only visible in `A` but not in `A @ V` or task metrics, it may
  be value-subspace redundant.

### Experiment 3: Training-time Emergence

Use checkpoints across training if available.

Track:

- emergence time of sink columns and outlier dimensions;
- whether LayerNorm/activation outliers precede attention sinks or follow them;
- whether cyclic/local R2 increases with training;
- whether dynamic sparse routes appear after semantic competence improves.

Decision rule:

- early stable outliers suggest architectural/optimization workaround;
- late task-correlated routes suggest learned circuits;
- geometry-aligned cyclic components suggest inductive bias.

## Immediate Next Engineering Steps

1. Add full-sweep sink/local/sparse metrics directly to
   `structured_attention_probe.py`, not only representative NPZ examples.
2. Extend `wan_bccb_activation_probe.py` to export sink/top-k/local metrics
   alongside 3D BCCB R2.
3. Add a `V`-subspace stress test: original `V`, random `V`, permuted `V`,
   whitened `V`.
4. Run a small register-token intervention on ViT if a trainable/evaluable ViT
   path is available.
5. Push this study update to the GitHub package after the next remote probe.

## Status Before Full-sweep Extension

This note and Fig.11 are an evidence-backed first pass over existing logs. They
were extended with a full-sweep ViT/Qwen pattern probe on 2026-07-07. The study
still does not complete the full objective because we need causal interventions
and broader Wan/Qwen sweeps to distinguish functional inductive bias from
architectural workaround with high confidence.

## Full-sweep ViT/Qwen Pattern Probe

Script:

- `scripts/attention_pattern_full_probe.py`

Outputs:

- `remote_logs/attention_pattern_full_vit_20260707.json/csv`
- `remote_logs/attention_pattern_full_qwen_20260707.json/csv`
- `figures/fig12_attention_pattern_full_sweep.png/pdf`

Scope:

- ViT: `192` maps = 8 samples x 4 layers (`0,1,2,5`) x 6 heads.
- Qwen3-VL visual: `96` maps = 3 videos x 2 temporal slices x 4 layers
  (`0,8,16,26`) x 4 heads.

Additional metrics:

- top-2/top-4 column mass for sink strength;
- row-argmax unique fraction for collapse vs dynamic routing;
- local radius-1/2 mass;
- row top-4 mass for sparse routing;
- effective rank;
- oracle union mask = sink-2 columns + local radius-1 + row top-4;
- union-mask `A @ V` error under true `V` and deterministic random `V`.

### Aggregate Results

| Family | Maps | Top-2 col mass | Argmax unique | Local r1 mass | Row top-4 mass | Eff-rank frac | Union `A@V` err | Union random-V err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT | 192 | 0.448 | 0.081 | 0.140 | 0.691 | 0.073 | 0.093 | 0.420 |
| Qwen3-VL visual | 96 | 0.204 | 0.210 | 0.141 | 0.440 | 0.265 | 0.600 | 1.133 |

Interpretation:

- ViT attention in this checkpoint is much more collapsed/sink-heavy and
  low-rank: high top-2 column mass, low row-argmax diversity, low effective
  rank.
- Qwen visual attention is less collapsed and more dynamic: higher argmax
  diversity and much higher effective-rank fraction.
- The union oracle mask works much better for ViT than Qwen. This implies that
  ViT maps are dominated by sink/top-k structures, while Qwen needs additional
  content-dependent routing beyond sink/local/top-k masks.
- Random-V stress strongly increases error, especially for Qwen. This confirms
  that `A @ V` success is value-subspace dependent and cannot be read as
  faithful attention replacement by itself.

### Layer-wise Results

ViT:

| Layer | Top-2 col mass | Argmax unique | Local r1 mass | Row top-4 mass | Union true-V err | Union random-V err |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.643 | 0.092 | 0.143 | 0.862 | 0.134 | 0.150 |
| 1 | 0.456 | 0.075 | 0.131 | 0.749 | 0.071 | 0.337 |
| 2 | 0.360 | 0.080 | 0.138 | 0.585 | 0.085 | 0.581 |
| 5 | 0.333 | 0.079 | 0.148 | 0.570 | 0.084 | 0.612 |

Qwen3-VL visual:

| Layer | Top-2 col mass | Argmax unique | Local r1 mass | Row top-4 mass | Union true-V err | Union random-V err |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.131 | 0.116 | 0.107 | 0.240 | 0.886 | 1.796 |
| 8 | 0.109 | 0.386 | 0.162 | 0.403 | 0.672 | 1.386 |
| 16 | 0.343 | 0.158 | 0.149 | 0.589 | 0.432 | 0.600 |
| 26 | 0.233 | 0.181 | 0.146 | 0.529 | 0.409 | 0.751 |

Interpretation:

- ViT sink strength is strongest in layer 0 and decreases with depth, but
  row-argmax diversity remains low. This is consistent with a no-op/scratch
  mechanism rather than semantic dynamic routing.
- Qwen layer 8 is the clearest dynamic-routing layer: argmax unique fraction
  reaches `0.386`, while sink mass is low (`0.109`) and local mass is relatively
  higher (`0.162`).
- Qwen layers 16/26 become more sparse/sink-like than layer 8, consistent with
  late-layer consolidation or routing into a smaller set of visual keys.
- The true-V vs random-V gap is large. For Qwen layer 0, union true-V error is
  `0.886` and random-V error is `1.796`; for ViT layer 5, true-V is `0.084`
  but random-V is `0.612`. This directly supports H5: value-subspace
  degeneracy is real.

### Sink Stability Across Inputs

The full probe also computes pairwise Jaccard overlap of top-2 sink columns
within each layer/head group.

Examples:

- ViT groups have low top-2 sink Jaccard in this sampled set; the highest shown
  group is only about `0.083`. This means ViT has strong sink mass but not
  necessarily the same sink columns across all images in the attention-only
  rollout.
- Qwen groups are more stable in some heads: examples include layer 0 head 0
  Jaccard `0.422`, layer 0 head 1 `0.356`, and layer 16 head 0 `0.333`.

Interpretation:

- "Sink" is not synonymous with one universal fixed token. Some heads have
  stable sink locations; others have high sink mass that follows input/grid
  conditions.
- This makes a pure fixed-sink cache policy risky for vision/video. A deployable
  router should allow head/layer-specific sink prediction rather than assuming
  the first or same token is always the sink.

### Updated Mechanism Judgment

The current evidence separates the patterns:

- **ViT/SCTM current checkpoint**: dominated by sink/sparse low-rank behavior;
  likely a no-op/scratch or model-specific routing workaround. The very low
  effective-rank fraction and low argmax diversity argue against rich semantic
  routing in the sampled maps.
- **Qwen3-VL visual layer 8**: dynamic/local routing is stronger. This looks
  more like an input-dependent circuit or visual grouping behavior than a pure
  sink artifact.
- **Qwen3-VL visual late layers**: sparse/sink behavior increases. This may be
  consolidation into salient visual keys, but causal ablation is needed.
- **Wan2.2**: remains the stronger candidate for geometry-induced
  local/cyclic bias because 3D BCCB R2 is much higher than Qwen/ViT, and
  high-low timestep R2 is partially correlated.

The practical compression implication is unchanged but sharper:

- BCCB/circulant should be gated to heads/layers/timesteps with measured
  cyclic structure.
- Sink/global paths must be explicit rather than folded into a cyclic kernel.
- Sparse routing must be dynamic for Qwen/video understanding.
- `A @ V` metrics must be reported with random/permuted/whitened-V stress tests
  before claiming a replacement is faithful.

## Current Status

This study now has:

- literature-backed mechanism hypotheses;
- representative-matrix diagnostics;
- full-sweep ViT/Qwen sink/local/sparse/value-subspace metrics;
- Wan high/low timestep cyclic-stability evidence;
- concrete intervention experiment designs;
- a first matrix-level component-intervention probe over the saved hybrid
  decomposition.

## Matrix-Level Component Intervention

Script:

- `scripts/attention_component_intervention_probe.py`

Outputs:

- `remote_logs/attention_component_intervention_20260707.json/csv`
- `figures/fig13_attention_component_intervention.png/pdf`

Method:

- Load the saved representative oracle decomposition from
  `hybrid_attention_decomposition_20260704`.
- Compare full hybrid against ablations:
  `no_sink_global`, `no_local_cyclic`, and `no_sparse_routing`.
- Also test `only_sink_global`, `only_local_cyclic`, and
  `only_sparse_routing`.
- All variants are row-normalized nonnegative matrices and are compared to the
  original dense attention matrix by relative Frobenius error.

Aggregate over the four representative matrices:

| Variant | Mean matrix error |
|---|---:|
| Grid BCCB | 0.876 |
| Monarch-like proxy | 0.743 |
| Full hybrid | 0.147 |
| No sink/global | 1.233 |
| No local-cyclic | 0.202 |
| No sparse-routing | 0.236 |
| Only sink/global | 0.300 |
| Only local-cyclic | 1.385 |
| Only sparse-routing | 2.078 |

Representative interpretation:

- ViT L0 H0 and ViT L5 H0 are overwhelmingly sink/global dominated. Removing
  sink/global raises error from `0.038` to `1.079` and from `0.134` to `1.458`.
- Qwen L8 H0 F1 is more mixed: removing sink/global is still worst
  (`0.291` to `1.297`), but removing local-cyclic (`0.488`) or sparse-routing
  (`0.456`) also hurts. This matches the dynamic/local-routing interpretation.
- Qwen L26 H0 F0 is sink/global plus sparse: removing sparse-routing raises
  error from `0.125` to `0.303`, larger than removing local-cyclic (`0.154`).

Why current replacement results are poor:

- Grid BCCB and flat BCM do not include an explicit sink/global low-rank path.
- Fixed permutations do not capture row-specific sparse routes.
- The Monarch-like proxy can look better on some maps because it keeps observed
  entries through a mask, but it is still not a learned router and fails on
  low-rank/global maps such as ViT L5 H0.
- The hybrid diagnostic works because it is oracle: sink columns and sparse
  routes are selected from known dense `A`. This explains the failure modes but
  does not yet provide a deployable kernel.

Remaining gap:

This still does not prove the final mechanistic distinction between functional
inductive bias and architectural workaround. That requires task-level causal
interventions: sink masking during real forward/loss evaluation,
register/gate/clipped-softmax variants, RoPE-axis perturbation for Wan, and
`V` subspace stress tests beyond random `V`.
