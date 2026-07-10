# Video Circulant-Attention Probe Report

## Question

Reference: `LeapLabTHU/Circulant-Attention`.

The reference repository argues that ViT self-attention maps often approximate
Block Circulant matrices with Circulant Blocks (BCCB), enabling a DFT-based
attention replacement. For video models, the analogous measurable question is:

- image/vision tokens: does each 2D frame attention map approximate a cyclic
  shift-invariant 2D kernel?
- video latent tokens: does attention over a `T x H x W` latent grid approximate
  a 3D cyclic shift-invariant kernel?

## Metric

For an attention matrix `A` over a grid, the nearest cyclic-shift matrix under
Frobenius norm is obtained by averaging `A[i, j]` over equal cyclic offsets
`coord(j) - coord(i) mod grid_shape`.

Reported metrics:

- `relative_fro_error = ||A - P_circ(A)||_F / ||A||_F`; lower is more circulant.
- `circulant_r2 = 1 - SSE/SST`; `1.0` is a perfect cyclic/BCCB fit.

Implementation:

- Local scripts: `E:\Codex_work\ssh_experiment\video_circulant_analysis_20260701\scripts`
- Key metric script: `circulant_metrics.py`
- Qwen3-VL visual probe: `qwen3vl_visual_circulant_probe.py`
- Wan2.2 direct Q/K probe: `wan_bccb_activation_probe.py`
- Probe result JSONs include command-line arguments, package versions, script
  hash, model-index hash, video hash, device request, and the explicit
  token-order assumption.

## Assets Found

### Video Understanding

Server: `210` / `172.25.5.210`

Model: `/home/wangmeiqi/dqy/Qwen3-VL-30B-A3B-FP8`

Useful facts from `config.json` and safetensor index:

- architecture: `Qwen3VLMoeForConditionalGeneration`
- vision depth: 27 blocks
- vision hidden size: 1152
- vision heads: 16
- patch size: 16
- temporal patch size: 2
- visual `attn.qkv.weight` shape: `[3456, 1152]`
- visual weights: 351 keys, mostly in shards 3 and 4

Environment used:

- conda env: `qwen_omni`
- torch: `2.2.2+cu121`
- transformers: `4.57.3`
- safetensors: `0.6.2`
- GPU: A800 80GB

Input samples:

- `/home/wangmeiqi/.cache/huggingface/videomme/data/fFYNmVb3NCQ.mp4`
- `/home/wangmeiqi/.cache/huggingface/videomme/data/VP4GtrEsefk.mp4`
- `/home/wangmeiqi/.cache/huggingface/videomme/data/HProiNnmGwI.mp4`

Important structure finding:

Qwen3-VL visual attention forms `cu_seqlens` by repeating `H*W` for each
temporal patch. Therefore the visual encoder attention is per-temporal-slice
2D spatial attention, not one global `T x H x W` video attention. The probe
therefore measures per-frame 2D BCCB fit only. It does not test 3D video BCCB
or cross-frame attention in Qwen3-VL because that attention path is not present
in the visual tower.

### Video Generation

Servers: `435` / `172.25.4.35`, `434` / `172.25.4.34`

Model/code assets:

- `/data6/user24215463/Wan2.2/Wan2.2`
- `/data6/user24215463/Wan2.2/Wan2.2_SA3_fullrepo_fp16_edges_sagetriton_v0`
- `/data6/zmf/Wan2.2-T2V`
- `/data6/zmf/Wan2.2_SA3_hif4_edges8_fp16_noscale_norowmax_baseline_para01`

Wan2.2 T2V A14B facts from source/config:

- transformer dim: 5120
- heads: 40
- layers: 40
- patch size: `(1, 2, 2)`
- VAE stride: `(4, 8, 8)`
- window size: `(-1, -1)`, i.e. global self-attention
- self-attention uses `grid_sizes = (F, H, W)` and 3D RoPE split over time,
  height, and width.
- For the historical 832x480, 81-frame run, rowmax logs show
  `query_len = key_len = 32760`, consistent with a 3D latent grid.

Existing Wan attention-stat artifact:

- `rowmax_scalar_presets_stable_heads_plus_partial_offset_m10.txt`
- header reports `new_preset_slots: 241917 / total_slots: 256000`
- coverage: `94.498828%`
- historical generation run completed on 2026-06-12 for `t2v-A14B`, 832x480,
  81 frames, 40 denoising steps. The log shows about 24 minutes for generation.

This rowmax artifact supports strong QK-logit distribution regularity, but it
does not by itself measure BCCB or cyclic relative-offset structure.

## Qwen3-VL Results

Aggregate over 3 Video-MME clips, 4 selected layers (`0, 8, 16, 26`), first 4
or 8 heads per layer, and two temporal slices per video:

| Scope | Maps | Mean R2 | Mean relative error |
|---|---:|---:|---:|
| all Qwen3-VL visual maps | 160 | 0.0814 | 0.8842 |
| layer 0 | 40 | 0.0527 | 0.8280 |
| layer 8 | 40 | 0.2086 | 0.8008 |
| layer 16 | 40 | 0.0298 | 0.9573 |
| layer 26 | 40 | 0.0346 | 0.9509 |

Per-run summaries:

| Video/size | Grid | Maps | Mean R2 | Mean relative error |
|---|---|---:|---:|---:|
| `fFYNmVb3NCQ.mp4`, 128 | `[2, 8, 8]` | 32 | 0.0738 | 0.8632 |
| `VP4GtrEsefk.mp4`, 224 | `[2, 14, 14]` | 64 | 0.0875 | 0.8844 |
| `HProiNnmGwI.mp4`, 224 | `[2, 14, 14]` | 64 | 0.0791 | 0.8945 |

Interpretation:

- There is no strong CA-like/BCCB phenomenon in the measured Qwen3-VL visual
  attention maps.
- Layer 8 shows a weak spatial cyclic-offset component, but the fit is still
  poor in absolute terms.
- Later visual layers are especially non-circulant under this metric.
- Because Qwen3-VL visual attention is split per temporal slice, this experiment
  does not support a 3D video-circulant conclusion for the understanding model.
- The probe manually patchifies frames in temporal-major, raster H/W order.
  Qwen3-VL's visual forward path does not reorder tokens before `attn.qkv`, but
  this remains a processor-order assumption rather than a byte-for-byte
  validation against the official processor output.

## Visualization Figures

Generated figures live under
`E:\Codex_work\ssh_experiment\video_circulant_analysis_20260701\figures`.
Each figure is saved as both PNG and PDF:

- `fig1_qwen_layer_cyclic_fit`: layer-wise cyclic/BCCB R2 and relative error.
- `fig2_qwen_head_heatmaps`: layer/head heatmaps for R2 and relative error.
- `fig3_qwen_attention_bccb_examples`: representative attention maps, nearest
  BCCB projections, residuals, and estimated cyclic kernels.
- `fig4_wan_rowmax_context`: Wan2.2 rowmax preset coverage context, explicitly
  framed as QK-scale regularity rather than BCCB proof.
- `fig5_wan_direct_bccb_probe`: direct Wan2.2 self-attention Q/K cyclic fit
  over a `2 x 30 x 52` latent token grid.
- `fig6_structured_matrix_fit`: 210-side projection-weight structure fit for
  block-circulant, fixed-permutation block-circulant, and a Monarch-like proxy.
- `fig7_structured_attention_replacement`: 210-side attention-space replacement
  probe for grid-BCCB, flat block-circulant, fixed-permutation block-circulant,
  and a Monarch-like mask proxy.
- `fig8_attention_matrix_failure_modes`: representative attention matrices,
  structured approximations, residuals, and row-argmax diagnostics explaining
  why the zero-shot structured replacements fail.
- `fig9_hybrid_attention_decomposition`: oracle sink/global-SVD +
  local-cyclic + sparse-routing decomposition of the same representative
  attention matrices.
- `fig10_hybrid_attention_tradeoff`: representative matrix-error comparison
  between the previous structured baselines and the hybrid oracle diagnostic.
- `fig11_attention_pattern_stability`: method/layer/head stability summaries.
- `fig12_attention_pattern_full_sweep`: full ViT/Qwen sink/local/sparse/value
  metrics.
- `fig13_attention_component_intervention`: matrix-level hybrid component
  ablation.
- `fig14_value_subspace_stress`: true/permuted/orthogonalized/random `V`
  stress.
- `fig15_head_output_intervention`: head-output keep/drop intervention.
- `fig16_wan_delta_perturbation`: Wan coordinate perturbation over a bounded
  small-grid latent.
- `fig17_hybrid_transfer_probe`: source-to-target transfer test for saved
  hybrid supports/templates.
- `fig18_wan_noise_branch_stability`: high/low noise branch stability of Wan
  3D cyclic R2 on overlapping layer/head records.
- `fig19_sink_noop_correlation`: sink-strength correlations with entropy,
  drop-sink intervention, and value-subspace stress.

## Wan2.2 Generation-Side Assessment

Evidence strength: direct selected-layer Q/K probe, but with synthetic latent
and text context rather than a full saved denoising trajectory.

Probe setup:

- Server/profile: 34 / `434`.
- Python: `/data2/wangmeiqi/anaconda3/envs/cambrians/bin/python`.
- Checkpoint: `/data6/zmf/Wan2.2-T2V`.
- Source: `/data6/zmf/Wan2.2_SA3_hif4_edges8_fp16_noscale_norowmax_baseline_para01`.
- Method: blockwise safetensors loader. The probe loads true Wan DiT weights
  for patch/time/text embeddings and then one transformer block at a time,
  runs real block forward, captures RoPE-applied self-attention Q/K, and
  computes the 3D cyclic projection metric over the patch grid.
- Grid: `2 x 30 x 52` tokens, from 5 frames at `832 x 480`.
- Heads sampled: `0, 10, 20, 30`.

Direct results:

| Branch/timestep | Layers | Heads | Mean attention R2 | Mean attention relative error | Mean logits R2 | Mean logits relative error |
|---|---|---|---:|---:|---:|---:|
| high-noise / 999 | 0, 8, 20, 39 | 0, 10, 20, 30 | 0.6075 | 0.3486 | 0.5683 | 0.3428 |
| low-noise / 100 | 0, 8 | 0, 10, 20, 30 | 0.6169 | 0.4259 | 0.5713 | 0.3600 |

Layer/head observations:

- High-noise layer 39 is the strongest measured region: attention R2 ranges
  from 0.6347 to 0.8124 across sampled heads, and relative error ranges from
  0.1028 to 0.3545.
- Low-noise layer 8 also shows a strong cyclic component in several heads:
  head 10 reaches R2 0.8706 and head 20 reaches R2 0.8342.
- The phenomenon is not uniform. Layer 0 head 20 is almost non-cyclic in both
  branches: high-noise R2 0.0919 and low-noise R2 0.0009.
- Compared with Qwen3-VL visual attention mean R2 0.0814, Wan2.2 self-attention
  has a much stronger 3D cyclic/relative-offset component.
- Compared with an ideal BCCB matrix, the fit is still imperfect: mean relative
  error is about 0.35 to 0.43, and some heads/layers remain poor.

Interpretation:

- Wan2.2 is a better architectural candidate for a video-circulant phenomenon
  than Qwen3-VL visual attention, because its self-attention is global over a
  latent `F x H x W` token grid with 3D RoPE.
- Existing rowmax preset coverage is high: 94.50% of slots can use preset
  rowmax scalars in the available run artifacts. This indicates substantial
  regularity in QK-logit scale/statistics across branch, attention type,
  timestep, layer, and head.
- However, rowmax stability is not equivalent to BCCB/cyclic relative-offset
  structure. It says logits have stable row maxima, not that `A[i, j]` is mostly
  a function of `j - i`.
- The direct Q/K probe changes the Wan conclusion from "only a candidate" to
  "measured moderate-to-strong 3D cyclic component in selected heads/layers".
- It still does not justify a hard circulant-attention replacement. A strict
  replacement would require consistently high R2 and low residual over many
  denoising timesteps, prompts, layers, and heads.

Current conclusion for Wan2.2:

- Wan2.2 does show a real 3D cyclic/relative-offset component in self-attention.
- The component is substantially stronger than in the Qwen3-VL visual tower.
- The measured structure is head- and layer-dependent, not a universal strict
  BCCB property.
- A small-grid coordinate perturbation (`2 x 15 x 26` patch grid) supports the
  geometry interpretation: high-noise true F-H-W attention R2 is `0.515` and
  random-coordinate R2 drops to `0.012`; low-noise true F-H-W R2 is `0.603` and
  random-coordinate R2 drops to `0.079`. Axis reinterpretations also reduce R2.
  Reverse-coordinate is not destructive because cyclic offset grouping is close
  to invariant under global coordinate reversal.
- A high/low noise-branch stability check over the overlapping small-grid
  layer/head records (`layers 0/8`, `heads 0/10/20/30`) shows mean high-noise
  R2 `0.433`, mean low-noise R2 `0.603`, Pearson high/low R2 correlation
  `0.548`, and Spearman `0.214`. Random-coordinate R2 drop remains large in
  both branches (`0.415` high, `0.524` low). This says the geometry-dependent
  signal is not a one-branch accident, but it is still strongly head/layer and
  denoising-regime dependent.
- The next probe should run on actual denoising latents from several prompts and
  timesteps, then test whether a hybrid policy can route only high-R2 heads to a
  circulant/FFT attention path.

Noise-branch stability details:

| Metric | Value |
|---|---:|
| overlap records | 8 |
| mean high-noise attention R2 | 0.433 |
| mean low-noise attention R2 | 0.603 |
| mean low-minus-high R2 delta | 0.170 |
| Pearson high/low R2 | 0.548 |
| Spearman high/low R2 | 0.214 |
| high random-coordinate R2 drop | 0.415 |
| low random-coordinate R2 drop | 0.524 |
| high axis-mean R2 drop | 0.116 |
| low axis-mean R2 drop | 0.142 |

Interpretation: low-noise layer 8 heads strengthen substantially, layer 0 head
20 is near-zero in both branches, and layer 0 head 30 is high in both branches.
This supports a gated-head/timestep policy rather than a universal circulant
replacement.

## 210 Attention-Replacement Probe

Correction to scope: the replacement target is the attention operator, not
visual projection weights. This section is therefore the main 210-side result
for `block-circulant / simple permutation / proxy Monarch-like`.

What was measured:

- `A`: actual attention matrix after softmax, or logits before softmax for
  reference. The table below reports attention `A`.
- `A @ V` output error: relative error after replacing `A` with the structured
  approximation and multiplying by the true value vectors `V`.
- `grid_cyclic_bccb`: nearest cyclic relative-offset projection over the 2D
  token grid, closest to the Circulant-Attention hypothesis.
- `flat_block_circulant`: nearest per-block circulant projection over flattened
  token order.
- `permuted_flat_block_circulant`: the same projection after the fixed
  row/column permutation cross product; no learned permutation.
- `monarch_like_mask_proxy`: row-renormalized attention retained by the union of
  identity and fixed-permutation block-diagonal masks. This is a cheap proxy,
  not a trained Monarch factorization.

Assets and setup:

- Server/profile: 210 / `/home/spco/sow_linear`.
- Script:
  `E:\Codex_work\ssh_experiment\video_circulant_analysis_20260701\scripts\structured_attention_probe.py`.
- Remote script:
  `/home/spco/sow_linear/video_structured_probe_20260702/scripts/structured_attention_probe.py`.
- ViT result:
  `remote_logs/structured_attention_probe_vit_20260703.json` and `.csv`,
  73344 rows.
- Qwen3-VL result:
  `remote_logs/structured_attention_probe_qwen_20260703.json` and `.csv`,
  82368 rows.

Important ViT scope note:

- The available ViT checkpoint is an SCTM attention variant, not a standard
  dense ViT baseline. Layer 0 patch-patch QK attention is exact from the
  checkpoint input. Layers 1/2/5 use a dense attention-only rollout without the
  logic FFN, so they are diagnostic rather than an exact full-model forward.

Qwen3-VL scope note:

- The Qwen3-VL visual tower is true model forward on real video frames, but its
  attention is per-temporal-slice 2D spatial attention, not global 3D video
  attention.

Attention-space results:

| Model/scope | Method | Maps | Mean attention error | Mean `A @ V` output error | Mean compression |
|---|---|---:|---:|---:|---:|
| ViT layer 0 exact | grid cyclic BCCB | 24 | 0.9794 | 0.8378 | 64.0x |
| ViT layer 0 exact | flat block-circulant | 24 | 0.8509 | 0.6385 | 4.0x |
| ViT layer 0 exact | fixed-perm flat block-circulant | 24 | 0.8216 | 0.6292 | 4.0x |
| ViT layer 0 exact | Monarch-like mask proxy | 24 | 0.5391 | 0.4984 | 2.3x |
| ViT attention-only rollout | grid cyclic BCCB | 72 | 0.9081 | 0.6774 | 64.0x |
| ViT attention-only rollout | flat block-circulant | 72 | 0.7383 | 0.3994 | 4.0x |
| ViT attention-only rollout | fixed-perm flat block-circulant | 72 | 0.7171 | 0.3764 | 4.0x |
| ViT attention-only rollout | Monarch-like mask proxy | 72 | 0.6458 | 0.1895 | 2.3x |
| Qwen3-VL visual | grid cyclic BCCB | 96 | 0.8815 | 0.7654 | 152.0x |
| Qwen3-VL visual | flat block-circulant | 96 | 0.7338 | 0.5937 | 4.0x |
| Qwen3-VL visual | fixed-perm flat block-circulant | 96 | 0.7135 | 0.5753 | 4.1x |
| Qwen3-VL visual | Monarch-like mask proxy | 96 | 0.5209 | 0.4703 | 3.0x |

Interpretation:

- Direct grid-BCCB attention replacement is poor for both 210-side ViT and
  Qwen3-VL visual attention. The `A @ V` output error remains very high:
  `0.8378` for exact ViT layer 0 and `0.7654` for Qwen3-VL visual attention.
- Flat block-circulant attention is better than grid-BCCB under this metric, but
  the output perturbation is still too large for a zero-shot replacement:
  `0.6385` on exact ViT layer 0 and `0.5937` on Qwen3-VL.
- Fixed simple permutations help only mildly. They do not reveal a hidden
  circulant ordering strong enough for replacement.
- The Monarch-like mask proxy gives the lowest output error in this diagnostic,
  but it is not a valid conclusion that Monarch attention will work: the proxy
  is a row-renormalized mask, has low compression, and is not trained.

Why the current methods are poor:

- Grid-BCCB assumes `A[i, j]` is mostly a function of cyclic relative offset
  `j - i`. In the visualized matrices this is false: rows are not cyclic shifts
  of one shared kernel. ViT layer 0 has strong vertical sink columns and very
  low row-argmax diversity, while Qwen layers show content-dependent diagonals,
  sparse off-diagonal stripes, and late-layer sink-like columns.
- The row-argmax diagnostic is a compact check for this failure. If grid-BCCB
  were a good model, row argmaxes would follow stable offset diagonals. Instead
  they collapse to a few columns in some heads or jump irregularly with content.
- Flat block-circulant ignores the real 2D grid and only assumes repeated
  circulant structure inside flattened contiguous blocks. It can match some
  local diagonal texture, but it misses cross-block stripes, global sink columns,
  and content-specific routing. This is why it improves over grid-BCCB but still
  leaves large `A @ V` errors.
- Fixed simple permutations are too weak. They are global and hand-designed,
  while the attention ordering that would be needed is layer/head/input
  dependent. The best fixed permutation therefore only slightly changes the
  block-circulant fit.
- The row-renormalized Monarch-like proxy sometimes has lower `A @ V` error
  than its matrix error suggests because `V` can make several attention
  differences output-equivalent for a particular head. That is not evidence of a
  valid attention replacement: the proxy is still only a fixed mask pattern, not
  a trained Monarch factorization, and it discards many cross-block
  interactions.

Representative visual examples:

- ViT layer 0 head 0: grid-BCCB matrix error `0.9849`, output error `1.0982`.
  The original attention is dominated by a few sink columns, so averaging by
  cyclic offset destroys the row-specific pattern. The row-normalized proxy
  matrix error is still `0.6654`.
- ViT layer 5 head 0 in the attention-only rollout: flat block-circulant is the
  best visual match (`0.4279` matrix error), but the row-normalized proxy has
  matrix error `1.2265` while output error is only `0.0375`. This is a warning
  that low `A @ V` error can be value-subspace-specific rather than a faithful
  attention approximation.
- Qwen layer 8 head 0 frame 1: grid-BCCB is less bad but still high-error
  (`0.6954` matrix, `0.4678` output). It contains local diagonal structure plus
  content-specific stripes, which a single cyclic kernel cannot express. The
  row-normalized proxy matrix error is `0.2441`.
- Qwen layer 26 head 0 frame 0: grid-BCCB is almost unusable (`0.9813` matrix,
  `0.7166` output). Late visual attention is sparse and sink-like, not
  translation-invariant; the row-normalized proxy matrix error is `0.8348`.

Current conclusion for replacing attention on 210 assets:

- Do not zero-shot replace ViT or Qwen3-VL visual attention with
  grid-BCCB/circulant attention.
- Do not treat fixed simple permutation as a sufficient path to circulant
  attention.
- A real replacement experiment would need training or calibration of the
  structured attention operator, then task accuracy or video-quality evaluation.

## Hybrid Attention Decomposition Diagnostic

Motivation:

The failure visualizations show that real attention is not well described by a
single translation-invariant BCCB kernel or by one fixed permutation. A more
faithful structural hypothesis is a mixture:

`attention ~= sink/global component + local/cyclic neighborhood + sparse
content-dependent routing`.

Implementation:

- Script:
  `E:\Codex_work\ssh_experiment\video_circulant_analysis_20260701\scripts\hybrid_attention_decomposition.py`.
- Outputs:
  `remote_logs/hybrid_attention_decomposition_20260704.json` and `.npz`.
- Representative `A @ V` re-export outputs:
  `remote_logs/structured_attention_visual_vit_examples_hybrid_20260704.*`
  and `remote_logs/structured_attention_visual_qwen_examples_hybrid_20260704.*`.
- Figures:
  `fig9_hybrid_attention_decomposition` and
  `fig10_hybrid_attention_tradeoff`.
- Scope: representative attention matrices only; this script consumes the
  saved matrix examples from Fig.8.
- Important limitation: this is an oracle diagnostic. Sink columns and sparse
  top-k routes are selected from the observed attention matrix `A`. Therefore
  the result tests whether this mixed structure can explain the attention, not
  whether a deployable kernel can predict the same routes without forming `A`.
- The global SVD component is clipped/capped for nonnegativity. This improves
  diagnostic fit but destroys the strict rank constraint, so the reported
  "budget ratio" is only a nominal component-budget proxy, not a real
  compression or storage claim.

Components:

- `sink`: exact attention mass on the top column-mass keys.
- `global_svd`: a small SVD-derived global component on the post-sink,
  post-local residual, then clipped/capped to avoid negative attention mass. It
  should not be read as a deployable low-rank factorization in its current form.
- `local_cyclic`: a radius-limited cyclic-offset average on the 2D token grid.
  This keeps the Circulant-Attention idea, but only as the local component
  rather than as the whole attention matrix.
- `sparse_routing`: exact top-k residual entries per row, used as an oracle
  proxy for content-dependent routes.
- The final approximation is row-normalized before the matrix error is
  measured.
- Component masses below are raw sequential fitted magnitudes before final row
  normalization. They are not mutually exclusive probability shares and can
  sum to slightly above or below one.

Representative results:

| Example | Grid BCCB err | Best fixed-perm BCM err | Monarch-like proxy err | Hybrid balanced err | Hybrid balanced nominal budget |
|---|---:|---:|---:|---:|---:|
| ViT L0 H0 | 0.9849 | 0.8618 | 0.6654 | 0.0378 | 6.3x |
| ViT L5 H0 | 0.8429 | 0.4284 | 1.2265 | 0.1342 | 6.3x |
| Qwen L8 H0 F1 | 0.6954 | 0.5996 | 0.2441 | 0.2905 | 6.3x |
| Qwen L26 H0 F0 | 0.9813 | 0.8520 | 0.8348 | 0.1249 | 9.8x |

Component observations:

- ViT L0 H0 is mostly a sink-column case: the balanced hybrid assigns about
  `0.88` raw fitted mass to the sink component. This explains why BCCB
  fails so severely; cyclic averaging destroys a few dominant global columns.
- ViT L5 H0 is dominated by a global pattern: the balanced hybrid has about
  `0.71` global-SVD mass and only `0.18` sink mass. Flat BCM catches part of
  the visual texture, but it cannot represent this global component.
- Qwen L8 H0 F1 is a mixed local/global case: global-SVD mass is about `0.46`,
  local-cyclic mass about `0.26`, sparse-routing mass about `0.10`, and sink
  mass about `0.12`. This is the counterexample where the earlier Monarch-like
  proxy has lower matrix error (`0.2441`) than the balanced hybrid (`0.2905`),
  because its mask happens to keep many useful stripes.
- Qwen L26 H0 F0 is a late-layer sparse/sink case: sink mass is about `0.44`,
  global-SVD mass about `0.26`, and sparse-routing mass about `0.16`. This is
  exactly the pattern a single BCCB kernel cannot express.
- `hybrid_plus` is not monotonically better. For ViT L5 H0, adding a wider
  radius and larger rank increases error from `0.1342` to `0.1959`, showing
  that naive component expansion can oversmooth or double-count residual
  structure.

Difference from the previous 210 methods:

- Grid-BCCB/Circulant Attention assumes one global cyclic kernel. The hybrid
  keeps the cyclic kernel only for the local residual and explicitly separates
  sink/global and sparse content routes.
- Flat BCM and fixed-permutation BCM are static matrix families. The hybrid
  includes data-dependent sparse routing, which is necessary for the observed
  row-specific jumps and stripes.
- The previous Monarch-like proxy is only a fixed mask over `A`. The hybrid is
  a semantic decomposition of the error modes: sink columns, global-SVD
  bias, local cyclic texture, and sparse content routes are measured
  separately.
- Unlike true trained Monarch or VMonarch-style methods, the current hybrid is
  not a trained factorization and does not yet provide a fast path. It is a
  diagnostic target for what a real replacement must learn.

Supplementary `A @ V` re-export:

After adding the hybrid metrics to the representative visual-export script, the
ViT and Qwen examples were re-exported with true value activations:

| Example | Proxy matrix err | Proxy `A @ V` err | Hybrid balanced matrix err | Hybrid balanced `A @ V` err | Hybrid plus `A @ V` err |
|---|---:|---:|---:|---:|---:|
| ViT L0 H0 | 0.6654 | 0.6893 | 0.0378 | 0.0320 | 0.0372 |
| ViT L5 H0 | 1.2265 | 0.0375 | 0.1342 | 0.0522 | 0.0921 |
| Qwen L8 H0 F1 | 0.2441 | 0.1771 | 0.2905 | 0.1586 | 0.1570 |
| Qwen L26 H0 F0 | 0.8348 | 0.5075 | 0.1249 | 0.0729 | 0.0845 |

This reinforces two points. First, the hybrid is not uniformly lower matrix
error than the prior proxy: Qwen L8 is the matrix-error exception. Second,
output error can disagree with matrix fidelity: ViT L5 has much better hybrid
matrix error, but the proxy has lower `A @ V` error for this particular value
subspace. The deployability caveat remains unchanged because the route
selection is still oracle.

Relation to existing work:

- Circulant Attention explicitly models ViT attention as the nearest BCCB
  matrix and uses DFT-based multiplication, achieving `O(N log N)` under that
  hypothesis. Our negative Qwen/ViT cases show that video/visual attention
  often needs additional non-cyclic components.
- StreamingLLM-style attention-sink work motivates preserving dedicated sink
  keys instead of treating them as noise. Our visual matrices show similar
  sink-column failure modes, although here the setting is visual/video
  attention rather than autoregressive LLM streaming.
- Sparse-attention methods such as BigBird motivate local/global/sparse
  patterns. The difference is that our sparse routes are not a fixed pattern;
  they are strongly head/input dependent in the measured matrices.
- Monarch and MonarchAttention-style methods provide hardware-friendly
  structured matrix families. Our result suggests why a fixed Monarch-like
  mask is insufficient on these examples: the dynamic sparse/sink structure
  must be fitted or predicted, not just masked by a static layout.
- VMonarch is the closest video-generation-side direction because it explicitly
  targets sparse spatio-temporal attention in Video DiTs using structured
  Monarch matrices and minimal tuning. Our Wan2.2 finding is compatible with
  that direction, but the current local hybrid probe is much weaker: it only
  decomposes saved 2D attention examples and does not implement alternating
  minimization, online updates, or video-quality evaluation.
- MonarchRT and RoPeSLR are also aligned with the current conclusion:
  video-generation attention has geometry/periodicity, but practical
  acceleration needs dynamic sparse semantic routing and low-rank/global
  structure. This supports our `sink/global + local-cyclic + sparse routing`
  direction over a pure BCCB replacement.
- Vision-register literature explains why visual sinks can be functional
  scratch space, while 2026 follow-up work warns that registers are not needed
  by every ViT family. This matches the measured split between ViT-like
  sink/register behavior and more dynamic Qwen visual routing.

Current bottlenecks and defects:

- The hybrid diagnostic needs the dense `A` to choose sinks and top-k routes.
  A deployable method must predict these routes from Q/K statistics, cached
  routing state, or a learned router without materializing full attention.
- `A @ V` output error has only been recomputed for the representative
  re-exported examples above. The next step is adding the hybrid method to the
  full `structured_attention_probe.py` sweep so all sampled heads/layers/frames
  have matrix and output-error metrics.
- The current nominal budget count is not a true compression ratio. The clipped
  SVD-derived global component can have high numerical rank after clipping, so
  a deployable version must use a genuine factorized low-rank/global path or a
  different nonnegative parameterization.
- The residual components are fitted sequentially, so they can double-count or
  oversmooth. A real method should solve a joint constrained fit or train the
  components end to end.
- The probe is representative, not aggregate. It should be expanded to all
  sampled heads/layers/frames, then to Wan2.2 true denoising latents.
- The best practical direction is therefore not "replace attention with
  circulant attention", but "gate attention heads into sink/global,
  local-cyclic, and sparse-routing paths, and train/calibrate the router".

## Hybrid Transfer Probe

Script:

- `scripts/hybrid_transfer_probe.py`

Outputs:

- `remote_logs/hybrid_transfer_probe_20260708.json/csv`
- `figures/fig17_hybrid_transfer_probe.png/pdf`

Question:

The oracle hybrid decomposition fits representative matrices well, but it
selects sink columns and sparse routes from the target dense attention matrix.
This transfer probe asks whether those supports/templates behave like reusable
structure or like target-specific routing.

Method:

- Load the four saved representative matrices from
  `hybrid_attention_decomposition_20260704`.
- For same-grid source/target pairs, compare:
  - target oracle hybrid error;
  - target support-only error using the target's own sink/local/sparse support
    but target attention values;
  - source-support transfer error using another map's support on the target
    attention values;
  - source hybrid template error using another map's fixed hybrid matrix.
- Measure support, sink-column, and sparse-route Jaccard overlap.

Aggregate results over the six same-grid ordered pairs:

| Scope | Oracle hybrid | Target support only | Source support transfer | Source hybrid template | Sink Jaccard | Sparse-route Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| all same-grid | 0.154 | 0.825 | 1.569 | 2.007 | 0.000 | 0.009 |
| same family | 0.086 | 0.761 | 1.565 | 2.461 | 0.000 | 0.015 |
| cross family | 0.188 | 0.857 | 1.571 | 1.780 | 0.000 | 0.006 |

Interpretation:

- The gap between target oracle hybrid (`0.154`) and source-support transfer
  (`1.569`) is large. The successful hybrid decomposition is not a reusable
  fixed mask/template in this saved example set.
- Sink columns do not overlap across the same-grid pairs, and sparse route
  overlap is near zero. This supports the dynamic-routing bottleneck rather
  than a hidden fixed permutation explanation.
- Target support-only error is also much higher than oracle hybrid, especially
  for ViT L5 H0. The low-rank/global component is not just a binary support;
  the fitted weights/profiles matter.
- Scope caveat: this is a transfer diagnostic over saved representative
  matrices, not a full train/test sweep. It is strong evidence against directly
  deploying the current oracle hybrid as a static replacement, but it does not
  rule out a learned router or calibrated sink/global module.

## 210 Structured Weight-Fit Probe (Supplement Only)

This is a separate weight-space experiment and does not answer the attention
replacement question above. It asks only whether existing attention projection
matrices can be zero-shot approximated by the 210-side structured linear
families. It does not measure attention maps and does not replace any module
during inference.

Assets and setup:

- Server/profile: 210 / `/home/spco/sow_linear`.
- Environment: `qwen_omni`, CPU execution.
- Local script:
  `E:\Codex_work\ssh_experiment\video_circulant_analysis_20260701\scripts\structured_matrix_probe.py`.
- Remote script:
  `/home/spco/sow_linear/video_structured_probe_20260702/scripts/structured_matrix_probe.py`.
- Results:
  `remote_logs/structured_matrix_probe_vit_qwen_20260702.json` and `.csv`.
  The final run produced 6816 matrix/method/block/permutation records.
- ViT matrices: 16 attention projection weights from layers `0, 1, 2, 5`
  (`q/k/v/out`, each `192 x 192`) in the CIFAR-10 ViT-LGN checkpoint.
- Video-understanding matrices: 16 Qwen3-VL visual projection weights from
  layers `0, 8, 16, 26` (`q/k/v/out`, each `1152 x 1152`; `q/k/v` split from
  `attn.qkv.weight`).

Methods:

- `block_circulant`: nearest Frobenius projection where each `b x b` block is
  independently circulant. This matches the parameterization style of the
  `BlockCirculantLinear` implementation found on 210.
- `permuted_block_circulant`: the same projection after a fixed deterministic
  row/column permutation. The tested set is the row-by-column cross product of
  `identity`, `bit_reverse_padded`, `stride5`, `stride7`,
  `interleave_groups16`, and `interleave_groups32`, excluding identity x
  identity. The permutation is not learned; the table reports the best fixed
  pair.
- `monarch_like_proxy`: energy retained by the union of two fixed
  block-diagonal layouts, identity plus a fixed permutation. This is only a
  cheap Monarch-like proxy score, not a trained Monarch product factorization.

Best mean results over the sampled matrices:

| Family | Method | Best block | Mean relative error | Mean fit score | Mean compression |
|---|---|---:|---:|---:|---:|
| ViT | block-circulant | 8 | 0.9348 | 0.1261 | 8.0x |
| ViT | fixed-perm block-circulant | 8 | 0.9326 | 0.1303 | 8.0x |
| ViT | Monarch-like proxy | 32 | 0.8284 | 0.3137 | 3.3x |
| Qwen3-VL visual | block-circulant | 16 | 0.9681 | 0.0627 | 16.0x |
| Qwen3-VL visual | fixed-perm block-circulant | 16 | 0.9679 | 0.0632 | 16.0x |
| Qwen3-VL visual | Monarch-like proxy | 64 | 0.9439 | 0.1090 | 9.3x |

Interpretation:

- The ViT and Qwen3-VL visual projection weights are not naturally close to the
  simple block-circulant family. The fit score is close to the dimensional
  baseline expected from an unconstrained matrix projected into a `1/b`
  block-circulant subspace: ViT with `b=8` gives about `0.126`, and Qwen3-VL
  with `b=16` gives about `0.063`.
- Fixed simple permutations do not uncover a hidden block-circulant ordering.
  Even after testing the fixed row/column cross product, the improvement is
  negligible: ViT mean error changes from `0.9348` to `0.9326`, and Qwen3-VL
  changes from `0.9681` to `0.9679`.
- The Monarch-like proxy retains more energy, especially for ViT, but this comes
  with much lower compression and is only a mask-energy proxy. It should not be
  read as evidence that a real Monarch replacement would work without training.
- This weight-space result is consistent with the Qwen3-VL attention-map result:
  the visual model shows weak circulant structure. It also clarifies the Wan2.2
  story: the stronger Wan signal appears in attention activations over the 3D
  latent grid, not in a generic claim that projection weights are already
  block-circulant.

Current conclusion for 210 structured linear probes:

- Do not zero-shot replace ViT or Qwen3-VL visual projection weights with
  block-circulant matrices.
- Simple fixed permutations are not enough.
- A meaningful Monarch/block-circulant replacement test would need trained
  adapters or fine-tuning, plus accuracy/generation-quality evaluation, rather
  than an offline projection alone.

## Review Follow-Up

An independent read-only review flagged over-broad scans, evidence-scope
ambiguity, and reproducibility metadata gaps. Actions taken:

- Patched `remote_model_scan.py` to match against relative paths rather than
  absolute user/home prefixes, changed the default `wan` keyword to `wan2` /
  `wan2.2`, skipped hidden/cache/credential-like paths, and uploaded the fixed
  script to 210/34/35.
- Moved the over-broad local and remote `scan_210.jsonl` and `scan_434.jsonl`
  files, plus older pre-fix `scan_*.jsonl` files, into task-local
  `trash/...` directories instead of deleting them. Replaced them with
  root-scoped `scan_210_safe.jsonl`, `scan_434_safe.jsonl`, and
  `scan_435_safe.jsonl`.
- Patched and reran `qwen3vl_visual_circulant_probe.py` so result JSONs carry
  reproducibility metadata and explicitly state the per-temporal-slice 2D
  metric scope.
- Tightened this report so Qwen3-VL is treated as negative/weak 2D evidence and
  Wan2.2 is treated as a measured moderate/partial 3D cyclic-attention case.

## Bottom Line

1. Video understanding, measured on Qwen3-VL visual attention: no strong
   Circulant-Attention-like phenomenon. The visual tower is per-frame spatial
   attention, and measured 2D BCCB fit is weak.
2. Video generation, directly probed on Wan2.2: selected self-attention
   heads/layers show a clear 3D cyclic component, with mean attention R2 around
   0.61 on the sampled high/low-noise probes.
3. 210 attention-space replacement probes on ViT and Qwen3-VL visual attention
   are negative for zero-shot circulant replacement: grid-BCCB, flat
   block-circulant, and fixed-permutation block-circulant all leave large
   `A @ V` output errors. The Monarch-like mask proxy is lower-error but is not
   a trained Monarch attention replacement.
4. The new oracle hybrid diagnostic supports a better structural hypothesis:
   real visual/video attention is closer to sink/global + local-cyclic
   + sparse content routing than to a single BCCB or fixed-permutation BCM
   family. On representative matrices, hybrid balanced reduces matrix error to
   `0.0378`, `0.1342`, `0.2905`, and `0.1249`, but it still needs dense `A` to
   choose routes and is not deployable yet.
5. The 2026-07-07 matrix-level component intervention explains why the
   replacement is poor. Across the same four representative matrices, the full
   hybrid mean error is `0.147`; removing sink/global raises it to `1.233`,
   while removing local-cyclic raises it to `0.202` and removing sparse-routing
   to `0.236`. Thus the biggest missing term in plain BCCB/BCM is an explicit
   sink/global low-rank path; Qwen L8/L26 also need local and sparse routing.
6. The value-subspace stress test makes the `A @ V` caveat concrete. ViT
   union-mask error changes from `0.093` on true `V` to `0.184`, `0.371`, and
   `0.419` on permuted, orthogonalized, and random `V`; Qwen3-VL visual changes
   from `0.600` to `0.502`, `1.113`, and `1.128`. A low output error on the
   observed value vectors is therefore not sufficient evidence of faithful
   attention replacement.
7. Head-output keep/drop interventions sharpen the mechanism split. ViT
   keep-only union error is `0.093` and drop-union error is `0.486`, so the
   oracle sparse/union route is close to sufficient and functionally important
   at the head-output level. Qwen3-VL visual keep-only union error remains
   `0.600` and drop-union is `0.702`, so fixed sink/local/top-k routes are not
   enough; dynamic content routing remains the missing term.
8. The projection-weight fit is only a supplement; it should not be used as the
   main evidence for or against replacing attention.
9. This is partial BCCB-like structure, not strict BCCB. Do not replace all Wan
   attention with circulant attention; a plausible next direction is a
   head/layer/timestep-gated hybrid path for consistently high-R2 heads and a
   learned or calibrated router for sink/sparse residuals.
10. Wan coordinate perturbation strengthens the inductive-bias interpretation:
    the measured 3D cyclic component depends on coherent F-H-W geometry rather
    than arbitrary token coordinates. It still needs actual denoising-latent and
    quality/loss validation before being treated as a deployable replacement
    criterion.
11. The 2026-07-08 hybrid transfer probe explains an additional deployment
    bottleneck: target oracle hybrid error is low (`0.154` mean), but using a
    different same-grid map's support raises mean error to `1.569`, and a fixed
    source hybrid template reaches `2.007`. Sink-column overlap is zero and
    sparse-route overlap is near zero. The current hybrid result is therefore a
    mechanism diagnostic, not a static reusable attention layout.
12. The 2026-07-08 Wan noise-branch stability probe adds a partial answer to
    "is it constant or occasional": over overlapping layers 0/8 and heads
    0/10/20/30, high/low R2 has moderate Pearson correlation (`0.548`) but low
    Spearman (`0.214`), low-noise is stronger on average (`0.603` vs `0.433`),
    and random-coordinate destruction remains strong in both branches. The
    cyclic structure is real and geometry-dependent, but not universal across
    all heads or timesteps.
13. The sink/no-op correlation probe strengthens the sink mechanism reading.
    In ViT, top-2 sink-column mass correlates strongly with lower entropy
    (`r=-0.952`), drop-sink output error (`r=0.772`), and sink raw component
    norm (`r=0.946`). In Qwen3-VL visual, drop-sink correlation remains high
    (`r=0.775`), but true-`V` vs random-`V` union error also correlates strongly
    (`r=0.852`), consistent with additional value-subspace and dynamic-routing
    effects. This is correlation evidence, not task-loss causality.
14. The 2026-07-10 ViT/SCTM route causal probe adds the first task-level
    intervention evidence for this repository. On the actual saved
    ViT-LGN/SCTM checkpoint, evaluated through the real SCTM top-k route
    selection, auxiliary accumulator, logic FFN, and classifier path, baseline
    CIFAR-10 loss/accuracy over 256 test samples are `1.235/0.559`. Dropping
    the strongest selected CLS-to-patch route raises loss by `0.214`, reduces
    accuracy by `0.055`, and flips `24.2%` of predictions; dropping the weakest
    selected route changes loss by only `0.001`, and the one-random-selected
    route control averaged over 8 seeds changes loss by `0.023 +/- 0.024`.
    Dropping the top two selected routes raises loss by `0.421`; zeroing all
    selected SCTM CLS routes raises loss by `3.386` and reduces accuracy to
    `0.102`. This supports the mechanism claim that the ranked SCTM sparse
    routes are functional, not just visualization artifacts. It does not yet
    close the analogous task-level causal question for Wan denoising or Qwen
    multimodal understanding.
