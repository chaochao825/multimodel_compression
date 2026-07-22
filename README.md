# Multimodal Compression: Video Circulant/Hybrid Attention Probe

本目录整理了当前一轮关于视频生成、视频理解和 ViT 注意力结构化替换的实验代码、日志、图和报告。

核心问题来自 `LeapLabTHU/Circulant-Attention`：视觉/视频模型的 attention 是否也能近似为 BCCB/circulant attention，从而用 FFT 或结构化矩阵替换。当前结论是：Qwen3-VL visual 与 ViT 不适合直接零训练替换为单一 circulant attention；Wan2.2 在部分 heads/layers/timesteps 上有更强 3D cyclic 成分，但仍不是严格 BCCB；更合理的方向是 `sink/global + local-cyclic + sparse routing` 的混合 attention。

新增的 `online_video_state_decomposition/` 将这条负面边界推进到在线
视频理解：它实现并审计了 query-conditioned bounded memory、LLaVA native
feature cache、calibration-only PCA 和 sparse residual event probe。当前最佳
压缩状态在最终未使用的 300 样本 MVBench reserve 上缩小 `7.84x`，并以
1.571% 的单侧 95% loss 上界通过严格的 `2%` finite-sample
non-inferiority gate。查询记忆在相同压缩状态下提升 `+2.0` 个百分点，但
McNemar `p=0.1094`，因此仍是独立正向信号而非确定优越结论。

## 目录

- `ANALYSIS_REPORT.md`：完整实验报告和结论。
- `ATTENTION_PATTERN_MECHANISM_STUDY.md`：attention sink/outlier/local/sparse
  pattern 的文献机制、全量 probe 结果和实验设计。
- `online_video_state_decomposition/`：在线理解的固定预算记忆、原生视觉状态
  压缩、稀疏事件残差、验证器、精选聚合结果和研究边界。
- `scripts/`：所有探测、结构拟合、hybrid 分解和可视化脚本。
- `figures/`：论文风格 PNG/PDF 图，包括 BCCB 拟合、attention 替换、失败模式和 hybrid 分解图。
- `remote_logs/`：210/34/35 服务器回传的 JSON/CSV/NPZ 结果。包含大文件原始 attention-space probe 结果。

## 方法

1. Qwen3-VL visual 2D cyclic/BCCB probe：测量每个 temporal slice 内的 spatial attention 是否近似 cyclic relative-offset kernel。
2. Wan2.2 direct Q/K probe：在 latent `F x H x W` grid 上测量 3D cyclic/BCCB 成分。
3. 210 structured attention replacement probe：比较 grid-BCCB、flat block-circulant、fixed-permutation BCM、Monarch-like mask proxy 对真实 attention `A` 和 `A @ V` 的替换误差。
4. 210 structured weight-fit probe：补充性分析 projection weight 是否天然接近 block-circulant/Monarch-like 结构。
5. Hybrid diagnostic：将代表 attention matrix 分解为 oracle `sink/global-SVD + local-cyclic + sparse-routing`，用于解释为什么单一 BCCB/BCM/固定 proxy 表现差。
6. Full-sweep pattern probe：统计 ViT/Qwen 全量采样中的 sink mass、row-argmax collapse、local mass、row top-k sparsity、effective rank，以及 true-`V` / random-`V` stress test。
7. Matrix-level component intervention：在已保存的 hybrid 分解上分别去掉 sink/global、local-cyclic、sparse-routing 分量，分析当前结构化替换为什么失败。
8. Value-subspace stress：在同一批 ViT/Qwen attention 上测试 true/permuted/orthogonalized/random `V`，检验低 `A @ V` error 是否依赖当前 value 子空间。
9. Head-output intervention：对 sink/local/row-topk/union mask 做 keep-only 和 drop+renorm，测量 head-level `A @ V` 输出变化。
10. Wan coordinate perturbation：在 Wan2.2 small-grid latent 上重解释/打乱 3D token 坐标，验证 cyclic R2 是否依赖真实 F-H-W 几何对齐。

## 关键结果

- Qwen3-VL visual attention：平均 BCCB R2 约 `0.0814`，2D cyclic 结构弱。
- Wan2.2 selected self-attention： sampled high/low-noise probes 中平均 attention R2 约 `0.61`，有明显但非严格的 3D cyclic 成分。
- Zero-shot attention replacement：
  - Qwen3-VL visual grid-BCCB `A @ V` error `0.7654`。
  - ViT layer0 grid-BCCB `A @ V` error `0.8378`。
  - fixed permutation BCM 只轻微改善，不足以替换。
  - Monarch-like proxy 较低，但只是 row-renormalized mask，不是训练后的 Monarch attention。
- Hybrid diagnostic：
  - Qwen L26 H0 F0：proxy `A @ V` error `0.5075`，hybrid balanced `0.0729`。
  - ViT L0 H0：proxy `A @ V` error `0.6893`，hybrid balanced `0.0320`。
  - Qwen L8 H0 F1 是反例：proxy matrix error `0.2441` 低于 hybrid balanced `0.2905`，说明 hybrid 是解释性诊断，不是统一最优近似。
- Full-sweep mechanism probe：
  - ViT top-2 column mass mean `0.448`，row-argmax unique mean `0.081`，更像 sink/no-op/scratch 主导。
  - Qwen3-VL top-2 column mass mean `0.204`，row-argmax unique mean `0.210`，更动态，layer 8 的 dynamic/local routing 最明显。
  - random-`V` stress 明显放大 union-mask output error，说明低 `A @ V` error 不能单独证明 attention 替换忠实。
- Component intervention：
  - 四个代表矩阵上 full hybrid 平均 matrix error `0.147`，Grid BCCB `0.876`，Monarch-like proxy `0.743`。
  - 去掉 sink/global 后平均 error 升至 `1.233`，去掉 local-cyclic 为 `0.202`，去掉 sparse-routing 为 `0.236`。
  - 这说明当前 BCCB/BCM 替换差的主因是缺少显式 sink/global 低秩通道；Qwen L8/L26 还需要 local/sparse routing。
- Value-subspace stress：
  - ViT union-mask output error：true/permuted/orthogonalized/random `V` 为 `0.093/0.184/0.371/0.419`。
  - Qwen3-VL visual 对应为 `0.600/0.502/1.113/1.128`。
  - 结论是 `A @ V` 单点低误差不可靠；ViT 后层明显依赖当前 value 子空间，Qwen 还缺动态 routing。
- Head-output intervention：
  - ViT keep-only union error `0.093`、drop-union error `0.486`，row-top4/union 对 head output 近似充分且被 drop 后影响明显。
  - Qwen keep-only union error `0.600`、drop-union error `0.702`，sink/local/row-topk 都不能单独解释，支持动态 content routing。
- Wan coordinate perturbation：
  - small-grid patch grid `2x15x26` 上，high-noise true F-H-W attention R2 `0.515`，random-coordinate R2 `0.012`。
  - low-noise true F-H-W R2 `0.603`，random-coordinate R2 `0.079`。
  - 轴重解释也降低 R2，说明 Wan 的 cyclic 成分依赖 3D latent/RoPE 坐标对齐；reverse-coordinate 近似不变，不能作为破坏性对照。

## 注意事项

- Hybrid 是 oracle diagnostic：sink columns 和 sparse top-k routes 来自已知 dense `A`，暂时不是可部署 kernel。
- `global_svd` 经过 clipping/capping，nominal budget ratio 不是实际低秩压缩率。
- ViT 后续层结果来自 dense attention-only rollout，不是完整 SCTM+FFN forward。
- Qwen3-VL visual tower 是 per-temporal-slice 2D spatial attention，不是全局 3D video attention。

## 2026-07-08 Update: Hybrid Transfer Probe

- New script: `scripts/hybrid_transfer_probe.py`.
- New outputs: `remote_logs/hybrid_transfer_probe_20260708.json/csv`.
- New figure: `figures/fig17_hybrid_transfer_probe.png/pdf`.
- Result over six same-grid source-target pairs: target oracle hybrid mean
  error is `0.154`, but source-support transfer error is `1.569` and fixed
  source-hybrid-template error is `2.007`.
- Sink-column Jaccard is `0.000`; sparse-route Jaccard is about `0.009`.
- Interpretation: the hybrid decomposition is a useful mechanism diagnostic,
  but its non-local sink/sparse routing is target-specific. A deployable
  replacement needs a learned/calibrated sink/global path and content-aware
  sparse router, not a static BCCB/BCM/Monarch-like layout.

## 2026-07-08 Update: Wan Noise-Branch Stability

- New script: `scripts/wan_noise_branch_stability.py`.
- New outputs: `remote_logs/wan_noise_branch_stability_20260708.json/csv`.
- New figure: `figures/fig18_wan_noise_branch_stability.png/pdf`.
- Overlapping Wan small-grid records: layers `0/8`, heads `0/10/20/30`.
- Mean high-noise attention R2 is `0.433`; mean low-noise attention R2 is
  `0.603`.
- High/low R2 Pearson correlation is `0.548`, but Spearman is only `0.214`.
- Random-coordinate R2 drop remains strong in both branches: `0.415` high and
  `0.524` low.
- Interpretation: Wan's 3D cyclic signal is geometry-dependent and appears in
  both noise branches, but it is still head/layer/timestep dependent. This
  supports a gated hybrid policy rather than universal circulant attention.

## 2026-07-08 Update: Literature Alignment

- Updated `ATTENTION_PATTERN_MECHANISM_STUDY.md` with a source-to-mechanism
  table connecting attention sinks, outlier features, registers, BCCB geometry,
  and video DiT sparse/low-rank attention to the current probes.
- Corrected the 2025 attention-sink title to `Attention Sinks and Outlier
  Features: A 'Catch, Tag, and Release' Mechanism for Embeddings`.
- Added recent video structured-attention context: `VMonarch`, `MonarchRT`,
  `RoPeSLR`, and `MonarchAttention`.
- Main conclusion is unchanged: current evidence favors learned/calibrated
  `sink/global + local-cyclic + sparse routing`, not a static universal
  circulant or Monarch-like mask.

## 2026-07-08 Update: Sink/No-op Correlation

- New script: `scripts/sink_noop_correlation_probe.py`.
- New outputs: `remote_logs/sink_noop_correlation_20260708.json/csv` and
  `remote_logs/sink_noop_quartiles_20260708.csv`.
- New figure: `figures/fig19_sink_noop_correlation.png/pdf`.
- ViT sink strength strongly anti-correlates with entropy (`r=-0.952`) and
  correlates with drop-sink output error (`r=0.772`) and raw sink component
  output norm (`r=0.946`).
- Qwen3-VL visual also shows high sink-vs-drop-sink correlation (`r=0.775`),
  but true-`V` vs random-`V` union error is strongly coupled (`r=0.852`),
  indicating stronger value-subspace/dynamic-routing effects.
- Interpretation: sinks are functional partial-update/scratch routes in these
  heads, not pure noise; causal task-loss masking remains the next missing
  test.

## 2026-07-10 Update: ViT/SCTM Route Causal Probe

- New script: `scripts/vit_sctm_route_causal_probe.py`.
- New outputs: `remote_logs/vit_sctm_route_causal_20260708.json/csv`.
- New figure: `figures/fig20_vit_sctm_route_causal.png/pdf`.
- This probe uses the actual saved ViT-LGN/SCTM checkpoint forward path rather
  than the earlier dense-attention proxy: SCTM top-k route selection, value
  aggregation, auxiliary accumulator, logic FFN, classifier head, and CIFAR-10
  CE loss are all active.
- On 256 CIFAR-10 test samples, baseline loss/accuracy are `1.235/0.559`.
  Dropping the strongest selected route raises loss by `0.214`, drops accuracy
  by `0.055`, and flips `24.2%` of predictions. Dropping the weakest selected
  route changes loss by only `0.001`; the one-random-selected-route control,
  averaged over 8 seeds, changes loss by `0.023 +/- 0.024`.
- Dropping the top two selected routes raises loss by `0.421`; zeroing all SCTM
  CLS routes raises loss by `3.386` and reduces accuracy to `0.102`.
- Interpretation: in this ViT/SCTM model, the ranked selected CLS-to-patch
  routes are task-functional. This upgrades the ViT evidence from matrix/output
  proxy diagnostics to a small-batch task-level causal intervention, while Wan
  denoising-loss and Qwen multimodal task causal probes remain open.
