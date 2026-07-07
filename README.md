# Multimodal Compression: Video Circulant/Hybrid Attention Probe

本目录整理了当前一轮关于视频生成、视频理解和 ViT 注意力结构化替换的实验代码、日志、图和报告。

核心问题来自 `LeapLabTHU/Circulant-Attention`：视觉/视频模型的 attention 是否也能近似为 BCCB/circulant attention，从而用 FFT 或结构化矩阵替换。当前结论是：Qwen3-VL visual 与 ViT 不适合直接零训练替换为单一 circulant attention；Wan2.2 在部分 heads/layers/timesteps 上有更强 3D cyclic 成分，但仍不是严格 BCCB；更合理的方向是 `sink/global + local-cyclic + sparse routing` 的混合 attention。

## 目录

- `ANALYSIS_REPORT.md`：完整实验报告和结论。
- `ATTENTION_PATTERN_MECHANISM_STUDY.md`：attention sink/outlier/local/sparse
  pattern 的文献机制、全量 probe 结果和实验设计。
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
