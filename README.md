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

## 注意事项

- Hybrid 是 oracle diagnostic：sink columns 和 sparse top-k routes 来自已知 dense `A`，暂时不是可部署 kernel。
- `global_svd` 经过 clipping/capping，nominal budget ratio 不是实际低秩压缩率。
- ViT 后续层结果来自 dense attention-only rollout，不是完整 SCTM+FFN forward。
- Qwen3-VL visual tower 是 per-temporal-slice 2D spatial attention，不是全局 3D video attention。
