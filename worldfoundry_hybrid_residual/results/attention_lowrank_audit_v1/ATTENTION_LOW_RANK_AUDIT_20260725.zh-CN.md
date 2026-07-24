# World Foundry 视频 DiT Attention 与低秩加速审计

日期：2026-07-25

硬件：NVIDIA H200 NVL 143 GB

模型：Wan2.1-T2V-1.3B / World Foundry，30 blocks，hidden size 1536，12 heads，head dim 128，FFN size 8960

结论状态：实测结论；论文数字均标为作者报告值，不能与本实验直接横向比较

## 1. 核心结论

1. **长视频的核心应该放在 self-attention。** 完整 20-step profile 显示，F17（7,800 tokens）每个增量去噪步中 self-attention 占 21.81%，F81（32,760 tokens）升至 53.88%。F81 已经是明确的 attention-bound；F17 则仍由 elementwise/copy/GELU/modulation 等碎片化 kernel 主导（47.76%）。
2. **此前“attention 只占约 6.3%”是 profile 口径错误。** 旧 profile 实际使用 `--sampling-steps 1`，固定的 VAE encode/decode、初始化和框架开销稀释了 attention 占比。本文用相同 F17/F81 的 20-step trace 减去 1-step trace，隔离 19 个增量去噪步。
3. **现有结果不说明“低秩对视频模型无效”。** 它只否定了当前的统一静态 rank-16 FFN/量化残差，以及把低秩 correction 作为独立 kernel 叠加到已有计算之上的实现。
4. **纯低秩不是最合理的 attention 近似。** 真实 Wan QKV oracle 显示，纯 rank-16 的 attention 输出误差在 F17/F81 分别为 31.01%/36.06%；纯 top-128 稀疏误差为 60.00%/64.56%；但 top-128 + rank-16 residual 在仅 4.97%/3.56% 的表示预算下把误差降到 4.91%/6.55%。这与 SLA 的“少量大权重高秩 + 大量边际权重低秩”观察一致。
5. **建议主线改为 fused sparse-high-rank + low-rank-tail + cache-aware refresh。** BCM/CM 只能作为 marginal branch 的候选结构；在没有证明频域/循环结构且没有 fused kernel 前，不应继续作为主线。

## 2. 为什么当前加速结果显得异常

### 2.1 完整去噪 profile

| GPU kernel 类别 | F17 占比 | F81 占比 | 解释 |
|---|---:|---:|---|
| self-attention core | 21.81% | 53.88% | 随 token 数近似二次增长 |
| cross-attention core | 2.42% | 1.38% | 文本长度短，不是主瓶颈 |
| linear GEMM | 23.59% | 14.29% | QKVO + FFN 投影 |
| elementwise / memory | 47.76% | 27.99% | copy、cat、GELU、调制与大量小 kernel |
| normalization | 3.69% | 2.11% | LayerNorm/RMSNorm |
| other | 0.72% | 0.36% | 未归类 kernel |

这里的差分来自两次独立 trace，适合做组件占比，不应当替代端到端 wall-clock。VAE 项在相减后被显式排除。

### 2.2 理论尺度与实测一致

对 hidden size `d=1536`、FFN size `m=8960`、token 数 `N` 的 Wan block，粗略 FLOPs 为：

```text
self-attention core      ~= 4 N^2 d
QKV/O projections       ~= 8 N d^2
FFN                     ~= 4 N d m
```

attention core 与 FFN 的交点约为 `N=m=8960`；与 FFN+QKVO 的交点约为 `N=m+2d=12032`。F17 的 `N=7800` 低于两个交点，F81 的 `N=32760` 远高于交点，因此 F81 attention 占比跃升是模型复杂度决定的，不是 profiler 偶然波动。

### 2.3 H200 实测说明问题在实现而非 FP8 理论

| 算子 | 相对匹配 BF16 的实测速度 | 判断 |
|---|---:|---|
| F17 FA3 FP8 attention | 0.99x | 短序列无法摊薄转换开销 |
| F17 SageAttention | 1.27x | attention kernel 本身仍有可兑现空间 |
| F81 FA3 FP8 attention | 1.51x | 长序列开始受益 |
| World Foundry dynamic FP8 QKV | 0.22x | 动态 scale/cast/launch 完全吞掉收益 |
| World Foundry dynamic FP8 FFN-down | 0.24x | 同上 |
| static-input FP8 Q 下界 | 1.50x | 证明 H200 FP8 GEMM 本身可以更快 |
| static FP8 + independent rank-16 Q | 0.46x | correction 的额外读写与 launch 反而更慢 |

这复现了 [SVDQuant](https://arxiv.org/abs/2411.05007) 的系统结论：低秩高精度分支如果独立运行，额外 activation traffic 会抵消量化收益；必须像 Nunchaku 一样融合低比特主分支和低秩分支。

## 3. 当前结果究竟否定了什么

### 3.1 否定：统一静态 rank-16 activation defect

F17 activation-defect probe 的全 block 统计为：

| 缺陷类型 | rank-16 能量 | rank-64 能量 | 90% 能量所需 rank |
|---|---:|---:|---:|
| FP8 quantization defect `Q` | 34.22% | 43.78% | 758 |
| cache reuse defect `C` | 52.93% | 61.03% | 556 |

少数晚层是例外，例如 block 24 的 cache defect rank-16 能量达到 79.64%，但 block 6/12 分别只有 26.19%/22.98%。因此“所有层、所有 step 用同一个 rank-16 correction”在统计上不完备；合理策略必须按 layer/head/timestep bucket 选择是否启用和分配 rank。

### 3.2 否定：加法式 correction 自动带来加速

若原算子为 `XW`，低秩替换 `XUV^T` 的复杂度是 `O(Nr(d_in+d_out))`，只有它**替换** dense work 且 `r` 足够小时才可能加速。当前实现实际计算的是：

```text
Q(XW) + XUV^T
```

它没有删除主分支，只新增两次 GEMM、activation 读取和 kernel launch。因此即使 reconstruction error 改善，也没有结构性加速上限。实测 static FP8 Q 的相对误差从 3.926% 降到 rank-16 correction 的 3.846%，仅改善约 2.0%，速度却从 1.50x 降到 0.46x。

### 3.3 否定：静态 FFN row-block sparsity 是主线

八组多 prompt/seed F17 实验中：

| 方法 | 几何平均速度 | frame SSIM |
|---|---:|---:|
| FP8 | 1.017x | 0.7966 |
| FP8 middle-1 | 0.998x | 0.9542 |
| hybrid middle-1（含低秩 correction） | 0.990x | 0.9483 |

低秩 correction 没有提高质量，反而稍降 SSIM 并增加耗时。静态 row-block 作用在 FFN 权重/输出维度，既没有触及 F81 的 `N^2` 主项，也没有匹配视频 attention 随 prompt/head/layer 变化的支持集。

## 4. 真实 Attention Oracle：纯低秩为何失败，混合为何可行

probe 使用真实 Wan2.1 layer 0、timestep 1000、conditional branch 的 Q/K/V，覆盖全部 12 heads，并在 512 个均匀采样 query 上比较表示能力。它是**表示 oracle**，不包含 mask search、basis construction、RoPE、调度和 kernel 开销，不能当成真实速度。

| 长度 | 方法 | 表示预算 | attention mass | 输出相对 L2 | 输出 cosine |
|---|---|---:|---:|---:|---:|
| F17 | pure rank-16 | 3.33% | 100% | 31.01% | 0.8564 |
| F17 | top-128 only | 1.64% | 38.37% | 60.00% | 0.6801 |
| F17 | top-128 + rank-16 tail | 4.97% | 38.37% exact | **4.91%** | **0.9982** |
| F81 | pure rank-16 | 3.17% | 100% | 36.06% | 0.8013 |
| F81 | top-128 only | 0.39% | 34.28% | 64.56% | 0.6269 |
| F81 | top-128 + rank-16 tail | 3.56% | 34.28% exact | **6.55%** | **0.9969** |

解释是 attention map 并非整体低秩：运动边界、主体关联、时空对角线和少量远程匹配形成高秩尖峰；去掉这些尖峰后，剩余平滑长尾更接近低秩。纯 top-k 又丢掉了约 60% 以上的概率质量。最合适的分解是：

```text
A = softmax(QK^T) = A_critical + A_marginal + A_negligible
O_hat = A_critical V + LowRank(A_marginal) V
```

其中 `A_critical` 用 tile/block sparse exact softmax 保留高秩运动与实体关系；`A_marginal` 用 linear/low-rank branch；`A_negligible` 跳过。误差满足一个直接的算子界：

```text
||(A - A_hat)V||_F
  <= (||A_marginal - A_lowrank||_2 + ||A_negligible||_2) ||V||_F
```

但扩散轨迹最终误差还会被后续 Jacobian 传播：

```text
||delta z_0|| <= sum_(layer,step) ||J_(layer,step -> z_0)|| ||delta_(layer,step)||
```

所以局部 L2 只能做筛选，最终必须用 rollout error、VBench 和运动一致性校准 refresh/controller。

## 5. 低秩在视频模型中是否有成功先例

答案是有，但成功方法几乎都满足“选对对象 + 替换计算 + 融合 kernel”，并非对任意权重做静态 SVD。

| 方法 | 低秩/稀疏对象 | 训练要求 | 作者报告的收益 | 对本项目的启示 |
|---|---|---|---|---|
| [SLA](https://arxiv.org/abs/2509.24006) | high-rank critical sparse + low-rank marginal attention | 少量 fine-tuning | Wan2.1-1.3B attention 13.7x，E2E 2.2x | 与本 oracle 最一致，首要 paper-faithful baseline |
| [VSA](https://arxiv.org/abs/2505.13389) | coarse-to-fine critical-token block sparsity | 可训练 | attention 6x，Wan 31s→18s | tile layout 和 85% FA3 MFU 比理论 sparsity 更关键 |
| [AdaSpa](https://arxiv.org/abs/2502.21079) | input/layer/head adaptive sparse mask；跨 step 缓存 | 免训练 | 作者报告显著长视频加速 | mask/LSE 的跨 step 稳定性可直接用于 cache-aware refresh |
| [Sparse-vDiT](https://arxiv.org/abs/2506.03065) | diagonal/multi-diagonal/vertical-stripe + head skip | 离线搜索 | Wan2.1 实际 1.58x | 视频 attention 有结构稀疏，不是随机 unstructured top-k |
| [Radial Attention](https://arxiv.org/abs/2506.19852) | 时空能量衰减的 `O(N log N)` mask | minimal LoRA tuning | 最多 1.9x | 低成本适配比硬套静态低秩更可靠 |
| [Attention Surgery](https://openreview.net/forum?id=KFXpfw5k1f) | softmax/linear token hybrid | 数 GPU-day 级适配 | Wan1.3B attention FLOPs -40% | linear attention 应按 token/branch 选择，不必全层替换 |
| [SVD-Cache](https://arxiv.org/abs/2601.07396) | 跨 timestep 平滑 principal feature subspace | 免训练/校准 | FLUX/HunyuanVideo 最多 5.55x | 低秩更适合预测平滑主子空间，而非拟合全部局部 defect |
| [SVDQuant](https://arxiv.org/abs/2411.05007) | 高精度低秩 outlier + 4-bit residual | 校准/量化 | 图像 DiT 上依赖 Nunchaku fusion | 证明 correction 不融合就可能完全没有速度收益 |
| [VideoMLA](https://arxiv.org/abs/2605.30351) | 训练强制的 low-rank latent KV cache | 重训练/蒸馏 | KV memory -92.7%，B200 throughput 1.23x | Wan 原始 `[W_K;W_V]` 并不低秩；瓶颈必须由训练塑造 |

VideoMLA 还给出一个重要反例：Wan 的 `[W_K;W_V]` 在 latent rank 192 时中位层只保留 45.8% 能量，所有层的 99%-energy rank 都超过 1300。它仍能成功，是因为训练把模型适配到新的 rank budget，而不是因为预训练权重天然低秩。这与我们的静态 Q-defect 谱平坦结论一致。

## 6. 最合理的下一版系统

### 6.1 Attention 主路径

1. 用低分辨率/coarse QK 生成 **block-tiled critical mask**，按 head/layer/prompt 自适应，不使用固定 FFN row mask。
2. 对 critical tiles 执行 exact sparse softmax，保留 motion boundary、主体和远程高秩关系。
3. 对 marginal tail 使用 rank-adaptive linear attention；只有 oracle 显示频谱/循环结构时，才比较 BCM basis 与普通低秩 basis。
4. 将 sparse 与 low-rank branch 融为一个 H200 kernel，复用 Q/K/V、LSE 和输出 accumulator，禁止 Python `.item()`、CPU sync 和独立 residual launch。
5. 缓存 mask、LSE、basis 与 routing，利用跨 denoising step 的稳定性；在 motion score、cache age、saturation 或 rollout-risk 超阈值时 refresh。

### 6.2 与既有 tri-mode runtime 的关系

外层仍可使用 `{D, Q, C}` trajectory-budget controller：

- `D`：BF16 dense anchor，刷新 reference 和风险观测；
- `Q`：fused FP8 sparse-linear recompute，刷新 cache age；
- `C`：复用/forecast 已有 attention 输出、mask 或 marginal basis。

三态互斥，避免在同一个 block-step 同时叠加 cache error 和新量化误差。attention kernel 是主执行器，controller 负责何时选择它，不应把两者混成一组无约束 trick。

### 6.3 执行优先级与停止门槛

1. **先做 F81 attention oracle Pareto**：扩展到多 layer、early/mid/late step、多个 prompt/seed，并测最终 rollout，而不是立即写通用 controller。
2. **实现 paper-faithful baseline**：SLA（低成本适配）与 AdaSpa/Sparse-vDiT 风格 block sparse（免训练），再比较 SageAttention/FA3 dense。
3. **先过 kernel gate**：F81 attention kernel 至少 3x；若表示 FLOPs 很低但 kernel 不超过 2x，立即转向 tile layout、fusion 和 occupancy，不再加算法模块。
4. **再过端到端 gate**：在多 prompt/seed 上 E2E 至少 1.3x，并用 VBench、运动一致性和 LPIPS/FVD 评价；dense-relative SSIM 0.98 保留为严格诊断指标，不作为唯一质量定义。
5. **停止条件**：若 per-sample oracle 在目标质量下也低于 1.2x，停止免训练高保真路线；若 oracle 高而 kernel 低，停止改 controller；若 kernel 高而 universal schedule 低，转向 sample-adaptive routing。

## 7. 最终判断

- “attention 应该是核心”在 F81/长视频上成立；F17 只能说 attention 是最大结构性目标之一，pointwise fusion 同样关键。
- 当前方法不奏效的根因是 **target、operator replacement、rank allocation 和 kernel fusion 四者同时错位**，不是简单的“视频模型不低秩”。
- 纯静态权重低秩对 Wan 缺乏谱依据；attention marginal tail、layer-specific cache defect 和跨 timestep principal subspace 仍有明确低秩机会。
- 最值得保留的不是现有独立 rank-16/BCM correction，而是“高秩关键稀疏路径 + 低秩边际路径 + 风险触发 refresh”的结构；这也是当前实验和公开论文共同支持的方向。

## 8. 证据与复现说明

- 可视化：`attention_lowrank_audit.png/.pdf`
- 图表原始数据：`profile_component_shares.csv`、`operator_speedups.csv`
- attention oracle：`raw/attention_lowrank_sparse_oracle_v1/`
- defect spectrum：`raw/defect_spectrum_summary.csv`
- 绘图脚本：`scripts/plot_attention_lowrank_audit.py`
- 完整远端来源和运行路径：`evidence_manifest.json`
