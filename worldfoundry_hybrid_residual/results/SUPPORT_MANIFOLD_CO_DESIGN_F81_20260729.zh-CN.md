# F81 Support-Manifold Co-Design Probe

日期：2026-07-29

模型与设置：World Foundry / Wan2.1-T2V-1.3B，F81 self-attention capture
状态：**注册的 rank-16 binary-support 前置门槛失败；高 payload 的 post-hoc weighted-support 容量点通过；部署与 H200 加速未验证。**

## 1. 核心结论

本轮实验没有证明一个可部署的新 Attention 加速方法，但将原命题拆成了三个清晰结论。

1. 在注册的 `12.5%/25%` 执行密度、adaptive rank-16 tail 下，固定 `64x64`、shifted tile、任意 `32x32` tile union、THW tile、motion-warp tile 和逐记录 support-family oracle 都无法通过 Layer 14 的 `0.5% aggregate / 1% worst-record` 前置门槛。
2. 更细的不规则 support 并非完全无效，但收益远低于成本。`hierarchical32` 相对 `fixed64` 使用约 `4x` kernel tile，25% 密度最坏误差仅从 `5.312%` 改为 `5.280%`；逐记录 family oracle 也只能到 `5.110%`。
3. 若允许 dense-AV 后验访问、adaptive rank-16，以及每个 64-query tile 最多 `1152` 个动态正权重，在 `56.25%` tile 密度上重新拟合 amplitude，三个被选中的 Layer-14 最坏记录可达到最大 `0.951%` output error。其 tile 算术上限为 `1.778x`，但这不是实测 H200 加速，也不是可泛化 router。

因此，最准确的方法状态是：

> **binary support shaping + payload-bounded rank-16 chart 尚未成立；support + dense-AV oracle amplitude + adaptive rank-16 的函数类存在容量，但需要先把 1152-scalar 后验 payload 压缩为可由 Q/K/V 预测的低维、可迁移表示。**

## 2. 为什么这轮实验必要

此前 restricted-rotation probe 已证明，在固定 support 下，任何 rank-16 rotation family 都不可能优于 fresh adaptive rank-16 SVD：

\[
\min_{R,C}\|D-RU_0C\|_F
\ge
\min_{\operatorname{rank}(L)\le16}\|D-L\|_F.
\]

因此 Layer 14 的约 5% worst error 不能靠增加 Givens、Householder、Butterfly 或 orthogonal BCM 因子解决。必须先改变 support，使

\[
D_{\Omega}=Y_{\mathrm{dense}}-Y_{\mathrm{sparse}}(\Omega)
\]

本身更容易被 rank-16 tail 表示。

本轮直接检验两个问题：

- 在相同执行预算下，更灵活且仍能映射到规则 GPU tile 的 support 是否能显著降低 `D_Omega` 的 intrinsic rank？
- 如果 binary support 不够，增加 tile amplitude 是否能形成质量与算术速度的交集？

## 3. 协议与证据边界

注册 screen 覆盖：

| 维度 | 设置 |
|---|---|
| 样本 | 4 个 sample，2 prompts x 2 seeds |
| CFG branch | `cond` |
| cell | Layer 0 / step 0 control；Layer 14 / step 0、9、19 |
| head | 12 |
| query tile | 每个 capture 3 个 `64-query` tile |
| density | 12.5%、25% |
| tail | 每条记录 fresh adaptive rank-16 |
| support family | fixed64、shifted64、hierarchical32、shifted32、thw8x8、motion_warp8x8 |
| 原始记录 | 8064 |

所有 capture、配置、merged artifact 均有 SHA256；两个 sample-disjoint H200 shard 在合并前检查精确样本覆盖和重复 record key。

必须强调：

- support、family choice、rank-16 basis 和 coefficient 都允许查看被评价记录的 dense `AV` defect，因此是 post-hoc capacity oracle。
- 连续权重实验只选择每个 Layer-14 cell 在注册 screen 中的一个最坏记录，不能代表跨 prompt/seed 泛化。
- H200 只承担数值 oracle；没有 fused sparse kernel、router、完整 rollout、VBench、SSIM 或 wall-clock 结果。
- 25% screen 的 support search 是单调启发式 witness，不是离散全局最优证书。

## 4. 注册 Binary-Support Screen

### 4.1 Layer 14 主结果

下表汇总 25% 密度下三个 Layer-14 cell 中的最大值：

| Family | Max aggregate error | Max worst error | Rank required for 1% worst gate | Tile multiplier |
|---|---:|---:|---:|---:|
| fixed64 | 2.081% | 5.312% | 51 | 1.00x |
| shifted64 | 2.084% | 5.231% | 51 | 1.00x |
| hierarchical32 | 2.117% | 5.280% | 48 | 4.00x |
| shifted32 | 2.129% | 5.240% | 49 | 4.00x |
| thw8x8 | 2.338% | 7.402% | 54 | 1.00x |
| motion_warp8x8 | 3.762% | 11.539% | 57 | 1.00x |
| per-record family oracle | **2.044%** | **5.110%** | **48** | up to 3.94x |

结论不是“tile 粒度完全无用”，而是“粒度收益不足以跨过 rank-16 门槛”。family oracle 已允许逐记录选择最有利 family，仍比 `0.5%/1%` gate 差约 4-5 倍。

### 4.2 Capacity control 排除实现错误

Layer 0 / step 0 的 25% control 明显更容易压缩：

| Family | Aggregate | Worst | Gate |
|---|---:|---:|---:|
| fixed64 | 0.096% | 0.396% | PASS |
| shifted32 | 0.096% | 0.396% | PASS |
| family oracle | 0.092% | 0.391% | PASS |

这说明 Layer-14 失败不是统一的代码符号、normalization 或 SVD 实现错误，而是中层 defect 的实际结构更复杂。

### 4.3 粒度与几何 insight

family oracle 在 25% 密度的选择分布显示：

- early cell 仍混合选择 fixed64、shifted64、hierarchical32 和 shifted32；
- middle cell 中 hierarchical32 + shifted32 占约 `85.4%`；
- late cell 中二者占约 `97.9%`；
- motion-warp 在三个 Layer-14 cell 中都没有成为最佳 family。

因此更细 support 在中后 step 有真实价值，但当前 motion/THW 构造没有对齐高杠杆 `V` 与内容相关位移。它不能被解释为“视频几何先验已经成功”。

### 4.4 增加 rank 也不能直接救回 25% 路径

在 25% family-oracle support 上，将 tail 从 rank-16 增到 rank-32 后：

| Cell | Rank-16 aggregate / worst | Rank-32 aggregate / worst |
|---|---:|---:|
| early | 2.044% / 5.110% | 0.835% / 2.291% |
| middle | 1.584% / 4.648% | 0.677% / 2.330% |
| late | 1.163% / 4.581% | 0.508% / 1.932% |

rank-32 仍不满足 `0.5%/1%`，而记录级 1% 门槛需要最高 rank `48`。这排除了“只把 rank 从 16 调到 32”作为低成本修复。

### 4.5 Dense fallback 不形成通用 schedule

若要求同一 family、同一 head 在所有测试 sample/step/query tile 上都通过 worst gate，则 Layer 14 的 12 个 head 全部至少失败一次。post-hoc fallback 只能把 12/12 heads 全部退回 dense，理想加速为 `1.0x`。

## 5. Continuous Support 与 Amplitude Refit

### 5.1 连续松弛说明容量存在，但权重场很稠密

对每个 Layer-14 cell 选择注册 screen 中最坏的 hierarchical32 记录，并在 25% weight-sum 约束下优化全部 2048 个候选 tile 权重：

| Cell | Binary rank-16 error | Full fractional error | Effective atoms / 2048 | Near-binary fraction |
|---|---:|---:|---:|---:|
| early | 5.280% | 0.321% | 1650.4 | 6.9% |
| middle | 4.648% | 0.165% | 1118.6 | 35.8% |
| late | 4.581% | 0.201% | 1088.1 | 36.2% |

连续函数类很强，但解广泛分布在候选 tile 上，并不是接近 25% binary mask 的松弛。这解释了直接 top-k 截断为何不稳定：删除大量小权重后，shared numerator/denominator 同时改变，原连续最优点不再成立。

### 5.2 固定 support 后重新拟合 amplitude

为避免把“截断后未重拟合”误判为函数类失败，本轮对每个密度保留 top-k support，并在 support、rank 和 normalization 不变的条件下重新拟合正权重 300 steps。

| Executed density | Early | Middle | Late | Max | Tile arithmetic upper bound |
|---:|---:|---:|---:|---:|---:|
| 50.00% | 1.288% | 0.949% | 0.472% | 1.288% | 2.000x |
| 56.25% | 0.951% | 0.426% | 0.231% | **0.951%** | **1.778x** |
| 62.50% | 0.708% | 0.179% | 0.188% | 0.708% | 1.600x |

56.25% 是细化网格中首次通过 `1% worst-record` 的点。这是本轮最重要的正面发现：**tile amplitude 与 rank-16 tail 的联合表示比 binary support 强得多。**

但该点仍需要：

- 每个 64-query tile 最多 1152 个后验 amplitude；
- 后验 support 与 amplitude 均查看 dense `AV`；
- fresh adaptive rank-16 output tail；
- 忽略 router、top-k、shared normalization、tail 和 kernel 调度开销的算术速度。

因此 verdict 只能是 `POSTHOC_WEIGHTED_SUPPORT_CAPACITY_GATE_PASSES`，不能写成质量或 H200 speed gate 通过。

### 5.3 直接在 50% 预算优化没有自动生成更好的 sparse support

进一步将连续 weight-sum 预算直接设为 50%，再 top-k + 300-step amplitude refit：

| Cell | 50% binary | 50% fractional | 50% refit top-k |
|---|---:|---:|---:|
| early | 2.807% | 0.156% | 1.627% |
| middle | 0.831% | 0.159% | 0.940% |
| late | 1.643% | 0.136% | 4.767% |

直接 50% 松弛虽然 full fractional 很低，但离散化后的 late support 更差。其原因是 weight-sum 松弛优化的是广泛重加权，不是 support 可离散性；更大的预算反而允许更多非二值质量分散。

这说明下一步不能简单“把 density 调到 50%”，必须显式优化 support 的可执行性与 amplitude 的低 payload 可生成性。

## 6. 理论解释

### 6.1 为什么 binary support 与 weighted support 差异很大

严格共享归一化下：

\[
\hat Y_i=
\frac{N_{\Omega_i}^{\mathrm{exact}}+\hat N_i^{\mathrm{tail}}}
     {Z_{\Omega_i}^{\mathrm{exact}}+\hat Z_i^{\mathrm{tail}}}.
\]

binary mask 只决定 `0/1` inclusion；weighted support 还能调整每个 tile 对 numerator 和 denominator 的相对贡献。对于中层非平稳 Attention，这相当于学习一个粗粒度 kernel correction，而不仅是选择 critical mass。

但 1152 个独立 amplitude 已接近通用拟合器。它证明容量，不证明低维流形，也不证明能从 Q/K/V 低成本预测。

### 6.2 为什么 BCM/rotation 仍不能进入主阶段

原注册 25% adaptive rank-16 pre-gate 仍失败，所以任何受限 rank-16 Grassmann rotation 都没有机会弥补 5% 级 residual。BCM、Toeplitz、wavelet 只能作为 localized geometry expert；在 support/tail 主容量门槛之前继续增加 expert 会混淆失败原因。

### 6.3 速度空间非常紧

56.25% 密度对应的 `1.778x` 只是 tile arithmetic upper bound。要达到实测 Whole-Attention `1.5x`，全部额外开销必须小于 dense attention 时间约 `10.4%`：

\[
T_{\mathrm{extra}}/T_{\mathrm{dense}}
< 1/1.5 - 0.5625
\approx 0.1042.
\]

这 10.4% 还要同时容纳 amplitude predictor、top-k、索引整理、rank-16 tail、shared normalization 和 kernel 效率损失。可行但余量很小，必须先做 payload-bounded transfer，再值得开发 fused kernel。

## 7. 与现有工作的边界

- [Scatterbrain](https://arxiv.org/abs/2110.15343) 已覆盖一般 sparse + low-rank attention，不能主张首次组合。
- [VSA](https://arxiv.org/abs/2505.13389) 已实现可训练硬件友好 tile router、约 6x attention 加速和 Wan 31s 到 18s；普通动态 tile 不是新贡献。
- [SLA2](https://arxiv.org/abs/2602.12675) 已覆盖 learnable sparse/linear routing、branch ratio 与 QAT；单纯加入 amplitude gate 与其高度重叠。
- [SALAD](https://arxiv.org/abs/2601.16515) 已说明少量适配的 sparse-linear tail 是现实路线。
- [Veda](https://arxiv.org/abs/2605.30325) 已将 tile selection 表述为 full-attention reconstruction，并结合 statistics-aware scoring、head-aware tiling、distillation 和 tile-skipping kernel。因此“重建 full attention 的 tile router”本身也不再构成创新。

仍可能站住的差异仅是：

> **在实测 latency 约束下，联合选择 support 与低 payload amplitude/tail chart，使剩余 AV defect 的 intrinsic dimension 最小，并给出跨 prompt/seed 的校准证书。**

## 8. 下一步 Go/Stop Gate

不应立即开发 H200 kernel。下一轮只做一个 payload-bounded transfer probe：

\[
w_x=\sigma\left(w_{c(x)}^0+\sum_{m=1}^{M}\alpha_m(x)B_{c(x),m}\right),
\qquad K\le4,\ M\le16.
\]

其中 chart、atoms 和 tail basis 只能从 calibration 学得；held-out 只允许 Q/K/V predictor 输出 chart ID 和最多 16 个坐标。support 由 `top-k(w_x)` 得到，所有分支使用严格共享 normalization。

继续条件：

1. 至少覆盖完整 `4 samples x 3 steps x 12 heads x 3 query tiles`，不能只测三个最坏记录。
2. calibration / validation / test 严格分离；test 不得选择 chart、basis、density 或 fallback head。
3. 在 density `<=50%` 时达到 aggregate `<=0.5%`、worst `<=1%`；56.25% 只作为容量控制。
4. predictor + routing + tail 的 H200 实测开销预算 `<=10%` dense attention。
5. 通过后才实现 fused kernel，并要求 Whole-Attention 实测 `>=1.5x`。

停止条件：

- `M<=16` 的 2/4-chart transfer 仍超过 `1%/2%`；
- 需要数百至上千 per-tile amplitudes 才能保持质量；
- 50% density 的 per-sample oracle 仍不能过 1%；
- fused cost model 已显示 Whole-Attention 上限低于 1.5x。

若停止，应转向 VSA/SLA2/SALAD/Veda 风格的少量适配与 fused sparse attention，并保留 FP8/BF16 dense fallback；BCM 仅保留在 localized expert 支路。

## 9. Artifact 与复现

主要 artifact：

- `results/support_manifold_oracle_f81_screen_v1_merged/`
- `results/support_manifold_oracle_f81_screen_v1_analysis_v1/`
- `results/continuous_support_relaxation_f81_worst_weighted_refit300_v1/`
- `results/continuous_support_relaxation_f81_worst_weighted_refit300_analysis_v1/`
- `results/continuous_support_direct50_f81_worst_refit300_v1/`
- `results/continuous_support_direct50_f81_worst_refit300_analysis_v1/`

关键图：

- `support_quality_kernel_tile_pareto.png`
- `support_rank_budget_curve.png`
- `support_intrinsic_rank_requirement.png`
- `continuous_support_density_tradeoff.png`

本轮代码使用 fresh-output guard、atomic artifact、SUCCESS hash、sample-shard exact merge、固定随机种子和显式 claim boundary。历史上被替代或不完整的运行均移入远端或本地 `trash/`，未覆盖正式 artifact。

## 10. 最终判断

这轮不是简单的失败，也不是方法已经成功：

- **失败部分**：25% binary irregular support 不能把 Layer-14 defect 塑造成 rank-16 可压缩流形；更细 tile、motion/THW geometry、family oracle 和 rank-32 都不足。
- **成功部分**：support amplitude 是此前缺失的自由度；56.25% post-hoc weighted support + adaptive rank-16 确有小于 1% 的容量点。
- **尚未解决**：如何用 `<=16` 坐标和当前 Q/K/V 生成可迁移 support、amplitude 与 tail，以及如何在 H200 上保住至少 1.5x Whole-Attention speedup。

所以最合理的创新主线应进一步收窄为：**payload-bounded support-amplitude-manifold co-design**，而不是继续堆叠更多 BCM、rotation 或任意不规则 mask。
