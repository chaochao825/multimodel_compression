# Wan whole-block sufficient-state 路线：历史证据、理论边界与 EXP-046

日期：2026-08-20
状态：`EXP-046 / G-025` 执行前分析；不包含正式 Gate 结果

## 1. 直接判断

“Wan 晚层计算可能存在低维、可递推的充分状态”是一个有条件合理的直觉，
但现有证据还不能支持直接训练 recurrent state student。过去的失败混合了五种
不同误差：

\[
E_{\rm total}
=E_{\rm representation}
+E_{\rm observability}
+E_{\rm transition}
+E_{\rm rollout}
+E_{\rm system}.
\]

`EXP-046` 只隔离第一项：允许看到目标 defect，问它本身能否由 rank 64 表示。
若这一最宽松上界失败，observer、transition 和 rollout 不可能挽救同一状态定义；
若它通过，也只说明值得继续测坐标可观测性，不能直接证明状态可递推。

## 2. 历史实验到底关闭了什么

| 证据 | 主要结果 | 被定位的瓶颈 | 没有否定什么 |
|---|---|---|---|
| EXP-000 scalar state | F81 H2 最强 target-visible scalar oracle `8.344%`，`0/192` cells 通过 | 标量/低阶时间系数的表示能力 | 条件化高维状态 |
| EXP-002 module target | H2 `14.305% -> 14.018%`；SA/FFN 仍为主要误差 | 不是漏算 AdaLN/timestep 语义 | 当前输入驱动的完整 block state |
| EXP-004 structure scout | target-leaking envelope `0.928%/1.994%` 但 deployable 仅 `1.043x` | sample-specific action risk 不可由便宜静态结构读出 | learned/full-observability routing |
| EXP-040 temporal rank 64 | adaptive capacity `1.380%/2.829%` | 单个 attention cell 的 rank-64 边界 | heterogeneous rank/dense fallback |
| EXP-041 heterogeneous rank | target-exposed `0.486%/0.874%`，46.02% optimistic work | 表示存在但 support/coefficients 不可执行 | target-free coordinate transport |
| EXP-043/044 finite jump | 修复采样后最优仍 `8.190%` | 局部 adapter 函数类不足，不只是 sampler starvation | released full few-step student、训练原生 state |
| EXP-045 current input | DPLR-16 output `7.348%`；75-shift oracle `5.53%-6.86%`；H3 失稳 | 当前输入只有部分 Jacobian 信息，旧 expert family 上限过低 | 目标 defect 本身是否低秩 |

这些结果共同反对“继续增加 DPLR rank/secant/shift 就会自然逼近”的路线，
但尚未回答一个更基础的问题：缺失计算是否存在紧凑表示。

### 2.1 跨路线 profile：失败并非彼此独立

更早的 attention、量化与结构化实验给出了同一个机制线索：主要瓶颈通常不是
参数或 rank 的绝对数量，而是低成本规则能否观测当前 defect 的动态坐标。

| 路线 | 最有信息量的结果 | 机制判断 |
|---|---|---|
| 固定 BCCB/分层 BCM | 参数/head 从 `2,184` 增至 `80,250`，误差仅从 `57.20%` 降至 `50.41%`；`530,070` 参数仍为 `50.47%` | 固定 Fourier eigenvectors 与内容相关 attention eigenvectors 错配，不是容量不足 |
| sparse critical + adaptive rank-16 | 单个 F81 attention cell 达到 `0.629%` aggregate、`1.85%` worst | 稀疏高秩主体加低维输出尾部存在表示 witness |
| 同一分解的 frozen basis | aggregate 退化到 `2.68%-2.76%`，worst `11.75%` | support 较可迁移，tail subspace 随内容旋转 |
| support-manifold shaping | Layer 14、25% support 加 adaptive rank-16 仍约 `2.04%/5.11%` | 简单移动规则 support 不能使难层 defect 自动低秩化 |
| positive/content-generated tail | rank `16 -> 64` 基本平台化；transductive capacity 仍为 `1.179%/3.385%`，frozen/proxy 为 `1.332%/1.488%` aggregate | 主要是函数类失配，不是 feature width 不足 |
| INT4 + rank/sparse correction | block 0 从 `3.029%` 降至 `2.555%`，block 24 从 `11.950%` 降至 `8.014%`；rank-64 + 5% sparse 仍为 `2.265%/7.112%` 且增加 `1.741B` ops | defect 含低维成分，但可修复能量、最终误差和可兑现速度不是同一件事 |

这条证据链解释了为何 `EXP-046` 不再尝试新的 BCM、Butterfly、shift bank
或 predictor：这些方法都同时混入了“表示是否存在”和“坐标能否生成”。本次先用
target-visible SVD 给表示能力一个最宽松且保守的判别；只有该判别通过，才值得重新
讨论 current-h observer、共享 basis、状态转移和训练原生 student。

## 3. 为什么 EXP-046 与 EXP-045 不重复

设完整 Wan block residual 为：

\[
r_{\ell,k}=F_\ell(h_{\ell,k},t_k,c).
\]

冻结 EXP-045 的因果 current-input diagonal renderer：

\[
\tilde r_{\ell,k}
=D_{\ell,k}r_{\ell,k-1}
+E_{\ell,k}(h_{\ell,k}-h_{\ell,k-1}),
\]

并在 H2/H3 中只从最后 exact anchor 连续展开。定义 endpoint defect：

\[
\Delta_{\ell,k}^{(H)}
=r_{\ell,k}-\tilde r_{\ell,k}^{(H)}.
\]

`EXP-045` 试图从历史和当前输入直接生成 correction；`EXP-046` 则先计算
target-visible 最优受限表示：

\[
\Delta_{\ell,k}^{(H)}\approx U_rV_r^T,
\qquad r\in\{8,16,32,64,96\}.
\]

两者回答不同问题：

- `EXP-045`：旧 observable/expert family 能否预测 correction？
- `EXP-046`：在不考虑坐标生成时，correction 是否至少有 rank-64 表示？

因此 rank-state PASS 不会推翻 EXP-045；它只会把下一瓶颈明确移动到
`U,V` 的 current-h observability 和跨样本旋转。

## 4. 算术上为什么值得先测

F17 latent grid 为 `N=5x30x52=7800`，hidden width `D=1536`。rank 64 时：

- factor payload：`r(N+D)=597,504` values，FP16 约 `1.14 MiB`；
- 冻结 state-render proxy：`2NDr=1.5335G`；
- 估计完整 block：
  `4ND^2 + 2NDM + 2N^2D = 475.206G` MAC；
- render/exact proxy：约 `0.323%`。

因此如果 rank-state 存在并且坐标可低成本生成，renderer 算术本身很便宜。
但这仍不是 latency 结论：factor 生成、HBM、kernel launch、融合和 fallback
可能主导真实成本，target-visible factor 也不允许保存成部署 payload。

## 5. 为什么选 whole block，而不是继续 attention-only

历史 H200 profile 显示：

- F81 self-attention 占 denoiser `53.88%`，因此长序列的主系统路线仍是
  FP8/BF16 dense 或已验证的 sparse attention；
- F17 self-attention 约 `21.81%`，linear GEMM `23.59%`，elementwise/memory
  约 `47.76%`，只加速 attention 的 Amdahl 杠杆有限；
- exact dual-H200 CFG parallel 已得到 `1.7743x` 且逐值一致，是系统基线；
- full-Wan batch-2 为 `1.952x/1.990x` cost，说明 LLM 式 speculative batch
  在当前 F17/F81 上几乎没有免费摊薄。

因此 F17 的 temporal state 若有意义，必须能替代 whole-block computation；
只预测 attention output 会留下 FFN、cross-attention 和 pointwise 主成本。

## 6. 与相关工作的正确关系

- [TaylorSeer](https://arxiv.org/abs/2503.06923) 与
  [L2P](https://arxiv.org/abs/2604.26365) 说明选择安全区域、学习逐 timestep
  predictor 和周期刷新可以有效；它们不意味着每个 Wan late block 都能统一
  线性展开。
- [Shortcut Models](https://arxiv.org/abs/2410.12557) 直接把 step size 作为
  训练条件，说明 aggressive skipping 往往需要改变训练分布，而非事后外推。
- [MeanFlow](https://arxiv.org/abs/2505.13447) 学习有限区间的平均速度，进一步
  支持“应训练可积分动力学对象”，但它是重新训练的生成建模方法，不是 frozen
  Wan block cache。

当前路线的潜在差异不在“首次使用 hidden state”，而在于先把 pretrained video
DiT 的 representation、observability、open-loop stability 与 H200 cost 分层认证，
再决定是否值得训练 native state。

## 7. G-025 的解释边界

Rank 64 必须在每个 target step、每个 H1/H2/H3 上至少覆盖 `6/10` 层，并同时满足：

- aggregate block-output relative L2 `<=0.5%`；
- worst identity/CFG branch `<=1%`；
- state-render arithmetic proxy `<=10%` estimated exact block。

结果映射：

- `PASS`：只开启 current-h coordinate-observability Gate；
- `BOUNDARY`：只有 rank 96 通过，不能自动扩 rank 挽救；
- `FAIL`：停止当前 renderer 下 rank-64 whole-block state；
- `INVALID`：dense 等价性、split、SVD 或完整性失败，不能形成科学结论。

无论结果如何，都不能外推到物理视频时间状态、训练原生 architecture、完整
few-step student 或所有视频 DiT。

还必须保留三项限制：每个 endpoint 的 SVD basis 独立拟合，不要求跨身份或跨步
共享；H2/H3 的低秩项只拟合最终 endpoint defect，并未作为中间状态递推；所有
block input 来自 dense teacher trajectory，而非近似 rollout 分布。因此 PASS 只
证明低秩 endpoint representation headroom，不等价于 Markov sufficient state。

## 8. 当前执行状态

- 冻结配置、12 个新身份、runner、分析器与可视化已实现；final 4 个身份锁定。
- synthetic rank recovery、随机 SVD 精度/单调性、Gate 完整性与原有 open-loop
  leakage tests 已通过远端环境。
- H200 smoke 只在共享锁释放且 GPU 空闲后执行；正式 selection 不自动串接。
