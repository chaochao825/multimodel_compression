# F81 Content Tail 的 ICLR 价值与后续异构认证路线

## 直接判断

当前结果有研究价值，但不足以单独支撑一篇 ICLR 主会论文。它最适合作为
完整论文中的关键否定实验：当前 positive sparse-linear tail 的主要瓶颈是
所测试函数类与 Layer 14/step 9 内容相关 attention 的失配，而不是 feature
rank 不够。

这一结论必须限制在以下证据范围：

- Wan2.1-T2V-1.3B，F81；
- `layer 14 × step 9 × conditional CFG`；
- 4 个此前探索过的 capture；
- 每个 capture 的 3 个 query tile；
- 25% exact support、positive separable Q/K feature map 和所测试 layout family。

因此不能外推为“Wan 不可压缩”“视频 DiT 的低秩无效”或“所有
sparse-linear attention 都失败”。

## 同 Test 的正确误差核算

原报告中的：

\[
1.179\%,\quad 1.332\%,\quad 1.488\%
\]

分别来自 transductive 全样本 aggregate、frozen test 和 proxy test，不能直接
相减。此外 relative L2 本身也不是可加误差量。

在完全相同的 test capture、route granularity 和 per-tile adaptive rank-16
定义下，rank-64 结果为：

| 阶段 | Per-tile aggregate | Worst tile |
| --- | ---: | ---: |
| Transductive capacity + dense-AV support | 1.3253% | 3.2292% |
| Calibration-frozen + dense-AV support | 1.3317% | 3.2338% |
| Calibration-frozen + validation-selected QKV proxy | 1.4878% | 3.6316% |

以最终 proxy 的平方 aggregate error 为分母，做描述性 accounting：

| 来源 | 最终 proxy 平方误差占比 |
| --- | ---: |
| 当前函数类容量 floor | 79.3% |
| calibration 到 test 的冻结增量 | 0.8% |
| proxy routing 增量 | 19.9% |

这比原来的跨 split 对比更有力地支持“容量失配主导”。但该分解仍不是因果
分解或数学不可能性证明：support search 是 monotone projected-rank heuristic，
不是所有组合 support 的全局最优；transductive 训练也不保证找到函数类全局
最优点。

## ICLR 完整度

| 维度 | 当前状态 |
| --- | --- |
| 实验协议 | 强：门槛预注册、split 无泄漏、oracle/proxy 分离 |
| 否定证据 | 强：rank 16 到 64 平台，函数类扩容收益耗尽 |
| 正向算法 | 尚缺：没有通过质量门槛的新 attention 方法 |
| 覆盖范围 | 不足：一个 layer/step/CFG cell |
| 实际速度 | `0x` 可声明：没有 fused kernel 或完整 rollout |
| 论文潜力 | 取决于全模型认证图谱、异构执行和 H200 落地 |

现有正向工作也说明，不能把本结果扩展为对 sparse-linear 路线的普遍否定：

- [VSA](https://arxiv.org/abs/2505.13389) 使用可训练 coarse-to-fine tile
  router 和单一可微 kernel，论文报告 Wan attention `6x`，端到端约
  `31s -> 18s`。
- [SLA](https://arxiv.org/abs/2509.24006) 使用可微调 sparse critical 与
  linear marginal 分支及融合 kernel，报告 attention `13.7x`、端到端 `2.2x`。
- [SALAD](https://arxiv.org/abs/2601.16515) 使用轻量 linear branch 与
  static-dynamic scaling，报告 `1.52-2.03x` 推理加速，训练少于 1,600 step。
- [SLA2](https://arxiv.org/abs/2602.12675) 进一步采用 learned routing、
  branch ratio 和 QAT，报告 attention `18.6x`。

这些数字来自不同模型配置、序列长度、硬件、训练预算和质量协议，不能直接
与本项目的 strict dense-relative gate 横向排名。它们支持的结论是：强正向
结果通常依赖可学习 router、改变函数类和融合 kernel，而不是继续扩大当前
免训练 positive-linear feature rank。

## Amdahl 边界

若 self-attention 占 denoiser `53.88%`，则：

\[
S_{\rm e2e}=\frac{1}{(1-0.5388)+0.5388/S_{\rm attn}}.
\]

当前 `3.94x` 只是单算子的算术上界，并且该 cell 已被分配到 dense fallback。
即便假设局部路径完全兑现：

| 可加速 attention 比例 | Whole-Attention | Denoiser 上限 |
| ---: | ---: | ---: |
| 100% | 3.94x | 1.67x |
| 43.1% | 1.47x | 1.21x |
| 25% | 1.23x | 1.11x |

router、fallback、permutation 和未融合 launch 会进一步降低实际速度。因此
进入系统论文主结果的最低门槛应保持为：H200 Whole-Attention `>=1.5x`，
端到端至少 `>=1.3x`；与强 baseline 正面对比时，应争取端到端 `>=1.5x`。

## 最合理的论文主线

下一主线应从统一近似改为：

> **Oracle-certified heterogeneous attention：对每个
> layer × step × head × CFG cell 认证可压缩性，再在 learned sparse-linear、
> FP8 dense 和 BF16 fallback 之间选择，以轨迹风险和 H200 实测时延优化。**

这里的创新不能只是一个 heuristic gate。至少需要：

1. 将总误差分为函数类容量、跨样本迁移、router、量化、kernel 数值和轨迹
   传播六部分，并在相同样本和 granularity 上核算。
2. 先建立 per-cell oracle Pareto frontier，证明异构策略严格优于统一 sparse、
   统一 linear 和统一 FP8。
3. 使用 trajectory-weighted risk，而不是仅局部 `AV` rel-L2；高风险 cell
   自动回退 BF16。
4. 将认证器校准成 coverage-aware 风险上界，报告误接受率和 fallback 率。
5. 在 H200 上以真实 fused kernel latency 作为动作成本，禁止用 FLOPs 代替。
6. 完成多 prompt、多 seed、F17/F81、VBench/PSNR/SSIM 和完整 wall-clock。

BCCB 只保留在 localized head 的 shift/router candidate 中。频带能量稳定最多
支持 coarse feature；只有固定相位、位移平稳性和边界误差也同时通过，才允许
BCCB 直接参与输出近似。

## 下一注册实验

下一步不是立即训练新 tail，而是建立低成本 certification atlas：

- 第一阶段在线统计代表性 `layer × step × head × CFG` cell，不落盘完整 QKV；
- 对每个 cell 比较 BF16 dense、FP8 dense、25% sparse oracle、当前 tail oracle
  和可学习 sparse-linear upper bound；
- 同时记录 local error、trajectory-weighted error、fallback rate 和 H200 latency；
- 只有 oracle 证明 Whole-Attention `>=1.5x` 且质量预算可行，才实现 runtime；
- 若可压缩 cell 覆盖率不足，则停止异构 attention 主线，转向 fused FP8、NFE
  和 exact system optimization。

当前实验的最佳定位是高质量止损证据，以及上述认证框架中的第一个
`DENSE_REQUIRED` cell，而不是独立的方法贡献。
