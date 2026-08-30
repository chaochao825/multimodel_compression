# 条件冗余、任务风险与渐进精确证据：VSI/OneVision 审计

日期：2026-08-30

状态：固定风险基、帧级取回、免训练标量控制器与低带宽 tiny risk controller 均已关闭；任务风险教师仅保留为训练原生 memory writer 的动机

## 直接结论

理论与历史实验是一致的，但需要把方法定位再收紧：

> 结构化拟合失败并不否定视频冗余；它说明稳定低维 bulk 只能作为记忆索引，不能替代随问题旋转的 task-sensitive innovation。下一条有依据的路线不是更大的 BCM/低秩基，而是风险监督的渐进精确证据取回。

本轮得到四个判决：

| 判决对象 | 结果 | 结论 |
|---|---:|---|
| 固定 reader-risk basis 的 CMRQ selection | `NO_GO` | 固定任务边界不跨问题稳定 |
| 1/2 个精确帧 oracle | `NO_GO` | 关键信息不是集中在少数整帧 |
| 同预算 target-gradient group oracle | 接近通过但仍 `NO_GO` | task metric 显著有效，但一阶、43.75% 预算仍有重尾错误 |
| query-score groups + scalar-margin fallback transfer | `NO_GO` | 免训练低带宽观测无法泛化，回退成本失控 |

因此当前不能做 formal、速度或论文正向结论。后续 prospective Gate 已进一步否定固定 quotient/query/residual metadata 上的 width-32 group-risk controller：它只把 top-98 recall 提升到 `30.91%`，并以 `75%` fallback 换得 `95.83%` delivered agreement。仍有潜力的方向必须改变存储接口，让 writer 在写入时主动生成 task-risk-observable innovation key，而不是继续扩大同一 controller。

## 与 EXP-004/005 的一致性

历史结果已经逐步定位了同一个 observability bottleneck：

| 实验 | 关键结果 | 对当前方法的含义 |
|---|---:|---|
| `EXP-004` past-residual-only | 最强 causal 方法仅约 `1.001x` AR(2) | 增大固定函数容量不能补回缺失条件变量 |
| `EXP-004` target-visible oracle | 晚层可达 `5.271x`，部分 step 更高 | 动态坐标存在，但推理时不可见 |
| `EXP-005` current-input diagonal field | 总体 `1.937x`，恢复 `0.877` oracle gap | 当前状态包含高价值信息，且逐通道 field 比标量/sketch 更有效 |
| `EXP-005` breadth gate | 仅 layers `21/24/25` 通过 | 条件冗余不是全层统一接口 |
| 本轮 target-gradient group risk | `95.83%` agreement，对比 residual 的 `70.83%` | 当前问题诱导的任务 metric 能定位高价值 innovation |
| 本轮 query/margin transfer | evaluation raw `58.33%`，fallback `66.67%` | 廉价标量观测不能稳定恢复该动态坐标 |

这组证据支持：

\[
\text{statistical redundancy}
\not\Rightarrow
\text{fixed low rank}
\not\Rightarrow
\text{cheap observability}
\not\Rightarrow
\text{deployable speedup}.
\]

## 理论统一

### 1. 条件风险分解

令完整视觉状态为 \(X\)，问题为 \(q\)，历史/元数据为 \(H\)，reader 决策为 \(Y\)。给定可观测状态 \(Z\) 的最优风险可写为：

\[
R(\hat Y)=
\underbrace{\mathbb E\,\operatorname{tr}\operatorname{Cov}(Y\mid Z,q,H)}
_{\text{条件创新下限}}
+
\underbrace{\mathbb E\|\mathbb E[Y\mid Z,q,H]-\hat Y(Z,q,H)\|^2}
_{\text{函数族/估计误差}}.
\]

增加 BCM block、rank 或 expert 主要降低第二项。若 quotient 丢失了决定当前任务边界的 innovation，第一项不会随函数容量下降。

本轮将视觉状态写成：

\[
X_g=\mu+Uz_g+e_g,
\]

其中 \(Uz_g\) 是稳定、低成本的 quotient bulk，\(e_g\) 是精确但冷存的 group innovation。实验说明 \(Uz_g\) 可以压缩通道状态，但不能直接决定应读取哪个 \(e_g\)。

### 2. 为什么任务风险优于欧氏误差

对 full reader 的 teacher margin \(m_c=\ell_y-\ell_c\)，省略 group \(g\) 的一阶风险为：

\[
r_g=
\max_c
\frac{
[-\langle \nabla_{X_g}m_c,\delta X_g\rangle]_+
}{
\max(m_c,m_0)
}.
\]

它同时编码了：

- 当前问题与选项；
- reader 的局部 Jacobian；
- 当前决策边界距离；
- 该 group 的真实 innovation 方向。

而 residual energy 只测 \(\|\delta X_g\|\)，query cosine 只测语义相似度。实际数据中，它们与 target risk 的 Spearman 相关分别只有 `0.0712` 和 `-0.0091`，top-98 overlap 都约 `29.3%`。这解释了为什么 raw L2 或 query 相似度无法稳定替代任务风险。

### 3. 热力学视角的有效边界

用户给出的 diffusion 路径 KL 与 drift metric 对 Wan 是合理背景：同一 feature MSE 在不同噪声阶段具有不同路径代价。但在当前视频理解 reader 中不存在去噪 SDE，不能声称物理 entropy production。

可迁移的核心不是“温度”，而是任务诱导 metric：

\[
G_q=J_{X\rightarrow\ell(q)}^\top H_\ell J_{X\rightarrow\ell(q)}.
\]

它对应 reader 决策边界上的 Fisher/GGN 风险。本轮 target-gradient group 的正面改善是这一观点的有限数值证据；它不是对任意视频任务的定理。

## 新实验结果

### 1. 固定 CMRQ selection：任务边界不共享

49 个冻结 selection 问题上：

| 方法 | mean KL | P95 KL | agreement | harmful |
|---|---:|---:|---:|---:|
| boundary mix | `0.002656` | `0.010470` | `93.88%` | `1` |
| permuted-risk null | `0.003007` | `0.007542` | `93.88%` | `1` |
| VSI PCA | `0.003031` | `0.008960` | `93.88%` | `0` |

mix 相对 permuted 的 paired mean delta 为 `-0.000351`，bootstrap 95% CI `[-0.001309, 0.000459]`，跨零。progressive fallback 只有 `6.12%`，但 delivered agreement 为 `97.959%`，略低于冻结的 `98%` 门槛，且 P95 未胜过 null/VSI。

因此固定 reader-risk basis 是有效 `NO_GO`。它支持“稳定 bulk 存在”，不支持“固定风险边界可泛化”。

### 2. 整帧精确取回：粒度错误

基础 7x7 pooling 将 token 保留率降到 `25%`。精确取回一个或两个完整帧：

| 方法 | token retention | mean KL | agreement | harmful |
|---|---:|---:|---:|---:|
| quotient pool | `25.00%` | `0.12060` | `54.17%` | `3` |
| oracle exact frame-1 | `34.38%` | `0.03554` | `83.33%` | `1` |
| oracle exact frame-2 | `43.75%` | `0.01696` | `87.50%` | `0` |
| residual-energy frame-2 | `43.75%` | `0.07556` | `66.67%` | `2` |
| query-score frame-2 | `43.75%` | `0.15394` | `62.50%` | `3` |

oracle KL 大幅改善但决策仍失败，说明答案证据跨帧、局部空间分散。整帧 retrieval 不是合适的硬件/语义粒度。

### 3. 规则 group 的任务风险：首次出现接近门槛的信号

每帧 196 token 按原序分成 49 个连续 4-token group。压缩态每组一个 quotient mean，98 个 group 恢复四个 exact token，token 保留率仍为 `43.75%`。

| 方法 | mean KL | P95 KL | agreement | harmful |
|---|---:|---:|---:|---:|
| quotient group means | `0.09696` | `0.57047` | `66.67%` | `1` |
| residual-energy groups | `0.07996` | `0.27885` | `70.83%` | `3` |
| query-score groups | `0.06351` | `0.29367` | `91.67%` | `1` |
| target-gradient risk groups | `0.04117` | `0.08305` | `95.83%` | `1` |

target risk 将 mean KL 相对 quotient 降低 `57.5%`，并把 mismatch 从 8 个降到 1 个。由于 24 个样本下 `95.83%` 仍低于 `98%`，且存在 1 个 harmful flip，冻结 gate 仍是 `NO_GO`。

这个结果的意义不是“梯度方法已成功”，而是首次证明 task metric 比 L2 和静态相似度更接近正确证据分配。

### 4. 免训练 query + scalar margin：不具备 transfer

前 24 个 calibration 问题用来按 mismatch margin 拟合阈值 `0.25`，后 24 个新 calibration 问题只评估一次：

| 指标 | fit | fresh evaluation |
|---|---:|---:|
| raw agreement | `91.67%` | `58.33%` |
| progressive delivered agreement | `100%` | `91.67%` |
| fallback rate | `58.33%` | `66.67%` |
| effective token retention | `76.56%` | `81.25%` |
| remaining harmful | `0` | `0` |

evaluation 上仍有 2 个 mismatch，且理想 reader token reduction 仅 `1.23x`。因此 scalar margin 没有给出稳定风险证书，不能进入 formal 或测速。

## 与领域工作的边界

现有工作已经覆盖多个相邻机制：

- [LongVU](https://arxiv.org/abs/2410.17434) 使用跨模态 query、帧间依赖与空间 token reduction；
- [FrameFusion](https://arxiv.org/abs/2501.01986) 组合相似度 merging 与 importance pruning，并报告 70% token reduction、`1.6-3.6x` 端到端加速和小于 3% 的平均性能影响；
- [StreamingTOM](https://arxiv.org/abs/2510.18269) 同时压缩 prefill token 和 4-bit online memory，按需检索相关组；
- [FlexMem](https://arxiv.org/abs/2603.29252) 已提出 training-free visual KV memory、双路径压缩和任务相关读取。

因此不能主张：首次 query-aware compression、首次 progressive retrieval、首次 dual-path memory，或首次 exact fallback。

仍有潜力的差异是：

> 不用 attention mass、相似度或 reconstruction error 定义 evidence value，而用 full reader 的 adverse decision-boundary risk 作为教师；运行时从 quotient、低比特 innovation sketch 和 query 预测每组风险上界，并在累计遗漏风险低于当前 margin 时停止。

这把“压缩多少”变成一个带可校准 task-risk 约束的 rate-distortion 问题：

\[
\min_{\Omega} C_{\mathrm{reader}}(\Omega)
\quad
\text{s.t.}\quad
\Pr\left[
\sum_{g\notin\Omega}r_g
\le \hat m-\epsilon
\right]\ge 1-\alpha.
\]

## 建议的核心方法：Risk-Certified Quotient Memory

### 运行时结构

1. Stable quotient bulk：常驻 \(z_g=U^\top(X_g-\mu)\)，用于低成本语义索引。
2. Cold exact innovation：精确 \(e_g\) 按规则 group/tile 冷存，不默认进入 reader。
3. Tiny risk controller：输入 query、quotient、位置和 4-16 bit residual sketch，输出 \(\hat r_g\) 与不确定度。
4. Progressive exact read：按风险/成本比取回 exact groups，保持规则 gather 和连续 token 布局。
5. Calibrated stop/fallback：只有当遗漏风险上界低于 compressed reader margin 时停止，否则继续读取或 full fallback。

### 训练目标

full reader 只在 calibration 生成 teacher：

\[
r_g^*=
\max_c
\frac{[-\langle\nabla_{X_g}m_c,\delta X_g\rangle]_+}
{\max(m_c,m_0)}.
\]

controller 不拟合 feature，而拟合排序和漏检风险：

\[
\mathcal L=
\mathcal L_{\mathrm{rank}}(\hat r,r^*)
+\lambda_1\mathcal L_{\mathrm{tail}}
+\lambda_2\mathcal L_{\mathrm{coverage}}.
\]

使用独立 calibration split 对 \(r_g^*-\hat r_g\) 做 conformal/quantile 校准，得到上界 \(u_g\)。运行时使用 \(\sum_{g\notin\Omega}u_g\) 而不是一个脆弱的 scalar margin threshold。

### 为什么它保留了最初结构化动机

- 低秩保留为稳定 bulk，而不是强行解释 task-sensitive tail；
- 稀疏保留为 exact innovation 的规则 group/tile；
- residual 不再事后相加，而决定渐进取回；
- 风险 metric 决定怎样叠加，而不是把 Q/S/L/Hessian 机械堆叠；
- BCM/BCCB 可作为离线编码或局部索引布局，但不能再充当跨问题 reader-risk basis。

## Prospective Gate 结果与下一边界

冻结的 tiny controller Gate 已执行并有效 `NO_GO`：

| 指标 | prospective | gate |
|---|---:|---:|
| controller top-98 recall | `30.91%` | 必须胜过 proxy，已满足 |
| delivered agreement | `95.83%` | `>=98%`，失败 |
| harmful | `0` | `0`，满足 |
| fallback | `75.00%` | `<=15%`，失败 |
| effective retention | `85.94%` | `<=53%`，失败 |
| task accuracy loss | `0 pp` | `<=1 pp`，满足 |

验证最大 mismatch margin 为 `0.625`，prospective 仍出现 margin `0.75` 的 mismatch，因此 scalar compressed margin 也不是可迁移证书。selection/formal 保持未读。

下一步没有继续调 controller width、sketch dimension、BCM block 或 fallback threshold，而是在 fresh calibration positions 73--96 上测试了 risk-observable writer。learned-writer-only、learned-controller-only 与 joint writer-controller 的 prospective recall 分别为 `30.02%/30.23%/30.70%`；joint reader agreement 为 `70.83%`，同预算 target-gradient oracle 也只有 `91.67%`。因此 follow-up 同样 `NO_GO`，完整判决见 `RISK_OBSERVABLE_WRITER_AUDIT_20260830.zh-CN.md`。

## 结论边界

当前支持：

- 视频理解状态中存在稳定 bulk 与任务条件 innovation；
- reader-induced risk 比欧氏误差更能定位高价值精确证据；
- 整帧取回、固定风险基和标量 margin 均不是充分接口；
- task-risk teacher 有 oracle headroom，但当前低带宽 metadata controller 无法兑现该 headroom；
- 稳定 bulk 可作为索引，不能被称为任务充分统计量。

当前不支持：

- formal generalization；
- 端到端准确率保持；
- reader 或系统 wall-clock 加速；
- diffusion 热力学对视频理解 reader 的物理解释；
- Risk-Certified Quotient Memory 已优于 LongVU、FrameFusion、StreamingTOM 或 FlexMem。

可视化与原始数据：

- `figures/progressive_evidence_gates.{png,pdf,svg}`
- `figures/progressive_evidence_gate_metrics.csv`
- `figures/group_proxy_correlations.csv`
- `figures/tiny_group_risk_controller_gate.{png,pdf,svg}`
- `figures/tiny_group_risk_controller_{selector,reader,gate}_metrics.csv`
- `figures/risk_observable_writer_gate.{png,pdf,svg}`
- `figures/risk_observable_writer_{selector,reader}_metrics.csv`
- `analysis/onevision_reader_quotient_stage_a_20260830/`
