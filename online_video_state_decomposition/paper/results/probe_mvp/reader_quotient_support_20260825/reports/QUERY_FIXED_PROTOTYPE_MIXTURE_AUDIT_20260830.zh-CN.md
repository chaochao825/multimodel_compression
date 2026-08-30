# Query-fixed prototype mixture 容量审计

日期：2026-08-30
状态：exposed capacity diagnostic closed
判决：`NO_PROTOTYPE_MIXTURE_PATH`

## 1. 结论

本轮结果与“条件冗余存在，但 raw/shared state 不是正确接口”的理论一致，并进一步
排除了一个比 Gaussian moment 更强的函数族：

> 即使 writer 能看到当前样本的全部 K/V、逐 head 自适应地形成 32--128 个正值
> prototypes，并允许 target-visible local oracle 在 25% active-read 预算内展开 exact
> clusters，cluster mean 仍不能保存严格 reader interchangeability 所需的簇内
> query-value coupling。

最佳 eligible 配置是 K-only、128 prototypes 和 `oracle_local` support：

| 指标 | 结果 | 容量门槛 |
|---|---:|---:|
| active-read ratio | `4.002x` | `>=3.8x` |
| visual mean | `7.544%` | `<=0.5%` |
| visual P95 | `13.059%` | `<=1%` |
| visual worst | `13.834%` | `<=2%` |
| full-output mean | `0.861%` | `<=0.25%` |
| full-output P95 | `1.413%` | `<=0.5%` |

`0/12` candidates 通过 capacity 或 deployable Gate。按照冻结协议，不训练这个
prototype writer 或 selector，也不读取 calibration confirmation、official selection
或 official formal 数据。

## 2. 协议与有效性

- 数据仅为 exposed calibration positions `73--96`，共 24 个场景。
- 每个场景评估 Qwen2 layers `0/13/27`，共 72 个 sample-layer cells。
- writer 是 query-independent 的 K-only 或 scale-normalized K+V deterministic
  k-means；prototype counts 为 `32/64/128`。
- current query 只读取 prototypes；selected clusters 用 exact leaves 替换 prototype，
  numerator 与 denominator 始终共享归一化。
- 每个 head 的 active K/V budget 不超过 392，对应 dense 1,568 visual leaves 的 25%。
- `prototype_mass` 是可部署 selector；`oracle_local` 可见真实局部 AV defect，只用于
  函数族容量上界。
- 最大 attention replay error 为 `0`；全部 24 个样本完成，无数值或显存失败。

writer 的 k-means 成本、cold exact storage、reader accuracy、TTFT 与 latency 均未计入，
所以本轮即使通过也不能形成加速声明；当前结果只是更强的容量 null。

`oracle_local` 不是所有 cluster subset 的全局组合最优。为避免把 support optimizer
误差归因于 state，本轮另外运行了两个 post-hoc exposed-only sensitivity diagnostics；
它们不改变注册判决，也不构成数学不可能性证明。

## 3. 关键结果

### 3.1 增加 prototypes 只缓慢改善

| writer / selector | 32 | 64 | 128 |
|---|---:|---:|---:|
| K-only / oracle | `12.03%` | `9.10%` | **`7.54%`** |
| K-only / mass | `20.62%` | `16.50%` | `13.84%` |
| K+V / oracle | `20.45%` | `14.86%` | `11.97%` |
| K+V / mass | `29.04%` | `25.44%` | `22.48%` |

这里不是纯粹的 prototype 数不足。prototype 数从 32 增至 128 会消耗更多 active-read
预算，使 exact-token fraction 从约 `23.1%` 降到 `18.5%`；即使在这个公平预算下
允许 oracle 选择 exact clusters，误差仍比门槛高一个数量级。

### 3.2 K+V 欧氏聚类反而更差

K+V clustering 使用归一化 K 与 V 的拼接距离，但真实目标不是这个欧氏距离，而是：

\[
N_g(q)=\sum_{j\in g}e^{q^Tk_j}v_j,
\qquad
Z_g(q)=\sum_{j\in g}e^{q^Tk_j}.
\]

把 \(k_j=\bar k+\delta k_j\)、\(v_j=\bar v+\delta v_j\) 展开，有：

\[
e^{-q^T\bar k}\frac{N_g(q)}{|g|}
=\bar v
+\mathbb E[\delta v\,\delta k^T]q
+\frac12\mathbb E[v(q^T\delta k)^2]+\cdots.
\]

mean prototype 只保留第一项。K+V k-means 虽然让 V 更近，却没有保存
\(\mathbb E[\delta v\delta k^T]\)、query-direction curvature 或 reader-risk metric，
还会牺牲决定 softmax 权重的 K geometry。因此它系统性差于 K-only 并不矛盾。

### 3.3 mass 不是 downstream risk

在最佳 K-only/128 配置中，`prototype_mass` 相比 oracle 的 visual mean penalty 为
`1.835x`，worst 从 `13.83%` 增到 `58.55%`。在全部 row-level 数据上，selected
attention mass 与 visual error 的 Spearman 相关仅为：

- K-only oracle：`0.263`；
- K-only mass：`-0.098`；
- K+V oracle：`-0.076`；
- K+V mass：`-0.015`。

因此“读取更多 mass”既不是单调风险证书，也不是可靠的 exact fallback 排序目标。
遗漏的低 mass token 仍可能携带高杠杆 V，反之高 mass cluster 也可能被 prototype
较好表示。

### 3.4 深层误差更重

最佳候选在 layers `0/13/27` 的 visual mean 分别为 `3.18%/8.04%/11.41%`，
worst 分别为 `3.82%/9.66%/13.83%`。这与深层 token 表征更 task-conditioned、
简单相似性逐渐失去充分性的观察一致，而不是一个跨层共享平均原型可以解决的问题。

### 3.5 support optimizer 只解释小部分缺口

对最佳 K-only/128 state 又比较了两个 target-visible post-hoc selector：

| support optimizer | visual mean | P95 | worst |
|---|---:|---:|---:|
| prototype mass | `13.84%` | `41.89%` | `58.55%` |
| registered local defect | `7.54%` | `13.06%` | `13.83%` |
| forward residual greedy | `12.17%` | `22.03%` | `26.42%` |
| reverse exact-to-coarse greedy | **`6.16%`** | **`10.38%`** | **`11.03%`** |

reverse greedy 从 full exact 开始，按单位 token 节省造成的最小输出损伤逐步 coarse，
比 registered local oracle 的 mean 改善 `18.3%`。这证明 support interaction 不是零，
但最强结果仍是 `0.5%` 容量门槛的 `12.3x`，full-output mean/P95 也仍为
`0.916%/1.463%`。因此当前证据支持“主要缺口来自 unread-cluster state”，而不是
“support 已全局最优”；任意组合 oracle 未穷举，不能声称所有 positive mixture 在
数学上都不可能。

## 4. 与条件创新和热力学审计是否一致

一致，但要保持适用边界。

对 writer state \(S\)、query \(Q\) 与 dense reader output \(Y\)：

\[
R(\hat Y)
=\mathbb E\operatorname{tr}\operatorname{Cov}(Y\mid S,Q)
+\mathbb E\|\mathbb E[Y\mid S,Q]-\hat Y(S,Q)\|^2.
\]

自适应 prototypes 已显著扩大了 state 的函数族，但 mean K/V 仍留下很高的条件创新；
oracle support 只能降低其中少量 cluster 的创新，不能恢复未读 clusters 的交叉矩和
高阶 query response。这正是增加 rank、BCM 或 prototype count 不能凭空创造缺失条件
变量的一个 reader-side 实例。

Wan diffusion 中的 path-KL、noise metric 与 thermodynamic length 不能原样搬到
video understanding reader，因为这里没有去噪时间和概率流。但共同原则成立：

> 误差必须在最终任务诱导的 metric 下度量，并把昂贵 exact computation 当作一次
> 降低条件不确定性的测量；feature MSE、attention mass 或状态维数本身都不是风险。

因此该理论不是用来证明某种固定结构，而是用来选择运行时信息接口和 fallback 条件。

## 5. 与长视频理解工作的边界

[Quest](https://arxiv.org/abs/2406.10774) 已证明 query-dependent page selection 可通过
K min/max metadata 减少 exact KV 读取；本实验说明这种 metadata 在严格 OneVision
reader interchangeability 下不能兼作 tight output certificate。

[LongVU](https://arxiv.org/abs/2410.17434) 使用跨模态 query、帧间依赖和空间压缩，
[FrameFusion](https://arxiv.org/abs/2501.01986) 组合相似性 merging 与 importance
pruning；它们支持“相似性与重要性必须分开建模”，但其平均性能容忍度不能替代这里的
严格 paired reader-risk Gate。

[CacheFlow](https://arxiv.org/abs/2511.13644) 使用轻量 recurrent index，并在查询时
rehydrate cold exact K/V。它与当前结果共同支持“compact index + exact leaves”，但
本轮证明普通 K/KV centroid 不是足够的 compact state。新意不能是重新命名这种架构，
而必须来自 state、support 和 downstream risk 的联合目标。

## 6. 下一候选：Risk-certified Hierarchical Evidence Memory

不再把 node state 限制成 K/V mean。对 query-independent event node \(g\)：

\[
s_g=E_\theta(\{k_j,v_j,p_j,t_j\}_{j\in g}),
\]

query-conditioned reader 直接预测未展开 node 的 shared numerator/denominator 与风险：

\[
(\hat N_g(q),\hat Z_g(q),\hat U_g(q))=R_\theta(q,s_g).
\]

当前 support \(\Omega\) 的输出为：

\[
\hat A_\Omega(q)=
\frac{N_{\rm exact}(\Omega,q)+\sum_{g\notin\Omega}\hat N_g(q)}
     {Z_{\rm exact}(\Omega,q)+\sum_{g\notin\Omega}\hat Z_g(q)}.
\]

风险控制器不按 mass 排序，而按单位 read-byte 的校准风险下降：

\[
g^*=\arg\max_g
\frac{\hat U(\Omega,q)-\hat U(\Omega\cup g,q)}{C_{\rm read}(g)}.
\]

当 upper quantile 超过预算时逐层展开 semantic/event node，最终回退 cold exact leaves。
这保留最初目标中的三个成分：稳定 bulk quotient、校准 boundary risk、渐进 exact
fallback；但把 bulk state 从手工矩阵结构改为可学习的 conditional sufficient state。

## 7. 继续条件与停止规则

下一 Gate 只能作为新的、单独冻结的 learned-memory Gate，不能复用本轮
`NO_PROTOTYPE_MIXTURE_PATH` 来授权当前 prototype writer 训练。四个等预算 arm 为：

1. fixed-page Quest-style exact retrieval；
2. support-only：learned nodes，未读部分无 learned state；
3. state-only：fixed nodes + learned numerator/denominator state；
4. joint：learned nodes + state + risk path。

训练只使用 calibration positions `1--72`，开发只使用 `73--96`；一次性 confirmation
使用 `97--120`。只有 joint 在相同 active read、state bytes 和估计成本下相对最佳单组件
改善至少 `25%`，且达到 visual mean/P95 `<=1%/2%`、reader KL mean/P95
`<=0.01/0.02`、`0` harmful flips，才读取 official selection。否则关闭 learned
hierarchical memory，不再增加 prototypes、Gaussian rank 或固定 BCM。

## 8. 工件

- `analysis/.../query_fixed_prototype_mixture_exposed_v1/summary.json`
- `analysis/.../query_fixed_prototype_mixture_exposed_v1/prototype_measure_rows.csv`
- `analysis/.../query_fixed_prototype_mixture_exposed_v1/prototype_measure_summary.csv`
- `analysis/.../query_fixed_prototype_residual_greedy_diagnostic_v1/`
- `analysis/.../query_fixed_prototype_reverse_greedy_diagnostic_v1/`
- `figures/query_fixed_prototype_mixture.{png,pdf,svg}`
- `figures/prototype_mixture_frontier.csv`
- `figures/prototype_mixture_best_layer_summary.csv`
- `figures/prototype_mixture_selector_gap.csv`
- `figures/prototype_mixture_support_sensitivity.csv`
- `figures/prototype_mixture_mass_error_correlation.csv`
- `protocols/vsi_query_fixed_prototype_mixture_20260830.md`
