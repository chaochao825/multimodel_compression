# Query-conditioned measure 与 progressive exact memory 综合判决

日期：2026-08-30
状态：exposed diagnostic closed；下一 learned-memory Gate 待接受
范围：OneVision reader exposed positions `73--96`，以及 Wan `EXP-004/005` 的已验证结论。

## 1. 结论

新的实验与“条件创新下限 + 函数族误差”的理论是一致的，但它们把可行方向收得比
原先更窄：

> 条件冗余存在；失败的是把它编码成规则 page 上的 frozen Gaussian/moment state，
> 或只靠 attention mass 选择 exact pages。下一步必须学习语义节点、value-coupled
> innovation 和下游风险，不能再增加 BCM、moment rank 或 hand-designed selector。

这不是“结构化方法全部失败”。它是两个清楚的 scoped null：

- `NO_POSITIVE_GAUSSIAN_MEASURE_PATH`：严格正的 Gaussian closure 在最有利
  target-visible support 下仍远离 `1%/2%`；rank 越大越差。
- `NO_PROGRESSIVE_EXACT_PAGE_PATH`：只读取 exact K/V 的规则 page 即使覆盖
  `94.40%` attention mass，visual mean error 仍为 `3.37%`。

两者均为 fixed-query、single-layer、exposed calibration diagnostic；它们不建立任务
精度、TTFT、端到端延迟或 Wan 视频质量结论。

## 2. 与 Wan 历史结果的统一解释

对目标 `Y`、历史状态 `H` 和运行时可观测量 `Z`：

\[
R(\hat Y)
=
\mathbb E\operatorname{tr}\operatorname{Cov}(Y\mid Z,H)
+
\mathbb E\|\mathbb E[Y\mid Z,H]-\hat Y(Z,H)\|^2.
\]

`EXP-004` 增加 past-residual predictor 容量几乎无效，说明它主要没有改变第一项；
target-visible late-layer oracle 很强，说明冗余并未消失。`EXP-005` 加入 current-input
channel field 后达到 `1.937x` risk improvement，并恢复 `0.877` oracle gap，但仅
layers `21/24/25` 通过连续层 Gate。这表明相关坐标存在于高带宽 current state，尚未
形成跨层稳定、低成本接口。

reader probes 得到同一种机制边界：query-aware headwise support 显著优于 shared
support，但 low-order state 不足；current-support marginal 有信息，但路径不单调；
positive Gaussian closure 与 exact page retrieval 又分别排除了“更高 moment rank”和
“只读高 mass pages”两条简单修复。

因此统一解释不是“模型没有冗余”，而是：

\[
\text{冗余的条件坐标}
\quad\text{尚未被压成}\quad
\text{稳定、低带宽、闭环安全的接口}.
\]

## 3. 为什么 Gaussian state 与 exact mass 都失败

### 3.1 Gaussian closure 丢失联合高阶结构

softmax memory 需要同时估计：

\[
Z(q)=\mathbb E[e^{q^TK}],
\qquad
N(q)=\mathbb E[e^{q^TK}V].
\]

key 的 mean/covariance 只有在合适的单峰分布假设下才能闭合 `Z`，更不能一般地闭合
`N`。真实节点包含多模态 key、重尾 score 和 query-dependent value coupling。
Gaussian covariance 被指数 MGF 放大，因此 rank `0/2/4/8/16` 在 `25%` exact pages
下从 `15.50%` 依次恶化到约 `19.83/46.18/62.26/70.98%` visual mean error。

### 3.2 attention mass 不是 AV error certificate

令 `S` 为已读 exact pages，`A_S` 为在 `S` 上重新归一化的 output，则：

\[
A-A_S
=
\sum_{j\notin S}a_j(v_j-A_S).
\]

tail mass 小并不足够；遗漏 token 的 value leverage 也必须小。实验中 exact-mass
oracle 在 `75%` pages 上覆盖 `94.40%` mass，visual mean 仍为 `3.37%`，正是该项的
直接证据。

### 3.3 worst-case box 随维度失去用途

Quest 风格 coordinate min/max box 对 page score 是有效上界，但它把不同 key 维的
极值当作可同时出现，忽略协方差和低维流形。`25%` exact pages 下 bound looseness
从 layer 0 的约 `10^5.62` 增长到 layer 27 的 `10^16.34`。因此它可做安全排序元数据，
却不能在本模型中提供有用的 early-stop 风险证书。

## 4. 核心下一方法：Value-coupled Hierarchical Evidence Memory

只保留一个候选接口。视频先被划分为可学习的 event/semantic nodes，而不是固定
`7x7` pages。每个 node `g` 保存：

\[
s_g=E_\theta(X_g),
\qquad
c_g=C_\theta(q,s_g,p_g,m_g),
\]

其中 `s_g` 是对 value-coupled innovation 训练的 compact state，`c_g` 同时预测
query relevance 与 adverse reader risk。原始 tokens 仍作为 cold exact leaves。
对嵌套 support `Omega_b`，输出使用共享 numerator/denominator：

\[
\hat A_b
=
\frac{N_{\rm exact}(\Omega_b)+\hat N_\theta(\bar\Omega_b,q)}
     {Z_{\rm exact}(\Omega_b)+\hat Z_\theta(\bar\Omega_b,q)}.
\]

训练目标直接约束路径，而不是假设 split 自动改善：

\[
\mathcal L
=
\sum_b w_b D_{\rm reader}(p_{\rm dense},p_b)
+\lambda\sum_b[D_{b+1}-D_b+\epsilon]_+
+\eta\,\ell_{\rm quantile}(\hat U_b,D_b)
+\beta C_{\rm measured}(b).
\]

运行时若校准 upper quantile `U_b` 超过剩余风险预算，就展开下一个 semantic node；
仍不安全则读取 exact leaves 或 full fallback。这个方法同时解决了四个已测瓶颈：

- event nodes 减少固定 page 跨语义边界造成的重尾 tail；
- learned state 不再假设 Gaussian/低阶 moment closure；
- value-coupled loss 不再把 attention mass 当 output error；
- path loss 与 quantile fallback 直接处理 frozen reader 中的非单调 refinement。

## 5. 与相关工作的边界

[Quest](https://arxiv.org/abs/2406.10774) 已覆盖 query-dependent page metadata 与
exact KV retrieval；[QTSplus](https://arxiv.org/abs/2511.11910) 已覆盖语义压缩；
[MemoryCard](https://arxiv.org/abs/2606.05917) 已覆盖 event/topic memory；
[S3-Attention](https://arxiv.org/abs/2601.17702) 也提醒原型 retrieval 可能慢于优化后的
full KV。因此“query-aware event memory + fallback”本身不够新。

唯一值得验证的独立主张是：

> **以 value-coupled downstream innovation 联合优化 semantic node、compact state
> 和 exact support，并用 reader-risk path consistency 校准 progressive fallback。**

若 joint 方法在相同 read/state/latency 预算下不能显著胜过 support-only、
re-encoder-only 和 Quest/QTSplus 风格 baseline，就应把该路线判为增量拼接并关闭。

## 6. 下一 Gate

基础 OneVision vision encoder、QKV 和 LLM 全部冻结，只训练 node constructor、tiny
re-encoder、support scorer 和 risk quantile head。数据边界固定为：

- train/calibration：positions `1--72`；
- model selection：positions `73--96`；
- formal：positions `97--120`，selection 通过前禁止读取。

同预算比较四组：fixed-page Quest baseline、support-only、re-encoder-only、joint。
event nodes 与 fixed pages 必须具有相同 active-token/read-byte budget。selection Gate：

- visual-output mean/P95 `<=1%/2%`；
- reader KL mean/P95 `<=0.01/0.02`；
- `0` harmful top-1 flips；
- joint 相对最佳单组件至少 `25%` 改善；
- compact state、routing 和 exact reads 的总算术/字节成本低于 dense visual memory 的
  `35%`。

任何一项失败都不读取 formal，也不开发 kernel。selection 全部通过后，formal 只运行
一次；随后才在 H200 上测完整 operator，而不是用 active-read proxy 声称加速。

## 7. 工件

- `analysis/.../query_fixed_positive_gaussian_exposed_v1_repair4/`
- `analysis/.../query_fixed_progressive_exact_pages_exposed_v1/`
- `figures/query_fixed_positive_gaussian_measure.{png,pdf,svg}`
- `figures/query_fixed_progressive_exact_pages.{png,pdf,svg}`
- `protocols/vsi_query_fixed_positive_gaussian_measure_20260830.md`
- `protocols/vsi_query_fixed_progressive_exact_pages_20260830.md`
