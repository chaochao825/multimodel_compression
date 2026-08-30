# Tiny Group-Risk Controller 前瞻性审计

日期：2026-08-30

状态：`NO_GO`；关闭当前 quotient/query/residual-metadata controller 家族；后续 risk-observable writer Gate 亦为 `NO_GO`

## 直接结论

本轮结果与“条件冗余而非固定结构”的理论判断一致，但否定了当前最小实现：

> 稳定 quotient bulk 中确实保留了一部分任务信息，低比特 residual metadata 也能让小控制器略好于随机和静态 proxy；但这些低带宽观测不足以恢复随问题旋转的 adverse decision-boundary risk，更不能形成可迁移的 fallback 证书。

controller 在 prospective 上将 mean candidate KL 相对 residual/query selector 降低 `31.6%/37.5%`，top-98 recall 也从 `28.61%/28.32%` 提升到 `30.91%`。这排除了“完全没有可学习信号”。然而 controller 仍漏掉约 `69%` 的 top-risk groups；为清零 harmful case，验证集校准阈值导致 prospective `18/24` full fallback，最终保留 `85.94%` token，只剩 `1.16x` 理想 reader token reduction，并仍有 `1/24` mismatch。

因此不能进入 selection、formal、速度或部署实验，也不能把结果写成视频状态没有冗余。被关闭的是：

- 冻结 rank-456 quotient；
- 9-scalar/group residual metadata；
- 固定 8D quotient/query/residual sketches；
- width-32 MLP group selector；
- compressed top-1 scalar margin fallback。

## 冻结协议与无泄漏顺序

72 个 calibration-only 问题在运行前固定为：

- first 24：controller training；
- next 24：epoch selection 与 fallback threshold calibration；
- following 24：只读一次的 prospective endpoint；
- selection/formal：保持未读。

视觉状态使用 8 帧、每帧 196 token，共 1,568 token。每 4 个连续 token 构成一个规则 group，共 392 groups；每组常驻一个 quotient mean，并恢复 98 个 exact groups，对应无 fallback 时 686 token、`43.75%` retention。

teacher 使用 full reader 的一阶 adverse margin risk：

\[
r_g^*=\max_c
\frac{[-\langle\nabla_{X_g}m_c,\delta X_g\rangle]_+}
{\max(m_c,0.05)}.
\]

运行顺序经过日志复核：train/validation gradient capture 完成后训练 controller；validation reader 完成并冻结 threshold `0.625` 后，才首次读取 prospective。没有 prospective label、gradient 或 reader outcome 进入训练、epoch selection 或阈值校准。

## 数值结果

### 1. Selector 学到弱而稳定的信号

| selector | validation recall | prospective recall | prospective risk mass |
|---|---:|---:|---:|
| query cosine | `28.57%` | `28.32%` | `26.66%` |
| residual energy | `28.66%` | `28.61%` | `32.08%` |
| tiny controller | **`31.08%`** | **`30.91%`** | **`32.55%`** |

由于每个样本选择 98/392 groups，随机 recall 为 `25%`。controller 的 prospective lift 只有 `1.236x` random。validation 到 prospective 几乎不掉，说明主要问题不是普通 train/validation overfit，而是绝对可观测信息不足。

### 2. Soft output 改善，但决策边界未被控制

| prospective selector | mean KL | agreement | harmful |
|---|---:|---:|---:|
| query cosine | `0.06677` | `83.33%` | `1` |
| residual energy | `0.06106` | `83.33%` | `3` |
| tiny controller | **`0.04174`** | `83.33%` | `1` |

controller 明显改善 soft candidate distribution，却没有改善 top-1 agreement。这是“平均失真下降”与“adverse boundary crossing”分离的直接证据，也说明只优化 group BCE/recall 不能替代端点决策约束。

### 3. Scalar margin 不是可迁移证书

validation controller 的 9 个 mismatch margin 为：

`[0, 0.125, 0.125, 0.25, 0.375, 0.375, 0.5, 0.625, 0.625]`。

按冻结规则取最大值 `0.625` 作为 fallback threshold。prospective 的 4 个 raw mismatch margin 为：

`[0.125, 0.375, 0.375, 0.75]`。

前三个被 fallback 修复，但 margin `0.75` 的错误穿过证书。因此阈值既不安全又过度保守：

| progressive endpoint | actual | registered gate |
|---|---:|---:|
| delivered agreement | `95.83%` | `>=98%` |
| remaining harmful | `0` | `0` |
| fallback rate | `75.00%` | `<=15%` |
| effective token retention | `85.94%` | `<=53%` |
| task accuracy loss | `0 pp` | `<=1 pp` |
| controller MACs | `501,760` | `<1,000,000` |

`23/24` delivered agreement 的 Wilson 95% interval 约为 `[79.76%, 99.26%]`；样本数也不足以支持高可靠性声明。

## 与历史结果的统一解释

| 证据 | 排除的解释 | 保留的解释 |
|---|---|---|
| 固定 CMRQ selection `NO_GO` | 跨问题共享固定 risk basis | bulk 可共享，边界随问题旋转 |
| 1/2 exact-frame oracle `NO_GO` | 证据集中在少数整帧 | 高价值 innovation 空间分散 |
| gradient-risk groups 达 `95.83%` | 欧氏能量足以定义价值 | reader-induced metric 更接近任务价值 |
| query/margin transfer `NO_GO` | 标量相似度与 margin 可认证 | 需要 groupwise risk 与不确定度 |
| tiny controller `NO_GO` | 39 个廉价 metadata scalars 足以恢复 risk | 当前存储接口丢失决定边界的方向信息 |

这与此前 Wan `EXP-004/005` 的 observability 结论同构：增加固定函数容量不能补回未观测的当前 mode；高价值坐标在 target-visible oracle 中存在，但只有提供足够 current state 才能部分恢复。

形式上，令 (R_g) 是真实 group risk，(Z_g) 是当前 39D metadata，(q) 是问题。最优 controller 仍受限于：

\[
\mathbb E\operatorname{Var}(R_g\mid Z_g,q)
\]

这一条件创新下限。controller 的稳定小幅 lift 表明 (I(R_g;Z_g\mid q)>0)，但 `30.91%` recall 和 `32.55%` risk-mass capture 表明该互信息远不足以支撑严格 rate-distortion gate。

## 与热力学/路径风险理论是否一致

一致的是 metric 原则，不应机械搬用物理术语。

在 diffusion 中，路径代价由 drift error 在噪声 metric 下累积；在当前视频理解 reader 中，没有 SDE 或物理温度，正确对象是问题条件的 decision metric：

\[
G_q=J_{X\rightarrow\ell(q)}^\top H_\ell J_{X\rightarrow\ell(q)}.
\]

target-gradient oracle 相对 residual/query 的优势证明 (G_q) 比 (I) 更适合作为 evidence value；tiny controller 的失败则证明 metric 本身正确并不意味着 metric 可由低带宽 metadata 观测。理论因此帮助我们把瓶颈从“矩阵容量”定位到“状态接口与风险可观测性”。

不能从本轮推出：

- 视频理解状态不可压缩；
- 低秩、稀疏或 progressive retrieval 普遍无效；
- entropy production 已在 reader 内部得到物理验证；
- gradient teacher 已形成可部署方法。

## 当前架构应如何收敛

仍应保留三部分，但必须改变中间接口：

1. Stable quotient bulk：继续作为常驻低成本语义索引，而非完整任务充分统计量。
2. Cold exact innovation：继续以规则 group/tile 保存，供精确渐进读取。
3. Risk-observable writer：不再从固定 metadata 事后猜风险，而是在写入时学习一个与 adverse boundary 对齐的 innovation key。

下一代最小形式为：

\[
b_g=E_b(X_g),\qquad
k_g=E_\phi(X_g,b_g),\qquad
e_g=X_g-D_b(b_g),
\]

\[
\hat r_g=f_\psi(q,k_g,b_g),\qquad
\hat Y=\operatorname{Reader}(\{b_g\},\{e_g:g\in\Omega(q)\}).
\]

其中 reader 与原模型冻结，只训练 writer key (E_\phi)、group scorer (f_\psi) 和风险上界；exact payload 仍冷存。训练目标必须联合：

\[
\mathcal L=
\mathcal L_{\rm reader-distill}
+\lambda_r\mathcal L_{\rm adverse-risk}
+\lambda_c C_{\rm bytes/read}
+\lambda_u\mathcal L_{\rm coverage}.
\]

关键对照必须是相同 bytes、density 与 reader cost 下的：

- only fixed writer + learned controller；
- learned writer + simple similarity read；
- joint learned writer-controller；
- full reader、query cosine、residual energy、FrameFusion/FlexMem 类 proxy。

只有 joint 在新 prompt/video 上显著优于两个单独分支，且保持 `>=98%` agreement、`0` harmful、`<=15%` fallback、`<=53%` retention，才能支持“怎样叠加”而非组件堆叠的创新主张。

## 与领域工作的创新边界

LongVU、FrameFusion、StreamingTOM 和 FlexMem 已覆盖 query-aware reduction、dual-path memory、progressive retrieval 与 training-free memory。不能把这些机制本身作为新意。

仍可能形成差异的主线是：

> 用 full-reader adverse boundary 训练 risk-observable memory writer，使稳定 quotient 与 cold exact innovation 在存储时就按任务风险可分；再用校准遗漏风险而非相似度或 scalar margin 决定渐进读取。

若 learned writer 仍不能在严格预算下显著提升 prospective risk coverage，应停止该视频理解主线，而不是继续扩大 controller width、sketch dimension、BCM block 或 fallback threshold。

## 最终判决

本轮是有效 `NO_GO`，但它不是对核心动机的否定。它完成了一个关键定位：

\[
\boxed{
\text{稳定 bulk 存在}
\;\land\;
\text{task-risk signal 可学习}
\;\land\;
\text{当前低带宽接口不充分}
}
\]

所以当时最有判别力的进一步实验不是拟合更复杂的固定结构，也不是给 scalar fallback 调阈值，而是让 memory writer 主动编码“未来问题可能需要的风险方向”。该 follow-up 已在 fresh calibration positions 73--96 上完成：joint writer-controller recall 仅 `30.70%`，reader agreement `70.83%`，同预算 target-gradient oracle 也只有 `91.67%` agreement，判决仍为 `NO_GO`。完整结果见 `RISK_OBSERVABLE_WRITER_AUDIT_20260830.zh-CN.md`；当前 post-hoc writer/controller 主线因此应 park。

## 2026-08-30 后续校正：从风险可观测性到路径可实现性

后续实验保留本报告的 controller `NO_GO`，同时排除了一个更隐蔽的假设：即使风险
信号可见，也不能默认各 group 的 exact-read value 独立可加。固定槽位
target-gradient 只是一阶 teacher；实际 compact reader 的 singleton benefit 在
support 增长后出现强交互，注册路径共有 `71` 次 KL regression，24/24 样本均受影响。

因此新的最小充分问题是：能否在原始位置与 token mass 被保留的条件下，构造
reader-aligned、current-support-conditioned、无 adverse regression 的嵌套 split path。
若该 train-free ceiling 不存在，增加 controller 容量没有意义；若 ceiling 存在而
低带宽规则无法生成，才进入 quotient tokenizer、mass/position adapter 与 reader
normalization 的小步联合训练。详见
`TARGET_RISK_BUDGET_AND_COMPACTION_GEOMETRY_AUDIT_20260830.zh-CN.md`。
