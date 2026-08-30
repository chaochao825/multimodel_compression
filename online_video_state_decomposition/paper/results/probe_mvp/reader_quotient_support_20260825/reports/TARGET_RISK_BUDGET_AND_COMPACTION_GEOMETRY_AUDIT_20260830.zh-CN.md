# 条件冗余理论、压缩路径几何与 reader-aligned 边际审计

日期：2026-08-30
范围：Wan `EXP-002`--`EXP-005` 的理论边界，以及 VSI/OneVision stable quotient、
target-risk budget、位置几何和 reader-aligned singleton 诊断。本文不改变 Wan
`L-026` 主线，不读取 VSI positions 97--120、selection 或 formal endpoint。

## 0. 直接结论

用户提出的“条件冗余、双时间几何和 diffusion 非平衡路径风险”框架在抽象层面
与历史结果一致，而且确实帮助我们继续定位了核心问题；但必须做三项收紧：

1. 它说明应寻找高信息增益的条件观测和正确 endpoint metric，不自动推出 BCM、
   BCCB、low-rank 或任何固定算子。
2. Girsanov/path-KL 只适用于满足条件的同扩散矩阵 SDE。Wan 当前的离散确定性
   sampler 仍应使用 suffix Jacobian、rollout 或 terminal loss，不能把内部 feature
   直接赋予物理温度。
3. 视频理解 reader 不是 diffusion。这里可借用的是条件风险、successive
   refinement 和 information-per-cost，不是 entropy production 的物理解释。

本轮新增证据把瓶颈推进了两层：

> 失败不仅来自局部 writer 看不到全局 reader state；当前 frozen reader 的可变长度
> quotient 路径本身也不满足单调 successive refinement。恢复更多 exact token
> 可能使决策和 KL 反而变差。

同预算的真实空间 `2x2` 控制进一步表明，几何邻接不是独立有效因素。它只在
proportional group mass 同时存在时获得决策级 headroom；equal mass 下反而恶化。
因此可复用对象更接近“带质量和位置的局部测度”，而不是一个无权 token 均值。
随后执行的 paper-faithful `K=4` PPE 虽将 mismatch 从 `5` 降至 `3`，却增加
harmful 并放大 P95 KL，正式判决为 `NO_PPE_HEADROOM`。因此位置完整性也不能单独
建立安全的 frozen-reader refinement path。

因此当前最有潜力的核心不再是“更好的静态 support teacher”，而是：

> **Position/measure-aware、reader-aligned、path-consistent 的层次 quotient memory。**
> 训练时让每个 quotient node 显式携带原始坐标与 token mass，并在嵌套拆分路径上
> 直接约束 reader 分布单调接近 dense；运行时才预测 current-support-conditioned
> 边际收益并做校准 fallback。

这保留了最初的“稳定 bulk + sparse exact innovation + progressive read”动机，但
否定了 frozen reader 上把各 group 风险独立相加的版本。

后续 query-fixed 两个 Gate 又把这一判断收紧了一步。正值 Gaussian moment state
即使允许 target-visible support，在 `25%` exact pages 下仍有
`15.50%/25.28%/29.61%` 的 visual mean/P95/worst error；增加 covariance rank
反而持续恶化。完全放弃 bulk 近似、只读取 exact pages 时，target-visible
attention-mass oracle 在读取 `75%` pages、覆盖 `94.40%` attention mass 后仍有
`3.37%` visual mean error。因此 frozen、规则 page 上的 hand-designed measure
closure 和 exact-only retrieval 都已有效关闭。剩余方向必须同时改变节点语义、
bulk state 和下游风险目标，而不能只增加 moment rank 或改 page selector。

## 1. 理论与历史实验是否一致

### 1.1 条件风险分解是正确主轴

对固定正半定任务度量 `W`，平方风险可分解为：

\[
R_W(\hat Y)
=
\mathbb E\operatorname{tr}\operatorname{Cov}_W(Y\mid Z,H)
+
\mathbb E\|\mathbb E[Y\mid Z,H]-\hat Y(Z,H)\|_W^2.
\]

它准确解释了历史结果：

| 证据 | 结果 | 对应解释 |
|---|---:|---|
| `EXP-004` past-only full-rank/MoE | 最佳约 `1.001x` | 增加函数容量没有补回 current mode |
| `EXP-004` target-visible oracle | late layer 最高 `11.077x` | 目标条件下存在强冗余 |
| `EXP-005` current-input diagonal field | pooled `1.937x` | current input 显著降低部分层的条件创新 |
| `EXP-005` breadth gate | 仅 layers `21/24/25` 通过 | 该 observer 不是十层连续接口 |
| VSI stable quotient | bulk 可压缩 | 存在统计冗余 |
| VSI local metadata writer/controller | recall 约 `30%` | 局部低带宽状态看不到 reader boundary |

因此“统计冗余存在”与“存在便宜、共享、闭环稳定的算子”确实是不同命题。

### 1.2 双时间曲率是有用诊断，不是已测事实

把物理视频时间记为 `tau`、去噪时间记为 `lambda`，交换缺陷

\[
\mathcal F_{\tau\lambda}
=-
\partial_\tau L_\lambda
+\partial_\lambda L_\tau
+[L_\tau,L_\lambda]
\]

能解释为什么物理 transport 不自动成为跨 denoising-step residual predictor。
RoPE、AdaLN、CFG、遮挡和 routing 都可使该曲率随条件变化。但当前实验没有直接
估计这一定义中的两个生成元，所以它应被称为机制模型，而不是经验测量结果。

### 1.3 diffusion 路径风险需要严格适用边界

若 exact 与 approximate SDE 具有相同、非退化扩散矩阵，并满足 Girsanov 条件，
漂移差的路径代价由

\[
\frac12\mathbb E\int
\delta b_t^\top a_t^{-1}\delta b_t\,dt
\]

控制。这支持“低噪阶段同样 feature MSE 可能更危险”的判断。[Score-SDE](https://arxiv.org/abs/2011.13456)
与随机热力学可提供背景，[Seifert](https://arxiv.org/abs/1205.4176)；但 Wan 的离散
UniPC/rCM 评估不能直接用该式替代真实 rollout。正确实现仍需：

\[
\mathcal R_{\ell,k,H}
=e_{\ell,k}^\top G_{\ell,k,H}e_{\ell,k}
+\operatorname{tr}(G_{\ell,k,H}\Sigma_{\rm inn})
+\beta H(M\mid Z).
\]

这里 `G` 必须由 suffix/terminal 任务诱导，而不是默认 `I`。

## 2. 新增 budget 与几何诊断

### 2.1 dense-gradient budget frontier：没有可靠容量窗口

在已暴露 positions 73--96 上，把 exact group 从 `0` 扫到 `392`：

| exact groups | token retention | agreement | harmful | KL mean / P95 |
|---:|---:|---:|---:|---:|
| 0 | 25.00% | 70.83% | 2 | 0.0600 / 0.2154 |
| 98 | 43.75% | 91.67% | 2 | 0.0270 / 0.0874 |
| 196 | 62.50% | 95.83% | 1 | 0.0113 / 0.0321 |
| 245 | 71.88% | 95.83% | 0 | 0.0106 / 0.0263 |
| 343 | 90.63% | 91.67% | 1 | 0.0104 / 0.0329 |
| 392 | 100.00% | 100% | 0 | 0 / 0 |

即使 target-risk mass 在 `k=343` 已约 100%，决策仍会错误；路径也不是单调的。
预注册判决为 `NO_USEFUL_CAPACITY_WINDOW`。

### 2.2 保留原始位置显著改善离散决策，但没有恢复分布

同一 gradient support 在三种路径下比较：

| `k=196` 机制 | reader slots | agreement | harmful | KL mean / P95 |
|---|---:|---:|---:|---:|
| compact contiguous positions | 62.5% | 91.67% | 1 | 0.0122 / 0.0374 |
| compact original positions | 62.5% | **100%** | **0** | 0.0213 / 0.0468 |
| fixed repeated quotient | 100% | **100%** | **0** | 0.0329 / 0.1631 |

原始位置把离散 reader 决策恢复到 `24/24`，证明 quotient 不是完全无效；但 KL
未达 `0.01/0.02`，且 `k=294/343` 又出现错误，预注册判决仍为
`NO_GEOMETRY_RECOVERY`。

这说明位置是必要状态之一，却不是充分状态。token multiplicity、attention
normalization 和 group interaction 仍未解决。

### 2.3 dense-gradient teacher 本身不稳定

两个独立运行的 `compact_contiguous` 在 `k=0/392` 完全一致，说明 base quotient
和 dense endpoint 稳定；一旦使用非零 gradient support，排序差异产生：

| `k` | 改变 top-1 的问题数 | mean absolute KL delta | max delta |
|---:|---:|---:|---:|
| 98 | 2/24 | 0.00366 | 0.02371 |
| 128 | 0/24 | 0.00967 | **0.14269** |
| 196 | 1/24 | 0.00355 | 0.03160 |
| 343 | 3/24 | 0.00356 | 0.03164 |

因此此前“target-gradient oracle”必须收窄为 **BF16/SDPA dense-path 一阶 teacher**，
不是可变长度 compact path 的真实 oracle，也不够稳定到承担主 capacity claim。

## 3. reader-aligned singleton Gate：真实局部收益仍不能形成路径

为排除 teacher/objective mismatch，本轮对 24 个已暴露问题、每个 392 个 group
逐一执行真实 compact reader：

\[
\Delta_g(\varnothing,q)
=D_q(\varnothing)-D_q(\{g\}),
\qquad
D_q(\Omega)=\operatorname{KL}(p_{\rm dense}^q\Vert p_\Omega^q).
\]

按该真实 singleton benefit 排序后形成静态嵌套路径，结果为：

| exact groups | retention | agreement | harmful | KL mean / P95 |
|---:|---:|---:|---:|---:|
| 0 | 25.00% | 75.00% | 3 | 0.0392 / 0.0974 |
| 49 | 34.38% | 83.33% | 1 | 0.0365 / 0.1369 |
| 98 | 43.75% | 79.17% | 2 | 0.0381 / 0.1601 |
| 196 | 62.50% | 87.50% | 0 | 0.0369 / 0.1392 |
| 245 | 71.88% | 87.50% | 0 | 0.0427 / 0.1527 |
| 343 | 90.63% | 91.67% | 0 | 0.0141 / 0.0729 |
| 392 | 100.00% | 100% | 0 | 0 / 0 |

预注册判决为 `NO_STATIC_READER_PATH`，dense/full equivalence 与重复推理误差均为
`0`。

更关键的是交互量：

\[
I_q(\Omega_k)
=D_q(\Omega_k)
-\left[D_q(\varnothing)-\sum_{g\in\Omega_k}\Delta_g(\varnothing,q)\right].
\]

| `k` | mean interaction residual | median | positive samples |
|---:|---:|---:|---:|
| 49 | **0.555** | 0.324 | 22/24 |
| 98 | **0.878** | 0.358 | 21/24 |
| 196 | **1.213** | 0.424 | 21/24 |

前 49 个 singleton 收益相加会预测平均 KL 为 `-0.518`，实际为 `0.0365`；到
196 个时预测为 `-1.176`，实际仍为 `0.0369`。24/24 问题均至少出现一次 KL
反向增加，共 `71` 次 budget-to-budget KL regression；另有 `6` 次
match-to-mismatch regression。

这不是 predictor 太弱，而是 set function 本身不满足当前方法需要的单调可加性。
实际边际必须写为：

\[
\Delta_g(\Omega,q)
=D_q(\Omega)-D_q(\Omega\cup\{g\}),
\]

它依赖当前 support、序列长度、位置、token mass 和全部 reader state。

### 3.1 与 adaptive submodularity 的关系

若收益函数单调且满足 diminishing returns，current-support greedy 才有经典近似
保证。[Adaptive Submodularity](https://arxiv.org/abs/1003.3967) 给出了这一类信息
获取问题的理论基础。当前结果更严重：某些 exact split 直接增大 `D`，因此连
单调性都不成立；不能继续把一阶 score 相加成 certificate。

### 3.2 baseline reader 本身较弱

本组 dense reader task accuracy 只有 `25%`。所以 agreement/KL 只是“复现该
reader”的 fidelity，不等于任务质量。任何最终方法都必须在更强 reader、完整
VSI/VideoMME 等 endpoint 和 wall-clock 上重新验证，不能从当前 24 个问题得出
通用视频理解结论。

## 4. 为什么现有 compaction path 会违反单调性

压缩 reader 实际计算的是：

\[
p_\Omega^q
=\operatorname{softmax}
f\left(
T_\Omega(X),
P_\Omega,
M_\Omega,
q
\right),
\]

其中：

- `T_Omega` 决定 exact/quotient token；
- `P_Omega` 决定保留后的 RoPE/position geometry；
- `M_Omega` 是每个 token 代表的原始 patch 数量；
- support 改变序列长度和后续所有层的 attention denominator。

旧 gradient teacher 在 dense fixed-slot 点上线性化；singleton teacher在空 support
点测量。两者都没有估计从 `Omega` 到 `Omega union {g}` 的真实条件边际。

ToMe 已明确跟踪 token size，并在 attention logits 中做 proportional attention；
其消融表明该机制对 supervised model 很重要。[ToMe](https://arxiv.org/abs/2210.09461)
因此 token mass 不是可选元数据，也不是我们的新意。

本轮 measure-preserving 实现尝试了显式 4D additive mask，但 all-mass-one 路径
相对普通 2D SDPA path 的 candidate logits 仍变化 `0.25`。两次尝试均按协议归类
为 **invalid engineering**，不能解释为 proportional attention 无效。后续必须先
在同一 attention kernel path 中通过 all-mass-one dense-equivalence，才可比较 mass。

## 5. 核心改进：Path-Consistent Successive-Refinement Quotient

### 5.1 状态表示

每个层次 quotient node 不只保存一个 mean，而保存：

\[
n=(\mu_n,\;m_n,\;p_n,\;c_n,\;\text{children}_n),
\]

其中：

- `mu_n`：bulk representative；
- `m_n`：代表的原始 token mass；
- `p_n`：原始时空坐标或可组合位置统计；
- `c_n`：低成本 content/query key；
- children：冷存的 exact innovation 或下一层 quotient。

BCM/BCCB 可以作为离线布局或局部编码工具，但不再承担 reader-risk basis。

### 5.2 训练目标直接作用于嵌套路径

训练时采样一条嵌套路径

\[
\Omega_0\subset\Omega_1\subset\cdots\subset\Omega_B,
\]

并优化：

\[
\mathcal L
=
\sum_b w_b D_q(\Omega_b)
+\lambda\sum_b
\left[D_q(\Omega_{b+1})-D_q(\Omega_b)+\epsilon\right]_+
+\beta C_{\rm reader}(\Omega_b)
+\gamma\mathcal L_{\rm coverage}.
\]

第一项保证各预算 fidelity；第二项直接惩罚“读得更多反而更坏”；第三项约束真实
token/latency；第四项校准遗漏风险。这个目标把 progressive read 变成训练原生的
successive refinement，而不是假定 frozen reader 自动具有该性质。信息论中的
[successive refinement](https://arxiv.org/abs/1707.09567) 是理论参照，但神经
reader 是否可逐级精化必须靠训练和实验建立。

### 5.3 teacher 与 deployable router 分离

离线 teacher 使用真实 current-support marginal：

\[
g_b^*=\arg\max_g
\frac{\Delta_g(\Omega_b,q)}{\Delta C_g}.
\]

部署 router 只能看 `(q, c_n, m_n, p_n, current support summary)`，预测边际收益
与上分位风险。只有当嵌套路径本身先稳定，预测器问题才有意义；否则再强的 writer
也只是在拟合一个不一致接口。

### 5.4 exact anchor 与校准 fallback

运行时维护剩余风险上界 `U_b`：

\[
U_b=Q_{1-\alpha}
\left(D_q(\Omega_b)\mid z_b\right).
\]

若 `U_b` 超过预算，则继续拆分或 full fallback。这里 exact read 相当于高成本
测量/anchor，作用是降低条件创新与校准不确定度；这正是热力学/信息增益框架真正
能指导的地方。

## 6. 与相关工作的边界

| 工作 | 已覆盖 | 本方向不能声称 | 尚可验证的差异 |
|---|---|---|---|
| [ToMe](https://arxiv.org/abs/2210.09461) | merging、token size、proportional attention | 首次 mass-aware merging | reader-risk path consistency |
| [FrameFusion](https://arxiv.org/abs/2501.01986) | similarity merge + importance prune | 首次 merge/prune 混合 | adverse-risk nested split |
| [LongVU](https://openreview.net/pdf/584946fd40dfa8c20c7c527f77e27a340b88664f.pdf) | query/inter-frame adaptive compression | 首次 query-aware video compression | actual compact-reader marginal与校准 fallback |
| [MMInference](https://openreview.net/forum?id=me6PfbATWM) | modality-aware permutation sparse attention与kernel | 首次视频 grid/sparse layout | memory quotient的successive-refinement语义 |

FrameFusion 报告高相似 token 在早层可 merge、深层再按重要性 prune；LongVU
使用 query 与 inter-frame dependency；这些都与“稳定 bulk + task-sensitive
innovation”相容。但它们没有让我们的当前结果自动成立。创新必须由相同
reader、相同 token/latency budget 下的独立增益证明。

## 7. 下一轮严格顺序

### Gate M0：同 kernel 的 measure equivalence

1. 在 eager 或可控 SDPA attention path 中，让 equal-mass 2D/4D 或 fused bias
   candidate logits 等价到 `1e-5`。
2. 再测试 `log m` proportional attention；不先比较质量。
3. 若无法获得同路径等价，停止该实现，不把 kernel 数值差异当算法证据。

### Gate M1：train-free monotonicity ceiling

在已暴露数据上比较：

- original-position only；
- original-position + proportional mass；
- current-support greedy marginal；
- fixed repeated diagnostic ceiling。

继续条件：在 `k<=196` 达到 `24/24`、`0 harmful`、KL `<=0.01/0.02`，且无
match/KL regression。若 mass-aware current-support oracle 仍失败，关闭 frozen
reader 的 train-free progressive quotient，不再增加 controller/BCM 容量。

### Gate M2：低成本 path-consistency adaptation

仅在 M1 表明函数类有上限、但 frozen path 不稳定时进行：

- 冻结视觉 encoder、原始 QKV 和绝大多数 reader 参数；
- 只训练 quotient tokenizer、mass/position adapter、small support router 与
  normalization ratio；
- 训练 positions 1--72，positions 73--96 只做 selection，positions 97--120
  保持一次性最终验证；
- 公平比较 tokenizer-only、router-only、joint path-consistency，以及 ToMe、
  FrameFusion、LongVU 风格 baseline；
- 最终必须补 stronger reader、完整任务、真实 token latency 和 end-to-end wall-clock。

## 8. 对 Wan 主线的影响

该 reader 结果不能用于宣称 Wan cache/attention 失败或成功。它只强化一个共同
原则：

\[
\text{条件冗余}
\not\Rightarrow
\text{raw-space 固定算子}
\not\Rightarrow
\text{单调可部署路径}.
\]

Wan 仍应先完成 released rCM/few-step H200 incumbent；若后续训练 state model，
应把 persistent world/causal state 与 per-step renderer 分离，并用完整 rollout
风险训练，而不是继续 post-hoc 拟合 raw residual。

## 9. 最终判决

当前 frozen-reader、post-hoc 静态 support 路线应继续 `park`。但最初目标并没有
完全失败：stable quotient bulk、exact innovation 和 conditional risk 都有实证
基础。真正需要改变的是它们之间的接口：

\[
\boxed{
\text{stable measure quotient}
+\text{reader-aligned nested refinement}
+\text{path-consistency training}
+\text{calibrated exact fallback}
}
\]

这是一条比“再加一个 low-rank/BCM/controller”更简洁、更接近核心 field problem
的路线。当前最优动作是先完成 M0/M1 的函数类与数值等价诊断；只有它们通过，
才值得投入小步适配与系统 kernel。

## 9.1 M0 后续校正：同 kernel 的 mass 代数等价已通过

M0 最终在 OneVision Qwen2 的 eager attention path 中通过。普通路径、显式
all-mass-one 路径和重复执行在 24 个已暴露问题上的 full-vocabulary logits 最大
误差均为 `0`；`k=392` 的 full-refinement 路径与 dense endpoint 误差也为 `0`。
正式判决为 `SAME_KERNEL_MASS_VALID`。

在此基础上加入 quotient token 的 `log m` bias 后，weighted diagnostics 为：

| exact groups | agreement | KL mean / P95 |
|---:|---:|---:|
| 0 | 70.83% | 0.05464 / 0.23258 |
| 196 | 75.00% | 0.02651 / 0.13432 |
| 392 | 100% | 0 / 0 |

这些数值只说明 proportional mass 已在相同 attention 实现中被正确表达，不说明
它优于 equal mass，也不说明路径单调。eager harness 不是生产延迟候选；标准
SDPA/FlashAttention 也没有直接暴露任意 per-key `log m` 接口，真实部署仍需融合
`exp(score) * mass`、预变换或等价 kernel。M0 首次尝试在全部 forward 完成后因
summary 聚合错误退出，按协议只修复一次聚合逻辑；失败尝试保留，第二次运行才是
上述有效结果。

## 9.2 理论导出的核心候选：风险有界的层次测度 memory

M0/M1 的更深作用不是再证明一种 token score，而是指出当前 decoder
self-attention 接口本身不利于渐进精化。一个更可证明的接口是把视觉状态作为经验
测度，由独立 cross-attention reader 消费：

\[
\mu=\sum_i\delta_{(p_i,k_i,v_i)},\qquad
Z(q)=\int e^{q^\top k}\,d\mu,\qquad
N(q)=\int e^{q^\top k}v\,d\mu,\qquad
A(q,\mu)=N(q)/Z(q).
\]

对层次节点 `g`，保存 mass `m_g`、centroid `bar k_g/bar v_g`、位置状态及半径
`r_k/r_v`。若 `||q||<=Q`，centroid 近似的局部 denominator 余项满足：

\[
\epsilon^Z_g
\le
m_g e^{q^\top\bar k_g}
\left(e^{Q r_{k,g}}-1\right),
\]

numerator 余项可保守界为：

\[
\epsilon^N_g
\le
m_g e^{q^\top\bar k_g}
\left[
e^{Q r_{k,g}}r_{v,g}
+
\left(e^{Q r_{k,g}}-1\right)\|\bar v_g\|
\right].
\]

`N/Z` 的全局误差再由 `sum_g epsilon_g`、`Z` 的正下界和 triangle inequality
控制。实际误差可能因不同节点抵消而非单调，但可以递归保存 envelope：

\[
b_g=\max\left(\tilde b_g,\sum_{c\in\operatorname{child}(g)}b_c\right).
\]

这样把父节点拆成 children 后，未读风险上界按构造不增加。运行时选择：

\[
g^*=\arg\max_g
\frac{b_g-\sum_{c\in\operatorname{child}(g)}b_c}
{\Delta T_{\rm measured}(g)},
\]

并在校准后的 task-risk 上界低于门槛时停止，否则继续拆分或 exact fallback。
这个结构把“稳定 bulk、精确 innovation、条件风险”统一成 adaptive quadrature，
而不是 BCM、低秩和 controller 的并列叠加。

它有四个必须保留的边界：

1. 上述可加性只直接适用于 query 固定的单个 cross-attention memory 接口；在当前
   多层 decoder self-attention 中，压缩会改变后续 query 和状态，不能套用该证明。
2. 几何半径给出的 bound 可能很松，必须用 reader Jacobian/adverse loss 做校准；
   校准证书只对相同数据分布提供统计覆盖，不是任意输入的绝对保证。
3. [FMMformer](https://arxiv.org/abs/2108.02347)、
   [H-Transformer-1D](https://arxiv.org/abs/2107.11906)、
   [Fast Multipole Attention](https://arxiv.org/abs/2310.11960) 和
   [MuSe](https://arxiv.org/abs/2509.10406) 已覆盖层次近远场、multipole 或
   centroid/dipole 近似；不能声称首次层次 attention 或 multipole memory。
4. 可能成立的独立点只能是 actual-reader adverse-risk calibration、可组合 remainder
   envelope、progressive exact split 与实测成本共同决定的 fallback，并且必须在
   相同 reader、token budget 和 latency 下胜过 ToMe/PPE/FMM 类 baseline。

因此 M1 仍是必要 Gate：它先判断当前 frozen decoder 是否已经存在可利用的条件
路径。若 M1 失败，先做同预算 `2x2` geometry/PPE 控制；该控制也失败后，应停止
继续扩充 frozen self-attention controller，转而用单层 external memory prototype
直接验证上述可加 bound 与一次性 support 选择。只有 prototype 同时通过风险和
成本门槛，才训练小型 quotient/risk adapter。

## 9.3 M1 结果：current-support 有 headroom，但没有安全渐进路径

M1 在 positions 73--96 的 24 个已暴露问题上有效完成，未读取 positions
97--120、selection 或 formal。每个样本包含 8 帧、1,568 个视觉 token 和 392 个
展平连续的 4-token group；两种路径都保留原始位置，并在同一个 eager attention
实现中比较 equal mass 与 group mass。离线 teacher 在当前 support 上逐一执行真实
compact reader，按条件 KL benefit 选择下一批 49 个 group：

\[
\Delta_g(\Omega,q)
=D_q(\Omega)-D_q(\Omega\cup\{g\}).
\]

该实验是 transductive 的 batched receding-horizon capacity diagnostic，不是精确
逐 group greedy，也不是部署 router。完整运行耗时 `5,416.10 s`，得到 240 条路径
记录和 61,152 条条件边际记录；正式判决为
`NO_BATCHED_CURRENT_SUPPORT_PATH`。

| mode | exact groups | retention | agreement | mismatch | harmful | KL mean / P95 |
|---|---:|---:|---:|---:|---:|---:|
| equal mass | 0 | 25.00% | 79.17% | 5 | 1 | 0.03348 / 0.09814 |
| equal mass | 49 | 34.38% | 87.50% | 3 | 1 | 0.03162 / 0.07118 |
| equal mass | 98 | 43.75% | 87.50% | 3 | 2 | 0.02487 / 0.08613 |
| equal mass | 147 | 53.13% | 83.33% | 4 | 1 | 0.00946 / 0.02978 |
| equal mass | 196 | 62.50% | **95.83%** | **1** | **1** | 0.01054 / 0.04010 |
| group mass | 0 | 25.00% | 70.83% | 7 | 3 | 0.05464 / 0.23258 |
| group mass | 49 | 34.38% | 91.67% | 2 | 0 | 0.02157 / 0.06893 |
| group mass | 98 | 43.75% | 79.17% | 5 | 2 | 0.02218 / 0.08285 |
| group mass | 147 | 53.13% | 91.67% | 2 | 1 | **0.00810** / 0.03330 |
| group mass | 196 | 62.50% | 91.67% | 2 | 1 | 0.00951 / 0.02810 |

两个 mode 的 strict budget 均为空。最接近门槛的 equal-mass `k=196` 仍有一个
match-to-mismatch harmful flip，且 P95 是注册门槛 `0.02` 的约两倍；group-mass
虽把 mean KL 降至 `0.00951`，却有两个 mismatch 和一个 harmful。两个 harmful
样本的 compressed top-1 margin 都为 `0.125`，因此也不能用“只在近零 margin
处失败”解释。

### 9.3.1 路径非单调不是均值偶然波动

| mode | match-to-mismatch transitions | KL-increase transitions | 有 KL 回退的样本 |
|---|---:|---:|---:|
| equal mass | 6 | 36 | 24/24 |
| group mass | 8 | 33 | 22/24 |

虽然每轮被选中的 49 个 singleton 中有 `71.7%--90.8%` 具有正条件收益，把这些
单项收益相加仍严重高估联合更新。令：

\[
I_{49}
=D_q(\Omega_{b+1})
-\left[D_q(\Omega_b)-\sum_{g\in B_b}\Delta_g(\Omega_b,q)\right].
\]

equal-mass 四轮的 median `I_49` 为 `0.349/0.251/0.192/0.206`，正值样本分别为
`24/24, 23/24, 23/24, 23/24`；group-mass 为
`0.335/0.195/0.092/0.077`，正值样本为 `22/24, 22/24, 23/24, 19/24`。
因此条件边际确有信息，但 49-group batch 内高度冗余并存在 error cancellation；
它不是可直接相加的风险 certificate。

这保留了一个很窄的可能性：更小 batch 或联合 support optimizer 可能改善
transductive oracle。但精确逐 group teacher 约需把四轮扩展到近 196 轮，既昂贵
又不可部署；当前证据不支持直接投入这种 brute-force oracle。

### 9.3.2 proportional mass 是必要表示，不是通过机制

在 `k=49/98/147/196`，group mass 相对 equal mass 的 mean KL 分别变化
`-0.01005/-0.00269/-0.00135/-0.00102`，但逐样本 KL 胜出仅为
`15/12/12/11` 个，决策净胜负分别为 `2:1, 2:4, 3:1, 1:2`。因此 mass 的均值
收益由少量样本驱动且不稳定，不能把 M1 的剩余 headroom 归因于 mass；M0 只证明
它被正确表达。

### 9.3.3 结论边界与下一动作

M1 相对历史 static singleton 明显降低 KL，说明 current support 是有价值的条件
变量；但 historical static 使用 SDPA，而 M1 使用 eager，因此该比较只能是描述性
诊断，不能作为严格 method win。M1 的有效 null 只关闭：

> 原始位置、可选 proportional mass、展平连续 4-token group、每轮 49-group
> current-support singleton teacher 组成的 frozen-reader 路径。

它不关闭 exact sequential greedy、true `2x2` geometry、PPE、多节点 joint
optimizer、external cross-attention memory 或训练原生 path consistency。由于当前
展平连续 group 会在部分位置跨越 `14x14` 行边界，因此先执行一个便宜、相同
预算的 `flat-4 vs true-2x2` 几何控制。PPE 不能用代表位置近似，必须在 RoPE 内
真正聚合多个 constituent position；只有几何控制达到预注册结果，才单独授权该
paper-faithful PPE 控制。若无改善，则冻结 self-attention 的 train-free progressive
quotient 应保持 parked，后续只比较：

1. 单层 external memory 上的可加 numerator/denominator remainder bound；
2. 冻结视觉 encoder/QKV，仅训练 quotient、position/mass adapter 和一次性 support
   router 的 path-consistency 适配；
3. ToMe、PPE、DyCoke、ForestPrune 和 FMM/MuSe 类同预算 baseline。

![M1 current-support 路径、尾部误差与 49-group 交互](../figures/batched_current_support_marginal_audit.png)

## 9.4 真实空间 2x2 控制：质量与几何必须联合表示

冻结的 topology control 在相同 reader、rank、八帧和 `25%` token retention 下，
比较原有展平连续四元组与每帧 `14x14` 网格上的非重叠真实 `2x2`。两条 flat
路径逐位复现 M1 `k=0`：maximum KL repeat error 为 `0`，预测不一致数为 `0`。
实验只读取 positions 73--96；positions 97--120、selection 和 formal 未读取。

| mass mode | geometry | agreement | mismatch | harmful | KL mean / P95 |
|---|---|---:|---:|---:|---:|
| equal mass | flat contiguous-4 | 79.17% | 5 | 1 | 0.03348 / 0.09814 |
| equal mass | spatial 2x2 | 75.00% | 6 | 2 | 0.05105 / 0.11792 |
| group mass | flat contiguous-4 | 70.83% | 7 | 3 | 0.05464 / 0.23258 |
| group mass | spatial 2x2 | **79.17%** | **5** | **1** | **0.04702 / 0.16398** |

正式判决为 `TRUE_2X2_DECISION_HEADROOM`，不是 strict
`TRUE_2X2_GEOMETRY_HEADROOM`。group-mass 路径满足 mismatch `7→5`、harmful
`3→1` 和 mean/P95 ratio `0.861/0.705`；mean ratio 未达到 strict `0.8`。
equal-mass 的 mean/P95 ratio 则为 `1.525/1.202`，决策和 harmful 也同时恶化。

逐样本审计进一步限制了结论。group-mass 的 spatial `2x2` 在 `16/24` 个样本上
降低 KL，单侧 sign-test 为 `p=0.0758`；mean KL delta 的 paired bootstrap 95%
区间为 `[-0.0331, 0.0173]`，mean-ratio 区间为 `[0.434, 1.317]`。equal-mass
仅 `12/23` 个非平局样本胜出，mean-ratio 区间为 `[0.748, 2.592]`。所以当前证据
支持的是一个交互机制：

\[
\text{spatial topology} \times \text{proportional mass},
\]

而不是“真实 `2x2` 已经稳定优于 flat grouping”。这与层次测度解释一致：group
mean 改变 support geometry，`log m_g` 恢复合并节点代表的 softmax measure；只做
其中一个会产生位置或归一化 mismatch。

按照预注册 outcome mapping，decision-only headroom **不授权**重跑 M1 的
current-support teacher，只授权一次 paper-faithful multi-position/PPE 控制。该控制
必须在 RoPE 变换中保留四个 constituent position 的贡献，不能继续用单个代表
位置冒充 PPE。若 PPE 不能把 group-mass spatial `2x2` 推到 strict 门槛，则 frozen
self-attention topology tuning 保持 parked，转向 query 固定的 external-memory
remainder bound 或低成本 path-consistency adaptation。

![真实空间 2x2 与展平四元组的配对审计](../figures/true_2x2_geometry_control.png)

## 9.5 Paper-faithful PPE：top-1 改善不等于风险改善

按照 9.4 的 decision-only outcome mapping，本轮只执行一次 PPE 控制。模型仍为
一维 Qwen2 RoPE，因此对每个真实 `2x2` quotient 使用 `K=4`：四个 member 按
reconstructed feature 到 group mean 的平方 L2 距离由近到远稳定排序，64 个 RoPE
frequency pair 均分成四组，每组绑定一个原始位置。非视觉 token、token value、
group mass、causal mask、eager attention 和所有模型参数保持不变。将普通标量位置
扩展到全部 frequency pair 时，24 个样本的 maximum full-vocabulary logit error 为
`0`；incumbent 的 KL 和预测也逐位复现，说明差异只来自 multi-position rotation。

| method | agreement | mismatch | harmful | KL mean / P95 |
|---|---:|---:|---:|---:|
| representative position | 79.17% | 5 | 1 | 0.04702 / 0.16398 |
| PPE center-ranked K=4 | **87.50%** | **3** | 2 | 0.04891 / 0.23526 |

PPE 获得三次 match gain、一次 match loss，因此 mismatch 净减少两项；但该 loss
同时新增一个 harmful。mean/P95 ratio 为 `1.040/1.435`，逐样本 KL 仅 `10/24`
胜出，单侧 sign-test 为 `p=0.798`。paired mean KL delta 的 bootstrap 95% 区间为
`[-0.0115, 0.0144]`，mean-ratio 区间为 `[0.816, 1.408]`。正式判决为
`NO_PPE_HEADROOM`。

该结果不能解释为 PPE 一般无效；[PPE](https://openreview.net/forum?id=OV0AoK7QEr)
在训练/架构适配过的 MLLM token compression 中保留多位置线索。本轮只说明：对该
OneVision frozen-reader、该 true-`2x2` group-mass quotient 和暴露 endpoint，PPE
在离散 top-1 与分布风险之间发生 trade-off，不能满足 adverse-risk guard。

这也给条件冗余理论一个更精确的修正：

\[
I(Y;\text{position}\mid H)>0
\quad\not\Rightarrow\quad
R_{\mathrm{reader}}(\text{PPE})\le R_{\mathrm{reader}}(\text{representative}).
\]

位置确实携带信息，但 frozen self-attention 会让 token merge 改变后续 query、
normalization 和跨层状态；局部信息增益不保证 suffix risk 单调。按预注册 stop rule，
继续增加 train-free position/topology variant 现已 parked，不重跑 M1 current-support
路径。后续只有两条与证据一致的路线：

1. 在 query 固定的 external cross-attention memory 上验证可组合
   numerator/denominator remainder envelope，使 split 按构造不增加风险上界；
2. 冻结视觉 encoder/QKV，只训练 quotient position/mass adapter 与嵌套路径损失，
   直接把 top-1/KL trade-off 纳入 path consistency，而不是事后选择 PPE 规则。

![真实空间 2x2 上的 paper-faithful PPE 配对审计](../figures/true_2x2_ppe_control.png)

## 9.6 Query-fixed visual measure：可组合性成立，但 centroid state 不充分

PPE null 后没有继续修改 frozen self-attention，而是在 final-token query 固定的单层
attention measure 上测试 9.2 的可加分解。每个样本仍有 1,568 个视觉 token，并以
真实空间 `2x2` 组成 392 个节点；coarse 节点只保存 `mass=4`、mean post-RoPE key
和 mean value。对 layers `0/13/27`、全部 28 heads，逐步将 coarse 节点恢复成四个
exact leaves。测试的保留率为 `25/34.375/43.75/53.125/62.5/100%`。

v1 engineering smoke 发现原 `exact_local_oracle` 只是一次性一阶分数，不是容量
ceiling，因此正式 run 前冻结 v2：保留该方法为 `exact_local_score`，新增每次 split
后重算真实 visual-output error 的 `exact_greedy_oracle`。第一次 formal attempt 又被
预注册 replay guard 正确拒绝：只 replay 最后 query 的 BF16 GEMM 与模型一次计算全
sequence 的 GEMM 形状不同，最大相对差为 `1.382e-3`。唯一一次 implementation
repair 改为捕获模型真实 Q/K/V 和 pre-`o_proj` 输入，并按模型完整 query 形状 replay；
repair run 的最大 replay error 为 `0`，局部 `Z/N` bound violation 和 certificate
increase 均为 `0`。失败目录和日志均保留，未覆盖。

在 `k=196`、即 62.5% visual-token retention 下，72 个 sample-layer cells 的结果为：

| selector | visual mean / P95 / worst | full mean / P95 / worst |
|---|---:|---:|
| analytic remainder | 5.520% / 10.474% / 21.857% | 0.607% / 1.102% / 1.768% |
| attention mass | 4.043% / 8.489% / 9.416% | 0.497% / 0.938% / 1.646% |
| exact local score | **3.450% / 6.433% / 7.395%** | **0.424% / 0.746% / 1.236%** |
| sequential exact greedy | 7.041% / 24.505% / 41.176% | 0.766% / 1.500% / 2.220% |
| fixed random | 25.141% / 55.504% / 65.375% | 2.086% / 4.308% / 5.711% |

正式判决为 `NO_REGISTERED_QUERY_FIXED_MEASURE_PATH`。该判决不能仅靠 sequential
greedy 解释，因为 nested greedy 会陷入路径局部最优；但更有利的 cell-wise
registered envelope 允许每个 cell 事后选择四种非随机路径，visual mean/P95/worst
仍为 `3.252%/5.841%/6.332%`，没有一个 cell 达到 `<=1%`。所以 narrow null 不依赖
某一个 selector 的异常，而是当前 true-`2x2` centroid state 与共享 regular split
family 的共同边界；它仍不是所有 cardinality subset 的全局最优证明。

### 9.6.1 深层退化和 full-output 稀释

`exact_local_score` 的 visual error 随层显著增加：

| layer | mean | P95 | worst |
|---:|---:|---:|---:|
| 0 | 1.300% | 1.464% | 1.497% |
| 13 | 3.337% | 4.042% | 4.383% |
| 27 | 5.714% | 7.161% | 7.395% |

相反，full-attention mean 已降至 `0.424%`。这不是 visual memory 已安全，而是 exact
non-visual numerator/denominator 稀释了视觉分支缺陷；若只看 full mean，会得到错误
的 positive。visual measure 因而是本 Gate 的 primary capacity endpoint，full output
只用于说明系统相关尺度。

解析不等式本身没有失败：所有 local `epsilon_Z/epsilon_N` 都覆盖真实局部 defect，
路径 certificate 也没有增加。但 finite denominator certificate 只覆盖 `11.01%`
heads；分层为 layer 0 的 `33.0%`、layers 13/27 的 `0%`。analytic path 虽有单调
certificate，真实 visual error 仍发生 10 次路径回退，说明不同节点之间存在明显
error cancellation；对局部绝对值求和的 triangle envelope 在深层过于保守。

### 9.6.2 与条件冗余和双时间理论的一致性

这一结果没有反驳条件冗余分解，反而把缺失接口进一步定位。固定 query 后，后续
query feedback、去噪时间和物理时间的交换缺陷已被排除；剩余误差来自 coarse state
不是 softmax measure 的充分统计量。令 `k=bar k+delta k`、`v=bar v+delta v`，则

\[
\sum_j e^{q^T k_j}v_j
=e^{q^T\bar k}
\sum_j e^{q^T\delta k_j}(\bar v+\delta v_j).
\]

只保存 `bar k, bar v, mass` 会丢掉至少

\[
\Sigma_{vk}q
=\mathbb E[\delta v\,\delta k^T]q
\]

以及更高阶 score-value coupling。layer 27 的退化说明该条件 innovation 随深度变得
更重尾，而不是 query-fixed measure 不可加。与 EXP-004/005 的共同结论是：增加固定
rank 或静态结构不能恢复未观测的条件坐标；`EXP-005` 的 current-input diagonal field
之所以能达到 `1.937x`，正是因为它提高了当前状态带宽，而非因为 diagonal 比 DPLR
更强。

热力学/路径风险视角仍只提供 metric：应按下游 Jacobian、噪声阶段和 suffix 传播给
innovation 加权。本 Gate 使用 `W=I` 的局部 attention-output 范数，因而只判断接口
容量；它不能推出 reader task risk、video quality、denoising suffix 稳定或硬件收益。

### 9.6.3 决策和唯一高信息后续

train-free true-`2x2` centroid certificate 现已 parked，不再增加 BCM、PPE 或更多
position rule。下一项若继续，只应先做一个更强上界分解：在相同 `k=196` 下允许
per-head target-visible exact support，仍使用相同 coarse state。它回答：

1. 若 per-head ceiling 通过 `1%/2%`，主要瓶颈是共享 support，后续才值得训练低带宽
   head/group router；
2. 若 per-head ceiling 仍失败，主要瓶颈是 coarse state，下一步只能比较
   query-conditioned dipole/cross-moment state，而不能继续优化 selector；
3. 若 richer state 只有在接近四个原 token 的存储和 MAC 下才通过，则该方向没有
   压缩价值，应停止 external-measure 路线。

cross-moment 或 multipole attention 本身已有 FMM 类先例；潜在独立点只能是以
conditional innovation、下游风险和实测成本联合认证 state order，并对不可压缩
heads exact fallback，不能主张首次使用 moment hierarchy。

![Query-fixed visual measure 的预算曲线、分层退化与证书覆盖](../figures/query_fixed_measure_remainder.png)

## 9.7 Per-head support ceiling：共享路由是主瓶颈，但不是全部瓶颈

按照 9.6.3 的唯一后续，本轮保持同一 query、coarse state、true-`2x2` groups 和
`k=196` 预算，只允许 28 个 heads 分别选择 exact nodes。上一轮
`shared_exact_local` 在全部 `sample x layer x budget` cells 上逐项复现；full-shape
Q/K/V replay error 为 `0`。position-73 smoke 首次因 permutation guard 的 CPU/CUDA
device mismatch 在写结果前停止，唯一修复只改变检查张量 device；失败目录保留。

24 个 exposed 样本的正式判决为 `HEADWISE_SUPPORT_PARTIAL`：

| selector | visual mean / P95 / worst | full mean / P95 / worst |
|---|---:|---:|
| shared exact-local | 3.450% / 6.433% / 7.395% | 0.424% / 0.746% / 1.236% |
| headwise attention mass | 1.493% / 2.905% / 3.280% | 0.197% / 0.316% / 0.432% |
| headwise exact-local | **1.222% / 2.493% / 2.700%** | **0.145% / 0.239% / 0.368%** |
| headwise sequential greedy | 9.165% / 16.935% / 21.343% | 1.617% / 2.897% / 4.727% |

headwise exact-local 在所有 72 个 cells 上都是注册 headwise envelope 的 winner，相对
shared support 将 visual mean 降低 `64.58%`。但 mean/P95 仍高于 `1%/2%`，因此
不授权训练 support router。sequential greedy 再次显著恶化，说明 exact split 的误差
集合不是 submodular；逐步最小化当前 error 会消耗后续需要的 cancellation，不能把
“每步 target-visible”误称为全局 oracle。

分层结果给出异构边界：

| layer | shared mean | headwise mean / P95 / worst | relative gain |
|---:|---:|---:|---:|
| 0 | 1.300% | **0.400% / 0.459% / 0.482%** | 69.24% |
| 13 | 3.338% | 1.142% / 1.425% / 1.511% | 65.80% |
| 27 | 5.714% | 2.125% / 2.652% / 2.700% | 62.81% |

因此共享 support 是三层的主要误差源，但 layer 27 即使拥有 target-visible per-head
support 仍不能通过。一个统一 sparse router 不再合理；可行系统最多是 shallow
headwise support、mid/deep richer state 或 exact fallback。更重要的是，本轮 support
由 exact output 生成，不具备部署可观测性，所以 layer 0 的 capacity pass 也不是
runtime method win。

### 9.7.1 从 centroid 到 query-conditioned cross moment

令 `a_j=q^T delta k_j` 且节点内 `sum delta k=sum delta v=0`，则

\[
Z_g(q)=e^{q^T\bar k}\sum_j e^{a_j},
\qquad
N_g(q)=e^{q^T\bar k}\sum_j e^{a_j}(\bar v+\delta v_j).
\]

centroid 是零阶近似。展开到二阶得到：

\[
Z_g(q)\approx e^{q^T\bar k}
\left(m+\frac12\sum_j a_j^2\right),
\]

\[
N_g(q)\approx e^{q^T\bar k}
\left[
\bar v\left(m+\frac12\sum_j a_j^2\right)
+\underbrace{\sum_j a_j\delta v_j}_{m\Sigma_{vk}q}
+\frac12\sum_j a_j^2\delta v_j
\right].
\]

第一阶 cross term `Sigma_vk q` 会随当前 query/head 旋转，正好对应本轮观测到的
head-specific support；固定 BCM、固定低秩 basis 和共享 support 都无法表达它。
这不是把 low-rank 重新叠加回去，而是从 softmax measure 的充分统计量推导出的首个
缺项。

但成本边界同样严格：对 4-token node，`Sigma_vk` 的样本秩最多 3，直接保存/应用
其 factors 与重新读取四个 leaves 同阶，几乎没有压缩意义。只有在更大的空间或时间
node 中，cross-moment effective rank `r << m` 时才可能摊薄成本。因此下一 Gate 只
先验证 order-1/2/3 Taylor state 的容量；低阶不通过就停止，低阶通过也必须再做
larger-node rank/cost sweep，不能直接宣称加速。

![Per-head support 的预算曲线、分层边界与相对收益](../figures/query_fixed_headwise_support_ceiling.png)

## 9.8 Taylor smoke 的有效止损：odd-order expansion 不是正测度

后续 Taylor capacity Gate 没有产生 decision-bearing 结果。v1 position-73 smoke 在
写 row 前发现某些 odd-order truncated exponential 产生非正 group mass；没有 clamp，
而是按正测度前提停止。v2 试图逐 cell 标记无效 order，但把 float32 `exp` underflow
产生的数值零也判为物理负质量，导致 order-0 identity control 在汇总前停止。按照
预注册的一次 repair 上限，本 Gate 记录为 invalid engineering/function-family attempt，
不再进行第三次修复或 formal run；两个失败目录均保留。

这不是 cross-moment 的负结果。它只说明直接使用 odd Taylor polynomial 不满足我们
需要的 measure invariant。尤其

\[
p_2(a)=1+a+\frac12a^2
=\frac12\left[(a+1)^2+1\right]>0
\]

对所有实数成立，而 `p_1` 和 `p_3` 都可能给 member 负权重。下一项若获授权，应是
一个新的 **positive quadratic moment** Gate，而不是修补本 Gate：只测试 order-0
与 order-2，区分数学负值和数值 underflow，并先把 node 扩到 `4x4` 或时空层次节点，
同时限制 `Sigma_vk/Sigma_kk` factor rank 与实际 MAC/state bytes。只有在

\[
\text{visual mean/P95}\le 1\%/2\%,
\qquad
r\ll m,
\qquad
C_{\rm moment}<C_{\rm leaves}
\]

同时成立时，才训练 current-query-conditioned coefficient/router。这样得到的核心不再
是 BCM、low-rank、sparse 的并列叠加，而是：

> **以正的 query-conditioned measure state 表达稳定 bulk，以 per-head exact leaves
> 表达条件 innovation，并按层认证 state order 与 fallback。**

BCM/BCCB 只可能作为 larger-node 几何分组或离线 basis，不再进入 measure 主公式。

## 9.9 Positive Gaussian measure：失败来自闭包错误，不是 rank 不够

为避免 odd Taylor 的非正权重，本轮直接使用严格正的 Gaussian moment-generating
closure。对每个规则空间 node 保存 mean key/value 和低秩 key covariance，使用
当前 query 解析计算正的 numerator/denominator；exact pages 仍读取真实 K/V。
测试覆盖 exposed positions `73--96`、layers `0/13/27`、28 heads 和 rank
`0/2/4/8/16`。full-exact identity 与模型 Q/K/V replay 最大误差均为 `0`。

正式判决为 `NO_POSITIVE_GAUSSIAN_MEASURE_PATH`。最有利的 eligible 配置已经允许
target-visible `oracle_local` support，但在 `25%` exact spatial pages 下仍为：

| topology / rank / selector | active-read proxy | visual mean / P95 / worst | full mean / P95 |
|---|---:|---:|---:|
| spatial `7x7` / 0 / oracle-local | `3.698x` | `15.505% / 25.276% / 29.608%` | `1.747% / 2.535%` |

rank 不是缺少的容量。相同 topology、selector 和 `25%` exact budget 下，rank
`0/2/4/8/16` 的 visual mean 依次约为
`15.50%/19.83%/46.18%/62.26%/70.98%`。rank 增加使误差系统性恶化，而不是形成
逐渐饱和的 capacity curve。分层 rank-0 结果也从 layer 0 的
`4.71%/5.40%/5.45%` 恶化到 layer 27 的
`22.58%/27.77%/29.61%`。

原因可以直接从 Gaussian MGF 看出。若 node 内 key 真服从单峰 Gaussian，则：

\[
\mathbb E[e^{q^T K}]
=
\exp\left(q^T\mu_k+\tfrac12q^T\Sigma_{kk}q\right).
\]

但真实 node 同时包含多模态 key、偏态/重尾 score 与 value-score coupling。
covariance rank 越高，`exp(0.5 q^T Sigma q)` 越容易放大错误的方差方向；它不能
恢复 `E[e^{q^T K}V]` 中随 query 旋转的联合分布。rank-0 最优因此是明确的
**wrong-closure signal**，不是 low-rank state 的正向结果。继续增加 Gaussian
mixture 数量虽能扩展函数类，却会快速接近读取 leaves 的 state/MAC，并且没有当前
证据支持。

![Positive Gaussian measure 的 rank、预算和分层失配](../figures/query_fixed_positive_gaussian_measure.png)

## 9.10 Progressive exact pages：mass coverage 不能控制 value leverage

第二个 Gate 完全删除 compact bulk：metadata 只负责选择或认证 pages，输出只由
selected exact K/V 计算。它比较 deployable centroid score、Quest 风格 K-coordinate
min/max box bound，以及 target-visible exact-mass 与 local-output selectors。测试
使用同一 24 个 exposed 样本和三层；full-exact identity 为 `0`，Quest box 对所有
真实 page score 都形成上界，最大 violation 为 `-4.018`。

正式判决为 `NO_PROGRESSIVE_EXACT_PAGE_PATH`。`25%` spatial pages 的结果为：

| selector | selected visual mass | visual mean / P95 / worst | leaf-only read proxy |
|---|---:|---:|---:|
| centroid score | `54.72%` | `32.98% / 57.09% / 65.29%` | `4.0x` |
| Quest box bound | `48.66%` | `39.27% / 72.28% / 81.40%` | `4.0x` |
| exact-mass oracle | `61.81%` | `22.27% / 35.99% / 37.95%` | `4.0x` |
| local-output oracle | `60.81%` | `22.64% / 38.80% / 45.12%` | `4.0x` |

更关键的是 exact-mass 的预算曲线。读取 `50/62.5/75%` pages、覆盖
`83.04/89.57/94.40%` attention mass 时，visual mean error 仍为
`9.23/5.81/3.37%`。若 `S` 是已读集合、tail mass 为 `tau`，renormalized exact
output 满足：

\[
A-A_S
=
\sum_{j\notin S}a_j(v_j-A_S).
\]

所以小 `tau` 只有在遗漏 value 的 leverage 同时受控时才意味着小输出误差。当前
结果直接否定了“保留 95% attention mass 即安全”的经验替代目标；选择器必须建模
联合 numerator/denominator 缺陷或下游 reader risk。

Quest-style bound 数学上有效但不可用。`25%` exact-mass 配置的平均 tail-bound
looseness 约为 `10^10.63`；分层约为 `10^5.62/10^9.92/10^16.34`。高维
coordinate box 把 K 各维极值当作可同时达到，丢失维间相关性，深层越发保守。
因此 worst-case box 不能承担 progressive early-stop certificate；后续只能使用
分布校准的 reader-risk quantile，并明确其覆盖域。

![Progressive exact pages 的质量、mass 与证书松弛度](../figures/query_fixed_progressive_exact_pages.png)

## 9.11 统一判决：条件冗余仍成立，但可压缩接口必须学习

`EXP-004/005` 与 reader probes 并不矛盾。前者证明 current block input 可把部分 Wan
late-layer conditional risk 从 AR(2) 的 `0.141061` 降到 `0.072832`，但只在
layers `21/24/25` 通过 breadth gate；后者证明 query-aware support 有信息，却不能
由固定规则 page、低阶 moment 或 attention mass 转换成严格 fidelity。共同结论是：

\[
\boxed{
\text{conditional redundancy}
\neq
\text{fixed closure}
\neq
\text{cheap exact support}
\neq
\text{deployable speedup}
}
\]

因此应停止三项工作：继续增加 Gaussian/Taylor rank、继续优化固定 `7x7` page
selector、继续用 worst-case coordinate box 追求证书。当前唯一有信息增益的 reader
后续是一个小步训练 Gate：冻结 vision encoder 和 LLM，仅联合学习

1. semantic/event node construction；
2. query-conditioned value-aware scorer；
3. tiny node re-encoder，直接拟合联合 numerator/denominator innovation；
4. empirical upper-quantile risk gate 与 exact-leaf fallback。

训练目标必须比较 support-only、re-encoder-only 与 joint，并在相同 active-token、
state byte 和读带宽下证明 joint 的独立增益。positions `1--72` 只用于训练，
`73--96` 只用于选择，`97--120` 保持 untouched formal endpoint。selection 只有在
visual mean/P95 `<=1%/2%`、reader KL mean/P95 `<=0.01/0.02`、无 harmful flip 且
联合方法相对最佳单组件至少改善 `25%` 时才允许读取 formal。否则该 learned-memory
方向关闭，不进入 kernel 或 wall-clock。

这与 [Quest](https://arxiv.org/abs/2406.10774) 的 query-aware exact KV page selection、
[QTSplus](https://arxiv.org/abs/2511.11910) 的语义压缩和
[MemoryCard](https://arxiv.org/abs/2606.05917) 的 event/topic memory 有明显交集；不能
声称首次 query-aware retrieval 或 event memory。可验证的差异只可能是：**用
value-coupled downstream innovation 而非 attention mass 共同训练 node、support 与
calibrated exact fallback**。若 joint 不能在公平预算下显著胜过这些单独机制，就
没有足够的新方法贡献。

## 10. 工件

- `analysis/.../target_risk_budget_frontier_exposed_v1/`
- `analysis/.../group_compaction_geometry_exposed_v1/`
- `analysis/.../reader_aligned_singleton_marginal_exposed_v1/`
- `analysis/.../same_kernel_mass_equivalence_exposed_v1/`
- `analysis/.../batched_current_support_marginal_exposed_v1/`
- `analysis/.../true_2x2_geometry_exposed_v1/`
- `analysis/.../true_2x2_ppe_exposed_v1/`
- `analysis/.../query_fixed_measure_exposed_v2_repair1/`
- `analysis/.../query_fixed_headwise_exposed_v1/`
- `analysis/.../query_fixed_positive_gaussian_exposed_v1_repair4/`
- `analysis/.../query_fixed_progressive_exact_pages_exposed_v1/`
- `analysis/.../measure_preserving_compaction_invalid_attempts/`
- `figures/target_risk_compaction_geometry_audit.{png,pdf,svg}`
- `figures/reader_aligned_singleton_marginal_audit.{png,pdf,svg}`
- `figures/batched_current_support_marginal_audit.{png,pdf,svg}`
- `figures/true_2x2_geometry_control.{png,pdf,svg}`
- `figures/true_2x2_ppe_control.{png,pdf,svg}`
- `figures/query_fixed_measure_remainder.{png,pdf,svg}`
- `figures/query_fixed_headwise_support_ceiling.{png,pdf,svg}`
- `figures/query_fixed_positive_gaussian_measure.{png,pdf,svg}`
- `figures/query_fixed_progressive_exact_pages.{png,pdf,svg}`
- 每张图对应的 bound CSV 位于同一 `figures/` 目录。
