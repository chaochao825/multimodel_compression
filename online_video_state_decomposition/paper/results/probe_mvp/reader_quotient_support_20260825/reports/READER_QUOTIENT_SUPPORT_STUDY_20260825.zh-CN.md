# Reader-Quotient Support：同码率原生读出优化与静态迁移边界

日期：2026-08-25

## 1. 总判决

本轮验证了一个正命题，也关闭了它最简单的部署近似：

1. **表示上限为正。** 在固定 `PCA-r256 + 每帧 4 个 exact residual token`、固定 `1,048,704` 字节状态下，按当前 query/content 的原生 LLaVA Fisher 选择 support，可将聚合 candidate KL 降低 `72.07%`；与欧氏分数等权混合后降低 `75.93%`。
2. **不是 candidate-only 指标投机。** full-vocabulary first-token KL 同步降低 `71.95%/75.82%`，P95 candidate KL 降至欧氏基线的 `19.66%/13.60%`，candidate top-1 agreement 从 `97.5%` 提高到 `100%`。
3. **静态迁移失败。** 用旧五任务 40 个样本校准固定 `64x4096` Fisher prior，再在完全不重叠的新五任务 40 个样本上应用，完整 prior 与 mixed prior 分别使聚合 KL 恶化 `190.65%/122.56%`；P95 变为 `3.30x/1.33x`。
4. **channel marginal 近似为 null。** 它只改善 `2.47%`，95% CI 为 `[-2.98%, +8.88%]`。空间位置 prior 则是主要灾难来源，episodic reasoning 上 mixed prior 恶化约 `5.38x`。

因此可以保留的主张是：

> 低秩 bulk、稀疏 exact innovation 与原生 readout metric 的联合分配，在同字节下明显优于欧氏 codec；但有效 support 是 query/content-conditioned 的，不能用跨任务平均 Fisher 或固定位置结构替代。

这不是可部署方法。当前 exact Fisher 需要每个样本四次候选梯度，平均约 `439.8 ms`，而一次对应 reader forward 约 `24.5 ms`。它只证明方向值得在强 reader 上复现和学习廉价 scorer。

## 2. 实验一：transductive native-Fisher oracle

### 2.1 公平条件

- 冻结 LLaVA-v1.5-7B 与原 rank-256 codec；
- 冻结 `exact_recent` 八帧 evidence；
- 每帧均存四个 FP16 residual vectors 和四个 int16 indices；
- PCA latent、状态字节、prompt、candidate token 和 forward 路径相同；
- 梯度 instrumentation 与普通 forward 的 logits 最大差异为 `0`；
- 欧氏 top-k support 与原 codec `40/40` 完全一致；
- 40 个样本，每个 transfer task 八个。

对 full feature 下候选分布定义

\[
G_{i,d}=\sum_y p_y
\left(\frac{\partial \log p_y}{\partial X_{i,d}}\right)^2,
\qquad
s_i^F=\sum_dG_{i,d}(X_{i,d}-\hat X_{i,d})^2.
\]

`mixed_s4` 使用逐帧归一化后的欧氏分数与 Fisher 分数等权平均。没有按测试结果选择混合系数。

### 2.2 聚合结果

| Variant | Candidate KL reduction | Vocab KL reduction | 95% CI | P95 ratio | Top-1 delta | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| Fisher s4 | `+72.07%` | `+71.95%` | `[+51.67%, +83.26%]` | `0.197` | `+2.50pp` | 26/2/12 |
| Mixed s4 | `+75.93%` | `+75.82%` | `[+60.06%, +84.71%]` | `0.136` | `+2.50pp` | 25/3/12 |

五个 task 的聚合 KL 均改善。Mixed 的 task reduction 为：

| Task | Reduction |
|---|---:|
| action localization | `20.68%` |
| character order | `70.88%` |
| episodic reasoning | `78.76%` |
| moving count | `46.04%` |
| object shuffle | `87.72%` |

action localization 低于单任务 `25%`，而 12/40 样本仍会退化。因此它是明显的聚合上限，不是逐样本安全证书。

### 2.3 为什么这个结果重要

欧氏、Fisher、Mixed 的平均 feature rel-L2 分别为 `6.289% / 6.482% / 6.411%`。Fisher 在 feature L2 上更差，却在 reader KL 上好得多。这直接证明：

\[
\|X-\hat X\|_2
\quad\text{不是}\quad
D_{KL}(p_\theta(\cdot|X)\|p_\theta(\cdot|\hat X))
\]

的可靠 surrogate。历史 BCM、固定低秩 residual 和静态 sparse support 的一部分失败来自优化错误的几何，而不是所有功能信息都不可压缩。

Fisher 与欧氏 support 平均只重合 `42.58%`，Mixed 为 `55.94%`。这说明增益并非对相同 token 做微小重排，而是把 exact payload 移到了不同的 reader-sensitive token 上。

## 3. 实验二：calibration-only static prior transfer

### 3.1 无泄漏设计

- calibration 只使用旧五任务 40 个样本；
- evaluation 使用另一组新五任务 40 个样本；
- 先前 oracle 的 40 个 sample ID 全部排除；
- evaluation 不计算梯度、不更新 prior、不调整混合系数；
- 所有 s4 variant 状态字节相同；
- 固定 prior 为 FP32 `64x4096`，共 `1,048,576` 字节全局模型元数据；
- 同时测试 position、channel、separable、full diagonal 和 fixed mixed。

静态 scorer 一次计算全部五个候选平均只需约 `0.357 ms`，工程成本足够低，因此失败是科学失败，不是 Python overhead 掩盖收益。

### 3.2 结果

| Variant | Candidate KL reduction | Vocab KL reduction | P95 ratio | W/T/L | 判决 |
|---|---:|---:|---:|---:|---|
| position s4 | `-189.82%` | `-189.62%` | `3.297` | 13/1/26 | ablation |
| channel s4 | `+2.47%` | `+2.49%` | `0.919` | 8/25/7 | null ablation |
| separable s4 | `-186.46%` | `-186.24%` | `3.046` | 11/2/27 | ablation |
| static Fisher s4 | `-190.65%` | `-190.45%` | `3.297` | 15/0/25 | ADVERSE |
| mixed static s4 | `-122.56%` | `-122.57%` | `1.328` | 14/4/22 | ADVERSE |

static mixed 只在 character order 与 object shuffle 上改善 `13.78%/6.00%`，在 action localization 基本持平，在 moving count 恶化 `32.36%`，在 episodic reasoning 恶化 `538.47%`。

### 3.3 失败机制

固定 Fisher prior 不是普通意义上的噪声平均。它将旧任务中高杠杆的空间位置长期放大；当新 query 的敏感位置旋转时，top-k 是离散操作，少量排序变化会把 exact residual 从当前关键 token 移走。其风险可写为：

\[
\bar G=E_{q,x}[G(q,x)],
\qquad
\operatorname{TopK}(\bar G\odot e^2)
\ne
E_{q,x}[\operatorname{TopK}(G(q,x)\odot e^2)].
\]

Top-K 与期望不交换，而且错误位置 prior 会产生重尾退化。channel marginal 接近欧氏，说明跨任务稳定部分主要是弱通道缩放；真正有用的增益来自随 query/content 变化的 token support。

这与历史结果一致：

- DiT 的 adaptive rank-16 tail 很强，但 frozen basis 失败；
- WAM 的 target-visible temporal coefficient 很强，但 past-only causal predictor 失败；
- CLIP selector proxy 为正，但原生 LLaVA reader 不迁移；
- 本轮 query-specific Fisher 很强，但 frozen Fisher prior 失败。

共同瓶颈不是矩阵容量，而是**低成本可观测性**。

## 4. 与相关工作的边界

[ForestPrune](https://arxiv.org/abs/2603.22911) 使用 training-free 时空 forest 进行视频 token pruning；[Script](https://arxiv.org/abs/2512.01949) 组合图结构与 query-conditioned semantic pruning；[CRAFT](https://arxiv.org/abs/2608.01644) 将 query-agnostic selection 与可训练的 position-aware/content-adaptive fusion 分开。它们都说明视频 token 冗余真实存在，但重点是减少 token 数或融合 token。

本轮对象不同：token 数不变，压缩的是已选 native visual token 的**内容状态**；低秩 bulk 与 sparse exact residual 的字节完全固定，只改变 sparse payload 在 reader metric 下的分配。它更接近 task-aware rate-distortion，而不是另一个 selector。

[CausalMem](https://arxiv.org/abs/2606.25658)、[SelectStream](https://arxiv.org/abs/2606.16353)、[StateKV](https://arxiv.org/abs/2605.31598)、[OASIS](https://arxiv.org/abs/2604.17052)、[StreamingTOM](https://arxiv.org/abs/2510.18269) 和 [STC](https://arxiv.org/abs/2512.00891) 已分别覆盖语义 basis、query retrieval、recurrent prefill state、事件 archive、token reduction/INT4 和 ViT cache/prune。因而不能主张首次低秩、稀疏、query-aware memory 或 token compression。

当前可形成差异的命题是：

> 在 exact-byte native-feature codec 中，联合优化 low-rank bulk 与 sparse innovation 的关键不是组件叠加，而是用冻结 reader 的功能商空间决定 exact payload；并显式分解 transductive capacity、静态迁移与可观测 scorer 三层差距。

## 5. 对 Q + S + L + Hessian + rotation 的重新定位

这轮已经验证了最简洁的 `L + S + H` 组合：

- `L`：PCA-r256 表示高维 bulk；
- `S`：固定四个 exact residual token；
- `H`：Fisher/GGN 只用于同码率 support allocation；
- `Q`：尚未改变 residual value 精度，避免和 support 结论混淆；
- rotation：没有加入，因为静态坐标变换不能解决 query-conditioned support 旋转。

这比机械叠加更优雅，因为三者求解同一个约束问题：

\[
\min_{U,a,\Omega}
D_{reader}\left(X,Ua+S_\Omega\right)
\quad
\text{s.t.}\quad B(Ua,S_\Omega)\le B_0.
\]

量化应只在 reader-aware support 可迁移后加入：先比较 residual value FP16/INT8/INT4 在同一 native KL 下的 bit allocation，再决定是否需要 rotation/noise shaping。现在加入量化只会混淆静态 support 失败的根因。

## 6. 下一步唯一高信息 gate

强 reader capacity replication 已执行：LLaVA-OneVision Mixed support 在同样约 `7.84x` 状态压缩下得到 `54.34%` aggregate KL reduction 与 `0.577x` P95，但只有 `3/5` 任务为正，一个答案翻转，bootstrap CI 跨零，因此判为 BOUNDARY，而不是 GO。

下一步不直接训练 scorer，而是先做同字节 rank/support allocation sweep：`(r,s)=(384,4),(402,3),(420,2),(438,1),(456,0)`。它检验 OneVision 的 `15.9%` feature residual 是否已经超出 diagonal Fisher 的局部二阶有效域。最多加入一个固定 winner-vs-runner margin sensitivity 消融，不能在当前 20 样本上调权。

只有配置同时满足 aggregate `>=25%`、P95 `<=1`、至少 `4/5` 任务为正且 top-1 不下降，才在剩余五个未使用任务上冻结确认；确认通过后才训练小型 `g_phi(q,x_i,e_i,p_i)` ranking scorer。若 allocation sweep 失败，应停止 diagonal-Fisher support 主线，保留欧氏 codec 并转向 encoder/prefill 系统优化。

## 7. 可视化与产物

- `figures/reader_quotient_support_oracle.{png,pdf,svg}`：同码率 oracle 结果、逐样本 KL、任务分解和 support overlap。
- `figures/reader_quotient_transfer_gap.{png,pdf,svg}`：oracle 到静态迁移的落差与 P95 风险。
- `results/understanding_reader_quotient_support_oracle/analysis_v2/`：正式 oracle CSV/JSON。
- `results/understanding_reader_quotient_static_prior/analysis_v2/`：正式静态迁移 CSV/JSON。
- `results/understanding_onevision_reader_quotient_replication/analysis_v2/`：强 reader BOUNDARY 复现。
- `figures/onevision_reader_quotient_replication.{png,pdf,svg}`：跨 reader、P95、任务异质性与逐样本重尾。
- 三个冻结协议位于 `protocols/`，实现和测试位于 source repository 的 `experiments/probes/` 与 `experiments/tests/`。
