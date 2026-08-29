# 视频理解结构状态迁移、baseline 与后续方法收敛

日期：2026-08-25

## 1. 判决

本轮得到的是两个有边界的正结果、一个强读者边界复现、三个明确负结果和一个低上限诊断：

1. `exact recent + rank-256 slow state + 4 sparse spatial innovations` 的写入/编解码结构可迁移。旧五任务与新五任务共 800 个原生 LLaVA 样本中，压缩状态均未出现明显任务损失，状态从 `8.024 MiB` 降至 `1.024 MiB`，约 `7.84x`。
2. 旧任务上 query read 相对 recent 的 `+2.75pp` 没有迁移到新任务；新任务为 `0.00pp`。
3. CLIP 代理的 `+2.25pp` 不能替代原生 reader 结论。相同 400 组证据帧下，CLIP 与 LLaVA 的逐样本正确性 delta 相关仅 `0.146`；冻结 CLIP-confidence fallback 迁移收益仍为 `0.00pp`。
4. 原生 LLaVA first-token/candidate confidence 能弱地识别两条读取路径的差异，但最好的 confidence rule 仅为 `+0.25pp`；即使使用答案正确性选择路径，理论上限也只有 `+1.25pp`。
5. 在固定 `PCA-r256 + s4`、固定 `1,048,704` 字节下，当前 query/content 的原生 Fisher support oracle 将 candidate KL 降低 `72.07%`，Euclidean/Fisher 混合降低 `75.93%`；full-vocabulary KL、P95 和 candidate top-1 同向改善。
6. 把旧五任务 Fisher 平均成冻结 prior 后，在零重叠的新五任务上反而使 candidate KL 恶化 `190.65%`；mixed static prior 也恶化 `122.56%`。因此容量成立，但最简单的静态可观测近似明确失败。
7. 在 LLaVA-OneVision 和五个全新任务上，同字节 Mixed support 仍将聚合 KL 降低 `54.34%`、P95 降至 `0.577x`，但只有 `3/5` 任务为正、一个答案发生翻转，且 bootstrap CI 跨零，正式判决为 `BOUNDARY`。
8. 同预算 rank/support sweep 仍未使 Fisher 过 GO：最佳 `Fisher r438+s1` 配对改善 `24.19%`、`4/5` 任务为正且无 top-1 损失，差 `0.81pp` 未过门；但纯 `PCA-r456+s0` 的绝对 KL 相对 `r384+s4` Euclidean 降低 `64.52%`，说明 OneVision 当前更应把字节分给共享 bulk，而不是少量高维 exact token。

因此当前可保留的是**有界结构状态 writer/codec**以及**原生 reader 度量下的同码率 support 分配容量**，不能保留的是当前 CLIP query reader或冻结 Fisher 位置先验。强 reader 结果还说明，在训练 scorer 前必须先优化 low-rank/sparse 的同字节分配，使有限 residual 回到局部 Fisher 的可信范围。

## 2. 已完成证据

### 2.1 原生模型非劣性

| 数据 | N | full | compressed | 差异与 95% CI | benefit/harm | 状态 | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| 旧五任务、两个零重叠 split | 400 | 47.75% | 47.50% | `-0.25pp [-1.25,+0.50]` | 1/2 | 1.024 MiB | PASS |
| 新五任务 transfer | 400 | 32.75% | 33.00% | `+0.25pp [-0.75,+1.25]` | 3/2 | 1.024 MiB | BOUNDARY |

两批数据的 harmful-event one-sided 95% 上界均为 `1.5655%`，低于预注册 `2%`。新任务之所以是 `BOUNDARY` 而非 PASS，仅因为 full reader 准确率 `32.75%` 低于预注册的 `35%` 参考质量地板；不能在弱参考上宣称非劣。

### 2.2 同预算 mechanism proxy

在新 400 样本、8 evidence、16 pool、3 recent anchor 的冻结 CLIP feature proxy 中：

| 方法 | 相对 exact recent | 95% CI | 证据等级 |
|---|---:|---:|---|
| CausalMem proxy | +1.25pp | `[-2.00,+4.50]` | 机制代理 |
| StreamingTOM proxy | -0.25pp | `[-3.25,+2.75]` | 机制代理 |
| STC proxy | +0.25pp | `[-2.75,+3.25]` | 机制代理 |
| SelectStream proxy | -0.25pp | `[-2.50,+2.00]` | 未训练机制代理 |
| OASIS proxy | -0.50pp | `[-2.75,+1.75]` | 机制代理 |
| StateKV proxy | +0.25pp | `[-2.25,+2.75]` | 机制代理 |
| ours frozen selector | +2.25pp | `[+0.25,+4.50]` | CLIP 代理，不是 VLM 质量 |

这张表只能比较冻结 feature-level 行为，不能与论文官方准确率、TTFT 或 GPU latency 混合。尤其 ours 的正结果与 CLIP candidate embedding 同时参与选择和评分，存在 metric coupling；原生 LLaVA 已证明它不能迁移成 reader gain。

### 2.3 代理置信 fallback

只用旧五任务 200 个 proxy/native 对拟合

\[
\Delta m=m_{query}-m_{recent}
\]

阈值，得到 `tau=-0.01075`，实际等价于 99.5% 采用 query。它在旧任务复现 `+4.00pp`，但在新任务变为 100% query、`+0.00pp [-1.50,+1.50]`。`Delta m` 对 native benefit/harm 的 AUC 分别只有 `0.419/0.587`。因此 CLIP confidence 不是可靠风险证书。

### 2.4 原生置信度与两路径上限

在新任务 400 个样本上，保持证据帧、prompt、模型和解码完全不变，只额外记录 LLaVA 首 token 与候选 `A-D` 的 log-probability、margin 和 entropy。原有答案逐项复核为 `800/800` 一致，因此 instrumentation 没有改变数值语义。

| 策略 | 准确率 | 相对 exact recent | 95% CI | benefit/harm |
|---|---:|---:|---:|---:|
| exact recent | 33.00% | 0.00pp | - | 0/0 |
| always query | 33.00% | 0.00pp | `[-1.50,+1.50]` | 5/5 |
| candidate margin | 33.25% | +0.25pp | `[-0.75,+1.50]` | 3/2 |
| token margin | 33.25% | +0.25pp | `[-0.75,+1.50]` | 3/2 |
| correctness-visible oracle | 34.25% | +1.25pp | `[+0.25,+2.50]` | 5/0 |

candidate/token margin 对 benefit 的 AUC 约为 `0.587`，说明原生置信度比 CLIP margin 的方向更合理，但信号仍很弱。更重要的是，答案可见 oracle 已给出这两条路径的严格经验上限：它仍低于预设 `2pp` 的继续阈值。因而应停止开发 `recent/query` 二选一 controller；这个结论不否定新的 native-space selector、不同 evidence set 或 readout-aware codec。

### 2.5 Reader-Quotient support：容量与迁移分离

冻结 LLaVA-v1.5-7B、证据帧、PCA bulk、每帧四个 exact residual token 和状态字节，只改变 sparse support 的打分方式。对 40 个新任务样本，transductive native-Fisher oracle 得到：

| Variant | Candidate KL reduction | Vocab KL reduction | P95 ratio | Top-1 delta |
|---|---:|---:|---:|---:|
| Fisher s4 | `+72.07%` | `+71.95%` | `0.197` | `+2.50pp` |
| Mixed s4 | `+75.93%` | `+75.82%` | `0.136` | `+2.50pp` |

Fisher/Mixed 的 feature rel-L2 分别为 `6.482%/6.411%`，反而高于欧氏 support 的 `6.289%`。因此收益来自 native reader 的功能几何，而非更低的 feature 重构误差。

随后用旧五任务 40 个样本构造一次固定 `64x4096` Fisher prior，在与校准和 oracle 都零重叠的新五任务 40 个样本上冻结使用：

| Variant | Candidate KL reduction | 95% CI | P95 ratio | 判决 |
|---|---:|---:|---:|---|
| Channel prior | `+2.47%` | `[-2.98%,+8.88%]` | `0.919` | NULL ablation |
| Static Fisher | `-190.65%` | `[-568.94%,-15.14%]` | `3.297` | ADVERSE |
| Mixed static | `-122.56%` | `[-502.03%,+20.32%]` | `1.328` | ADVERSE |

这关闭了“跨任务平均 Fisher 就能部署”的路径。它没有否定 content-conditioned scorer，因为

\[
\operatorname{TopK}(E[G]\odot e^2)
\ne E[\operatorname{TopK}(G\odot e^2)],
\]

且实验证明真正有用的是随 query/content 旋转的 token support，而不是稳定但很弱的 channel marginal。详细证据见 `READER_QUOTIENT_SUPPORT_STUDY_20260825.zh-CN.md`。

### 2.6 LLaVA-OneVision 强读者复现

使用 Qwen2-7B backbone 的冻结 LLaVA-OneVision，在旧五任务 20 个样本上拟合 PCA-r384，并在五个此前未使用的 MVBench 任务、20 个新样本上评估。native state 为 `[16,196,3584]`，s4 状态 `2,867,328` bytes，相对 BF16 dense 为 `7.83965x`。

| Variant | KL reduction | 95% CI | P95 ratio | Positive tasks | Top-1 delta | 判决 |
|---|---:|---:|---:|---:|---:|---|
| Fisher s4 | `+44.24%` | `[-93.59%,+88.03%]` | `0.613` | 3/5 | -5pp | BOUNDARY |
| Mixed s4 | `+54.34%` | `[-44.12%,+88.79%]` | `0.577` | 3/5 | -5pp | BOUNDARY |

Standard pixel forward、手工 token injection 与 gradient instrumentation 的最大 logit 差异均为 `0`，reference accuracy 为 `60%`。但 Mixed 的点估计显著依赖一个样本；删除该样本后 leave-one-out reduction 为 `-15.02%`。该结果支持 capacity 跨架构存在，不支持稳定可部署性。详见 `ONEVISION_READER_QUOTIENT_REPLICATION_20260825.zh-CN.md`。

### 2.7 OneVision 同预算 rank/support allocation

使用单一 rank-456 calibration basis 的 ordered prefix，以每减少一个 exact residual token 换约 18 个 PCA rank。四个 shard、20 个样本全部完成。`Fisher r438+s1` 是最接近 GO 的 reader-aware endpoint，但 aggregate `24.19%` 低于冻结 `25%` 门槛，bootstrap CI 跨零，因此不授权 scorer 或 confirmation。

所有 Euclidean endpoint 的绝对 KL 随 bulk rank 单调下降，`PCA-r456+s0` 达到 `0.043739`，相对 anchor 的 `0.123267` 改善 `64.52%`，且答案准确率为 `60%`。这是 selection-set positive，不是跨任务 confirmation。详见 `ONEVISION_RANK_SUPPORT_ALLOCATION_20260825.zh-CN.md`。

### 2.8 官方/上游证据保持分组

- CausalMem 正式子集为 `206/250=82.4%`；OASIS 为 `209/250=83.6%`，但二者 backbone 不同，配对 McNemar `p=0.755`，不能归因到 memory 方法。
- STC 上游 A800 stage benchmark 中，ViT P50 `1681.37 -> 527.95 ms`，prefill `7587.77 -> 6551.79 ms`，stage sum `9761.02 -> 7062.34 ms`，约 `1.38x`；它不是 end-to-end TTFT。
- StreamingTOM 的 CTR、OQM write 和 selection 已有上游核心微基准，但测量域不同，不能相加成请求延迟。
- StateKV 明确还有完整 per-frame decode cache，因此“prefill recurrent state 固定”不等于系统总状态固定。

### 2.9 OneVision PCA-r456 untouched-task confirmation（2026-08-29）

冻结的 `PCA-r456+s0` 在五个未参与 calibration、reader replication 或
rank/support selection 的任务上完成了 500 样本确认。Full 与压缩态准确率分别为
`54.20%` 和 `55.20%`，配对差为 `+1.00pp`，bootstrap 95% 区间为
`[-0.20,+2.40]pp`。3 个 harmful flip 的单侧 95% Clopper-Pearson 上界为
`1.543%`，低于 2% 非劣门槛；tensor payload 从 `21.44 MiB` 降至
`2.73 MiB`，为 `7.86x`。

但 prediction agreement 只有 `96.80%`，未通过冻结的 98% 门槛，因此正式判决
仍为 `BOUNDARY`。这增强了“强 reader 的 diffuse semantic bulk 可压缩”证据，
没有证明压缩态与完整态逐样本可交换，也没有授权 serialization、prefill 或 TTFT
profiling。16 个变化预测中有 3 个 harmful、8 个 beneficial 和 5 个
wrong-to-wrong；不一致样本的平均 KL 约为一致样本的 7.5 倍，而 feature L2
几乎相同，进一步否定了 raw feature MSE 作为唯一风险度量。

## 3. 为什么历史结构在这里表现不同

跨 DiT、AR、WAM 和视频理解的结果可由同一个条件宽度解释：

\[
W_r^{cond}
=\inf_{phi,psi_r}
E_c\left\|
G_c^{1/2}[d_c-psi_r(phi(c))]
\right\|_2^2.
\]

- 固定 BCM/BCCB 失败，是因为 hidden channel 或内容 attention 不共享固定 Fourier eigenvectors。
- 静态 low-rank residual 失败，是因为单样本低秩子空间随内容、step、动作或 query 旋转。
- WAM frozen action condition 只让动态对角 defect 改善 `5.06%`，却没有恢复 action Jacobian；它学到相关性而非受控动力学。
- 视频理解不需要重建物理状态或完整 attention，只要保存对问题足够的证据，所以较大的 feature L2 误差仍可位于 reader 的任务 null space。

新 transfer 进一步说明：即使 writer 的任务商空间成立，外部 CLIP 定义的商空间也不等于原生 LLaVA 的商空间。Reader-Quotient oracle 又补充了更细的结论：原生功能度量可以显著改善同码率 codec，但其空间 support 随 query/content 旋转，跨任务静态平均会制造重尾错误。跨 DiT、WAM 与视频理解反复出现的瓶颈因此不是表示容量，而是**低成本可观测性**。

## 4. 与最新方法的创新边界

| 工作 | 已覆盖核心 | 我们不能再主张 | 尚可形成差异的位置 |
|---|---|---|---|
| [CausalMem](https://arxiv.org/abs/2606.25658) | training-free online semantic basis、固定预算 memory | 首个低秩/语义 memory | 原生 reader 风险加权的结构 codec |
| [SelectStream](https://arxiv.org/abs/2606.16353) | surprise write、固定图 memory、query retrieval、evidence calibration | 首个 query-conditioned evidence allocation | codec 内的功能等价与非劣证书 |
| [StateKV](https://arxiv.org/abs/2605.31598) | pretrained VLM 的固定 recurrent prefill state | 首个固定状态/线性 prefill | 完整状态字节与 readout-aware residual coding |
| [OASIS](https://arxiv.org/abs/2604.17052) | 层次事件 archive、uncertainty-triggered intent retrieval | 首个事件层次或按需 retrieval | 可与 archive 正交组合的 native feature codec |
| [StreamingTOM](https://arxiv.org/abs/2510.18269) | causal token reduction + INT4 online memory | 首个稀疏加量化 | 同一 reader metric 下联合分配 Q/S/L 误差 |
| [STC](https://arxiv.org/abs/2512.00891) / [EarlyTom](https://arxiv.org/abs/2605.30010) | ViT cache/prune 与 early encoder token compression | 仅压 LLM 前 token 即足够 | encoder、state、prefill 的完整系统核算 |
| [NovaCov](https://arxiv.org/abs/2608.01169) | 有界历史参考下的 set-wise submodular token selection | 首个 set-aware/coverage selector | 被选 native token 的功能状态压缩 |
| [ForestPrune](https://arxiv.org/abs/2603.22911) / [Script](https://arxiv.org/abs/2512.01949) | training-free 时空 pruning 与 query-conditioned semantic pruning | 首个 query-aware 视频 token 选择 | token 数固定后的 native-state rate-distortion |
| [CRAFT](https://arxiv.org/abs/2608.01644) | query-agnostic selection 与 position/content fusion | 首个位置/内容融合 selector | exact-byte codec 内 reader-sensitive residual allocation |

因此不再设计另一个 selector。最清晰的主线是 **Reader-Quotient Structured Memory (RQSM)**。

## 5. RQSM：单一目标，不是组件堆叠

令冻结 reader 为 `p_theta(y|q,X)`，状态 codec 为 `X_hat=D(C(X))`。目标是

\[
\min_C\; B[C(X)] + lambda T[C,D]
\quad\text{s.t.}\quad
E_q D_{KL}\left(
p_theta(\cdot|q,X)\|p_theta(\cdot|q,X_hat)
\right)\le epsilon.
\]

二阶近似使用原生 reader 的 query-conditioned GGN/Fisher：

\[
D_{task}(delta X)\simeq
delta X^T G_{theta,q} delta X.
\]

状态仍保持简单分解：

\[
M=R_{recent}\oplus U a\oplus I_{sparse}.
\]

- `R_recent` 精确保护当前场景；
- `U a` 保存原生 reader 敏感的长期共享方向；
- `I_sparse` 保存 GGN leverage 高的空间/事件 innovation；
- `a` 可量化，bit、rank 和 sparse support 在同一个 task distortion 下联合分配；
- rotation 仅作为 codec 内部可逆坐标变换，decode 后仍回到冻结 LLaVA 原坐标，不声称任意旋转是模型 symmetry；
- BCM/BTTB 只在空间 residual 已显示位移稳定时作为编码 basis。

本轮已经验证了最小的 `L + S + H`：PCA-r256 负责 bulk，四个 exact residual token 负责 sparse innovation，Fisher/GGN 只负责同字节 support allocation。Q 与 rotation 尚未加入，避免把有效的 support 结论和 value quantization 混在一起。Q、S、L、Hessian 和 rotation 的创新不在“同时存在”，而在它们共同求解同一个 exact-byte、native-readout rate-distortion 问题；若 joint 不能在同字节下显著优于各单分支，就不能保留联合创新主张。

## 6. 当前主线决策

同字节 nested allocation 已完成且没有 GO，因此停止 diagonal-Fisher support/scorer 主线。结果不支持继续调 mixture 权重或加入 rotation/BCM；它支持的更简单假设是 OneVision residual 在该预算下更偏 diffuse bulk。

`PCA-r456+s0` 的 untouched-task confirmation 已完成，但因 96.8% agreement
未过 98% 门槛而停留在 `BOUNDARY`。不能事后放宽门槛，也不进入
serialization、decode、LLM prefill、峰值显存或 TTFT profiling。若未来继续，
应换独立模型或视频域验证 reader quotient；若目标是计算加速，则应转向
STC/Script 一类 encoder/token-count 优化，而不是恢复动态 Fisher。

## 7. 当前结论边界

可以主张：冻结结构 codec 在十个 MVBench 任务、两批共 800 个原生样本上保持了预测，且状态约缩小 `7.84x`；强 reader 的简单 PCA bulk 又在五个 untouched tasks、500 样本上实现 `7.86x` payload 缩减、无 aggregate accuracy loss 和低于 2% 的 harmful 上界；query-specific native Fisher 在弱 reader 上显著优于欧氏 support，并在强 reader 上保留正向但不稳定的 capacity 信号。

不能主张：压缩态与 strong reader 逐样本等价、query reader 跨任务有效、原生 confidence 可以可靠选路、静态 Fisher 可部署、learned scorer 已有效、优于 CausalMem/SelectStream/StateKV 官方实现、已有真实 TTFT 加速、BCM/低秩存在普遍视频定理，或当前结果足以支持完整论文。

最重要的新 insight 是：

> 原生 reader 的功能等价类确实可压缩，但 OneVision 在约 2.86 MB 预算下首先需要更宽的共享 low-rank bulk；动态 sparse curvature 只有在 residual 已足够小、且跨任务稳定通过后才值得付出可观测性成本。
