# LLaVA-OneVision Reader-Quotient 强读者复现实验

日期：2026-08-25

## 1. 判决

按冻结协议，本轮判决为 **BOUNDARY**：功能容量信号跨 reader 复现，但稳定性、任务覆盖和离散答案保护没有同时过关。

在固定 `PCA-r384 + 每帧 4 个 exact residual token`、固定 `2,867,328` 字节状态下，相对 Euclidean s4：

| Variant | Candidate KL reduction | Vocab KL reduction | 95% CI | P95 ratio | Positive tasks | Top-1 delta |
|---|---:|---:|---:|---:|---:|---:|
| Fisher s4 | `+44.24%` | `+44.24%` | `[-93.59%,+88.03%]` | `0.613` | `3/5` | `-5.0pp` |
| Mixed s4 | `+54.34%` | `+54.34%` | `[-44.12%,+88.79%]` | `0.577` | `3/5` | `-5.0pp` |

这满足 aggregate `>=25%` 与 P95 不恶化，但没有满足至少 `4/5` 任务为正。20 个样本中还有一个 `object_interaction_0088` 从 reference 的正确答案 C 翻为 B；因此不能晋级为 GO。

更重要的是，Mixed 的 aggregate 点估计由 `fine_grained_pose_0061` 的大幅改善显著驱动。删除该样本后 leave-one-out reduction 为 `-15.02%`；bootstrap CI 跨零。当前正确表述是“跨架构 capacity 有复现迹象”，不是“强 reader 上方法已经稳定成立”。

## 2. 实验完整性

- 强 reader：冻结的 LLaVA-OneVision Qwen2-7B BF16，本地完整四 shard 权重；
- PCA calibration：旧五任务各 4 个样本，共 20 个样本、62,720 个 projected token；
- evaluation：五个此前未参与旧实验的新任务各 4 个样本，共 20 个样本；
- evaluation tasks：fine-grained pose、object interaction、action prediction、egocentric navigation、moving attribute；
- 每个样本：32 个 uniform source positions，后 16 帧构成 feature pool，后 8 帧进入 reader；
- native feature shape：`[16,196,3584]`；PCA-r384 calibration explained energy `97.49%`；
- BF16 dense state：`22,478,848` bytes；s4 state：`2,867,328` bytes；压缩比 `7.83965x`；
- standard pixel forward 与手工 projected-token injection 的最大 logit 误差为 `0`；
- gradient/no-gradient instrumentation 的最大 logit 误差为 `0`；
- 四个 shard 均 exit 0，20/20 checkpoints，failure files 全为空；
- reference 与 Euclidean s4 accuracy 均为 `60%`，Fisher/Mixed 为 `55%`。

首次 v1 执行在 16/20 样本后被一致性守卫判为 INVALID：support 打分前过早把 PCA reconstruction 转为 BF16，导致一个样本的 top-k 与正式 codec 的 FP32 residual 选择不一致。v1 被保留但完全排除；v2 修正为 FP32 support scoring、reader 注入时才转 BF16。

## 3. 为什么结果弱于 LLaVA-v1.5

### 3.1 局部度量遇到较大的有限扰动

Diagonal Fisher 使用局部二阶近似：

\[
D_{KL}(p_X\|p_{X+\delta})
\simeq \frac12\delta^T G_X\delta.
\]

弱 reader 的 Euclidean s4 feature rel-L2 为 `6.289%`，OneVision 上升到 `15.902%`；PCA calibration explained energy 也从约 `99.44%` 降到 `97.49%`。当 `delta` 已是 16% 量级时，对角局部曲率会忽略：

- token/channel 间 Fisher cross terms；
- 多层非线性传播；
- 由压缩引起的敏感方向旋转；
- top-k support 的离散改变。

因此 Fisher 在平均上仍能找到高杠杆 residual，却不能稳定排序所有样本。

### 3.2 KL 改善不保证 argmax 保持

Candidate KL 是平滑分布距离，答案选择是非连续的 argmax。`object_interaction_0088` 中 Fisher/Mixed KL 从 `0.007802` 小幅降至 `0.007614`，但近边界的 B/C 排序发生翻转。因而后续目标至少需要同时考虑：

\[
D_{KL}(p\|\hat p)
\quad\text{与}\quad
m_{c^*,j}=\ell_{c^*}-\ell_j.
\]

这不是把 accuracy 当可微损失，而是为接近零的 reference margin 提供显式保护。

### 3.3 Reader-sensitive support 跨架构变化更大

OneVision Fisher 与 Euclidean support 的平均 overlap 只有 `7.66%`，Mixed 为 `25.16%`；弱 reader 分别为 `42.58%/55.94%`。这说明“哪些 native token 值得 exact payload”高度依赖 vision projector、LLM reader 与 prompt 几何，不能把一个模型上的 Fisher prior 或固定 BCM/位置结构迁移到另一个模型。

## 4. 与历史实验的统一解释

到目前为止，DiT、WAM 和视频理解反复出现三个不同层级：

1. **表示容量**：adaptive rank、target-visible coefficient、query-specific Fisher 经常很强；
2. **低成本可观测性**：frozen basis、past-only predictor、CLIP proxy、static Fisher prior 经常失败；
3. **局部 surrogate 有效域**：即使可观测 oracle 为正，有限压缩误差也可能超出 SVD/Fisher/Taylor 的局部近似范围。

本轮把第三层显式暴露出来。此前不能简单总结为“低秩或 Hessian 无效”；更准确的是：

> low-rank bulk、sparse exact innovation 和 reader curvature 的组合有真实容量，但 rank/sparsity 的字节分配必须先把 residual 控制在局部度量的可信范围内，随后才有资格学习 scorer。

这也重新定位了 `Q + S + L + Hessian + rotation`：当前只验证了 `L + S + H` 的容量；Q、rotation 和 noise shaping 尚未进入。现在加入它们只会掩盖 rank/support 分配错误，不能形成“怎样叠加”的清晰贡献。

## 5. 与相关工作的边界

[ForestPrune](https://arxiv.org/abs/2603.22911)、[Script](https://arxiv.org/abs/2512.01949) 和 [CRAFT](https://arxiv.org/abs/2608.01644) 分别覆盖 training-free 时空 pruning、query-conditioned semantic pruning，以及 position/content-aware token fusion。它们主要减少 token 数或融合 token；本轮固定 reader token 数，优化的是被存储 native feature 的 exact-byte 内容状态。

[CausalMem](https://arxiv.org/abs/2606.25658)、[SelectStream](https://arxiv.org/abs/2606.16353)、[StateKV](https://arxiv.org/abs/2605.31598) 与 [StreamingTOM](https://arxiv.org/abs/2510.18269) 已覆盖 semantic basis、query retrieval、recurrent prefill state 和 pruning/INT4 memory。因此创新边界不能是“首次低秩、稀疏或 query-aware memory”，而应是：

> exact-byte、native-readout rate-distortion 下的 rank-support-curvature 协同分配，以及 capacity、observability 与 finite-perturbation 三层误差的可验证分解。

## 6. 后续 allocation gate 已完成

未训练 scorer。后续使用同一个 rank-456 calibration basis 的前缀，完成了近似同字节的 nested allocation sweep：

| Rank | Exact residual/frame | Estimated bytes |
|---:|---:|---:|
| 384 | 4 | 2,867,328 |
| 402 | 3 | 2,865,504 |
| 420 | 2 | 2,863,680 |
| 438 | 1 | 2,861,856 |
| 456 | 0 | 2,860,032 |

结果仍为 `BOUNDARY`。`Fisher r438+s1` 达到 `24.19%` 配对 KL 改善、P95 `0.972x`、`4/5` 任务为正且无 top-1 损失，但低于冻结 `25%` 门槛；因此不训练 scorer。与此同时，纯 `PCA-r456+s0` 的绝对 KL 相对 `r384+s4` Euclidean 降低 `64.52%`，成为更简单的 selection-set candidate。完整结果见 `ONEVISION_RANK_SUPPORT_ALLOCATION_20260825.zh-CN.md`。

当前 20 样本仍只可作为 allocation selection 集。由于没有配置达到 GO，diagonal-Fisher support 主线停止。若继续验证纯 PCA-r456，必须另行冻结剩余任务，且不能把该选择集结果表述为 confirmation。

## 7. 产物

- 协议：`protocols/understanding_onevision_reader_quotient_replication_20260825.md`；
- 正式结果：`results/understanding_onevision_reader_quotient_replication/remote_v2/`；
- 统计：`results/understanding_onevision_reader_quotient_replication/analysis_v2/`；
- 图：`figures/onevision_reader_quotient_replication.{png,pdf,svg}`；
- 原始绘图表：`figures/onevision_reader_quotient_replication_data.csv`；
- 实现：`mvbench_onevision_utils.py`、`fit_mvbench_onevision_feature_pca.py`、`mvbench_onevision_reader_quotient_oracle.py`。
