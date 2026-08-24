# OneVision 同预算 Low-Rank / Sparse Support 分配实验

日期：2026-08-25

## 1. 判决

按冻结协议，本轮为 **BOUNDARY**，不授权新任务 confirmation，也不授权训练 support scorer。

最接近门槛的是 `Fisher r438+s1`：

- 相对同 allocation 的 Euclidean support，candidate KL 改善 `24.19%`，低于冻结 `25%` 门槛 `0.81pp`；
- P95 ratio `0.972`，`4/5` 任务为正，无新增 candidate top-1 或答案准确率损失；
- 最差 leave-one-out 改善仍为 `+9.78%`，不是由单个样本单独制造；
- 但 bootstrap 95% CI 为 `[-86.25%,+55.53%]`，egocentric navigation 的 task KL 恶化 `568%`。

因此它支持“减小 residual 后 diagonal Fisher 更接近有效”，但不支持“Fisher support 已成为稳定可部署规则”。

本轮更强、也更简单的新发现是：纯 `PCA-r456+s0` 在 `2,860,032` bytes 下，将绝对 candidate KL 从 `r384+s4` 的 `0.123267` 降到 `0.043739`，改善 `64.52%`，答案准确率从 `55%` 回到 `60%`。它不使用测试时梯度、动态 support 或外部 proxy，但这是在当前 20 个 allocation-selection 样本上选出的配置，尚不能宣称跨任务确认成立。

## 2. 完整结果

| Variant | Bytes | KL sum | Paired reduction | Min LOO | P95 ratio | Absolute/anchor | Positive tasks | Top-1 delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Euclidean r384+s4 | 2,867,328 | 0.123267 | - | - | 1.000 | 1.000 | - | - | Reference |
| Fisher r384+s4 | 2,867,328 | 0.049616 | +59.75% | -2.11% | 0.463 | 0.403 | 2/5 | 0pp | BOUNDARY |
| Mixed r384+s4 | 2,867,328 | 0.055772 | +54.76% | -16.17% | 0.757 | 0.452 | 3/5 | 0pp | BOUNDARY |
| Euclidean r402+s3 | 2,865,504 | 0.092289 | - | - | 1.000 | 0.749 | - | - | Reference |
| Fisher r402+s3 | 2,865,504 | 0.058773 | +36.32% | -22.07% | 0.526 | 0.477 | 3/5 | 0pp | BOUNDARY |
| Mixed r402+s3 | 2,865,504 | 0.071683 | +22.33% | -51.34% | 0.990 | 0.582 | 2/5 | 0pp | BOUNDARY |
| Euclidean r420+s2 | 2,863,680 | 0.069919 | - | - | 1.000 | 0.567 | - | - | Reference |
| Fisher r420+s2 | 2,863,680 | 0.053202 | +23.91% | -16.29% | 0.387 | 0.432 | 4/5 | -5pp | ADVERSE |
| Mixed r420+s2 | 2,863,680 | 0.055251 | +20.98% | -21.00% | 0.484 | 0.448 | 4/5 | -5pp | ADVERSE |
| Euclidean r438+s1 | 2,861,856 | 0.049421 | - | - | 1.000 | 0.401 | - | - | Reference |
| Fisher r438+s1 | 2,861,856 | **0.037465** | +24.19% | +9.78% | 0.972 | **0.304** | 4/5 | 0pp | BOUNDARY |
| Mixed r438+s1 | 2,861,856 | 0.040751 | +17.54% | +0.23% | 0.972 | 0.331 | 3/5 | 0pp | BOUNDARY |
| Euclidean r456+s0 | 2,860,032 | **0.043739** | - | - | 1.000 | **0.355** | - | - | Reference |

四个远端 shard 均 exit `0`，共 `20/20` checkpoints，failure 数为 `0`。rank-456 basis 仅使用旧五任务 20 个 calibration 样本拟合，解释 calibration token energy `97.949%`。手工 feature injection 和 gradient instrumentation 的最大 logit 差异均为 `0`。

本轮 `r384` 是同一个 rank-456 basis 的 ordered prefix，不是上一轮独立拟合的 rank-384 basis。这一设计保证 allocation 只改变 prefix width 和 support budget；两轮 `r384+s4` 的绝对数值差异不能解释成方法增益，反而说明 PCA basis realization 也会改变后续 residual geometry。

## 3. 为什么更多 bulk rank 胜过 sparse exact token

设 feature pool 为 `F=16` 帧、每帧 `T=196` token、hidden `D=3584`。FP16 下增加一个 PCA rank 的状态成本为：

\[
\Delta B_{rank}=2FT=6,272\ \text{bytes}.
\]

每帧增加一个 exact residual token 的成本为：

\[
\Delta B_{sparse}=F(2D+2)=114,720\ \text{bytes},
\]

约等于 `18.29` 个 rank。冻结 sweep 正好以每少一个 sparse token 换约 18 个共享 rank。

Euclidean 绝对 KL 随该交换单调下降：`0.1233 -> 0.0923 -> 0.0699 -> 0.0494 -> 0.0437`。feature rel-L2 只从 `15.85%` 降到 `15.25%`，功能 KL 却降低 `64.5%`。这说明：

1. OneVision projected feature 的剩余误差更接近跨 token 共享的 diffuse bulk，而不是少量空间 token 的重尾 innovation；
2. 原生 reader 对该 bulk 高度各向异性，小幅 feature-L2 改善可带来很大的 readout-KL 改善；
3. 在 support 很少时，单个 exact 3584-D token 的机会成本过高；
4. 这不是“低秩普遍优于稀疏”，而是该 reader、该 state axis 和该字节预算下的 rate-allocation 结论。

这也解释了为什么历史 DiT/WAM 结果不能直接迁移。DiT 中要拟合的是随 step/content 旋转的运行时 defect，固定 basis 经常失败；这里压缩的是冻结视觉 projector 已经组织好的持久 feature state，reader 只需要其任务商空间，跨样本共享 bulk 因而更稳定。

## 4. Fisher 结果的准确边界

Diagonal Fisher 仍有真实容量：所有非零-support allocation 上，Fisher 的 aggregate KL 都优于 paired Euclidean。但稳定性不随 rank 单调改善：

- `r384+s4` aggregate 很强，却只有 `2/5` 任务为正；
- `r420+s2` 达到 `4/5`，但产生一个 top-1/accuracy 翻转；
- `r438+s1` 满足任务和离散答案 guard，却差 `0.81pp` 未达到 aggregate 门槛；
- 固定 50/50 Mixed 在本 sweep 中始终不优于 Fisher，说明混合不是免费的稳健化。

更高 rank 降低了有限扰动，却没有消除对角 Fisher 忽略 cross-token/cross-channel 曲率和任务重尾的问题。继续调 mixture 权重、追逐 `0.81pp` 或训练 scorer 都会违反当前证据边界。

## 5. 与相关工作的边界

[ForestPrune](https://arxiv.org/abs/2603.22911) 和 [Script](https://arxiv.org/abs/2512.01949) 是 training-free token pruning；[CRAFT](https://arxiv.org/abs/2608.01644) 通过位置/内容感知融合减少 token 数。它们优化“保留哪些/融合哪些 token”，本实验保持 reader token 数不变，优化持久 native state 中每个 token 的低秩坐标与 exact residual 字节分配。

[CausalMem](https://arxiv.org/abs/2606.25658) 使用在线 semantic basis 构造固定预算 memory，[StateKV](https://arxiv.org/abs/2605.31598) 使用固定 recurrent prefill state，[SelectStream](https://arxiv.org/abs/2606.16353) 做 query-conditioned evidence allocation。它们已覆盖低秩/固定状态/动态检索，因而不能主张这些组件首次出现。当前尚可形成差异的是：**在冻结 native reader 下，用 exact tensor bytes 和功能 KL 联合判定 low-rank bulk 与 sparse innovation 的 rate allocation，并显式分开 capacity、observability 与 finite-perturbation。**

[StreamingTOM](https://arxiv.org/abs/2510.18269) 联合 causal token reduction 与 INT4 memory，[STC](https://arxiv.org/abs/2512.00891) 同时优化 ViT cache 与 LLM prefill。它们说明最终工作必须测 TTFT/显存/reader quality；本轮只有 state quality，不含 kernel、prefill 或端到端速度声明。

## 6. 后续决策

1. 按 protocol 停止 diagonal-Fisher support/scorer 主线，不做阈值追逐。
2. `PCA-r456+s0` 作为最简单的 deployable allocation 候选保留，但只能称 selection-set positive；若继续，需另行冻结 untouched-task confirmation，不能复用这 20 个样本。
3. confirmation 若成立，再比较真实 serialization、decode、LLM prefill、峰值显存和 TTFT；若系统收益很小，转向 STC/Script 类 encoder/token-count 优化。
4. Q、rotation、BCM 不在这一 gate 中追加。只有 residual 的 token-space displacement 或量化误差出现稳定证据，才允许作为 codec 内部 basis，而不是为了组件完整性叠加。

## 7. 产物

- 冻结协议：`protocols/understanding_onevision_rank_support_allocation_20260825.md`；
- 原始结果：`results/understanding_onevision_rank_support_allocation/remote_v1/`；
- 分析表：`results/understanding_onevision_rank_support_allocation/analysis_v1/`；
- 可视化：`figures/onevision_rank_support_allocation.{png,pdf,svg}`；
- runner：`mvbench_onevision_rank_support_allocation.py`；
- launcher：`run_onevision_rank_support_allocation.sh`。
