# OneVision PCA-r456 untouched-task 确认与潜力判决

日期：2026-08-29
正式判决：`BOUNDARY`

## 1. 直接结论

`PCA-r456+s0` 在五个从未参与 calibration、Reader-Quotient replication
或 rank/support selection 的 MVBench 任务上显示出真实的 reader-preservation
潜力，但没有通过预注册的逐样本等价门槛：

| 指标 | Full | PCA-r456 | 判决 |
|---|---:|---:|---|
| 样本 | 500 | 500 | 完整 |
| 准确率 | 54.20% | 55.20% | `+1.00pp [-0.20,+2.40]` |
| prediction agreement | - | 96.80% | 未过 98% 门槛 |
| harmful / beneficial flips | - | 3 / 8 | harmful 上界 1.543%，通过 2% 门槛 |
| 平均 / P95 candidate KL | - | 0.004554 / 0.016917 | 描述性 |
| feature relative L2 | - | 16.43% | 描述性 |
| tensor payload | 21.44 MiB | 2.73 MiB | 7.860x |

五个任务的最坏准确率变化仅为 `-1pp`，其余分别为 `0/+1/+3/+2pp`。
因此结果不支持“几乎逐样本等价”，但支持更窄的命题：在冻结 reader 的
任务正确性下，宽共享 PCA bulk 可以跨任务保存大部分功能状态。

## 2. 首轮无效尝试与无泄漏修复

初始 `v1` 从原始 MVBench 记录直接抽样。运行到 398 个 checkpoint 时，两个
shard 在 `counterfactual_inference` 的单候选记录上被多选 helper 正确拒绝；第三
个 shard 正常完成。该尝试没有聚合或读取科学指标，完整日志保留。

`v2` 在任何模型输出读取前冻结一个统一的 schema eligibility：候选数必须在
`[2,26]`，且答案必须存在于候选列表。五个任务中只排除 5 条
`counterfactual_inference` 单候选记录，随后用原 seed 重新抽取每任务 100 个。
`v2` 的 500 个唯一 checkpoint、三个空失败列表和三个退出码 0 构成正式证据。

## 3. 为什么是 BOUNDARY，而不是失败

16 个变化预测可分为：

- `3` 个 full-correct / compressed-wrong；
- `8` 个 full-wrong / compressed-correct；
- `5` 个 wrong-to-wrong。

冻结的 98% agreement 门槛对任何预测变化一视同仁，因此即使净准确率提高，
`96.8%` 仍必须判为 `BOUNDARY`，不能事后放宽门槛。

不一致样本平均 candidate KL 为 `0.02839`，一致样本为 `0.00377`，约高
`7.5x`；但对应 feature L2 仅为 `16.59%` 与 `16.42%`。这再次说明 raw
feature L2 不能可靠表示 reader 风险。另一方面，三个 harmful flip 的平均 KL
只有 `0.00592`，所以单一全局 KL 阈值也不能保证捕获全部有害事件。

## 4. 与此前结构结果的统一解释

令原生视觉状态为 `X`，PCA codec 为

\[
Z=(X-\mu)U_r,\qquad \hat X=ZU_r^\top+\mu,
\]

冻结 reader 的局部缺陷近似为

\[
\delta \ell
\simeq
J_{X\rightarrow\ell}(I-U_rU_r^\top)(X-\mu).
\]

PCA 只最小化被丢弃的欧氏能量；本实验的正向准确率表明，大量 `16.43%`
feature 误差落在 reader 的低敏感商空间中。它不证明视频状态普遍 rank-456，
也不证明这一 basis 对别的 reader、prompt 或视频域成立。

这和 DiT 生成侧并不矛盾：生成要求每一步向量场误差在 rollout 中稳定，固定
basis 的小误差会累计；视频理解只要求最终冻结 reader 的条件分布或答案保持，
可以丢弃 reader-null 方向。此前 BCM/BCCB 失败是固定 Fourier 特征向量与内容
坐标错配；这里成功的是从 calibration 数据学到的共享 semantic bulk，不是循环
位移算子。

## 5. 潜力分层

### 表示与内存：中高潜力

- 五个新任务、500 样本上没有准确率损失，且 harmful 上界通过；
- 7.86x tensor payload 缩减跨 weak/strong reader 均出现；
- 方法无需 query gradient、动态 support、BCM 或训练 scorer。

### 部署等价：边界

- 96.8% agreement 未过 98%；
- `action_count` agreement 只有 95%，`fine_grained_action` 为 96%；
- 当前不能把 compressed state 当作逐样本可交换的完整 state。

### 计算加速：尚未证明，潜力有限于系统实现

当前 codec 压缩 hidden width，但 reconstruction 后仍向 LLM 提交相同数量的视觉
token，因此 attention/FFN prefill FLOPs 基本不变。它直接节省的是持久状态、
传输和缓存带宽；只有 reconstruction 融合、压缩态直读或进一步减少 token 数，
才可能形成明显 TTFT 收益。根据冻结协议，`BOUNDARY` 不授权系统 profiling，
本轮没有产生速度主张。

### 论文创新：单独 PCA 不足

低秩视觉 memory 已有大量先例。可保留的研究贡献只能是：以原生 reader
rate-distortion 和有限样本非劣证书联合分配 exact bytes，并明确区分
representation capacity、低成本 observability、finite-perturbation 和系统收益。
当前强 reader 结果仍不足以闭合这一整条链。

## 6. 决策

按冻结协议，强 reader `PCA-r456+s0` 停留在 `BOUNDARY`，不进入
serialization/prefill/TTFT profiling，也不重启 Fisher scorer、rotation 或 BCM。
可以保留的正结论仍是：视频理解的 reader quotient 具有明显可压缩性，而且简单
共享 bulk 比当前动态 sparse support 更稳；若未来重新打开，应先更换独立模型或
视频域做 replication，而不是在这 500 个样本上调 rank 或 agreement 阈值。

## 7. 证据

- 冻结协议：`protocols/understanding_onevision_pca_r456_confirmation_20260829.md`；
- 正式结果：`analysis/onevision_pca_r456_confirmation/`；
- 正式运行元数据：`metadata/onevision_pca_r456_confirmation_v2/`；
- 首轮无效运行：本报告第 2 节记录原因、停止点和修复边界，原始日志保留在运行端与本地工作区但不随仓库发布；
- 可视化：`figures/onevision_pca_r456_confirmation.{png,pdf,svg}`。
