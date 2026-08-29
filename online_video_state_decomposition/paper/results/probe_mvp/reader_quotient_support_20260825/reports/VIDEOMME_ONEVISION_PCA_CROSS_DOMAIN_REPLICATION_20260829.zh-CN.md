# OneVision PCA-r456 Video-MME 跨域复现与方法判决

日期：2026-08-29
正式判决：`BOUNDARY`

## 1. 直接结论

未经重拟合的 MVBench calibration-only `PCA-r456+s0` codec 在 600 个唯一
Video-MME 视频上保留了大部分冻结 OneVision reader 行为，并继续提供 `7.86x`
状态压缩，但没有通过预注册的逐样本与有害事件门槛：

| 指标 | Full | PCA-r456 | 判决 |
|---|---:|---:|---|
| 样本 / 唯一视频 | 600 | 600 | 完整，零失败 |
| 准确率 | 55.17% | 54.17% | `-1.00pp [-2.17,+0.17]` |
| prediction agreement | - | 95.83% | 未过 98% |
| harmful / beneficial flips | - | 9 / 3 | harmful 上界 2.603%，未过 2% |
| 平均 / P95 candidate KL | - | 0.010884 / 0.043238 | 描述性 |
| feature relative L2 | - | 19.42% | 描述性 |
| tensor payload | 21.44 MiB | 2.73 MiB | 7.860x |

结果不是 `ADVERSE`：总体只损失 1 个百分点，short/medium/long 分别变化
`-2/-1/0pp`，均未触发 adverse 门槛。它也不是 `PASS`：agreement 和 harmful
上界均失败。因此最准确的结论是，MVBench 学到的 shared semantic bulk 能部分
跨视频域迁移，但它不是一个对冻结 reader 普遍成立的固定商空间。

## 2. 实验身份与无泄漏边界

- codec、rank、reader、候选 token scoring 均与此前 OneVision confirmation 相同；
- codec 只使用旧 MVBench calibration 样本拟合，Video-MME 不参与拟合或选择；
- 在任何正式模型输出产生前，以 seed `20260829` 冻结 short/medium/long 各
  200 个不同视频，每个视频只取一个问题；
- 排除此前 Qwen/CLIP probe 使用过的 30 个 Video-MME 视频，正式 split 与其
  重叠为 0；
- 两条路径使用相同的完整视频均匀 16-frame feature pool、相同 8-frame reader
  输入、问题、prompt 和候选；字幕均关闭；
- 600 个 checkpoint 全部完成，退出码为 0，失败、重复、非有限指标均为 0；
- 手工 feature 注入的最大 logit 差异为 0。

`paired_samples.csv` 由 checkpoint 文件名按字典序聚合，因此行顺序与冻结 split
顺序不同；独立复算确认二者样本集合完全一致，所有统计量与顺序无关。

## 3. 与 MVBench untouched confirmation 的对比

| 指标 | MVBench N=500 | Video-MME N=600 | 变化 |
|---|---:|---:|---:|
| Full/PCA accuracy delta | `+1.00pp` | `-1.00pp` | 方向反转 |
| prediction agreement | 96.80% | 95.83% | `-0.97pp` |
| harmful flips | 3 | 9 | 样本率 `0.60% -> 1.50%` |
| harmful upper 95% | 1.543% | 2.603% | 跨过 2% 门槛 |
| candidate KL mean | 0.004554 | 0.010884 | `2.39x` |
| feature relative L2 | 16.43% | 19.42% | `1.18x` |
| state compression | 7.86x | 7.86x | 不变 |

reader KL 增长远快于 feature L2，说明跨域失配不是简单的“视频更难重建”。固定
PCA 仍删除相似数量的欧氏能量，但这些方向在 Video-MME 问题分布下具有更高的
reader leverage。换言之，商空间取决于 reader、query 和数据域：

\[
D_q(X,\hat X)
\simeq
(X-\hat X)^\top G_{\theta,q}(X-\hat X),
\]

而 PCA 只优化 \(\|X-\hat X\|_2^2\)。同一个 \(U_r\) 能否迁移，取决于其丢弃
空间是否同时接近不同域的 \(G_{\theta,q}\)-null space；本轮结果只支持部分重叠。

## 4. 错误结构

25 个 prediction mismatch 可分为 9 harmful、3 beneficial 和 13
wrong-to-wrong。mismatch 样本的平均 candidate KL 为 `0.06942`，match 样本为
`0.00834`，相差 `8.32x`；两组 feature L2 却只有 `19.83%` 与 `19.40%`。
因此 raw feature error 仍不是可靠的 finite-perturbation 风险证书。

分组结果显示：

- duration 不是主要失配轴：short/medium/long 损失 `2/1/0pp`；
- Sports Competition 损失 `4pp`、agreement `92%`，Life Record 损失约
  `2.08pp`，Knowledge 反而提高约 `0.56pp`；
- Counting Problem 损失约 `5.88pp`，而 Information Synopsis、OCR、空间与
  时间感知点估计不变。

这些小分组只能定位候选风险，不能据此宣称计数或运动必然不可压缩。更稳健的
结论是：跨域错误呈 task/query 重尾，而不是随视频时长单调增长。

## 5. 与此前理论和失败实验的一致性

本轮支持“条件冗余不等于固定共享算子”的统一解释：

1. DiT 中固定 BCM/BCCB、跨样本静态 low-rank residual 失败，是因为内容坐标、
   step 和 query 改变了敏感方向；
2. 视频理解允许丢弃冻结 reader 的 null directions，因此简单 PCA bulk 比 raw
   residual predictor 更容易出现正结果；
3. 但 null space 仍随问题域旋转，所以 MVBench calibration basis 转到
   Video-MME 后，欧氏误差只温和增加，reader distortion 却显著放大；
4. 这对应条件风险分解中的函数族/域迁移项，而不是证明状态没有统计冗余。

热力学或 trajectory metric 对生成模型强调的是条件路径风险；这里相应的对象是
reader-induced metric。两者共同的核心不是某个固定矩阵族，而是：压缩必须在最终
功能度量下评估，低成本 observable 能否刻画该 metric 才决定可部署性。

## 6. 方法与工程决策

- 保留 `PCA-r456+s0` 作为简单、无 query 梯度的状态/传输压缩 baseline；
- 不把它称为严格 reader-equivalent state，也不从本轮启动直接读压缩态、prefill
  或 TTFT profiling；冻结协议只有 `PASS` 才授权该阶段；
- 不在这 600 个样本上调 rank、重新拟合 basis 或恢复动态 Fisher/BCM；
- 若继续科学验证，应换不同 reader；若提出新方法，则需另行冻结 multi-domain
  calibration 或 reader-weighted bulk，并用新的 untouched endpoint 检验；
- 若目标是明显计算加速，应减少 encoder/token 数或设计压缩态直读 kernel。当前
  codec decode 后仍恢复相同 token 数，7.86x payload 不等于 7.86x 计算加速。

## 7. 证据位置

- 冻结协议：`protocols/videomme_onevision_pca_cross_domain_replication_20260829.md`；
- 冻结 split：`configs/videomme/onevision_pca_cross_domain_600_20260829.json`；
- 正式汇总与逐样本结果：`analysis/videomme_onevision_pca_replication/`；
- 运行身份与零失败记录：`metadata/videomme_onevision_pca_replication_v1/`；
- 工程 smoke：`metadata/videomme_onevision_pca_smoke/`；
- 可视化：`figures/videomme_onevision_pca_replication.{png,pdf,svg}`。
