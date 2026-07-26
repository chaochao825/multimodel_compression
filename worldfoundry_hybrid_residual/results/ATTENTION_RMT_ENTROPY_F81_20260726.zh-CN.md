# F81 Attention 随机矩阵与 Token Support Probe

日期：2026-07-26

样本：Wan2.1-T2V-1.3B，F81，layer 0，timestep 1000，conditional branch，32,760 tokens，12 heads，每头 128 个 stratified queries。

## 核心结论

F81 self-attention 不是“所有 head 都稀疏”或“所有 head 都低秩”，而是呈现强烈的 head-class 异质性：部分 head 几乎是全局 diffuse，部分 head 高度局部，另一些处于过渡状态。因此整层统一的 geometry mask、统一 low-rank residual 或统一量化策略都会浪费这种结构。

## Token Support

| Head 类别 | Heads | 典型 normalized entropy | Participation support / N | temporal-PM2 geometry mass | 含义 |
|---|---|---:|---:|---:|---|
| diffuse/global | 0, 2, 3, 8, 9, 10 | 88%–99.7% | 19%–93% | 4%–17% | 必须保留全局路径，固定局部稀疏不可用 |
| transitional | 1, 6, 7 | 62%–72% | 0.8%–2.9% | 65%–73% | 可能使用更宽 geometry、少量 global tiles 或动态 router |
| localized | 4, 5, 11 | 34%–49% | 低于 0.2% | 94% 左右 | 最适合完整 64x64 tile 的静态 geometry sparse kernel |

跨 12 个 head 平均：

- normalized entropy：75.60%；
- participation support：34.30% 的全部 token；
- top-1024 mass：51.56%；
- temporal-PM2 geometry mass：44.82%，与前一轮 geometry probe 一致；
- matched-Gaussian normalized entropy：69.91%。

真实 attention 在平均意义下甚至比保持逐 query logit 均值/方差的 Gaussian baseline 更 diffuse。部分 head 的集中主要由 logit scale 决定；但 localized heads 的 geometry mass 接近 94%，又证明它们确实包含空间/时间局部结构，而不只是随机尖峰。

## 随机矩阵结果

对每个 head 的 Q/K token-by-channel 矩阵逐通道标准化，再分析 128x128 correlation spectrum。样本数 32,760、维度 128 时，Marchenko-Pastur 上界为 1.1289。

| 指标 | Q | K |
|---|---:|---:|
| 最大特征值，head mean | 10.96 | 10.58 |
| 超过 MP 上界的特征值数量，head mean | 34.67 | 34.92 |
| stable rank，head mean | 14.16 | 14.80 |
| top-8 spectrum energy | 35.38% | 33.17% |

所以 Q/K 通道明显不是各向同性随机矩阵，存在强 channel subspace；但该结论只支持“通道存在结构”，不支持“token attention 可统一稀疏”。softmax、RoPE、head specialization 和时空位置共同决定最终 token support。

## 加速含义

当前 rank-16 current-replay oracle 在 2% error gate 下仍需 7/12 dense heads，effective execution density 为 61.32%。即使假设 sparse/dense head 完美融合且 correction、routing、gather 全部免费，结合 F81 self-attention profile share 53.88%，端到端 Amdahl 上限也只有：

```text
1 / ((1 - 0.5388) + 0.5388 * 0.6132) = 1.263x
```

若未来能稳定把 dense heads 降到 6/12，则理想 effective density 约 53.59%，端到端上限约 1.334x。这个空间存在，但距离目标 1.20x 的工程余量很窄，必须满足：

1. frozen defect basis 在独立 prompt/seed 上迁移；
2. dense 与 sparse heads 在同一 fused kernel 内执行，不能拆成多次 launch/gather；
3. localized/transitional head 分类在 layer、step、CFG branch 上稳定；
4. correction 同时处理 numerator 与 partition/LSE defect；
5. cache-aware refresh 只在 trajectory risk 允许时使用。

如果 held-out basis 或 head class 不稳定，固定 sparse-high-rank 主线应停止，转向低成本 sample-adaptive routing；不能再靠增加静态 low-rank/CM residual 强行修补。

本结果仅是单 replay 的结构筛查，不是多 trajectory 泛化或端到端视频质量证据。
