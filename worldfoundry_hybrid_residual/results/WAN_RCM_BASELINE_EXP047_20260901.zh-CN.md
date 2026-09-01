# EXP-047：released rCM 四步 H200 质量与速度判决

日期：2026-09-01
状态：有效 `speed-boundary`

## 结论

官方 BF16 rCM 四步模型已经证明了一个很强的正向机制：在冻结的四个
prompt、两个 seed、F81 设置下，它将 denoiser 时间从 `32.202s` 降到
`3.177s`，同时 8 维项目集 VBench 的 teacher-normalized 均值为
`0.9969`、最低维度为 `0.9706`，跨 seed 多样性也没有坍塌。

但完整 G-026 没有通过。warm end-to-end 从 `56.126s` 降到 `25.729s`，
加速为 `2.181x`，低于预注册的 `2.5x`。因此准确表述是：

> rCM 已经达到质量保持和 denoiser 加速预期，但尚未达到当前软件栈的
> 完整端到端目标；结果属于系统速度边界，不是质量失败。

## 速度分解

| 方法 | 网络调用 | denoiser | text | VAE | 序列化 | warm E2E |
|---|---:|---:|---:|---:|---:|---:|
| teacher20 | 40 | 32.202s | 17.386s | 4.181s | 1.928s | 56.126s |
| native4 | 8 | 6.357s | 14.332s | 4.221s | 2.377s | 27.791s |
| rcm4 | 4 | 3.177s | 16.119s | 4.214s | 1.827s | 25.729s |

三者每次网络 forward 都约为 `0.79--0.81s`。因此 rCM 的 `10.135x`
denoiser 加速不是 kernel 偷换，而是 20 到 4 步以及每步 CFG 调用数从两次
降为一次共同产生的调用数压缩。相对 native4，rCM 又获得约 `2x`
denoiser 收益。

与此同时，rCM 的 denoiser 只剩 `3.18s`，文本编码、VAE 和序列化已占主要
wall-clock。继续只优化 block/attention，即使局部无限快，也无法解决新的
Amdahl 瓶颈。优先级应转为 prompt embedding 精确缓存、VAE/序列化并行、
输入输出流水化和完整 runtime fusion。

## 质量结果

| 维度 | native4 / teacher | rcm4 / teacher |
|---|---:|---:|
| subject consistency | 1.0366 | 1.0062 |
| background consistency | 1.0094 | 0.9880 |
| temporal flickering | 1.0215 | 0.9887 |
| motion smoothness | 1.0078 | 0.9939 |
| dynamic degree | 0.5000 | 1.0000 |
| aesthetic quality | 0.8283 | 0.9706 |
| imaging quality | 0.7245 | 0.9820 |
| overall consistency | 0.8073 | 1.0455 |

`native4` 说明直接把原模型改成四步并不可行：虽然局部一致性指标看起来不差，
动态度、成像质量和美学质量明显受损，8 维均值只有 `0.8669`。rCM 的训练
改变了四步有限时间映射的函数，而不是简单减少同一个 solver 的步数。

rCM 四个 prompt 的多样性复合比值为 `1.572/1.120/1.113/1.362`，全部通过。
该指标只能证明没有 seed collapse，不能解释为 rCM 的所有分布属性优于 teacher。
paired SSIM/PSNR 较低同样不构成失败，因为 student 与 teacher 会生成不同但
都可能有效的视频轨迹。

## 与此前结构化实验的关系

EXP-048 中 rCM/rCM4 的 rank-64 late-block capacity/H1 误差仍为
`22.460%/30.843%`，说明蒸馏并没有把内部 residual 变成低秩 Markov state。
EXP-047 却能在端点质量上通过。两者并不矛盾：

\[
\text{finite-time flow map 可由一个学生网络摊销}
\not\Rightarrow
\text{学生的任意中间 residual 可由固定低秩状态表示}.
\]

因此 Wan 侧真正通过验证的对象是训练原生的 flow map / learned solver，
不是固定 BCM/BCCB、跨样本共享低秩 basis 或 post-hoc block cache。

## 结论边界

- 这是四 prompt 项目集，不是完整官方 VBench 排名。
- 没有量化、稀疏 attention、cache、compile、CUDA Graph 或自定义低精度 kernel。
- 结果支持以 rCM 作为质量较强的部署基线，并围绕其剩余 exact overhead 做系统优化。
- 结果不支持把 released rCM 重新包装成本项目的新算法，也不支持恢复已经失败的
  post-hoc whole-block state 路线。
