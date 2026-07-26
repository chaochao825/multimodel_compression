# Exact Cross-Attention K/V Cache 的 Amdahl 边界

日期：2026-07-26

模型：Wan2.1-T2V-1.3B，`dim=1536`，`ffn_dim=8960`，30 层，text length 512，20 个 denoising steps。

## 结论

Exact text cross-attention K/V cache 是应当保留的系统卫生优化，但不是 F81/F17 的主要加速来源。

| Case | 视频 token | 每步可缓存 text K/V FLOP 占比 | 19/20 步复用后消除的 FLOP | 理想 FLOP 加速上限 |
|---|---:|---:|---:|---:|
| F17 | 7,800 | 0.4587% | 0.4358% | 1.0044x |
| F81 | 32,760 | 0.0512% | 0.0487% | 1.0005x |

这个计算与实测行为一致：cache 每个样本有 60 次首步 miss 和 1,140 次后续 hit，即成功复用了 95% 的文本 K/V 投影；但这些投影只处理 512 个文本 token，而 self-attention、视频 Q/O 投影和 FFN 处理 7,800 或 32,760 个视频 token。F81 的 self-attention `QK/PV` 还随视频 token 数平方增长，因此 text K/V 的相对覆盖率进一步下降。

此前受资源污染的 F17 观测均值 `1.009x` 高于 FLOP proxy 的 `1.0044x`，且置信区间跨越 `1.0x`。这不能被解释为稳定收益；其主要价值是 bit-exact、低风险和可与其他优化组合。

## 计算口径

每层每次 denoiser call 采用 dense 算术量：

```text
self Q/K/V/O        = 8 N d^2
self QK/PV          = 4 N^2 d
cross Q/O           = 4 N d^2
cross text K/V      = 4 L d^2
cross QK/PV         = 4 N L d
FFN up/down         = 4 N d d_ff
```

其中乘加记为 2 FLOPs。cache 只消除 `cross text K/V`，并且首个 step 必须计算，因此实际可消除比例再乘 `19/20`。

这只是结构 FLOP proxy，不是 H200 时延预测。它不包含 kernel 启动、归一化、调制、内存层级、VAE 和 scheduler；小 GEMM 的低效率可能使实测占比略大，但不足以改变“F81 主线必须优化视频 self-attention”的结论。
