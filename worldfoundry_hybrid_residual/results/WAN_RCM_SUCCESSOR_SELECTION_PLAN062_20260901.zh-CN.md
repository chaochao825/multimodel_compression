# rCM exact runtime 后续候选选择：先验证 VAE 因果时序调度

日期：2026-09-01
状态：`PLAN-062` 完成候选选择；`RDR-036 / EXP-053 / G-032` 待研究者接受

## 1. 决策结论

以 `EXP-052` 的 exact resident rCM4 `9.637995s` 为唯一基线，第一项候选应是
**Wan VAE 的因果时序分块调度**，而不是 FP8 attention、serialization pipeline
或新的近似状态。它不改权重、dtype、卷积、attention、cache 内容或输出 codec，
只测试同一 decoder 能否把首个 latent frame 保持独立，并把后续 frame 合并成
较大的因果 chunk。

这不是提前认定 chunking 数值等价。不同时间 batch 可能触发不同 cuDNN kernel
或归约顺序，因此每个 chunk 都必须先通过 `torch.equal`；任何非 bitwise 候选立即
淘汰，不能用质量指标或放宽 tolerance 修复。

## 2. Amdahl 排序

`EXP-052` resident rCM4 的组件中位数为：VAE `4.308300s`、denoiser
`3.205365s`、serialization `1.796082s`、CPU transfer `0.253741s`、text
`0.064420s`。

| 候选 | 当前端点占比/估计占比 | 局部假设 | 端点增量上限或估计 | 判决 |
|---|---:|---:|---:|---|
| exact VAE scheduling | `44.7%` | VAE `2x` | `1.288x`；完全消除上限 `1.808x` | 第一候选 |
| serialization | `18.6%` | codec `2x` | `1.103x`；完全消除上限 `1.229x` | 第二顺位；还需区分 latency/throughput |
| FP8/fused self-attention | 约 `17.9%` endpoint | self-attn `1.51x` 且全覆盖 | 约 `1.064x` | 后置；已有质量覆盖不足 |
| CPU transfer | `2.6%` | 完全消除 | `1.027x` | 不单独开 Gate |

要让 resident endpoint 至少再快 `1.10x`，VAE 至少需要约 `1.2553x` 局部
加速，因此 Gate 将完整 VAE 中位数门槛冻结为 `1.26x`。

## 3. 为什么当前 VAE 有可测空间

官方 Wan VAE 对 F81 的 21 个 latent frame 逐 frame 调用 `Decoder3d`，并在每次
调用后对累计输出执行 `torch.cat`。decoder 内部的 spatial attention 本来就按
`(batch × time)` 独立处理；时间依赖由 `CausalConv3d` 的两帧 feature cache 与
两个 temporal upsample 层维护。

候选始终单独执行第 0 帧，以保留 upsample cache 的 `Rep` sentinel。之后测试：

| chunk | F81 decoder 调用数 | 作用 |
|---:|---:|---|
| `1` | `21` | 只移除渐进式累计 `cat`，一次性拼接 |
| `2` | `11` | 合并后续两帧 |
| `4` | `6` | 与 VAE temporal window 对齐 |
| `8` | `4` | 更强 launch amortization，显存风险更高 |

该候选的潜力来自减少完整 decoder 调用边界、小 Conv3d/normalization/attention
dispatch 和累计复制，不来自减少数学 FLOPs。若 larger chunk 不 bitwise，`chunk=1`
仍能单独判断重复 `cat` 是否值得；若所有 exact 候选低于门槛，应直接关闭该线。

## 4. 已完成的非 GPU 工程证据

- 新 runner 独立于官方 rCM source，不污染固定 commit。
- 单元测试验证第 0 帧独立、后续 chunk 边界、输出顺序和 cache 对象传递。
- `pytest`: `8 passed`（包含既有 EXP-052 policy tests）。
- `ruff` 与 `git diff --check` 通过。
- 当前 236 GPU3 是隔离空闲 H200，但在 RDR 接受前未运行科学/工程 GPU stage。

## 5. 决策门

`EXP-053` 只允许以下顺序：

1. F17 在同一 latent 上比较官方 framewise decode 与 `1/2/4/8`，选择最快的
   bitwise-exact 候选；
2. F81 四个冻结 prompt、同一 seed 上确认 bitwise equality、完整 VAE latency
   和显存；
3. 仅当 VAE `>=1.26x` 且投影 resident endpoint `>=1.10x` 时，运行完整 resident
   rCM endpoint；
4. 完整 endpoint 仍须相对 `9.637995s` 达到 `>=1.10x`，否则只记 boundary。

不允许在该 Gate 中加入 `torch.compile`、FP8、CUDA Graph、attention 修改、codec
并行或 tolerance relaxation。这样失败也能回答一个明确问题：官方逐帧 VAE 的
launch/copy 边界是否是 exact、可兑现的 H200 冗余。
