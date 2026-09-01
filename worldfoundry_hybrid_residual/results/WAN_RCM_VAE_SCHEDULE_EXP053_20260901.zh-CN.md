# Wan rCM VAE 时间分组实验判决

日期：2026-09-01
实验：`EXP-053`
判决：`exactness-null`

## 核心结论

官方 BF16 Wan VAE 的因果时间分组在 F17 上出现了短程 bitwise 等价，但该等价
没有迁移到 F81 的多 chunk 解码。冻结选择的 `chunk_size=4` 在四个 F81 prompt
上均与官方逐帧输出不一致，完整 VAE 仅达到 `1.196837x`，投影到 resident rCM4
请求仅为 `1.098601x`。因此它同时未通过精确性、VAE `1.26x` 和请求 `1.10x`
三个门槛，未进入完整 endpoint timing。

## 结果

F17 选择阶段中，chunk 1/4/8 bitwise exact，chunk 2 不 exact。冻结规则选择了
最快的 chunk 4，其 F17 VAE 时间为 `0.753553s`，相对官方 `0.894316s` 为
`1.186800x`。

F81 四个 prompt 的最大绝对误差为 `0.111328–0.164062`，相对 L2 为
`0.220%–0.413%`，VAE 局部速度为 `1.189–1.198x`。这些误差不能在“exact
system optimization”名义下接受，也不能用未注册的视频质量容差覆盖。

## 新发现

F17 的 chunk 4 只包含一个 sentinel 后分块，F81 则包含多个连续分块，并复用候选
路径的内部 cache state。结果说明：

\[
\text{short-horizon output equality}
\not\Rightarrow
\text{long-horizon hidden-state closure}.
\]

这与此前 cross-step predictor 的开放环边界一致：只比较一次输出会高估可复用性，
任何 temporal cache/schedule 都必须验证连续更新后的内部状态，而不是只验证首个
chunk。当前证据尚不能区分 chunk-shape CUDA 数值路径和 cache 演化各自的贡献，
因此不作更强根因主张。

## 决策边界

- `C-031` 在注册候选类内 refuted；`L-031` parked。
- 不事后改选 F17 的 chunk 1/8，也不运行未获授权的容差质量实验。
- 不推广为“VAE 不可优化”；结论只覆盖当前官方 backend、F17 选择规则和冻结
  chunk-4 F81 确认。
- 后续近似 kernel 仍必须以 EXP-052 的 `9.637995s` resident rCM4 为基线。
