# Wan FFN Fusion H200 上限分析

日期：2026-07-26

## 结论

1. **Standalone post-GEMM pointwise kernel 是 NO-GO。** 已测 Triton bias+GELU 路径不是 bit-exact（相对 L2 约 0.286%，最大绝对误差 1.0），且投影到完整 FFN 后，F17/F81 局部速度分别只有 `0.978x/0.997x`。
2. **只消除 FFN hidden activation 的一次等价 BF16 copy，端到端上限很低。** F17/F81 的局部理想上限为 `1.103x/1.095x`，Amdahl 端到端仅为 `1.019x/1.011x`。
3. **即使把 up projection 的全部 bias+GELU overhead 免费消除，仍不足以形成主要加速。** F17/F81 的完整 FFN 理想上限为 `1.230x/1.143x`，端到端仅为 `1.038x/1.016x`。
4. F17 值得继续的是跨 normalization、modulation、residual、copy 和 FFN 的 **whole-block/whole-segment fusion**；F81 的主线仍应是 self-attention。单独优化 FFN epilogue 不足以改变系统瓶颈。

## 数据与假设

- H200 实测完整 FFN：F17 `0.7737 ms`，F81 `3.3209 ms`。
- H200 实测 up linear + bias + GELU：F17 `0.4553 ms`，F81 `1.8806 ms`。
- H200 实测无 bias up linear：F17 `0.3108 ms`，F81 `1.4649 ms`。
- H200 实测 FFN hidden BF16 copy：F17 `0.07225 ms`，F81 `0.2890 ms`。
- 每个增量 denoise step 按 30 blocks、2 个 CFG branch 估计 60 次完整 FFN。
- profile 每步总时延来自已有 F17/F81 增量 step profiler；由此估计完整 FFN 占 F17 `19.49%`、F81 `12.39%`。
- “删除完整 FFN”和“删除全部 elementwise/memory”仅为不可部署的绝对上界，不是实现预期。

## 分层上限

| Case | Hidden traffic 局部 / E2E | 全 epilogue 局部 / E2E | 删除完整 FFN E2E | 删除全部 elementwise E2E |
|---|---:|---:|---:|---:|
| F17 | 1.103x / 1.019x | 1.230x / 1.038x | 1.242x | 1.914x |
| F81 | 1.095x / 1.011x | 1.143x / 1.016x | 1.141x | 1.389x |

F17 的 elementwise/memory profile 占比为 `47.76%`，明显大于估计完整 FFN 占比 `19.49%`。这说明可利用空间分散在整个 DiT block 的 normalization、调制、布局变换、residual 和 launch 碎片中，而不是集中在 GELU。F81 的 self-attention 已占 `53.88%`，任何 FFN-only exact fusion 即使局部成功，也只能提供约 1%--2% 的端到端收益。

## 工程决策

- 保留正在排队的 checkpoint-faithful 完整 FFN `torch.compile` / CUDA Graph 基准，用它判断现成 exact path 是否能兑现任何收益。
- 停止 standalone bias+GELU kernel；它增加 launch，且当前实现不 exact。
- F17 下一 kernel 单元应至少覆盖一个完整 residual segment，并报告 kernel count、HBM bytes、median/p95 和端到端 Amdahl 收益。
- F81 不为 FFN epilogue 开发专用 kernel；优先完成 CFG exact 与 attention 工作量削减。
- 若 whole-block exact prototype 的实测端到端收益仍低于 `1.05x`，则 F17 系统主线应转向 CUDA Graph、布局消除和更大范围的 execution graph capture，而不是继续堆叠近似残差。

## 产物

- 原始派生数据：`ffn_fusion_ceiling_h200_v1/ffn_fusion_ceiling.csv`
- 机器可读结论：`ffn_fusion_ceiling_h200_v1/ffn_fusion_ceiling.json`
- 可视化：`ffn_fusion_ceiling_h200_v1/ffn_fusion_ceiling.png` 与 `.pdf`
