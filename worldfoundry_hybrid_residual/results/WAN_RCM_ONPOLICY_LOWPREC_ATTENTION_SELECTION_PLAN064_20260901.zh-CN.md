# rCM 四步 on-policy 低精度 dense attention 候选选择

日期：2026-09-01
状态：`PLAN-064` 完成，`RDR-037 / EXP-054 / G-033` 已冻结

## 判决

第一项值得在 resident rCM4 上验证的近似 kernel 不是普通 FA3 FP8，也不是逐 head
precision island，而是：

> 以 `rCM step x layer` 为 cell，在 calibration 上静态选择整 cell Sage SM90 dense
> attention 或官方 FA3 BF16，并在未见轨迹上要求零 false-safe。

它只测试一个新问题：rCM 蒸馏后的四步 on-policy 分布，是否比历史 teacher20 轨迹具有
更广的低精度安全区。它不主张新 attention kernel，也不重开已失败的结构残差路线。

## 已有证据

- 普通 FA3 FP8 在 F81 的全 attention output relative L2 约 `13.76%`，不具备高保真
  起点。
- Sage per-thread INT8 Q/K、smooth-K、FP8 V/PV、FP32+FP32 accumulation 曾达到
  `0.9824% / 1.9204% / 1.0629%` aggregate/head/tile，并在 H200 上达到约
  `1.5890x`。
- 实际 11-head Sage + 1-head BF16 双调用在一个 exposed cell 达到 `1.4931x`，但
  teacher20 prospective atlas 只认证约 `33.3%` cells，静态 precision island 整体失败。
- 后续 V correction、rank tail、rotation、scale granularity、SAP/EAR 均没有同时解决
  teacher20 的质量、覆盖和真实调用成本。

因此旧结论仍有效：不能继续优化 teacher20 上的静态 island，也不能把 exposed cell
当作全模型证据。但 rCM 权重和四个宏步是新的部署分布，需要一次独立 on-policy Gate。

## resident rCM 上的物质性

EXP-052 给出 `T=9.637995s`、denoiser `D=3.205365s`。沿用历史 self-attention
占 denoiser `a=0.5388086760` 和 Sage speed `s=1.5906`，覆盖率 `c` 的乐观请求为：

\[
T'=T-Dac(1-1/s).
\]

| 安全覆盖 | 投影请求 | 增量速度 |
|---:|---:|---:|
| 33.3% | `9.424s` | `1.0227x` |
| 50% | `9.317s` | `1.0344x` |
| 75% | `9.157s` | `1.0525x` |
| 100% | `8.997s` | `1.0713x` |

达到 `1.05x` 至少需要 `71.57%` 安全覆盖。因此 Gate 固定为至少 `87/120` cells，
并且仍要用本轮实测速率重新计算，不能只引用历史数字。

## 为什么选择整 cell

- 每个 cell 只有一次 fused call，不引入 head gather、两次 kernel、拼接和 CPU routing。
- rCM F81 每层每步形状一致，静态 cell atlas 可以直接映射到部署调用。
- 校准时同时计算 Sage/FA3 但只返回 FA3，使后续层始终看到 exact baseline trajectory。
- Atlas 冻结后，candidate 才真正返回 Sage，因而 local transfer 和 on-policy rollout
  被明确分开。

## 停止边界

若 S1 出现任意 false-safe、覆盖不足 `87/120`，或投影不足 `1.05x`，立即停止，不生成
视频。只有 S1 通过，才运行独立四 prompt x 两 seed 的质量与 resident timing。

这条路线的最高价值是一次高信息判决：成功则给 rCM 增加约 `5%--7%` 系统速度；失败
则关闭 released rCM 上的 train-free static low-precision dense atlas，后续只能考虑
训练态量化或转向 VAE/serialization 的不同函数类。
