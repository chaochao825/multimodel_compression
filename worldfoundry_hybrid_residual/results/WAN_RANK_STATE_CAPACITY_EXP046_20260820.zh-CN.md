# EXP-046：Wan whole-block rank-state 容量 Gate

日期：2026-08-20
状态：`VALID PROSPECTIVE NULL`
决策：`G-025 = FAIL`

## 1. 结论

当前直觉只有一半成立：Wan 晚层 residual 确实含有可被低秩修复的相关成分，
但在当前 whole-block renderer 下，它远未形成满足严格质量门槛的 rank-64/96
endpoint state。

正式 selection 结果为：

| rank | aggregate block-output L2 | worst identity/branch | passing cells | render/exact MAC proxy |
|---:|---:|---:|---:|---:|
| 0 | 42.957% | 392.024% | 0/60 | 0.000% |
| 8 | 7.455% | 15.279% | 0/60 | 0.040% |
| 16 | 6.541% | 13.504% | 0/60 | 0.081% |
| 32 | 5.678% | 12.424% | 0/60 | 0.161% |
| 64 | **4.833%** | **11.093%** | **0/60** | **0.323%** |
| 96 | **4.345%** | **10.210%** | **0/60** | **0.484%** |

rank 64 和诊断 rank 96 在 step 4/6、H1/H2/H3 的六个覆盖条件中均为
`0/10` 层通过。最好的 rank-96 cell 是 L29、step 4、H1，也只有
`1.189%` aggregate、`1.716%` worst，仍同时超过 `0.5%/1%` Gate。

因此应停止：

> 在冻结 Wan 上，先用 exact-history current-input diagonal renderer 展开，
> 再以 rank-64 whole-block state 修复 endpoint defect，并训练便宜 observer
> 生成该 state。

这个 null 不否定训练原生 state、完整 few-step distillation、物理视频时间状态，
也不否定 F81 attention 的异构稀疏/低精度加速。

## 2. 实验完整性

- 模型：Wan2.1-T2V-1.3B，F17 480x832，20-step UniPC，CFG 5.0。
- 范围：blocks 20--29，sampler steps 4/6，H1/H2/H3，两个 CFG branch。
- 数据：4 个全新 selection prompt/seed；与已有 prompt 文件无完全相同文本。
- final：4 个 final identity 从未打开。
- target-visible 边界：SVD correction 读取 endpoint defect；base renderer 不读取
  exact anchor 后的 residual。
- 每个 identity 产生 720 rows，共 2,880 rows；分析器验证完整 grid、有限值、
  rank 单调性、输出误差分母和 split 边界。
- H200 smoke v3 的 capture 与未插桩 dense latent `torch.equal=True`，relative L2
  为 0；正式 run 独占一张 NVIDIA H200 NVL。
- smoke v1/v2 分别因错误环境缺少 `easydict`、`diffusers` 在 Wan import 前退出，
  未产生科学数据，原日志保留。v3 和正式 run 使用此前 Wan 已验证的
  `/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python`。

## 3. Horizon 分解

以下为从原始 selection rows 重新聚合的 full-tensor output L2：

| rank | step 4 H1/H2/H3 | step 6 H1/H2/H3 |
|---:|---:|---:|
| 0 | 13.025% / 18.271% / 97.221% | 10.941% / 15.618% / 19.853% |
| 64 | 3.595% / 4.486% / 5.350% | 4.104% / 5.197% / 5.940% |
| 96 | 3.248% / 4.033% / 4.745% | 3.725% / 4.700% / 5.345% |

step-4 H3 的 diagonal base 出现明显开放环放大，但这不是 G-025 失败的唯一原因：
即使只看最容易的 H1，rank 64 仍为 `3.595%/4.104%`，rank 96 仍为
`3.248%/3.725%`。因此不能通过删除 H3 或换一个稳定性解释来挽救当前表示命题。

深度上 L29 最容易，L20 最难，但不存在通过层。rank-64 的 L29/step4/H1 为
`1.404%/2.085%`，L20/step6/H3 为 `9.807%/11.093%`。这再次说明
predictability 具有 layer role，而不是简单随深度单调增加；但这种趋势不足以形成
可部署的稀疏 late-layer schedule。

## 4. 为什么低剩余能量仍没有低输出误差

rank-64 的平均 remaining defect energy 为 `14.589%`，rank-96 为 `11.898%`。
部分 H3 cell 甚至只剩不到 1% 的 base-defect energy，但 output L2 仍超过 2%。
原因是 initial open-loop defect 相对 block output 已经非常大：

\[
\frac{\|\Delta-L_r\|}{\|h+r\|}
=
\frac{\|\Delta\|}{\|h+r\|}
\sqrt{1-\eta_r}.
\]

当第一项很大时，捕获 99% defect energy 也未必满足 0.5% output error。这与此前
INT4 joint shaping 的教训相同：`captured energy` 可以很好看，但最终 output error
才是成功标准。

从 rank 64 的 `4.833%` 降到 `0.5%`，还需在当前结果上再减少约 99% 的误差
能量。rank `64 -> 96` 只把 L2 从 `4.833%` 降到 `4.345%`，没有显示存在这种
陡峭尾谱。继续试 rank 128/256 既没有预注册依据，也会把“紧凑 state”逐步退化为
target-specific payload。

## 5. 与历史尝试的统一解释

| 历史路线 | 关键结果 | 与本次的共同机制 |
|---|---|---|
| fixed/hierarchical BCM | `57.20% -> 50.41%`，继续增至 530,070 参数/head 不改善 | 固定 Fourier basis 不能追踪内容相关坐标 |
| sparse + adaptive rank-16 attention tail | `0.629%/1.85%` | 单 cell 存在 target-adaptive witness |
| frozen tail basis | `2.68%-2.76%`，worst `11.75%` | witness 的子空间跨内容旋转 |
| positive/content-generated tail | rank 16--64 平台在 `1.179%/3.385%` capacity | 便宜函数类不能生成正确坐标 |
| INT4 + LR + sparse | block 24 `11.950% -> 8.014%` | 可修复能量不等于最终保真或加速 |
| EXP-045 current input | DPLR-16 `7.348%`；target-visible shift oracle仍 `5%-7%` | 当前输入暴露局部 Jacobian，但旧 expert family 不足 |
| EXP-046 target-visible rank state | rank-64 `4.833%/11.093%` | 即使去掉 observer 难题，whole-block endpoint defect 仍不够紧凑 |

这些实验没有说明“视频 DiT 没有冗余”，而是连续排除了三类简单假设：固定全局
结构、跨样本共享静态 tail、以及冻结模型上的小 rank whole-block endpoint state。
可利用冗余更可能来自异构安全区域、训练后改变的向量场、精确系统并行和低精度
dense kernel，而不是一个统一 post-hoc state。

## 6. 对“最小充分隐藏状态”直觉的判断

理论上，确定性 denoising 过程当然有状态，完整 latent 与条件就是一个状态；真正的
问题是是否存在同时满足低维、可观测、闭合递推和低渲染成本的充分状态：

\[
s_k=E(h_{\le k},c),\qquad
s_{k+1}=T(s_k,h_{k+1},t_{k+1},c),\qquad
r_{k+1}=R(s_{k+1}).
\]

本次只给 `R` 的每-endpoint、target-visible、非共享低秩上界，而且该上界失败。
它没有测试共享 encoder/transition，也没有在近似 latent 分布上 rollout。因此：

- **对冻结 Wan 的当前 renderer**：直觉不成立，不应训练 observer/router。
- **对训练原生 student**：仍合理，但必须视为新的函数类和训练问题；训练可以主动
  让 dynamics 更平滑、更可观测，并让 student 适应自己的预测分布。
- **对已有 few-step 模型**：应作为真实竞争基线，而不是当前方法的证据。

[Shortcut Models](https://arxiv.org/abs/2410.12557) 和
[MeanFlow](https://arxiv.org/abs/2505.13447) 都通过改变训练目标学习有限步长映射；
[T2V-Turbo](https://arxiv.org/abs/2405.18750) 则说明视频 few-step distillation 可以
形成强实际基线。这些成功不意味着 pretrained Wan 的任意 late block 可以被事后
线性/低秩展开。

## 7. 下一决策

1. 关闭 C-025/L-025，不打开 final split，不训练 current-h coordinate observer。
2. 工程主线优先接入一个公开、可复现的 Wan-compatible few-step student，测真实
   H200 end-to-end、DiT/VAE/text 分解、VRAM、VBench/SSIM 与多 prompt/seed Pareto。
3. 若继续研究 state student，必须注册新 claim：训练一个共享 decoder/transition，
   使用多步开放环和 own-state distribution loss，并与 few-step baseline 比较；不能
   继续称为 EXP-046 的 rank rescue。
4. F81 production 加速继续保留 FP8/BF16 dense attention、精确 dual-H200 CFG 和
   已通过独立 Gate 的异构 sparse attention 候选；它们与本 null 正交。

## 8. 证据与可视化

- `rank_state_capacity_exp046_gate_v1/all_cell_metrics.csv`
- `rank_state_capacity_exp046_gate_v1/run_manifest.json`
- `rank_state_capacity_exp046_gate_v1_analysis/rank_summary.csv`
- `rank_state_capacity_exp046_gate_v1_analysis/cell_rank_metrics.csv`
- `rank_state_capacity_exp046_gate_v1_analysis/coverage_gate.csv`
- `rank_state_capacity_exp046_gate_v1_analysis/rank_capacity_frontier.png`
- `rank_state_capacity_exp046_gate_v1_analysis/rank64_layer_map.png`
- `rank_state_capacity_exp046_smoke_v3/run_manifest.json`
- `WAN_RANK_STATE_CAPACITY_EXP046_PREANALYSIS_20260820.zh-CN.md`
