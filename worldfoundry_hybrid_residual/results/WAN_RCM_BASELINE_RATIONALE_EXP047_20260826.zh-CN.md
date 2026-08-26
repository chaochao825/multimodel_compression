# Wan 少步基线 Gate：历史证据、理论边界与 EXP-047 设计

日期：2026-08-26  
状态：`EXP-047 / G-026` 执行中；本文不包含尚未产生的质量或速度结果。

## 1. 直接判决

当前方向仍有潜力，并且比继续扩展 `EXP-005` 的 grouped field、DPLR、
Broyden 或 rank-state 更接近最初的“几乎不损质量且明显加速 Wan”目标。
但潜力来自**训练后直接学习少步 flow map**，不是来自已经关闭的事后
低秩状态定义。

因此下一项最高信息量实验不是再设计一个结构，而是先测量官方四步 rCM
在同一 H200、同一 Wan 实现和同一冻结 prompt/seed 上的真实质量-速度
Pareto。该结果定义任何新结构化学生必须超过的部署基线。

## 2. EXP-005 的正结果与边界

外部 `EXP-005` 与规范仓库 `EXP-045` 回答的是同一类“当前输入可观测性”
问题，但后者进一步加入完整候选、连续层覆盖和开放环稳定性 Gate。

外部记录中，校准冻结的逐通道 field 将 pooled risk 从 AR(2) 的
`0.141061` 降到 `0.072832`，即 `1.937x`，并恢复 target-visible
coefficient oracle gap 的 `87.7%`。这个信号真实且重要：

\[
r_{\ell,k,d}\approx
a_{\ell,k,d}r_{\ell,k-1,d}
+b_{\ell,k,d}r_{\ell,k-2,d}
+c_{\ell,k,d}(h_{\ell,k,d}-h_{\ell,k-1,d}).
\]

它说明当前 block input drift 暴露了 past-residual-only 方法缺失的局部
Jacobian 坐标。但它只通过 L21、L24、L25 三层，未达到连续十层中六层
的 breadth Gate，也没有验证后缀稳定性、视频质量或速度。

规范 `EXP-045` 对这个机制进行了更强检验：

| 方法 | step-4 risk | step-6 risk | aggregate block-output L2 |
|---|---:|---:|---:|
| calibration-frozen AR(2) | 35.082% | 27.040% | 14.124% |
| current-input diagonal | 26.194% | 19.489% | 10.415% |
| Broyden-2 | 26.121% | 18.526% | 10.222% |
| DPLR-16 | 18.062% | 14.324% | 7.348% |
| target-visible transport oracle | 15.342% | 12.239% | 6.86% / 5.53% |

DPLR-16 仍只通过 L21/L24/L25；完整可执行的 Broyden-2 只通过两层，且
三步开放环最坏 risk ratio 达到 `1.530`。所以瓶颈不再是是否存在局部
相关性，而是相关性是否足够广、足够稳定、且能被便宜状态表达。

## 3. 为什么继续增加 rank 也不成立

`EXP-046` 已把 observer 问题移除：每个 endpoint 都允许看见目标 defect，
再拟合最乐观的 adaptive rank state。即使如此：

| rank | aggregate block-output L2 | worst | passing cells |
|---:|---:|---:|---:|
| 8 | 7.455% | 15.279% | 0/60 |
| 16 | 6.541% | 13.504% | 0/60 |
| 32 | 5.678% | 12.424% | 0/60 |
| 64 | 4.833% | 11.093% | 0/60 |
| 96 | 4.345% | 10.210% | 0/60 |

rank-64 的 H1 仍为 `3.595% / 4.104%`，rank-96 最容易 cell 也未通过
`1% / 2%` 门槛。因为这是 target-visible capacity ceiling，新的坐标
predictor、chart、rotation 或更多 past statistics 不可能拯救同一个
renderer。重复 grouped-field 或 rank sweep 会违反已接受的 stop rule。

## 4. 与更早结构化路线的统一解释

此前负结果不是“生成模型没有冗余”，而是多次把**局部条件冗余**误写成
**全局共享表示**：

| 路线 | 有效信号 | 失败的强假设 |
|---|---|---|
| BCM/BCCB/Butterfly | 局部几何、频带和部分 head role 稳定 | 固定 Fourier/结构特征向量可跨内容、step、head 共享 |
| sparse + low-rank attention | per-sample sparse critical + adaptive tail 有容量 | frozen tail basis 或廉价正值 kernel 可跟随旋转子空间 |
| Q/S/L residual repair | 部分量化或稀疏 defect 可被低秩捕获 | 独立分支的局部 MSE 改善会自动转成融合 kernel 和轨迹收益 |
| cache/AR/Broyden | 少数 layer-step 存在强当前输入相关性 | 单一过去状态可覆盖连续深层并在开放环保持稳定 |
| whole-block rank state | defect 确含相关低维能量 | 小 rank 是完整 defect 的充分状态 |

共同数学原因可写为：

\[
e_{k+m}\approx
\left(\prod_{j=1}^{m}J_{k+j}\right)e_k
+\sum_{j=1}^{m}
\left(\prod_{q=j+1}^{m}J_{k+q}\right)\delta_{k+j}.
\]

局部投影误差 \(\delta\) 不仅随内容旋转，还被后续非平稳 Jacobian 放大。
因此低 local MSE、较高 captured energy 或单层 risk gain 都不足以推出
稳定跳步。之前的止损实验正是逐步排除了这种错误外推。

## 5. 为什么 rCM 是不同且合理的函数类

事后 cache/predictor 固定原向量场，只尝试从历史近似一次昂贵调用：

\[
\hat r_k=P_\psi(h_k,t_k,r_{k-1:k-p}).
\]

少步 consistency/distillation 则直接训练参数 \(\phi\)，让网络学习有限时间
flow map：

\[
F_\phi(x_t,t\!\rightarrow\!s,c)\approx x_s,
\qquad s<t,
\]

并在训练分布中吸收曲率、内容条件和误差传播。它不要求一个 post-hoc
rank-64 state 同时重建所有 block defect，也不要求 20-step teacher 的
每一步轨迹保持不变。官方 rCM 进一步结合 continuous-time consistency、
score regularization 与 diversity-oriented training，公开了 Wan2.1 1.3B
四步 checkpoint 和原生推理代码。

这使 `EXP-047` 成为必要基线，但不是我们的新算法贡献：

- 它直接测试 NFE reduction 能否兑现；
- 它把“训练能否重塑可预测性”与“事后结构是否能拟合”分开；
- 它给后续 per-step FP8、exact CFG parallel 或训练原生结构状态提供真实
  Amdahl 基线；
- 它不能反向证明 `EXP-045/046` 的状态假设正确。

官方来源：[NVlabs/rCM](https://github.com/NVlabs/rcm)、
[Wan inference examples](https://github.com/NVlabs/rcm/blob/main/Wan.md)、
[released 1.3B checkpoint](https://huggingface.co/worstcoder/rcm-Wan/blob/main/rCM_Wan2.1_T2V_1.3B_480p.pt)。

## 6. EXP-047 的判别力

三种方法共享官方 rCM commit `ed3cb14dd936f92cdc9f9381af7369991509b41f`
中的 Wan 网络与 BF16 dense attention 路径：

1. `teacher20`：本地冻结 Wan 权重，20-step UniPC，CFG=5；
2. `native4`：同权重、同 sampler，仅把 NFE 降到 4；
3. `rcm4`：官方未量化 rCM 权重和四步更新，无 cache、稀疏、量化或
   自定义低精度 kernel。

`native4` 区分“少算所以快”与“训练使少步仍有质量”；`teacher20` 给出质量
锚点；`rcm4` 才是候选。runner 分别计量 text、denoiser、VAE、serialization、
warm end-to-end、每 step/forward 时间和 peak VRAM。

Gate 要求 `rcm4` 同时达到 denoiser `>=3.5x`、end-to-end `>=2.5x`，冻结
四提示集合上的八个 VBench evaluator teacher-normalized 平均 `>=0.90`、
任一维度 `>=0.80`，并保持跨 seed diversity。这里明确不是完整官方
VBench benchmark 排名，而是官方 evaluator 的项目内配对测量。

## 7. 结果出来后的唯一合理分叉

- **质量与速度均通过**：rCM 成为部署 incumbent。后续只研究能否在其上
  叠加 exact/system 优化，或训练一个结构原生学生超过该 Pareto；不再复活
  post-hoc rank-state。
- **速度通过、质量失败**：这是 few-step quality boundary。按冻结协议结束
  本 Gate，再单独决策是否比较 8-step/新版 checkpoint，不能在看到结果后
  临时放宽阈值。
- **质量通过、速度失败**：先 profile backend、text/VAE 和实现，不改算法。
- **两者都失败**：在该发布版本和严格 F81 条件下，官方少步基线也没有
  兑现目标；训练原生新架构的门槛与风险随之提高。

最初的结构化动机仍可保留，但只能以新的形式进入：结构必须在训练中塑造
flow map 或作为同一融合算子的硬件约束，而不能再作为冻结模型后的固定
BCM/低秩补丁。是否值得打开这条训练主线，应由 `G-026` 的实测 Pareto
决定，而不是由更多局部 oracle 决定。
