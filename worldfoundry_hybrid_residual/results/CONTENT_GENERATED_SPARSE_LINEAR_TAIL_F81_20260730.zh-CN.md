# F81 内容生成 Sparse-Linear Tail 与逐 Tail Support Probe

## 结论

本轮在不恢复 rotation/chart、保持 `25%` exact support 的前提下，完成了
Layer 14、step 9、conditional CFG 分支上的内容生成 tail 与 support 重选实验。
最终注册结论为：

> **停止在该 cell 上继续扩大免训练 positive-linear tail；Layer 14/step 9
> 进入 FP8/BF16 dense fallback。**

这一结论不否定 sparse-critical + learned sparse-linear tail 的一般路线，
但否定了继续增加当前正值可分离 feature rank、固定 BCM family 或
rotation/chart 来修补该函数类。

## 实验协议

- 模型与捕获：Wan2.1-T2V-1.3B，F81，Layer 14，step 9，conditional 分支。
- 样本：此前已注册的 4 个 prompt/seed capture；`s00/s01` calibration，
  `s02` validation，`s03` test。
- query 范围：每个 capture 固定 3 个 query tile。
- exact support：固定 `25%` key budget；每种 tail 都重新运行 support selector，
  不复用旧 mask。
- tail rank：`16/32/48/64`。
- support/layout：fixed contiguous-64、SVG2-style Q/K semantic proxy、
  value-aware K/V semantic proxy，以及 projected error-to-width oracle。
- 归一化：exact sparse 与 linear tail 直接共享 softmax numerator/denominator，
  不做两个已归一化分支的后验拼接。
- 容量门槛：transductive oracle 的 aggregate `<=0.5%`、worst `<=1%`。
- 免训练部署门槛：aggregate `<=1%`、worst `<=2%`。

这里的 SVG2-style 仅指语义排序 proxy，不是 Sparse VideoGen2 的
paper-faithful k-means、top-p 与专用 kernel 复现。

## 实现要点

内容 tail 使用逐 head 的正值 Q/K feature map：

\[
\phi_q(q)=\operatorname{softplus}(qW_q+b_q),\qquad
\phi_k(k)=\operatorname{softplus}(kW_k+b_k).
\]

linear branch 先计算完整 tail numerator/denominator；对选中的 exact
interaction，再用真实指数核贡献替换对应 linear contribution：

\[
\hat Y_i=
\frac{N_i^{\rm linear}
      +N_{i,\Omega}^{\rm exact}
      -N_{i,\Omega}^{\rm linear}}
     {Z_i^{\rm linear}
      +Z_{i,\Omega}^{\rm exact}
      -Z_{i,\Omega}^{\rm linear}}.
\]

support oracle 的目标不是只最大化 attention mass，而是估计 exact tile
对 post-tail、post-rank-16 剩余输出缺陷的降低。为保证 selector 不会因近似
目标退化，若 rank-aware 候选比低成本 proxy 更差，执行单调回退。

## 主要结果

与粒度一致的旧基线相比：

| 方法 | Raw content error | Per-tile adaptive rank-16 | Worst tile |
| --- | ---: | ---: | ---: |
| 旧 `25% support-family oracle + rank-16` | 9.569% | 1.584% | 4.648% |
| rank-64 transductive content tail + support oracle | 8.399% | 1.179% | 3.385% |

本轮确实改善了 tail shaping：raw error 相对下降 `12.23%`，per-tile
rank-16 aggregate 相对下降 `25.53%`，worst tile 相对下降 `27.18%`。
但三个指标均没有达到注册容量门槛。

rank sweep 显示明显平台：

| Feature rank | Transductive per-tile adaptive rank-16 error |
| ---: | ---: |
| 16 | 1.180% |
| 32 | 1.177% |
| 48 | 1.179% |
| 64 | 1.179% |

从 rank 16 增至 64 仅产生约 `0.106%` 的相对改善，说明当前瓶颈不是
feature width，而是正值可分离 kernel、低成本 semantic support 与 Layer 14
内容相关 attention 的函数类失配。

更严格的迁移结果为：

| 设置 | Aggregate | Worst tile |
| --- | ---: | ---: |
| rank-64 calibration-frozen tail + held-out support oracle | 1.332% | 3.234% |
| validation-selected train-free value-aware proxy | 1.488% | 3.632% |
| rank-64 transductive，3 个 query tile 共享 basis | 1.851% | - |

在 transductive rank-64 run 中，达到单记录 `1%` 误差所需输出 rank 的
均值约为 `28`，最大为 `59`。这进一步说明可被 rank-16 修复的低维 witness
存在，但不足以形成跨 tile、跨样本稳定且廉价的固定 tail。

## 解释与方法边界

1. 语义 permutation 有价值，但只是 candidate layout 改善，不是完整 tail。
2. 共享 numerator/denominator 消除了 branch-scale mismatch 的一个来源，
   剩余误差仍然很大，因此失败不能归因于简单归一化实现错误。
3. transductive 和 dense-support 结果使用了不可部署信息，只能作为函数类
   容量诊断，不能表述为 train-free 推理精度。
4. error-to-width selector 是单调 projected-rank heuristic，不是组合 support
   的全局最优证明；但 rank-16 到 rank-64 的平台与 frozen/test 退化共同说明，
   继续扩大同一 family 的收益不足。
5. 本轮只覆盖一个 layer/step/CFG cell 和三个 query tile，不能外推为所有
   Wan attention cell 都不可压缩。

## 系统决策

- `Layer 14 / step 9`：分配给 fused FP8 dense；风险过高时 BF16 dense。
- rotation/chart：保持停止，不训练坐标预测器。
- 当前 positive-linear tail：停止继续扩大 rank，也不进入 1k-2k step 适配，
  因为 all-record transductive oracle 本身没有通过容量门槛。
- fixed BCM/support family：不再扩展。
- localized heads：仍可保留 confidence-gated geometry/semantic sparse 分支。
- transitional heads：只有新的、独立注册的 learned sparse-linear 函数类通过
  oracle 与 transfer 门槛后，才能重新进入 kernel 阶段。
- diffuse heads：以 fused FP8/BF16 dense attention 为主路径。

rank-64 的 attention 算术上界约为 `3.94x`，但本轮没有 fused H200 kernel、
没有完整 denoising rollout，也没有端到端 wall-clock 测量，因此不作任何
实际加速声明。

## 质量保证与复现

- 四个正式 rank artifact 与四个 deterministic re-evaluation artifact 均有
  `SUCCESS` marker。
- 每个 re-evaluation 包含 `672` 条 record、`28` 条 summary，无 NaN。
- shared-basis re-evaluation 与原始结果误差小于 `1e-12`。
- checkpoint 可加载；5 个核心单元测试通过。
- 可视化均由 CSV/JSON 原始结果生成，同时保存 PNG/PDF。

核心入口：

```bash
bash scripts/run_content_generated_tail_f81_v1.sh
python scripts/reevaluate_content_generated_tail.py --help
python scripts/analyze_content_generated_tail.py --help
python scripts/plot_content_generated_tail.py --help
python scripts/test_content_generated_tail.py
```

## 相关工作边界

- [Sparse VideoGen2](https://arxiv.org/abs/2505.18875)：语义聚类、重排、top-p
  与自定义 kernel；本实验只使用显式标注的 SVG2-style proxy。
- [VSA](https://arxiv.org/abs/2505.13389)：可训练 coarse-to-fine tile router
  与专用稀疏 kernel。
- [SLA2](https://arxiv.org/abs/2602.12675)：可学习 sparse-linear routing、
  branch ratio 与量化适配。
- [DynamicRad](https://arxiv.org/abs/2604.20470)：locality prior 与动态内容路由。

本轮最重要的研究结论不是“低秩对视频 DiT 无效”，而是：

> 对 Layer 14/step 9，当前免训练正值 sparse-linear family 无法低成本生成
> 随内容变化的 tail；继续增加 feature width 或固定结构专家不再是合理投入。
