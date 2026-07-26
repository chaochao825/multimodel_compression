# H200 Timing 有效性追溯审计

日期：2026-07-26

## 结论

当前 `strict_phase2_h200_v1` 中已经完成的正确性和质量证据可以保留，但本轮 cross-attention cache 时延与被终止的 F81 CFG 时延均不能用于论文加速比。

| 阶段 | 外来进程采样点 | 估计重叠时间 | Timing 判断 |
|---|---:|---:|---|
| F17 exact cross-attention K/V cache | 12 / 21（57.1%） | 约 120 秒 | INVALID |
| F81 exact CFG partial | 40 / 41（97.6%） | 约 400 秒 | INVALID |

F17 cache 的 latent、pixel、逐帧 SSIM 均为 exact，因而“缓存语义正确”仍成立；其观测均值 `1.009x` 和置信区间跨越 `1.0x`，在资源污染下只能说明该 exact 优化不是主要加速来源，不能作为干净 H200 数字。

F81 CFG 在检测到几乎全程资源重叠后已正常终止，只终止本项目的 `torchrun`，未影响外部任务。部分生成文件保留用于故障审计，不进入正式聚合。

## 根因

原 runner 只要求启动前连续三次空闲。另一项目恰好在空闲门通过后、Wan 进程创建前抢占 H200，形成 TOCTOU 竞态。单纯延长空闲轮询不能消除该问题。

修复后的 runner 同时采用：

1. 启动前连续空闲门；
2. 运行期每 5 秒记录 GPU、PID、进程路径和显存；
3. 通过 `/proc/<pid>/status` 祖先链区分本 runner 子进程与外来进程；
4. 发现外来 PID 后只终止本次自有 benchmark，并保留污染事件；
5. 生成 `gpu_exclusivity_audit.json`，只有 `timing_valid=true` 才允许进入时延汇总。

这将“数值正确性/视频质量”和“硬件性能有效性”拆成两个独立证据门，避免再次把共享 GPU 干扰误判为算法本身无效。
