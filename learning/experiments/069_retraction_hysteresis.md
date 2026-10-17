# 实验 069：Safe Decode Retraction 的滞回控制

## 1. 实验目的

实验 068 引入了 safe decode retraction：当新的 prefill 请求无法获得足够 page 时，暂停一个尚未提交到当前 GPU batch 的 decode 请求，只保留它已经拥有的 KV page，待 prefill 压力下降后再恢复。这个机制保持了 KV 正确性，但在线压力测试出现了明显的 reservation thrashing：同一批 decode 请求可能被暂停、恢复、再次暂停，调度开销抵消了内存回收收益。

本实验增加两个独立的滞回条件：

1. **恢复冷却窗口**：被恢复的 decode 请求在若干次 scheduler 调度周期内不能再次成为 retraction 候选。
2. **pending token 压力门槛**：只有待处理 prefill token 达到门槛时，才允许触发 safe retraction；低压力状态不为了少量收益改变 decode 集合。

目标是减少反复暂停/恢复，同时保持实验 068 的 page accounting 和输出正确性。

## 2. 实现

新增配置和 CLI 参数：

```text
--prefill-retraction-cooldown-steps N
--prefill-retraction-min-pending-tokens N
```

默认值都是 `0`，因此不改变原有行为。实验配置使用：

```text
--prefill-safe-retraction
--prefill-reserve-policy page-aware
--prefill-retraction-cooldown-steps 4
--prefill-retraction-min-pending-tokens 2048
```

`DecodeManager` 为恢复请求维护冷却计数。每次调度周期递减一次，冷却期间请求仍然可以 decode，但不会再次被 `retract_candidate()` 选中。请求完成、abort 或移出 paused 集合时清理计数，避免生命周期泄漏。

`Scheduler` 在 safe retraction 路径上读取 pending prefill token 数；当配置门槛非零且压力不足时，跳过 retraction。负数配置在启动时直接拒绝。整个机制只影响显式打开 safe retraction 的实验路径，默认配置仍关闭。

## 3. 正确性验证

新增测试覆盖：

- CLI 正确解析两个滞回参数；
- 恢复请求在冷却周期内不会再次被选为 retraction 候选；
- page-aware safe retraction 的原有请求生命周期和 page reservation 测试继续通过。

验证结果：

```text
tests/core focused: 16 passed
tests/core full:    112 passed
compileall:         passed
git diff --check:    passed
```

实验启动过程中曾发现一个实现遗漏：`Scheduler` 没有把新配置复制到实例字段，导致所有 rank 在第一次在线启动时抛出 `AttributeError`。该次启动没有产生有效 benchmark 数据，修复后重新运行并只采用修复后的结果。

## 4. L20 在线实验

硬件和模型：4 张 L20（GPU 2/3/6/7），Qwen3-32B，FP8 W8A8，TP=4，BF16 runtime，4096 pages，page size=1。请求输入长度均衡覆盖 256/1024/2048，输出 64 token，Poisson 到达率 1.5 和 2.5 req/s，12 请求，seed `6800042`。

对照组使用实验 068 的 safe-retraction control，处理组使用本实验的 cooldown=4、pending gate=2048。完整原始结果见 [`069_cooldown_poisson.md`](../results/069_cooldown_poisson.md)。

| 到达率 | 068 control 吞吐 | 069 吞吐 | 变化 | 068 control P90 TTFT | 069 P90 TTFT | 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 req/s | 66.91 tok/s | 66.95 tok/s | +0.06% | 1234.77 ms | 1235.23 ms | +0.04% |
| 2.5 req/s | 72.30 tok/s | 73.23 tok/s | +1.28% | 3766.34 ms | 3678.23 ms | -2.34% |

遥测中的 retraction 次数为 144 次、涉及 3972 token；实验 068 pause 组为 849 次、涉及 24204 token，分别下降约 83.0% 和 83.6%。两组请求都保持 HTTP 200、输出长度正确和 page integrity 正常；069 的 TTFT-SLO 达成为 100%/50%，与 068 control 的 100%/50% 相同。

## 5. 结论和限制

本实验支持将“冷却窗口 + pending pressure gate”作为 safe retraction 的条件性改进：它显著减少了 reservation thrashing，在本次 L20 轨迹上没有吞吐回退，并在高负载点略有收益。它不是通用默认策略，原因是：

- 当前在线矩阵每个到达率只运行 1 次，方差尚未建立；
- 2048 token 门槛和 4 个调度周期是当前 Qwen3-32B/L20/4096-page 配置的调优点，不能直接推广到其他显存容量、page size 或输出长度；
- safe retraction 本身仍然是实验性功能，默认保持关闭。

面试中可将本实验概括为：**通过 hysteresis 避免资源调度中的 oscillation，用状态冷却和压力门槛换取更少的集合变更；最终是否合入默认路径必须以多重复在线 tail latency 和正确性 gate 为准。**
