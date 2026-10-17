# 实验 066：L20 memory-aware CUDA Graph shape policy

日期：2026-10-17  
状态：第一阶段完成；`l20` policy 进入条件候选，默认仍为 `coarse`

## 目标

实验 035/040 已经说明，固定增加少数 Graph shape 不一定能改善端到端尾延迟。本实验换
一个可验证的假设：L20 在线服务经常出现 3、5、7、9 等小 decode batch，若只捕获
`1/2/4/8/16/24/32`，这些 batch 会被 padding 到更大的 shape。精确捕获小 batch 可以
减少无效 token 计算，同时通过显存和 P99 TPOT 验证额外 Graph 是否值得。

本实验不改变默认策略，新增显式 `--cuda-graph-shape-policy`：

- `coarse`：现有 `1/2/4/8/...` 策略，默认值。
- `l20`：捕获 `1..16`，再捕获 `20/24/28/32`，32 以上保持 8 的步长。
- `explicit`：要求同时传入 `--cuda-graph-bs`，用于复现实验列表。

## 原理

CUDA Graph 只能 replay 固定形状的地址和张量。逻辑 batch 为 `B` 时，scheduler 会选择
不小于 `B` 的最小已捕获 shape `P`，并补充 `P-B` 个 dummy request。padding ratio 定义为：

```text
padding_ratio = (P - B) / B
```

例如 coarse policy 下 `B=7` 会使用 `P=8`，而 `l20` policy 直接 replay `P=7`。这减少了
模型、attention 和 collective 对 dummy token 的计算，但每新增一个 Graph shape 都会增加
capture 时间、Graph pool 显存和启动复杂度。因此 shape policy 必须同时观察：

1. padding ratio 和 padding token 数。
2. Graph hit rate，确认没有因为 shape 缺失退回 eager。
3. capture 后剩余显存和运行峰值显存。
4. 吞吐、P90 TTFT、P99/P99.9 TPOT。

## 实现

- `EngineConfig`/server parser 增加 `cuda_graph_shape_policy` 和
  `--cuda-graph-shape-policy`。
- `GraphRunner._determine_cuda_graph_bs` 增加 `coarse/l20/explicit` 三种策略；显式
  `cuda_graph_bs` 仍优先于自动策略。
- scheduler telemetry 新增：
  - `requested_batch_size`
  - `padded_batch_size`
  - `padding_ratio`
  - `graph_hit`
- `bench_l20.py` 和 `bench_poisson_l20.py` 将 shape policy 写入结果元数据。
- 新增 Graph policy 单元测试，覆盖细粒度 shape 列表和 explicit 边界。

默认配置和旧的 `--cuda-graph-bs` 行为保持不变。

## 单元与启动验证

- Graph/调度/KV focused tests：16 passed，5 个 subtests passed。
- `coarse` TP4 Qwen3-32B FP8 Graph capture：7 个 shape，启动成功。
- `l20` TP4 Qwen3-32B FP8 Graph capture：20 个 shape，启动成功。
- Qwen3-32B FP8 TP4，L20 GPU 2/3/6/7，`cuda-graph-max-bs=32`：
  - coarse capture 后 rank-0 剩余显存约 8.53 GiB；
  - l20 capture 后约 8.20 GiB，额外约 0.33 GiB/card；
  - 运行峰值显存 36,767 -> 37,097 MiB，增加约 330 MiB/card。

## Poisson/open-loop 对照

为了产生非规则 decode batch，使用相同 seed 的 Poisson trace：32 请求、输入长度混合
128/256/512、输出 64，arrival rate 4/8/12 req/s，每组 2 次，FP8 W8A8 + FP8 KV，TP4，
`max_prefill_streak=1`、decode-active prefill budget 1024。

### Telemetry

| policy | decode forwards | Graph hit | 平均 padding ratio | padding tokens |
| --- | ---: | ---: | ---: | ---: |
| coarse | 1022 | 100% | 19.4848% | 1825 |
| l20 | 1042 | 100% | 0.8661% | 204 |

两组请求总 token 数相同，forward 数量的轻微差异来自 scheduler 在不同运行时长下的
batch 合并边界。l20 将 padding token 减少约 88.8%，但没有改变 Graph hit rate。

### 端到端结果

| Arrival rate | coarse tok/s | l20 tok/s | 吞吐变化 | coarse P90 TTFT ms | l20 P90 TTFT ms | TTFT 变化 | coarse P99 TPOT ms | l20 P99 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 223.38 | 224.66 | +0.57% | 328.62 | 298.51 | -9.16% | 142.52 | 156.01 |
| 8 | 372.41 | 372.53 | +0.03% | 378.05 | 368.01 | -2.65% | 334.37 | 474.89 |
| 12 | 463.62 | 463.46 | -0.03% | 961.77 | 941.52 | -2.11% | 428.39 | 421.27 |

TTFT P90 在三档负载都略有改善，但吞吐没有达到预注册的 +3% gate；8 req/s 的 P99 TPOT
上升约 42%，说明 padding 减少没有自动转化为稳定的尾延迟收益。两次重复不足以解释
所有波动，因此不能把 l20 policy 宣称为默认最优。

## 结论

1. `l20` policy 成功消除了大部分小 batch padding，padding ratio 从 19.48% 降到 0.87%，
   并通过了 Graph capture/replay 和 correctness smoke。
2. 额外 Graph 显存约 0.33 GiB/card，低于 1 GiB/card 的预算目标。
3. 当前 Poisson workload 上吞吐基本不变，P90 TTFT 小幅下降，但 P99 TPOT 存在负向波动；
   端到端性能 gate 未通过。
4. 保留 `--cuda-graph-shape-policy l20` 作为 L20 profiling/候选部署开关，默认仍为
   `coarse`。后续需要至少 3--5 次重复、更多 arrival rate 和更长输出，再决定是否推广。

## 原始结果

- [coarse benchmark](../results/066_poisson_coarse.md)
- [l20 benchmark](../results/066_poisson_l20.md)
- [coarse JSONL telemetry](../results/066_poisson_coarse_pipeline.jsonl)
- [l20 JSONL telemetry](../results/066_poisson_l20_pipeline.jsonl)
- [同步 C=1/8/32 对照](../results/066_coarse_graph.md)

## Stage 2：更长输出与三次重复

第一阶段的输出长度为 64，且每个 arrival rate 只有两次重复。为确认尾延迟结论，Stage 2
使用相同的 FP8 TP4 配置和 `input_lens=128/256/512`，将输出长度提高到 128，request count
为 24，每个 arrival rate 做 3 次重复。coarse 与 l20 使用相同 seed `6602042`，覆盖
4/8/12 req/s。

### Stage 2 telemetry

| policy | decode forwards | Graph hit | 平均 padding ratio | padding tokens |
| --- | ---: | ---: | ---: | ---: |
| coarse | 1847 | 100% | 15.6507% | 3171 |
| l20 | 1870 | 100% | 1.4769% | 505 |

在相同请求 token 数下，l20 将 padding token 减少约 84.1%。由于精确 shape 会改变 scheduler
的 batch 边界，forward 次数略有增加，但没有产生 eager fallback。

### Stage 2 端到端结果

| Arrival rate | coarse tok/s | l20 tok/s | 吞吐变化 | coarse P90 TTFT ms | l20 P90 TTFT ms | TTFT 变化 | coarse P99 TPOT ms | l20 P99 TPOT ms | P99 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 361.75 | 365.37 | +1.00% | 283.27 | 284.61 | +0.47% | 226.60 | 251.59 | +11.03% |
| 8 | 484.36 | 488.59 | +0.87% | 469.78 | 464.88 | -1.04% | 314.11 | 298.18 | -5.07% |
| 12 | 543.22 | 546.88 | +0.67% | 583.18 | 580.95 | -0.38% | 261.09 | 218.24 | -16.41% |

P99.9 TPOT 变化为 4 req/s +0.08%、8 req/s -13.17%、12 req/s -3.40%。l20 在中高负载的
尾延迟表现更好，但低负载 4 req/s 的 P99 TPOT 回退仍然存在；吞吐提升也未达到预注册的
3% gate。因此结论仍是条件接受，不修改默认 coarse policy。

Stage 2 原始结果：

- [coarse 128-token benchmark](../results/066_stage2_coarse.md)
- [l20 128-token benchmark](../results/066_stage2_l20.md)
- [coarse Stage 2 JSONL telemetry](../results/066_stage2_coarse_pipeline.jsonl)
- [l20 Stage 2 JSONL telemetry](../results/066_stage2_l20_pipeline.jsonl)
