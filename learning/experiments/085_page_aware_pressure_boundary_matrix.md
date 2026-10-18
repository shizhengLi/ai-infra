# 实验 085：page-aware reservation 的真实 L20 压力边界矩阵（阶段 1）

## 1. 实验目的

实验 084 已经用 CPU-only deterministic trace 证明了 capacity trim、deferred 和 UID
fairness cursor 的选择逻辑。但这并不能证明真实 serving workload 会进入同一条路径。
实验 085 的第一阶段因此只回答两个问题：

1. 在 L20 TP4、Qwen3-32B-AWQ、`page_size=16` 下，缩小 KV cache 是否会让
   `token`/`page-aware` reservation 进入真实 capacity trim 路径；
2. 在没有 trim 的情况下，page-aware 是否会带来稳定的端到端收益，还是只是 admission
   排队形态不同造成的单次波动。

本阶段是边界 pilot，不把单次 A/B 的最快数字写成默认优化结论。实验计划中要求的“至少
一个真实 trim-path workload”尚未满足，因此 page-aware 继续保持 opt-in，默认策略仍为
`token`。

## 2. 环境与可复现配置

- GPU：L20，`CUDA_VISIBLE_DEVICES=2,3,6,7`，TP=4；这些卡在实验期间为空闲状态。
- 模型：Qwen3-32B-AWQ（ModelScope snapshot）。
- dtype：`float16`；量化：AWQ；CUDA Graph max batch size=8。
- PyTorch：`2.9.1+cu128`；CUDA runtime：12.8；Python：3.12.14。
- workload：Poisson arrival，16 个请求，input=17 token，arrival rate=80 req/s。
- 所有新点均使用同一随机种子和同一提示，只有 reservation policy 或 cache 容量改变。
- prefill：normal budget=64，decode-active budget=32，maximum prefill streak=1。
- 指标：output throughput、P90/P99 TTFT、P99 TPOT、P90 E2E、GPU 利用率和 telemetry。

### CUDA 工具链边界

机器上的 `/usr/bin/nvcc` 是 CUDA 11.5，而 PyTorch 和 FlashInfer 运行时是 cu12.8。
使用默认 `PATH` 启动时，PyNCCL AOT 编译在 `-std=c++20` 处失败；显式设置
`CUDA_HOME=/usr/local/cuda-12.8` 并把对应 `bin` 放在 `PATH` 后，服务可以完成编译、
初始化和 Graph capture。这是环境修复，不是本实验的算法变量；后续 L20 实验必须记录
实际 `nvcc --version`，否则结果不可复现。

### 128-token 容量的请求上限

`num_pages=8`、`page_size=16` 只有 128 个 KV token。聊天模板和输入实际占用约 25 token，
所以请求 `output_len=128` 会被引擎调整为 103，benchmark 的严格长度校验会失败。128-token
点改用 `output_len=96`；这不是吞吐调参，而是满足 `input + output <= max_seq_len` 的合法性
边界。512-token 点可以合法使用 output=128。

## 3. 结果矩阵

### 3.1 新增的 128/512-token 单次 A/B

| KV cache | output | policy | throughput (tok/s) | P90 TTFT (ms) | P99 TTFT (ms) | P99 TPOT (ms) | P90 E2E (s) | GPU util | trim/deferred |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 96 | token | 87.87 | 15183.72 | 16268.17 | 12.86 | 16.226 | 95.41% | 0/0 |
| 128 | 96 | page-aware | 89.46 | 14948.03 | 15981.62 | 11.61 | 15.969 | 94.72% | 0/0 |
| 512 | 128 | token | 221.58 | 6151.48 | 7603.49 | 34.29 | 7.642 | 92.45% | 0/0 |
| 512 | 128 | page-aware | 216.20 | 6320.81 | 7824.23 | 36.80 | 7.837 | 92.68% | 0/0 |

相对 token，128-token 点的 page-aware 单次结果为 throughput +1.80%、P90 TTFT -1.55%、
P99 TTFT -1.76%、P99 TPOT -9.76%；512-token 点则为 throughput -2.42%、P90 TTFT +2.75%、
P99 TTFT +2.90%、P99 TPOT +7.31%。两组都只有一个 repeat，且 telemetry 没有 trim/deferred，
所以这两组只能作为边界探索，不能作为策略收益的统计证据。

### 3.2 与实验 083 的 256-token 中间点

实验 083 已经在相同的 TP4/16 请求/80 req/s 压力下做过三次重复：

| KV cache | output | policy | throughput (tok/s) | P90 TTFT (ms) | P99 TPOT (ms) | decode batch | trim/deferred |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 128 | token | 88.99 ± 0.39 | 19956.95 | 12.09 | 全部 batch=1 | 0/0 |
| 256 | 128 | page-aware | 90.14 ± 0.58 | 19718.82 | 11.73 | 全部 batch=1 | 0/0 |

256-token 点的表面 throughput 差异约 +1.30%，但两组 telemetry 完全相同，6160 个 decode
batch 全为 1。这个结果说明 scheduler 在 decode fit guard 之前已经将请求串行化；page-aware
没有获得多个候选请求，也就没有机会使用实验 084 验证的公平性 cursor。

## 4. 结果解释

### 4.1 真实压力的主要瓶颈仍是 admission

把 KV 容量从 512 降到 256/128 后，TTFT 和 E2E 显著变差，但 trim/deferred 仍为 0。也就是
“容量很小”不等于“decode fit 会裁剪”：当前调度器先让请求在 admission 阶段等待，真正进入
decode 的 batch 通常只有 1 个。因而本阶段没有验证到 page-aware 的核心收益点——在多个 decode
候选中按未来 page demand 选择子集并提供公平性。

### 4.2 page-aware 没有稳定的在线收益

128-token 单次结果看起来更快，512-token 单次结果反而更慢；方向相反且均只有一次重复。
结合 256-token 的三次中性 A/B，合理结论是 admission-serialized workload 下两种策略基本等价，
单次差异属于测量波动或请求完成时序差异，不能据此改变默认策略。

### 4.3 正确性和 SLO

四个新运行均 16/16 请求完成，TTFT/TPOT SLO attainment 均为 100%，没有 allocator assertion
或 stale result。这个结果支持“缩小容量不会破坏现有 guard”的 correctness 结论，但不支持
“page-aware 提升吞吐”的结论。

## 5. 验收状态

| Gate | 状态 | 说明 |
| --- | --- | --- |
| 不同容量点可启动并完成请求 | 通过 | 128/256/512 token 均完成；128 点遵守 input+output 边界 |
| correctness / allocator safety | 通过 | 新增四次 TP4 运行无 assertion，SLO 100% |
| 真实 `decode_capacity_trimmed/deferred` | 未通过 | 128/256/512 点均为 0/0 |
| page-aware fairness 改善 | 未验证 | 没有多个 decode candidate，实验 084 cursor 未被线上 workload 使用 |
| throughput >= token 的 98% 且 P99 TTFT 不回退 5% | 未接受 | 128/512 单次结果方向相反，缺少重复和 trim-path |
| 修改默认策略 | 拒绝 | 继续 `token` 默认、`page-aware` opt-in |

## 6. 原始产物

- 128 token：`learning/results/085_pilot_token_128c_out96_rerun.{json,md}`、
  `085_pilot_page_aware_128c_out96.{json,md}`。
- 512 token：`learning/results/085_pilot_token_512c_out128.{json,md}`、
  `085_pilot_page_aware_512c_out128.{json,md}`。
- 每个运行都有对应的 `*_telemetry.jsonl`；可用 `rg` 搜索
  `decode_capacity_trimmed` 和 `decode_capacity_deferred` 复核计数为 0。
- 先前失败的 CUDA 11.5 编译日志和 128-token output=128 的非法边界日志也保留在
  `learning/results/`，作为环境/参数边界证据，不纳入性能均值。

## 7. 下一步

实验 085 的阶段 1 已完成，但真实 trim gate 未完成。下一步有两条明确路线：

1. 做 085b 定向 workload：增加异构 output/input、提高同时 pending 数，或限制
   `max_running_req`，让多个 decode request 同时成为候选并自然触发 trim/deferred；
2. 若优先推进更容易独立审查的 upstream 改动，则进入实验 086 的 `Req` 预分配 buffer，
   同时把 085 的真实 trim gate 保留为 page-aware PR 的前置条件。

在没有 085b 真实 trim 样本前，不把 page-aware 扩大为默认，也不把本阶段的 +1.80% 单次
结果写入性能宣传或简历数字。
