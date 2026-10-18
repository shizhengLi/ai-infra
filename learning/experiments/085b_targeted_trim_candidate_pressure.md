# 实验 085b：定向构造 decode candidate 的真实 L20 压力

## 1. 目的

实验 085 阶段 1 的 128/256/512-token pressure matrix 没有产生
`decode_capacity_trimmed/deferred`。本实验不再扩大普通 arrival-rate 矩阵，而是根据
`PrefillAdder` 和 `DecodeManager.inflight_tokens` 的实际公式构造边界：让多个请求尽可能
同时运行，并让 token policy 的理论 token 预算接近 KV page 容量。

目标不是寻找最快 throughput，而是回答：

- token policy 是否会 admission 过多请求，随后由 `_fit_decode_batch` 做真实 trim；
- page-aware policy 是否会提前拒绝同一请求，从而避免 trim；
- `max_running_req`、tokenizer 并发和 prefill budget 是否会改变这个边界。

## 2. 关键推导

本地请求经过 chat template 后实际输入长度是 25 token，而 benchmark 的 `--input-lens 17`
只是生成器的目标长度。`page_size=16` 时，若每个请求的总长度为 `T`，最终需要的 page
数是 `ceil(T/16)`。但 admission 还会通过 `DecodeManager.inflight_tokens` 为每个运行请求
保留约 `page_size - 1 = 15` token 的边界安全空间：

```text
inflight_tokens ~= sum(remain_len) + 15 * active_decode_requests
```

因此单纯满足 `N * T <= capacity` 并不能保证 token policy 会 admission `N` 个请求；实际
还必须满足剩余 decode reservation 和已分配输入 page 的共同约束。这个额外预算正是本实验
需要验证的 guard。

## 3. 环境与配置

- GPU：L20，`CUDA_VISIBLE_DEVICES=2,3,6,7`，TP=4。
- 模型：Qwen3-32B-AWQ；dtype=float16；CUDA 12.8 nvcc；AWQ；FlashInfer attention。
- `page_size=16`，`num_pages=64`，总 KV cache=1024 token。
- `max_running_requests=8`，`num_tokenizer=4`，CUDA Graph max batch=8。
- prefill budget=64，decode-active prefill=32，maximum prefill streak=1。
- workload：Poisson rate=1000 req/s，6 个请求，实际 input=25，output=136，seed=8500088。
- token 与 page-aware 使用完全相同的 arrival trace 和 prompt；每组一次重复，仅作路径验证。

另外做了两个更小容量的定向试探：512-token cache 下 5 请求 output=83 和 output=75。
这两组也没有命中 trim，且最大 decode batch 为 4；它们用于确认结论不是 1024-token 单点
特例，不纳入正式 A/B 性能均值。

## 4. 结果

### 4.1 正式配对运行

| policy | throughput (tok/s) | P90 TTFT (ms) | P99 TTFT (ms) | P99 TPOT (ms) | P90 E2E (s) | GPU util | max decode batch | trim/deferred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token | 237.69 | 1965.16 | 1965.16 | 42.73 | 3.429 | 83.82% | 5 | 0/0 |
| page-aware | 233.60 | 2012.82 | 2012.82 | 43.67 | 3.487 | 70.91% | 5 | 0/0 |

page-aware 相对 token 的单次差异为 throughput -1.72%、P90 TTFT +2.43%、P99 TPOT +2.19%。
两组都 6/6 完成、TTFT/TPOT SLO attainment=100%，没有 allocator assertion 或 stale result。
因为没有 trim/deferred 且只有一个 repeat，这些差异不能视为策略性能回归的统计结论。

### 4.2 定向试探

| cache | requests | output | policy | max decode batch | trim/deferred |
| ---: | ---: | ---: | --- | ---: | ---: |
| 512 | 5 | 83 | token | 4 | 0/0 |
| 512 | 5 | 75 | token | 4 | 0/0 |
| 1024 | 6 | 136 | token | 5 | 0/0 |

512-token 的两个试探使用相同的 5-request burst，token policy 仍未形成超过 page capacity 的
decode batch；它们和正式 1024-token A/B 一致指向 admission guard，而不是 telemetry 丢失。

## 5. 为什么没有命中 trim

`_fit_decode_batch` 只有在 decode manager 已经返回一个无法由当前 page allocator 容纳的
完整 batch 时才会裁剪。实际运行中，prefill admission 先检查：

1. 已运行请求的 `remain_len`；
2. 每个 active decode request 的 15-token page 边界安全预算；
3. 新请求的输入和 output 估计；
4. 当前已经分配的输入 page 和可驱逐 cache。

因此 token policy 并没有把“理论 token 总量低于容量”直接转换成过大的 decode batch。正式
运行最多同时形成 5 个 decode request，allocator 在每轮都能容纳这 5 个请求，trim 分支
没有理由被调用。page-aware 也形成同样的最大 batch，两个策略的公平 cursor 仍没有线上
触发条件。

这个结果是一个重要的负证据：实验 084 的 trim/fairness 逻辑在确定性 trace 中正确，但
当前真实 admission 设计会在它之前吸收 page pressure。要构造真实 trim，可能需要改变
overlap 边界、制造跨批次的 page allocation race，或显式注入一个用于 scheduler integration
test 的多候选 batch；继续提高 arrival rate本身不会绕过这个 guard。

## 6. 验收与决策

| Gate | 状态 | 说明 |
| --- | --- | --- |
| 多请求同时 decode | 部分通过 | 最大 decode batch=5，已超过实验 085 普通 workload 的 batch=1 |
| 真实 trim/deferred | 未通过 | 1024 正式 A/B 和 512 定向试探均为 0/0 |
| token/page-aware 安全性 | 通过 | 两组均完成，SLO=100%，无 allocator assertion |
| page-aware fairness 收益 | 未验证 | 没有进入 `_fit_decode_batch`，cursor 未被使用 |
| 改变默认策略 | 拒绝 | 保持 token 默认、page-aware opt-in |

085b 不接受 page-aware 性能 gate，但接受为 admission guard 的边界证据。它将后续工作从
“继续调 arrival rate”收敛到“验证 overlap/page allocation 边界或补充 deterministic integration
test”。如果目标是上游 PR，telemetry、inflight reservation 和 focused fairness test 比
宣称 page-aware 吞吐收益更有证据价值。

## 7. 原始产物

- 正式 token：`learning/results/085b_token_1024c_6x161_tok4.{json,md}` 和对应 telemetry；
- 正式 page-aware：`learning/results/085b_page_aware_1024c_6x161_tok4.{json,md}` 和对应 telemetry；
- 定向试探：`085b_token_512c_5x100.{json,md}`、`085b_token_512c_5x100_out75.{json,md}`、
  `085b_token_1024c_6x161.{json,md}`；JSON/JSONL 保留完整 pipeline event。

复核方式：对 telemetry JSONL 搜索 `decode_capacity_trimmed` 和
`decode_capacity_deferred`，四个正式/试探文件计数均为 0。
