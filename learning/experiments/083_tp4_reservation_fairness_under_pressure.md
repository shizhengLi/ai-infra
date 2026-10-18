# 实验 083：256-token KV cache 下 reservation 公平性 A/B

## 实验目的

实验 082 证明在 256-token cache、16 个长请求的压力下，scheduler 会把 decode batch
压到 1，并通过 admission 排队保护 allocator，但没有命中 capacity trim 分支。本实验在
完全相同的压力下比较两种 prefill reservation：

- `token`：按 token 数预留未来需求；
- `page-aware`：按 page 边界向上取整后预留。

重点观察 reservation 是否改变吞吐、TTFT/TPOT tail、请求完成顺序和容量 telemetry。

## 固定条件

- GPU：L20，`CUDA_VISIBLE_DEVICES=2,3,6,7`
- 模型：Qwen3-32B-AWQ；TP4；dtype `float16`
- 通信：默认 PyNCCL；Gloo 只负责 CPU rank/NCCL UID bootstrap
- KV cache：`page_size=16`、`num_pages=16`，总容量 256 token
- `max_seq_len_override=256`
- workload：16 个请求，输入 17 token，输出 128 token，arrival rate=80 req/s
- 重复次数：3；每组 48/48 请求完成；TTFT/TPOT SLO 均为 120 s
- prefill：normal budget=64、decode-active budget=32、最大 prefill streak=1
- CUDA Graph：coarse，batch shapes `[1,2,4,8]`
- benchmark seed：8300071

完整结果：

- [`083_tp4_token.md`](../results/083_tp4_token.md)
- [`083_tp4_page_aware.md`](../results/083_tp4_page_aware.md)

## 性能结果

| policy | 吞吐 tok/s | 标准差 | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P90 E2E s | GPU util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token | 88.99 | 0.39 | 19956.95 | 21390.44 | 12.09 | 21.350 | 96.89% |
| page-aware | 90.14 | 0.58 | 19718.82 | 21123.13 | 11.73 | 21.089 | 96.62% |

相对 `token`，`page-aware` 吞吐高 1.30%，P90 TTFT 低 1.19%，P99 TPOT 低 2.98%。两组
差异小于当前极端排队场景的运行波动，不能据此宣称 page-aware 有稳定性能收益。

## Telemetry 与完成顺序

两组各记录 4 个 trace（含 warmup），结果完全一致：

- `decode_capacity_trimmed=0`、`decode_capacity_deferred=0`、retraction=0；
- 最大 pending prefill 为 15 个请求、375 token；
- 6160 个 selected decode batch 全部为 batch size=1；
- token 和 page-aware 都按请求 UID 顺序完成前 3 轮，page-aware 第 3 轮只有 UID 12/13
 发生相邻交换，未形成长期饥饿或明显公平性改善。

这说明在当前 cache pressure 下，请求已经在 admission 阶段串行化。两种 reservation 都没有
进入容量裁剪/fairness cursor 分支，因此 page-aware 没有足够机会改变服务顺序。

## 结论

1. 在 256-token 极小 cache 下，`token` 和 `page-aware` 都保持 48/48 请求完成，且没有
   allocator assertion 或容量裁剪事件。
2. page-aware 的小幅指标改善不具备稳定收益证据；两组 decode batch 和 pending 状态完全
   相同，不能把这次结果解释为 reservation 优化成功。
3. 要验证 073-076 的容量公平性，需要构造“同一 decode round 有多个候选请求，但未来 page
   demand 只有部分请求能容纳”的确定性压力，而不是让 admission 直接退化为单请求 batch。
4. 默认继续保留 `token` reservation；`page-aware` 作为安全/尾延迟优先场景的 opt-in，
   等能命中容量裁剪分支的 trace 后再评估。

## 面试表达要点

这个 A/B 的价值在于它是一个中性结果：我保持模型、容量、请求轨迹和并行配置完全一致，
只切换 reservation policy。两组最终都被 admission 串行化，因此 page-aware 几乎没有改变
性能或公平性。这说明做调度优化时必须先确认实验真的命中了目标代码路径，否则“看起来更快”
可能只是噪声；telemetry 既要记录指标，也要证明优化分支被执行。
