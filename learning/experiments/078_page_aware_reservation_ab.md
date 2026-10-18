# 实验 078：页感知 KV 预留策略 A/B

## 1. 实验目的

实验 077 说明在线 admission 可能在 `_fit_decode_batch` 之前消化容量压力。本实验继续
研究 `prefill_reserve_policy`：

- `token`：按输入扩展 token 加最大输出 token 预留，保留旧路径；
- `page-aware`：把最终序列和已有 cache 分别向 page 边界取整后再预留，避免低估未来
  page 分配。

目标是验证页感知策略是否能减少真实 decode 容量裁剪，以及它对吞吐和 tail latency 的
代价。该实验不修改默认策略。

## 2. 公平性与环境

- GPU：L20 GPU 2，单卡 TP1；
- 模型：Qwen3-4B-AWQ ModelScope snapshot；`float16`；
- `page_size=32`，`num_pages=4`，KV 容量 128 token；
- `max_seq_len_override=128`，`max_prefill_length=64`，`max_prefill_streak=1`；
- `decode_active_prefill_length=32`，`cuda_graph_max_bs=8`，shape policy=`coarse`；
- 负载：8 个请求，输入 17 token，输出 16 token，arrival rate 1000 req/s；
- seed：7800082；重复 3 次；TTFT/TPOT SLO 均为 60 s；
- 两组唯一变量为 `--prefill-reserve-policy token/page-aware`，其他参数和请求 trace 相同。

选择 `input=17/output=16` 是为了制造页边界：token 策略估算约 33 token，而 page-aware
策略会把最终 33 token 向上取整为 64 token。这样 4 page cache 下会出现“token admission
看似还能容纳，decode 下一步实际需要新 page”的场景。

## 3. 结果

| 策略 | 平均输出吞吐 tok/s | 跨重复标准差 | P90 TTFT ms | P99 TPOT ms | P90 E2E s | SLO | decode_capacity_trimmed |
|---|---:|---:|---:|---:|---:|---:|---:|
| token | 338.39 | 3.23 | 268.51 | 42.89 | 0.372 | 100%/100% | 每轮 16 次 |
| page-aware | 294.93 | 10.11 | 358.77 | 5.69 | 0.426 | 100%/100% | 0 次 |

相对 `token`，`page-aware` 平均吞吐下降 12.84%，P90 TTFT 上升 33.62%，但 P99 TPOT
从 42.89 ms 降到 5.69 ms。两组均为 8/8 请求完成，没有 allocator assertion、请求丢失
或进程异常。

原始结果：

- `learning/results/078_token_boundary_r3.md`；
- `learning/results/078_page_aware_boundary_r3.md`；
- 对应 JSON 文件保存逐重复数据；
- `learning/results/078_token_boundary_r3_telemetry.jsonl`；
- `learning/results/078_page_aware_boundary_r3_telemetry.jsonl`。

## 4. Telemetry 证据

`token` 三个 burst trace 都产生 16 次 `decode_capacity_trimmed`，每次请求 batch 从
3 裁到 1 或 2；最大请求 batch 为 3，最小执行 batch 为 1。事件中记录了
`deferred_uids` 和 `selected_wait_rounds`，说明实验 076 的线上字段已经真正命中。

`page-aware` 三个 trace 均只有 `selected/forward_*/result_sent` 事件，没有容量裁剪或
deferred 事件。它通过更早的预留把风险转化成 admission 排队，而不是让 decode 阶段临时
裁剪。这是稳定性与利用率之间的明确 trade-off。

## 5. 结论与决策

1. 页感知预留确实消除了本页边界负载下的 decode capacity trimming，并显著降低 TPOT tail。
2. 在极小 KV cache 的 L20 压力点上，它以吞吐 -12.84%、P90 TTFT +33.62% 为代价；当前
   三次重复不支持把它作为全局默认策略。
3. 建议保留 `page-aware` 作为长输出、严格尾延迟或 allocator 安全优先场景的 opt-in
   开关；常规负载继续使用 `token`，并依靠 071/073/076 的容量保护、公平 cursor 和
   telemetry 兜底。
4. 该实验没有证明所有 workload 都有同样比例的收益；下一步若要推广，必须在更大 page
   数、不同输出长度和真实 TP4 模型上做至少三次重复。

## 6. 面试表达

我把 KV cache 的容量模型从 token 粒度和 page 粒度做了 A/B。token 粒度吞吐更高，但在
跨 page 边界时 decode batch 会被临时裁剪，三轮都产生 16 次容量事件，P99 TPOT 达到
42.89 ms。page-aware 通过 admission 阶段提前预留整页，裁剪次数降为 0，P99 TPOT 降到
5.69 ms，但吞吐下降 12.84%、P90 TTFT 上升 33.62%。因此没有简单追求“更安全”，而是
把它设计成 opt-in 策略：根据 tail latency 与吞吐目标选择 reservation 粒度。
