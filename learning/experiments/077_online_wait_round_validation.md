# 实验 077：L20 在线页边界与等待轮数 telemetry 验证

## 1. 实验目的

实验 076 增加了 `decode_capacity_trimmed` 和 `decode_capacity_deferred` 的等待轮数字段。本实验把 KV cache 压缩到极小规模，在 L20 上运行真实 HTTP 请求，验证服务正确性、telemetry 链路以及没有命中裁剪时的负向证据。

本实验只验证正确性和观测链路，不以单次 benchmark 宣称吞吐提升。

## 2. 环境与配置

- GPU：L20，`CUDA_VISIBLE_DEVICES=2`；模型：Qwen3-4B-AWQ ModelScope snapshot；
- TP：1；运行 dtype：`float16`；量化：AWQ；
- `page_size=32`，`num_pages=4`，理论 KV 容量为 128 token；`max_seq_len_override=128`；
- `max_prefill_length=64`，`decode_active_prefill_length=32`，`max_prefill_streak=1`；
- `cuda_graph_max_bs=8`，shape policy=`coarse`，`prefill_reserve_policy=token`；
- telemetry：`learning/results/077_page_boundary_telemetry.jsonl`；代码版本：`977594e`。

启动时曾误用 `--prefill-safe-retraction=false`；该参数是无值布尔开关。删除参数后服务正常启动，正式配置默认值为 `False`。

## 3. 压测结果

### 3.1 短请求边界冒烟

3 个请求，输入长度 `[8, 16, 24]`，输出 12 token，Poisson arrival rate 50 req/s，seed 为 7700071。3/3 请求完成，TTFT/TPOT SLO 均为 100%，输出吞吐 184.60 token/s，P90 TTFT 123.29 ms，P90 E2E 0.172 s。

### 3.2 并发 burst

8 个请求，输入长度 16，输出 32 token，arrival rate 1000 req/s，seed 为 7700072。8/8 请求完成，TTFT/TPOT SLO 均为 100%，输出吞吐 315.08 token/s，P90 TTFT 657.95 ms，P90 E2E 0.803 s，GPU 平均利用率 60.33%。

原始结果见 `learning/results/077_page_boundary_pressure.md`、`077_page_boundary_pressure_burst.md` 及对应 JSON 文件。

## 4. Telemetry 观察

JSONL 共记录 4 个 scheduler trace：两次请求批次各对应一个 trace。事件计数为：

- 第一轮：`request_received=1`，`selected=16`；
- 第二轮：`request_received=3`，`selected=25`；
- 第三轮：`request_received=1`，`selected=16`；
- 第四轮：`request_received=8`，`selected=160`。

本次 `decode_capacity_trimmed=0`、`decode_capacity_deferred=0`，没有 `selected_wait_rounds` 或 `deferred_uids` 的线上样本。并发 burst 中 decode 批次达到 2，但 admission 在形成候选 batch 时已经把活跃集合限制在当前可容纳范围内，没有把超容量 batch 交给 `_fit_decode_batch`。这与实验 072 的线上观察一致。

## 5. 结论

1. 在 4 page/128 token 的极小 KV cache 下，Qwen3-4B-AWQ 单卡服务可以完成短请求和并发 burst，没有 allocator assertion、进程退出或请求丢失。
2. 低并发 P90 TTFT 为 123.29 ms，高并发 P90 TTFT 为 657.95 ms；小 cache 下 tail latency 随并发上升。
3. 本实验没有命中 `decode_capacity_trimmed/deferred`，不能证明等待轮数 telemetry 已经在线采样，也不能宣称性能收益。它验证了 admission 可能在 `_fit_decode_batch` 前消化容量压力。
4. 确定性单测仍是等待轮数字段的覆盖来源；真实裁剪样本需要专门构造多个活跃请求同时跨 page 边界的 trace，不能为了制造 telemetry 而改变生产默认 admission。

## 6. 面试表达

我把 L20 上的 KV cache 缩到 4 个 32-token page，用真实 HTTP burst 验证容量保护和 telemetry。所有请求都完成，SLO 保持 100%，但裁剪事件计数为 0。通过 selected trace 发现 admission 在 `_fit_decode_batch` 之前已把 decode batch 限制到可容纳大小，所以没有把“没有触发事件”误报成公平性收益，而是把它作为调度路径分层的证据：单测覆盖裁剪公平性，线上测试覆盖 allocator 安全和 tail latency，二者结论分开。
