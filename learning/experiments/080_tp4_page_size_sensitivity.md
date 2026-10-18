# 实验 080：TP4 下 KV cache page size 敏感性

## 实验目的

在 L20 四卡 TP4、Qwen3-32B-AWQ 的真实在线请求上，比较 `page_size=8/16/32`
对吞吐、首 token 延迟（TTFT）、单 token 延迟（TPOT）和调度行为的影响。三组实验
固定总 KV cache 容量为 128 token，只改变 page 的粒度：

| page size | num pages | 总容量 |
| ---: | ---: | ---: |
| 8 | 16 | 128 token |
| 16 | 8 | 128 token |
| 32 | 4 | 128 token |

这样可以把“容量变大”与“page 粒度变化”区分开来。`page_size=16` 复用实验 079 的
token reservation 结果，`page_size=8/32` 为本实验新增的两组运行。

## 背景与假设

Paged KV cache 用固定大小的 page 保存每个请求的 K/V。page 越小，分配和释放的粒度
越细，短请求的内部碎片通常越少，但 page table 条目、地址计算和 allocator 操作会增多；
page 越大，元数据和分配次数减少，但一个请求可能因为最后一个 page 没有用满而浪费更多
容量，且 admission 的容量跳变更大。

在总容量固定为 128 token 时，假设中等粒度 `page_size=16` 会在碎片和管理开销之间取得
较好的平衡；`page_size=8` 可能增加管理开销，`page_size=32` 可能放大内部碎片和 admission
跳变。

## 固定条件

- GPU：L20，`CUDA_VISIBLE_DEVICES=2,3,6,7`
- 模型：`Qwen3-32B-AWQ`
- Tensor parallel：4
- dtype：`float16`
- 量化：AWQ
- 通信：默认 PyNCCL；Gloo 只用于 CPU rank/NCCL UID bootstrap
- `max_seq_len_override=128`，`memory_ratio=0.9`
- prefill：`max_prefill_length=64`、`max_prefill_streak=1`、decode-active budget=32
- reservation：`token`
- CUDA Graph：`cuda_graph_max_bs=8`
- workload：每轮 8 个请求，输入 17 token，输出 16 token，Poisson arrival rate=20 req/s
- 重复次数：3；每组总计 24 个请求；TTFT/TPOT SLO 均设为 120 s

除 page 参数外，启动参数和 benchmark seed 保持一致。完整机器结果见：

- [`080_tp4_page8.md`](../results/080_tp4_page8.md)
- [`080_tp4_page32.md`](../results/080_tp4_page32.md)
- 实验 079 的 `learning/results/079_tp4_token_8page.md`（page size=16 基线）

## 吞吐与延迟结果

| page size | 吞吐 tok/s | 重复标准差 | P90 TTFT ms | P99 TPOT ms | P90 E2E s | GPU util |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 132.04 | 6.64 | 403.79 | 36.58 | 0.585 | 66.83% |
| 16（079 基线） | 136.63 | 3.90 | 511.49 | 35.01 | 0.677 | 64.40% |
| 32 | 131.74 | 2.13 | 416.96 | 80.04 | 0.628 | 66.85% |

相对 `page_size=16`：

- `page_size=8` 吞吐低 3.36%，P90 TTFT 低 21.06%，P99 TPOT 高 4.48%；
- `page_size=32` 吞吐低 3.58%，P90 TTFT 低 18.48%，P99 TPOT 高 128.58%；
- 两组新实验的 GPU 利用率约高 3.8 个百分点，但没有转化为更高的 token 吞吐。

这些数据来自单一短输入/短输出 workload，且 page size=16 来自相邻实验，不能据此宣称
page size=8 或 32 是全局最优。尤其 `page_size=32` 的 TPOT P99 明显变差，说明大 page
可能把容量管理的离散跳变传递到 decode 尾延迟。

## 调度与 telemetry

三组实验都完成 24/24 请求，均没有 allocator assertion、请求丢失或进程异常退出。
遥测汇总如下：

| page size | trace 数 | decode capacity trimmed | deferred | retraction | 备注 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 4 | 0 | 0 | 0 | decode batch 主要为 1/2 |
| 16 | 4 | 0 | 0 | 0 | decode batch 主要为 1/2 |
| 32 | 4 | 0 | 0 | 0 | 6 次 prefill budget-limited，decode 出现少量 batch=3 |

这里没有命中 `decode_capacity_trimmed`，因为 admission 在进入后续 decode fit guard 前已经
把活跃请求控制在可见容量内。因此本实验验证的是不同 page 粒度下的稳定性和常规调度开销，
不是容量裁剪公平性实验。

## 结论

1. 在 128-token 极小 KV cache 和当前短请求 workload 下，三种 page size 都稳定运行，
   没有安全性回归。
2. `page_size=16` 的吞吐最高且 TPOT 尾部最低，但 TTFT 在这次重复中高于 8/32；差异
   受 Poisson 到达和短样本影响，不能只看一个指标做默认值决策。
3. `page_size=32` 的 P99 TPOT 明显偏高，不适合作为低尾延迟场景的默认粒度；`page_size=8`
   没有表现出足够吞吐收益来抵消更细粒度管理的复杂度。
4. 暂不修改默认 `page_size`。下一步应使用更长输出（128/512 token）、更高到达率和更大
   page 数做重复矩阵，并分别观察 page 内部碎片、allocator 操作次数和 P99/P99.9 延迟。

## 面试表达要点

可以把该实验概括为：**在固定 KV token 容量的条件下做 page 粒度 A/B，而不是把 page 数
和容量一起改动。** 这样实验变量可归因；结果显示大 page 减少了元数据粒度，却可能增加
容量跳变和 decode 尾延迟，小 page 则没有在本 workload 上带来吞吐收益。工程决策是保留
`page_size=16` 作为当前基线，同时把 page-aware reservation 和更细粒度 page 保持为需要
额外 workload 证据后再启用的优化，而不是凭一次 benchmark 修改默认路径。
