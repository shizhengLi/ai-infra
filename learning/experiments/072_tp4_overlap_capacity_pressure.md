# 实验 072：TP4 overlap 限页压力与容量保护可观测性

## 1. 实验目标

实验 071 增加了 decode admission guard，但单卡 4B 冒烟中 decode batch 基本只有一个请求，
没有证明 guard 在真实 overlap 压力下会介入。实验 072 在 L20 四卡 TP4 上运行限页压力，验证：

1. 真实 Qwen3-32B-AWQ TP4 路径是否能在小 KV cache 下完成请求；
2. 新增的 `decode_capacity_trimmed` telemetry 是否能观察到裁剪前后的 batch；
3. 高并发下是否仍出现实验 070 的 `0 evictable` assertion、请求丢失或 starvation。

## 2. 实现改动

在 scheduler 的 pipeline telemetry 中加入容量裁剪事件：

```text
decode_capacity_trimmed:
  requested_batch_size = 裁剪前 decode batch 大小
  selected_batch_size  = 容量保护后实际执行大小
```

同时修正 `_record_pipeline_event`，允许事件保留外部传入的 `requested_batch_size`，而不是被
实际 batch size 覆盖。新增单测验证一个实际大小为 2、原始大小为 4 的事件可以被正确记录。
默认调度逻辑和容量判断公式没有变化。

## 3. 实验配置

- 硬件：L20 GPU 2/3/6/7，TP=4，Gloo 通信（本机没有可链接的 NCCL library）。
- 模型：本地 Qwen3-32B-AWQ，AWQ 权重，FP16 runtime，FlashInfer attention。
- KV：`page_size=1`，`num_pages=96`，最大序列长度 512。
- 调度：`page-aware`、safe retraction、cooldown=4、pending gate=32、prefill budget=64、
  decode-active budget=32，CUDA Graph 默认 capture。
- 服务端口：1928；原始 telemetry：`learning/results/072_tp4_capacity_telemetry.jsonl`。

## 4. 结果

### 4.1 短 burst

12 请求、输入 `[8, 16, 24]`、输出 16 token、到达率 20 req/s：

| 指标 | 结果 |
| --- | ---: |
| 完成请求 | 12/12 |
| 输出吞吐 | 123.84 token/s |
| P90 TTFT | 1006.40 ms |
| P99 TPOT | 53.89 ms |
| allocator assertion | 0 |
| `decode_capacity_trimmed` | 0 |
| 最大 decode batch | 3 |

### 4.2 长 burst

32 请求、输入 `[8, 16, 24]`、输出 64 token、到达率 100 req/s：

| 指标 | 结果 |
| --- | ---: |
| 完成请求 | 32/32 |
| 输出吞吐 | 89.62 token/s |
| P90 TTFT | 19710.35 ms |
| P99 TTFT | 21797.22 ms |
| P99 TPOT | 11.51 ms |
| allocator assertion | 0 |
| `decode_capacity_trimmed` | 0 |
| 最大 decode batch | 3 |
| safe retraction | 0 次 |

原始 benchmark 结果见 `learning/results/072_tp4_capacity_pressure.md` 和
`learning/results/072_tp4_capacity_pressure_long.md`。

## 5. 解释与边界

两轮压力均完成全部请求，说明在这组配置下没有复现实验 070 的 allocator 崩溃。第二轮的
P90/P99 TTFT 已明显恶化，表明 96 page cache 和 100 req/s burst 主要表现为排队，而不是
吞吐不足；因此不能把这次稳定性结果包装成性能优化。

更重要的是，telemetry 没有出现 `decode_capacity_trimmed`。这说明当前 prefill admission、
safe retraction 和 decode manager 已经把可同时运行的 decode batch 限制在 3 个以内，测试没有
真正进入 071 guard 的“候选 batch 超过可见容量”分支。换言之，072 验证了高压路径不崩溃和
观测能力，但没有完成 guard 的真实命中验证。

## 6. 结论与下一步

实验 072 通过稳定性 gate：TP4 限页压力 44 个工作请求全部完成，两个 trace 都没有 assertion
或请求丢失；性能 gate 不适用，长 burst 的 TTFT tail 很差，且没有观察到 guard 裁剪。

下一步应做确定性边界实验，而不是继续盲目增加请求数：构造已占用大部分 page、同时拥有多个
`running_reqs` 的 scheduler 状态，直接验证 `_fit_decode_batch` 能把候选 batch 从 N 裁到可用
前缀；再补一个极小 `num_pages` 的 TP1/TP4 在线 trace，要求 telemetry 至少出现一次
`decode_capacity_trimmed`，并检查裁剪是否导致某个 UID 长时间饥饿。

