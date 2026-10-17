# 实验 067：批量 TokenizeManager

日期：2026-10-17  
状态：Stage 3 完成；1 ms 微批窗口改善轻负载尾延迟但未达到 gate，默认保持 0 ms

## 背景与目标

`python/minisgl/tokenizer/tokenize.py` 原来对每条 `TokenizeMsg` 单独调用
`encode()`，chat 请求还要先单独执行 `apply_chat_template()`。这会重复进入 Python 和
Rust tokenizer 边界；当多个请求已经在 ZMQ 队列中时，CPU tokenizer 成为前端到 scheduler
的串行小瓶颈。上游 mini-sglang 的 #117/#55 也把批量 tokenization 作为独立改进方向。

本实验的目标是：

1. plain text 和 chat template 分开批量编码，避免 chat 输入格式互相影响。
2. 使用 ragged `input_ids`，不引入 padding token，也不改变 scheduler 看到的长度。
3. 按原消息索引恢复结果顺序，保证 uid、abort 和 streaming 的消息关联不受影响。
4. 用服务端可配置的 batch 上限控制一次 tokenizer worker 最多消费的消息数。

## 实现原理

`tokenize_worker` 原来每轮从 ZMQ 拉取第一条消息，然后最多继续拉取 `local_bs - 1` 条。
此前 launch 始终传 `local_bs=1`，所以即使 `TokenizeManager.tokenize()` 接收的是列表，
列表也几乎总是只有一个元素。

本实验做了两层修改：

- `ServerArgs` 增加 `--tokenizer-batch-size`，默认 32，并把它传给 tokenizer worker；
  第一条消息仍立即触发处理，只有队列中已经有消息时才继续凑批，因此没有人为的固定等待
  窗口。`num_tokenizer=0` 的共享 detokenizer 也使用该上限。
- `TokenizeManager` 先按 `str` / chat message list 分组。plain 组调用 tokenizer 的
  batch API；chat 组直接批量调用 `apply_chat_template(..., tokenize=True)`。两组分别得到
  ragged token id 列表，再写回原始索引。每个结果最终转换成 CPU `torch.int32` 一维 tensor，
  与旧路径的后端协议一致。

没有使用 `padding=True` 或 `return_tensors="pt"`：不同输入长度不需要补齐，避免把 padding
带入 KV cache 和 prefill 预算。`TokenizeMsg.uid` 不在 TokenizeManager 内重排，只依赖结果
列表与输入消息的同序关系；server 随后用同一个 `zip(tokenize_msg, tensors)` 构造
`UserMsg`，所以 uid、sampling 参数和输入 token 一一对应。

## 正确性验证

新增 `tests/core/test_tokenize_batch.py`，覆盖：

- mixed plain/chat 输入被分成两次 batch 调用；
- 交错输入在返回时恢复原顺序；
- 结果是 `torch.int32`，空 batch 返回空列表；
- `--tokenizer-batch-size` 默认值、显式配置和 `0` 的非法值检查。

测试结果：

```text
tests/core/test_tokenize_batch.py                 3 passed
tokenize_batch + kv_cache_dtype focused tests   13 passed
```

另外使用本地 Qwen3-0.6B tokenizer，对 3 条 plain 和 2 条 chat 请求逐条执行旧实现，再与
批量实现比较。两条路径的每个请求 token id 完全一致：

```text
plain_equal True
chat_equal True
```

因此这次改动没有改变 chat generation prompt、special token 或输入长度。

## CPU tokenizer 微基准

固定 32 条混合请求（16 plain + 16 chat），每条请求包含 2--6 次重复短语，预热 5 次后
测量 20 次。机器上的 tokenizer 为本地 Qwen3-0.6B tokenizer，结果单位为 wall time：

可复现命令：

```bash
PYTHONPATH=python .venv/bin/python benchmark/offline/bench_tokenizer_batch.py \
  --model /data2/lszlsz/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --requests 32 --repeats 20
```

| 路径 | median (ms) | mean (ms) |
| --- | ---: | ---: |
| 旧实现：32 次逐条 encode/template | 6.153 | 7.097 |
| 新实现：plain/chat 两个 batch | 2.725 | 2.826 |

median 加速为 **2.26x**，mean 加速为 **2.51x**，达到预注册的“32 条混合请求至少 2x”
门槛。这个结果只代表 CPU tokenizer 阶段；它不等价于端到端生成吞吐提升，因为低到达率
时 tokenizer worker 可能只收到一条消息，GPU prefill/decode 仍然是主要耗时。

## Stage 2：真实 L20 在线验证

在 GPU 2/3/6/7 上启动 Qwen3-32B FP8 TP4，固定 FlashInfer、coarse Graph 32、
`max_prefill_streak=1`、decode-active prefill 2304 和相同 seed。每个 profile 使用
12 请求、输入长度 128/256/512、输出 64 token、1.5/2.5 req/s、每档 2 次重复。

| Arrival rate | batch=1 tok/s | batch=32 tok/s | 吞吐变化 | batch=1 P90 TTFT ms | batch=32 P90 TTFT ms | TTFT 变化 | batch=1 P99 TPOT ms | batch=32 P99 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 102.86 | 102.81 | -0.06% | 223.44 | 229.32 | +2.63% | 107.92 | 107.89 |
| 2.5 | 179.89 | 179.66 | -0.13% | 226.37 | 226.85 | +0.21% | 166.96 | 167.88 |

两组 profile 的 TTFT 和 TPOT SLO 均为 100%，但 batch=32 没有达到预注册的 TTFT P90
下降 5% 门槛。原因符合队列模型：1.5/2.5 req/s 下 tokenizer worker 大多数时候只收到
一条消息，`local_bs=32` 不会等待额外窗口，因此 batch API 很少能形成 32 条 batch；CPU
阶段的 2.26x 微基准收益没有传递到端到端 TTFT。

### 多 tokenizer streaming smoke

使用 `--dummy-weight`、TP4、`max-seq-len=1024` 的轻量服务，不用于性能结论：

- `num_tokenizer=2`：8 个并发请求均返回 8 个 streaming chunk，均有文本且
  `finish_reason=stop`。
- `num_tokenizer=4`：同样的 8/8 请求全部通过，未观察到 uid 交叉或输出丢失。
- `num_tokenizer=0`：真实 Qwen3-32B FP8 在线矩阵的 24 个请求全部 HTTP 200，streaming
  正常。

本次 smoke 没有强行中断正在生成的请求，因此 abort 专项仍需后续用可控客户端断开连接和
短输出请求补测；当前结果足以确认多 tokenizer 的正常路由和 streaming 基本路径。

原始在线结果：

- [batch=1 Markdown](../results/067_bs1.md)
- [batch=32 Markdown](../results/067_bs32.md)

## Stage 3：短微批等待窗口

为验证“请求太稀疏导致凑不成 batch”的假设，增加显式 `--tokenizer-batch-wait-ms`。首条
消息是 `TokenizeMsg` 时，worker 最多等待该时长收集后续消息；纯 detokenize/abort 首条消息
不等待。默认值为 `0.0`，因此原有行为完全兼容。

在同一 Qwen3-32B FP8 TP4、相同 seed 和 workload 下比较 `batch_size=32, wait=1.0 ms`
与此前 `batch_size=1, wait=0 ms`：

| Arrival rate | wait=0 tok/s | wait=1 ms tok/s | 吞吐变化 | wait=0 P90 TTFT ms | wait=1 ms P90 TTFT ms | TTFT 变化 | wait=0 P99 TPOT ms | wait=1 ms P99 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 102.86 | 102.93 | +0.07% | 223.45 | 213.79 | -4.32% | 107.92 | 107.50 |
| 2.5 | 179.89 | 179.83 | -0.03% | 226.37 | 226.55 | +0.08% | 166.96 | 165.71 |

1.5 req/s 的 P90 TTFT 已接近但没有达到预注册的 -5% 门槛；2.5 req/s 几乎没有变化。吞吐
没有显著回归，说明 1 ms 等待窗口的成本可控，但当前 workload 下不足以成为默认优化。

原始结果见 [wait=1 ms Markdown](../results/067_wait1.md)。

## 结论

批量 TokenizeManager 是一个边界清晰、可回滚的前端 CPU 优化：默认 batch 上限为 32，
输入协议和输出 token 完全兼容，混合请求 CPU tokenizer wall time 降低约一半。Stage 2
证明它在低到达率下没有显著端到端回归；1 ms 等待窗口只将 1.5 req/s P90 TTFT 改善
4.32%，仍未达到 -5% gate。因此保留 `--tokenizer-batch-size 32` 作为 CPU 高并发候选，
`--tokenizer-batch-wait-ms` 默认保持 0.0；不把本实验宣称为低负载 L20 TTFT 优化。

## 代码与测试

- `python/minisgl/tokenizer/tokenize.py`
- `python/minisgl/tokenizer/server.py`
- `python/minisgl/server/args.py`
- `python/minisgl/server/launch.py`
- `tests/core/test_tokenize_batch.py`
