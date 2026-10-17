# 实验 067：批量 TokenizeManager

日期：2026-10-17  
状态：Stage 1 完成；CPU tokenizer 门槛通过，在线 TTFT gate 待在固定服务矩阵中验证

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

## 尚未完成的在线 gate

预注册的完整 gate 还包括在相同 Qwen3-32B FP8 TP4 服务、相同 seed 和 1.5/2.5 req/s
Poisson trace 下比较 `--tokenizer-batch-size 1` 与 `32` 的 TTFT P90，并在
`num_tokenizer=0/2/4` 下验证输出顺序、abort 和 streaming。当前 Stage 1 已完成实现、协议
单测和 CPU 等价性/性能验证；在线矩阵需要占用四张 L20 并单独记录服务启动、Graph 和
tokenizer worker 配置，不能用本地微基准替代。

## 结论

批量 TokenizeManager 是一个边界清晰、可回滚的前端 CPU 优化：默认 batch 上限为 32，
输入协议和输出 token 完全兼容，混合请求 CPU tokenizer wall time 降低约一半。当前接受
为 Stage 1 候选；在在线 TTFT P90 达到至少 5% 改善、且多 tokenizer/abort/streaming gate
通过前，不宣称它是最终 L20 默认部署配置。

## 代码与测试

- `python/minisgl/tokenizer/tokenize.py`
- `python/minisgl/tokenizer/server.py`
- `python/minisgl/server/args.py`
- `python/minisgl/server/launch.py`
- `tests/core/test_tokenize_batch.py`
