# 实验 086：Req 预分配 token buffer 的上游化验收

## 1. 实验目的

`Req.append_host()` 位于 scheduler 处理每个 decode token 的 CPU 路径。旧实现用
`torch.cat([ids, next_token])` 扩展 token tensor，每一步都会重新分配并复制已有 prefix，
理论复制量为 O(output_len^2)。当前实现按 `max_device_len` 一次分配 `_input_buffer`，
通过有效长度 view 暴露 `input_ids`，每次 append 只写入一个位置。

本实验验证四件事：

1. 4096/8192 长输出下 allocation 次数和复制量是否仍然是预期的线性/常数级；
2. append 后 token identity、有效长度和容量边界是否正确；
3. 预分配是否真的降低本机 CPU wall-time，还是只降低 allocator/copy pressure；
4. 结合实验 060 的 L20 A/B，确定该改动适合怎样的 upstream PR 描述。

## 2. 实现语义

- `Req.__post_init__()` 分配 `torch.empty(max_device_len)`，并复制一次 prompt。
- `input_ids` 始终是 `_input_buffer[:device_len]` 的 view，不暴露未生成的 capacity 区域。
- `Engine.forward_batch()` 完成一个 decode step 后调用 `complete_one()`，scheduler 再调用
  `append_host()` 写入采样 token；两步顺序保持旧生命周期不变。
- `append_host()` 强制只接受一个 token，并在 capacity 越界时失败。
- `MINISGL_REQ_APPEND_MODE=cat` 仍可用于 legacy regression，但不是部署推荐选项。

## 3. 离线长输出 benchmark

命令：

```text
PYTHONPATH=python .venv/bin/python benchmark/offline/bench_req_append.py \
  --input-len 1024 --output-len 4096 --repeats 5

PYTHONPATH=python .venv/bin/python benchmark/offline/bench_req_append.py \
  --input-len 1024 --output-len 8192 --repeats 5
```

结果如下。`speedup` 定义为 `cat_mean / preallocated_mean`，小于 1 表示预分配更慢。

| input | output | 方法 | 平均 wall-time | allocation 次数 | 理论 copy bytes | copy reduction |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | 4096 | preallocated | 44.76 ms | 1 | 20,480 | 2,457x |
| 1024 | 4096 | legacy cat | 13.64 ms | 4,096 | 50,323,456 | - |
| 1024 | 8192 | preallocated | 91.70 ms | 1 | 36,864 | 4,551x |
| 1024 | 8192 | legacy cat | 29.58 ms | 8,192 | 167,755,776 | - |

预分配在 4096/8192 输出下分别把 allocation 降低 4096/8192 倍，把理论复制量降到原来的
约 1/2457 和 1/4551；但本机 Python microbenchmark wall-time 分别是 legacy cat 的
3.28x 和 3.10x。原因是这个 benchmark 的单 token 写入主要落在 Python loop 和
`next_token.item()`，而 `torch.cat` 的小型一维 tensor 拼接由 C++ 实现，不能把它当成
scheduler 端到端 CPU 结论。

## 4. 正确性回归

命令：

```text
PYTHONPATH=python .venv/bin/pytest -o addopts='' -q \
  tests/core/test_req_buffer.py tests/core/test_safe_retraction.py
```

结果：`10 passed in 4.20s`。

覆盖内容包括：

- capacity 分配但只暴露有效长度；
- 连续 append 后 token 内容和 `device_len`；
- `_input_buffer.data_ptr()` 在整个生命周期保持稳定；
- append 到最大 capacity 后停止 decode；
- legacy `cat` 回归模式的 allocation/copy 计数；
- 单 token 输入边界；
- safe retraction 场景下的 append 生命周期。

实验 060 已经完成完整 `tests/core` 回归和 L20 长输出 A/B，因此本实验不重复启动 GPU
服务，只补充长输出 CPU 数据和 focused tests。

## 5. 与真实 L20 serving 的关系

实验 060 的 Qwen3-32B FP8 TP4 L20 A/B 已经给出端到端边界：

| 指标 | preallocated | legacy cat | 变化 |
| --- | ---: | ---: | ---: |
| throughput | 209.65 | 210.10 token/s | -0.21% |
| host append time | 60.71 | 62.22 ms | -2.43% |
| host append allocations | 0 | 3,088 | -100% |
| host copied tokens | 3,088 | 6,319,880 | -99.95% |

两组都 6/6 成功，TPOT SLO=100%，但 TTFT SLO 只有 50%，端到端吞吐没有改善。结合本次
4096/8192 CPU microbenchmark，合理定位是：预分配降低了长期运行和超长输出下的 allocator
pressure、copy traffic 与内存抖动风险，但当前 GPU-bound serving workload 的主要瓶颈不在
append wall-time。

## 6. Upstream PR 决策

该改动适合独立 upstream PR，但 PR 描述必须准确：

- 可以宣称：消除每 token 的 tensor allocation；把 token copy 从 O(n) 累积降为一次 prompt
  copy 加每 token 一次写入；保留有效长度和 capacity 不变量；
- 不能宣称：当前 L20 serving throughput 提升，或单线程 Python append wall-time 必然下降；
- 必须带上 abort、最大 capacity、token identity、streaming 顺序和 legacy regression 测试；
- benchmark 同时报告 allocation/copy、append CPU 时间、scheduler CPU 时间和 E2E tok/s，
  将“基础设施收益”和“服务收益”分开。

验收结论：correctness gate 通过，allocation/copy gate 通过，CPU wall-time -50% gate
未通过，端到端吞吐 gate 未通过。保留预分配实现和 `cat` regression mode，不修改其它
scheduler 默认策略。

## 7. 原始依据

- `benchmark/offline/bench_req_append.py`：本次 4096/8192 benchmark 实现；
- `tests/core/test_req_buffer.py`、`tests/core/test_safe_retraction.py`：10/10 focused tests；
- `learning/experiments/060_req_token_buffer.md`：完整 L20 A/B、全量 core 回归和历史边界。
