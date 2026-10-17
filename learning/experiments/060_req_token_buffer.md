# 实验 060：预分配 Req token buffer

日期：2026-10-17  
状态：实现和离线正确性完成；CPU wall-time/L20 长输出验证待做

## 动机

上游 [#143](https://github.com/sgl-project/mini-sglang/pull/143) 指出，逐 token
`torch.cat` 会在每次生成时重新分配并复制已有 CPU token。当前 `Req.append_host()` 位于
scheduler 处理采样结果的路径中，理论复制复杂度接近 O(output_len^2)。

## 实现

- `Req.__post_init__()` 按 `max_device_len` 一次创建 `_input_buffer`，复制 prompt 后只把
  `input_ids` 暴露为当前有效长度 view。
- `append_host()` 将新 token 写到已分配 buffer 的下一个位置，再更新有效 view；底层
  `data_ptr()` 在完整生成过程中保持不变。
- 仍保留 `device_len`/`max_device_len` 语义。`Engine.forward_batch()` 先调用
  `complete_one()`，scheduler 再调用 `append_host()`，与原生命周期兼容。
- append 接口现在明确要求单 token，并在 capacity 越界时拒绝。

## 离线正确性

```text
.venv/bin/pytest -o addopts='' -q tests/core
91 passed in 5.05s
```

新增 `tests/core/test_req_buffer.py` 覆盖：有效长度 view、容量大小、连续 append、最大
capacity、底层指针稳定、单 token 边界。`compileall` 和 `git diff --check` 通过。

## 复杂度微基准

命令：

```text
.venv/bin/python benchmark/offline/bench_req_append.py \
  --input-len 1024 --output-len 32768 --repeats 3
```

当前实现与旧 `torch.cat` 的 wall-time 结果（本机 CPU、Python 循环）如下：

| 方法 | 平均 wall time | 分配次数 | 理论复制字节 |
| --- | ---: | ---: | ---: |
| 预分配 buffer | 0.38 s | 1 | 135,168 B |
| 每次 `torch.cat` | 0.19 s | 32,768 | 2,281,635,840 B |

预分配将理论复制量降低约 16,880 倍、分配次数降低 32,768 倍，但这个微基准中 wall-time
反而慢约 1.9 倍。原因是 benchmark 主要测 Python 层单 token 写入，而 `torch.cat` 的 C++
实现对这种小的一维 tensor 操作很快；这不是端到端 scheduler CPU 结论。实验 060 的
`append CPU time -50%` gate 尚未通过，不能仅凭复杂度数字宣称性能优化成功。

## 下一步

1. 在真实 scheduler 中加入 append 分配/复制 telemetry，测 Qwen3-32B FP8 TP4 长输出下的
   scheduler CPU 时间，而不是只测孤立 Python 循环。
2. 运行 256/1024/4096 prompt、512/2048 output 的 L20 对照，检查 token 完全一致、TTFT、
   TPOT 和吞吐。
3. 若 scheduler CPU 仍无收益，评估批量化 host token 写入或减少 Python 调度边界；保留本次
   稳定 buffer 作为低风险基础改动。
