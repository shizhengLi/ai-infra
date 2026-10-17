# 实验 060：预分配 Req token buffer

日期：2026-10-17  
状态：实现、离线正确性和 L20 长输出 A/B 完成；未达到 CPU wall-time -50% gate

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
92 passed (latest run)
```

新增 `tests/core/test_req_buffer.py` 覆盖：有效长度 view、容量大小、连续 append、最大
capacity、底层指针稳定、单 token 边界和 legacy `cat` 回归模式。`compileall` 和
`git diff --check` 通过。

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

## L20 长输出 A/B

为区分复杂度收益和真实服务收益，增加了实验开关 `MINISGL_REQ_APPEND_MODE`：默认值为
`preallocated`，设置为 `cat` 时恢复旧实现，仅用于 A/B 回归。两组使用空闲 GPU 2/3/6/7，
Qwen3-32B FP8 TP4、FlashInfer、CUDA Graph max batch 32、相同 prefill/decode 参数和
`CUDA_HOME=/usr/local/cuda-12.8`。workload 为 256/1024/4096 混合输入、512 输出 token、
1.5 req/s、6 请求、seed `6000042`，每组 1 次；开启 prefill telemetry。

| 指标 | preallocated | legacy cat | 变化 |
| --- | ---: | ---: | ---: |
| 输出吞吐 token/s | 209.65 | 210.10 | -0.21% |
| 平均 TTFT ms | 1473.00 | 1455.77 | +1.18% |
| P90 TTFT ms | 2494.32 | 2468.68 | +1.04% |
| 平均 TPOT ms | 21.844 | 21.836 | +0.03% |
| host append calls | 3088 | 3088 | 0% |
| host append time | 60.71 ms | 62.22 ms | -2.43% |
| host append allocations | 0 | 3088 | -100% |
| host copied tokens | 3088 | 6,319,880 | -99.95% |
| max pending prefill tokens | 8221 | 8221 | 0% |

两组均为 6/6 请求成功，TTFT SLO 为 50%，TPOT SLO 为 100%，峰值显存均为 16413 MiB/card。
结果文件为 `learning/results/060_preallocated_poisson.*` 和
`learning/results/060_cat_poisson.*`。端到端吞吐和 TTFT 没有改善，说明当前 L20 workload
的主要瓶颈不是 host append；但 telemetry 验证了预分配消除了每 token allocation 和
O(n) token copy。实验 060 的 CPU wall-time `-50%` gate **未通过**，保留该改动作为降低
allocator pressure 的基础设施，不改变其它 scheduler 默认策略。

## 下一步

1. 若要继续优化 CPU path，评估批量化 host token 写入或减少 Python 调度边界；单纯替换
   allocator 已经证明不会改变当前 GPU-bound workload 的主要指标。
2. 把 legacy `cat` 模式保留为回归工具，不作为部署选项文档推荐。
