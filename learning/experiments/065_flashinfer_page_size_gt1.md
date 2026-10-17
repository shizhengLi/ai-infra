# 实验 065：FlashInfer 支持 page size 大于 1

日期：2026-10-17  
状态：第一阶段完成；page size 32 进入候选，page size 16 暂不接受，默认值仍为 1

## 目标

当前 mini-sglang 的 FlashInfer 路径把每个 token 当成一个 page，因此只能使用
`page_size=1`。这会让 page table、KV cache 元数据和 FlashInfer 的索引都带有较大的
元素数量。实验 065 参考上游 page-size 相关改动，打通 `page_size=16/32`，并在 L20
上验证正确性、BF16/FP8 服务、CUDA Graph 初始化以及 Qwen3-32B 的长输入性能。

## 原理

KV cache 的物理布局为 `[num_pages, page_size, num_kv_heads, head_dim]`。旧实现中
`page_size=1`，所以逻辑 token 下标可以直接作为 page 下标。page size 大于 1 后，必须
同时维护两套长度：

- `seq_lens`：每个请求的真实 token 长度，例如 17、33。
- `page_counts`：每个请求占用的 page 数，计算为 `ceil(seq_len / page_size)`，例如
  page size 16 时为 2、3。

FlashInfer 的 `plan` 需要的 `paged_kv_indptr` 是 page 级别的前缀和，而不是 token 级别
的前缀和。以两个请求长度 17、33、page size 16 为例：

```text
page_counts       = [2, 3]
paged_kv_indptr   = [0, 2, 5]
last_page_len     = [1, 1]
seq_lens          = [17, 33]
```

page table 仍然保存 token 级物理位置，因此每个请求需要按 `page_size` 步长抽取首个
token 对应的 page，并除以 page size 得到 FlashInfer 的 page id。最后一个 page 的有效
token 数由 `last_page_len` 单独传入；满 page 时该值等于 page size。

## 实现内容

- `python/minisgl/attention/fi.py`
  - 新增 `_paged_lengths`，统一计算 page 数和最后一页有效长度。
  - `FIMetadata.page_size` 从仅允许字面量 1 改为正整数。
  - 直接把 `[num_pages, page_size, heads, dim]` 的 KV cache 传给 FlashInfer，删除旧的
    flatten workaround。
  - `cu_seqlens_k_cpu` 改为 page 级前缀和；无 cache 命中的 prefill 保留 token 级
    `cu_seqlens_q_cpu`，避免把 query 长度误当成 page 长度。
  - page table 按 page 步长生成索引，正确填充 `last_page_len_cpu`。
  - CUDA Graph capture 的 page table 容量按 `ceil(max_seq_len/page_size)` 分配。
- `python/minisgl/engine/engine.py`
  - 移除 FP8 KV cache 对 `page_size=1` 的硬限制。
  - 增加 `--page-size` 必须为正数的配置校验；FP8 仍要求 FlashInfer attention backend。
- `tests/core/test_kv_cache_dtype.py`
  - 增加边界长度测试，覆盖 1、满 page、跨 page 和空余尾页。

## 正确性与服务验证

- `tests/core`：102 passed。
- `compileall` 和 `git diff --check`：通过。
- FlashInfer 合成 prefill：page size 16，长度 17/33，cache shape 为
  `[5, 16, 2, 64]`，输出 shape `[50, 4, 64]`，结果全部 finite。
- Qwen3-0.6B BF16：page size 1/16/32 均完成 64、256、384 token 请求；page size 32
  还完成了 `cuda-graph-max-bs=1` 的 capture/replay smoke。
- Qwen3-0.6B FP8 KV：page size 16 完成 64、256、384 token 请求，HTTP 请求全部成功。

这些 smoke 验证了 metadata、非满 page、BF16/FP8 和最小 Graph 路径，但不等价于完整
长上下文质量评测。

## L20 Qwen3-32B FP8 TP4 对照

环境：L20 GPU 2/3/6/7，FP8 W8A8 权重，FP8 E4M3 KV，`first_batch` calibration，
FlashInfer，CUDA Graph disabled，`memory_ratio=0.8`，并发 1，输入 4096，输出 32，
每组 2 次。单组重复次数较少，下面的数据只能作为候选筛选依据。

| page size | 吞吐 tok/s | 平均 TPOT ms | P90 TPOT ms | 平均 TTFT ms | 峰值显存 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.3405 | 47.73 | 52.98 | 1114.49 | 36625 |
| 16 | 11.0389 | 56.46 | 64.07 | 1148.83 | 36605 |
| 32 | 12.6121 | 46.43 | 51.29 | 1098.27 | 36605 |

相对 page size 1，page size 32 在本次画像上吞吐提升约 2.20%，P90 TPOT 降低约 3.20%，
page size 16 吞吐下降约 10.53%，P90 TPOT 上升约 20.94%。显存差异约 20 MiB，说明本次
收益主要来自元数据/访问行为，而不是显著的 KV cache 容量变化。

原始结果：

- [page size 1](../results/065_page1_4096.md)
- [page size 16](../results/065_page16_4096.md)
- [page size 32](../results/065_page32_4096.md)

## 结论与后续

1. page size 大于 1 的 FlashInfer metadata、BF16/FP8 服务和最小 CUDA Graph 路径已经
   打通，实验的实现 gate 通过。
2. page size 32 是当前 4096 输入、并发 1 的候选值，但证据只有两次重复，不能直接改成
   全局默认。
3. page size 16 在同一画像下明显回退，暂不接受；这说明 page size 不能只凭“减少索引
   数量”推断收益，必须结合 FlashInfer kernel 的 tile、occupancy 和请求长度分布。
4. 默认值继续保持 1。下一轮应补充 C=1/8/32、8K/16K 输入、open-loop 负载、page
   eviction/retraction telemetry，以及 page size 32 的真实 Graph 形状矩阵，再决定是否
   进入主线默认配置。

