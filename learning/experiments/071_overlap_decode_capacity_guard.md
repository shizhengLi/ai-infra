# 实验 071：Overlap decode batch 容量保护

## 1. 背景与问题

实验 070 在开启 overlap pipeline、增加并发和限制 KV cache 页数后，出现了如下 allocator
断言：

```text
Cannot satisfy 1/2/4 cache tokens with only 0 evictable
```

这不是普通的显存不足，而是调度器在一个 GPU batch 尚未完成、其 KV 写入结果尚未完全回收时，
又选择了下一批 decode 请求。旧路径直接把整个 decode batch 交给 `allocate_paged`，分配器只能在
当前可见的 free slots 和 evictable radix pages 中找空间；当 batch 的未来 token 总量超过这个
瞬时容量时，内部 assertion 会终止 scheduler。

本实验的目标是先恢复安全不变量：**送入 paged allocator 的 decode batch，必须能在当前可见容量内
完成下一步 KV 分配**。这一步是正确性修复，不宣称已经提升吞吐。

## 2. 原理

对每个 decode 请求，下一步需要新增的 page 数为：

```text
needed_pages(req) = max(0,
    ceil(device_len / page_size) - ceil(cached_len / page_size))
```

整个候选 batch 的需求是所有请求之和。当前可用容量由两部分组成：

```text
available_size = free_slots * page_size + evictable_radix_tokens
```

`CacheManager.can_allocate_paged` 只读这两个权威状态，不提前修改 cache。scheduler 在 decode
batch 进入 `_prepare_batch` 之前做一次 admission check：

1. 整个 batch 能容纳时保持原 batch，避免正常路径额外分支开销。
2. 整体超限时按原请求顺序逐个试加，只保留仍能容纳的前缀。
3. 一个请求都不能容纳时返回 `None`，下一轮 scheduler 再等待已提交 batch 释放可见容量。

这是一个保守的安全阀。它没有把 in-flight CUDA 写入量伪装成已释放空间，也没有做复杂的全局
最优装箱；代价是超限时可能出现 batch 变小和请求顺序偏置，后续可再做 round-robin 或老化策略。

## 3. 实现变更

- `CacheManager.can_allocate_paged(reqs)`：根据 `cached_len/device_len` 计算 page demand，并与
  `available_size` 比较。
- `Scheduler._fit_decode_batch(batch)`：在 decode batch 准备和 telemetry 记录前进行容量裁剪。
- telemetry 的 `selected` 事件现在记录实际裁剪后的 batch，避免“记录为 8、实际执行为 3”这类
  观测不一致。
- 新增两个 focused tests：可见容量判断，以及 scheduler 按容量裁剪 decode batch。

## 4. 验证过程

### 4.1 离线测试

执行：

```bash
.venv/bin/pytest -o addopts='' -q \
  tests/core/test_cache_allocate.py \
  tests/core/test_safe_retraction.py \
  tests/core/test_prefill_reserve.py
```

结果：`24 passed`。完整 `tests/core` 也在最终提交前重新执行。

### 4.2 在线功能冒烟

Qwen3-32B FP8 TP4 的重新启动耗时过长，无法在本轮窗口内形成公平 benchmark，因此没有把它
当成有效性能样本。为了验证真实 API、scheduler、CUDA Graph 和 cache 生命周期，使用本地已缓存
的 Qwen3-4B-AWQ 单卡 L20 运行：

- GPU：L20，单卡 GPU 2；TP=1；AWQ；FP16 runtime。
- `num_pages=128`，`page_size=1`，`max_seq_len_override=2048`。
- `prefill-reserve-policy=page-aware`，safe retraction 开启，cooldown=4，pending gate=64。
- 6 个请求，输入长度 `[8, 16, 24]` 的平衡混合，输出 4 token，Poisson 到达率 2 req/s。

原始结果保存在 `learning/results/071_awq_guard_smoke.md` 和对应 JSON 中。6/6 请求完成，
TTFT/TPOT SLO 均为 100%；输出吞吐为 6.99 token/s，P90 TTFT 为 596.91 ms，P99 TPOT 为
5.41 ms。safe-retraction telemetry 记录了 2 次 retraction、8 个 retracted token，服务端没有
再出现 allocator assertion，进程也能优雅退出。

这次冒烟的结论是“路径可运行且没有复现崩溃”，不是“071 带来 6.99 token/s 性能提升”。由于
这个 workload 的 decode batch 实际多为单请求，未声称已经覆盖实验 070 的高压 overlap 边界。

## 5. 结论与下一步

实验 071 通过 correctness gate：新增容量保护、单测和真实在线 smoke 均通过，默认行为只在
batch 超出当前可见容量时介入。当前实现保留为安全修复，暂不打开新的全局性能默认值。

下一步应在可复现的 4 卡 TP4 限页环境中增加明确的 overlap pressure trace，要求同时记录：

- 被裁剪前后的 `requested_batch_size/selected_batch_size`；
- allocator assertion、请求错误数和完成率；
- retraction 次数、pending tokens、P90/P99 TTFT 与 TPOT；
- 不同 `page_size` 和 output length 下的 starvation 情况。

只有在高压 trace 不崩溃、无请求丢失，并且至少三次重复没有明显 tail regression 后，才考虑把
裁剪策略扩展成更公平的 admission policy。

