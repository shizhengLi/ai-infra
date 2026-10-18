# 实验 073：Decode 容量裁剪的公平性

## 1. 动机

实验 072 的 TP4 限页压力没有触发容量裁剪，因为 prefill admission 和 safe retraction 已经把
decode batch 限制在很小的规模。但 071 的实现仍有一个调度风险：当容量只能容纳部分请求时，
它按 `batch.reqs` 的固定顺序取前缀。`DecodeManager` 当前按 UID 排序，因此低 UID 请求可能
持续占据唯一可用的 decode slot，后面的请求会饥饿。

这类问题不能靠吞吐数字发现，必须用确定性选择序列验证。实验 073 的目标是：在保持容量安全
不变量的前提下，增加一个轻量 round-robin 起点，让受限 decode batch 在多轮调度中公平轮转。

## 2. 基线

使用三个 UID 为 0/1/2 的 decode 请求，并让 fake cache manager 每轮只能接受一个请求。原来的
前缀策略连续执行 9 轮，选择结果为：

```text
selection_order = [0, 0, 0, 0, 0, 0, 0, 0, 0]
selection_counts = {0: 9, 1: 0, 2: 0}
```

这不是理论上的可能性，而是当前 `_fit_decode_batch` 的实际运行结果。

## 3. 实现

`Scheduler` 增加 `_decode_capacity_cursor`。当完整 decode batch 无法容纳时：

1. 从 cursor 指向的请求开始环形遍历，而不是永远从 UID 0 开始。
2. 仍按原来的 `can_allocate_paged(candidate)` 检查逐个加入，容量不足的请求跳过。
3. 记录最后选中请求的后继位置，下一轮从该位置继续。

容量足够容纳完整 batch 时保持原路径，不改变正常 workload 的请求顺序和 batch shape。这个算法
仍然是贪心策略，不承诺不同请求 page demand 不同时的全局最优装箱。

## 4. 确定性结果

同一组 3 请求、单 slot、9 轮实验结果：

```text
selection_order = [0, 1, 2, 0, 1, 2, 0, 1, 2]
selection_counts = {0: 3, 1: 3, 2: 3}
```

另外使用真实 `CacheManager(num_pages=4, page_size=1)`，预先占用 3 page，只保留 1 page：

- 候选 decode 请求数：3；
- `can_allocate_paged` 可见容量：1 token/page；
- 实际裁剪结果：只选择 UID 0；
- cache 可见容量未被检查过程修改，仍为 1。

这证明 round-robin 只改变选择顺序，不绕过 cache admission，也不会在检查阶段提前分配 page。

## 5. 测试与结论

新增测试覆盖：

- 容量为 1 时的 3 请求轮转，确认无 starvation；
- 真实 `CacheManager` 剩余 1 page 时裁剪到单请求；
- telemetry 保留裁剪前 `requested_batch_size` 和裁剪后 `selected_batch_size`。

focused tests 为 34/34，完整 `tests/core` 为 117/117，compileall 和 diff 检查通过。

实验 073 通过 fairness/correctness gate。该改动是容量受限场景的调度质量修复，不会提升容量
本身，也没有修改默认 cache/page 参数。下一步应在 TP4 小 page 在线 trace 中打开可观测裁剪，
记录每个 UID 的等待轮数和 P99 TTFT，确认真实运行中的请求完成时间不会因轮转而恶化。

