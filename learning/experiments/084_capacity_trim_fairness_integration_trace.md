# 实验 084：确定性 capacity trim/fairness 集成 trace

## 1. 实验目的

实验 077、082、083 在真实 L20 HTTP 压力下都没有命中
`decode_capacity_trimmed` 或 `decode_capacity_deferred`。原因不是容量保护失效，而是
prefill admission 已经在 decode fit guard 之前把活跃请求集合限制到可容纳范围，导致 073-076
实现的公平 cursor 和等待轮数 telemetry 没有真实线上样本。

本实验不继续盲目压缩 GPU KV cache，而是先构造一个可重复的 scheduler 集成 trace，验证以下
问题：

1. 混合 page demand 下，多个候选请求能否真正进入 `_fit_decode_batch` 并产生 trim；
2. 所有候选都无法容纳时，是否产生 deferred 事件且不破坏 cursor；
3. 请求完成、取消和新 UID 加入后，等待状态是否只保留当前集合；
4. `selected_wait_rounds`、`deferred_uids`、`wait_rounds` 是否和实际选择一致。

这是一项 CPU-only 的 deterministic integration test，不宣称 GPU 吞吐收益，也不替代后续
L20 真实压力实验。

## 2. Trace 设计

使用一个容量检查器模拟 `CacheManager.can_allocate_paged`，只按候选请求的 page demand 求和：

```text
UID 0: demand = 2 pages
UID 1: demand = 1 page
UID 2: demand = 1 page
UID 3: demand = 2 pages
UID 4: demand = 1 page
UID 5: demand = 1 page
```

主阶段固定可见容量为 3 pages，候选集合为 `[0, 1, 2]`。完整 demand 为 4，因此每轮必然
进入 trim；随后进行动态集合阶段：

- UID 0 视为完成，UID 3 加入；
- UID 1 视为取消，UID 4 加入；
- 将可见容量暂时设为 0，制造一个全量 deferred round；
- 恢复容量并加入 UID 5，验证 cursor 和等待表恢复正常。

测试通过 `_fit_decode_batch` 和 `_record_pipeline_event`，使用真实 scheduler 的 cursor、等待
轮数和事件字段，不直接伪造事件结果。测试代码位于
`tests/core/test_cache_allocate.py` 的
`test_scheduler_capacity_trace_records_dynamic_fairness`。

## 3. 运行命令

```bash
PYTHONPATH=python .venv/bin/pytest -q tests/core/test_cache_allocate.py
PYTHONPATH=python .venv/bin/pytest -q tests/core
```

环境为仓库 `.venv`、Python 3.12、pytest；本实验不需要 CUDA、模型权重或 L20 空闲卡。

## 4. 结果

focused 测试为 **15/15 通过**，完整 `tests/core` 为 **121/121 通过**。

### 4.1 固定集合的 9 轮结果

容量为 3 pages，候选需求为 2/1/1：

| round | requested | selected | deferred | selected wait rounds |
| ---: | --- | --- | --- | --- |
| 1 | `[0,1,2]` | `[0,1]` | `[2]` | `{0:1, 1:1}` |
| 2 | `[0,1,2]` | `[2,0]` | `[1]` | `{0:1, 2:2}` |
| 3 | `[0,1,2]` | `[1,2]` | `[0]` | `{1:2, 2:1}` |
| 4 | `[0,1,2]` | `[0,1]` | `[2]` | `{0:2, 1:1}` |
| 5 | `[0,1,2]` | `[2,0]` | `[1]` | `{0:1, 2:2}` |
| 6 | `[0,1,2]` | `[1,2]` | `[0]` | `{1:2, 2:1}` |
| 7 | `[0,1,2]` | `[0,1]` | `[2]` | `{0:2, 1:1}` |
| 8 | `[0,1,2]` | `[2,0]` | `[1]` | `{0:1, 2:2}` |
| 9 | `[0,1,2]` | `[1,2]` | `[0]` | `{1:2, 2:1}` |

9 轮中 UID 0、1、2 各被选择 6 次，服务次数差异为 0。重需求 UID 0 不会因为需要
2 pages 而永久饥饿，轻需求 UID 1/2 也不会固定占据前缀位置。

### 4.2 动态集合与 deferred 结果

| round | requested | capacity | selected | event | deferred / wait |
| ---: | --- | ---: | --- | --- | --- |
| 10 | `[1,2,3]` | 3 | `[1,2]` | trimmed | UID 3；selected waits `{1:1,2:1}` |
| 11 | `[2,3,4]` | 3 | `[3,4]` | trimmed | UID 2；selected waits `{3:2,4:1}` |
| 12 | `[3]` | 0 | `[]` | deferred | UID 3；wait rounds `{3:1}` |
| 13 | `[3,4,5]` | 3 | `[3,4]` | trimmed | UID 5；selected waits `{3:2,4:1}` |

总计产生 13 个 pipeline capacity 事件：12 个 `decode_capacity_trimmed`、1 个
`decode_capacity_deferred`。容量为 0 的 round 没有修改 cursor；恢复后 UID 3/4 可以继续被
选择，新增 UID 5 只进入当前等待表，不继承已完成或取消请求的旧计数。

## 5. Gate 判定

| gate | 结果 | 说明 |
| --- | --- | --- |
| 至少 9 个 capacity trim round | 通过 | 固定阶段连续 9 轮 trim |
| 混合 demand 服务公平 | 通过 | UID 0/1/2 各 6 次，差异 0 |
| deferred telemetry 实际命中 | 通过 | 第 12 轮记录 `{3:1}` |
| wait rounds 单调且服务后更新 | 通过 | UID 3 在恢复轮携带等待值，未出现 stale UID |
| 完成/取消/新 UID 交错 | 通过 | 0 移除、1 移除、3/4/5 加入均覆盖 |
| allocator assertion / stale result | 通过 | 纯 scheduler 集成模型，无异常或 stale batch |
| GPU 吞吐收益 | 不适用 | 本实验没有运行模型和 GPU |

## 6. 结论

实验 084 通过 deterministic integration gate，证明 073-076 的公平 cursor 和 wait-round
telemetry 能在同一条 scheduler 选择链路中同时处理 trim、全量 deferred、请求集合变化和恢复。
这补上了此前只有单元测试、没有完整事件 trace 的证据缺口。

但这仍然是容量检查器驱动的 CPU trace，不能证明 L20 在线 workload 会自然进入该分支，也不能
证明 page-aware reservation 有吞吐收益。实验 085 继续使用真实 L20 pressure matrix，重点
寻找真实 trim-path 的边界；在此之前，page-aware 仍保持 opt-in，token reservation 仍为默认。

