# 实验 076：Decode 容量等待轮数 telemetry

## 1. 动机

实验 075 解决了动态请求集合下的 cursor 漂移，但仍缺少一个可量化的公平性指标：一个 UID
被容量裁剪延迟了多少轮？仅记录 selected batch 无法区分偶发跳过和长期 starvation。

## 2. 实现

Scheduler 增加两个内部状态：

- `_decode_capacity_wait_rounds`：当前仍未被选中的 UID 到目前为止连续等待的轮数；
- `_last_decode_capacity_wait_rounds`：本轮被选中的 UID 在服务前累计等待轮数。

当 `can_allocate_paged` 使 decode batch 被裁剪时：

1. 所有候选 UID 的等待轮数加一；
2. 被选中的 UID 写入 `selected_wait_rounds` 并从 deferred 状态移除；
3. 未选中的 UID 保留在等待表中；
4. `decode_capacity_trimmed` 事件记录 `deferred_uids` 和 `selected_wait_rounds`。

如果一个请求都无法容纳，则记录 `decode_capacity_deferred`，保留每个 UID 的等待轮数。

## 3. 确定性结果

容量每轮只能容纳 1 个请求，UID 为 0/1/2，连续三轮结果为：

```text
round 1: selected=[0], deferred_wait={1:1, 2:1}, selected_wait={0:1}
round 2: selected=[1], deferred_wait={0:1, 2:2}, selected_wait={1:2}
round 3: selected=[2], deferred_wait={0:2, 1:1}, selected_wait={2:3}
```

这说明等待时间是可观测且单调的：UID 2 连续等待两轮后在第三轮获得服务，服务事件携带
`selected_wait_rounds=3`。请求被移除并加入新 UID 后，等待表只保留当前候选集合，不会把旧
UID 的计数泄漏给新请求。

## 4. 验证与结论

新增等待轮数单测后，`tests/core/test_cache_allocate.py` 通过 14/14；完整核心回归结果为
120/120，compileall 与 diff 检查通过。

实验 076 通过 telemetry/correctness gate。该改动不改变 admission 结果，只增加可观测状态，
为下一阶段在线公平性和 P99 TTFT 对照提供数据基础。当前仍没有线上吞吐收益结论；后续应在
真实裁剪事件中检查最大等待轮数和 UID 完成时间分布。

