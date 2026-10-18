# 实验 074：混合 page demand 下的容量裁剪公平性

## 1. 问题

实验 073 只验证了每个请求都需要一个 slot 时的 round-robin。真实 decode 请求的下一步需求
可能不同：有的请求刚好跨 page 边界，需要 2 个 page；另一些请求只需 1 个 page。如果仍然
使用简单前缀或固定顺序，重请求可能长期被跳过，轻请求则反复占满剩余容量。

## 2. 确定性模型

构造三个请求：

```text
UID 0: demand = 2 pages
UID 1: demand = 1 page
UID 2: demand = 1 page
每轮容量上限 = 3 pages
```

`can_allocate_paged` 使用 demand 总和模拟真实 cache admission，连续调用
`Scheduler._fit_decode_batch`，不修改 cache，也不引入 GPU 或模型执行噪声。

## 3. 结果

三轮选择为：

```text
round 1: [0, 1]  # 2 + 1 = 3
round 2: [2, 0]  # 1 + 2 = 3
round 3: [1, 2]  # 1 + 1 = 2
```

继续运行六轮得到：

```text
[[0, 1], [2, 0], [1, 2], [0, 1], [2, 0], [1, 2]]
```

因此 UID 0、1、2 在相同周期内都获得服务；重请求不会因为需要更多 page 而永久饥饿，轻请求
也不会永远占据容量。算法仍是贪心装箱，不保证任意 demand 分布下的最大吞吐。

## 4. 验证

新增混合 page demand 单测后，`tests/core/test_cache_allocate.py` 通过 12/12；实验 073 已有
的完整核心回归保持通过。代码路径只使用 `can_allocate_paged` 的只读判断，未在 admission 检查
阶段分配或释放 page。

## 5. 结论

实验 074 通过确定性 fairness/correctness gate，确认 round-robin 对异构 page demand 仍能避免
固定 UID 饥饿。尚未宣称线上吞吐或 tail latency 收益；下一步在线实验应采集每个 UID 的等待
轮数、实际 demand 和完成时间，并验证请求集合动态变化时 cursor 不会失效。

