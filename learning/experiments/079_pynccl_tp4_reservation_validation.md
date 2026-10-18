# 实验 079：PyNCCL 修复与 TP4 页预留验证

## 1. 实验目标

本实验处理两个实际问题：

1. TP4 启动时不能再因为系统没有 `libnccl.so` unversioned symlink 而退回 Gloo；
2. 将实验 078 的 `token/page-aware` reservation 对比放到 Qwen3-32B-AWQ、4 张 L20 的
   PyNCCL TP4 路径，观察策略差异是否仍然存在。

## 2. PyNCCL 修复与通信边界

环境中实际存在：

```text
.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
```

但系统 linker cache 没有 NCCL，原始 `-lnccl` 会寻找不存在的 `libnccl.so`。当前
`python/minisgl/kernel/pynccl.py` 会从 `NCCL_HOME/NCCL_ROOT`、Python NCCL wheel 和
`LD_LIBRARY_PATH` 查找版本化库，找到后传入绝对路径并加入 rpath：

```text
/.../nvidia/nccl/lib/libnccl.so.2
-Wl,-rpath,/.../nvidia/nccl/lib
```

AOT 编译验证：

```text
_find_nccl_library() -> .../nvidia/nccl/lib/libnccl.so.2
_load_nccl_module() -> ffi.Module(imports_=())
```

TP4 服务使用默认 `use_pynccl=True` 启动成功，4 个 rank 加载 Qwen3-32B-AWQ、完成 CUDA
Graph capture 并进入 API ready。启动日志中的 Gloo 只用于 CPU 控制面：rank 初始化以及把
NCCL unique ID 广播给各进程；GPU 上的 tensor-parallel all-reduce/all-gather 由 PyNCCL
调用 NCCL 完成。关闭 PyNCCL 的 `--disable-pynccl` 只是 fallback，不作为本实验路径。

## 3. TP4 对比配置

- GPU：L20 2/3/6/7；TP=4；模型 Qwen3-32B-AWQ；runtime `float16`；
- `page_size=16`，`num_pages=8`，每 rank 的 KV 容量 128 token；
- `max_seq_len_override=128`，prefill budget 64，decode-active budget 32，最大 prefill streak 1；
- CUDA Graph `max_bs=8`；
- 负载：8 请求，输入 17 token，输出 16 token，arrival rate 20 req/s；seed 7900072；
- `token` 和 `page-aware` 各运行 3 次，TTFT/TPOT SLO 均为 120 s。

## 4. 结果

| reservation | 吞吐 tok/s | 标准差 | P90 TTFT ms | P99 TPOT ms | P90 E2E s | 完成率 | capacity 事件 |
|---|---:|---:|---:|---:|---:|---:|---:|
| token | 136.63 | 3.90 | 511.49 | 35.01 | 0.677 | 24/24 | 0 |
| page-aware | 135.50 | 1.90 | 519.38 | 35.53 | 0.685 | 24/24 | 0 |

`page-aware` 相对 `token` 吞吐下降约 0.83%，P90 TTFT 上升约 1.54%，均在本次单一 workload
的重复波动范围内。两组均无 allocator assertion、请求丢失或进程异常。

原始结果：

- `learning/results/079_tp4_token_8page.md`；
- `learning/results/079_tp4_page_aware_8page.md`；
- `learning/results/079_tp4_token_telemetry.jsonl`；
- `learning/results/079_tp4_token_pressure_8p_telemetry.jsonl`；
- `learning/results/079_tp4_page_aware_pressure_8p_telemetry.jsonl`。

## 5. Telemetry 与解释

TP4/8-page 两组 trace 都只有 `selected`、forward 和 result 事件，没有
`decode_capacity_trimmed/deferred`。这不是 PyNCCL 失效，而是 admission 在 decode guard
之前已经把活跃请求集合限制到可见容量内；因此 078 单卡页边界中观察到的 16 次/轮裁剪，在
这个 TP4 trace 中没有出现。

这次结果把两个层次分开：PyNCCL 解决的是跨 GPU 通信初始化和执行，reservation policy 解决
的是 KV admission；前者成功不能推导后者一定产生收益。

## 6. 决策与面试表达

PyNCCL wheel-linking 修复接受：不需要额外安装系统 NCCL development package，TP4 默认路径
即可启动。Gloo 保留为 CPU bootstrap，不把它误称为 GPU 通信后端。

`page-aware` 在本 TP4/8-page 负载没有统计显著收益，因此不改默认策略；继续保留为单卡或
更激进页边界压力下的 opt-in。面试中可以说明：先用绝对库路径和 rpath 修复动态链接，再
用真实 TP4 端到端验证；同时从 telemetry 证明 TP4 的 admission 先行，避免把没有触发 guard
解释成优化成功。
