# Experiment 032: TP collective attribution and overlap baseline

## Status

Complete. The profile identified a material TP collective cost and validated a direct PyNCCL
communication path as an L20 TP=4 deployment optimization. The generic default remains unchanged;
the optimized path is enabled explicitly with `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`.

## Objective

The L20 TP=4/8 results show diminishing scaling, but the existing benchmark does not attribute the
cost to individual NCCL collectives. Before changing communication semantics, measure how much of a
forward pass is spent in all-reduce/all-gather, where those collectives sit on the engine stream,
and whether the cost changes with decode batch size and TP topology.

## Hypothesis

Small decode batches are dominated by collective latency, while larger prefill batches amortize
PCIe transfer cost. The current row-parallel and output-projection all-reduces are serialized on the
model stream; if profiling confirms a material communication fraction, an asynchronous communication
stream or batched collective design becomes a justified optimization target.

## Implementation

- Added NVTX ranges around TorchDistributed and PyNCCL all-reduce/all-gather calls in
  `python/minisgl/distributed/impl.py`. Labels include operation and message bytes, so Nsight can
  attribute communication without changing the CPU-only test path.
- Added `benchmark/online/bench_tp_collective_profile.py` for repeatable TP=4/8 PyNCCL
  microbenchmarks. It sets `LOCAL_RANK` after process-group initialization to avoid all workers
  accidentally using GPU 0 under `torchrun`.
- The existing PyNCCL implementation already has a direct path when the symmetric scratch-buffer
  cap is zero. This experiment measures that path rather than adding a second communication
  implementation.

## Stage A: collective microbenchmark

Qwen3-32B BF16 shape (`hidden_size=5120`), 20 warmup and 100 measured all-reduces per point:

| TP | Mode | 5 MiB | 20 MiB | 80 MiB |
| --- | --- | ---: | ---: | ---: |
| 4 | symmetric buffer | 0.636 ms / 7.68 GiB/s | 2.531 ms / 7.72 GiB/s | 10.534 ms / 7.42 GiB/s |
| 4 | direct (`cap=0`) | 0.398 ms / 12.26 GiB/s | 1.608 ms / 12.15 GiB/s | 6.418 ms / 12.17 GiB/s |
| 8 | symmetric buffer | 0.681 ms / 7.17 GiB/s | 2.625 ms / 7.44 GiB/s | 10.707 ms / 7.30 GiB/s |
| 8 | direct (`cap=0`) | 0.677 ms / 7.21 GiB/s | 2.586 ms / 7.55 GiB/s | 10.166 ms / 7.68 GiB/s |

The direct path is consistently faster for TP=4 (about 37% lower collective latency), while TP=8
is effectively neutral to slightly better. This points to a TP=4 local-copy overhead rather than a
universal improvement across the topology.

## Stage A: Nsight attribution

The TP=4 Qwen3-32B server was traced with CUDA Graphs disabled and calibrated prefill/decode
budgets. NVTX ranges were visible in the trace. Cumulative `NCCL:ncclAllReduce` kernel time was
19.896 seconds versus 47.530 seconds for all GPU kernels, approximately 41.9% of cumulative kernel
time. All-gather kernel time was only about 0.085 seconds. This clears the pre-registered 10%
communication gate and makes communication a credible optimization target.

Artifacts: `learning/results/032_tp4_server.nsys-rep`, `032_tp4_server.sqlite`, and
`032_tp4_server_stats.txt`.

## Stage B: paired online validation

Both modes used Qwen3-32B BF16, TP=4 on GPUs 0-3, CUDA Graphs disabled, `max_seq_len=8192`,
prefill budget 2048, active decode-prefill budget 1024, input length 256, output length 32,
concurrency `[1, 8, 32]`, seed `3100042`, and three repetitions. The control used the default
symmetric buffer; treatment used `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`.

| Concurrency | Throughput change | Avg TTFT change | Avg TPOT change | P90 TPOT change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +9.54% | -37.53% | -3.81% | -4.57% |
| 8 | +11.82% | -17.63% | -7.00% | -5.05% |
| 32 | +11.34% | -11.28% | -8.00% | -22.41% |

Absolute aggregate throughput was 31.50 -> 34.50, 158.22 -> 176.92, and 158.61 -> 176.59
tokens/s for concurrency 1/8/32. Mean GPU memory was unchanged within measurement noise
(32,193 MiB control versus 32,185 MiB treatment).

Raw and structured results:

- `learning/results/032_tp4_online_symmetric_seed3100042.json`
- `learning/results/032_tp4_online_direct_seed3100042.json`
- `learning/experiments/032_tp4_online_symmetric_seed3100042_raw.md`
- `learning/experiments/032_tp4_online_direct_seed3100042_raw.md`

## Stage B gate

The Stage A attribution gate passed, but the measured win came from removing the PyNCCL symmetric
buffer copy path; an asynchronous stream or batched collective is not justified yet. The direct
path passes the Stage B rule for TP=4: throughput improves by at least 9.5% at every tested load,
and P90 TPOT improves at every load with no measured memory increase. Keep it as an explicit L20
TP=4 deployment profile:

```bash
MINISGL_PYNCCL_MAX_BUFFER_SIZE=0 \
  python -m minisgl --model Qwen/Qwen3-32B --tp 4 ...
```

Experiment 033 completed the TP=8 and Graph checks. TP8 remained positive with Graphs disabled, but
TP4 Graph P90 TPOT regressed. Keep the generic default unchanged and continue with the Graph-tail
variance study in experiment 034.
