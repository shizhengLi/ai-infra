# Experiment 033: direct PyNCCL validation at TP8 and with CUDA Graphs

## Status

Complete. The direct PyNCCL path generalizes to TP8 for the graph-disabled serving profile, but the
TP4 CUDA Graph result has tail-latency regressions. Keep direct communication as an explicit,
graph-disabled L20 deployment profile; do not make it an unconditional default.

## Objective

Experiment 032 found a 9.5-11.8% TP4 throughput improvement from
`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`, but only measured TP4 with CUDA Graphs disabled. This experiment
checks two risks:

1. whether the improvement survives TP8;
2. whether direct communication changes CUDA Graph replay behavior or tail latency.

## Reproducibility and toolchain

The workload was Qwen3-32B BF16 with input length 256, output length 32, seed `3100042`,
concurrency `[1, 8, 32]`, three repetitions, `fi`, memory ratio 0.7, maximum prefill 2048,
prefill streak 1, and decode-active prefill budget 1024. The online TP8 comparison used GPUs 0-7
and CUDA Graphs disabled. The TP4 Graph comparison used GPUs 0-3 and graph sizes through batch 64.

The host has multiple CUDA installations: `/usr/bin/nvcc` is CUDA 11.5 while PyTorch and the
driver path use CUDA 12.8. The successful runs explicitly used:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Without this environment, PyNCCL JIT compilation either rejects `c++20` or cannot target
`compute_89`. This is a reproducibility prerequisite, not a runtime optimization.

## TP8 online comparison, CUDA Graphs disabled

| Concurrency | Symmetric tok/s | Direct tok/s | Throughput change | Avg TPOT change | P90 TPOT change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.70 | 32.72 | +10.20% | -8.00% | -8.35% |
| 8 | 167.90 | 177.17 | +5.52% | -6.60% | -5.46% |
| 32 | 172.80 | 181.89 | +5.26% | -5.81% | -14.52% |

The direct path improves every TP8 load. GPU memory remained effectively unchanged (32,053 MiB
versus 32,047 MiB). This confirms that the optimization is not limited to TP4, although the gain is
smaller than TP4's 9.5-11.8% because TP8's collective topology has a lower copy overhead.

Artifacts:

- `learning/results/033_tp8_online_symmetric_seed3100042.json`
- `learning/results/033_tp8_online_direct_seed3100042.json`
- `learning/experiments/033_tp8_online_symmetric_seed3100042_raw.md`
- `learning/experiments/033_tp8_online_direct_seed3100042_raw.md`

## TP4 online comparison, CUDA Graphs enabled

Both servers captured graph sizes `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]` successfully.

| Concurrency | Symmetric tok/s | Direct tok/s | Throughput change | Avg TPOT change | P90 TPOT change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.98 | 35.79 | -5.78% | -0.19% | +0.01% |
| 8 | 166.69 | 179.43 | +7.64% | -2.11% | +4.49% |
| 32 | 267.87 | 303.65 | +13.36% | -8.03% | +9.69% |

The direct path is faster at C=8/32, but P90 TPOT regresses at both loads and C=1 has a large
first-repeat TTFT outlier. Therefore the strict no-regression gate is not met for Graph mode. The
result is useful: direct communication and Graph replay are compatible, but their scheduling/tail
interaction needs a separate investigation before automatic selection.

Artifacts:

- `learning/results/033_tp4_online_symmetric_graph_seed3100042.json`
- `learning/results/033_tp4_online_direct_graph_seed3100042.json`
- `learning/experiments/033_tp4_online_symmetric_graph_seed3100042_raw.md`
- `learning/experiments/033_tp4_online_direct_graph_seed3100042_raw.md`

## Decision

- Accept `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0` as an explicit L20 TP4/TP8 profile when CUDA Graphs are
  disabled and low tail latency is not being traded against peak throughput.
- Keep the generic default and Graph-mode default unchanged. The TP4 Graph result does not satisfy
  the pre-registered P90 TPOT rule.
- Do not start asynchronous or batched collective work yet: direct-buffer elimination already
  provides a measurable baseline, and the remaining issue is Graph-mode tail behavior rather than
  raw collective bandwidth.

Next experiment 034 will isolate Graph-mode tail variance with longer repeats and separate C=1
warmup from steady-state measurements before considering a graph-aware automatic policy.
