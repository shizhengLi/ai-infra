# Experiment 003: CUDA Graph value on L20

## Hypothesis

CUDA Graph replay materially improves Qwen3-0.6B decode throughput on L20 by removing per-layer
Python/CUDA kernel launch overhead, especially at a fixed decode batch size.

## Principle

Autoregressive decode launches many short kernels once per generated token. For a small model, host
launch latency can be a large fraction of token time. CUDA Graph captures the fixed-address decode
execution once and replays it with one host submission. Disabling graph capture removes startup
cost and graph memory, but exposes eager launch overhead.

## Controlled variables

Same as experiment 001 post-change. The only changed argument is `--cuda-graph-max-bs 0`.

## Command

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 proxychains .venv/bin/python benchmark/offline/bench_l20.py \
  --model Qwen/Qwen3-0.6B \
  --num-prompts 128 --input-len 256 --output-len 128 --repeats 3 \
  --attention-backend fi --cuda-graph-max-bs 0 --max-running-requests 128 \
  --max-seq-len 2048 --memory-ratio 0.8 \
  --markdown-out learning/experiments/003_graph_disabled_raw.md \
  --json-out learning/results/003_graph_disabled.json
```

The graph-enabled comparison is `learning/experiments/001_optimized_raw.md`.

## Results

| Metric | Graph enabled | Graph disabled | Graph effect |
| --- | ---: | ---: | ---: |
| Engine initialization | 7.3553 s | 6.7014 s | +0.6538 s |
| Median throughput | 10145.02 token/s | 7746.01 token/s | +30.97% |
| Mean throughput | 10142.23 token/s | 7742.62 token/s | +30.99% |
| Throughput population stddev | 11.44 token/s | 10.36 token/s | both stable |

Graph-disabled runs were 7728.59, 7746.01, and 7753.27 token/s. Graph-enabled runs were 10127.02,
10145.02, and 10154.64 token/s.

Raw report: `learning/experiments/003_graph_disabled_raw.md`; machine-readable result:
`learning/results/003_graph_disabled.json`.

## Interpretation

At this fixed batch and small-model shape, CUDA Graph removes enough host launch overhead to improve
output throughput by about 31%. The extra startup cost is recovered after roughly 21.4k output
tokens: about 2.1 seconds with graphs versus 2.8 seconds in eager mode. This break-even estimate is
workload-specific and does not include request arrival gaps or a larger model's longer kernels.

## Decision

Keep CUDA Graph enabled for serving and offline batch profiles. Preserve `--cuda-graph-max-bs 0` for
cold-start-sensitive diagnostics and as an eager-mode control. Cap capture at reachable concurrency
as implemented in experiment 001.

## Next step

Measure graph maximum/concurrency under an online arrival process, where batch size changes over
time, then repair FlashAttention compatibility in its own environment experiment.
