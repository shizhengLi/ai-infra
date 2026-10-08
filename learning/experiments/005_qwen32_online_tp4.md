# Experiment 005: Qwen3-32B TP=4 online baseline

## Hypothesis

On four PCIe L20 GPUs, increasing concurrent requests should improve output throughput until model
compute, memory bandwidth, or PyNCCL collectives saturate. TTFT and per-request E2E latency will rise
with batch size, so the useful operating point is not necessarily maximum throughput.

## Principle

Qwen3-32B BF16 requires tensor parallelism on L20. Every transformer layer introduces TP
collectives, while decode repeatedly reads weights and KV cache. Concurrency amortizes kernel launch
and weight access but increases queueing and batch latency. CUDA Graph removes launch overhead for
decode batches up to its configured maximum.

The test launches each concurrency group as a closed batch. TTFT is measured to the first streamed
token. TPOT excludes both TTFT and the final OpenAI finish event. Unique random prompts prevent
Radix Cache hits from turning later repetitions into a different workload.

## Controlled variables

- Model: Qwen3-32B BF16, local ModelScope snapshot
- Hardware: GPUs 0-3, four NVIDIA L20 cards
- Tensor parallelism: 4, default PyNCCL
- Attention backend: automatic (`fi` on SM89)
- Memory ratio: 0.8
- CUDA Graph maximum batch size: 64
- Maximum running requests: 128
- Prompt content: 1024 tokens plus chat-template tokens
- Output: 256 tokens, EOS ignored, greedy/top-k 1
- Concurrency: 1, 8, 32, 64
- Repeats: 3 with distinct deterministic seeds

## Commands

Server:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m minisgl \
  --model /data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master \
  --tp 4 --memory-ratio 0.8 --cuda-graph-max-bs 64 \
  --max-running-requests 128 --port 1919
```

Client:

```bash
python benchmark/online/bench_l20.py \
  --base-url http://127.0.0.1:1919/v1 \
  --input-len 1024 --output-len 256 \
  --concurrency 1 8 32 64 --repeats 3 \
  --gpu-indices 0,1,2,3 \
  --server-tp 4 --server-memory-ratio 0.8 --server-graph-max-bs 64 \
  --markdown-out learning/experiments/005_qwen32_tp4_raw.md \
  --json-out learning/results/005_qwen32_tp4.json
```

## Results

The server loaded 17 weight shards in about 18 seconds, allocated a 18.62 GiB KV cache per rank,
captured 11 CUDA Graph sizes up to batch 64 in about 16 seconds, and retained about 8.25 GiB free
memory per GPU after graph capture.

| Concurrency | Output token/s | Stddev | P90 TTFT | P90 TPOT | P90 E2E | Avg GPU util | Peak GPU memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.08 | 0.72 | 438.90 ms | 24.84 ms | 6.725 s | 98.88% | 37059 MiB |
| 8 | 212.16 | 0.09 | 2610.39 ms | 27.88 ms | 9.652 s | 98.46% | 37715 MiB |
| 32 | 415.30 | 0.25 | 10346.97 ms | 37.13 ms | 19.708 s | 96.88% | 38117 MiB |
| 64 | 503.35 | 0.33 | 20701.20 ms | 46.95 ms | 32.512 s | 98.11% | 38117 MiB |

Throughput gain was about 5.57x from concurrency 1 to 8, 1.96x from 8 to 32, and only 1.21x from
32 to 64. The first concurrency-1 run measured 37.07 token/s and 618 ms TTFT; the next two stabilized
at 38.59 token/s and about 349 ms TTFT, indicating one shape-specific warm-up effect.

Raw aggregate report: `learning/experiments/005_qwen32_tp4_raw.md`. Machine-readable per-repeat
result: `learning/results/005_qwen32_tp4.json`.

## Interpretation

All concurrency levels kept the four GPUs at roughly 97-99% average utilization, so the server was
not idle. Concurrency 64 maximized throughput, but it added only 21% over concurrency 32 while
doubling P90 TTFT and raising P90 TPOT by 26%. Concurrency 8 is the best measured point for an
interactive SLO near 3 seconds TTFT; concurrency 32 is the stronger throughput/latency compromise
when 10-second TTFT is acceptable.

The default prefill budget is 8192 tokens. Eight 1024-token message bodies plus chat-template tokens
already exceed it, and concurrency 32/64 must enter prefill in several waves. This explains the
near-stepwise TTFT growth. At concurrency 32 and 64, mean TPOT (about 49.7 and 79.5 ms) was also
higher than P99 TPOT (about 40 and 52 ms), proving that fewer than 1% of decode intervals contained
large stalls. The initial harness did not retain P99.9/max or raw timestamps, so their exact size is
not reconstructed after the run; the harness now records these fields for the next experiment.

## Decision

Accept this as the TP=4 baseline. Do not optimize PyNCCL yet: GPU utilization is already high, while
the latency shape directly identifies prefill chunking as the next controlled variable.

## Next step

Experiment 006 will compare prefill budgets 8192 and 32768 at concurrency 32, measuring whether a
larger prefill batch reduces TTFT and decode stalls without unacceptable memory or throughput cost.
