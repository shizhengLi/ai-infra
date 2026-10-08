# Experiment 007: Qwen3-32B tensor-parallel scaling on L20

## Status

Complete.

## Hypothesis

Increasing tensor parallelism from 2 to 4 to 8 reduces the per-GPU matrix workload and weight
traffic, but every transformer layer requires collectives over PCIe. Throughput should improve
sublinearly, while small decode batches may eventually become communication-bound.

## Principle

Tensor parallelism shards linear layers across GPUs. Each rank stores fewer weights and performs
less GEMM work, then uses NCCL all-reduce/all-gather operations to combine partial results. These
L20 GPUs do not have NVLink, so collective latency and bandwidth use the PCIe topology. Scaling is
useful only when saved compute time exceeds the added communication and synchronization cost.

## Controlled variables

- Model: Qwen3-32B BF16, local ModelScope snapshot
- Hardware: NVIDIA L20, using GPUs 0 through TP-1
- Tensor parallelism: 2, 4, 8
- Communication: PyNCCL/NCCL
- Attention backend: automatic (`fi` on SM89)
- Memory ratio: 0.8
- CUDA Graph maximum batch size: 64
- Maximum running requests: 128
- Prefill budget: 8192 tokens
- Prompt content: 1024 tokens plus chat-template tokens
- Output: 256 tokens, EOS ignored, greedy/top-k 1
- Concurrency: 32
- Repeats: 3 with distinct deterministic seeds

## Procedure

Restart the server for each TP size. Run the same deterministic workload and compare output
throughput, TTFT/TPOT tails, E2E latency, per-GPU utilization, peak memory, and total estimated GPU
power. The TP=4 result is reused from experiment 006 because its server and client configuration is
identical.

Server, changing TP and `CUDA_VISIBLE_DEVICES` for each run:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_VISIBLE_DEVICES=GPU_LIST \
python -m minisgl \
  --model /data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master \
  --tp TP --memory-ratio 0.8 --cuda-graph-max-bs 64 \
  --max-running-requests 128 --max-prefill-length 8192 --port 1919
```

Client:

```bash
python benchmark/online/bench_l20.py \
  --base-url http://127.0.0.1:1919/v1 \
  --input-len 1024 --output-len 256 --concurrency 32 --repeats 3 \
  --gpu-indices GPU_LIST --server-tp TP --server-memory-ratio 0.8 \
  --server-graph-max-bs 64 --server-max-extend-tokens 8192
```

## Results

| Metric | TP=2 | TP=4 | TP=8 |
| --- | ---: | ---: | ---: |
| Output throughput | 159.09 token/s | 415.34 token/s | 507.19 token/s |
| Throughput stddev | 0.10 | 0.18 | 6.38 |
| Average TTFT | 15.934 s | 7.036 s | 5.664 s |
| P90 TTFT | 24.967 s | 10.345 s | 8.274 s |
| Average TPOT | 56.12 ms | 49.65 ms | 41.03 ms |
| P90 TPOT | 53.37 ms | 37.23 ms | 31.11 ms |
| P99.9 TPOT | 1820.77 ms | 4490.44 ms | 3628.40 ms |
| Worst observed TPOT | 4782.50 ms | 9637.84 ms | 7592.95 ms |
| P90 E2E | 38.449 s | 19.709 s | 16.140 s |
| Average GPU utilization | 98.82% | 96.91% | 96.09% |
| Peak memory per GPU | 37955 MiB | 37833 MiB | 37339 MiB |
| KV capacity per rank | 20005 tokens | 305066 tokens | 878675 tokens |
| Average power per GPU | 270.90 W | 227.39 W | 178.61 W |
| Estimated total GPU power | 541.80 W | 909.55 W | 1428.86 W |
| Output token/J | 0.2936 | 0.4566 | 0.3550 |

TP=4 delivered 2.61x the TP=2 throughput with twice the GPUs. TP=8 delivered 1.22x the TP=4
throughput with twice the GPUs, or about 61% incremental parallel efficiency. The TP=8 first repeat
measured 498.18 token/s; repeats two and three stabilized at 511.7 token/s, showing an additional
first-batch warm-up effect.

The PCIe topology has two four-GPU groups. GPUs 0-3 and GPUs 4-7 are connected as `PIX` inside
their respective NUMA nodes, while communication between the groups is `SYS` and crosses the CPU
interconnect. There is no NVLink.

Raw reports:

- TP=2: `learning/experiments/007_tp2_raw.md`
- TP=4 reused: `learning/experiments/006_prefill_8192_raw.md`
- TP=8: `learning/experiments/007_tp8_raw.md`

Machine-readable results are stored in `learning/results/007_tp2.json`,
`learning/results/006_prefill_8192.json`, and `learning/results/007_tp8.json`.

## Interpretation

The 2-to-4 GPU result is not a pure compute-scaling measurement. At TP=2, each rank stores roughly
twice the weight shard and a wider KV shard. Only 2.44 GiB remained for KV cache, enough for 20005
tokens. The 32 requests need more than 40000 prompt-plus-output tokens, so not all requests can be
resident and some wait for earlier requests to finish. This capacity effect explains both the
39-second worst TTFT and the apparently superlinear 2.61x throughput gain at TP=4.

TP=4 has enough KV capacity for this workload and is the efficiency optimum. It produces 55% more
tokens per joule than TP=2. TP=8 lowers P90 TTFT by 20% and P90 E2E by 18% relative to TP=4, but
throughput rises only 22% while estimated total GPU power rises 57%. Output token/J consequently
falls by 22%. The collective must also cross the `SYS` boundary between NUMA nodes, making PCIe and
host-interconnect communication a likely source of the diminishing return.

P99.9/max TPOT is not monotonic with TP because the strict prefill-priority scheduler and the TP=2
capacity limit change which requests can decode concurrently. These tail values should not be used
alone to infer NCCL performance.

## Decision

Use TP=4 as the default Qwen3-32B L20 deployment and optimization target. It is the best measured
throughput/latency/power compromise and stays inside one `PIX` four-GPU group. Use TP=8 only when
the 18% P90 latency reduction or maximum throughput justifies four additional GPUs and lower energy
efficiency. Do not use TP=2 for this concurrency and sequence shape at memory ratio 0.8.

## Next step

Experiment 008 will target the strict prefill-priority behavior identified in experiment 006. Add
an open-loop mixed-arrival benchmark, establish the TP=4 starvation baseline, then evaluate a
bounded prefill/decode interleaving policy without sacrificing steady-state throughput.
