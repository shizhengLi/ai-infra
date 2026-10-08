# Experiment 006: Qwen3-32B TP=4 prefill budget

## Status

Complete.

## Hypothesis

At concurrency 32, increasing `max_extend_tokens` from 8192 to 32768 should admit the initial
prefill in fewer scheduler rounds. This may reduce TTFT, but a larger prefill batch can block decode
for longer and increase peak workspace memory or TPOT tail latency.

## Principle

Mini-SGLang uses `max_extend_tokens` as the scheduler prefill budget. Requests whose combined
uncached prompt tokens exceed the budget are chunked or scheduled in waves. A larger budget reduces
the number of waves, while each individual prefill step becomes larger. This experiment changes
only that budget so the latency/throughput effect can be attributed to prefill scheduling.

The online harness retains raw stream timestamps for this experiment. TPOT P99.9 and maximum values
can therefore expose rare decode interruptions that P90/P99 hide.

## Controlled variables

- Model: Qwen3-32B BF16, local ModelScope snapshot
- Hardware: GPUs 0-3, four NVIDIA L20 cards
- Tensor parallelism: 4, PyNCCL
- Attention backend: automatic (`fi` on SM89)
- Memory ratio: 0.8
- CUDA Graph maximum batch size: 64
- Maximum running requests: 128
- Prompt content: 1024 tokens plus chat-template tokens
- Output: 256 tokens, EOS ignored, greedy/top-k 1
- Concurrency: 32
- Repeats: 3 with distinct deterministic seeds
- Independent variable: `max_extend_tokens` = 8192, 32768

## Procedure

For each prefill budget, restart the server to avoid cross-run cache and allocator state. Run the
same deterministic online workload and record output throughput, TTFT, TPOT tails, E2E latency, GPU
utilization, memory, and power.

Server, with the budget changed between runs:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m minisgl \
  --model /data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master \
  --tp 4 --memory-ratio 0.8 --cuda-graph-max-bs 64 \
  --max-running-requests 128 --max-prefill-length BUDGET --port 1919
```

Client, run once for each budget:

```bash
python benchmark/online/bench_l20.py \
  --base-url http://127.0.0.1:1919/v1 \
  --input-len 1024 --output-len 256 --concurrency 32 --repeats 3 \
  --gpu-indices 0,1,2,3 --server-tp 4 --server-memory-ratio 0.8 \
  --server-graph-max-bs 64 --server-max-extend-tokens BUDGET
```

## Results

| Metric | 8192 | 32768 | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 415.34 token/s | 414.06 token/s | -0.31% |
| Average TTFT | 7036.29 ms | 8723.97 ms | +23.99% |
| P50 TTFT | 5923.96 ms | 8086.65 ms | +36.51% |
| P90 TTFT | 10344.71 ms | 10424.47 ms | +0.77% |
| Average TPOT | 49.65 ms | 43.29 ms | -12.81% |
| P90 TPOT | 37.23 ms | 37.15 ms | -0.21% |
| P99.9 TPOT | 4490.44 ms | 2388.04 ms | -46.82% |
| Worst observed TPOT | 9637.84 ms | 9001.27 ms | -6.60% |
| P90 E2E | 19.709 s | 19.772 s | +0.32% |
| Average GPU utilization | 96.91% | 96.95% | +0.05% |
| Peak GPU memory | 37833 MiB | 39669 MiB | +4.85% |
| Average power per GPU | 227.39 W | 229.76 W | +1.05% |

The 8192 run allocated 305066 KV-cache tokens per rank. The 32768 run allocated 301994 tokens,
3072 fewer tokens per rank, and used 1836 MiB more peak memory during the workload.

Raw reports:

- `learning/experiments/006_prefill_8192_raw.md`
- `learning/experiments/006_prefill_32768_raw.md`

Machine-readable results with stream timestamps:

- `learning/results/006_prefill_8192.json`
- `learning/results/006_prefill_32768.json`

## Interpretation

The larger budget did not improve throughput or P90 completion latency. It traded first-token
fairness for fewer decode interruptions. With 8192, the combined TTFT distribution formed four
groups around 3, 6, 9, and 10.3 seconds. With 32768, 75 of 96 requests received their first token
between 8 and 10.44 seconds; average and median TTFT therefore regressed substantially.

The scheduler implementation explains this shape. `_schedule_next_batch` tries
`prefill_manager.schedule_next_batch()` before decode. As long as pending prefill exists, decode is
not selected. A small budget lets an early request finish prefill and emit its first token, but its
decode is then starved by later prefill batches. A large budget handles more prompts in one prefill
batch, so fewer long interruptions occur after decode starts, but most requests wait longer for the
first token.

Across all three repeats, intervals longer than one second fell from 72 to 67 out of 24480 decode
intervals, and intervals longer than four seconds fell from 48 to 2. However, the worst interval
remained about nine seconds and varied substantially by repeat. Raising the budget alone therefore
does not solve strict prefill priority.

## Decision

Keep the default 8192 budget for the next scaling baseline. It has equivalent throughput and E2E,
lower average/median TTFT, and 1.8 GiB less peak memory. Do not adopt 32768 as a general L20
optimization solely because its P99.9 TPOT is lower.

The source-level optimization opportunity is scheduler fairness: interleave decode with prefill or
limit consecutive prefill batches. That change needs a mixed-arrival workload rather than only a
closed batch, so it will be evaluated separately after the TP scaling baseline.

## Next step

Experiment 007 will measure Qwen3-32B with TP=2/4/8 at the retained 8192 prefill budget and
concurrency 32. This will separate compute scaling from PCIe collective overhead before modifying
the scheduler.
