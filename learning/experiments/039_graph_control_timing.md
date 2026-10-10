# Experiment 039: Graph-mode GPU path attribution

## Status

Complete. This is an attribution experiment, not a deployment optimization. The default scheduler
policy and Graph configuration are unchanged.

## Question

The previous scheduler experiments moved work around the padded-32 decode/prefill boundary, but did
not identify whether the remaining P90 cost came from Graph replay, sampling, or copying the sampled
token to CPU. This experiment adds per-batch CUDA-event timing below the scheduler policy:

- `gpu_model_ms`: model forward or CUDA Graph replay;
- `gpu_sample_ms`: sampler execution after the model output;
- `gpu_copy_ms`: sampled-token device-to-host copy completion;
- `gpu_execution_kind`: `graph` or `eager`.

Timing is enabled only when prefill telemetry is enabled on the primary rank. The normal serving
path does not allocate timing events. The existing copy-completion event is timing-enabled only in
this profiling path so CUDA event elapsed-time queries remain valid.

## Method

Qwen3-32B, TP=4, four L20 GPUs, symmetric PyNCCL, CUDA Graph max batch 64, Graph-tail priority 0,
`max-prefill-streak=1`, decode-active prefill budget 1024, input/output lengths 256/32, concurrency
1/8/32, three measured repeats after two steady-state warmups, seed `3900042`.

```text
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
.venv/bin/python benchmark/online/bench_l20.py \
  --base-url http://127.0.0.1:1935/v1 \
  --input-len 256 --output-len 32 --concurrency 1 8 32 \
  --repeats 3 --steady-warmup-repeats 2 --seed 3900042 \
  --gpu-indices 0,1,2,3 --server-tp 4 --server-memory-ratio 0.7 \
  --server-graph-max-bs 64 --server-max-extend-tokens 2048 \
  --server-max-prefill-streak 1 --server-decode-active-prefill-tokens 1024 \
  --server-decode-graph-tail-prefill-priority-batch-size 0
```

## Serving baseline

| C | Throughput tok/s | P90 TPOT ms | P99.9 TPOT ms | TTFT avg ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 38.040 | 24.762 | 26.128 | 101.618 |
| 8 | 166.885 | 27.370 | 361.601 | 532.534 |
| 32 | 267.630 | 211.305 | 383.556 | 1600.276 |

The C=32 P90 value retains the known Graph tail, so this control trace is suitable for attribution
but is not evidence that the tail has been fixed.

## GPU attribution

The telemetry contained 596 timed `copy_ready` events. Decode batches were all Graph executions:

| Padded decode batch | Model/Graph mean ms | Sampling mean ms | Copy mean ms |
| ---: | ---: | ---: | ---: |
| 1 | 24.613 | 0.011 | 0.007 |
| 4 | 26.359 | 0.013 | 0.007 |
| 8 | 27.033 | 0.013 | 0.006 |
| 16 | 28.623 | 0.016 | 0.006 |
| 24 | 32.540 | 0.018 | 0.006 |
| 32 | 34.075 | 0.020 | 0.007 |

At padded batch 32, model/Graph replay is over 1,200 times the combined sampling and copy means.
Sampling and copy are therefore not credible targets for the current P90 tail. Prefill timing also
showed the expected eager model cost (about 93 ms at batch 1 and 293--315 ms at batches 4--5), which
is useful context but was not treated as a decode optimization target.

Scheduler completion intervals still show the tail concentration at padded batch 32: 130 decode
samples exceeded the 30 ms interval range in this trace. This is consistent with Graph replay and
batch-shape scheduling, not with CPU token-copy latency.

## Decision and next step

- Do not change the sampler or token copy path based on this experiment.
- Keep Graph-tail priority disabled by default; experiment 038 failed the P90 gate.
- Keep the timing instrumentation opt-in through existing telemetry; it is an attribution tool.
- Next, test the padded-32 Graph replay/batch-shape path with a paired control and treatment. The
  candidate must improve C=32 P90 TPOT without a material C=1/8 or throughput regression.

## Artifacts

- `learning/results/039_graph_control_timing_seed3900042.json`
- `learning/results/039_graph_control_timing_telemetry.jsonl`
- `learning/experiments/039_graph_control_timing_seed3900042_raw.md`
- `python/minisgl/engine/engine.py`
- `python/minisgl/scheduler/scheduler.py`
