# Experiment 041: padded-32 Graph replay kernel attribution

## Status

Complete as an attribution experiment. No serving default was changed. The result reopens a narrow
communication hypothesis for experiment 042, but does not yet justify changing the Graph profile.

## Question

Experiment 039 measured about 34 ms for padded-32 Graph model execution, and experiment 040 showed
that adding capture shapes 20/28 did not reduce that cost. This experiment uses Nsight Systems and the
existing NVTX ranges to identify the kernel classes inside `MiniSGL.GraphReplay.bs=32`.

The replay marker was added around `CUDAGraph.replay()` only; graph contents, scheduler policy, and
communication implementation are unchanged.

## Method

Qwen3-32B, TP=4, four L20 GPUs, symmetric PyNCCL, Graph max batch 64, input/output 256/32,
concurrency 32, one warmup and one probe request batch. The probe is only to trigger the Nsight
capture and is not used as a performance result because profiler overhead changes serving latency.

Nsight capture used `--trace=cuda,nvtx` and the report was filtered with:

```text
nsys stats --filter-nvtx 'MiniSGL.GraphReplay.bs=32' \
  --report cuda_gpu_kern_sum --format column \
  learning/results/041_graph_replay_full.nsys-rep
```

## Kernel attribution

The filtered Graph replay interval contained 29.629 ms of summed GPU kernel time:

| Kernel class | Total time | Share |
| --- | ---: | ---: |
| NCCL all-reduce (`ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16`) | 29.629 ms | 52.0% |
| BF16 GEMM `s16816gemm 128x64` | 14.280 ms | 25.1% |
| BF16 GEMM `s1688gemm 64x128` | 9.232 ms | 16.2% |
| BF16 GEMM `s1688gemm 128x64` | 2.673 ms | 4.7% |
| RMSNorm, FlashInfer attention/activation, and copies | 0.846 ms | 1.4% |

The 23 all-reduce instances averaged 1.288 ms each in the filtered interval. The three dominant GEMM
classes account for another 46.0%, so communication is the largest single category but not the only
material cost. This is a new measured residual bottleneck after the earlier shape experiments.

## Decision and next step

- Keep the default symmetric PyNCCL and Graph configuration unchanged for now.
- Do not optimize sampler, copy, or capture shapes based on this trace.
- Run experiment 042 as a narrow paired comparison of the direct PyNCCL path versus symmetric PyNCCL
  under the same Graph workload, with special attention to padded-32 replay and the strict C=32 P90
  gate. Earlier direct-communication experiments improved throughput but regressed Graph P90, so this
  experiment is a re-measurement motivated by kernel attribution, not an assumed acceptance.
- If communication does not pass the P90 gate, focus the following experiment on the dominant BF16
  GEMM/linear path rather than adding more scheduler heuristics.

## Artifacts

- `learning/results/041_graph_replay_full.nsys-rep`
- `learning/results/041_graph_replay_full.sqlite`
- `learning/results/041_graph_replay_full_telemetry.jsonl`
- `learning/results/041_graph_replay_probe_seed4100042.json`
- `learning/experiments/041_graph_replay_full_probe_raw.md`
- `python/minisgl/engine/graph.py`

