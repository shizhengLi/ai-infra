# Experiment 046: scheduler and Graph replay Nsight attribution

## Question

Experiment 045 rejected eager fallback for padded-24/32. The remaining uncertainty was whether the
tail came from work around `CUDAGraph.replay()` or from kernels inside the Graph. This experiment
adds an opt-in scheduler NVTX range around the decode forward and captures a short Nsight Systems
probe.

## Instrumentation

When prefill telemetry is enabled on rank 0, decode forwards now emit:

```text
MiniSGL.SchedulerForward.decode.bs=<padded batch>
```

The existing nested marker remains:

```text
MiniSGL.GraphReplay.bs=<padded batch>
```

Normal serving without telemetry does not create the scheduler NVTX range. No Graph contents,
collective path, or scheduler policy was changed.

## Probe method

- Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- Symmetric PyNCCL, automatic attention (`fi`), Graph max batch 64
- Input/output 256/32, concurrency 32
- One unrecorded warmup and one profiler probe batch
- Seed 4600042
- Nsight Systems: `--trace=cuda,nvtx --sample=none --cpuctxsw=none`

The profiler run is attribution-only; its throughput is not a performance result.

## Nsight result

Filtered `nvtx_pushpop_sum` ranges on the primary scheduler showed:

| Shape | SchedulerForward wall range | Nested GraphReplay range | Sampler ranges |
| ---: | ---: | ---: | ---: |
| 24 | 59.600 ms | 56.425 ms | 2 |
| 32 | 59.418 ms | 56.195 ms | 2 |

The roughly 3.2 ms difference includes input-buffer preparation, attention metadata preparation,
sampling, and host-side scheduler work. It is not a new large GPU kernel. `nvtx_gpu_proj_sum` did
not return data for the CUDA Graph range, so kernel attribution uses `cuda_gpu_kern_sum` filtered by
the scheduler range.

For both padded-24 and padded-32, the kernel mix was effectively identical:

| Kernel class | Padded-24 total | Padded-32 total | Approximate share |
| --- | ---: | ---: | ---: |
| NCCL symmetric all-reduce | 30.934 ms | 30.801 ms | 51.5% |
| BF16 `s16816gemm 128x64` | 15.465 ms | 15.465 ms | 25.9% |
| BF16 `s1688gemm 64x128` | 9.444 ms | 9.436 ms | 15.8% |
| BF16 `s1688gemm 128x64` | 2.923 ms | 2.922 ms | 4.9% |
| Norm, attention, activation, copies | about 1.1 ms | about 1.1 ms | about 1.8% |

The fresh event telemetry had 53 padded-32 and 8 padded-24 Graph samples, with model means of
34.905 ms and 33.194 ms respectively. These values include profiler overhead and are not used as a
serving comparison.

## Decision

The scheduler marker confirms that the extra interval around Graph replay is small relative to the
GPU work, and no shape-specific synchronization kernel appears. Do not add another scheduler
priority or result barrier. Keep the NVTX marker as an opt-in profiling aid and target the measured
NCCL/BF16 GEMM implementation in the next experiment.

## Artifacts

- `learning/results/046_graph_scheduler.nsys-rep`
- `learning/results/046_graph_scheduler.sqlite`
- `learning/results/046_scheduler_bs24_stats.txt`
- `learning/results/046_scheduler_bs32_stats.txt`
- `learning/results/046_graph_scheduler_telemetry.jsonl`
- `learning/results/046_graph_scheduler_probe.json`
