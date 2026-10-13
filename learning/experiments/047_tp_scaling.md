# Experiment 047: Qwen3-32B TP Scaling on L20

## Question

How does the accepted graph-disabled direct-PyNCCL profile scale from TP=2 to
TP=4 and TP=8 on eight L20 GPUs? The goal is to separate model parallel scaling
from the communication cost already observed in Nsight, and to produce an
interview-ready scaling result rather than another isolated parameter sweep.

## Hypothesis

Increasing tensor parallelism should increase aggregate decode throughput, but
the gain should be sublinear because every decode step contains TP collectives.
The direct PyNCCL path should be measurable with CUDA Graphs disabled, where the
Graph-mode tail regression is not part of the comparison.

## Method

- Model: Qwen3-32B BF16 from the ModelScope snapshot.
- Hardware: L20, one process per GPU, TP=2/4/8.
- Workload: input length 256, output length 32, distinct generated prompts.
- Concurrency: 1, 8, 32; two unrecorded warmups and three recorded repeats.
- Seed: `4700042`.
- CUDA Graphs: disabled with `--cuda-graph-max-bs 0`.
- Communication: direct PyNCCL via `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`.
- Scheduler controls: `--max-prefill-streak 1`, decode-active prefill budget
  1024, maximum prefill length 2048.
- Primary metrics: output token throughput, average TTFT, P90/P99.9 TPOT, and
  throughput standard deviation.

The TP2 run initially used `--memory-ratio 0.7` and failed during KV-cache
initialization with `Not enough memory for KV cache`. It was rerun at
`--memory-ratio 0.9`; TP4 and TP8 fit and used 0.7. This changes reserved KV
capacity, but not the model, request workload, or communication path, so the
memory-ratio difference is a limitation of this scaling comparison and must be
reported with the result.

TP1 is omitted: Qwen3-32B does not fit on a 48 GiB L20 at the production
configuration. The raw benchmark and telemetry artifacts are:

- `results/047_tp2_seed4700042.json` and `.md`
- `results/047_tp4_seed4700042.json` and `.md`
- `results/047_tp8_seed4700042.json` and `.md`
- `results/047_tp2_telemetry.jsonl`, `047_tp4_telemetry.jsonl`,
  `047_tp8_telemetry.jsonl`

## Results

| TP | C | Output tok/s | Stddev | Avg TTFT ms | P90 TPOT ms | P99.9 TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 21.03 | 0.00 | 149.10 | 45.81 | 45.95 |
| 2 | 8 | 107.34 | 0.01 | 658.71 | 51.16 | 441.31 |
| 2 | 32 | 196.69 | 0.05 | 1967.53 | 346.03 | 459.96 |
| 4 | 1 | 32.31 | 3.87 | 86.90 | 32.70 | 35.94 |
| 4 | 8 | 151.73 | 2.91 | 440.15 | 40.93 | 288.26 |
| 4 | 32 | 297.74 | 3.93 | 1315.59 | 246.65 | 303.82 |
| 8 | 1 | 36.20 | 0.18 | 86.56 | 27.22 | 29.61 |
| 8 | 8 | 179.96 | 0.67 | 433.76 | 29.92 | 268.61 |
| 8 | 32 | 326.81 | 0.32 | 1275.44 | 233.59 | 293.17 |

Scaling efficiency is computed against the TP2 throughput at the same
concurrency: `throughput(TP) / throughput(TP2) / (TP / 2)`.

| Concurrency | TP4 throughput scaling | TP4 efficiency | TP8 throughput scaling | TP8 efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.54x | 76.8% | 1.72x | 43.0% |
| 8 | 1.41x | 70.7% | 1.68x | 41.9% |
| 32 | 1.51x | 75.7% | 1.66x | 41.5% |

## Interpretation

TP4 is consistently useful: it reaches 1.41-1.54x TP2 throughput with roughly
71-77% parallel efficiency. TP8 still increases throughput, but only to
1.66-1.72x TP2, or about 42% efficiency. The diminishing return is visible even
at C=32, where the workload has enough requests to keep the GPUs busy. This is
consistent with a communication-bound decode path: more GPUs reduce local GEMM
work while TP collectives remain on every step.

The TP8 C=32 result has lower P90 TPOT than TP4 (233.59 vs 246.65 ms) and a
slightly lower P99.9 (293.17 vs 303.82 ms), so TP8 is not a latency regression
under this short-output workload. However, its aggregate efficiency is too low
to call it a cost-effective scaling point without a longer-output and arrival
rate study.

This online run does not provide kernel-level NCCL percentage. The available
Nsight experiment 046 attribution remains the communication context: NCCL was
about 51.5% of the measured padded Graph replay GPU kernel time. A future
communication sweep must collect an equivalent kernel-level profile before
claiming that a buffer or algorithm setting reduced NCCL cost.

## Decision

- Keep the explicit direct-PyNCCL, graph-disabled profile as the current L20
  TP4/TP8 deployment candidate.
- Treat TP4 as the best measured efficiency point for this model and workload.
- Do not change generic defaults based on this experiment; TP2 used a different
  memory ratio and the benchmark uses only 32 generated tokens.
- Proceed to Experiment 048: one-variable PyNCCL buffer/algorithm tuning in
  graph-disabled mode, with C=32 P90 TPOT and correctness gates.

## Reproducibility

All servers exited cleanly after each run. After TP8 shutdown, `nvidia-smi`
reported zero GPU utilization and zero used memory on GPUs 0-7. The benchmark
JSON files contain the per-repeat observations and aggregate fields used above.
