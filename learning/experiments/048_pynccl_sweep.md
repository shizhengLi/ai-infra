# Experiment 048: PyNCCL Buffer and Protocol Sweep

## Question

Can the accepted graph-disabled direct-PyNCCL profile be improved by changing
the symmetric buffer threshold or the NCCL protocol on the L20 PCIe topology?

## Hypothesis

Mini-SGLang's PyNCCL wrapper copies small all-reduce tensors through an internal
symmetric buffer when the configured threshold is non-zero. That copy may be
more expensive than direct NCCL for decode-sized tensors. Separately, a fixed
NCCL protocol may improve the 80--320 KiB messages common in TP4 decode, but it
must be validated end to end because protocol changes can affect scheduling and
tail latency.

## Method

### Microbenchmark

The existing `benchmark/online/bench_tp_collective_profile.py` was run on four
L20 GPUs with TP=4, BF16, hidden size 5120, 30 warmups, and 200 timed
repetitions. Message sizes were 1, 8, 32, 128, and 512 hidden-size tokens
(10 KiB to 5 MiB). The primary variable was changed one run at a time:

- PyNCCL maximum internal buffer: `0`, `1M`, and `20M`.
- `NCCL_ALGO`: default, `Ring`, and `Tree`.
- `NCCL_PROTO`: default, `Simple`, and `LL128`.

The online candidate was selected from this screen: direct buffer `0` plus
`NCCL_PROTO=LL128`. The control was direct buffer `0` with the default NCCL
protocol.

### Online validation

Qwen3-32B BF16, TP=4, four L20 GPUs, CUDA Graphs disabled, direct PyNCCL,
`memory_ratio=0.7`, maximum prefill length 2048, prefill streak 1, and
decode-active prefill budget 1024. The workload used input/output lengths
256/32, concurrency 1/8/32, seed `4800042`, two warmups, and three recorded
repeats per concurrency.

The first treatment startup hit an out-of-memory error because orphaned
scheduler processes from the previous server still occupied the GPUs. Those
known PIDs were terminated, all GPU memory returned to zero, and the treatment
was rerun successfully. This was an environment cleanup issue, not a treatment
result.

## Microbenchmark results

Average all-reduce latency in milliseconds, in message-size order 10 KiB,
80 KiB, 320 KiB, 1.25 MiB, and 5 MiB:

| Configuration | 10 KiB | 80 KiB | 320 KiB | 1.25 MiB | 5 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, default protocol | 0.0112 | 0.0186 | 0.0542 | 0.1201 | 0.3967 |
| Internal buffer 1M | 0.0188 | 0.0319 | 0.0451 | 0.1200 | 0.3979 |
| Internal buffer 20M | 0.0336 | 0.0437 | 0.0498 | 0.1663 | 0.6368 |
| Direct, `NCCL_ALGO=Ring` | 0.0129 | 0.0188 | 0.0540 | 0.1202 | 0.3981 |
| Direct, `NCCL_ALGO=Tree` | 0.0123 | 0.0510 | 0.1232 | 0.4044 | 0.8362 |
| Direct, `NCCL_PROTO=Simple` | 0.0341 | 0.0511 | 0.0579 | 0.1201 | 0.3976 |
| Direct, `NCCL_PROTO=LL128` | 0.0185 | 0.0250 | 0.0394 | 0.1155 | 0.4329 |

The internal buffer is slower for most sizes because it adds device-to-device
copies before and after the collective. LL128 helps the 80--320 KiB points but
loses at 10 KiB and 5 MiB. Tree is a poor match for this four-way PCIe path.

## Online results

| C | Control tok/s | LL128 tok/s | Throughput change | Control P90 TPOT ms | LL128 P90 TPOT ms | P90 change | P99.9 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.29 | 36.56 | +0.76% | 26.72 | 26.50 | -0.80% | +4.43% |
| 8 | 178.20 | 171.85 | -3.56% | 29.07 | 29.51 | +1.50% | +3.54% |
| 32 | 303.15 | 300.51 | -0.87% | 244.68 | 250.01 | +2.18% | +2.49% |

The C=32 treatment also increased average TTFT by 3.69%. The small C=1 gain is
within the range where it cannot justify a deployment change, while C=8 and
C=32 fail the throughput/tail gate.

## Decision

- Reject `NCCL_PROTO=LL128` as a serving optimization for this workload.
- Reject `NCCL_ALGO=Tree` and `NCCL_PROTO=Simple` based on the microbenchmark;
  neither merits an online matrix.
- Keep `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0` as the explicit graph-disabled profile.
  The microbenchmark supports the direct path and no non-zero buffer threshold
  improved the complete message-size range.
- Do not modify generic NCCL or Mini-SGLang defaults.
- Proceed to Experiment 049: open-loop arrival rates with the accepted TP4
  direct-PyNCCL profile. It tests whether the measured kernel choice remains
  useful when requests arrive over time rather than in synchronized bursts.

## Artifacts

Microbenchmarks:

- `results/048_micro_direct.json`
- `results/048_micro_1m.json`
- `results/048_micro_20m.json`
- `results/048_micro_ring.json`
- `results/048_micro_tree.json`
- `results/048_micro_simple.json`
- `results/048_micro_ll128.json`

Online validation:

- `results/048_control_seed4800042.json` and `.md`
- `results/048_ll128_seed4800042.json` and `.md`
- `results/048_control_telemetry.jsonl`
- `results/048_ll128_telemetry.jsonl`

Both online servers were stopped, and all eight GPUs reported zero utilization
and zero used memory after cleanup.
