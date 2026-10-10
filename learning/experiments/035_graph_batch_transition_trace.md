# Experiment 035: identify CUDA Graph P90 batch transitions

## Status

Complete. The P90 tail is concentrated in the largest decoded Graph batches, especially logical
batch sizes 20/24/28/31/32 that replay padded Graph sizes 24/24/32/32/32. Direct PyNCCL makes the
same transition path slightly slower in scheduler result processing, so it should not be enabled
automatically in Graph mode until that transition is addressed.

## Objective and instrumentation

Experiment 034 proved that the Graph-mode P90 regression survives steady-state warmup. This
experiment records the scheduler pipeline events for the same TP4 Graph workload and adds two fields
to telemetry events:

- `padded_batch_size`: the Graph shape selected by `GraphRunner.pad_batch`;
- `uses_cuda_graph`: whether the batch is eligible for Graph replay.

The fields are emitted only when `--prefill-telemetry-path` is enabled. The default serving path is
unchanged. Both symmetric and direct PyNCCL runs used two unrecorded warmups plus three measured
repetitions at C=1/8/32, seed `3500042`.

## Observed Graph transitions

The symmetric and direct traces had exactly the same 530 decoded Graph batches:

| Logical -> padded batch | Count |
| --- | ---: |
| 1 -> 1 | 180 |
| 4 -> 4 | 20 |
| 7 -> 8 | 5 |
| 8 -> 8 | 155 |
| 12 -> 16 | 10 |
| 16 -> 16 | 10 |
| 20 -> 24 | 10 |
| 24 -> 24 | 10 |
| 28 -> 32 | 10 |
| 31 -> 32 | 5 |
| 32 -> 32 | 115 |

This rules out a different Graph-shape distribution as the explanation for the direct-mode P90
regression. The transition sequence is scheduler-driven and identical between treatments.

## Tail attribution

Using `process_start -> result_sent` as a scheduler-side completion interval, batches with duration
above 30 ms were concentrated in the padded-32 path:

| Path | Slow samples | Mean slow interval |
| --- | ---: | ---: |
| Symmetric, padded 32 | 130 | 32.73 ms |
| Direct, padded 32 | 130 | 33.10 ms |

The remaining six slow samples were the same logical transition groups (`20 -> 24`, `24 -> 24`,
`28 -> 32`, `31 -> 32`) in both treatments. Direct mode lowers the extreme P99.9 token tail in the
online result, but the padded-32 completion interval is about 0.4 ms slower, which is consistent
with the observed P90 TPOT regression at C=32. This is an attribution signal, not proof that the
collective alone causes every token-level tail.

Artifacts:

- `learning/results/035_graph_symmetric_telemetry.jsonl`
- `learning/results/035_graph_direct_telemetry.jsonl`
- `learning/results/035_graph_symmetric_batchtrace_seed3500042.json`
- `learning/results/035_graph_direct_batchtrace_seed3500042.json`
- `learning/experiments/035_graph_symmetric_batchtrace_seed3500042_raw.md`
- `learning/experiments/035_graph_direct_batchtrace_seed3500042_raw.md`

## Decision

- Treat padded-32 decode transitions as the next scheduler investigation target.
- Keep direct PyNCCL as an explicit graph-disabled TP4/TP8 profile; do not auto-enable it for Graph
  mode.
- Do not change Graph padding or remove the 32-shape capture yet. A padding change would alter the
  scheduler contract and needs a correctness/throughput matrix first.

Next experiment 036 should test a scheduler policy that avoids mixing a 32-shape decode transition
with pending prefill work, using the new telemetry to verify whether the P90 tail moves without
changing the collective implementation.
