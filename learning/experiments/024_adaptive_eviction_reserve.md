# Experiment 024: Output-aware adaptive eviction reserve

## Status

Complete, accepted for further production-shaped testing. The policy, capacity normalization,
workload matrix, selection rule, and acceptance rules below were fixed before adaptive-treatment
data was collected.

## Objective

Determine whether experiment 023's fixed 16-page reserve generalizes beyond 16-token outputs, and
whether a bounded output-aware reserve can reduce repeated eviction for 64/128-token and mixed
output workloads without sacrificing more useful prefixes than the fixed policy.

## Policy

At each paged allocation, the adaptive policy derives near-term demand from requests already
admitted to the batch:

```text
adaptive_reserve_pages = min(
    sum(ceil(req.remain_len / page_size) for req in batch),
    adaptive_reserve_max_pages,
)
```

The experiment sets `adaptive_reserve_max_pages=128`. The existing fixed reserve must be zero when
adaptive mode is enabled. Adaptive mode requires partial radix eviction, remains opt-in, and is
disabled by default.

The reserve is recomputed only when `allocate_paged()` asks the allocator for pages. Existing free
pages are consumed first. As in experiment 023, the eviction target is capped at currently
evictable capacity, and telemetry separates immediate deficit, policy target, and actual
reclamation.

## Capacity normalization

Longer outputs add persistent tokens to each cached unique branch. Keeping the pool at 4096 would
change the number of prefixes that can survive, confounding reserve policy with capacity. The
pressure-revisit workload contains nine unique prefixes, so uniform-output pools use:

```text
num_pages(output_len) = 4096 + 9 * (output_len - 16)
```

With page size 1:

| Output length | KV pages |
| ---: | ---: |
| 16 | 4096 |
| 64 | 4528 |
| 128 | 5104 |

The mixed mapping is `A16 B64 C128 D16 E64 F128 G16 H64 I128`. Its additional unique-branch
output footprint is 480 tokens, so it uses 4576 pages. This normalization is an experimental
control, not a production sizing formula.

## Fixed environment

- Model: Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- Page size: 1
- CUDA Graph maximum batch size: 64
- Scheduling: prefill streak 1, active budget 2304, result-before-prefill enabled
- Pressure sequence: fill A-F, refresh A/B, pressure G-I, probe I/H/G/B/A/F/E, then D/C
- Greedy sampling with EOS ignored
- Seeds: 3100042, 3101042, 3102042
- One fresh server for every workload/policy run

## Stage A: uniform-length screen

Using seed 3100042, compare fixed reserve 16 and adaptive-max-128 at output lengths 16, 64, and 128.
Experiment 023 seed 3100042 is reused as the fixed-16/output-16 cell; all other cells use fresh
servers.

Adaptive passes the screen only if, at both 64 and 128 tokens:

1. it has no integrity/output failure;
2. its survivor-probe matched-token total is at least the fixed policy's total;
3. complete-sequence eviction calls are at least 40% lower than fixed 16;
4. actual reclamation is no more than 5% above fixed 16;
5. target overshoot remains exactly 1.000x.

The 16-token adaptive run is a compatibility check: it should be behaviorally equivalent to fixed
16 because its derived reserve is 16 pages.

## Stage B: mixed-length confirmation

If adaptive passes Stage A, compare fixed 16 and adaptive-max-128 on the mixed mapping across all
three seeds. Seed 3100042 is the first repetition; seeds 3101042 and 3102042 are confirmation runs.

Primary metrics use the complete 20-request sequence because per-request output length changes
which phase consumes reserve. Report:

- eviction call count;
- immediate deficit, policy target, actual reclamation, and target overshoot;
- per-prefix matched/matchable tokens for survivor and D/C probes;
- TTFT, TPOT, sequential output throughput, and output correctness.

## Final acceptance rules

Accept adaptive reserve for further production-shaped testing only if the three mixed repetitions:

- complete without integrity or output failures;
- never reduce any survivor probe's matched tokens versus its paired fixed-16 run;
- reduce mean complete-sequence eviction calls by at least 40%;
- keep mean actual reclamation within 5% of fixed 16;
- keep target overshoot at 1.000x;
- regress sequential output throughput by no more than 1%;
- regress pooled hot-survivor TTFT by no more than 5%.

If uniform outputs pass but mixed outputs fail, retain fixed reserve as the explicit workload-tuned
option and reject the proposed adaptive formula. Both generic defaults remain disabled regardless
of outcome.

## Planned artifacts

- CLI/config option `--radix-partial-eviction-adaptive-reserve-max-pages`
- Unit tests for adaptive reserve derivation, cap, batching, and configuration conflicts
- Benchmark support for per-prefix pressure output lengths
- Uniform screen raw Markdown/JSON/JSONL artifacts
- Three paired mixed-length repetitions and aggregate analysis

## Implementation

The allocator now accepts an optional adaptive reserve cap through
`--radix-partial-eviction-adaptive-reserve-max-pages`. At each `allocate_paged()` call it sums the
remaining output demand of the currently admitted requests in pages, caps that sum at the
configured maximum, and passes the result to the existing bounded partial-eviction allocator. A
zero cap preserves the old path. Fixed and adaptive reserves are mutually exclusive, and both
still require partial-leaf eviction.

The benchmark gained `--pressure-output-lens`, which assigns independent A-I output lengths while
retaining the same cache-pressure sequence. Unit tests cover page-size rounding, batch summation,
the cap, the disabled path, negative values, missing partial eviction, and fixed/adaptive conflicts.
The generated report also records the server KV-page count so capacity normalization is visible in
raw artifacts.

## Stage A results

All runs used seed 3100042 and fresh servers. Metrics cover the complete 20-request sequence, not
only the pressure phase. The fixed-16/output-16 cell reuses experiment 023 as pre-registered.

| Output | Policy | Calls | Requested | Target / reclaimed | Hot survivors | D / C match | Throughput |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Fixed 16 | 6 | 1416 | 1512 / 1512 | 7/7 | 337 / 3 | 33.52 tok/s |
| 16 | Adaptive 128 | 6 | 1416 | 1500 / 1500 | 7/7 | 349 / 3 | 33.82 tok/s |
| 64 | Fixed 16 | 31 | 1529 | 2025 / 2025 | 7/7 | 209 / 3 | 38.67 tok/s |
| 64 | Adaptive 128 | 6 | 1359 | 1731 / 1731 | 7/7 | 349 / 3 | 38.70 tok/s |
| 128 | Fixed 16 | 68 | 1714 | 2802 / 2802 | 7/7 | 3 / 3 | 39.53 tok/s |
| 128 | Adaptive 128 | 11 | 1700 | 2821 / 2821 | 7/7 | 3 / 3 | 39.53 tok/s |

At 64 tokens, adaptive reserve reduced calls by 80.65% and reclamation by 14.52%. At 128 tokens it
reduced calls by 83.82% while reclamation increased only 0.68%. Every survivor remained a complete
519/519-token hit, and every eviction reclaimed exactly its target. The 16-token compatibility
pair had the same call count and immediate deficit; its small target and D-retention differences
are run-to-run allocation variation. Stage A therefore passed all screening rules.

## Stage B results

The mixed mapping was `A16 B64 C128 D16 E64 F128 G16 H64 I128`, with 4576 KV pages. Each policy and
seed used a fresh server. Hot TTFT is the mean of the seven `probe-survivor` requests.

| Seed | Policy | Calls | Requested | Target / reclaimed | Min hot match | Hot TTFT | Throughput |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3100042 | Fixed 16 | 36 | 1351 | 1927 / 1927 | 519/519 | 45.49 ms | 38.840 tok/s |
| 3100042 | Adaptive 128 | 6 | 1216 | 1690 / 1690 | 519/519 | 43.37 ms | 38.706 tok/s |
| 3101042 | Fixed 16 | 37 | 1417 | 2009 / 2009 | 519/519 | 42.98 ms | 38.650 tok/s |
| 3101042 | Adaptive 128 | 6 | 1274 | 1760 / 1760 | 519/519 | 41.04 ms | 38.745 tok/s |
| 3102042 | Fixed 16 | 29 | 1211 | 1675 / 1675 | 519/519 | 41.92 ms | 38.769 tok/s |
| 3102042 | Adaptive 128 | 6 | 1208 | 1693 / 1693 | 519/519 | 40.47 ms | 38.717 tok/s |
| Mean / pooled | Fixed 16 | 34.0 | 1326.3 | 1870.3 / 1870.3 | 21/21 complete | 43.46 ms | 38.753 tok/s |
| Mean / pooled | Adaptive 128 | 6.0 | 1232.7 | 1714.3 / 1714.3 | 21/21 complete | 41.63 ms | 38.723 tok/s |

Across the three repetitions, aggregate dispersion was:

| Metric | Fixed 16 mean +/- sample SD (median) | Adaptive 128 mean +/- sample SD (median) |
| --- | ---: | ---: |
| Eviction calls | 34.0 +/- 4.4 (36) | 6.0 +/- 0.0 (6) |
| Actual reclamation | 1870.3 +/- 174.1 (1927) | 1714.3 +/- 39.6 (1693) |
| Hot-survivor TTFT | 43.46 +/- 1.84 ms (42.98) | 41.63 +/- 1.54 ms (41.04) |
| Sequential throughput | 38.753 +/- 0.096 tok/s (38.769) | 38.723 +/- 0.020 tok/s (38.717) |

Relative to fixed 16, adaptive reserve reduced mean complete-sequence eviction calls by 82.35%
and mean actual reclamation by 8.34%. Sequential output throughput changed by -0.078%, while pooled
hot-survivor TTFT improved by 4.23%. There were no request, integrity, or output failures. No paired
survivor lost matched tokens, and the largest per-operation difference between actual reclamation
and policy target was zero pages.

## Acceptance decision

| Rule | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Integrity/output failures | 0 | 0 | Pass |
| Paired survivor regressions | 0 | 0/21 | Pass |
| Mean eviction-call reduction | >=40% | 82.35% | Pass |
| Mean reclamation increase | <=5% | -8.34% | Pass |
| Target overshoot | 1.000x | 1.000x | Pass |
| Throughput regression | <=1% | 0.078% | Pass |
| Hot TTFT regression | <=5% | -4.23% | Pass |

The adaptive formula is accepted for the next, production-shaped validation stage. It solves the
fixed reserve's core weakness: a 16-page cushion is sufficient for 16-token decode but causes many
small reallocations as output demand grows. Reserving admitted near-term demand amortizes those
allocations without blindly evicting more cache; in the mixed workload it actually reclaimed less.

This is not a generic-default decision. The cap of 128 and the sequential pressure workload are
still calibrated controls, not evidence for concurrent production traffic. Keep partial eviction
and adaptive reserve disabled by default. Experiment 025 should test concurrent, variable-length
arrivals and verify that summed admitted demand does not over-reserve when multiple decodes overlap.

## Artifacts

- Raw reports: `learning/experiments/024_*_raw.md`
- Machine-readable request data: `learning/results/024_*.json`
- Rank-0 cache telemetry: `learning/results/024_*_telemetry.jsonl`
- Focused tests: `tests/core/test_partial_leaf_eviction.py`
