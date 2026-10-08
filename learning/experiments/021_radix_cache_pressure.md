# Experiment 021: Radix Cache pressure and prefix retention

## Status

Complete. All five acceptance rules passed in all three fresh-server repetitions.

## Objective

Force Radix Cache eviction with a controlled KV pool, then determine whether recently reused and
recently inserted prefixes survive while older prefixes are evicted. Measure eviction amplification
before considering a policy change.

## Principle

`RadixPrefixCache` orders evictable leaf nodes by their last tree-walk timestamp. A cache hit updates
the timestamp of every fully traversed node. Under pressure, the heap therefore intends to evict the
least recently accessed leaf first. Eviction happens at node granularity, so the actual number of
tokens reclaimed can exceed the immediate allocation deficit.

The experiment separates three questions:

1. **Retention:** do useful, recently touched prefixes remain matchable after pressure?
2. **Ordering:** do untouched older prefixes disappear before refreshed A/B and newly inserted G-I?
3. **Amplification:** how many tokens are actually reclaimed per requested eviction token?

Primary metrics are derived from rank-0 telemetry:

```text
useful_retention = useful probe requests with >=95% token hit rate / useful probe requests
eviction_amplification = sum(evicted_tokens) / sum(eviction_requested_tokens)
```

Latency is secondary because this experiment deliberately constrains capacity and changes the
server maximum sequence length.

## Fixed workload

Each prefix is a deterministic 512-token user prompt. Requests generate 16 tokens with greedy
sampling and run sequentially. A distinct 64-token warmup is excluded.

| Phase | Sequence | Purpose | Expected state |
| --- | --- | --- | --- |
| Fill | A B C D E F | Establish six equal-sized branches | Cold/template-only matches |
| Refresh | A B | Advance two old branches in LRU order | Near-full input-prefix hits |
| Pressure | G H I | Exceed the 4096-token pool | Cold matches and runtime eviction |
| Survivor probe | I H G B A F E | Probe likely survivors before misses can mutate state | Near-full hits |
| Evicted probe | D C | Probe oldest untouched branches last | Template-only misses |

The expected state is a hypothesis, not a filter. Every observed mismatch will be retained and
explained. Probes are ordered so all expected survivors are measured before a missed probe inserts a
new branch and causes another eviction.

## Fixed environment

- Model: Qwen3-32B BF16
- Hardware: 4 x NVIDIA L20, GPUs 0-3
- Tensor parallelism: 4 with PyNCCL
- CUDA 12.8 and FlashInfer attention
- Explicit KV pool: 4096 pages, page size 1
- CUDA Graph maximum batch size: 64
- Normal/decode-active prefill budgets: 8192/2304
- Maximum prefill streak: 1
- Decode result before prefill: enabled
- Output length: 16 tokens
- Repetitions: 3 fresh server processes
- Seeds: 3100042, 3101042, 3102042

The explicit page override is the only capacity variable. Each repetition starts a new server and
uses a new prompt seed while preserving token counts and request order.

## Acceptance rules

The baseline is accepted if all three repetitions:

1. Complete without allocation, integrity, or telemetry errors.
2. Record at least one eviction and exactly 20 measured match operations (UIDs 1-20).
3. Preserve A/B through the pressure phase, demonstrating that refresh affects retention.
4. Preserve the expected survivor set before miss probes mutate the cache.
5. Show materially lower matched-token ratios for C/D than for survivor probes.

No policy will be changed during experiment 021. If behavior is stable, the result will decide
whether experiment 022 should compare eviction policies, page sizes, or production-sized cache
allocation profiles.

## Planned artifacts

- Updated `benchmark/online/bench_radix_l20.py`
- `learning/experiments/021_rep{1,2,3}_raw.md`
- `learning/results/021_rep{1,2,3}.json`
- `learning/results/021_rep{1,2,3}_telemetry.jsonl`

## Results

### Prefix retention

Token-level results were identical across all three seeds.

| Request group | Per repetition | Matched / matchable | Useful hit rate |
| --- | ---: | ---: | ---: |
| Refresh A/B | 2/2 | 519/519 each | 100% |
| Survivor probes I/H/G/B/A/F/E | 7/7 | 519/519 each | 100% |
| Evicted probes D/C | 0/2 useful | 3/519 each | 0.578% template-only |

Pooled useful retention was 21/21 survivor probes, or 100%. Refreshed A/B survived while older,
untouched C/D did not, so timestamp refresh materially affected eviction order. The policy behavior
is deterministic across token content for this equal-length workload.

### Eviction amplification

The pressure phase includes only H and I, before probe misses modify the cache.

| Repetition | Requested tokens | Reclaimed tokens | Pressure amplification | Max resident tokens |
| --- | ---: | ---: | ---: | ---: |
| 1 | 374 | 1,142 | 3.053x | 3,805 |
| 2 | 372 | 1,140 | 3.065x | 3,803 |
| 3 | 372 | 1,140 | 3.065x | 3,803 |
| Mean | 372.7 | 1,140.7 | 3.061x | 3,803.7 |

Pressure amplification mean/sample standard deviation was `3.0608 +/- 0.0064x`. The first pressure
allocation evicted the small warmup branch and one old prompt branch; the next requested about 148
tokens but reclaimed an entire 532-token leaf. This is a property of compressed-node granularity,
not incorrect LRU ordering.

The later D/C miss probes intentionally mutated the cache and added two more eviction calls. Across
the complete 20-request sequence, total amplification was 3.265x, 3.293x, and 3.299x. These totals
are recorded for accounting but are not used as the pressure-policy metric.

### Latency corroboration

| Metric | Mean | Sample standard deviation | Range |
| --- | ---: | ---: | ---: |
| Hot-prefix TTFT | 43.52 ms | 1.37 ms | 42.42-45.06 ms |
| Steady cold-prefix TTFT* | 184.94 ms | 0.78 ms | 184.40-185.83 ms |
| Output throughput | 33.54 token/s | 0.15 token/s | 33.38-33.66 token/s |
| Average TPOT | 23.046 ms | 0.023 ms | 23.031-23.072 ms |

`*` Excludes UID 1's fresh-shape overhead. Hot-prefix TTFT was 76.47% lower than steady cold-prefix
TTFT, independently corroborating the telemetry classification. Latency remains secondary to exact
token matches in this experiment.

## Interpretation

Current leaf-level LRU makes the correct retention decision: all recently used prefixes survive and
the two oldest untouched branches are selected first. Replacing LRU is therefore not justified by
this workload.

The measurable inefficiency is eviction granularity. A 148-token deficit can discard a 532-token
leaf, throwing away 384 tokens that could remain as a useful partial prefix. Because Radix nodes
already store key/value tensors and page size is explicit, an opt-in policy could trim an aligned
tail from an evictable leaf instead of always deleting the entire node. Correctness requires keeping
the parent-child key stable, updating node tensors and size accounting, and never trimming protected
nodes.

## Decision

Accept the existing LRU ordering baseline and reject a wholesale policy replacement. Experiment 022
should implement default-disabled partial-leaf tail eviction and compare it against this exact
workload. Acceptance must require lower amplification, increased C/D partial-prefix retention,
lower miss-probe TTFT, unchanged hot-prefix retention, and full cache-integrity tests.

## Verification

- Three fresh Qwen3-32B TP=4 servers completed and shut down cleanly
- Each JSONL contains exactly measured UIDs 1-20 plus excluded warmup UID 0
- All repetitions recorded four evictions and zero cache integrity failures
- Formal JSON and JSONL artifacts parse successfully
- Full `tests/core`: 28 passed
- Benchmark compilation, Black formatting, and `git diff --check`: passed
- Post-run cleanup: no GPU compute processes and no listener on port 1919
