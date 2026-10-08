from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from minisgl.benchmark.client import (
    BenchmarkTrace,
    RawResult,
    benchmark_one,
    benchmark_trace,
    generate_prompt,
)
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poisson-arrival Mini-SGLang benchmark for L20")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--request-count", type=int, default=16)
    parser.add_argument("--arrival-rates", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-indices", default="0,1,2,3")
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=1000.0)
    parser.add_argument("--server-tp", type=int, default=4)
    parser.add_argument("--server-memory-ratio", type=float, default=0.8)
    parser.add_argument("--server-graph-max-bs", type=int, default=64)
    parser.add_argument("--server-max-extend-tokens", type=int, default=8192)
    parser.add_argument("--server-max-prefill-streak", type=int, default=1)
    parser.add_argument("--server-decode-active-prefill-tokens", type=int, required=True)
    parser.add_argument("--server-decode-overload-prefill-tokens", type=int, default=0)
    parser.add_argument("--server-decode-overload-prefill-threshold", type=int, default=0)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def generate_arrival_offsets(rate: float, count: int, seed: int) -> list[float]:
    if rate <= 0 or count < 1:
        raise ValueError("arrival rate and request count must be positive")
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + rng.expovariate(rate))
    return offsets


def token_timestamps(result: RawResult, output_len: int) -> list[float]:
    if len(result.tics) < output_len + 1:
        raise RuntimeError(
            f"request returned {len(result.tics) - 1} stream events, "
            f"expected {output_len} tokens"
        )
    return result.tics[1 : output_len + 1]


def summarize_run(
    raw: list[RawResult],
    offsets: list[float],
    output_len: int,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[str, float]:
    starts: list[float] = []
    last_tokens: list[float] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    request_max_tpots: list[float] = []
    e2es: list[float] = []
    for result in raw:
        tokens = token_timestamps(result, output_len)
        request_tpots = [b - a for a, b in zip(tokens, tokens[1:])]
        starts.append(result.tics[0])
        last_tokens.append(tokens[-1])
        ttfts.append(tokens[0] - result.tics[0])
        tpots.extend(request_tpots)
        request_max_tpots.append(max(request_tpots))
        e2es.append(tokens[-1] - result.tics[0])

    duration = max(last_tokens) - min(starts)
    ttft_slo_s = ttft_slo_ms / 1000
    tpot_slo_s = tpot_slo_ms / 1000
    return {
        "scheduled_arrival_span_s": offsets[-1] - offsets[0],
        "observed_duration_s": duration,
        "output_throughput_tok_s": len(raw) * output_len / duration,
        "request_throughput_req_s": len(raw) / duration,
        "ttft_avg_ms": statistics.mean(ttfts) * 1000,
        "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
        "ttft_p90_ms": percentile(ttfts, 0.90) * 1000,
        "ttft_p99_ms": percentile(ttfts, 0.99) * 1000,
        "tpot_avg_ms": statistics.mean(tpots) * 1000,
        "tpot_p90_ms": percentile(tpots, 0.90) * 1000,
        "tpot_p99_ms": percentile(tpots, 0.99) * 1000,
        "tpot_p999_ms": percentile(tpots, 0.999) * 1000,
        "tpot_max_ms": max(tpots) * 1000,
        "e2e_avg_s": statistics.mean(e2es),
        "e2e_p90_s": percentile(e2es, 0.90),
        "e2e_p99_s": percentile(e2es, 0.99),
        "ttft_slo_attainment_pct": 100 * sum(value <= ttft_slo_s for value in ttfts) / len(ttfts),
        "tpot_slo_attainment_pct": 100
        * sum(value <= tpot_slo_s for value in request_max_tpots)
        / len(request_max_tpots),
        "token_gaps_over_100ms": float(sum(value > 0.1 for value in tpots)),
        "token_gaps_over_1s": float(sum(value > 1.0 for value in tpots)),
    }


async def sample_gpus(
    stop: asyncio.Event,
    gpu_indices: str,
    interval: float,
    samples: list[dict[str, float]],
) -> None:
    query = "index,utilization.gpu,memory.used,power.draw"
    while not stop.is_set():
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "-i",
            gpu_indices,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        if process.returncode == 0:
            for line in stdout.decode().splitlines():
                index, utilization, memory, power = (part.strip() for part in line.split(","))
                samples.append(
                    {
                        "gpu": float(index),
                        "utilization_pct": float(utilization),
                        "memory_mib": float(memory),
                        "power_w": float(power),
                    }
                )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass


def summarize_gpu_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {"gpu_util_avg_pct": 0.0, "gpu_memory_peak_mib": 0.0, "gpu_power_avg_w": 0.0}
    return {
        "gpu_util_avg_pct": statistics.mean(sample["utilization_pct"] for sample in samples),
        "gpu_memory_peak_mib": max(sample["memory_mib"] for sample in samples),
        "gpu_power_avg_w": statistics.mean(sample["power_w"] for sample in samples),
    }


def git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, check=False, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], capture_output=True, check=False, text=True
    ).stdout
    return {"git_revision": revision or "unknown", "git_dirty": bool(status.strip())}


def mean_metric(runs: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(run[key]) for run in runs)


def build_aggregates(
    runs: list[dict[str, Any]], arrival_rates: list[float]
) -> list[dict[str, float]]:
    aggregates = []
    for rate in arrival_rates:
        group = [run for run in runs if run["arrival_rate_req_s"] == rate]
        throughputs = [run["output_throughput_tok_s"] for run in group]
        aggregates.append(
            {
                "arrival_rate_req_s": rate,
                "throughput_tok_s": statistics.mean(throughputs),
                "throughput_std": statistics.pstdev(throughputs),
                "request_throughput_req_s": mean_metric(group, "request_throughput_req_s"),
                "ttft_avg_ms": mean_metric(group, "ttft_avg_ms"),
                "ttft_p90_ms": mean_metric(group, "ttft_p90_ms"),
                "ttft_p99_ms": mean_metric(group, "ttft_p99_ms"),
                "tpot_avg_ms": mean_metric(group, "tpot_avg_ms"),
                "tpot_p99_ms": mean_metric(group, "tpot_p99_ms"),
                "tpot_p999_ms": mean_metric(group, "tpot_p999_ms"),
                "tpot_max_ms": max(run["tpot_max_ms"] for run in group),
                "e2e_p90_s": mean_metric(group, "e2e_p90_s"),
                "e2e_p99_s": mean_metric(group, "e2e_p99_s"),
                "ttft_slo_attainment_pct": mean_metric(group, "ttft_slo_attainment_pct"),
                "tpot_slo_attainment_pct": mean_metric(group, "tpot_slo_attainment_pct"),
                "token_gaps_over_1s": mean_metric(group, "token_gaps_over_1s"),
                "gpu_util_avg_pct": mean_metric(group, "gpu_util_avg_pct"),
                "gpu_memory_peak_mib": max(run["gpu_memory_peak_mib"] for run in group),
                "gpu_power_avg_w": mean_metric(group, "gpu_power_avg_w"),
            }
        )
    return aggregates


def markdown_report(report: dict[str, Any]) -> str:
    config = report["config"]
    rows = []
    for aggregate in report["aggregates"]:
        rows.append(
            "| {arrival_rate_req_s:.1f} | {throughput_tok_s:.2f} | {throughput_std:.2f} | "
            "{ttft_p90_ms:.2f} | {ttft_p99_ms:.2f} | {tpot_p99_ms:.2f} | "
            "{tpot_p999_ms:.2f} | {tpot_max_ms:.2f} | {e2e_p90_s:.3f} | "
            "{ttft_slo_attainment_pct:.1f} | {tpot_slo_attainment_pct:.1f} | "
            "{gpu_util_avg_pct:.2f} |".format(**aggregate)
        )
    table = "\n".join(rows)
    return f"""# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `{report['timestamp_utc']}`
- Git revision: `{report['environment']['git_revision']}`
- Git worktree dirty: `{report['environment']['git_dirty']}`
- Python: `{report['environment']['python']}`
- PyTorch: `{report['environment']['torch']}` (`cu{report['environment']['torch_cuda']}`)
- Model: `{report['model']}`

## Configuration

- Tensor parallelism: `{config['server_tp']}`
- Normal prefill budget: `{config['server_max_extend_tokens']}`
- Maximum prefill streak: `{config['server_max_prefill_streak']}`
- Decode-active prefill budget: `{config['server_decode_active_prefill_tokens']}`
- Decode-overload prefill budget: `{config['server_decode_overload_prefill_tokens']}`
- Decode-overload threshold: `{config['server_decode_overload_prefill_threshold']}` pending tokens
- Workload: `{config['request_count']} x {config['input_len']} input / {config['output_len']} output tokens`
- Arrival rates: `{config['arrival_rates']}` requests/s
- Repeats: `{config['repeats']}`
- SLOs: TTFT <= `{config['ttft_slo_ms']}` ms; per-request maximum TPOT <= `{config['tpot_slo_ms']}` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
"""


async def main() -> None:
    args = parse_args()
    if min(args.input_len, args.output_len, args.request_count, args.repeats) < 1:
        raise ValueError("token lengths, request count, and repeats must be positive")
    if any(rate <= 0 for rate in args.arrival_rates):
        raise ValueError("arrival rates must be positive")
    if args.ttft_slo_ms <= 0 or args.tpot_slo_ms <= 0:
        raise ValueError("SLO thresholds must be positive")

    async with OpenAI(base_url=args.base_url, api_key="dummy") as client:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("server returned no models")
        model = models.data[0].id
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

        random.seed(args.seed - 1)
        warmup = generate_prompt(tokenizer, min(128, args.input_len))
        await benchmark_one(client, warmup, 16, model, pbar=False)

        runs: list[dict[str, Any]] = []
        for rate_index, rate in enumerate(args.arrival_rates):
            for repeat in range(args.repeats):
                trace_seed = args.seed + rate_index * 100000 + repeat * 10000
                offsets = generate_arrival_offsets(rate, args.request_count, trace_seed)
                random.seed(trace_seed)
                prompts = [
                    generate_prompt(tokenizer, args.input_len) for _ in range(args.request_count)
                ]
                trace = [
                    BenchmarkTrace(offset, prompt, args.output_len)
                    for offset, prompt in zip(offsets, prompts, strict=True)
                ]

                stop = asyncio.Event()
                gpu_samples: list[dict[str, float]] = []
                monitor = asyncio.create_task(
                    sample_gpus(stop, args.gpu_indices, args.monitor_interval, gpu_samples)
                )
                try:
                    raw = await benchmark_trace(client, trace, model, pbar=False)
                finally:
                    stop.set()
                    await monitor

                metrics = summarize_run(
                    raw,
                    offsets,
                    args.output_len,
                    args.ttft_slo_ms,
                    args.tpot_slo_ms,
                )
                metrics.update(summarize_gpu_samples(gpu_samples))
                run: dict[str, Any] = {
                    "arrival_rate_req_s": rate,
                    "repeat": repeat + 1,
                    "trace_seed": trace_seed,
                    **metrics,
                }
                print(json.dumps(run), flush=True)

                origin = min(result.tics[0] for result in raw)
                run["scheduled_arrival_offsets_s"] = offsets
                run["raw_tics_s"] = [
                    [timestamp - origin for timestamp in result.tics] for result in raw
                ]
                runs.append(run)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **git_state(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "model": model,
        "config": {
            "base_url": args.base_url,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "request_count": args.request_count,
            "arrival_rates": args.arrival_rates,
            "repeats": args.repeats,
            "seed": args.seed,
            "gpu_indices": args.gpu_indices,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "server_tp": args.server_tp,
            "server_memory_ratio": args.server_memory_ratio,
            "server_graph_max_bs": args.server_graph_max_bs,
            "server_max_extend_tokens": args.server_max_extend_tokens,
            "server_max_prefill_streak": args.server_max_prefill_streak,
            "server_decode_active_prefill_tokens": args.server_decode_active_prefill_tokens,
            "server_decode_overload_prefill_tokens": args.server_decode_overload_prefill_tokens,
            "server_decode_overload_prefill_threshold": (
                args.server_decode_overload_prefill_threshold
            ),
        },
        "runs": runs,
        "aggregates": build_aggregates(runs, args.arrival_rates),
    }
    console_report = {key: report[key] for key in ("timestamp_utc", "config", "aggregates")}
    print(json.dumps(console_report, indent=2), flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_report(report))


if __name__ == "__main__":
    asyncio.run(main())
