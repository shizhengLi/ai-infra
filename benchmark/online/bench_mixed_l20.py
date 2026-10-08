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
    parser = argparse.ArgumentParser(description="Mixed-arrival Mini-SGLang benchmark for L20")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--anchor-count", type=int, default=8)
    parser.add_argument("--anchor-input-len", type=int, default=128)
    parser.add_argument("--anchor-output-len", type=int, default=512)
    parser.add_argument("--burst-count", type=int, default=24)
    parser.add_argument("--burst-input-len", type=int, default=1024)
    parser.add_argument("--burst-output-len", type=int, default=64)
    parser.add_argument("--burst-delay", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-indices", default="0,1,2,3")
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--server-tp", type=int, default=4)
    parser.add_argument("--server-max-extend-tokens", type=int, default=8192)
    parser.add_argument("--server-max-prefill-streak", type=int, default=0)
    parser.add_argument("--server-decode-active-prefill-tokens", type=int, default=0)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def token_timestamps(result: RawResult, output_len: int) -> list[float]:
    if len(result.tics) < output_len + 1:
        raise RuntimeError(
            f"request returned {len(result.tics) - 1} stream events, "
            f"expected {output_len} tokens"
        )
    return result.tics[1 : output_len + 1]


def summarize_role(results: list[RawResult], output_len: int) -> dict[str, float]:
    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []
    for result in results:
        tokens = token_timestamps(result, output_len)
        ttfts.append(tokens[0] - result.tics[0])
        tpots.extend(b - a for a, b in zip(tokens, tokens[1:]))
        e2es.append(tokens[-1] - result.tics[0])
    return {
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
    }


def summarize_run(
    raw: list[RawResult],
    anchor_count: int,
    anchor_output_len: int,
    burst_output_len: int,
) -> dict[str, Any]:
    anchors = raw[:anchor_count]
    bursts = raw[anchor_count:]
    burst_start = min(result.tics[0] for result in bursts)
    anchor_post_burst_gaps = []
    for result in anchors:
        tokens = token_timestamps(result, anchor_output_len)
        anchor_post_burst_gaps.extend(
            b - a for a, b in zip(tokens, tokens[1:]) if b >= burst_start
        )

    starts = [result.tics[0] for result in raw]
    last_tokens = [
        token_timestamps(
            result,
            anchor_output_len if index < anchor_count else burst_output_len,
        )[-1]
        for index, result in enumerate(raw)
    ]
    duration = max(last_tokens) - min(starts)
    output_tokens = anchor_count * anchor_output_len + len(bursts) * burst_output_len
    return {
        "duration_s": duration,
        "output_throughput_tok_s": output_tokens / duration,
        "anchor": summarize_role(anchors, anchor_output_len),
        "burst": summarize_role(bursts, burst_output_len),
        "anchor_post_burst_tpot_p99_ms": percentile(anchor_post_burst_gaps, 0.99) * 1000,
        "anchor_post_burst_tpot_p999_ms": percentile(anchor_post_burst_gaps, 0.999) * 1000,
        "anchor_post_burst_tpot_max_ms": max(anchor_post_burst_gaps) * 1000,
        "anchor_post_burst_gaps_over_100ms": sum(x > 0.1 for x in anchor_post_burst_gaps),
        "anchor_post_burst_gaps_over_1s": sum(x > 1.0 for x in anchor_post_burst_gaps),
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


def mean_metric(runs: list[dict[str, Any]], *path: str) -> float:
    values = []
    for run in runs:
        value: Any = run
        for key in path:
            value = value[key]
        values.append(float(value))
    return statistics.mean(values)


def build_aggregates(runs: list[dict[str, Any]]) -> dict[str, float]:
    throughputs = [run["output_throughput_tok_s"] for run in runs]
    return {
        "throughput": statistics.mean(throughputs),
        "throughput_std": statistics.pstdev(throughputs),
        "anchor_ttft_avg_ms": mean_metric(runs, "anchor", "ttft_avg_ms"),
        "anchor_tpot_p99_ms": mean_metric(runs, "anchor", "tpot_p99_ms"),
        "anchor_tpot_p999_ms": mean_metric(runs, "anchor", "tpot_p999_ms"),
        "anchor_tpot_max_ms": max(run["anchor"]["tpot_max_ms"] for run in runs),
        "anchor_post_burst_p999_ms": mean_metric(
            runs, "anchor_post_burst_tpot_p999_ms"
        ),
        "anchor_post_burst_max_ms": max(
            run["anchor_post_burst_tpot_max_ms"] for run in runs
        ),
        "burst_ttft_avg_ms": mean_metric(runs, "burst", "ttft_avg_ms"),
        "burst_ttft_p90_ms": mean_metric(runs, "burst", "ttft_p90_ms"),
        "burst_e2e_p90_s": mean_metric(runs, "burst", "e2e_p90_s"),
        "gpu_util_avg_pct": mean_metric(runs, "gpu_util_avg_pct"),
        "gpu_memory_peak_mib": max(run["gpu_memory_peak_mib"] for run in runs),
        "gpu_power_avg_w": mean_metric(runs, "gpu_power_avg_w"),
    }


def markdown_report(report: dict[str, Any]) -> str:
    config = report["config"]
    aggregate = report["aggregates"]
    anchor_workload = (
        f"{config['anchor_count']} x {config['anchor_input_len']} input / "
        f"{config['anchor_output_len']} output tokens"
    )
    burst_workload = (
        f"{config['burst_count']} x {config['burst_input_len']} input / "
        f"{config['burst_output_len']} output tokens at {config['burst_delay']} seconds"
    )
    return f"""# L20 mixed-arrival benchmark result

## Principle

Anchor requests begin decoding before a timed burst of long-prefill requests arrives. Anchor TPOT
after burst dispatch measures decode starvation directly. Stream finish events are excluded.

## Configuration

- Model: `{report['model']}`
- Timestamp: `{report['timestamp_utc']}`
- Git revision: `{report['environment']['git_revision']}`
- Tensor parallelism: `{config['server_tp']}`
- Prefill budget: `{config['server_max_extend_tokens']}`
- Maximum consecutive prefill batches: `{config['server_max_prefill_streak']}`
- Decode-active prefill tokens: `{config['server_decode_active_prefill_tokens']}`
- Anchors: `{anchor_workload}`
- Burst: `{burst_workload}`
- Repeats: `{config['repeats']}`

## Aggregate results

| Metric | Value |
| --- | ---: |
| Output throughput | {aggregate['throughput']:.2f} token/s |
| Throughput stddev | {aggregate['throughput_std']:.2f} |
| Anchor average TTFT | {aggregate['anchor_ttft_avg_ms']:.2f} ms |
| Anchor P99 TPOT | {aggregate['anchor_tpot_p99_ms']:.2f} ms |
| Anchor P99.9 TPOT | {aggregate['anchor_tpot_p999_ms']:.2f} ms |
| Anchor worst TPOT | {aggregate['anchor_tpot_max_ms']:.2f} ms |
| Post-burst anchor P99.9 TPOT | {aggregate['anchor_post_burst_p999_ms']:.2f} ms |
| Post-burst anchor worst TPOT | {aggregate['anchor_post_burst_max_ms']:.2f} ms |
| Burst average TTFT | {aggregate['burst_ttft_avg_ms']:.2f} ms |
| Burst P90 TTFT | {aggregate['burst_ttft_p90_ms']:.2f} ms |
| Burst P90 E2E | {aggregate['burst_e2e_p90_s']:.3f} s |
| Average GPU utilization | {aggregate['gpu_util_avg_pct']:.2f}% |
| Peak GPU memory | {aggregate['gpu_memory_peak_mib']:.0f} MiB |
| Average power per GPU | {aggregate['gpu_power_avg_w']:.2f} W |

Per-repeat measurements and normalized raw stream timestamps are stored in the JSON result.
"""


async def main() -> None:
    args = parse_args()
    positive_values = (
        args.anchor_count,
        args.anchor_input_len,
        args.anchor_output_len,
        args.burst_count,
        args.burst_input_len,
        args.burst_output_len,
        args.repeats,
    )
    if min(positive_values) < 1 or args.burst_delay <= 0:
        raise ValueError("counts, lengths, repeats, and burst delay must be positive")

    async with OpenAI(base_url=args.base_url, api_key="dummy") as client:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("server returned no models")
        model = models.data[0].id
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

        random.seed(args.seed - 1)
        warmup = generate_prompt(tokenizer, min(128, args.anchor_input_len))
        await benchmark_one(client, warmup, 16, model, pbar=False)

        runs: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            random.seed(args.seed + repeat * 10000)
            anchors = [
                generate_prompt(tokenizer, args.anchor_input_len)
                for _ in range(args.anchor_count)
            ]
            bursts = [
                generate_prompt(tokenizer, args.burst_input_len) for _ in range(args.burst_count)
            ]
            trace = [
                BenchmarkTrace(0.0, prompt, args.anchor_output_len) for prompt in anchors
            ] + [
                BenchmarkTrace(args.burst_delay, prompt, args.burst_output_len)
                for prompt in bursts
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
                args.anchor_count,
                args.anchor_output_len,
                args.burst_output_len,
            )
            metrics.update(summarize_gpu_samples(gpu_samples))
            run: dict[str, Any] = {"repeat": repeat + 1, **metrics}
            print(json.dumps(run), flush=True)
            origin = min(result.tics[0] for result in raw)
            run["raw_requests"] = [
                {
                    "role": "anchor" if index < args.anchor_count else "burst",
                    "index": index if index < args.anchor_count else index - args.anchor_count,
                    "tics_s": [timestamp - origin for timestamp in result.tics],
                }
                for index, result in enumerate(raw)
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
            "anchor_count": args.anchor_count,
            "anchor_input_len": args.anchor_input_len,
            "anchor_output_len": args.anchor_output_len,
            "burst_count": args.burst_count,
            "burst_input_len": args.burst_input_len,
            "burst_output_len": args.burst_output_len,
            "burst_delay": args.burst_delay,
            "repeats": args.repeats,
            "seed": args.seed,
            "gpu_indices": args.gpu_indices,
            "server_tp": args.server_tp,
            "server_max_extend_tokens": args.server_max_extend_tokens,
            "server_max_prefill_streak": args.server_max_prefill_streak,
            "server_decode_active_prefill_tokens": args.server_decode_active_prefill_tokens,
        },
        "runs": runs,
        "aggregates": build_aggregates(runs),
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
