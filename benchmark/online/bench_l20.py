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
from minisgl.benchmark.client import RawResult, benchmark_one, benchmark_one_batch, generate_prompt
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online Mini-SGLang benchmark for NVIDIA L20")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-indices", default="0,1,2,3")
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--server-tp", type=int, default=4)
    parser.add_argument("--server-memory-ratio", type=float, default=0.8)
    parser.add_argument("--server-graph-max-bs", type=int, default=64)
    parser.add_argument("--server-max-extend-tokens", type=int, default=8192)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_requests(results: list[RawResult], output_len: int) -> dict[str, float]:
    starts: list[float] = []
    last_tokens: list[float] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []

    for result in results:
        if len(result.tics) < output_len + 1:
            raise RuntimeError(
                f"request returned {len(result.tics) - 1} stream events, "
                f"expected {output_len} tokens"
            )
        token_tics = result.tics[1 : output_len + 1]
        starts.append(result.tics[0])
        last_tokens.append(token_tics[-1])
        ttfts.append(token_tics[0] - result.tics[0])
        tpots.extend(b - a for a, b in zip(token_tics, token_tics[1:]))
        e2es.append(token_tics[-1] - result.tics[0])

    duration = max(last_tokens) - min(starts)
    output_tokens = len(results) * output_len
    return {
        "duration_s": duration,
        "output_throughput_tok_s": output_tokens / duration,
        "request_throughput_req_s": len(results) / duration,
        "ttft_avg_ms": statistics.mean(ttfts) * 1000,
        "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
        "ttft_p90_ms": percentile(ttfts, 0.90) * 1000,
        "ttft_p99_ms": percentile(ttfts, 0.99) * 1000,
        "tpot_avg_ms": statistics.mean(tpots) * 1000,
        "tpot_p50_ms": percentile(tpots, 0.50) * 1000,
        "tpot_p90_ms": percentile(tpots, 0.90) * 1000,
        "tpot_p99_ms": percentile(tpots, 0.99) * 1000,
        "tpot_p999_ms": percentile(tpots, 0.999) * 1000,
        "tpot_max_ms": max(tpots) * 1000,
        "e2e_avg_s": statistics.mean(e2es),
        "e2e_p90_s": percentile(e2es, 0.90),
        "e2e_p99_s": percentile(e2es, 0.99),
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


def markdown_report(report: dict[str, Any]) -> str:
    config = report["config"]
    table_header = (
        "| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | "
        "P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |"
    )
    rows = []
    for aggregate in report["aggregates"]:
        rows.append(
            "| {concurrency} | {throughput:.2f} | {throughput_std:.2f} | {ttft_avg:.2f} | "
            "{ttft:.2f} | {tpot_avg:.2f} | {tpot:.2f} | {tpot_p999:.2f} | "
            "{tpot_max:.2f} | {e2e:.3f} | {util:.2f} | {memory:.0f} |".format(**aggregate)
        )
    return f"""# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `{config['output_len']}` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `{report['timestamp_utc']}`
- Git revision: `{report['environment']['git_revision']}`
- Git worktree dirty: `{report['environment']['git_dirty']}`
- Python: `{report['environment']['python']}`
- PyTorch: `{report['environment']['torch']}` (`cu{report['environment']['torch_cuda']}`)
- Model: `{report['model']}`

## Server configuration

- Tensor parallelism: `{config['server_tp']}`
- Memory ratio: `{config['server_memory_ratio']}`
- CUDA Graph max batch: `{config['server_graph_max_bs']}`
- Maximum prefill tokens: `{config['server_max_extend_tokens']}`
- GPUs monitored: `{config['gpu_indices']}`

## Workload

- Prompt content length: `{config['input_len']}` tokens plus chat-template tokens
- Requested output: `{config['output_len']}` tokens with EOS ignored
- Concurrency: `{config['concurrency']}`
- Repeats: `{config['repeats']}`

## Aggregate results

{table_header}
{chr(10).join(rows)}

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
"""


async def main() -> None:
    args = parse_args()
    if args.repeats < 1 or min(args.input_len, args.output_len, *args.concurrency) < 1:
        raise ValueError("lengths, concurrency, and repeats must be positive")

    async with OpenAI(base_url=args.base_url, api_key="dummy") as client:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("server returned no models")
        model = models.data[0].id
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

        random.seed(args.seed - 1)
        warmup = generate_prompt(tokenizer, min(128, args.input_len))
        await benchmark_one(client, warmup, min(16, args.output_len), model, pbar=False)

        runs: list[dict[str, Any]] = []
        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                random.seed(args.seed + concurrency * 1000 + repeat)
                prompts = [generate_prompt(tokenizer, args.input_len) for _ in range(concurrency)]
                stop = asyncio.Event()
                gpu_samples: list[dict[str, float]] = []
                monitor = asyncio.create_task(
                    sample_gpus(stop, args.gpu_indices, args.monitor_interval, gpu_samples)
                )
                try:
                    raw = await benchmark_one_batch(
                        client,
                        prompts,
                        args.output_len,
                        model,
                        pbar=False,
                    )
                finally:
                    stop.set()
                    await monitor
                metrics = summarize_requests(raw, args.output_len)
                metrics.update(summarize_gpu_samples(gpu_samples))
                run = {"concurrency": concurrency, "repeat": repeat + 1, **metrics}
                print(json.dumps(run), flush=True)
                trace_origin = min(result.tics[0] for result in raw)
                run["raw_tics_s"] = [
                    [timestamp - trace_origin for timestamp in result.tics] for result in raw
                ]
                runs.append(run)

    aggregates = []
    for concurrency in args.concurrency:
        group = [run for run in runs if run["concurrency"] == concurrency]
        throughputs = [run["output_throughput_tok_s"] for run in group]
        aggregates.append(
            {
                "concurrency": concurrency,
                "throughput": statistics.mean(throughputs),
                "throughput_std": statistics.pstdev(throughputs),
                "ttft_avg": statistics.mean(run["ttft_avg_ms"] for run in group),
                "ttft": statistics.mean(run["ttft_p90_ms"] for run in group),
                "tpot_avg": statistics.mean(run["tpot_avg_ms"] for run in group),
                "tpot": statistics.mean(run["tpot_p90_ms"] for run in group),
                "tpot_p999": statistics.mean(run["tpot_p999_ms"] for run in group),
                "tpot_max": max(run["tpot_max_ms"] for run in group),
                "e2e": statistics.mean(run["e2e_p90_s"] for run in group),
                "util": statistics.mean(run["gpu_util_avg_pct"] for run in group),
                "memory": max(run["gpu_memory_peak_mib"] for run in group),
            }
        )

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
            "concurrency": args.concurrency,
            "repeats": args.repeats,
            "seed": args.seed,
            "gpu_indices": args.gpu_indices,
            "server_tp": args.server_tp,
            "server_memory_ratio": args.server_memory_ratio,
            "server_graph_max_bs": args.server_graph_max_bs,
            "server_max_extend_tokens": args.server_max_extend_tokens,
        },
        "runs": runs,
        "aggregates": aggregates,
    }
    console_report = {
        "timestamp_utc": report["timestamp_utc"],
        "model": report["model"],
        "config": report["config"],
        "aggregates": report["aggregates"],
    }
    print(json.dumps(console_report, indent=2), flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_report(report))


if __name__ == "__main__":
    asyncio.run(main())
