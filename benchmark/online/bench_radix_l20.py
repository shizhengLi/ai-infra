from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from minisgl.benchmark.client import generate_prompt
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Radix Cache benchmark for L20")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["unique", "shared-system", "multi-turn", "pressure-revisit"],
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--request-count", type=int, default=12)
    parser.add_argument("--conversation-count", type=int, default=4)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--unique-tokens", type=int, default=1024)
    parser.add_argument("--shared-system-tokens", type=int, default=768)
    parser.add_argument("--shared-user-tokens", type=int, default=256)
    parser.add_argument("--multi-system-tokens", type=int, default=256)
    parser.add_argument("--multi-user-tokens", type=int, default=128)
    parser.add_argument("--pressure-prompt-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3000042)
    parser.add_argument("--server-tp", type=int, default=4)
    parser.add_argument("--server-num-pages", type=int, default=0)
    parser.add_argument("--server-radix-partial-eviction", action="store_true")
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def make_prompt(tokenizer: Any, token_count: int, seed: int) -> str:
    random.seed(seed)
    return generate_prompt(tokenizer, token_count)


def templated_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


async def run_request(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    output_len: int,
) -> tuple[list[float], str]:
    response = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=messages,
        max_tokens=output_len,
        temperature=0.0,
        extra_body={"ignore_eos": True, "top_k": 1},
    )
    timestamps = [time.perf_counter()]
    output_parts: list[str] = []
    async for chunk in response:
        timestamps.append(time.perf_counter())
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                output_parts.append(content)
    if len(timestamps) < output_len + 1:
        raise RuntimeError(
            f"request returned {len(timestamps) - 1} stream events, expected {output_len} tokens"
        )
    return timestamps[: output_len + 1], "".join(output_parts)


async def warmup(client: OpenAI, model: str, tokenizer: Any, seed: int) -> None:
    prompt = make_prompt(tokenizer, 64, seed)
    await run_request(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        output_len=8,
    )


async def run_unique(
    client: OpenAI,
    model: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    runs = []
    for index in range(args.request_count):
        messages = [
            {
                "role": "user",
                "content": make_prompt(
                    tokenizer,
                    args.unique_tokens,
                    args.seed + 1000 + index,
                ),
            }
        ]
        runs.append(await measure_one(client, model, tokenizer, args, messages, index))
    return runs


async def run_shared_system(
    client: OpenAI,
    model: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    system_prompt = make_prompt(tokenizer, args.shared_system_tokens, args.seed + 2000)
    runs = []
    for index in range(args.request_count):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": make_prompt(
                    tokenizer,
                    args.shared_user_tokens,
                    args.seed + 3000 + index,
                ),
            },
        ]
        runs.append(await measure_one(client, model, tokenizer, args, messages, index))
    return runs


async def run_multi_turn(
    client: OpenAI,
    model: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    runs = []
    request_index = 0
    for conversation in range(args.conversation_count):
        history = [
            {
                "role": "system",
                "content": make_prompt(
                    tokenizer,
                    args.multi_system_tokens,
                    args.seed + 4000 + conversation,
                ),
            }
        ]
        for turn in range(1, args.turns + 1):
            user_message = {
                "role": "user",
                "content": make_prompt(
                    tokenizer,
                    args.multi_user_tokens,
                    args.seed + 5000 + conversation * args.turns + turn,
                ),
            }
            messages = history + [user_message]
            run, assistant_text = await measure_one(
                client,
                model,
                tokenizer,
                args,
                messages,
                request_index,
                return_output=True,
            )
            run.update(conversation=conversation + 1, turn=turn)
            runs.append(run)
            history.extend(
                [
                    user_message,
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            request_index += 1
    return runs


async def run_pressure_revisit(
    client: OpenAI,
    model: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    labels = list("ABCDEFGHI")
    prompts = {
        label: make_prompt(
            tokenizer,
            args.pressure_prompt_tokens,
            args.seed + 6000 + index,
        )
        for index, label in enumerate(labels)
    }
    sequence = (
        [("fill", label, "cold") for label in "ABCDEF"]
        + [("refresh", label, "hit") for label in "AB"]
        + [("pressure", label, "cold") for label in "GHI"]
        + [("probe-survivor", label, "hit") for label in "IHGBAFE"]
        + [("probe-evicted", label, "miss") for label in "DC"]
    )

    runs = []
    for index, (phase, label, expected_cache_state) in enumerate(sequence):
        messages = [{"role": "user", "content": prompts[label]}]
        run = await measure_one(client, model, tokenizer, args, messages, index)
        run.update(
            phase=phase,
            prefix=label,
            expected_cache_state=expected_cache_state,
        )
        runs.append(run)
    return runs


async def measure_one(
    client: OpenAI,
    model: str,
    tokenizer: Any,
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    request_index: int,
    *,
    return_output: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    input_tokens = templated_token_count(tokenizer, messages)
    timestamps, output_text = await run_request(
        client,
        model=model,
        messages=messages,
        output_len=args.output_len,
    )
    token_timestamps = timestamps[1 : args.output_len + 1]
    run = {
        "request": request_index + 1,
        "expected_server_uid": request_index + 1,
        "input_tokens": input_tokens,
        "output_tokens": args.output_len,
        "ttft_ms": (token_timestamps[0] - timestamps[0]) * 1000,
        "e2e_s": token_timestamps[-1] - timestamps[0],
        "tpot_avg_ms": statistics.mean(
            later - earlier for earlier, later in zip(token_timestamps, token_timestamps[1:])
        )
        * 1000,
    }
    print(json.dumps(run), flush=True)
    if return_output:
        return run, output_text
    return run


def git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        check=False,
        text=True,
    ).stdout
    return {"git_revision": revision or "unknown", "git_dirty": bool(status.strip())}


def build_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    start_to_finish = sum(run["e2e_s"] for run in runs)
    return {
        "requests": len(runs),
        "input_tokens_avg": statistics.mean(run["input_tokens"] for run in runs),
        "input_tokens_total": sum(run["input_tokens"] for run in runs),
        "output_throughput_tok_s": sum(run["output_tokens"] for run in runs) / start_to_finish,
        "ttft_avg_ms": statistics.mean(run["ttft_ms"] for run in runs),
        "ttft_p50_ms": percentile([run["ttft_ms"] for run in runs], 0.50),
        "ttft_p90_ms": percentile([run["ttft_ms"] for run in runs], 0.90),
        "tpot_avg_ms": statistics.mean(run["tpot_avg_ms"] for run in runs),
        "e2e_avg_s": statistics.mean(run["e2e_s"] for run in runs),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = []
    for run in report["runs"]:
        label = str(run["request"])
        if "conversation" in run:
            label += f" (c{run['conversation']}t{run['turn']})"
        elif "phase" in run:
            label += f" ({run['phase']}:{run['prefix']})"
        rows.append(
            f"| {label} | {run['input_tokens']} | {run['ttft_ms']:.2f} | "
            f"{run['tpot_avg_ms']:.2f} | {run['e2e_s']:.3f} |"
        )
    return f"""# L20 Radix Cache workload result

## Configuration

- Scenario: `{report['config']['scenario']}`
- Model: `{report['model']}`
- Timestamp: `{report['timestamp_utc']}`
- Git revision: `{report['environment']['git_revision']}`
- Tensor parallelism: `{report['config']['server_tp']}`
- Partial-leaf eviction: `{report['config']['server_radix_partial_eviction']}`
- Output length: `{report['config']['output_len']}` tokens
- Seed: `{report['config']['seed']}`
- Expected measured server UIDs: `1-{summary['requests']}` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | {summary['requests']} |
| Average input tokens | {summary['input_tokens_avg']:.2f} |
| Total input tokens | {summary['input_tokens_total']} |
| Sequential output throughput | {summary['output_throughput_tok_s']:.2f} token/s |
| Average TTFT | {summary['ttft_avg_ms']:.2f} ms |
| P50 TTFT | {summary['ttft_p50_ms']:.2f} ms |
| P90 TTFT | {summary['ttft_p90_ms']:.2f} ms |
| Average TPOT | {summary['tpot_avg_ms']:.2f} ms |
| Average E2E | {summary['e2e_avg_s']:.3f} s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
"""


async def main() -> None:
    args = parse_args()
    if args.request_count < 1 or args.output_len < 2:
        raise ValueError("request count must be positive and output length must be at least two")
    if args.scenario == "multi-turn" and (
        args.conversation_count * args.turns != args.request_count
    ):
        raise ValueError("multi-turn request count must equal conversation count times turns")

    async with OpenAI(base_url=args.base_url, api_key="dummy") as client:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("server returned no models")
        model = models.data[0].id
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        await warmup(client, model, tokenizer, args.seed - 1)

        if args.scenario == "unique":
            runs = await run_unique(client, model, tokenizer, args)
        elif args.scenario == "shared-system":
            runs = await run_shared_system(client, model, tokenizer, args)
        elif args.scenario == "multi-turn":
            runs = await run_multi_turn(client, model, tokenizer, args)
        else:
            runs = await run_pressure_revisit(client, model, tokenizer, args)

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
            **vars(args),
            "request_count": len(runs),
            "markdown_out": str(args.markdown_out) if args.markdown_out else None,
            "json_out": str(args.json_out) if args.json_out else None,
        },
        "summary": build_summary(runs),
        "runs": runs,
    }
    print(json.dumps({"config": report["config"], "summary": report["summary"]}, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_report(report))


if __name__ == "__main__":
    asyncio.run(main())
