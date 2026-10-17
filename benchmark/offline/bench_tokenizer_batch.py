from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoTokenizer

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.tokenize import TokenizeManager


def build_messages(count: int) -> list[TokenizeMsg]:
    messages: list[TokenizeMsg] = []
    half = count // 2
    for index in range(half):
        suffix = "benchmark tokenization latency " * (2 + index % 5)
        messages.append(
            TokenizeMsg(
                uid=2 * index,
                text=f"Explain request {index}: {suffix}",
                sampling_params=SamplingParams(),
            )
        )
        messages.append(
            TokenizeMsg(
                uid=2 * index + 1,
                text=[{"role": "user", "content": f"Chat request {index}: {suffix}"}],
                sampling_params=SamplingParams(),
            )
        )
    return messages


def tokenize_one_by_one(tokenizer, messages: list[TokenizeMsg]) -> list[torch.Tensor]:
    results = []
    for message in messages:
        if isinstance(message.text, list):
            prompt = tokenizer.apply_chat_template(
                message.text,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = message.text
        results.append(tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32))
    return results


def measure(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scalar and batched tokenizer paths")
    parser.add_argument("--model", required=True, help="Local tokenizer directory or model ID")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages = build_messages(args.requests)
    manager = TokenizeManager(tokenizer)
    scalar = lambda: tokenize_one_by_one(tokenizer, messages)
    batched = lambda: manager.tokenize(messages)

    assert [x.tolist() for x in scalar()] == [x.tolist() for x in batched()]
    for _ in range(5):
        scalar()
        batched()
    scalar_samples = measure(scalar, args.repeats)
    batched_samples = measure(batched, args.repeats)
    scalar_median = statistics.median(scalar_samples)
    batched_median = statistics.median(batched_samples)
    print(f"requests={len(messages)} repeats={args.repeats}")
    print(f"scalar_median_ms={scalar_median:.3f}")
    print(f"batched_median_ms={batched_median:.3f}")
    print(f"median_speedup={scalar_median / batched_median:.2f}x")
    print(f"scalar_mean_ms={statistics.mean(scalar_samples):.3f}")
    print(f"batched_mean_ms={statistics.mean(batched_samples):.3f}")
    print(f"mean_speedup={statistics.mean(scalar_samples) / statistics.mean(batched_samples):.2f}x")


if __name__ == "__main__":
    main()
