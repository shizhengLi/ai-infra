from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import httpx
except ModuleNotFoundError:
    import httpx2 as httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.offline.bench_mmlu_quantization import (
    format_content,
    load_questions,
    parse_answer,
)
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online MMLU generation gate for TP servers")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--data-tar", type=Path, required=True)
    parser.add_argument("--questions-per-subject", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5300042)
    parser.add_argument("--mode", choices=["bf16", "fp8"], required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


async def evaluate_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model: str,
    item: dict[str, Any],
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    async with semaphore:
        response = await client.post(
            "chat/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_k": 1,
            },
        )
        response.raise_for_status()
    output_text = response.json()["choices"][0]["message"]["content"]
    prediction = parse_answer(output_text)
    return {
        "id": item["id"],
        "subject": item["subject"],
        "answer": item["answer"],
        "prediction": prediction,
        "correct": prediction == item["answer"],
        "output_text": output_text,
    }


async def main() -> None:
    args = parse_args()
    if min(args.questions_per_subject, args.concurrency, args.max_tokens) < 1:
        raise ValueError("questions per subject, concurrency, and max tokens must be positive")
    questions = load_questions(args.data_tar, args.questions_per_subject, args.seed)

    async with OpenAI(base_url=args.base_url, api_key="dummy") as client:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("server returned no models")
        model = models.data[0].id
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": format_content(item)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for item in questions
    ]
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/", timeout=120.0
    ) as client:
        semaphore = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()
        records = await asyncio.gather(
            *(
                evaluate_one(client, semaphore, model, item, prompt, args.max_tokens)
                for item, prompt in zip(questions, prompts, strict=True)
            )
        )
        duration_s = time.perf_counter() - started

    parsed = [record for record in records if record["prediction"] is not None]
    correct = [record for record in records if record["correct"]]
    subject_accuracy = {}
    for subject in sorted({record["subject"] for record in records}):
        group = [record for record in records if record["subject"] == subject]
        subject_accuracy[subject] = statistics.mean(record["correct"] for record in group)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": model,
            "data_tar": str(args.data_tar),
            "mode": args.mode,
            "questions_per_subject": args.questions_per_subject,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "metrics": {
            "questions": len(records),
            "subjects": len(subject_accuracy),
            "parsed": len(parsed),
            "parse_rate": len(parsed) / len(records),
            "correct": len(correct),
            "accuracy": len(correct) / len(records),
            "evaluation_s": duration_s,
            "subject_accuracy": subject_accuracy,
        },
        "records": records,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
