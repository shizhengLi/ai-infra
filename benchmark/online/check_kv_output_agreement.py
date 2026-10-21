from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path

from minisgl.benchmark.client import generate_prompt
from openai import AsyncOpenAI
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture deterministic Mini-SGLang output for KV-cache agreement checks"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--seed", type=int, default=8700042)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def main() -> None:
    args = parse_args()
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("input and output lengths must be positive")

    async with AsyncOpenAI(base_url=args.base_url, api_key="dummy") as client:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("server returned no models")
        model = models.data[0].id
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        random.seed(args.seed)
        prompt = generate_prompt(tokenizer, args.input_len)
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.output_len,
            temperature=0.0,
            extra_body={"ignore_eos": True, "top_k": 1},
        )
        fragments: list[str] = []
        async for chunk in response:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content is not None:
                fragments.append(content)

    output = "".join(fragments)
    output_token_ids = tokenizer.encode(output, add_special_tokens=False)
    result = {
        "model": model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "prompt_sha256": sha256_text(prompt),
        "output_sha256": sha256_text(output),
        "output_characters": len(output),
        "retokenized_output_tokens": len(output_token_ids),
        "stream_fragment_count": len(fragments),
        "stream_fragments": fragments,
        "output_token_ids": output_token_ids,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    console_result = {
        key: value
        for key, value in result.items()
        if key not in {"stream_fragments", "output_token_ids"}
    }
    print(json.dumps(console_result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
