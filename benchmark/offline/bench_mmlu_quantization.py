from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import statistics
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from transformers import AutoTokenizer


CHOICES = "ABCD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired-friendly MMLU generation gate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-tar", type=Path, required=True)
    parser.add_argument("--quantization", choices=["fp8"], default=None)
    parser.add_argument("--questions-per-subject", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=5300042)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=32)
    parser.add_argument("--num-pages", type=int, default=32768)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def parse_answer(text: str) -> str | None:
    match = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", text.strip().upper())
    return match.group(1) if match else None


def load_questions(data_tar: Path, questions_per_subject: int, seed: int) -> list[dict[str, Any]]:
    questions = []
    with tarfile.open(data_tar) as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile()
                and member.name.startswith("data/test/")
                and member.name.endswith("_test.csv")
            ),
            key=lambda member: member.name,
        )
        for subject_index, member in enumerate(members):
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {member.name}")
            rows = list(csv.reader(io.TextIOWrapper(extracted, encoding="utf-8")))
            rng = random.Random(seed + subject_index)
            selected = rng.sample(rows, min(questions_per_subject, len(rows)))
            subject = Path(member.name).stem.removesuffix("_test")
            for row_index, row in enumerate(selected):
                if len(row) != 6 or row[5] not in CHOICES:
                    raise ValueError(f"Unexpected MMLU row in {member.name}: {row}")
                questions.append(
                    {
                        "id": f"{subject}:{row_index}",
                        "subject": subject,
                        "question": row[0],
                        "choices": row[1:5],
                        "answer": row[5],
                    }
                )
    if not questions:
        raise ValueError(f"No MMLU test questions found in {data_tar}")
    return questions


def format_content(item: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(CHOICES, item["choices"], strict=True)
    )
    return (
        "Answer this multiple-choice question with exactly one letter: A, B, C, or D. "
        "Do not explain.\n\n"
        f"Question: {item['question']}\n{choices}\nAnswer:"
    )


def format_prompt(tokenizer: Any, item: dict[str, Any]) -> list[int]:
    content = format_content(item)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    args = parse_args()
    if min(args.questions_per_subject, args.batch_size, args.num_pages, args.max_seq_len) < 1:
        raise ValueError("question, batch, page, and sequence limits must be positive")

    questions = load_questions(args.data_tar, args.questions_per_subject, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [format_prompt(tokenizer, item) for item in questions]
    if max(map(len, prompts)) + 4 > args.max_seq_len:
        raise ValueError("MMLU prompt exceeds --max-seq-len")

    init_start = time.perf_counter()
    llm = LLM(
        args.model,
        attention_backend="fi",
        quantization=args.quantization,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        max_running_req=args.batch_size,
        max_seq_len_override=args.max_seq_len,
        max_extend_tokens=8192,
        num_page_override=args.num_pages,
    )
    initialization_s = time.perf_counter() - init_start
    sampling = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=4)

    started = time.perf_counter()
    records = []
    for offset in range(0, len(prompts), args.batch_size):
        outputs = llm.generate(prompts[offset : offset + args.batch_size], sampling)
        for item, output in zip(
            questions[offset : offset + args.batch_size], outputs, strict=True
        ):
            prediction = parse_answer(str(output["text"]))
            records.append(
                {
                    "id": item["id"],
                    "subject": item["subject"],
                    "answer": item["answer"],
                    "prediction": prediction,
                    "correct": prediction == item["answer"],
                    "output_text": output["text"],
                    "output_token_ids": output["token_ids"],
                }
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
            "model": args.model,
            "data_tar": str(args.data_tar),
            "quantization": args.quantization,
            "questions_per_subject": args.questions_per_subject,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "num_pages": args.num_pages,
            "max_seq_len": args.max_seq_len,
        },
        "metrics": {
            "questions": len(records),
            "subjects": len(subject_accuracy),
            "parsed": len(parsed),
            "parse_rate": len(parsed) / len(records),
            "correct": len(correct),
            "accuracy": len(correct) / len(records),
            "initialization_s": initialization_s,
            "evaluation_s": duration_s,
            "subject_accuracy": subject_accuracy,
        },
        "records": records,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
