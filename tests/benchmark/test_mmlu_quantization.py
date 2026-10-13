import csv
import io
import tarfile

from benchmark.offline.bench_mmlu_quantization import load_questions, parse_answer


def test_parse_answer_accepts_standalone_choice() -> None:
    assert parse_answer("B") == "B"
    assert parse_answer("Answer: c.") == "C"
    assert parse_answer("Because") is None


def test_load_questions_is_seeded_and_balanced(tmp_path) -> None:
    archive_path = tmp_path / "mmlu.tar"
    with tarfile.open(archive_path, "w") as archive:
        for subject in ("alpha", "beta"):
            data = io.StringIO()
            writer = csv.writer(data)
            for index in range(4):
                writer.writerow([f"question {index}", "a", "b", "c", "d", "ABCD"[index]])
            payload = data.getvalue().encode()
            info = tarfile.TarInfo(f"data/test/{subject}_test.csv")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    first = load_questions(archive_path, questions_per_subject=2, seed=7)
    second = load_questions(archive_path, questions_per_subject=2, seed=7)

    assert first == second
    assert len(first) == 4
    assert {item["subject"] for item in first} == {"alpha", "beta"}
