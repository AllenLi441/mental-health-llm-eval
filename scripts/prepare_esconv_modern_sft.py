#!/usr/bin/env python3
"""Build leakage-resistant ESConv train/dev SFT files and a test ID freeze.

This script deliberately writes no test conversations, labels, or responses to
the SFT files. The test output contains only item IDs and reconstructed dialogue
cluster IDs for later paired and cluster-bootstrap evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_response_eval.core import ESConvExample  # noqa: E402


EXPECTED_SHA256 = {
    "trainWithStrategy_short.tsv": "0ecf37462f8e3fa7f1dc5bfbecd70733abb3501c3bf83b27a957bec5a926ec21",
    "devWithStrategy_short.tsv": "625b511f40cf9a9285582e808e6bbb4c2f35d0b0cd063f3709db430b8d0c3bdc",
    "testWithStrategy_short.tsv": "b85ae888bf747cefa54bba2a6c3e2f6ccb4c1005d4e0b6d1d3be3823cf040aef",
}

LABEL_TO_LETTER = {
    "Questions": "A",
    "Restatement or Paraphrasing": "B",
    "Reflection of feelings": "C",
    "Self-disclosure": "D",
    "Affirmation and Reassurance": "E",
    "Providing Suggestions": "F",
    "Information": "G",
    "Other": "H",
}

STRATEGY_SYSTEM = (
    "You are being trained for ESConv next-strategy prediction. "
    "A=Questions, B=Restatement or Paraphrasing, "
    "C=Reflection of feelings, D=Self-disclosure, "
    "E=Affirmation and Reassurance, F=Providing Suggestions, "
    "G=Information, H=Other. Return exactly one letter."
)

GENERATION_SYSTEM = (
    "Generate the next ESConv supporter response conditioned on the specified "
    "support strategy. Return only the supporter response."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def target_turn(line: str) -> int:
    target = line.rstrip("\n").split("EOS")[-1].strip().split(None, 3)
    if len(target) != 4:
        raise ValueError("target segment does not contain three metadata fields")
    return int(target[2])


def transcript(example: ESConvExample) -> str:
    lines = []
    for message in example.context:
        if message["role"] == "assistant":
            strategy = message.get("strategy")
            prefix = f"[{strategy}] " if strategy else ""
            lines.append(f"Supporter: {prefix}{message['content']}")
        else:
            lines.append(f"Seeker: {message['content']}")
    return "\n".join(lines)


def parse_split(path: Path, split: str) -> list[dict]:
    rows = []
    previous_turn = None
    dialogue_number = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            turn = target_turn(line)
            if previous_turn is None or turn <= previous_turn:
                dialogue_number += 1
            previous_turn = turn
            example = ESConvExample.from_tsv_line(line, line_number)
            if example.gold_strategy not in LABEL_TO_LETTER:
                raise ValueError(
                    f"unknown strategy at {path}:{line_number}: {example.gold_strategy!r}"
                )
            rows.append(
                {
                    "sample_id": f"esconv-{split}-{line_number:06d}",
                    "conversation_id": f"esconv-{split}-dialog-{dialogue_number:04d}",
                    "line_number": line_number,
                    "target_turn": turn,
                    "example": example,
                }
            )
    return rows


def strategy_record(row: dict) -> dict:
    example = row["example"]
    user = f"Conversation:\n{transcript(example)}\n\nNext supporter strategy:"
    return {
        "messages": [
            {"role": "system", "content": STRATEGY_SYSTEM},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": LABEL_TO_LETTER[example.gold_strategy],
            },
        ]
    }


def generation_record(row: dict) -> dict:
    example = row["example"]
    letter = LABEL_TO_LETTER[example.gold_strategy]
    user = (
        f"Conversation:\n{transcript(example)}\n\n"
        f"Selected strategy: {letter} = {example.gold_strategy}\n"
        "Generate one supporter response."
    )
    return {
        "messages": [
            {"role": "system", "content": GENERATION_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": example.gold_response},
        ]
    }


def training_index(row: dict, split: str) -> dict:
    example = row["example"]
    return {
        "sample_id": row["sample_id"],
        "split": split,
        "conversation_id": row["conversation_id"],
        "source_line_number": row["line_number"],
        "target_turn": row["target_turn"],
        "gold_strategy": example.gold_strategy,
        "gold_letter": LABEL_TO_LETTER[example.gold_strategy],
    }


def test_index(row: dict) -> dict:
    return {
        "task_id": f"esconv-test-{row['line_number']:06d}",
        "conversation_id": row["conversation_id"],
        "source_line_number": row["line_number"],
        "target_turn": row["target_turn"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "tmp/official_benchmarks/esconv/codes/dataset",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/esconv/data"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {name: args.dataset_dir / name for name in EXPECTED_SHA256}
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    if source_hashes != EXPECTED_SHA256:
        raise RuntimeError(
            f"ESConv source hash mismatch: expected={EXPECTED_SHA256} actual={source_hashes}"
        )

    split_files = {
        "train": source_paths["trainWithStrategy_short.tsv"],
        "dev": source_paths["devWithStrategy_short.tsv"],
        "test": source_paths["testWithStrategy_short.tsv"],
    }
    parsed = {split: parse_split(path, split) for split, path in split_files.items()}

    outputs = {
        "strategy_train": args.output_dir / "esconv_strategy_train.jsonl",
        "strategy_dev": args.output_dir / "esconv_strategy_dev.jsonl",
        "generation_train": args.output_dir / "esconv_generation_train.jsonl",
        "generation_dev": args.output_dir / "esconv_generation_dev.jsonl",
        "train_dev_index": args.output_dir / "train_dev_index.jsonl",
        "test_item_dialog_ids": args.output_dir / "test_item_dialog_ids.jsonl",
    }
    write_jsonl(outputs["strategy_train"], [strategy_record(row) for row in parsed["train"]])
    write_jsonl(outputs["strategy_dev"], [strategy_record(row) for row in parsed["dev"]])
    write_jsonl(outputs["generation_train"], [generation_record(row) for row in parsed["train"]])
    write_jsonl(outputs["generation_dev"], [generation_record(row) for row in parsed["dev"]])
    write_jsonl(
        outputs["train_dev_index"],
        [training_index(row, split) for split in ("train", "dev") for row in parsed[split]],
    )
    write_jsonl(outputs["test_item_dialog_ids"], [test_index(row) for row in parsed["test"]])

    (args.output_dir / "dataset.sha256").write_text(
        "".join(f"{source_hashes[name]}  {source_paths[name]}\n" for name in source_paths),
        encoding="utf-8",
    )
    (args.output_dir / "line-counts.txt").write_text(
        "".join(f"{split}\t{len(rows)}\n" for split, rows in parsed.items()),
        encoding="utf-8",
    )

    output_manifest = {
        name: {
            "path": str(path.resolve()),
            "records": sum(1 for _ in path.open(encoding="utf-8")),
            "sha256": sha256_file(path),
        }
        for name, path in outputs.items()
    }
    manifest = {
        "protocol_id": "esconv-modern-strategy-sft-v1",
        "source_dataset": "original 1,053-dialogue author TSV split",
        "source_sha256": source_hashes,
        "label_to_letter": LABEL_TO_LETTER,
        "strategy_system": STRATEGY_SYSTEM,
        "generation_system": GENERATION_SYSTEM,
        "dialogue_counts": {
            split: len({row["conversation_id"] for row in rows})
            for split, rows in parsed.items()
        },
        "outputs": output_manifest,
        "test_leakage_controls": {
            "test_examples_written_to_sft": False,
            "test_index_contains_gold_labels": False,
            "test_index_contains_conversation_text": False,
            "training_splits": ["train", "dev"],
        },
        "external_api_upload_allowed": False,
        "note": (
            "These one-letter strategy SFT files define a new strategy-only track. "
            "They do not inherit scores from the existing JSON strategy+response API track."
        ),
    }
    manifest_path = args.output_dir / "modern_sft_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

