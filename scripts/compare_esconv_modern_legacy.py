#!/usr/bin/env python3
"""Compare corrected modern and legacy ESConv per-item predictions."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[int(record["item_id"])] = record
    return records


def compare(modern, legacy):
    common = sorted(set(modern) & set(legacy))
    if not common:
        raise RuntimeError("Prediction files have no common item_id")
    agreement = 0
    modern_correct = 0
    legacy_correct = 0
    modern_only = 0
    legacy_only = 0
    disagreements = []
    for item_id in common:
        left, right = modern[item_id], legacy[item_id]
        if left.get("gold") != right.get("gold"):
            raise RuntimeError("Gold mismatch for item_id %d" % item_id)
        same = left.get("prediction") == right.get("prediction")
        agreement += int(same)
        left_correct = bool(left.get("correct"))
        right_correct = bool(right.get("correct"))
        modern_correct += int(left_correct)
        legacy_correct += int(right_correct)
        modern_only += int(left_correct and not right_correct)
        legacy_only += int(right_correct and not left_correct)
        if not same:
            disagreements.append(
                {
                    "item_id": item_id,
                    "gold": left.get("gold"),
                    "modern": left.get("prediction"),
                    "legacy": right.get("prediction"),
                }
            )
    total = len(common)
    return {
        "items": total,
        "prediction_agreement": agreement / total,
        "modern_accuracy": modern_correct / total,
        "legacy_accuracy": legacy_correct / total,
        "accuracy_difference_percentage_points": (modern_correct - legacy_correct) * 100.0 / total,
        "modern_only_correct": modern_only,
        "legacy_only_correct": legacy_only,
        "disagreements": disagreements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modern-predictions", type=Path, required=True)
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read_jsonl(args.modern_predictions), read_jsonl(args.legacy_predictions))
    result["modern_sha256"] = sha256_file(args.modern_predictions)
    result["legacy_sha256"] = sha256_file(args.legacy_predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "disagreements"}, indent=2))


if __name__ == "__main__":
    main()
