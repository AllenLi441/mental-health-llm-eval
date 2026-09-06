#!/usr/bin/env python3
"""Paired ESConv fixed250 inference with dialogue-cluster bootstrap CIs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_response_eval.core import ESConvExample  # noqa: E402


LABELS = [
    "Questions",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Other",
]
TASK_RE = re.compile(r"^esconv-test-(\d{6})$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_turn(line: str) -> int:
    fields = line.rstrip("\n").split("EOS")[-1].strip().split(None, 3)
    if len(fields) != 4:
        raise ValueError("target segment does not contain three metadata fields")
    return int(fields[2])


def canonical_items(test_file: Path, n: int) -> list[dict]:
    items = []
    previous_turn = None
    dialogue_number = 0
    with test_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > n:
                break
            turn = target_turn(line)
            if previous_turn is None or turn <= previous_turn:
                dialogue_number += 1
            previous_turn = turn
            example = ESConvExample.from_tsv_line(line, line_number)
            items.append(
                {
                    "index": line_number - 1,
                    "task_id": f"esconv-test-{line_number:06d}",
                    "gold": example.gold_strategy,
                    "conversation_id": f"esconv-test-dialog-{dialogue_number:04d}",
                }
            )
    if len(items) != n:
        raise RuntimeError(f"test file has only {len(items)} rows, expected {n}")
    return items


def load_arm(path: Path, items_by_task: dict[str, dict]) -> dict[str, str | None]:
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record.get("task_id") or "")
            if task_id not in items_by_task:
                raise ValueError(f"{path}:{line_number} has unexpected task_id {task_id!r}")
            if task_id in predictions:
                raise ValueError(f"{path}:{line_number} duplicates {task_id}")
            stored_gold = record.get("gold_strategy")
            if stored_gold is not None and stored_gold != items_by_task[task_id]["gold"]:
                raise ValueError(
                    f"{path}:{line_number} gold mismatch for {task_id}: {stored_gold!r}"
                )
            prediction = record.get("predicted_strategy")
            predictions[task_id] = prediction if prediction in LABELS else None
    return predictions


def classification_metrics(gold: list[str], predicted: list[str | None]) -> dict:
    total = len(gold)
    correct = sum(g == p for g, p in zip(gold, predicted))
    per_class = {}
    weighted_sum = 0.0
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        support = sum(g == label for g in gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        weighted_sum += support * f1
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS),
        "weighted_f1": weighted_sum / total if total else 0.0,
        "invalid": sum(p is None for p in predicted),
        "per_class": per_class,
    }


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    correct_a: list[bool],
    correct_b: list[bool] | None,
    clusters: list[str],
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    by_cluster = defaultdict(list)
    for index, cluster in enumerate(clusters):
        by_cluster[cluster].append(index)
    cluster_ids = sorted(by_cluster)
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        numerator = 0
        denominator = 0
        for cluster in selected:
            for index in by_cluster[cluster]:
                numerator += int(correct_a[index])
                if correct_b is not None:
                    numerator -= int(correct_b[index])
                denominator += 1
        estimates.append(numerator / denominator)
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * raw[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("arm name is empty")
    return name.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = dict(args.arm)
    if len(arms) < 2 or len(arms) != len(args.arm):
        raise ValueError("provide at least two uniquely named arms")
    for path in arms.values():
        if not path.exists():
            raise FileNotFoundError(path)

    items = canonical_items(args.test_file, args.n)
    items_by_task = {item["task_id"]: item for item in items}
    gold = [item["gold"] for item in items]
    clusters = [item["conversation_id"] for item in items]
    predictions = {
        name: load_arm(path, items_by_task) for name, path in arms.items()
    }

    ordered_predictions = {
        name: [records.get(item["task_id"]) for item in items]
        for name, records in predictions.items()
    }
    correct = {
        name: [g == p for g, p in zip(gold, values)]
        for name, values in ordered_predictions.items()
    }
    arm_results = {}
    for offset, (name, values) in enumerate(ordered_predictions.items()):
        metrics = classification_metrics(gold, values)
        low, high = cluster_bootstrap(
            correct[name], None, clusters, args.bootstrap_repetitions, args.seed + offset
        )
        metrics["accuracy_cluster_bootstrap_95_ci"] = [low, high]
        metrics["coverage"] = len(predictions[name]) / args.n
        metrics["missing"] = args.n - len(predictions[name])
        arm_results[name] = metrics

    pair_results = {}
    raw_p = {}
    for offset, (name_a, name_b) in enumerate(itertools.combinations(arms, 2), start=100):
        a = correct[name_a]
        b = correct[name_b]
        only_a = sum(x and not y for x, y in zip(a, b))
        only_b = sum(y and not x for x, y in zip(a, b))
        both_correct = sum(x and y for x, y in zip(a, b))
        both_wrong = sum(not x and not y for x, y in zip(a, b))
        delta = sum(a) / args.n - sum(b) / args.n
        low, high = cluster_bootstrap(
            a, b, clusters, args.bootstrap_repetitions, args.seed + offset
        )
        key = f"{name_a}__vs__{name_b}"
        p_value = exact_mcnemar(only_a, only_b)
        raw_p[key] = p_value
        pair_results[key] = {
            "arm_a": name_a,
            "arm_b": name_b,
            "accuracy_delta_a_minus_b": delta,
            "cluster_bootstrap_95_ci": [low, high],
            "both_correct": both_correct,
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "both_wrong": both_wrong,
            "exact_mcnemar_two_sided_p": p_value,
        }
    adjusted = holm_adjust(raw_p)
    for key, value in adjusted.items():
        pair_results[key]["holm_adjusted_p"] = value

    result = {
        "report_id": "esconv-fixed250-paired-inference-20260812",
        "protocol": "provider_api_v2 JSON strategy+response; missing/invalid counts wrong",
        "n": args.n,
        "conversation_clusters": len(set(clusters)),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "test_file": str(args.test_file.resolve()),
        "test_file_sha256": sha256_file(args.test_file),
        "arms": {
            name: {
                "source": str(path.resolve()),
                "source_sha256": sha256_file(path),
                **arm_results[name],
            }
            for name, path in arms.items()
        },
        "pairwise": pair_results,
        "limitations": [
            "fixed250 is a development-sized benchmark, not the 2,775-row full test",
            "this analysis belongs to the existing JSON strategy+response API track",
            "it cannot be inherited by the new one-letter strategy-only LoRA track",
        ],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ESConv fixed250 paired inference",
        "",
        f"Protocol: `{result['protocol']}`",
        "",
        f"Items: {args.n}; reconstructed dialogue clusters: {result['conversation_clusters']}.",
        "",
        "## Arms",
        "",
        "| Model | Accuracy | Cluster-bootstrap 95% CI | Macro-F1 | Weighted-F1 | Missing/invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in arm_results.items():
        low, high = metrics["accuracy_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {metrics['accuracy']:.2%} | [{low:.2%}, {high:.2%}] | "
            f"{metrics['macro_f1']:.2%} | {metrics['weighted_f1']:.2%} | "
            f"{metrics['missing']}/{metrics['invalid']} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise",
            "",
            "| Comparison (A-B) | Accuracy delta | Cluster-bootstrap 95% CI | Only A correct | Only B correct | McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_results.values():
        low, high = row["cluster_bootstrap_95_ci"]
        lines.append(
            f"| {row['arm_a']} - {row['arm_b']} | "
            f"{row['accuracy_delta_a_minus_b']:+.2%} | [{low:+.2%}, {high:+.2%}] | "
            f"{row['only_a_correct']} | {row['only_b_correct']} | "
            f"{row['exact_mcnemar_two_sided_p']:.6g} | {row['holm_adjusted_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            "These are fixed250 development results. They do not replace a full-test score, and they do not transfer to the new one-letter LoRA protocol.",
            "",
        ]
    )
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")
    print(args.json_output)
    print(args.markdown_output)


if __name__ == "__main__":
    main()

