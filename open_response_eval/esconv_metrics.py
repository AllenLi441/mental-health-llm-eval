"""Pure, dependency-free metrics for ESConv strategy classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def compute_classification_metrics(
    records: Sequence[dict[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    """Score exact-match predictions with invalid rows retained in the denominator."""

    ordered_labels = list(labels)
    if not ordered_labels or len(ordered_labels) != len(set(ordered_labels)):
        raise ValueError("labels must be non-empty and unique")

    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    matrix = [[0 for _ in ordered_labels] for _ in ordered_labels]
    support_by_gold = [0 for _ in ordered_labels]
    invalid_by_gold = [0 for _ in ordered_labels]
    invalid = 0
    correct = 0
    for record in records:
        gold = record.get("gold")
        prediction = record.get("prediction")
        if gold in label_to_index:
            support_by_gold[label_to_index[gold]] += 1
        if (
            record.get("invalid")
            or gold not in label_to_index
            or prediction not in label_to_index
        ):
            invalid += 1
            if gold in label_to_index:
                invalid_by_gold[label_to_index[gold]] += 1
            continue
        gold_index = label_to_index[gold]
        prediction_index = label_to_index[prediction]
        matrix[gold_index][prediction_index] += 1
        correct += int(gold == prediction)

    total = len(records)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    weighted_sum = 0.0
    valid_support = 0
    for index, label in enumerate(ordered_labels):
        true_positive = matrix[index][index]
        support = support_by_gold[index]
        predicted = sum(row[index] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "correct": true_positive,
        }
        f1_values.append(f1)
        weighted_sum += f1 * support
        valid_support += support

    accuracy = correct / total if total else 0.0
    macro_f1 = sum(f1_values) / len(f1_values)
    weighted_f1 = weighted_sum / valid_support if valid_support else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "ACC": accuracy,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "invalid_rate": invalid / total if total else 0.0,
        "correct": correct,
        "total": total,
        "invalid": invalid,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": ordered_labels,
            "rows_gold_columns_predicted": matrix,
            "invalid_by_gold": {
                label: invalid_by_gold[index]
                for index, label in enumerate(ordered_labels)
            },
        },
    }
