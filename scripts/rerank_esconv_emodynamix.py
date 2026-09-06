#!/usr/bin/env python3
"""Train-only strategy-transition reranker for EmoDynamiX development.

This is a deliberately small, post-hoc experiment inspired by the strategy
sequence statistics present in MultiESC.  It does not reproduce MultiESC and it
does not expose a test-split option.  Transition counts come from the released
EmoDynamiX author train split; configurations are compared on its valid split.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from open_response_eval.esconv_metrics import compute_classification_metrics
from scripts.eval_esconv_emodynamix import (
    CANONICAL_LABELS,
    canonical_strategy,
    prepare_author_pickle,
    sha256_file,
)


PROTOCOL_ID = "emodynamix-multiesc-inspired-transition-v1"
DEVELOPMENT_SPLITS = ("train", "valid")


def validate_development_split(split: str) -> str:
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError("development preparation accepts train and valid only")
    return split


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(path)
    os.link(temporary, path)
    temporary.unlink()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(path)
    os.link(temporary, path)
    temporary.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            records.append(value)
    return records


def prepare_development_pickle(
    path: Path,
    *,
    split: str,
    expected_sha256: str,
    allow_unsafe_pickle: bool,
) -> list[dict[str, Any]]:
    """Prepare a pinned author train/valid pickle after hashing its raw bytes."""

    split = validate_development_split(split)
    path = Path(path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{split} pickle hash mismatch: expected={expected_sha256} "
            f"actual={actual_sha256}"
        )
    prepared = prepare_author_pickle(path, allow_unsafe_pickle=allow_unsafe_pickle)
    output = []
    for index, source in enumerate(prepared, start=1):
        record = dict(source)
        record["item_id"] = f"emodynamix-author-{split}-{index:06d}"
        record["source_kind"] = f"emodynamix_author_{split}_pickle"
        output.append(record)
    return output


def _strategy_history(model_input: dict[str, Any]) -> tuple[int, ...]:
    raw = model_input.get("strategy_history")
    try:
        parsed = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"invalid strategy_history {raw!r}") from error
    if not isinstance(parsed, (list, tuple)) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in parsed
    ):
        raise ValueError(f"invalid strategy_history {raw!r}")
    values = tuple(value for value in parsed if value >= 0)
    if any(value >= len(CANONICAL_LABELS) for value in values):
        raise ValueError(f"strategy_history contains an out-of-range ID: {raw!r}")
    return values


def _probabilities(counts: Sequence[int], smoothing: float) -> list[float]:
    denominator = sum(counts) + smoothing * len(CANONICAL_LABELS)
    return [(count + smoothing) / denominator for count in counts]


def fit_transition_prior(
    records: Sequence[dict[str, Any]], *, history_order: int, smoothing: float
) -> dict[str, Any]:
    if history_order < 1:
        raise ValueError("history_order must be at least 1")
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be finite and positive")
    label_to_index = {label: index for index, label in enumerate(CANONICAL_LABELS)}
    global_counts = [0] * len(CANONICAL_LABELS)
    context_counts: dict[str, list[int]] = {}
    for record in records:
        if record.get("source_kind") != "emodynamix_author_train_pickle":
            raise ValueError("transition fitting is train-only")
        gold = record.get("gold")
        if gold not in label_to_index:
            raise ValueError(f"invalid train gold label {gold!r}")
        label_index = label_to_index[str(gold)]
        global_counts[label_index] += 1
        history = _strategy_history(dict(record.get("model_input", {})))
        for order in range(1, min(history_order, len(history)) + 1):
            suffix = history[-order:]
            key = f"{order}:" + ",".join(str(value) for value in suffix)
            counts = context_counts.setdefault(key, [0] * len(CANONICAL_LABELS))
            counts[label_index] += 1
    if not records:
        raise ValueError("train records must not be empty")
    return {
        "protocol_id": PROTOCOL_ID,
        "labels": list(CANONICAL_LABELS),
        "history_order": history_order,
        "smoothing": smoothing,
        "train_rows": len(records),
        "global_counts": global_counts,
        "global_probabilities": _probabilities(global_counts, smoothing),
        "contexts": {
            key: {
                "counts": counts,
                "probabilities": _probabilities(counts, smoothing),
            }
            for key, counts in sorted(context_counts.items())
        },
    }


def prior_for_input(prior: dict[str, Any], model_input: dict[str, Any]) -> list[float]:
    if prior.get("labels") != list(CANONICAL_LABELS):
        raise ValueError("transition prior label order mismatch")
    history = _strategy_history(model_input)
    contexts = prior.get("contexts")
    if not isinstance(contexts, dict):
        raise ValueError("transition prior has no contexts")
    maximum = min(int(prior.get("history_order", 0)), len(history))
    for order in range(maximum, 0, -1):
        suffix = history[-order:]
        key = f"{order}:" + ",".join(str(value) for value in suffix)
        value = contexts.get(key)
        if isinstance(value, dict):
            probabilities = value.get("probabilities")
            if isinstance(probabilities, list) and len(probabilities) == len(
                CANONICAL_LABELS
            ):
                return [float(probability) for probability in probabilities]
    probabilities = prior.get("global_probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != len(
        CANONICAL_LABELS
    ):
        raise ValueError("transition prior global probability shape mismatch")
    return [float(probability) for probability in probabilities]


def _canonical_logits(logits: Any) -> list[float]:
    if not isinstance(logits, dict):
        raise ValueError("EmoDynamiX logits must be a label-to-score object")
    scores: dict[str, float] = {}
    for raw_label, raw_score in logits.items():
        label = canonical_strategy(str(raw_label))
        if label is None or label in scores:
            raise ValueError(f"invalid or duplicate logit label {raw_label!r}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"non-finite logit for {raw_label!r}")
        scores[label] = score
    missing = [label for label in CANONICAL_LABELS if label not in scores]
    if missing:
        raise ValueError(f"EmoDynamiX logits are missing labels: {missing}")
    return [scores[label] for label in CANONICAL_LABELS]


def rerank_one(
    model_input: dict[str, Any],
    logits_by_label: dict[str, float],
    prior: dict[str, Any],
    *,
    transition_weight: float,
    class_adjustment_tau: float,
) -> dict[str, Any]:
    """Rerank one row; the interface intentionally has no gold-label argument."""

    if not math.isfinite(transition_weight) or transition_weight < 0:
        raise ValueError("transition_weight must be finite and non-negative")
    if not math.isfinite(class_adjustment_tau) or class_adjustment_tau < 0:
        raise ValueError("class_adjustment_tau must be finite and non-negative")
    logits = _canonical_logits(logits_by_label)
    probabilities = prior_for_input(prior, model_input)
    train_class_probabilities = prior.get("global_probabilities")
    if not isinstance(train_class_probabilities, list) or len(
        train_class_probabilities
    ) != len(CANONICAL_LABELS):
        raise ValueError("transition prior global probability shape mismatch")
    scores = [
        logit
        + transition_weight * math.log(max(probability, 1e-300))
        - class_adjustment_tau
        * math.log(max(float(train_probability), 1e-300))
        for logit, probability, train_probability in zip(
            logits, probabilities, train_class_probabilities
        )
    ]
    predicted_index = max(range(len(scores)), key=scores.__getitem__)
    base_index = max(range(len(logits)), key=logits.__getitem__)
    return {
        "prediction": CANONICAL_LABELS[predicted_index],
        "base_prediction": CANONICAL_LABELS[base_index],
        "transition_probabilities": {
            label: probabilities[index]
            for index, label in enumerate(CANONICAL_LABELS)
        },
        "train_class_probabilities": {
            label: float(train_class_probabilities[index])
            for index, label in enumerate(CANONICAL_LABELS)
        },
        "reranked_scores": {
            label: scores[index] for index, label in enumerate(CANONICAL_LABELS)
        },
    }


def _by_item_id(
    records: Sequence[dict[str, Any]], description: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = record.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in output:
            raise ValueError(f"{description} has a missing or duplicate item_id")
        output[item_id] = record
    return output


def evaluate_configuration(
    valid_records: Sequence[dict[str, Any]],
    raw_predictions: Sequence[dict[str, Any]],
    prior: dict[str, Any],
    *,
    transition_weight: float,
    class_adjustment_tau: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_by_id = _by_item_id(valid_records, "valid records")
    raw_by_id = _by_item_id(raw_predictions, "valid predictions")
    if set(valid_by_id) != set(raw_by_id):
        raise ValueError("valid record and prediction item IDs differ")
    scored = []
    for source in valid_records:
        if source.get("source_kind") != "emodynamix_author_valid_pickle":
            raise ValueError("configuration evaluation is valid-only")
        raw = raw_by_id[str(source["item_id"])]
        try:
            if raw.get("invalid"):
                raise ValueError(str(raw.get("error") or "invalid base prediction"))
            planned = rerank_one(
                dict(source.get("model_input", {})),
                raw.get("logits"),
                prior,
                transition_weight=transition_weight,
                class_adjustment_tau=class_adjustment_tau,
            )
            record = {
                "item_id": source["item_id"],
                "gold": source.get("gold"),
                "prediction": planned["prediction"],
                "base_prediction": planned["base_prediction"],
                "correct": planned["prediction"] == source.get("gold"),
                "invalid": False,
                "error": None,
                "model_input_sha256": source.get("model_input_sha256"),
                "transition_probabilities": planned["transition_probabilities"],
                "train_class_probabilities": planned[
                    "train_class_probabilities"
                ],
                "reranked_scores": planned["reranked_scores"],
            }
        except (TypeError, ValueError) as error:
            record = {
                "item_id": source["item_id"],
                "gold": source.get("gold"),
                "prediction": None,
                "base_prediction": raw.get("prediction"),
                "correct": False,
                "invalid": True,
                "error": f"{type(error).__name__}: {error}",
                "model_input_sha256": source.get("model_input_sha256"),
            }
        scored.append(record)
    return scored, compute_classification_metrics(scored, CANONICAL_LABELS)


def choose_best_candidate(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidate list must not be empty")

    def key(
        candidate: dict[str, Any]
    ) -> tuple[float, float, float, float, float, float]:
        metrics = candidate["metrics"]
        config = candidate["config"]
        return (
            float(metrics["macro_f1"]),
            float(metrics["accuracy"]),
            -float(config["transition_weight"]),
            -float(config.get("class_adjustment_tau", 0.0)),
            -float(config["history_order"]),
            -float(config["smoothing"]),
        )

    return max(candidates, key=key)


def select_configuration(
    train_records: Sequence[dict[str, Any]],
    valid_records: Sequence[dict[str, Any]],
    raw_predictions: Sequence[dict[str, Any]],
    *,
    history_orders: Sequence[int],
    smoothings: Sequence[float],
    transition_weights: Sequence[float],
    class_adjustment_taus: Sequence[float] = (0.0,),
) -> dict[str, Any]:
    candidates = []
    predictions_by_config: dict[
        tuple[int, float, float, float], list[dict[str, Any]]
    ] = {}
    for history_order in sorted(set(history_orders)):
        for smoothing in sorted(set(smoothings)):
            prior = fit_transition_prior(
                train_records, history_order=history_order, smoothing=smoothing
            )
            for transition_weight in sorted(set(transition_weights)):
                for class_adjustment_tau in sorted(set(class_adjustment_taus)):
                    predictions, metrics = evaluate_configuration(
                        valid_records,
                        raw_predictions,
                        prior,
                        transition_weight=transition_weight,
                        class_adjustment_tau=class_adjustment_tau,
                    )
                    config = {
                        "history_order": history_order,
                        "smoothing": smoothing,
                        "transition_weight": transition_weight,
                        "class_adjustment_tau": class_adjustment_tau,
                    }
                    candidates.append({"config": config, "metrics": metrics})
                    predictions_by_config[
                        (
                            history_order,
                            smoothing,
                            transition_weight,
                            class_adjustment_tau,
                        )
                    ] = predictions
    best = choose_best_candidate(candidates)
    baseline_candidates = [
        candidate
        for candidate in candidates
        if float(candidate["config"]["transition_weight"]) == 0.0
        and float(candidate["config"]["class_adjustment_tau"]) == 0.0
    ]
    if not baseline_candidates:
        raise ValueError(
            "transition_weights and class_adjustment_taus must include the 0.0 "
            "base-model control"
        )
    baseline = choose_best_candidate(baseline_candidates)
    best_key = (
        int(best["config"]["history_order"]),
        float(best["config"]["smoothing"]),
        float(best["config"]["transition_weight"]),
        float(best["config"]["class_adjustment_tau"]),
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "DEVELOPMENTAL_AUTHOR_VALID_ONLY",
        "leaderboard_eligible": False,
        "selection_rule": (
            "max valid Macro-F1; then Accuracy; then smaller transition weight, "
            "class-adjustment tau, history order, and smoothing"
        ),
        "split_contract": {
            "transition_fit": "author_train_only",
            "selection": "author_valid_only",
            "author_test_read": False,
            "frozen_test_read": False,
        },
        "grid": candidates,
        "baseline": baseline,
        "best": best,
        "best_predictions": predictions_by_config[best_key],
        "commitments": {
            "train_records_sha256": _canonical_sha256(train_records),
            "valid_records_sha256": _canonical_sha256(valid_records),
            "valid_predictions_sha256": _canonical_sha256(raw_predictions),
        },
    }


def _float_values(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _int_values(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--split", choices=DEVELOPMENT_SPLITS, required=True)
    prepare.add_argument("--source-pickle", type=Path, required=True)
    prepare.add_argument("--expected-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest-output", type=Path, required=True)
    prepare.add_argument("--allow-unsafe-pickle", action="store_true")

    select = subparsers.add_parser("select")
    select.add_argument("--train-prepared", type=Path, required=True)
    select.add_argument("--valid-prepared", type=Path, required=True)
    select.add_argument("--valid-predictions", type=Path, required=True)
    select.add_argument("--history-orders", default="1,2,3")
    select.add_argument("--smoothings", default="0.1,0.5,1.0")
    select.add_argument(
        "--transition-weights", default="0.0,0.05,0.1,0.2,0.4,0.8"
    )
    select.add_argument(
        "--class-adjustment-taus",
        default="0.0,0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0",
    )
    select.add_argument("--summary-output", type=Path, required=True)
    select.add_argument("--predictions-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        records = prepare_development_pickle(
            args.source_pickle,
            split=args.split,
            expected_sha256=args.expected_sha256,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        )
        _write_jsonl(args.output, records)
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "status": "DEVELOPMENT_INPUT_PREPARED",
            "split": args.split,
            "source": {
                "path": str(args.source_pickle.resolve()),
                "sha256": sha256_file(args.source_pickle),
            },
            "output": {
                "path": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "rows": len(records),
            },
            "target_strategy_in_model_input": False,
            "target_response_in_model_input": False,
        }
        _write_json(args.manifest_output, manifest)
        return 0

    train_records = _read_jsonl(args.train_prepared)
    valid_records = _read_jsonl(args.valid_prepared)
    raw_predictions = _read_jsonl(args.valid_predictions)
    result = select_configuration(
        train_records,
        valid_records,
        raw_predictions,
        history_orders=_int_values(args.history_orders),
        smoothings=_float_values(args.smoothings),
        transition_weights=_float_values(args.transition_weights),
        class_adjustment_taus=_float_values(args.class_adjustment_taus),
    )
    predictions = result.pop("best_predictions")
    _write_jsonl(args.predictions_output, predictions)
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["artifacts"] = {
        "train_prepared": {
            "path": str(args.train_prepared.resolve()),
            "sha256": sha256_file(args.train_prepared),
            "rows": len(train_records),
        },
        "valid_prepared": {
            "path": str(args.valid_prepared.resolve()),
            "sha256": sha256_file(args.valid_prepared),
            "rows": len(valid_records),
        },
        "valid_predictions": {
            "path": str(args.valid_predictions.resolve()),
            "sha256": sha256_file(args.valid_predictions),
            "rows": len(raw_predictions),
        },
        "selected_predictions": {
            "path": str(args.predictions_output.resolve()),
            "sha256": sha256_file(args.predictions_output),
            "rows": len(predictions),
        },
    }
    _write_json(args.summary_output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
