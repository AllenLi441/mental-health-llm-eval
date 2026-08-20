#!/usr/bin/env python3
"""Auditable EmoDynamiX adapter for the frozen ESConv strategy benchmark.

The workflow is intentionally staged:

1. ``prepare`` converts either the frozen author TSV or EmoDynamiX's released
   ``test.pkl`` into a JSONL contract. Gold labels remain outside
   ``model_input`` and target responses are never written.
2. ``infer`` loads a caller-supplied third-party checkout and checkpoint, then
   passes only ``model_input`` to the model.
3. ``score`` joins predictions to the prepared gold labels and emits audited
   predictions plus a hash-rich summary.

No third-party repository, checkpoint, dataset, or generated JSONL is vendored
by this adapter.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import pickle
import platform
import random
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_response_eval.esconv_metrics import compute_classification_metrics


EXPECTED_FROZEN_ROWS = 2_775
EXPECTED_FROZEN_TEST_SHA256 = (
    "b85ae888bf747cefa54bba2a6c3e2f6ccb4c1005d4e0b6d1d3be3823cf040aef"
)
UPSTREAM_REPOSITORY = "https://github.com/cw-wan/EmoDynamiX-v2"

CANONICAL_LABELS = (
    "Questions",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Other",
)

# Frozen from cw-wan/EmoDynamiX-v2 data/esconv/strategies.json at audit time.
EMODYNAMIX_LABEL_TO_ID = {
    "Reflection of feelings": 0,
    "Self-disclosure": 1,
    "Question": 2,
    "Affirmation and Reassurance": 3,
    "Providing Suggestions": 4,
    "Restatement or Paraphrasing": 5,
    "Information": 6,
    "Others": 7,
}
EMODYNAMIX_ID_TO_LABEL = {
    label_id: label for label, label_id in EMODYNAMIX_LABEL_TO_ID.items()
}
CANONICAL_TO_EMODYNAMIX = {
    "Questions": "Question",
    "Restatement or Paraphrasing": "Restatement or Paraphrasing",
    "Reflection of feelings": "Reflection of feelings",
    "Self-disclosure": "Self-disclosure",
    "Affirmation and Reassurance": "Affirmation and Reassurance",
    "Providing Suggestions": "Providing Suggestions",
    "Information": "Information",
    "Other": "Others",
}
EMODYNAMIX_TO_CANONICAL = {
    upstream: canonical for canonical, upstream in CANONICAL_TO_EMODYNAMIX.items()
}

_CANONICAL_BY_CASEFOLD = {label.casefold(): label for label in CANONICAL_LABELS}
_EMODYNAMIX_BY_CASEFOLD = {
    label.casefold(): label for label in EMODYNAMIX_LABEL_TO_ID
}
_SEGMENT = re.compile(
    r"^\s*(?P<loss>[01](?:\.0)?)\s+(?P<role>[01])\s+"
    r"(?P<turn>\d+)\s+(?P<body>.*)\s*$"
)
_STRATEGY_PREFIX = re.compile(r"^\[(?P<strategy>[^\]]+)\]\s*(?P<text>.*)$")


def _normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def canonical_strategy(value: str) -> str | None:
    normalized = _normalize_space(str(value)).casefold()
    if normalized in _CANONICAL_BY_CASEFOLD:
        return _CANONICAL_BY_CASEFOLD[normalized]
    upstream = _EMODYNAMIX_BY_CASEFOLD.get(normalized)
    return EMODYNAMIX_TO_CANONICAL.get(upstream) if upstream else None


def normalize_prediction(value: Any) -> tuple[str | None, str | None]:
    """Return canonical and EmoDynamiX labels for an ID or label value."""

    if isinstance(value, bool):
        return None, None
    if isinstance(value, int):
        upstream = EMODYNAMIX_ID_TO_LABEL.get(value)
        return (EMODYNAMIX_TO_CANONICAL.get(upstream), upstream) if upstream else (None, None)
    if isinstance(value, str):
        stripped = _normalize_space(value)
        if stripped.isdigit():
            return normalize_prediction(int(stripped))
        canonical = canonical_strategy(stripped)
        if canonical is not None:
            return canonical, CANONICAL_TO_EMODYNAMIX[canonical]
    return None, None


def _parse_segment(raw_segment: str, line_number: int) -> dict[str, Any]:
    match = _SEGMENT.match(raw_segment)
    if not match:
        raise ValueError(
            f"ESConv row {line_number} has an invalid turn segment: "
            f"{raw_segment[:80]!r}"
        )
    role = int(match.group("role"))
    body = _normalize_space(match.group("body"))
    strategy = None
    if role == 1:
        strategy_match = _STRATEGY_PREFIX.match(body)
        if not strategy_match:
            raise ValueError(
                f"ESConv row {line_number} supporter turn lacks a strategy prefix"
            )
        strategy = canonical_strategy(strategy_match.group("strategy"))
        if strategy is None:
            raise ValueError(
                f"ESConv row {line_number} has unknown strategy "
                f"{strategy_match.group('strategy')!r}"
            )
        body = _normalize_space(strategy_match.group("text"))
    if not body:
        raise ValueError(f"ESConv row {line_number} contains an empty utterance")
    return {
        "role": role,
        "turn": int(match.group("turn")),
        "strategy": strategy,
        "text": body,
    }


def _parse_tsv_line(line: str, line_number: int) -> list[dict[str, Any]]:
    raw_segments = re.split(r"\s+EOS\s+", line.strip())
    if not raw_segments or not raw_segments[-1].strip():
        raise ValueError(f"ESConv row {line_number} is empty")
    segments = [_parse_segment(segment, line_number) for segment in raw_segments]
    target = segments[-1]
    if target["role"] != 1 or target["strategy"] is None:
        raise ValueError(f"ESConv row {line_number} target must be a supporter turn")
    return segments


def _merge_turn(
    turns: dict[int, dict[str, Any]], turn: dict[str, Any], line_number: int
) -> None:
    turn_number = int(turn["turn"])
    existing = turns.get(turn_number)
    if existing is not None and existing != turn:
        raise ValueError(
            f"ESConv row {line_number} conflicts with an earlier copy of turn "
            f"{turn_number}"
        )
    turns[turn_number] = turn


def _upstream_model_input(
    turns: dict[int, dict[str, Any]], target_turn: int, line_number: int
) -> dict[str, str]:
    prior_turns = [
        turn
        for turn_number, turn in sorted(turns.items())
        if turn_number < target_turn
    ]
    if target_turn < 5:
        history = [{"role": None, "strategy": None, "text": "<START>"}]
        history.extend(prior_turns)
    else:
        # A few frozen dialogues skip a raw turn number.  EmoDynamiX windows by
        # observed utterance position, so use the five most recent observable
        # prior turns rather than inventing text for a missing numeric index.
        history = prior_turns[-5:]
    if not history or len(history) > 5:
        raise ValueError(
            f"ESConv row {line_number} cannot form a one-to-five-turn history"
        )

    utterances = [turn["text"] for turn in history]
    speaker_turn = []
    strategy_history = []
    for turn in history:
        if turn["role"] is None:
            speaker_turn.append("None")
            strategy_history.append(-1)
        elif turn["role"] == 0:
            speaker_turn.append("seeker")
            strategy_history.append(-1)
        else:
            speaker_turn.append("supporter")
            upstream = CANONICAL_TO_EMODYNAMIX[turn["strategy"]]
            strategy_history.append(EMODYNAMIX_LABEL_TO_ID[upstream])
    return {
        "dialogue_history": " </s> ".join(utterances),
        "strategy_history": str(strategy_history),
        "speaker_turn": " ".join(speaker_turn),
    }


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_frozen_lines(
    lines: Iterable[str], expected_rows: int | None = EXPECTED_FROZEN_ROWS
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_target_turn: int | None = None
    conversation_number = 0
    turns: dict[int, dict[str, Any]] = {}

    for physical_line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        segments = _parse_tsv_line(line, physical_line_number)
        target = segments[-1]
        target_turn = int(target["turn"])
        if previous_target_turn is None or target_turn <= previous_target_turn:
            conversation_number += 1
            turns = {}

        for context_turn in segments[:-1]:
            if int(context_turn["turn"]) >= target_turn:
                raise ValueError(
                    f"ESConv row {physical_line_number} contains a non-prior context turn"
                )
            _merge_turn(turns, context_turn, physical_line_number)

        model_input = _upstream_model_input(
            turns, target_turn, physical_line_number
        )
        gold = str(target["strategy"])
        record = {
            "item_id": f"esconv-test-{physical_line_number:06d}",
            "line_number": physical_line_number,
            "conversation_id": (
                f"esconv-test-dialog-{conversation_number:04d}"
            ),
            "target_turn": target_turn,
            "source_kind": "frozen_tsv",
            "gold": gold,
            "gold_emodynamix": CANONICAL_TO_EMODYNAMIX[gold],
            "model_input": model_input,
            "model_input_sha256": _canonical_json_sha256(model_input),
        }
        records.append(record)

        # The current target becomes legitimate history only for later targets.
        _merge_turn(turns, target, physical_line_number)
        previous_target_turn = target_turn

    if expected_rows is not None and len(records) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} frozen ESConv rows, found {len(records)}"
        )
    return records


def prepare_frozen_file(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_FROZEN_TEST_SHA256:
        raise ValueError(
            "frozen ESConv test hash mismatch: "
            f"expected={EXPECTED_FROZEN_TEST_SHA256} actual={actual_hash}"
        )
    with path.open(encoding="utf-8") as handle:
        return prepare_frozen_lines(handle, expected_rows=EXPECTED_FROZEN_ROWS)


def _frozen_dialogue_sequences(
    lines: Iterable[str], expected_rows: int | None
) -> list[dict[str, Any]]:
    """Reconstruct dialogue text for split auditing, never for model inference."""

    dialogues: list[dict[str, Any]] = []
    turns: dict[int, dict[str, Any]] = {}
    row_count = 0
    previous_target_turn: int | None = None

    def finish_dialogue() -> None:
        if not turns:
            return
        dialogues.append(
            {
                "conversation_id": f"esconv-test-dialog-{len(dialogues) + 1:04d}",
                "utterances": [turn["text"] for _, turn in sorted(turns.items())],
                "rows": row_count,
            }
        )

    total_rows = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        segments = _parse_tsv_line(line, line_number)
        target_turn = int(segments[-1]["turn"])
        if previous_target_turn is None or target_turn <= previous_target_turn:
            finish_dialogue()
            turns = {}
            row_count = 0
        for turn in segments:
            _merge_turn(turns, turn, line_number)
        row_count += 1
        total_rows += 1
        previous_target_turn = target_turn
    finish_dialogue()

    if expected_rows is not None and total_rows != expected_rows:
        raise ValueError(
            f"expected {expected_rows} frozen ESConv rows, found {total_rows}"
        )
    return dialogues


def _normalized_utterances(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(_normalize_space(str(value)).casefold() for value in values)


def audit_split_contamination(
    frozen_lines: Iterable[str],
    author_dialogues: Any,
    *,
    expected_frozen_rows: int | None = EXPECTED_FROZEN_ROWS,
    expected_author_dialogues: int | None = 1_300,
    split_seed: int = 13,
) -> dict[str, Any]:
    """Replay EmoDynamiX's split and locate frozen dialogues in its raw corpus."""

    if not isinstance(author_dialogues, list):
        raise ValueError("EmoDynamiX ESConv.json must contain a list")
    if (
        expected_author_dialogues is not None
        and len(author_dialogues) != expected_author_dialogues
    ):
        raise ValueError(
            f"expected {expected_author_dialogues} author dialogues, "
            f"found {len(author_dialogues)}"
        )

    author_sequences = []
    for index, dialogue in enumerate(author_dialogues):
        raw_turns = dialogue.get("dialog") if isinstance(dialogue, dict) else None
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"author dialogue {index} has no dialog turns")
        contents = []
        for turn_number, turn in enumerate(raw_turns):
            if not isinstance(turn, dict) or "content" not in turn:
                raise ValueError(
                    f"author dialogue {index} turn {turn_number} has no content"
                )
            contents.append(turn["content"])
        author_sequences.append(_normalized_utterances(contents))

    shuffled_indices = list(range(len(author_sequences)))
    random.Random(split_seed).shuffle(shuffled_indices)
    valid_size = int(0.15 * len(shuffled_indices))
    test_size = int(0.15 * len(shuffled_indices))
    author_split_by_index: dict[int, str] = {}
    for position, dialogue_index in enumerate(shuffled_indices):
        if position < valid_size:
            split = "valid"
        elif position < valid_size + test_size:
            split = "test"
        else:
            split = "train"
        author_split_by_index[dialogue_index] = split

    frozen_dialogues = _frozen_dialogue_sequences(
        frozen_lines, expected_rows=expected_frozen_rows
    )
    matched_dialogues = {"train": 0, "valid": 0, "test": 0}
    matched_rows = {"train": 0, "valid": 0, "test": 0}
    unmatched_dialogues = 0
    unmatched_rows = 0
    ambiguous_dialogues = 0
    ambiguous_rows = 0
    dialogue_matches = []
    for frozen in frozen_dialogues:
        frozen_sequence = _normalized_utterances(frozen["utterances"])
        candidates = [
            index
            for index, author_sequence in enumerate(author_sequences)
            if author_sequence[: len(frozen_sequence)] == frozen_sequence
        ]
        match_record = {
            "conversation_id": frozen["conversation_id"],
            "frozen_rows": frozen["rows"],
            "frozen_turns": len(frozen_sequence),
        }
        if len(candidates) == 1:
            author_index = candidates[0]
            split = author_split_by_index[author_index]
            matched_dialogues[split] += 1
            matched_rows[split] += int(frozen["rows"])
            match_record.update(
                {
                    "status": "matched",
                    "author_dialogue_index": author_index,
                    "author_split": split,
                }
            )
        elif not candidates:
            unmatched_dialogues += 1
            unmatched_rows += int(frozen["rows"])
            match_record["status"] = "unmatched"
        else:
            ambiguous_dialogues += 1
            ambiguous_rows += int(frozen["rows"])
            match_record.update(
                {
                    "status": "ambiguous",
                    "candidate_author_dialogue_indices": candidates,
                }
            )
        dialogue_matches.append(match_record)

    train_overlap = matched_rows["train"] > 0
    eligibility_status = (
        "DIAGNOSTIC_ONLY_TRAIN_CONTAMINATED"
        if train_overlap
        else "DIAGNOSTIC_ONLY_AUTHOR_SPLIT_PROTOCOL"
    )
    return {
        "audit_status": "COMPLETE",
        "method": "casefold_whitespace_normalized_dialogue_prefix",
        "split_seed": split_seed,
        "split_rule": "random.shuffle(seed=13), valid=first 15%, test=next 15%, train=remaining 70%",
        "frozen_dialogues": len(frozen_dialogues),
        "frozen_rows": sum(int(dialogue["rows"]) for dialogue in frozen_dialogues),
        "author_dialogues": len(author_sequences),
        "author_split_dialogues": {
            "train": len(author_sequences) - valid_size - test_size,
            "valid": valid_size,
            "test": test_size,
        },
        "matched_dialogues_by_split": matched_dialogues,
        "matched_rows_by_split": matched_rows,
        "unmatched_dialogues": unmatched_dialogues,
        "unmatched_rows": unmatched_rows,
        "ambiguous_dialogues": ambiguous_dialogues,
        "ambiguous_rows": ambiguous_rows,
        "released_checkpoint_train_overlap": train_overlap,
        "frozen_leaderboard_eligible": False,
        "eligibility_status": eligibility_status,
        "dialogue_matches": dialogue_matches,
    }


def audit_split_contamination_files(
    *,
    frozen_test_source: Path,
    author_raw_json: Path,
    third_party_repo: Path,
) -> dict[str, Any]:
    frozen_test_source = Path(frozen_test_source)
    author_raw_json = Path(author_raw_json)
    frozen_hash = sha256_file(frozen_test_source)
    if frozen_hash != EXPECTED_FROZEN_TEST_SHA256:
        raise ValueError(
            "frozen ESConv test hash mismatch: "
            f"expected={EXPECTED_FROZEN_TEST_SHA256} actual={frozen_hash}"
        )
    with author_raw_json.open(encoding="utf-8") as handle:
        author_dialogues = json.load(handle)
    with frozen_test_source.open(encoding="utf-8") as handle:
        audit = audit_split_contamination(handle, author_dialogues)
    audit["frozen_source"] = {
        "path": str(frozen_test_source.resolve()),
        "sha256": frozen_hash,
    }
    audit["author_raw_dataset"] = {
        "path": str(author_raw_json.resolve()),
        "sha256": sha256_file(author_raw_json),
    }
    audit["third_party_repo"] = {
        "path": str(Path(third_party_repo).resolve()),
        "commit": git_commit(third_party_repo),
        "tree_sha": git_tree(third_party_repo),
    }
    return audit


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=repr)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    raise TypeError(f"unsupported preprocessed value type: {type(value).__name__}")


def prepare_author_pickle(
    path: Path, *, allow_unsafe_pickle: bool
) -> list[dict[str, Any]]:
    """Convert the released test.pkl; loading pickle requires explicit trust."""

    if not allow_unsafe_pickle:
        raise ValueError(
            "refusing unsafe pickle load; pass --allow-unsafe-pickle only after "
            "verifying the third-party file hash and provenance"
        )
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - explicit CLI opt-in above
    if not isinstance(payload, list):
        raise ValueError("EmoDynamiX test.pkl must contain a list")

    records = []
    required = {
        "dialogue_history",
        "strategy_history",
        "speaker_turn",
        "label",
        "parsed_dialogue",
        "erc_logits",
    }
    for index, source in enumerate(payload, start=1):
        if not isinstance(source, dict) or not required.issubset(source):
            missing = sorted(required - set(source) if isinstance(source, dict) else required)
            raise ValueError(f"EmoDynamiX test.pkl row {index} is missing {missing}")
        canonical, upstream = normalize_prediction(source["label"])
        if canonical is None or upstream is None:
            raise ValueError(
                f"EmoDynamiX test.pkl row {index} has invalid label {source['label']!r}"
            )
        model_input = {
            "dialogue_history": str(source["dialogue_history"]),
            "strategy_history": str(source["strategy_history"]),
            "speaker_turn": str(source["speaker_turn"]),
            "parsed_dialogue": _json_safe(source["parsed_dialogue"]),
            "erc_logits": _json_safe(source["erc_logits"]),
        }
        records.append(
            {
                "item_id": f"emodynamix-author-test-{index:06d}",
                "line_number": index,
                "source_kind": "author_test_pickle",
                "gold": canonical,
                "gold_emodynamix": upstream,
                "model_input": model_input,
                "model_input_sha256": _canonical_json_sha256(model_input),
            }
        )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> tuple[str, str]:
    path = Path(path)
    if path.is_file():
        return "file", sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = [item for item in path.rglob("*") if item.is_file()]
    for item in sorted(files, key=lambda candidate: candidate.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return "directory_tree", digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo)), "rev-parse", "HEAD"], text=True
    ).strip()


def git_tree(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo)), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def _unique_by_item_id(
    records: Sequence[dict[str, Any]], description: str
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        item_id = str(record.get("item_id") or "")
        if not item_id:
            raise ValueError(f"{description} row {index} has no item_id")
        if item_id in by_id:
            raise ValueError(f"{description} has duplicate item_id {item_id}")
        by_id[item_id] = record
    return by_id


def run_inference(
    prepared: Sequence[dict[str, Any]],
    predict_batch: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(prepared), batch_size):
        source_batch = prepared[start : start + batch_size]
        # This is the leakage boundary: the third-party callable receives only input.
        model_inputs = [dict(record["model_input"]) for record in source_batch]
        try:
            outputs = predict_batch(model_inputs)
            if len(outputs) != len(source_batch):
                raise ValueError(
                    f"predictor returned {len(outputs)} rows for {len(source_batch)} inputs"
                )
        except Exception as error:  # preserve terminal model failures as invalid rows
            outputs = [
                {"prediction": None, "error": f"{type(error).__name__}: {error}"}
                for _ in source_batch
            ]

        for source, output in zip(source_batch, outputs):
            raw_prediction = output.get("prediction", output.get("next_strategy"))
            canonical, upstream = normalize_prediction(raw_prediction)
            error = output.get("error")
            record = {
                "item_id": source["item_id"],
                "line_number": source.get("line_number"),
                "raw_prediction": raw_prediction,
                "prediction": canonical,
                "prediction_emodynamix": upstream,
                "invalid": bool(error) or canonical is None,
                "error": error,
            }
            if source.get("conversation_id") is not None:
                record["conversation_id"] = source["conversation_id"]
            if "logits" in output:
                record["logits"] = _json_safe(output["logits"])
            predictions.append(record)
    return predictions


def score_predictions(
    prepared: Sequence[dict[str, Any]], raw_predictions: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared_by_id = _unique_by_item_id(prepared, "prepared input")
    raw_by_id = _unique_by_item_id(raw_predictions, "predictions")
    if set(prepared_by_id) != set(raw_by_id):
        missing = sorted(set(prepared_by_id) - set(raw_by_id))
        extra = sorted(set(raw_by_id) - set(prepared_by_id))
        raise ValueError(
            "prediction item IDs do not match prepared input: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    audited = []
    for source in prepared:
        raw = raw_by_id[source["item_id"]]
        raw_value = raw.get("prediction", raw.get("raw_prediction"))
        canonical, upstream = normalize_prediction(raw_value)
        error = raw.get("error")
        invalid = bool(raw.get("invalid")) or bool(error) or canonical is None
        gold = source.get("gold")
        if gold not in CANONICAL_LABELS:
            raise ValueError(f"prepared input has invalid gold label {gold!r}")
        record = {
            "item_id": source["item_id"],
            "line_number": source.get("line_number"),
            "gold": gold,
            "gold_emodynamix": CANONICAL_TO_EMODYNAMIX[gold],
            "prediction": canonical,
            "prediction_emodynamix": upstream,
            "correct": not invalid and canonical == gold,
            "invalid": invalid,
            "error": error,
            "model_input_sha256": source.get("model_input_sha256"),
        }
        if source.get("conversation_id") is not None:
            record["conversation_id"] = source["conversation_id"]
        if "logits" in raw:
            record["logits"] = _json_safe(raw["logits"])
        audited.append(record)
    return audited, compute_metrics(audited)


def compute_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return compute_classification_metrics(records, CANONICAL_LABELS)


def _eligibility_for_run(
    *,
    source_kind: str,
    checkpoint_training_provenance: str,
    dataset_hash: str,
    contamination_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    if source_kind != "frozen_tsv":
        return {
            "frozen_leaderboard_eligible": False,
            "status": "AUTHOR_TEST_REPRODUCTION_ONLY",
            "reason": "The released author test.pkl is not the frozen 2,775-row protocol.",
        }
    if checkpoint_training_provenance == "author_seed13_1300":
        if contamination_audit is None:
            raise ValueError(
                "a contamination audit is required for an author_seed13_1300 "
                "checkpoint on the frozen TSV"
            )
        if contamination_audit.get("audit_status") != "COMPLETE":
            raise ValueError("contamination audit is not complete")
        audited_hash = contamination_audit.get("frozen_source", {}).get("sha256")
        if audited_hash != dataset_hash:
            raise ValueError(
                "contamination audit frozen source hash does not match test source"
            )
        train_rows = int(
            contamination_audit.get("matched_rows_by_split", {}).get("train", 0)
        )
        status = (
            "DIAGNOSTIC_ONLY_TRAIN_CONTAMINATED"
            if train_rows
            else "DIAGNOSTIC_ONLY_AUTHOR_SPLIT_PROTOCOL"
        )
        return {
            "frozen_leaderboard_eligible": False,
            "status": status,
            "reason": (
                f"The checkpoint's author seed-13 training split overlaps "
                f"{train_rows} frozen evaluation rows."
                if train_rows
                else "The released checkpoint follows the incompatible author seed-13 split."
            ),
        }
    if checkpoint_training_provenance == "frozen_train_dev_only":
        return {
            "frozen_leaderboard_eligible": True,
            "status": "ELIGIBLE_SELF_DECLARED_FROZEN_TRAIN_DEV_ONLY",
            "reason": (
                "The run declares that checkpoint training and selection used only "
                "the frozen train/dev partitions; independent provenance review remains required."
            ),
        }
    if checkpoint_training_provenance == "unknown":
        return {
            "frozen_leaderboard_eligible": False,
            "status": "DIAGNOSTIC_ONLY_UNKNOWN_PROVENANCE",
            "reason": "Checkpoint training-data provenance is unknown.",
        }
    raise ValueError(
        f"unsupported checkpoint training provenance {checkpoint_training_provenance!r}"
    )


def build_summary(
    *,
    run_name: str,
    source_kind: str,
    test_source: Path,
    prepared_input: Path,
    predictions_output: Path,
    third_party_repo: Path,
    checkpoint: Path,
    metrics: dict[str, Any],
    checkpoint_training_provenance: str,
    contamination_audit: dict[str, Any] | None = None,
    contamination_audit_path: Path | None = None,
) -> dict[str, Any]:
    dataset_hash = sha256_file(test_source)
    eligibility = _eligibility_for_run(
        source_kind=source_kind,
        checkpoint_training_provenance=checkpoint_training_provenance,
        dataset_hash=dataset_hash,
        contamination_audit=contamination_audit,
    )
    checkpoint_kind, checkpoint_hash = sha256_path(checkpoint)
    summary = {
        "audit_status": "COMPLETE",
        "run_name": run_name,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "paper": "EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics",
        "task": "8-way next-support-strategy prediction",
        "source_kind": source_kind,
        "protocol": {
            "frozen_2775": source_kind == "frozen_tsv",
            "target_response_excluded_from_model_input": True,
            "target_strategy_excluded_from_model_input": True,
            "history_window": "EmoDynamiX-compatible last five utterances; <START> inserted for early turns",
            "invalid_policy": "missing, errored, or out-of-label predictions count wrong",
            "checkpoint_training_provenance": checkpoint_training_provenance,
        },
        "dataset": {
            "path": str(Path(test_source).resolve()),
            "sha256": dataset_hash,
            "rows": metrics["total"],
        },
        "prepared_input": {
            "path": str(Path(prepared_input).resolve()),
            "sha256": sha256_file(prepared_input),
        },
        "third_party_repo": {
            "declared_upstream": UPSTREAM_REPOSITORY,
            "path": str(Path(third_party_repo).resolve()),
            "commit": git_commit(third_party_repo),
            "tree_sha": git_tree(third_party_repo),
        },
        "checkpoint": {
            "path": str(Path(checkpoint).resolve()),
            "kind": checkpoint_kind,
            "sha256": checkpoint_hash,
        },
        "label_contract": {
            "canonical_labels": list(CANONICAL_LABELS),
            "canonical_to_emodynamix": CANONICAL_TO_EMODYNAMIX,
            "emodynamix_label_to_id": EMODYNAMIX_LABEL_TO_ID,
        },
        "metrics": metrics,
        "eligibility": eligibility,
        "predictions": {
            "path": str(Path(predictions_output).resolve()),
            "sha256": sha256_file(predictions_output),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "adapter_repo_commit": git_commit(ROOT),
        },
    }
    if contamination_audit is not None:
        audit_copy = json.loads(json.dumps(contamination_audit))
        if contamination_audit_path is not None:
            audit_copy["artifact"] = {
                "path": str(Path(contamination_audit_path).resolve()),
                "sha256": sha256_file(contamination_audit_path),
            }
        summary["contamination_audit"] = audit_copy
    return summary


@contextlib.contextmanager
def _third_party_import_context(repo: Path):
    repo = Path(repo).resolve()
    previous_directory = Path.cwd()
    repo_string = str(repo)
    sys.path.insert(0, repo_string)
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(previous_directory)
        if sys.path and sys.path[0] == repo_string:
            sys.path.pop(0)


def make_third_party_predictor(
    *,
    third_party_repo: Path,
    checkpoint: Path,
    use_preprocessed: bool,
    device_name: str,
    allow_unsafe_checkpoint: bool,
) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    if not allow_unsafe_checkpoint:
        raise ValueError(
            "refusing third-party torch checkpoint load; verify its hash and pass "
            "--allow-unsafe-checkpoint"
        )
    third_party_repo = Path(third_party_repo).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not (third_party_repo / "modules/roberta/model.py").is_file():
        raise FileNotFoundError(
            f"not an EmoDynamiX checkout: {third_party_repo}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    with _third_party_import_context(third_party_repo):
        torch = importlib.import_module("torch")
        module = importlib.import_module("modules.roberta")
        model_class = module.RobertaHeterogeneousGraph
        args = SimpleNamespace(
            dataset="esconv-preprocessed" if use_preprocessed else "esconv",
            exclude_others=0,
            erc_temperature=0.5,
            erc_mixed=1,
            hg_dim=512,
        )
        model = model_class(args, lightmode=use_preprocessed)
        model.load(str(checkpoint))

    if dict(model.strategy2id) != EMODYNAMIX_LABEL_TO_ID:
        raise RuntimeError(
            "third-party strategy mapping differs from the frozen adapter contract"
        )
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    device = torch.device(device_name)
    model.to(device)
    model.device = device
    model.eval()

    def predict(model_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batch: dict[str, Any] = {
            "dialogue_history": [item["dialogue_history"] for item in model_inputs],
            "strategy_history": [item["strategy_history"] for item in model_inputs],
            "speaker_turn": [item["speaker_turn"] for item in model_inputs],
        }
        if use_preprocessed:
            batch["parsed_dialogue"] = [
                item["parsed_dialogue"] for item in model_inputs
            ]
            tensors = [
                torch.tensor(item["erc_logits"], dtype=torch.float32)
                for item in model_inputs
            ]
            batch["erc_logits"] = torch.cat(tensors, dim=0)
        with torch.no_grad():
            output = model(batch)
            logits = output["logits"].detach().cpu()
        predictions = logits.argmax(dim=-1).tolist()
        return [
            {
                "prediction": model.id2strategy[int(label_id)],
                "logits": {
                    EMODYNAMIX_ID_TO_LABEL[index]: float(score)
                    for index, score in enumerate(row.tolist())
                },
            }
            for label_id, row in zip(predictions, logits)
        ]

    return predict


def _prepare_manifest(
    source_kind: str,
    test_source: Path,
    output: Path,
    records: Sequence[dict[str, Any]],
    third_party_repo: Path | None,
) -> dict[str, Any]:
    manifest = {
        "source_kind": source_kind,
        "source": {
            "path": str(Path(test_source).resolve()),
            "sha256": sha256_file(test_source),
        },
        "prepared_output": {
            "path": str(Path(output).resolve()),
            "sha256": sha256_file(output),
            "rows": len(records),
        },
        "input_contract": {
            "target_response_in_model_input": False,
            "target_strategy_in_model_input": False,
            "canonical_to_emodynamix": CANONICAL_TO_EMODYNAMIX,
            "emodynamix_label_to_id": EMODYNAMIX_LABEL_TO_ID,
        },
    }
    if third_party_repo is not None:
        manifest["third_party_repo"] = {
            "path": str(Path(third_party_repo).resolve()),
            "commit": git_commit(third_party_repo),
            "tree_sha": git_tree(third_party_repo),
        }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--source-kind",
        choices=("frozen_tsv", "author_test_pickle"),
        required=True,
    )
    prepare.add_argument("--test-source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest-output", type=Path)
    prepare.add_argument("--third-party-repo", type=Path)
    prepare.add_argument("--allow-unsafe-pickle", action="store_true")

    contamination = subparsers.add_parser("audit-contamination")
    contamination.add_argument("--frozen-test-source", type=Path, required=True)
    contamination.add_argument("--third-party-repo", type=Path, required=True)
    contamination.add_argument(
        "--author-raw-json",
        type=Path,
        help="Defaults to THIRD_PARTY_REPO/data/esconv/ESConv.json",
    )
    contamination.add_argument("--output", type=Path, required=True)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--prepared-input", type=Path, required=True)
    infer.add_argument("--third-party-repo", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--batch-size", type=int, default=1)
    infer.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    infer.add_argument("--allow-unsafe-checkpoint", action="store_true")

    score = subparsers.add_parser("score")
    score.add_argument(
        "--source-kind",
        choices=("frozen_tsv", "author_test_pickle"),
        required=True,
    )
    score.add_argument("--test-source", type=Path, required=True)
    score.add_argument("--prepared-input", type=Path, required=True)
    score.add_argument("--prediction-input", type=Path, required=True)
    score.add_argument("--third-party-repo", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--run-name", required=True)
    score.add_argument(
        "--checkpoint-training-provenance",
        choices=("author_seed13_1300", "frozen_train_dev_only", "unknown"),
        required=True,
    )
    score.add_argument("--contamination-audit", type=Path)
    score.add_argument("--summary-output", type=Path, required=True)
    score.add_argument("--predictions-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        if args.source_kind == "frozen_tsv":
            records = prepare_frozen_file(args.test_source)
        else:
            records = prepare_author_pickle(
                args.test_source,
                allow_unsafe_pickle=args.allow_unsafe_pickle,
            )
        write_jsonl(args.output, records)
        manifest = _prepare_manifest(
            args.source_kind,
            args.test_source,
            args.output,
            records,
            args.third_party_repo,
        )
        if args.manifest_output:
            args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
            args.manifest_output.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "audit-contamination":
        author_raw_json = args.author_raw_json or (
            args.third_party_repo / "data/esconv/ESConv.json"
        )
        audit = audit_split_contamination_files(
            frozen_test_source=args.frozen_test_source,
            author_raw_json=author_raw_json,
            third_party_repo=args.third_party_repo,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "infer":
        prepared = read_jsonl(args.prepared_input)
        use_preprocessed_values = {
            "parsed_dialogue" in record.get("model_input", {})
            and "erc_logits" in record.get("model_input", {})
            for record in prepared
        }
        if len(use_preprocessed_values) != 1:
            raise ValueError("prepared input mixes raw and preprocessed records")
        use_preprocessed = next(iter(use_preprocessed_values), False)
        predictor = make_third_party_predictor(
            third_party_repo=args.third_party_repo,
            checkpoint=args.checkpoint,
            use_preprocessed=use_preprocessed,
            device_name=args.device,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
        )
        predictions = run_inference(
            prepared, predictor, batch_size=args.batch_size
        )
        write_jsonl(args.output, predictions)
        print(args.output)
        return

    prepared = read_jsonl(args.prepared_input)
    raw_predictions = read_jsonl(args.prediction_input)
    audited, metrics = score_predictions(prepared, raw_predictions)
    contamination_audit = None
    if args.contamination_audit is not None:
        contamination_audit = json.loads(
            args.contamination_audit.read_text(encoding="utf-8")
        )
        if not isinstance(contamination_audit, dict):
            raise ValueError("contamination audit must be a JSON object")
    _eligibility_for_run(
        source_kind=args.source_kind,
        checkpoint_training_provenance=args.checkpoint_training_provenance,
        dataset_hash=sha256_file(args.test_source),
        contamination_audit=contamination_audit,
    )
    write_jsonl(args.predictions_output, audited)
    summary = build_summary(
        run_name=args.run_name,
        source_kind=args.source_kind,
        test_source=args.test_source,
        prepared_input=args.prepared_input,
        predictions_output=args.predictions_output,
        third_party_repo=args.third_party_repo,
        checkpoint=args.checkpoint,
        metrics=metrics,
        checkpoint_training_provenance=args.checkpoint_training_provenance,
        contamination_audit=contamination_audit,
        contamination_audit_path=args.contamination_audit,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Summary: {args.summary_output}")
    print(f"Predictions: {args.predictions_output}")


if __name__ == "__main__":
    main()
