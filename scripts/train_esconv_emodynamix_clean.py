#!/usr/bin/env python3
"""Build the clean canonical train/dev inputs for an EmoDynamiX policy.

This entry point deliberately knows only the two author-published canonical
ESConv train/dev TSVs.  It never accepts a test split or a generic dataset
directory.  The current supporter response is parsed only long enough to
identify its strategy label and is never retained in a record or model input.

The default and currently only operation is a read-only data audit.  Model,
feature, and optimizer code is added behind later eval gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "tmp/official_benchmarks/esconv/codes/dataset"

OFFICIAL_SPLITS: dict[str, dict[str, Any]] = {
    "train": {
        "filename": "trainWithStrategy_short.tsv",
        "sha256": "0ecf37462f8e3fa7f1dc5bfbecd70733abb3501c3bf83b27a957bec5a926ec21",
        "rows": 8_562,
        "dialogues": 632,
        "mapping_sha256": (
            "ef6795ef6355909a684129eed027b9ee17afba50bec0175345efcd9a81599726"
        ),
    },
    "dev": {
        "filename": "devWithStrategy_short.tsv",
        "sha256": "625b511f40cf9a9285582e808e6bbb4c2f35d0b0cd063f3709db430b8d0c3bdc",
        "rows": 2_985,
        "dialogues": 211,
        "mapping_sha256": (
            "1de343795f59694f8ef298874621f6e4532e602ec41bf987516342980be13ee5"
        ),
    },
}

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
_CANONICAL_BY_CASEFOLD = {
    label.casefold(): label for label in CANONICAL_LABELS
}
_EMODYNAMIX_BY_CASEFOLD = {
    upstream.casefold(): canonical
    for canonical, upstream in CANONICAL_TO_EMODYNAMIX.items()
}

EXPECTED_CLEAN_CONTRACT = {
    "unique_overlap_count": 12,
    "overlap_set_sha256": (
        "c2499ae0abbfb3d492b3b9bbf2141d77ba7967bfc399db4092f93768c795a6c2"
    ),
    "removed_train_rows": 129,
    "removed_line_numbers_sha256": (
        "86655276f6f2456cdf49732ac00b3af2afd7c5e24b3cfaad74d3a008cfc8131b"
    ),
    "derived_train_rows": 8_433,
    "derived_train_mapping_sha256": (
        "28fed69839d9f0cf41c73957fdbb1a265f02e1b449a2dbbeb97ba72ba00ca810"
    ),
    "combined_unique_feature_keys": 11_366,
}

_SEGMENT = re.compile(
    r"^\s*(?P<loss>[01](?:\.0)?)\s+(?P<role>[01])\s+"
    r"(?P<turn>\d+)\s+(?P<body>.*)\s*$"
)
_STRATEGY_PREFIX = re.compile(r"^\[(?P<strategy>[^\]]+)\]\s*(?P<text>.*)$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

FEATURE_SCHEMA_VERSION = "esconv-emodynamix-causal-feature-v1"
FIXTURE_FEATURE_BACKEND = "deterministic_structural_fixture_v1"
VERIFIED_FEATURE_BACKEND = "verified_upstream_sddp_erc_v1"
FEATURE_ROW_FIELDS = {
    "schema_version",
    "model_input_sha256",
    "node_count",
    "parsed_dialogue",
    "upstream_erc_softmax_output",
    "feature_backend",
    "generator_manifest_sha256",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


FIXTURE_GENERATOR_SHA256 = canonical_json_sha256(
    {
        "backend": FIXTURE_FEATURE_BACKEND,
        "purpose": "shape_and_gradient_smoke_only",
        "selectable": False,
        "erc_vector": "deterministic_seven_way_probability_fixture",
        "discourse_edges": "root_plus_observed_order_continuation_fixture",
        "schema_version": FEATURE_SCHEMA_VERSION,
    }
)


def _normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def canonical_strategy(value: str) -> str | None:
    normalized = _normalize_space(str(value)).casefold()
    if normalized in _CANONICAL_BY_CASEFOLD:
        return _CANONICAL_BY_CASEFOLD[normalized]
    return _EMODYNAMIX_BY_CASEFOLD.get(normalized)


def _validate_split(split: str) -> None:
    if split not in OFFICIAL_SPLITS:
        raise ValueError("split must be exactly 'train' or 'dev'")


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
        history = prior_turns[-5:]
    if not history or len(history) > 5:
        raise ValueError(
            f"ESConv row {line_number} cannot form a one-to-five-turn history"
        )

    utterances: list[str] = []
    speaker_turn: list[str] = []
    strategy_history: list[int] = []
    for turn in history:
        utterances.append(str(turn["text"]))
        if turn["role"] is None:
            speaker_turn.append("None")
            strategy_history.append(-1)
        elif turn["role"] == 0:
            speaker_turn.append("seeker")
            strategy_history.append(-1)
        else:
            speaker_turn.append("supporter")
            upstream = CANONICAL_TO_EMODYNAMIX[str(turn["strategy"])]
            strategy_history.append(EMODYNAMIX_LABEL_TO_ID[upstream])
    return {
        "dialogue_history": " </s> ".join(utterances),
        "strategy_history": str(strategy_history),
        "speaker_turn": " ".join(speaker_turn),
    }


def build_emodynamix_records(
    lines: Iterable[str], *, split: str, expected_rows: int | None
) -> list[dict[str, Any]]:
    """Reconstruct causal author-compatible inputs without target responses."""

    _validate_split(split)
    records: list[dict[str, Any]] = []
    previous_target_turn: int | None = None
    conversation_number = 0
    turns: dict[int, dict[str, Any]] = {}

    for physical_line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        segments = _parse_tsv_line(raw_line, physical_line_number)
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
        upstream_gold = CANONICAL_TO_EMODYNAMIX[gold]
        records.append(
            {
                "item_id": f"esconv-{split}-{physical_line_number:06d}",
                "line_number": physical_line_number,
                "conversation_id": (
                    f"esconv-{split}-dialog-{conversation_number:04d}"
                ),
                "target_turn": target_turn,
                "source_kind": f"canonical_{split}_tsv",
                "source_line_sha256": sha256_text(raw_line.rstrip("\r\n")),
                "gold": gold,
                "gold_emodynamix": upstream_gold,
                "label_id": EMODYNAMIX_LABEL_TO_ID[upstream_gold],
                "model_input": model_input,
                "model_input_sha256": canonical_json_sha256(model_input),
            }
        )

        # The current target is history only for later prediction targets.
        _merge_turn(turns, target, physical_line_number)
        previous_target_turn = target_turn

    if expected_rows is not None and len(records) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} canonical {split} rows, found {len(records)}"
        )
    return records


def mapping_commitment(records: Sequence[dict[str, Any]]) -> str:
    return canonical_json_sha256(
        [
            {
                "item_id": record["item_id"],
                "conversation_id": record["conversation_id"],
                "target_turn": int(record["target_turn"]),
                "gold": record["gold"],
                "model_input_sha256": record["model_input_sha256"],
            }
            for record in records
        ]
    )


def load_official_split(
    path: Path, split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read, hash, decode, and parse one immutable train/dev byte snapshot."""

    _validate_split(split)
    path = Path(path)
    expected = OFFICIAL_SPLITS[split]
    if path.name != expected["filename"]:
        raise ValueError(
            f"official {split} filename must be {expected['filename']!r}"
        )
    if path.is_symlink():
        raise ValueError(f"official {split} file must not be a symlink")
    if not path.is_file():
        raise ValueError(f"official {split} file does not exist: {path}")

    payload = path.read_bytes()
    actual_hash = sha256_bytes(payload)
    if actual_hash != expected["sha256"]:
        raise ValueError(
            f"official {split} SHA-256 mismatch: "
            f"expected={expected['sha256']} actual={actual_hash}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"official {split} is not UTF-8") from error
    records = build_emodynamix_records(
        text.splitlines(keepends=True),
        split=split,
        expected_rows=int(expected["rows"]),
    )
    commitment = mapping_commitment(records)
    if commitment != expected["mapping_sha256"]:
        raise ValueError(
            f"official {split} causal mapping drift: "
            f"expected={expected['mapping_sha256']} actual={commitment}"
        )
    dialogue_count = len({record["conversation_id"] for record in records})
    if dialogue_count != expected["dialogues"]:
        raise ValueError(
            f"official {split} dialogue count drift: "
            f"expected={expected['dialogues']} actual={dialogue_count}"
        )
    history_nodes = Counter(
        len(record["model_input"]["speaker_turn"].split()) for record in records
    )
    return records, {
        "split": split,
        "filename": path.name,
        "sha256": actual_hash,
        "rows": len(records),
        "dialogues": dialogue_count,
        "unique_model_inputs": len(
            {record["model_input_sha256"] for record in records}
        ),
        "history_node_counts": {
            str(nodes): history_nodes[nodes] for nodes in sorted(history_nodes)
        },
        "mapping_sha256": commitment,
    }


def derive_train_without_dev_overlap(
    raw_train_records: Sequence[dict[str, Any]],
    dev_records: Sequence[dict[str, Any]],
    *,
    expected_contract: dict[str, Any] | None = EXPECTED_CLEAN_CONTRACT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove every train row whose complete Emo input appears in dev."""

    dev_hashes = {record["model_input_sha256"] for record in dev_records}
    overlap_hashes = sorted(
        {record["model_input_sha256"] for record in raw_train_records}
        & dev_hashes
    )
    removed = [
        record
        for record in raw_train_records
        if record["model_input_sha256"] in dev_hashes
    ]
    derived = [
        record
        for record in raw_train_records
        if record["model_input_sha256"] not in dev_hashes
    ]
    post_filter_overlap = (
        {record["model_input_sha256"] for record in derived} & dev_hashes
    )
    removed_line_numbers = [int(record["line_number"]) for record in removed]
    audit: dict[str, Any] = {
        "rule": (
            "remove every train row whose complete EmoDynamiX model-input "
            "SHA-256 occurs in canonical dev"
        ),
        "mandatory": True,
        "raw_train_rows": len(raw_train_records),
        "raw_dev_rows": len(dev_records),
        "unique_train_inputs": len(
            {record["model_input_sha256"] for record in raw_train_records}
        ),
        "unique_dev_inputs": len(dev_hashes),
        "unique_overlap_count": len(overlap_hashes),
        "unique_overlap_input_sha256": overlap_hashes,
        "overlap_set_sha256": canonical_json_sha256(overlap_hashes),
        "removed_train_rows": len(removed),
        "removed_train_line_numbers": removed_line_numbers,
        "removed_line_numbers_sha256": canonical_json_sha256(
            removed_line_numbers
        ),
        "derived_train_rows": len(derived),
        "post_filter_overlap_count": len(post_filter_overlap),
    }
    if all("item_id" in record for record in derived):
        audit["derived_train_mapping_sha256"] = mapping_commitment(derived)
    combined_unique = {
        record["model_input_sha256"] for record in derived
    } | dev_hashes
    audit["combined_unique_feature_keys"] = len(combined_unique)

    if expected_contract is not None:
        drift = {
            key: {"expected": expected, "actual": audit.get(key)}
            for key, expected in expected_contract.items()
            if audit.get(key) != expected
        }
        if drift:
            raise ValueError(
                "official train/dev EmoDynamiX overlap contract drift; "
                "refusing gradient data: "
                + json.dumps(drift, sort_keys=True)
            )
    if post_filter_overlap:
        raise AssertionError("derived train still intersects canonical dev")
    return derived, audit


def prepare_canonical_data(train_path: Path, dev_path: Path) -> dict[str, Any]:
    raw_train, train_audit = load_official_split(train_path, "train")
    dev, dev_audit = load_official_split(dev_path, "dev")
    train, overlap_audit = derive_train_without_dev_overlap(raw_train, dev)
    return {
        "raw_train_records": raw_train,
        "train_records": train,
        "dev_records": dev,
        "audit": overlap_audit,
        "source_audit": {"train": train_audit, "dev": dev_audit},
    }


def _require_hex_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def validate_feature_row(
    row: Any,
    *,
    expected_input_sha256: str | None = None,
    expected_generator_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one label-free, history-only precomputed feature row."""

    if not isinstance(row, dict):
        raise ValueError("feature row must be a JSON object")
    actual_fields = set(row)
    if actual_fields != FEATURE_ROW_FIELDS:
        missing = sorted(FEATURE_ROW_FIELDS - actual_fields)
        extra = sorted(actual_fields - FEATURE_ROW_FIELDS)
        raise ValueError(
            f"feature fields mismatch: missing={missing} extra={extra}"
        )
    if row["schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("feature schema version mismatch")

    input_sha = _require_hex_sha256(
        row["model_input_sha256"], "feature model input SHA"
    )
    if expected_input_sha256 is not None and input_sha != expected_input_sha256:
        raise ValueError(
            "feature model input SHA does not match the required causal input"
        )
    generator_sha = _require_hex_sha256(
        row["generator_manifest_sha256"], "generator manifest SHA"
    )
    if (
        expected_generator_manifest_sha256 is not None
        and generator_sha != expected_generator_manifest_sha256
    ):
        raise ValueError("feature generator manifest SHA mismatch")
    if row["feature_backend"] not in {
        FIXTURE_FEATURE_BACKEND,
        VERIFIED_FEATURE_BACKEND,
    }:
        raise ValueError("unknown feature backend")

    node_count = row["node_count"]
    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or not 1 <= node_count <= 5
    ):
        raise ValueError("feature node_count must be an integer from one to five")

    erc_rows = row["upstream_erc_softmax_output"]
    if not isinstance(erc_rows, list) or len(erc_rows) != node_count:
        raise ValueError("feature must have one ERC row per input node")
    for node_index, vector in enumerate(erc_rows):
        if not isinstance(vector, list) or len(vector) != 7:
            raise ValueError(
                f"ERC node {node_index} must contain exactly seven values"
            )
        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("ERC values must be finite numeric probabilities")
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError("ERC values must be finite probabilities in [0, 1]")
            values.append(numeric)
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"ERC node {node_index} probabilities must sum to one")

    parsed_dialogue = row["parsed_dialogue"]
    if not isinstance(parsed_dialogue, list):
        raise ValueError("parsed_dialogue must be a list")
    normalized_edges: list[tuple[int, int, int]] = []
    for edge in parsed_dialogue:
        if (
            not isinstance(edge, list)
            or len(edge) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in edge)
        ):
            raise ValueError("every discourse edge must contain three integers")
        head, tail, relation = edge
        if not 0 <= head <= node_count or not 1 <= tail <= node_count:
            raise ValueError("discourse edge endpoint is outside the input nodes")
        # SDDP emits 0..16. EmoDynamiX itself adds Self=17 and Inter=18 later.
        if not 0 <= relation <= 16:
            raise ValueError("SDDP discourse relation must be between 0 and 16")
        normalized_edges.append((head, tail, relation))
    if normalized_edges != sorted(normalized_edges):
        raise ValueError("discourse edges must be sorted canonically")
    if len(normalized_edges) != len(set(normalized_edges)):
        raise ValueError("discourse edges must be unique")
    return row


def make_structural_fixture_feature(
    model_input_sha256: str, model_input: dict[str, str]
) -> dict[str, Any]:
    """Create deterministic fake features for structural smoke tests only."""

    _require_hex_sha256(model_input_sha256, "model input SHA")
    if canonical_json_sha256(model_input) != model_input_sha256:
        raise ValueError("model input SHA does not match model_input bytes")
    if set(model_input) != {
        "dialogue_history",
        "strategy_history",
        "speaker_turn",
    }:
        raise ValueError("fixture model_input fields mismatch")
    speakers = model_input["speaker_turn"].split()
    utterances = [
        value.strip() for value in model_input["dialogue_history"].split("</s>")
    ]
    try:
        strategies = json.loads(model_input["strategy_history"])
    except json.JSONDecodeError as error:
        raise ValueError("fixture strategy_history is not a JSON list") from error
    if not (len(speakers) == len(utterances) == len(strategies)):
        raise ValueError("fixture input node fields have different lengths")
    node_count = len(speakers)
    if not 1 <= node_count <= 5:
        raise ValueError("fixture input must have one to five nodes")

    erc_rows: list[list[float]] = []
    for node_index in range(node_count):
        chosen = (int(model_input_sha256[node_index * 2 : node_index * 2 + 2], 16)
                  + node_index) % 7
        vector = [0.05] * 7
        vector[chosen] = 0.7
        erc_rows.append(vector)
    edges = [[0, 1, 16]]
    edges.extend([[node, node + 1, 0] for node in range(1, node_count)])
    row = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "model_input_sha256": model_input_sha256,
        "node_count": node_count,
        "parsed_dialogue": edges,
        "upstream_erc_softmax_output": erc_rows,
        "feature_backend": FIXTURE_FEATURE_BACKEND,
        "generator_manifest_sha256": FIXTURE_GENERATOR_SHA256,
    }
    return validate_feature_row(
        row,
        expected_input_sha256=model_input_sha256,
        expected_generator_manifest_sha256=FIXTURE_GENERATOR_SHA256,
    )


def feature_table_status(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("feature table is empty")
    if any(row.get("feature_backend") == FIXTURE_FEATURE_BACKEND for row in rows):
        return "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE"
    return "REQUIRES_VERIFIED_GENERATOR_RECEIPT"


def feature_table_commitment(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("feature table is empty")
    ordered = sorted(rows, key=lambda row: str(row.get("model_input_sha256", "")))
    keys: list[str] = []
    for row in ordered:
        validate_feature_row(row)
        keys.append(str(row["model_input_sha256"]))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate feature key in feature table")
    return canonical_json_sha256(ordered)


def load_feature_jsonl_bytes(
    payload: bytes,
    *,
    required_input_sha256: set[str],
    expected_generator_manifest_sha256: str,
    expected_table_sha256: str | None = None,
    allow_fixture: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate one immutable JSONL byte snapshot and close its key set."""

    _require_hex_sha256(
        expected_generator_manifest_sha256, "expected generator manifest SHA"
    )
    for digest in required_input_sha256:
        _require_hex_sha256(digest, "required model input SHA")
    if expected_table_sha256 is not None:
        _require_hex_sha256(expected_table_sha256, "expected feature commitment")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("feature JSONL is not UTF-8") from error

    by_key: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"feature JSONL line {line_number} is invalid JSON") from error
        validate_feature_row(
            row,
            expected_generator_manifest_sha256=(
                expected_generator_manifest_sha256
            ),
        )
        key = str(row["model_input_sha256"])
        if key in by_key:
            raise ValueError(f"duplicate feature key at JSONL line {line_number}: {key}")
        by_key[key] = row

    actual_keys = set(by_key)
    missing = sorted(required_input_sha256 - actual_keys)
    extra = sorted(actual_keys - required_input_sha256)
    if missing:
        raise ValueError(f"missing feature keys: {missing[:5]}")
    if extra:
        raise ValueError(f"extra feature keys: {extra[:5]}")
    rows = list(by_key.values())
    status = feature_table_status(rows)
    if status == "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE" and not allow_fixture:
        raise ValueError("fixture features require explicit smoke-only opt-in")
    commitment = feature_table_commitment(rows)
    if expected_table_sha256 is not None and commitment != expected_table_sha256:
        raise ValueError(
            "feature table commitment mismatch: "
            f"expected={expected_table_sha256} actual={commitment}"
        )
    return by_key, {
        "feature_rows": len(rows),
        "feature_table_sha256": commitment,
        "generator_manifest_sha256": expected_generator_manifest_sha256,
        "status": status,
        "payload_sha256": sha256_bytes(payload),
    }


def load_feature_jsonl(
    path: Path,
    *,
    required_input_sha256: set[str],
    expected_generator_manifest_sha256: str,
    expected_table_sha256: str,
    allow_fixture: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError("feature JSONL must not be a symlink")
    if not path.is_file():
        raise ValueError(f"feature JSONL does not exist: {path}")
    features, audit = load_feature_jsonl_bytes(
        path.read_bytes(),
        required_input_sha256=required_input_sha256,
        expected_generator_manifest_sha256=expected_generator_manifest_sha256,
        expected_table_sha256=expected_table_sha256,
        allow_fixture=allow_fixture,
    )
    audit["filename"] = path.name
    return features, audit


def resolve_record_features(
    records: Sequence[dict[str, Any]],
    features_by_input_sha256: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    for record in records:
        key = str(record["model_input_sha256"])
        if key not in features_by_input_sha256:
            raise ValueError(f"missing feature for record input: {key}")
        aligned.append(features_by_input_sha256[key])
    return aligned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of canonical ESConv train/dev inputs for clean "
            "EmoDynamiX retraining"
        )
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_DATA_ROOT / OFFICIAL_SPLITS["train"]["filename"],
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=DEFAULT_DATA_ROOT / OFFICIAL_SPLITS["dev"]["filename"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = prepare_canonical_data(args.train_file, args.dev_file)
    summary = {
        "status": "DATA_READY",
        "model_loaded": False,
        "optimizer_steps": 0,
        "frozen_test_accessed": False,
        "source_audit": prepared["source_audit"],
        "overlap_audit": prepared["audit"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
