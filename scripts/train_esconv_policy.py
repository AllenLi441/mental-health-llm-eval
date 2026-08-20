#!/usr/bin/env python3
"""Train an auditable train/dev-only RoBERTa ESConv strategy policy.

The program accepts exactly the original author ``train`` and ``dev`` TSV
files.  Their filenames, byte hashes, and row counts are fixed below.  There
is deliberately no evaluation/holdout argument or generic dataset directory.

Each TSV row is parsed structurally: all turns before the final supporter turn
become model input, the final strategy annotation becomes the label, and the
final supporter response is discarded before a training record is created.
The default invocation is a read-only audit.  ``--execute`` is required to
load a model or write a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_response_eval.esconv_metrics import (  # noqa: E402
    compute_classification_metrics,
)


PROTOCOL_ID = "esconv-policy-roberta-v1"
OFFICIAL_SPLITS: dict[str, dict[str, Any]] = {
    "train": {
        "filename": "trainWithStrategy_short.tsv",
        "sha256": "0ecf37462f8e3fa7f1dc5bfbecd70733abb3501c3bf83b27a957bec5a926ec21",
        "rows": 8_562,
    },
    "dev": {
        "filename": "devWithStrategy_short.tsv",
        "sha256": "625b511f40cf9a9285582e808e6bbb4c2f35d0b0cd063f3709db430b8d0c3bdc",
        "rows": 2_985,
    },
}
EXPECTED_DEV_INPUT_OVERLAP_SHA256 = (
    "3358ea28648260fd2f0aed0819cdfac9b55d44e55c6a9eead8dd28478dd3751d",
    "3dd472de131aea43923eafc47114def5bebeff71eed52de7ebd2c5a42cbcfb44",
    "4ab4aedefae16b8912287b3440a0325ca219c0ec33c704e9465c9bb729f2289e",
    "58a8f9e38ce5dcea33f500e22442e5c42ba134cbeebb7b1240d95a9c62e5bc4e",
    "62a369522f0374e2c1e39c23ffc01b84cb85ec464210a9ea7e545623906a851a",
    "75635a3047863753ee3993183f260d478bd5ce202d59548bcb83821d26c1384f",
    "89812797e3f4500b0cb41157be48c90ad549a57e5a7e556322ab9c07c61ae3a6",
    "9ad0444929093d274cc1bc0bac2886c5581fb5e5b4811e8cf9f69fb8e17cc65e",
    "b1a29810b05b1a1274a96612b9865386567d46513d77961df98ab575bfa8ee2a",
    "d2ac838857de6b685570a1e894f787be8f2637b956433b2e5d4228a5a86d2930",
    "d47c492b14a0d576a2447e5e3630bd6b2cf0f9789718131d7b5c0c017418798a",
    "f60d89294c3747ef88dcac250145dc0789423aec05b96d0ed784afad40a45341",
)
EXPECTED_DEV_INPUT_OVERLAP_SET_SHA256 = (
    "b9e1d891f736f1588d6386214413862c43b812c69466589b7f259e2e6d30f6cd"
)
EXPECTED_REMOVED_TRAIN_ROWS = 129
EXPECTED_DERIVED_TRAIN_ROWS = 8_433
EXPECTED_DERIVED_TRAIN_RECORDS_SHA256 = (
    "55c098b7a1cf1c9c9c9e8c9da49d95d4c8484552fbcc3d18319affda353b3480"
)
EXPECTED_EXACT_TSV_OVERLAP_SHA256 = (
    "77be9b60fae1f61a2bb3c1c733aaaad24b95bd4b505c4c45b43123727c80531c",
    "f1085cc07bcb4b13f49c732932b1f9a1a22cac9c6a77180872b021f92f626b0a",
)
EXPECTED_REMOVED_LINE_NUMBERS_SHA256 = (
    "86655276f6f2456cdf49732ac00b3af2afd7c5e24b3cfaad74d3a008cfc8131b"
)
EXPECTED_REMOVED_RECORDS_SHA256 = (
    "c09d39b627f6ff9a5848354bf1deed36ae35dafe8d8535fcc6ee5d8ed06e31f5"
)
LABELS = (
    "Questions",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Other",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ALLOWED_MODEL_TYPES = {"roberta", "xlm-roberta"}
LOSS_MODES = ("ce", "class_balanced", "logit_adjusted")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SEGMENT = re.compile(
    r"^\s*(?P<loss>[01](?:\.0)?)\s+(?P<role>[01])\s+"
    r"(?P<turn>\d+)\s+(?P<body>.*?)\s*$"
)
STRATEGY_PREFIX = re.compile(r"^\[(?P<strategy>[^\]]+)\]\s*(?P<text>.*)$")
MODEL_TREE_IGNORED_PARTS = {".git", ".cache", "__pycache__"}
MODEL_TREE_IGNORED_NAMES = {".DS_Store"}
CHECKPOINT_NAME = "best_model_state.pt"
MANIFEST_NAME = "best_checkpoint_manifest.json"

# These are the first pilot's frozen optimization defaults.  Changing any of
# them is allowed explicitly and is recorded as a developmental/non-frozen run.
FROZEN_PILOT_DEFAULTS = {
    "max_length": 256,
    "truncation_side": "left",
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "eval_batch_size": 16,
    "seed": 42,
    "class_balance_beta": 0.999,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit(repo: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_split_name(split: str) -> None:
    if split not in OFFICIAL_SPLITS:
        raise ValueError("ESConv policy training accepts train/dev splits only")


def verify_official_split_file(path: Path, split: str) -> dict[str, Any]:
    """Verify the immutable author TSV contract before parsing any examples."""

    _validate_split_name(split)
    path = Path(path)
    spec = OFFICIAL_SPLITS[split]
    if path.name != spec["filename"]:
        raise ValueError(
            f"official {split} filename mismatch: expected {spec['filename']!r}, "
            f"found {path.name!r}"
        )
    if path.is_symlink():
        raise ValueError(f"official {split} file must not be a symlink")
    if not path.is_file():
        raise ValueError(f"official {split} file does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != spec["sha256"]:
        raise ValueError(
            f"official {split} hash mismatch: expected={spec['sha256']} "
            f"actual={actual_hash}"
        )
    with path.open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != spec["rows"]:
        raise ValueError(
            f"official {split} row-count mismatch: expected={spec['rows']} actual={rows}"
        )
    return {
        "split": split,
        "filename": path.name,
        "path": str(path.resolve()),
        "sha256": actual_hash,
        "expected_sha256": spec["sha256"],
        "rows": rows,
        "expected_rows": spec["rows"],
    }


def _normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def _parse_segment(raw_segment: str, *, target: bool) -> dict[str, Any]:
    match = SEGMENT.match(raw_segment)
    if match is None:
        raise ValueError(f"invalid turn segment {raw_segment[:80]!r}")
    role = int(match.group("role"))
    turn = int(match.group("turn"))
    body = _normalize_space(match.group("body"))
    strategy = None
    text = body
    if role == 1:
        strategy_match = STRATEGY_PREFIX.match(body)
        if strategy_match is None:
            raise ValueError("supporter turn lacks a leading strategy annotation")
        strategy = strategy_match.group("strategy")
        if strategy not in LABEL_TO_ID:
            raise ValueError(f"unknown ESConv strategy {strategy!r}")
        text = _normalize_space(strategy_match.group("text"))
    if target and role != 1:
        raise ValueError("target is not a supporter turn")
    if not text:
        raise ValueError("empty utterance")
    parsed: dict[str, Any] = {"role": role, "turn": turn, "strategy": strategy}
    if not target:
        parsed["text"] = text
    return parsed


def _format_prior_turns(turns: Sequence[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for turn in turns:
        if turn["role"] == 0:
            rendered.append(f"Seeker: {turn['text']}")
        elif turn["role"] == 1 and turn["strategy"] in LABEL_TO_ID:
            rendered.append(
                f"Supporter [{turn['strategy']}]: {turn['text']}"
            )
        else:
            raise ValueError("history contains an invalid role or strategy")
    if not rendered:
        raise ValueError("history is empty")
    return "\n".join(rendered)


def parse_split_lines(
    lines: Iterable[str],
    *,
    split: str,
    expected_rows: int | None,
) -> list[dict[str, Any]]:
    """Create strategy records without retaining the final response text."""

    _validate_split_name(split)
    source_lines = [line for line in lines if line.strip()]
    if expected_rows is not None and len(source_lines) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} {split} rows, found {len(source_lines)}"
        )

    records: list[dict[str, Any]] = []
    previous_target_turn: int | None = None
    conversation_number = 0
    for item_id, raw_line in enumerate(source_lines):
        line_number = item_id + 1
        segments = re.split(r"\s+EOS\s+", raw_line.strip())
        if len(segments) < 2:
            raise ValueError(f"{split} line {line_number} has no history/target boundary")
        target = _parse_segment(segments[-1], target=True)
        target_turn = int(target["turn"])
        prior_turns = [
            _parse_segment(segment, target=False) for segment in segments[:-1]
        ]
        if any(int(turn["turn"]) >= target_turn for turn in prior_turns):
            raise ValueError(
                f"{split} line {line_number} context contains a non-prior turn"
            )
        if previous_target_turn is None or target_turn <= previous_target_turn:
            conversation_number += 1
        previous_target_turn = target_turn
        input_text = _format_prior_turns(prior_turns)
        label = str(target["strategy"])
        records.append(
            {
                "item_id": item_id,
                "line_number": line_number,
                "conversation_id": (
                    f"esconv-{split}-{conversation_number:04d}"
                ),
                "target_turn": target_turn,
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "input_text": input_text,
                "input_sha256": sha256_text(input_text),
                "source_line_sha256": sha256_text(raw_line.rstrip("\r\n")),
            }
        )
    return records


def records_commitment(records: Sequence[dict[str, Any]]) -> str:
    return canonical_json_sha256(
        [
            {
                "source_line_sha256": record["source_line_sha256"],
                "input_sha256": record["input_sha256"],
                "label": record["label"],
                "conversation_id": record["conversation_id"],
            }
            for record in records
        ]
    )


def load_official_split(
    path: Path, split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = verify_official_split_file(path, split)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        records = parse_split_lines(
            handle, split=split, expected_rows=OFFICIAL_SPLITS[split]["rows"]
        )
    label_counts = Counter(record["label"] for record in records)
    missing = [label for label in LABELS if label_counts[label] == 0]
    if missing:
        raise ValueError(f"official {split} is missing declared labels: {missing}")
    audit["conversation_count"] = len(
        {record["conversation_id"] for record in records}
    )
    audit["label_counts"] = {label: label_counts[label] for label in LABELS}
    audit["records_commitment_sha256"] = records_commitment(records)
    return records, audit


def derive_train_without_dev_overlap(
    raw_train_records: Sequence[dict[str, Any]],
    dev_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail closed while deriving the only records permitted for gradients."""

    dev_input_hashes = {record["input_sha256"] for record in dev_records}
    overlap_hashes = tuple(
        sorted(
            {record["input_sha256"] for record in raw_train_records}
            & dev_input_hashes
        )
    )
    removed = [
        record
        for record in raw_train_records
        if record["input_sha256"] in dev_input_hashes
    ]
    derived = [
        record
        for record in raw_train_records
        if record["input_sha256"] not in dev_input_hashes
    ]
    exact_tsv_overlap = tuple(
        sorted(
            {record["source_line_sha256"] for record in raw_train_records}
            & {record["source_line_sha256"] for record in dev_records}
        )
    )
    overlap_set_hash = canonical_json_sha256(list(overlap_hashes))
    derived_hash = records_commitment(derived)
    removed_line_numbers = [int(record["line_number"]) for record in removed]
    removed_record_fingerprints = [
        {
            "line_number": int(record["line_number"]),
            "source_line_sha256": record["source_line_sha256"],
            "input_sha256": record["input_sha256"],
            "label": record["label"],
        }
        for record in removed
    ]
    expected = {
        "unique_overlap_input_sha256": EXPECTED_DEV_INPUT_OVERLAP_SHA256,
        "overlap_input_set_sha256": EXPECTED_DEV_INPUT_OVERLAP_SET_SHA256,
        "removed_train_rows": EXPECTED_REMOVED_TRAIN_ROWS,
        "derived_train_rows": EXPECTED_DERIVED_TRAIN_ROWS,
        "derived_train_records_sha256": EXPECTED_DERIVED_TRAIN_RECORDS_SHA256,
        "exact_tsv_overlap_sha256": EXPECTED_EXACT_TSV_OVERLAP_SHA256,
    }
    actual = {
        "unique_overlap_input_sha256": overlap_hashes,
        "overlap_input_set_sha256": overlap_set_hash,
        "removed_train_rows": len(removed),
        "derived_train_rows": len(derived),
        "derived_train_records_sha256": derived_hash,
        "exact_tsv_overlap_sha256": exact_tsv_overlap,
    }
    drift = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if drift:
        raise ValueError(
            "official train/dev overlap contract drift; refusing gradient data: "
            + json.dumps(drift, sort_keys=True)
        )
    if {record["input_sha256"] for record in derived} & dev_input_hashes:
        raise AssertionError("derived train still intersects development inputs")

    train_counts = Counter(record["input_sha256"] for record in raw_train_records)
    dev_counts = Counter(record["input_sha256"] for record in dev_records)
    derived_label_counts = Counter(record["label"] for record in derived)
    audit = {
        "rule": "remove every train row whose input_sha256 occurs in official dev",
        "mandatory": True,
        "disable_option_exists": False,
        "raw_train_rows": len(raw_train_records),
        "raw_dev_rows": len(dev_records),
        "unique_train_inputs": len(train_counts),
        "unique_dev_inputs": len(dev_counts),
        "unique_overlap_count": len(overlap_hashes),
        "unique_overlap_input_sha256": list(overlap_hashes),
        "overlap_input_set_sha256": overlap_set_hash,
        "overlap_details": {
            digest: {
                "train_rows_removed": train_counts[digest],
                "dev_rows": dev_counts[digest],
                "removed_train_line_numbers": [
                    int(record["line_number"])
                    for record in removed
                    if record["input_sha256"] == digest
                ],
            }
            for digest in overlap_hashes
        },
        "exact_tsv_overlap_count": len(exact_tsv_overlap),
        "exact_tsv_overlap_sha256": list(exact_tsv_overlap),
        "removed_train_rows": len(removed),
        "removed_train_line_numbers": removed_line_numbers,
        "removed_train_line_numbers_sha256": canonical_json_sha256(
            removed_line_numbers
        ),
        "removed_train_records_sha256": canonical_json_sha256(
            removed_record_fingerprints
        ),
        "derived_train_rows": len(derived),
        "derived_train_records_sha256": derived_hash,
        "derived_label_counts": {
            label: derived_label_counts[label] for label in LABELS
        },
        "post_filter_overlap_count": 0,
    }
    if len(raw_train_records) == OFFICIAL_SPLITS["train"]["rows"]:
        extra_expected = {
            "removed_train_line_numbers_sha256": (
                EXPECTED_REMOVED_LINE_NUMBERS_SHA256
            ),
            "removed_train_records_sha256": EXPECTED_REMOVED_RECORDS_SHA256,
        }
        extra_drift = {
            key: {"expected": value, "actual": audit[key]}
            for key, value in extra_expected.items()
            if audit[key] != value
        }
        if extra_drift:
            raise ValueError(
                "official train/dev overlap detail drift; refusing gradient data: "
                + json.dumps(extra_drift, sort_keys=True)
            )
    return derived, audit


def _model_tree_paths(root: Path) -> list[Path]:
    if root.is_symlink():
        raise ValueError("base model directory must not be a symlink")
    if not root.is_dir():
        raise ValueError(f"base model directory does not exist: {root}")
    paths: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"base model tree contains a symlink: {path}")
        relative = path.relative_to(root)
        if any(part in MODEL_TREE_IGNORED_PARTS for part in relative.parts):
            continue
        if path.name in MODEL_TREE_IGNORED_NAMES:
            continue
        if path.is_file():
            paths.append(path)
    if not paths:
        raise ValueError("base model directory contains no committed files")
    return paths


def hash_model_tree(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash a materialized local model tree, rejecting indirection/symlinks."""

    root = Path(root)
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _model_tree_paths(root)
    ]
    return canonical_json_sha256(entries), entries


def validate_base_model_dir(
    path: Path, expected_commitment: str
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("base model path must be an absolute local directory")
    if not HEX64.fullmatch(expected_commitment):
        raise ValueError("base model commitment must be a lowercase SHA-256")
    commitment, entries = hash_model_tree(path)
    if commitment != expected_commitment:
        raise ValueError(
            "base model tree commitment mismatch: "
            f"expected={expected_commitment} actual={commitment}"
        )
    names = {entry["path"] for entry in entries}
    if "config.json" not in names:
        raise ValueError("base model tree lacks config.json")
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "vocab.json",
    }
    if not names.intersection(tokenizer_names):
        raise ValueError("base model tree lacks tokenizer vocabulary artifacts")
    weight_names = {
        name
        for name in names
        if name.endswith((".safetensors", ".bin"))
        and "optimizer" not in name.lower()
    }
    if not weight_names:
        raise ValueError("base model tree lacks model weights")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"base model must be RoBERTa/XLM-RoBERTa, found {model_type!r}"
        )
    return {
        "path": str(path.resolve()),
        "tree_sha256": commitment,
        "files": entries,
        "file_count": len(entries),
        "model_type": model_type,
        "weight_files": sorted(weight_names),
        "local_files_only": True,
        "trust_remote_code": False,
        "symlinks_allowed": False,
    }


def class_balanced_weights(
    class_counts: Sequence[int], *, beta: float = 0.999
):
    """Effective-number class weights from Cui et al., normalized to mean 1."""

    import torch

    if len(class_counts) != len(LABELS) or any(count <= 0 for count in class_counts):
        raise ValueError("class counts must contain eight positive integers")
    if not 0.0 <= beta < 1.0:
        raise ValueError("class-balance beta must be in [0, 1)")
    if beta == 0.0:
        values = [1.0 for _ in class_counts]
    else:
        values = [
            (1.0 - beta) / (1.0 - math.pow(beta, int(count)))
            for count in class_counts
        ]
    mean = sum(values) / len(values)
    return torch.tensor([value / mean for value in values], dtype=torch.float32)


def policy_loss(
    logits,
    labels,
    *,
    mode: str,
    class_counts: Sequence[int],
    class_balance_beta: float = 0.999,
    logit_adjustment_tau: float = 1.0,
):
    import torch
    import torch.nn.functional as functional

    if mode not in LOSS_MODES:
        raise ValueError(f"unknown loss mode {mode!r}; choose from {LOSS_MODES}")
    if logits.ndim != 2 or logits.shape[1] != len(LABELS):
        raise ValueError("policy logits must have shape [batch, 8]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("policy labels must have shape [batch]")
    if len(class_counts) != len(LABELS) or any(count <= 0 for count in class_counts):
        raise ValueError("class counts must contain eight positive integers")
    if mode == "ce":
        return functional.cross_entropy(logits, labels)
    if mode == "class_balanced":
        weights = class_balanced_weights(
            class_counts, beta=class_balance_beta
        ).to(device=logits.device, dtype=logits.dtype)
        return functional.cross_entropy(logits, labels, weight=weights)
    if logit_adjustment_tau < 0.0:
        raise ValueError("logit-adjustment tau must be non-negative")
    counts = torch.tensor(class_counts, device=logits.device, dtype=logits.dtype)
    priors = counts / counts.sum()
    adjusted_logits = logits + logit_adjustment_tau * torch.log(priors)
    return functional.cross_entropy(adjusted_logits, labels)


def metrics_from_ids(
    gold_ids: Sequence[int], predicted_ids: Sequence[int], *, loss: float
) -> dict[str, Any]:
    if len(gold_ids) != len(predicted_ids):
        raise ValueError("gold and prediction lengths differ")
    records = [
        {
            "gold": LABELS[int(gold)],
            "prediction": LABELS[int(predicted)],
            "invalid": False,
        }
        for gold, predicted in zip(gold_ids, predicted_ids)
    ]
    metrics = compute_classification_metrics(records, LABELS)
    return {
        "loss": float(loss),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "total": int(metrics["total"]),
        "correct": int(metrics["correct"]),
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def is_better_dev(
    candidate: dict[str, Any], incumbent: dict[str, Any] | None
) -> bool:
    """Macro-F1 is primary; accuracy then lower loss are deterministic ties."""

    if incumbent is None:
        return True
    candidate_key = (
        float(candidate["macro_f1"]),
        float(candidate["accuracy"]),
        -float(candidate["loss"]),
    )
    incumbent_key = (
        float(incumbent["macro_f1"]),
        float(incumbent["accuracy"]),
        -float(incumbent["loss"]),
    )
    return candidate_key > incumbent_key


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_checkpoint_manifest(
    checkpoint_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    manifest_path = Path(manifest_path)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("checkpoint must be a regular local file")
    payload = copy.deepcopy(manifest)
    payload["checkpoint"] = {
        "filename": checkpoint_path.name,
        "size": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    manifest_hash = sha256_file(manifest_path)
    receipt_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    _atomic_write_text(
        receipt_path, f"{manifest_hash}  {manifest_path.name}\n"
    )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "receipt_path": str(receipt_path.resolve()),
        "checkpoint_sha256": payload["checkpoint"]["sha256"],
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except RuntimeError:
            pass
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str):
    import torch

    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def determinism_disclosure(device: Any) -> dict[str, Any]:
    return {
        "determinism_mode": f"best_effort_{device}",
        "torch_deterministic_algorithms_enabled": True,
        "torch_deterministic_algorithms_warn_only": True,
        "bitwise_reproducible_not_guaranteed": True,
    }


class TokenizedPolicyDataset:
    """Small map-style dataset pretokenized from leakage-safe records."""

    def __init__(self, records, tokenizer, max_length: int):
        texts = [record["input_text"] for record in records]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.features = []
        for index, record in enumerate(records):
            feature = {key: values[index] for key, values in encoded.items()}
            feature["labels"] = int(record["label_id"])
            self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def _collator(tokenizer):
    def collate(features):
        labels = [int(feature["labels"]) for feature in features]
        without_labels = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = tokenizer.pad(without_labels, padding=True, return_tensors="pt")
        import torch

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    return collate


def evaluate_model(
    model,
    data_loader,
    *,
    device,
    loss_mode: str,
    class_counts: Sequence[int],
    class_balance_beta: float,
    logit_adjustment_tau: float,
) -> dict[str, Any]:
    import torch

    model.eval()
    gold_ids: list[int] = []
    predicted_ids: list[int] = []
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for batch in data_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = policy_loss(
                logits,
                labels,
                mode=loss_mode,
                class_counts=class_counts,
                class_balance_beta=class_balance_beta,
                logit_adjustment_tau=logit_adjustment_tau,
            )
            batch_rows = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_rows
            total_rows += batch_rows
            gold_ids.extend(int(value) for value in labels.detach().cpu().tolist())
            predicted_ids.extend(
                int(value) for value in logits.argmax(dim=-1).detach().cpu().tolist()
            )
    if total_rows == 0:
        raise ValueError("development loader is empty")
    return metrics_from_ids(
        gold_ids, predicted_ids, loss=total_loss / total_rows
    )


def _pilot_configuration_audit(args: argparse.Namespace) -> dict[str, Any]:
    actual = {key: getattr(args, key) for key in FROZEN_PILOT_DEFAULTS}
    deviations = {
        key: {"frozen": expected, "actual": actual[key]}
        for key, expected in FROZEN_PILOT_DEFAULTS.items()
        if actual[key] != expected
    }
    return {
        "frozen_defaults": FROZEN_PILOT_DEFAULTS,
        "actual": actual,
        "matches_frozen_hyperparameters": not deviations,
        "deviations": deviations,
        "run_scope": (
            "preregistered_pilot_arm"
            if not deviations
            else "developmental_smoke_or_nonfrozen_configuration"
        ),
    }


def _training_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "loss": args.loss,
        "class_balance": {
            "method": "effective_number",
            "beta": args.class_balance_beta,
        },
        "logit_adjustment_tau": args.logit_adjustment_tau,
        "max_length": args.max_length,
        "truncation_side": args.truncation_side,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.train_batch_size * args.gradient_accumulation_steps
        ),
        "eval_batch_size": args.eval_batch_size,
        "max_grad_norm": args.max_grad_norm,
        "early_stopping_patience": args.early_stopping_patience,
        "optimizer": "torch.optim.AdamW",
        "scheduler": "linear_warmup_then_linear_decay",
    }


def train_policy(
    args: argparse.Namespace,
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    train_audit: dict[str, Any],
    dev_audit: dict[str, Any],
    base_audit: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    train_records, overlap_audit = derive_train_without_dev_overlap(
        train_records, dev_records
    )
    seed_everything(args.seed)
    device = resolve_device(args.device)
    transformers.utils.logging.disable_progress_bar()
    model_dir = Path(args.base_model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_NAME
    manifest_path = output_dir / MANIFEST_NAME
    receipt_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    occupied = [
        path for path in (checkpoint_path, manifest_path, receipt_path) if path.exists()
    ]
    if occupied:
        raise ValueError(
            "refusing to overwrite existing training artifacts: "
            + ", ".join(str(path) for path in occupied)
        )

    config = AutoConfig.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    if config.model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(f"unsupported local model type {config.model_type!r}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    tokenizer.truncation_side = args.truncation_side
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_dataset = TokenizedPolicyDataset(
        train_records, tokenizer, args.max_length
    )
    dev_dataset = TokenizedPolicyDataset(dev_records, tokenizer, args.max_length)
    collate = _collator(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )

    class_counts = [
        sum(record["label_id"] == label_id for record in train_records)
        for label_id in range(len(LABELS))
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = round(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        train_loss_sum = 0.0
        train_rows_seen = 0
        for batch_index, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = policy_loss(
                logits,
                labels,
                mode=args.loss,
                class_counts=class_counts,
                class_balance_beta=args.class_balance_beta,
                logit_adjustment_tau=args.logit_adjustment_tau,
            )
            batch_rows = int(labels.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * batch_rows
            train_rows_seen += batch_rows
            (loss / args.gradient_accumulation_steps).backward()
            should_update = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        dev_metrics = evaluate_model(
            model,
            dev_loader,
            device=device,
            loss_mode=args.loss,
            class_counts=class_counts,
            class_balance_beta=args.class_balance_beta,
            logit_adjustment_tau=args.logit_adjustment_tau,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_rows_seen,
            "dev": dev_metrics,
            "elapsed_seconds": time.time() - epoch_started,
            "learning_rate_after_epoch": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_record)
        print(
            json.dumps(
                {
                    "event": "dev_epoch",
                    "epoch": epoch,
                    "train_loss": epoch_record["train_loss"],
                    "dev_loss": dev_metrics["loss"],
                    "dev_macro_f1": dev_metrics["macro_f1"],
                    "dev_accuracy": dev_metrics["accuracy"],
                    "dev_weighted_f1": dev_metrics["weighted_f1"],
                    "elapsed_seconds": epoch_record["elapsed_seconds"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

        if is_better_dev(dev_metrics, best_metrics):
            best_metrics = copy.deepcopy(dev_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            portable_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            _atomic_torch_save(
                {
                    "protocol_id": PROTOCOL_ID,
                    "selected_epoch": epoch,
                    "labels": list(LABELS),
                    "base_model_tree_sha256": base_audit["tree_sha256"],
                    "model_state_dict": portable_state,
                },
                checkpoint_path,
            )
            del portable_state
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.early_stopping_patience:
            break

    if best_metrics is None or best_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError("training completed without a selectable checkpoint")
    ending_commitment, ending_entries = hash_model_tree(model_dir)
    if ending_commitment != base_audit["tree_sha256"]:
        raise RuntimeError("base model tree changed during training")

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_scope": _pilot_configuration_audit(args),
        "data": {
            "accepted_splits": ["train", "dev"],
            "train": train_audit,
            "dev": dev_audit,
            "dev_overlap_filter": overlap_audit,
            "gradient_source": "mandatory_derived_train_after_dev_input_exclusion",
            "label_order": list(LABELS),
            "target_construction": (
                "final supporter strategy is label; final supporter response is "
                "discarded; only turns with turn index lower than target are input"
            ),
        },
        "base_model": {
            **base_audit,
            "post_training_tree_sha256": ending_commitment,
            "post_training_files": ending_entries,
        },
        "training": {
            **_training_configuration(args),
            "device": str(device),
            "class_counts_in_label_order": class_counts,
            "planned_optimizer_updates": total_updates,
            "warmup_steps": warmup_steps,
            "epochs_completed": len(history),
            "elapsed_seconds": time.time() - started,
        },
        "selection": {
            "primary": "dev_macro_f1",
            "tie_breaker_1": "higher_dev_accuracy",
            "tie_breaker_2": "lower_dev_loss",
            "selected_epoch": best_epoch,
            "selected_dev_metrics": best_metrics,
            "epoch_history": history,
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
            "git_commit_at_completion": git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "manual_pytorch_loop": True,
            "determinism": determinism_disclosure(device),
        },
        "leakage_controls": {
            "holdout_path_argument_exists": False,
            "holdout_loaded": False,
            "target_response_retained": False,
            "target_strategy_in_model_input": False,
            "model_selection_split": "dev",
        },
    }
    result = write_checkpoint_manifest(checkpoint_path, manifest_path, manifest)
    print(json.dumps({"event": "complete", **result}, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or execute train/dev-only ESConv RoBERTa policy training. "
            "Without --execute, no model is loaded and no artifact is written."
        )
    )
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--dev-file", type=Path)
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument("--base-model-sha256")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/esconv-policy-roberta-v1"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check", dest="selftest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=LOSS_MODES, default="ce")
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--truncation-side", choices=("left", "right"), default="left"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    return parser


def _require_training_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    missing = [
        option
        for option, value in (
            ("--train-file", args.train_file),
            ("--dev-file", args.dev_file),
            ("--base-model-dir", args.base_model_dir),
            ("--base-model-sha256", args.base_model_sha256),
        )
        if value is None
    ]
    if missing:
        parser.error("required for data audit/training: " + ", ".join(missing))


def validate_arguments(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    positive_values = {
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_batch_size": args.eval_batch_size,
        "max_grad_norm": args.max_grad_norm,
        "early_stopping_patience": args.early_stopping_patience,
    }
    invalid = {key: value for key, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"training values must be positive: {invalid}")
    if args.weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup ratio must be in [0, 1)")
    if not 0 <= args.class_balance_beta < 1:
        raise ValueError("class-balance beta must be in [0, 1)")
    if args.logit_adjustment_tau < 0:
        raise ValueError("logit-adjustment tau must be non-negative")
    if args.max_length < 384 and args.truncation_side != "left":
        raise ValueError(
            "max_length below 384 must use left truncation to preserve recent dialogue"
        )
    base = Path(args.base_model_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output == base or output.is_relative_to(base):
        raise ValueError("output directory must be outside the immutable base model")
    if Path(args.train_file).resolve() == Path(args.dev_file).resolve():
        raise ValueError("train and dev must be distinct official files")


def run_selftest() -> dict[str, Any]:
    rows = [
        "1.0 0 0 hello EOS 1.0 1 1 [Questions] discard-this-response\n",
        (
            "1.0 0 0 hello EOS 1.0 1 1 [Other] prior response EOS "
            "1.0 0 2 sad EOS 1.0 1 3 [Information] discard-second-response\n"
        ),
    ]
    records = parse_split_lines(rows, split="train", expected_rows=2)
    serialized = json.dumps(records, ensure_ascii=False)
    if "discard-this-response" in serialized or "discard-second-response" in serialized:
        raise AssertionError("target response survived structural parsing")
    if not is_better_dev(
        {"macro_f1": 0.6, "accuracy": 0.1, "loss": 9.0},
        {"macro_f1": 0.5, "accuracy": 0.9, "loss": 0.1},
    ):
        raise AssertionError("development selection is not Macro-F1 first")
    return {
        "status": "ok",
        "records": len(records),
        "target_response_retained": False,
        "selection_primary": "dev_macro_f1",
        "accepted_splits": sorted(OFFICIAL_SPLITS),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps(run_selftest(), sort_keys=True))
        return
    _require_training_arguments(parser, args)
    validate_arguments(args)
    train_records, train_audit = load_official_split(args.train_file, "train")
    dev_records, dev_audit = load_official_split(args.dev_file, "dev")
    _derived_train_records, overlap_audit = derive_train_without_dev_overlap(
        train_records, dev_records
    )
    base_audit = validate_base_model_dir(
        args.base_model_dir, args.base_model_sha256
    )
    audit = {
        "protocol_id": PROTOCOL_ID,
        "mode": "execute" if args.execute else "read_only_audit",
        "data": {
            "train": train_audit,
            "dev": dev_audit,
            "dev_overlap_filter": overlap_audit,
            "gradient_source": "mandatory_derived_train_after_dev_input_exclusion",
        },
        "base_model": base_audit,
        "training": _training_configuration(args),
        "run_scope": _pilot_configuration_audit(args),
        "leakage_controls": {
            "accepted_splits": ["train", "dev"],
            "target_response_retained": False,
            "only_prior_turns_in_model_input": True,
        },
    }
    print(json.dumps({"event": "audit", **audit}, ensure_ascii=False, sort_keys=True))
    if not args.execute:
        return
    train_policy(
        args,
        train_records,
        dev_records,
        train_audit,
        dev_audit,
        base_audit,
    )


if __name__ == "__main__":
    main()
