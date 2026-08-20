#!/usr/bin/env python3
"""Evaluate allow-listed Hugging Face 8-way classifiers on frozen ESConv.

The evaluator is intentionally profile-driven.  A profile freezes the Hub
repository, immutable revision, model-card label order, input template,
maximum length, truncation side, config hash, and weight hash.  Arbitrary
``AutoModel`` repositories are not accepted because a generic ``LABEL_0``
configuration is not enough to recover an ESConv label map safely.

Only prior dialogue turns are passed to a model.  The target strategy is kept
as scorer-only gold data and the target supporter response is discarded while
parsing.  ModernBERT artifacts are recorded below, but remain disabled until
their authors publish an input construction contract; the frozen test must not
be used to choose a template.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_FROZEN_ROWS = 2_775
EXPECTED_FROZEN_TEST_SHA256 = (
    "b85ae888bf747cefa54bba2a6c3e2f6ccb4c1005d4e0b6d1d3be3823cf040aef"
)

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
HEEGYU_SOURCE_LABELS = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
)
MODERNBERT_SOURCE_LABELS = (
    "Affirmation and Reassurance",
    "Information",
    "Others",
    "Providing Suggestions",
    "Question",
    "Reflection of feelings",
    "Restatement or Paraphrasing",
    "Self-disclosure",
)
SOURCE_TO_CANONICAL = {
    "Question": "Questions",
    "Questions": "Questions",
    "Restatement or Paraphrasing": "Restatement or Paraphrasing",
    "Reflection of feelings": "Reflection of feelings",
    "Self-disclosure": "Self-disclosure",
    "Affirmation and Reassurance": "Affirmation and Reassurance",
    "Providing Suggestions": "Providing Suggestions",
    "Information": "Information",
    "Other": "Other",
    "Others": "Other",
}
CANONICAL_TO_CARD = {
    "Questions": "Question",
    "Restatement or Paraphrasing": "Restatement or Paraphrasing",
    "Reflection of feelings": "Reflection of feelings",
    "Self-disclosure": "Self-disclosure",
    "Affirmation and Reassurance": "Affirmation and Reassurance",
    "Providing Suggestions": "Providing Suggestions",
    "Information": "Information",
    "Other": "Others",
}

GENERIC_CONFIG_ID2LABEL = tuple(f"LABEL_{index}" for index in range(8))
GENERIC_CONFIG_LABEL2ID = {label: index for index, label in enumerate(GENERIC_CONFIG_ID2LABEL)}


def _enabled_profile(
    *,
    model_id: str,
    revision: str,
    architecture: str,
    config_sha256: str,
    card_sha256: str,
    weight_sha256: str,
    weight_size: int,
    template: str,
    max_length: int,
    reported_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "model_id": model_id,
        "revision": revision,
        "expected_architecture": architecture,
        "expected_config_sha256": config_sha256,
        "expected_card_sha256": card_sha256,
        "expected_weights": {
            "model.safetensors": {
                "sha256": weight_sha256,
                "size": weight_size,
            }
        },
        "expected_config_id2label": GENERIC_CONFIG_ID2LABEL,
        "expected_config_label2id": GENERIC_CONFIG_LABEL2ID,
        "source_label_order": HEEGYU_SOURCE_LABELS,
        "canonical_label_order": CANONICAL_LABELS,
        "label_mapping_source": f"README.md at immutable revision {revision}",
        "template": template,
        "max_length": max_length,
        "truncation_side": "right",
        "reported_metrics": reported_metrics,
    }


def _blocked_modernbert_profile(
    *, model_id: str, revision: str, config_sha256: str
) -> dict[str, Any]:
    explicit_label2id = {
        label: index for index, label in enumerate(MODERNBERT_SOURCE_LABELS)
    }
    return {
        "enabled": False,
        "block_reason": (
            "the immutable model card does not define an input template or training "
            "split construction; do not select a template or mapping on frozen test"
        ),
        "model_id": model_id,
        "revision": revision,
        "expected_architecture": "ModernBertForSequenceClassification",
        "expected_config_sha256": config_sha256,
        "expected_card_sha256": None,
        "expected_weights": None,
        "expected_config_id2label": MODERNBERT_SOURCE_LABELS,
        "expected_config_label2id": explicit_label2id,
        "source_label_order": MODERNBERT_SOURCE_LABELS,
        "canonical_label_order": CANONICAL_LABELS,
        "label_mapping_source": f"config.json at immutable revision {revision}",
        "template": None,
        "max_length": 8_192,
        "truncation_side": None,
        "reported_metrics": None,
    }


PROFILES: dict[str, dict[str, Any]] = {
    "tinyllama-augesc-context": _enabled_profile(
        model_id="heegyu/TinyLlama-augesc-context",
        revision="4fd4cdc278812afd572e34040ecf433a31c8623e",
        architecture="LlamaForSequenceClassification",
        config_sha256="46a30993f08d288f45efe40cd41ed1c35f48360a28dd5842cf1d90bc0ddb4e87",
        card_sha256="84a4f2cecb97a5f39f876455cf4579b782a16c35a504612e1f284eba74bafa16",
        weight_sha256="be1b01ac696c8a4f8397286e5a9accd14bcff0aaa0b8f43ef3ae5a37b82fc483",
        weight_size=4_138_138_032,
        template="heegyu_context_without_strategy",
        max_length=2_048,
        reported_metrics={
            "dataset": "heegyu/augesc",
            "accuracy": 0.4158,
            "macro_f1": 0.2453,
            "source": "model card",
            "comparability": "not assumed to be the frozen original-ESConv test",
        },
    ),
    "esconv-xlm-roberta-base": _enabled_profile(
        model_id="heegyu/esconv-xlm-roberta-base",
        revision="8a21d3f0e1ec06aa147faa652e64b938a55c37dd",
        architecture="XLMRobertaForSequenceClassification",
        config_sha256="42af14b58e13586334a15de879473d83dee5fdd90e81d62fe578d8b5ea410ffd",
        card_sha256="431c04444f8d1c9bec9a24544cd721cc17803efc8df1b023876df53a90d36aaf",
        weight_sha256="4fd0d826664b28158fd907321fea2bd3b79144c503bc6b869148ec1e438816b9",
        weight_size=1_112_223_464,
        template="heegyu_context_with_prior_strategy",
        max_length=512,
    ),
    "esconv-xlm-roberta-large": _enabled_profile(
        model_id="heegyu/esconv-xlm-roberta-large",
        revision="0c1955b5f0d1aceb31cd4067432ed29bc936caa9",
        architecture="XLMRobertaForSequenceClassification",
        config_sha256="759ae2accfc0dc8d69baef1d4356373a0c4a6e8a2d12010e69dee844cac85d7f",
        card_sha256="dc375c0f9091f9db8c34939cf52622b98ba6ae5ca7bc78d10b6fb9c7e1613972",
        weight_sha256="561a372d63e1cacd4a85cd67642ad3e1df5aa18b8219dc7372f3953c81316869",
        weight_size=2_239_643_272,
        template="heegyu_context_with_prior_strategy",
        max_length=512,
    ),
    "modernbert-singleturn": _blocked_modernbert_profile(
        model_id=(
            "thanaphatt1/ModernBERT-base-esconv-strategy-classification-singleturn"
        ),
        revision="d51dc11656a769cd84b5dcf11b4a9953d0f63fbc",
        config_sha256="19563e558e9ca505326c9e25330014808cba8c60e42bc6a330c01dea7e229a6a",
    ),
    "modernbert-multiturn-upsampling": _blocked_modernbert_profile(
        model_id=(
            "thanaphatt1/ModernBERT-base-esconv-strategy-classification-"
            "multiturn-upsampling"
        ),
        revision="5f5d0641ae9c94f65b717aa1d11f3ee3192e51b4",
        config_sha256="505430ffcace18970a29122a841c1268db7f12b05fe33f2c0bef0dad5ab285a2",
    ),
    "modernbert-multiturn-upsampling-v3": _blocked_modernbert_profile(
        model_id=(
            "thanaphatt1/ModernBERT-base-esconv-strategy-classification-"
            "multiturn-upsampling-v3"
        ),
        revision="f4057d6e1cfaf62184b369e014ad6e0b26dec13a",
        config_sha256="505430ffcace18970a29122a841c1268db7f12b05fe33f2c0bef0dad5ab285a2",
    ),
}


_SEGMENT = re.compile(
    r"^\s*(?P<loss>[01](?:\.0)?)\s+(?P<role>[01])\s+"
    r"(?P<turn>\d+)\s+(?P<body>.*?)\s*$"
)
_STRATEGY_PREFIX = re.compile(r"^\[(?P<strategy>[^\]]+)\]\s*(?P<text>.*)$")
_HEX_REVISION = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def get_enabled_profile(key: str) -> dict[str, Any]:
    if key not in PROFILES:
        raise ValueError(
            f"unknown profile {key!r}; label mapping must be explicitly approved"
        )
    profile = PROFILES[key]
    if not profile["enabled"]:
        raise ValueError(
            f"profile {key!r} is blocked: {profile['block_reason']} (input template "
            "must be documented before frozen-test inference)"
        )
    if profile.get("expected_config_id2label") is None:
        raise ValueError(f"profile {key!r} has no explicitly approved label mapping")
    return copy.deepcopy(profile)


def canonicalize_source_label(label: str) -> str:
    try:
        return SOURCE_TO_CANONICAL[label]
    except KeyError as error:
        raise ValueError(f"unknown ESConv label {label!r}") from error


def _normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def _parse_segment(raw_segment: str, *, target: bool) -> dict[str, Any]:
    match = _SEGMENT.match(raw_segment)
    if not match:
        raise ValueError(f"invalid frozen turn segment {raw_segment[:80]!r}")
    role = int(match.group("role"))
    turn = int(match.group("turn"))
    body = _normalize_space(match.group("body"))
    strategy = None
    if role == 1:
        strategy_match = _STRATEGY_PREFIX.match(body)
        if strategy_match is None:
            raise ValueError("supporter turn lacks a leading strategy annotation")
        strategy = strategy_match.group("strategy")
        if strategy not in CANONICAL_LABELS:
            raise ValueError(f"unknown frozen strategy {strategy!r}")
        body = _normalize_space(strategy_match.group("text"))
    if target and role != 1:
        raise ValueError("frozen target is not a supporter turn")
    if not body:
        raise ValueError("empty frozen utterance")
    result = {"role": role, "turn": turn, "strategy": strategy}
    if not target:
        result["text"] = body
    return result


def _format_history(turns: Sequence[dict[str, Any]], profile: dict[str, Any]) -> str:
    lines: list[str] = []
    for turn in turns:
        if turn["role"] == 0:
            lines.append(f"usr: {turn['text']}")
            continue
        if turn["role"] != 1 or turn["strategy"] is None:
            raise ValueError("history contains an invalid supporter turn")
        if profile["template"] == "heegyu_context_without_strategy":
            lines.append(f"sys: {turn['text']}")
        elif profile["template"] == "heegyu_context_with_prior_strategy":
            card_label = CANONICAL_TO_CARD[turn["strategy"]]
            lines.append(f"sys[{card_label}]: {turn['text']}")
        else:
            raise ValueError(f"unsupported frozen input template {profile['template']!r}")
    if not lines:
        raise ValueError("history is empty")
    return "\n".join(lines)


def prepare_frozen_lines(
    lines: Iterable[str],
    *,
    profile: dict[str, Any],
    expected_rows: int | None = EXPECTED_FROZEN_ROWS,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Parse the frozen TSV while structurally discarding each target response."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    source_lines = [line for line in lines if line.strip()]
    if expected_rows is not None and len(source_lines) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} frozen rows, found {len(source_lines)}"
        )
    selected_lines = source_lines[:limit] if limit is not None else source_lines

    records: list[dict[str, Any]] = []
    conversation_number = 0
    previous_target_turn: int | None = None
    for item_id, raw_line in enumerate(selected_lines):
        line_number = item_id + 1
        base: dict[str, Any] = {
            "item_id": item_id,
            "line_number": line_number,
            "source_line_sha256": sha256_text(raw_line.rstrip("\r\n")),
        }
        try:
            raw_segments = re.split(r"\s+EOS\s+", raw_line.strip())
            if len(raw_segments) < 2:
                raise ValueError("row has no prior-history/target EOS boundary")
            target = _parse_segment(raw_segments[-1], target=True)
            target_turn = int(target["turn"])
            if previous_target_turn is None or target_turn <= previous_target_turn:
                conversation_number += 1
            previous_target_turn = target_turn
            conversation_id = f"esconv-test-{conversation_number:04d}"
            context = [
                _parse_segment(segment, target=False)
                for segment in raw_segments[:-1]
            ]
            if any(int(turn["turn"]) >= target_turn for turn in context):
                raise ValueError("context includes a non-prior or target turn")
            model_input = _format_history(context, profile)
            base.update(
                {
                    "conversation_id": conversation_id,
                    "target_turn": target_turn,
                    "gold": target["strategy"],
                    "input_sha256": sha256_text(model_input),
                    "_model_input": model_input,
                    "prediction": None,
                    "correct": False,
                    "invalid": False,
                }
            )
        except Exception as error:
            if conversation_number == 0:
                conversation_number = 1
            base.update(
                {
                    "conversation_id": f"esconv-test-{conversation_number:04d}",
                    "target_turn": None,
                    "gold": None,
                    "input_sha256": None,
                    "prediction": None,
                    "correct": False,
                    "invalid": True,
                    "invalid_reason": f"parse_error: {error}",
                }
            )
        records.append(base)
    return records


def prepare_frozen_file(
    path: Path, *, profile: dict[str, Any], limit: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_FROZEN_TEST_SHA256:
        raise ValueError(
            "frozen ESConv test hash mismatch: "
            f"expected={EXPECTED_FROZEN_TEST_SHA256} actual={actual_hash}"
        )
    with path.open(encoding="utf-8") as handle:
        records = prepare_frozen_lines(
            handle,
            profile=profile,
            expected_rows=EXPECTED_FROZEN_ROWS,
            limit=limit,
        )
    return records, {
        "path": str(path.resolve()),
        "sha256": actual_hash,
        "expected_sha256": EXPECTED_FROZEN_TEST_SHA256,
        "full_rows": EXPECTED_FROZEN_ROWS,
        "evaluated_rows": len(records),
        "limit": limit,
        "conversation_id_rule": (
            "sequential esconv-test-NNNN; increment when target turn resets/decreases"
        ),
    }


def validate_config_contract(config: dict[str, Any], profile: dict[str, Any]) -> None:
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("config.architectures is missing")
    forbidden = [
        name
        for name in architectures
        if "CausalLM" in name or "ConditionalGeneration" in name or "LMHead" in name
    ]
    if forbidden:
        raise ValueError(
            f"CausalLM/generation head is forbidden for 8-way classification: {forbidden}"
        )
    expected_architecture = profile["expected_architecture"]
    if architectures != [expected_architecture] or "SequenceClassification" not in expected_architecture:
        raise ValueError(
            f"sequence-classification architecture mismatch: expected "
            f"{expected_architecture!r}, found {architectures!r}"
        )

    approved_id2label = profile.get("expected_config_id2label")
    if approved_id2label is None:
        raise ValueError("config label mapping is not explicitly approved")
    actual_id2label = config.get("id2label")
    expected_id2label = {
        str(index): label for index, label in enumerate(approved_id2label)
    }
    if actual_id2label != expected_id2label:
        raise ValueError(
            f"config.id2label mismatch: expected={expected_id2label!r} "
            f"actual={actual_id2label!r}"
        )
    expected_label2id = profile.get("expected_config_label2id")
    if expected_label2id is None or config.get("label2id") != expected_label2id:
        raise ValueError("config.label2id is missing, unknown, or ambiguous")

    source_labels = tuple(profile["source_label_order"])
    mapped = tuple(canonicalize_source_label(label) for label in source_labels)
    if len(source_labels) != 8 or set(mapped) != set(CANONICAL_LABELS):
        raise ValueError("approved source label order is not a bijection over 8 labels")


def detect_local_hf_revision(model_dir: Path) -> str | None:
    """Read immutable revision evidence from HF local-dir metadata/snapshot path."""

    model_dir = Path(model_dir).resolve()
    revisions: set[str] = set()
    metadata_dir = model_dir / ".cache/huggingface/download"
    if metadata_dir.is_dir():
        for metadata in metadata_dir.glob("*.metadata"):
            try:
                first_line = metadata.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, IndexError, UnicodeDecodeError):
                continue
            if _HEX_REVISION.fullmatch(first_line):
                revisions.add(first_line)

    parts = model_dir.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts) and _HEX_REVISION.fullmatch(parts[index + 1]):
            revisions.add(parts[index + 1])

    if len(revisions) > 1:
        raise ValueError(f"local Hugging Face metadata contains mixed revisions: {sorted(revisions)}")
    return next(iter(revisions)) if revisions else None


def _resolve_revision(
    model_dir: Path, profile: dict[str, Any], declared_revision: str | None
) -> tuple[str, str]:
    expected = profile["revision"]
    detected = detect_local_hf_revision(model_dir)
    if declared_revision is not None and declared_revision != expected:
        raise ValueError(
            f"declared model revision differs from profile: {declared_revision} != {expected}"
        )
    if detected is not None and detected != expected:
        raise ValueError(
            f"local model revision differs from profile: {detected} != {expected}"
        )
    if detected is not None:
        return detected, "huggingface-local-metadata-or-snapshot-path"
    if declared_revision is not None:
        return declared_revision, "caller-declared-plus-byte-hash-verification"
    raise ValueError(
        "cannot prove the local model revision; retain Hugging Face local-dir metadata "
        "or pass --model-revision with the pinned 40-character revision"
    )


def _hash_named_files(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    entries = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths, key=lambda item: item.name)
    ]
    return entries, canonical_json_sha256(entries)


def audit_model_directory(
    model_dir: Path,
    *,
    profile: dict[str, Any],
    declared_revision: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    config_path = model_dir / "config.json"
    card_path = model_dir / "README.md"
    if not config_path.is_file() or not card_path.is_file():
        raise ValueError("model directory must contain config.json and immutable README.md")

    config_hash = sha256_file(config_path)
    if config_hash != profile["expected_config_sha256"]:
        raise ValueError(
            f"config hash mismatch: expected={profile['expected_config_sha256']} "
            f"actual={config_hash}"
        )
    card_hash = sha256_file(card_path)
    if card_hash != profile["expected_card_sha256"]:
        raise ValueError(
            f"model-card hash mismatch: expected={profile['expected_card_sha256']} "
            f"actual={card_hash}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config_contract(config, profile)

    expected_weights = profile["expected_weights"]
    weight_paths = [model_dir / filename for filename in expected_weights]
    missing = [path.name for path in weight_paths if not path.is_file()]
    if missing:
        raise ValueError(f"model download is incomplete; missing weight files: {missing}")
    weight_entries = []
    for path in weight_paths:
        expected = expected_weights[path.name]
        actual_size = path.stat().st_size
        if actual_size != expected["size"]:
            raise ValueError(
                f"weight size mismatch for {path.name}: expected={expected['size']} "
                f"actual={actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"weight hash mismatch for {path.name}: expected={expected['sha256']} "
                f"actual={actual_hash}"
            )
        weight_entries.append(
            {"name": path.name, "size": actual_size, "sha256": actual_hash}
        )

    tokenizer_names = (
        "tokenizer.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    tokenizer_paths = [model_dir / name for name in tokenizer_names if (model_dir / name).is_file()]
    if not tokenizer_paths:
        raise ValueError("model directory contains no tokenizer artifacts")
    tokenizer_entries, tokenizer_manifest_hash = _hash_named_files(tokenizer_paths)
    revision, revision_evidence = _resolve_revision(
        model_dir, profile, declared_revision
    )

    artifact = {
        "source_model_id": profile["model_id"],
        "revision": revision,
        "revision_evidence": revision_evidence,
        "path": str(model_dir.resolve()),
        "config": {
            "sha256": config_hash,
            "expected_sha256": profile["expected_config_sha256"],
            "architectures": config["architectures"],
            "model_type": config.get("model_type"),
            "id2label": config["id2label"],
            "label2id": config["label2id"],
        },
        "model_card": {
            "sha256": card_hash,
            "expected_sha256": profile["expected_card_sha256"],
        },
        "weights": {
            "files": weight_entries,
            "manifest_sha256": canonical_json_sha256(weight_entries),
        },
        "tokenizer": {
            "files": tokenizer_entries,
            "manifest_sha256": tokenizer_manifest_hash,
        },
    }
    return artifact, config


def select_device(requested: str, torch_module: Any) -> Any:
    if requested != "auto":
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return torch_module.device("mps")
    return torch_module.device("cpu")


def validate_logits_shape(shape: Sequence[int], *, batch_size: int) -> None:
    if tuple(shape) != (batch_size, 8):
        raise ValueError(
            f"model output is not an 8-class sequence-classification head: "
            f"expected={(batch_size, 8)} actual={tuple(shape)}"
        )


def load_hf_model(
    model_dir: Path, profile: dict[str, Any], requested_device: str
) -> tuple[Any, Any, Any, dict[str, str]]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "PyTorch and Transformers are required for model inference"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    tokenizer.truncation_side = profile["truncation_side"]
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    class_name = type(model).__name__
    if class_name != profile["expected_architecture"] or "SequenceClassification" not in class_name:
        raise ValueError(
            f"loaded class is not the approved sequence classifier: {class_name!r}"
        )
    if int(model.config.num_labels) != 8:
        raise ValueError(f"loaded classifier has {model.config.num_labels} labels, expected 8")
    device = select_device(requested_device, torch)
    model.eval().to(device)
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    return model, tokenizer, device, versions


def run_inference(
    records: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    profile: dict[str, Any],
    batch_size: int,
) -> None:
    import torch

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    source_to_canonical = [
        canonicalize_source_label(label) for label in profile["source_label_order"]
    ]
    valid_indices = [
        index for index, record in enumerate(records) if not record.get("invalid")
    ]
    with torch.inference_mode():
        for start in range(0, len(valid_indices), batch_size):
            indices = valid_indices[start : start + batch_size]
            texts = [records[index]["_model_input"] for index in indices]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=profile["max_length"],
                return_tensors="pt",
            )
            attention_mask = encoded.get("attention_mask")
            token_counts = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask is not None
                else [None] * len(indices)
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            validate_logits_shape(logits.shape, batch_size=len(indices))
            probabilities = torch.softmax(logits.float(), dim=-1)
            logits_cpu = logits.float().cpu()
            probabilities_cpu = probabilities.cpu()
            for offset, record_index in enumerate(indices):
                row_logits = logits_cpu[offset]
                row_probabilities = probabilities_cpu[offset]
                record = records[record_index]
                record["input_token_count"] = (
                    int(token_counts[offset]) if token_counts[offset] is not None else None
                )
                if not bool(torch.isfinite(row_logits).all()):
                    record.update(
                        {
                            "prediction": None,
                            "correct": False,
                            "invalid": True,
                            "invalid_reason": "non_finite_logits",
                        }
                    )
                    continue
                prediction_id = int(torch.argmax(row_logits).item())
                prediction = source_to_canonical[prediction_id]
                record.update(
                    {
                        "prediction_id": prediction_id,
                        "prediction": prediction,
                        "correct": prediction == record["gold"],
                        "invalid": False,
                        "logits": {
                            source_to_canonical[index]: round(float(value), 8)
                            for index, value in enumerate(row_logits.tolist())
                        },
                        "probabilities": {
                            source_to_canonical[index]: round(float(value), 10)
                            for index, value in enumerate(row_probabilities.tolist())
                        },
                    }
                )


def compute_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(CANONICAL_LABELS)}
    matrix = [[0 for _ in CANONICAL_LABELS] for _ in CANONICAL_LABELS]
    invalid = 0
    correct = 0
    for record in records:
        gold = record.get("gold")
        prediction = record.get("prediction")
        if (
            record.get("invalid")
            or gold not in label_to_index
            or prediction not in label_to_index
        ):
            invalid += 1
            continue
        gold_index = label_to_index[gold]
        prediction_index = label_to_index[prediction]
        matrix[gold_index][prediction_index] += 1
        correct += int(gold == prediction)

    total = len(records)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    weighted_sum = 0.0
    valid_support = 0
    for index, label in enumerate(CANONICAL_LABELS):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
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

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted_sum / valid_support if valid_support else 0.0,
        "invalid_rate": invalid / total if total else 0.0,
        "correct": correct,
        "total": total,
        "invalid": invalid,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(CANONICAL_LABELS),
            "rows_gold_columns_predicted": matrix,
        },
    }


def public_prediction(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_predictions(path: Path, records: Sequence[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(
            public_prediction(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )
    _atomic_write_text(path, content)


def write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _profile_summary(key: str, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": key,
        "source_model_id": profile["model_id"],
        "immutable_revision": profile["revision"],
        "input_template": profile["template"],
        "input_template_literal": (
            "usr: {seeker_text}\\nsys: {supporter_text}"
            if profile["template"] == "heegyu_context_without_strategy"
            else "usr: {seeker_text}\\nsys[{prior_strategy}]: {supporter_text}"
        ),
        "max_length": profile["max_length"],
        "truncation_side": profile["truncation_side"],
        "source_label_order": list(profile["source_label_order"]),
        "canonical_label_order": list(CANONICAL_LABELS),
        "label_mapping_source": profile["label_mapping_source"],
        "reported_metrics": profile.get("reported_metrics"),
    }


def build_summary(
    *,
    run_name: str,
    profile_key: str,
    profile: dict[str, Any],
    model_artifact: dict[str, Any],
    dataset_artifact: dict[str, Any],
    predictions_path: Path,
    records: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    invalid_reasons: dict[str, int] = {}
    for record in records:
        if record.get("invalid"):
            reason = str(record.get("invalid_reason", "unspecified"))
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    is_smoke = dataset_artifact["limit"] is not None
    return {
        "audit_status": (
            "SMOKE_ONLY"
            if is_smoke
            else ("COMPLETE_WITH_INVALID" if metrics["invalid"] else "COMPLETE")
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "task": "8-way next-support-strategy prediction",
        "profile": _profile_summary(profile_key, profile),
        "model": model_artifact,
        "dataset": dataset_artifact,
        "predictions": {
            "path": str(Path(predictions_path).resolve()),
            "sha256": sha256_file(predictions_path),
            "rows": len(records),
        },
        "metrics": metrics,
        "invalid_reasons": invalid_reasons,
        "protocol": {
            "frozen_2775": not is_smoke,
            "prefix_smoke": is_smoke,
            "target_strategy_excluded_from_model_input": True,
            "target_response_excluded_from_model_input_and_outputs": True,
            "prior_supporter_strategy_in_input": (
                profile["template"] == "heegyu_context_with_prior_strategy"
            ),
            "decision_rule": "argmax over exactly eight sequence-classification logits",
            "test_tuning_prohibited": True,
            "modernbert_template_selection_on_test_prohibited": True,
            "invalid_rows_retained_in_accuracy_denominator": True,
        },
        "runtime": runtime,
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
            "repository_commit": git_commit(ROOT),
        },
        "eligible_for_frozen_leaderboard": (
            not is_smoke and metrics["invalid"] == 0
        ),
    }


def _ensure_outputs_available(
    predictions_out: Path, summary_out: Path, *, overwrite: bool
) -> None:
    if predictions_out.resolve() == summary_out.resolve():
        raise ValueError("predictions and summary outputs must be different files")
    existing = [path for path in (predictions_out, summary_out) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"refusing to overwrite existing outputs: {[str(path) for path in existing]}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    profile = get_enabled_profile(args.profile)
    _ensure_outputs_available(
        args.predictions_out, args.summary_out, overwrite=args.overwrite
    )
    model_artifact, _ = audit_model_directory(
        args.model_dir,
        profile=profile,
        declared_revision=args.model_revision,
    )
    records, dataset_artifact = prepare_frozen_file(
        args.test_file, profile=profile, limit=args.limit
    )
    model, tokenizer, device, versions = load_hf_model(
        args.model_dir, profile, args.device
    )
    started = time.monotonic()
    run_inference(
        records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        profile=profile,
        batch_size=args.batch_size,
    )
    elapsed = time.monotonic() - started
    metrics = compute_metrics(records)
    write_predictions(args.predictions_out, records)
    runtime = {
        **versions,
        "platform": platform.platform(),
        "device": str(device),
        "batch_size": args.batch_size,
        "elapsed_seconds": round(elapsed, 3),
    }
    summary = build_summary(
        run_name=args.run_name,
        profile_key=args.profile,
        profile=profile,
        model_artifact=model_artifact,
        dataset_artifact=dataset_artifact,
        predictions_path=args.predictions_out,
        records=records,
        metrics=metrics,
        runtime=runtime,
    )
    write_json(args.summary_out, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auditable allow-listed HF classifier evaluation on frozen ESConv"
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--model-revision",
        help=(
            "Pinned 40-character Hub revision; needed only if HF local-dir metadata "
            "or a snapshots/<revision> path is unavailable"
        ),
    )
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "audit_status": summary["audit_status"],
                "accuracy": summary["metrics"]["accuracy"],
                "macro_f1": summary["metrics"]["macro_f1"],
                "weighted_f1": summary["metrics"]["weighted_f1"],
                "invalid": summary["metrics"]["invalid"],
                "summary": str(args.summary_out.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
