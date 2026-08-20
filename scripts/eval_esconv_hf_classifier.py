#!/usr/bin/env python3
"""Evaluate allow-listed Hugging Face 8-way classifiers on frozen ESConv.

The evaluator is intentionally profile-driven.  A profile freezes the Hub
repository, immutable revision, model-card label order, input template,
maximum length, truncation side, config hash, and weight hash.  Arbitrary
``AutoModel`` repositories are not accepted because a generic ``LABEL_0``
configuration is not enough to recover an ESConv label map safely.

Only prior dialogue turns are passed to a model.  The target strategy is kept
as scorer-only gold data and the target supporter response is discarded while
parsing.  XLM-R and ModernBERT artifacts are recorded below, but remain
disabled until an immutable frozen input construction contract is approved;
the frozen test must not be used to choose a template.

Before this process opens the frozen test or loads a model, it requires a
pre-committed, clean, externally hash-anchored candidate selection,
authorization, provenance audit, and armed consumption receipt.  The formal
runner has no prefix/smoke mode: its single campaign slot is atomically claimed
before the test is read, and it must publish exactly 2,775 predictions into one
pre-authorized run directory.  A persistent git-private claim plus the tracked
receipt fail closed on concurrent use and ordinary local replay.
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
import secrets
import stat
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
TRAINING_PROVENANCE_AUDIT_SCHEMA = (
    "esconv-checkpoint-training-provenance-audit-v1"
)
CANDIDATE_SELECTION_SCHEMA = "esconv-frozen-candidate-selection-v1"
CAMPAIGN_AUTHORIZATION_SCHEMA = "esconv-frozen-campaign-authorization-v2"
CONSUMPTION_RECEIPT_SCHEMA = "esconv-frozen-campaign-consumption-v1"
CHECKPOINT_TRAINING_PROVENANCE_CHOICES = (
    "frozen_train_dev_only",
    "external_unknown",
    "known_overlap",
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
    enabled: bool = True,
    block_reason: str | None = None,
    input_template_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "block_reason": block_reason,
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
        "profile_requires_frozen_input_contract": True,
        "input_template_source": input_template_source,
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
        "profile_requires_frozen_input_contract": True,
        "input_template_source": None,
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
        input_template_source={
            "kind": "immutable_model_card",
            "url": (
                "https://huggingface.co/heegyu/TinyLlama-augesc-context/blob/"
                "4fd4cdc278812afd572e34040ecf433a31c8623e/README.md"
            ),
            "revision": "4fd4cdc278812afd572e34040ecf433a31c8623e",
            "sha256": "84a4f2cecb97a5f39f876455cf4579b782a16c35a504612e1f284eba74bafa16",
            "evidence": "Top-1 strategy-prediction example with usr:/sys: turns",
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
        enabled=False,
        block_reason=(
            "profile requires an immutable frozen input contract; current adapter "
            "must not choose an XLM-R template on the frozen test"
        ),
        input_template_source=None,
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
        enabled=False,
        block_reason=(
            "profile requires an immutable frozen input contract; current adapter "
            "must not choose an XLM-R template on the frozen test"
        ),
        input_template_source=None,
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
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _open_regular_file(path: Path) -> int:
    """Open one immutable-by-fd regular-file view and reject symlinks."""

    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"cannot stat required file {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlink artifacts are forbidden: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact is not a regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open required file {path}: {error}") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise ValueError(f"opened artifact is not a regular file: {path}")
    if (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(descriptor)
        raise ValueError(f"artifact changed while it was opened: {path}")
    return descriptor


def _read_regular_file_bytes(path: Path) -> bytes:
    descriptor = _open_regular_file(path)
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError(f"artifact changed while it was read: {path}")
    return b"".join(chunks)


def _hash_regular_file(path: Path) -> tuple[int, str]:
    descriptor = _open_regular_file(path)
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        total != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError(f"artifact changed while it was hashed: {path}")
    return total, digest.hexdigest()


def sha256_file(path: Path) -> str:
    return _hash_regular_file(Path(path))[1]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json_artifact(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = _read_regular_file_bytes(path)
        document = json.loads(payload.decode("utf-8"))
    except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{description} must contain one JSON object")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "document": document,
    }


def _run_git(repo_root: Path, arguments: Sequence[str], *, text: bool) -> Any:
    command = ["git", "-C", str(repo_root), *arguments]
    try:
        return subprocess.check_output(
            command,
            text=text,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", b"" if not text else "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise ValueError(
            f"git verification failed for {' '.join(arguments)}: {str(stderr).strip()}"
        ) from error


def _repo_relative_artifact(path: Path, repo_root: Path) -> tuple[Path, str]:
    repo_root = Path(repo_root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"artifact must be inside repository {repo_root}: {path}") from error
    cursor = repo_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {cursor}")
    return resolved, relative.as_posix()


def _validate_expected_sha256(value: str, *, description: str) -> None:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise ValueError(f"{description} expected SHA-256 must be 64 lowercase hex characters")


def load_committed_file_artifact(
    path: Path,
    *,
    description: str,
    repo_root: Path,
    trusted_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load bytes once and prove that exact blob is tracked, committed, and clean."""

    _validate_expected_sha256(expected_sha256, description=description)
    if not isinstance(trusted_commit, str) or not _HEX_REVISION.fullmatch(
        trusted_commit
    ):
        raise ValueError(f"{description} trusted commit must be 40 lowercase hex characters")
    repo_root = Path(repo_root).resolve()
    actual_root = Path(
        _run_git(repo_root, ["rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    if actual_root != repo_root:
        raise ValueError(f"repository root mismatch: expected={repo_root} actual={actual_root}")
    _run_git(repo_root, ["cat-file", "-e", f"{trusted_commit}^{{commit}}"], text=False)
    resolved, relative = _repo_relative_artifact(path, repo_root)
    payload = _read_regular_file_bytes(resolved)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    status = _run_git(
        repo_root,
        ["status", "--porcelain", "--untracked-files=all", "--", relative],
        text=True,
    ).strip()
    if status:
        raise ValueError(f"{description} is not tracked, committed, and clean: {status}")
    try:
        committed_payload = _run_git(
            repo_root, ["show", f"{trusted_commit}:{relative}"], text=False
        )
    except ValueError as error:
        raise ValueError(
            f"{description} is not tracked and committed at the trusted commit"
        ) from error
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{description} does not match explicit expected SHA-256: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    if committed_payload != payload:
        raise ValueError(
            f"{description} working-tree bytes do not equal the trusted committed blob"
        )
    return {
        "path": str(resolved),
        "repo_relative_path": relative,
        "sha256": actual_sha256,
        "trusted_commit": trusted_commit,
        "_payload": payload,
    }


def load_committed_json_artifact(
    path: Path,
    *,
    description: str,
    repo_root: Path,
    trusted_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    artifact = load_committed_file_artifact(
        path,
        description=description,
        repo_root=repo_root,
        trusted_commit=trusted_commit,
        expected_sha256=expected_sha256,
    )
    payload = artifact.pop("_payload")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{description} must contain one JSON object")
    return {**artifact, "document": document}


def validate_training_provenance_audit(
    audit: dict[str, Any], *, dataset_hash: str
) -> None:
    if audit.get("schema_version") != TRAINING_PROVENANCE_AUDIT_SCHEMA:
        raise ValueError(
            "training provenance audit schema_version must equal "
            f"{TRAINING_PROVENANCE_AUDIT_SCHEMA!r}"
        )
    if audit.get("audit_status") != "COMPLETE":
        raise ValueError("training provenance audit status must be COMPLETE")
    frozen_test = audit.get("frozen_test")
    if not isinstance(frozen_test, dict):
        raise ValueError("training provenance audit lacks frozen_test binding")
    if frozen_test.get("sha256") != dataset_hash:
        raise ValueError("training provenance audit test hash does not match this run")
    if frozen_test.get("rows") != EXPECTED_FROZEN_ROWS:
        raise ValueError(
            f"training provenance audit must bind {EXPECTED_FROZEN_ROWS} test rows"
        )
    overlap = audit.get("train_test_overlap")
    if not isinstance(overlap, dict):
        raise ValueError("training provenance audit lacks train/test overlap result")
    overlap_rows = overlap.get("rows")
    if isinstance(overlap_rows, bool) or not isinstance(overlap_rows, int):
        raise ValueError("training provenance audit overlap rows must be an integer")
    if overlap_rows != 0:
        raise ValueError(
            f"training provenance audit reports train/test overlap={overlap_rows}, expected 0"
        )


def _campaign_candidate(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_key": context["profile_key"],
        "model_id": context["model_id"],
        "revision": context["revision"],
        "weights_manifest_sha256": context["weights_manifest_sha256"],
        "model_tree_sha256": context["model_tree_sha256"],
        "tokenizer_tree_sha256": context["tokenizer_tree_sha256"],
    }


def _campaign_training_audit_binding(context: dict[str, Any]) -> dict[str, Any] | None:
    path = context.get("training_provenance_audit_path")
    digest = context.get("training_provenance_audit_sha256")
    if path is None and digest is None:
        return None
    return {"path": path, "sha256": digest}


def _campaign_evaluator(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": context["evaluator_path"],
        "sha256": context["evaluator_sha256"],
        "commit": context["evaluator_commit"],
    }


def _campaign_formal_run(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": context["run_name"],
        "limit": context["limit"],
        "run_dir": context["run_dir"],
    }


def _campaign_receipt_binding(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": context["receipt_path"],
        "armed_sha256": context["receipt_armed_sha256"],
    }


def _expected_candidate_selection(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CANDIDATE_SELECTION_SCHEMA,
        "selection_status": "FROZEN",
        "campaign_id": context["campaign_id"],
        "unique_candidate_count": 1,
        "frozen_test": {
            "sha256": EXPECTED_FROZEN_TEST_SHA256,
            "rows": EXPECTED_FROZEN_ROWS,
        },
        "candidate": _campaign_candidate(context),
        "checkpoint_training_provenance": context[
            "checkpoint_training_provenance"
        ],
        "training_provenance_audit": _campaign_training_audit_binding(context),
        "evaluator": _campaign_evaluator(context),
        "formal_run": _campaign_formal_run(context),
        "authorization_artifact": {
            "path": context["authorization_path"],
            "sha256": context["authorization_sha256"],
        },
        "consumption_receipt": _campaign_receipt_binding(context),
    }


def _expected_campaign_authorization(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_AUTHORIZATION_SCHEMA,
        "authorization_status": "AUTHORIZED",
        "authorization_scope": "one_frozen_test_prediction_run",
        "campaign_id": context["campaign_id"],
        "candidate_frozen_before_test": True,
        "unique_candidate_count": 1,
        "authorized_prediction_runs": 1,
        "selection_artifact_path": context["selection_path"],
        "frozen_test": {
            "sha256": EXPECTED_FROZEN_TEST_SHA256,
            "rows": EXPECTED_FROZEN_ROWS,
        },
        "candidate": _campaign_candidate(context),
        "checkpoint_training_provenance": context[
            "checkpoint_training_provenance"
        ],
        "training_provenance_audit": _campaign_training_audit_binding(context),
        "evaluator": _campaign_evaluator(context),
        "formal_run": _campaign_formal_run(context),
        "consumption_receipt": _campaign_receipt_binding(context),
    }


def validate_campaign_documents(
    selection: dict[str, Any],
    authorization: dict[str, Any],
    *,
    campaign_context: dict[str, Any],
) -> None:
    if campaign_context.get("limit") is not None:
        raise ValueError("campaign must authorize a formal full run with limit=null")
    if selection != _expected_candidate_selection(campaign_context):
        raise ValueError(
            "candidate selection binding mismatch; the committed selection must "
            "exactly bind one candidate, audit, evaluator, authorization, receipt, "
            "test, and formal output directory"
        )
    if authorization != _expected_campaign_authorization(campaign_context):
        raise ValueError(
            "campaign authorization binding mismatch; every frozen-run input must "
            "match the committed candidate selection"
        )


def validate_campaign_authorization(
    manifest: dict[str, Any],
    *,
    dataset_hash: str,
    campaign_context: dict[str, Any],
) -> None:
    """Compatibility wrapper retained for callers that validate authorization only."""

    if dataset_hash != EXPECTED_FROZEN_TEST_SHA256:
        raise ValueError("campaign authorization test hash does not match this run")
    expected = _expected_campaign_authorization(campaign_context)
    if manifest != expected:
        raise ValueError("campaign authorization binding mismatch")


def _expected_armed_receipt(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CONSUMPTION_RECEIPT_SCHEMA,
        "campaign_id": context["campaign_id"],
        "authorization_sha256": context["authorization_sha256"],
        "run_name": context["run_name"],
        "run_dir": context["run_dir"],
        "status": "ARMED",
        "claimed_at_utc": None,
        "claim_nonce": None,
    }


def validate_armed_receipt(
    receipt: dict[str, Any], *, campaign_context: dict[str, Any]
) -> None:
    if receipt != _expected_armed_receipt(campaign_context):
        raise ValueError(
            "campaign consumption receipt is not the exact committed ARMED receipt"
        )


def assess_frozen_leaderboard_eligibility(
    *,
    checkpoint_training_provenance: str,
    dataset_artifact: dict[str, Any],
    metrics: dict[str, Any],
    training_provenance_audit: dict[str, Any] | None,
    campaign_selection_manifest: dict[str, Any] | None,
    campaign_authorization_manifest: dict[str, Any] | None,
    campaign_context: dict[str, Any],
) -> dict[str, Any]:
    if checkpoint_training_provenance not in CHECKPOINT_TRAINING_PROVENANCE_CHOICES:
        raise ValueError(
            "unsupported checkpoint training provenance "
            f"{checkpoint_training_provenance!r}"
        )

    def diagnostic(status: str, reason: str) -> dict[str, Any]:
        return {
            "frozen_leaderboard_eligible": False,
            "diagnostic_only": True,
            "status": status,
            "reason": reason,
            "checkpoint_training_provenance": checkpoint_training_provenance,
        }

    if checkpoint_training_provenance == "external_unknown":
        return diagnostic(
            "DIAGNOSTIC_ONLY_EXTERNAL_UNKNOWN",
            "The third-party checkpoint's original frozen train/dev membership is unproven.",
        )
    if checkpoint_training_provenance == "known_overlap":
        return diagnostic(
            "DIAGNOSTIC_ONLY_KNOWN_OVERLAP",
            "The checkpoint is known to overlap the frozen evaluation partition.",
        )
    if training_provenance_audit is None:
        return diagnostic(
            "DIAGNOSTIC_ONLY_MISSING_PROVENANCE_AUDIT",
            "frozen_train_dev_only was declared without a complete zero-overlap audit.",
        )

    validate_training_provenance_audit(
        training_provenance_audit,
        dataset_hash=str(dataset_artifact["sha256"]),
    )
    formal_dataset = (
        dataset_artifact.get("sha256") == EXPECTED_FROZEN_TEST_SHA256
        and dataset_artifact.get("limit") is None
        and dataset_artifact.get("full_rows") == EXPECTED_FROZEN_ROWS
        and dataset_artifact.get("evaluated_rows") == EXPECTED_FROZEN_ROWS
        and metrics.get("total") == EXPECTED_FROZEN_ROWS
    )
    if not formal_dataset:
        return diagnostic(
            "DIAGNOSTIC_ONLY_INCOMPLETE_FORMAL_RESULT",
            "Frozen eligibility requires the full 2,775-row test and metrics.total=2775.",
        )
    if int(metrics.get("invalid", 0)) != 0:
        return diagnostic(
            "DIAGNOSTIC_ONLY_INVALID_PREDICTIONS",
            "Formal frozen eligibility requires zero invalid predictions.",
        )
    if campaign_selection_manifest is None or campaign_authorization_manifest is None:
        return diagnostic(
            "DIAGNOSTIC_ONLY_MISSING_CAMPAIGN_AUTHORIZATION",
            "The committed candidate selection and one-run authorization are both required.",
        )
    if campaign_authorization_manifest.get("authorization_status") != "AUTHORIZED":
        return diagnostic(
            "DIAGNOSTIC_ONLY_UNAUTHORIZED_CAMPAIGN",
            "Campaign authorization is absent, UNKNOWN, or not AUTHORIZED.",
        )
    validate_campaign_documents(
        campaign_selection_manifest,
        campaign_authorization_manifest,
        campaign_context=campaign_context,
    )
    return {
        "frozen_leaderboard_eligible": True,
        "diagnostic_only": False,
        "status": "ELIGIBLE_AUDITED_FROZEN_TRAIN_DEV_ONLY",
        "reason": (
            "The complete zero-overlap audit and one-candidate/one-run campaign "
            "authorization both match this immutable checkpoint and frozen test."
        ),
        "checkpoint_training_provenance": checkpoint_training_provenance,
    }


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
            f"profile {key!r} is blocked: {profile['block_reason']}"
        )
    if profile.get("profile_requires_frozen_input_contract"):
        source = profile.get("input_template_source")
        if not isinstance(source, dict):
            raise ValueError(
                f"profile {key!r} has no immutable frozen input contract"
            )
        if (
            source.get("revision") != profile["revision"]
            or source.get("sha256") != profile["expected_card_sha256"]
            or source.get("kind") != "immutable_model_card"
        ):
            raise ValueError(
                f"profile {key!r} frozen input contract is not bound to its card/revision"
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
    payload = _read_regular_file_bytes(path)
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != EXPECTED_FROZEN_TEST_SHA256:
        raise ValueError(
            "frozen ESConv test hash mismatch: "
            f"expected={EXPECTED_FROZEN_TEST_SHA256} actual={actual_hash}"
        )
    try:
        lines = payload.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise ValueError(f"frozen ESConv test is not valid UTF-8: {error}") from error
    records = prepare_frozen_lines(
        lines,
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


_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def _is_tokenizer_artifact(relative_path: str) -> bool:
    name = Path(relative_path).name.lower()
    return (
        name in _TOKENIZER_FILENAMES
        or name.startswith("tokenizer")
        or name.startswith("vocab")
        or "sentencepiece" in name
    )


def snapshot_model_tree(model_dir: Path) -> dict[str, Any]:
    """Hash every regular file in a local model tree and reject indirection."""

    model_dir = Path(model_dir)
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError(f"model directory must be a real directory: {model_dir}")
    entries: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(
        model_dir, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"model tree contains a symlink directory: {child}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"model tree contains a non-directory node: {child}")
        for name in sorted(file_names):
            path = directory_path / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"model tree contains a symlink artifact: {path}")
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"model tree contains a non-regular artifact: {path}")
            size, digest = _hash_regular_file(path)
            entries.append(
                {
                    "path": path.relative_to(model_dir).as_posix(),
                    "size": size,
                    "sha256": digest,
                }
            )
    entries.sort(key=lambda item: item["path"])
    tokenizer_entries = [
        item for item in entries if _is_tokenizer_artifact(str(item["path"]))
    ]
    if not tokenizer_entries:
        raise ValueError("model directory contains no tokenizer artifacts")
    return {
        "files": entries,
        "tree_manifest_sha256": canonical_json_sha256(entries),
        "tokenizer_files": tokenizer_entries,
        "tokenizer_manifest_sha256": canonical_json_sha256(tokenizer_entries),
    }


def audit_model_directory(
    model_dir: Path,
    *,
    profile: dict[str, Any],
    declared_revision: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    snapshot = snapshot_model_tree(model_dir)
    entries_by_path = {item["path"]: item for item in snapshot["files"]}
    config_entry = entries_by_path.get("config.json")
    card_entry = entries_by_path.get("README.md")
    if config_entry is None or card_entry is None:
        raise ValueError("model directory must contain config.json and immutable README.md")

    config_hash = config_entry["sha256"]
    if config_hash != profile["expected_config_sha256"]:
        raise ValueError(
            f"config hash mismatch: expected={profile['expected_config_sha256']} "
            f"actual={config_hash}"
        )
    card_hash = card_entry["sha256"]
    if card_hash != profile["expected_card_sha256"]:
        raise ValueError(
            f"model-card hash mismatch: expected={profile['expected_card_sha256']} "
            f"actual={card_hash}"
        )
    config_payload = _read_regular_file_bytes(model_dir / "config.json")
    if hashlib.sha256(config_payload).hexdigest() != config_hash:
        raise ValueError("config.json changed during model audit")
    try:
        config = json.loads(config_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse audited config.json: {error}") from error
    validate_config_contract(config, profile)

    expected_weights = profile["expected_weights"]
    missing = [filename for filename in expected_weights if filename not in entries_by_path]
    if missing:
        raise ValueError(f"model download is incomplete; missing weight files: {missing}")
    weight_entries = []
    for filename, expected in sorted(expected_weights.items()):
        entry = entries_by_path[filename]
        actual_size = entry["size"]
        if actual_size != expected["size"]:
            raise ValueError(
                f"weight size mismatch for {filename}: expected={expected['size']} "
                f"actual={actual_size}"
            )
        actual_hash = entry["sha256"]
        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"weight hash mismatch for {filename}: expected={expected['sha256']} "
                f"actual={actual_hash}"
            )
        weight_entries.append(
            {"name": filename, "size": actual_size, "sha256": actual_hash}
        )
    revision, revision_evidence = _resolve_revision(
        model_dir, profile, declared_revision
    )

    artifact = {
        "source_model_id": profile["model_id"],
        "revision": revision,
        "revision_evidence": revision_evidence,
        "path": str(model_dir.resolve()),
        "tree_manifest_sha256": snapshot["tree_manifest_sha256"],
        "tokenizer_manifest_sha256": snapshot["tokenizer_manifest_sha256"],
        "tree": {"files": snapshot["files"]},
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
            "files": snapshot["tokenizer_files"],
            "manifest_sha256": snapshot["tokenizer_manifest_sha256"],
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


def load_hf_model_verified(
    model_dir: Path,
    *,
    profile: dict[str, Any],
    requested_device: str,
    declared_revision: str | None,
    pre_load_artifact: dict[str, Any],
) -> tuple[Any, Any, Any, dict[str, str]]:
    """Load only between two identical complete model-tree audits."""

    loaded = load_hf_model(model_dir, profile, requested_device)
    post_load_artifact, _ = audit_model_directory(
        model_dir,
        profile=profile,
        declared_revision=declared_revision,
    )
    for key in (
        "tree_manifest_sha256",
        "tokenizer_manifest_sha256",
    ):
        if post_load_artifact.get(key) != pre_load_artifact.get(key):
            raise ValueError(
                f"model artifacts changed during load ({key}); refusing inference"
            )
    if post_load_artifact.get("weights", {}).get(
        "manifest_sha256"
    ) != pre_load_artifact.get("weights", {}).get("manifest_sha256"):
        raise ValueError("model weights changed during load; refusing inference")
    return loaded


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
    _atomic_write_bytes(path, content.encode("utf-8"), mode=0o644)


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


def validate_formal_result_integrity(
    *,
    records: Sequence[dict[str, Any]],
    dataset_artifact: dict[str, Any],
    metrics: dict[str, Any],
    predictions_path: Path,
) -> dict[str, Any]:
    """Recompute and cross-check every count, metric, row, and prediction byte."""

    if len(records) != EXPECTED_FROZEN_ROWS:
        raise ValueError(
            f"formal result requires exactly {EXPECTED_FROZEN_ROWS} records, "
            f"found {len(records)}"
        )
    expected_dataset = {
        "sha256": EXPECTED_FROZEN_TEST_SHA256,
        "full_rows": EXPECTED_FROZEN_ROWS,
        "evaluated_rows": EXPECTED_FROZEN_ROWS,
        "limit": None,
    }
    for key, expected in expected_dataset.items():
        if dataset_artifact.get(key) != expected:
            raise ValueError(
                f"formal result dataset {key} must equal {expected!r}, "
                f"found {dataset_artifact.get(key)!r}"
            )
    if metrics.get("total") != EXPECTED_FROZEN_ROWS:
        raise ValueError(
            f"formal metrics.total must equal {EXPECTED_FROZEN_ROWS}"
        )
    recomputed_metrics = compute_metrics(records)
    if canonical_json_sha256(metrics) != canonical_json_sha256(recomputed_metrics):
        raise ValueError("reported metrics do not equal metrics recomputed from records")
    payload = _read_regular_file_bytes(Path(predictions_path))
    lines = payload.splitlines()
    if len(lines) != EXPECTED_FROZEN_ROWS:
        raise ValueError(
            f"prediction JSONL must contain {EXPECTED_FROZEN_ROWS} rows, "
            f"found {len(lines)}"
        )
    for index, (line, record) in enumerate(zip(lines, records, strict=True)):
        try:
            actual = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"prediction row {index} is invalid JSON: {error}") from error
        expected = public_prediction(record)
        if actual != expected:
            raise ValueError(f"prediction row {index} does not match scored record")
    return {
        "path": str(Path(predictions_path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(lines),
    }


def validate_run_directory(run_dir: Path) -> Path:
    """Require one new absolute directory beneath a real, non-symlink parent."""

    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        raise ValueError("authorized run directory must be an absolute path")
    parent = run_dir.parent
    if not parent.is_dir():
        raise ValueError(f"authorized run directory parent does not exist: {parent}")
    if parent.is_symlink():
        raise ValueError(f"authorized run directory parent is a symlink: {parent}")
    # Detect a direct-parent symlink even on systems where /var itself is an
    # intentional platform symlink (for example macOS temporary directories).
    expected_parent = parent.parent.resolve() / parent.name
    if parent.resolve() != expected_parent:
        raise ValueError(f"authorized run directory parent resolves through a symlink: {parent}")
    if run_dir.exists() or run_dir.is_symlink():
        raise ValueError(f"authorized run directory already exists: {run_dir}")
    if run_dir.name in {"", ".", ".."}:
        raise ValueError("authorized run directory needs a concrete final component")
    return run_dir


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        os.chmod(temporary, mode)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _git_directory(repo_root: Path) -> Path:
    raw = _run_git(Path(repo_root), ["rev-parse", "--git-dir"], text=True).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def campaign_claim_path(repo_root: Path) -> Path:
    """One repository-local claim for the frozen test, independent of campaign id."""

    claim_directory = _git_directory(Path(repo_root)) / "esconv-frozen-claims"
    if claim_directory.exists() or claim_directory.is_symlink():
        if claim_directory.is_symlink() or not claim_directory.is_dir():
            raise ValueError(f"campaign claim directory is not a real directory: {claim_directory}")
    else:
        claim_directory.mkdir(mode=0o700, parents=True)
    return claim_directory / f"{EXPECTED_FROZEN_TEST_SHA256}.claim.json"


def consume_campaign_slot(
    *,
    repo_root: Path,
    receipt_artifact: dict[str, Any],
    campaign_context: dict[str, Any],
) -> Path:
    """Persistently claim the sole frozen run, then atomically dirty its receipt."""

    if receipt_artifact.get("sha256") != campaign_context["receipt_armed_sha256"]:
        raise ValueError("campaign receipt does not match its armed SHA-256 binding")
    validate_armed_receipt(
        receipt_artifact["document"], campaign_context=campaign_context
    )
    receipt_path = Path(receipt_artifact["path"])
    current = load_json_artifact(receipt_path, description="campaign consumption receipt")
    if (
        current["sha256"] != receipt_artifact["sha256"]
        or current["document"] != receipt_artifact["document"]
    ):
        raise ValueError("campaign receipt changed before the slot could be claimed")

    claim_path = campaign_claim_path(repo_root)
    claim_nonce = secrets.token_hex(16)
    claimed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    claim_document = {
        "schema_version": CONSUMPTION_RECEIPT_SCHEMA,
        "status": "CLAIMED",
        "frozen_test_sha256": EXPECTED_FROZEN_TEST_SHA256,
        "campaign_id": campaign_context["campaign_id"],
        "authorization_sha256": campaign_context["authorization_sha256"],
        "receipt_path": receipt_artifact["repo_relative_path"],
        "run_name": campaign_context["run_name"],
        "run_dir": campaign_context["run_dir"],
        "claimed_at_utc": claimed_at,
        "claim_nonce": claim_nonce,
        "pid": os.getpid(),
    }
    payload = (
        json.dumps(claim_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(
            "frozen campaign slot was already claimed; replay, a second manifest, "
            "and concurrent runs are forbidden"
        ) from error
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(claim_path.parent)

    consumed = copy.deepcopy(receipt_artifact["document"])
    consumed.update(
        {
            "status": "CONSUMED",
            "claimed_at_utc": claimed_at,
            "claim_nonce": claim_nonce,
            "claim_path": str(claim_path),
        }
    )
    consumed_payload = (
        json.dumps(consumed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        _atomic_write_bytes(receipt_path, consumed_payload)
    except Exception as error:
        raise ValueError(
            "campaign claim persisted but tracked receipt consumption failed; "
            "campaign remains burned and requires audit"
        ) from error
    return claim_path


def _profile_summary(key: str, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": key,
        "source_model_id": profile["model_id"],
        "immutable_revision": profile["revision"],
        "input_template": profile["template"],
        "input_template_source": profile.get("input_template_source"),
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
    checkpoint_training_provenance: str,
    training_provenance_audit_artifact: dict[str, Any] | None,
    campaign_selection_artifact: dict[str, Any],
    campaign_authorization_artifact: dict[str, Any] | None,
    campaign_consumption_receipt_artifact: dict[str, Any],
    campaign_claim: Path,
    evaluator_artifact: dict[str, Any],
    campaign_context: dict[str, Any],
    published_predictions_path: Path | None = None,
) -> dict[str, Any]:
    invalid_reasons: dict[str, int] = {}
    for record in records:
        if record.get("invalid"):
            reason = str(record.get("invalid_reason", "unspecified"))
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    training_audit_document = (
        training_provenance_audit_artifact["document"]
        if training_provenance_audit_artifact is not None
        else None
    )
    campaign_document = (
        campaign_authorization_artifact["document"]
        if campaign_authorization_artifact is not None
        else None
    )
    selection_document = campaign_selection_artifact["document"]
    predictions_artifact = validate_formal_result_integrity(
        records=records,
        dataset_artifact=dataset_artifact,
        metrics=metrics,
        predictions_path=predictions_path,
    )
    if published_predictions_path is not None:
        predictions_artifact["path"] = str(published_predictions_path.resolve())
    eligibility = assess_frozen_leaderboard_eligibility(
        checkpoint_training_provenance=checkpoint_training_provenance,
        dataset_artifact=dataset_artifact,
        metrics=metrics,
        training_provenance_audit=training_audit_document,
        campaign_selection_manifest=selection_document,
        campaign_authorization_manifest=campaign_document,
        campaign_context=campaign_context,
    )
    if metrics["invalid"]:
        audit_status = "COMPLETE_WITH_INVALID"
    elif eligibility["diagnostic_only"]:
        audit_status = eligibility["status"]
    else:
        audit_status = "COMPLETE"
    return {
        "audit_status": audit_status,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "task": "8-way next-support-strategy prediction",
        "profile": _profile_summary(profile_key, profile),
        "model": model_artifact,
        "dataset": dataset_artifact,
        "predictions": predictions_artifact,
        "metrics": metrics,
        "invalid_reasons": invalid_reasons,
        "checkpoint_training_provenance": checkpoint_training_provenance,
        "training_provenance_audit": training_provenance_audit_artifact,
        "campaign_selection": campaign_selection_artifact,
        "campaign_authorization": campaign_authorization_artifact,
        "campaign_consumption": {
            "armed_receipt": campaign_consumption_receipt_artifact,
            "claim_path": str(campaign_claim),
            "status": "CONSUMED",
        },
        "eligibility": eligibility,
        "diagnostic_only": eligibility["diagnostic_only"],
        "protocol": {
            "frozen_2775": True,
            "prefix_smoke": False,
            "limit_disabled": True,
            "target_strategy_excluded_from_model_input": True,
            "target_response_excluded_from_model_input_and_outputs": True,
            "prior_supporter_strategy_in_input": (
                profile["template"] == "heegyu_context_with_prior_strategy"
            ),
            "decision_rule": "argmax over exactly eight sequence-classification logits",
            "test_tuning_prohibited": True,
            "modernbert_template_selection_on_test_prohibited": True,
            "invalid_rows_retained_in_accuracy_denominator": True,
            "campaign_authorization_required_before_any_frozen_test_prediction": True,
            "campaign_committed_clean_hash_anchors_required": True,
            "single_repository_claim_and_tracked_receipt": True,
            "receipt_reset_or_new_clone_cannot_be_prevented_by_local_filesystem_only": True,
            "formal_outputs_may_not_be_overwritten": True,
        },
        "runtime": runtime,
        "evaluator": evaluator_artifact,
        "eligible_for_frozen_leaderboard": eligibility[
            "frozen_leaderboard_eligible"
        ],
    }


def validate_static_run_arguments(args: argparse.Namespace) -> None:
    if getattr(args, "limit", None) is not None:
        raise ValueError(
            "--limit is disabled for the frozen evaluator; use a dev fixture for smoke tests"
        )
    if getattr(args, "overwrite", False):
        raise ValueError("frozen campaign outputs can never be overwritten")
    batch_size = getattr(args, "batch_size", None)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    provenance = getattr(args, "checkpoint_training_provenance", None)
    if provenance not in CHECKPOINT_TRAINING_PROVENANCE_CHOICES:
        raise ValueError(f"unsupported checkpoint training provenance {provenance!r}")
    required = (
        "campaign_trusted_commit",
        "campaign_selection_manifest",
        "campaign_selection_expected_sha256",
        "campaign_authorization_manifest",
        "campaign_authorization_expected_sha256",
        "campaign_consumption_receipt",
        "campaign_consumption_receipt_expected_sha256",
        "run_dir",
    )
    for name in required:
        if getattr(args, name, None) in (None, ""):
            raise ValueError(f"--{name.replace('_', '-')} is required")
    if not _HEX_REVISION.fullmatch(str(args.campaign_trusted_commit)):
        raise ValueError("campaign trusted commit must be 40 lowercase hex characters")
    for name in (
        "campaign_selection_expected_sha256",
        "campaign_authorization_expected_sha256",
        "campaign_consumption_receipt_expected_sha256",
    ):
        _validate_expected_sha256(
            str(getattr(args, name)), description=name.replace("_", " ")
        )
    audit_path = getattr(args, "training_provenance_audit", None)
    audit_sha = getattr(args, "training_provenance_audit_expected_sha256", None)
    if (audit_path is None) != (audit_sha is None):
        raise ValueError(
            "training provenance audit path and expected SHA-256 must be supplied together"
        )
    if provenance == "frozen_train_dev_only" and audit_path is None:
        raise ValueError(
            "frozen_train_dev_only requires a committed training provenance "
            "audit and expected SHA-256"
        )
    if audit_sha is not None:
        _validate_expected_sha256(str(audit_sha), description="training provenance audit")


def _assert_single_selection_artifact(
    repo_root: Path, *, trusted_commit: str, selected_path: str
) -> None:
    paths = _run_git(
        repo_root, ["ls-tree", "-r", "--name-only", trusted_commit], text=True
    ).splitlines()
    matches: list[str] = []
    marker = CANDIDATE_SELECTION_SCHEMA.encode("utf-8")
    for path in paths:
        if not path.endswith(".json"):
            continue
        payload = _run_git(repo_root, ["show", f"{trusted_commit}:{path}"], text=False)
        if marker not in payload:
            continue
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(document, dict)
            and document.get("schema_version") == CANDIDATE_SELECTION_SCHEMA
        ):
            matches.append(path)
    if matches != [selected_path]:
        raise ValueError(
            "trusted campaign commit must contain exactly one candidate selection; "
            f"selected={selected_path!r} found={matches!r}"
        )


def _build_runtime_campaign_context(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    selection_artifact: dict[str, Any],
    authorization_artifact: dict[str, Any],
    receipt_artifact: dict[str, Any],
    training_audit_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    selection = selection_artifact["document"]
    candidate = selection.get("candidate")
    evaluator = selection.get("evaluator")
    if not isinstance(candidate, dict) or not isinstance(evaluator, dict):
        raise ValueError("candidate selection lacks candidate/evaluator bindings")
    audit_path = (
        training_audit_artifact["repo_relative_path"]
        if training_audit_artifact is not None
        else None
    )
    audit_sha = (
        training_audit_artifact["sha256"]
        if training_audit_artifact is not None
        else None
    )
    return {
        "campaign_id": selection.get("campaign_id"),
        "trusted_commit": args.campaign_trusted_commit,
        "run_name": args.run_name,
        "profile_key": args.profile,
        "model_id": profile["model_id"],
        "revision": profile["revision"],
        "weights_manifest_sha256": candidate.get("weights_manifest_sha256"),
        "model_tree_sha256": candidate.get("model_tree_sha256"),
        "tokenizer_tree_sha256": candidate.get("tokenizer_tree_sha256"),
        "checkpoint_training_provenance": args.checkpoint_training_provenance,
        "training_provenance_audit_path": audit_path,
        "training_provenance_audit_sha256": audit_sha,
        "evaluator_path": evaluator.get("path"),
        "evaluator_sha256": evaluator.get("sha256"),
        "evaluator_commit": evaluator.get("commit"),
        "selection_path": selection_artifact["repo_relative_path"],
        "selection_sha256": selection_artifact["sha256"],
        "authorization_path": authorization_artifact["repo_relative_path"],
        "authorization_sha256": authorization_artifact["sha256"],
        "receipt_path": receipt_artifact["repo_relative_path"],
        "receipt_armed_sha256": receipt_artifact["sha256"],
        "limit": None,
        "run_dir": str(Path(args.run_dir).absolute()),
    }


def _load_campaign_preflight(
    args: argparse.Namespace, *, profile: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
]:
    trusted_commit = args.campaign_trusted_commit
    selection = load_committed_json_artifact(
        args.campaign_selection_manifest,
        description="candidate selection manifest",
        repo_root=ROOT,
        trusted_commit=trusted_commit,
        expected_sha256=args.campaign_selection_expected_sha256,
    )
    authorization = load_committed_json_artifact(
        args.campaign_authorization_manifest,
        description="campaign authorization manifest",
        repo_root=ROOT,
        trusted_commit=trusted_commit,
        expected_sha256=args.campaign_authorization_expected_sha256,
    )
    receipt = load_committed_json_artifact(
        args.campaign_consumption_receipt,
        description="campaign consumption receipt",
        repo_root=ROOT,
        trusted_commit=trusted_commit,
        expected_sha256=args.campaign_consumption_receipt_expected_sha256,
    )
    audit = None
    if args.training_provenance_audit is not None:
        audit = load_committed_json_artifact(
            args.training_provenance_audit,
            description="checkpoint training provenance audit",
            repo_root=ROOT,
            trusted_commit=trusted_commit,
            expected_sha256=args.training_provenance_audit_expected_sha256,
        )
    context = _build_runtime_campaign_context(
        args=args,
        profile=profile,
        selection_artifact=selection,
        authorization_artifact=authorization,
        receipt_artifact=receipt,
        training_audit_artifact=audit,
    )
    validate_campaign_documents(
        selection["document"],
        authorization["document"],
        campaign_context=context,
    )
    validate_armed_receipt(receipt["document"], campaign_context=context)
    _assert_single_selection_artifact(
        ROOT,
        trusted_commit=trusted_commit,
        selected_path=selection["repo_relative_path"],
    )
    if context["evaluator_path"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise ValueError("campaign evaluator path does not identify this evaluator")
    if not _HEX_REVISION.fullmatch(str(context["evaluator_commit"])):
        raise ValueError("campaign evaluator commit is not an immutable git commit")
    _run_git(
        ROOT,
        [
            "merge-base",
            "--is-ancestor",
            context["evaluator_commit"],
            trusted_commit,
        ],
        text=False,
    )
    evaluator = load_committed_file_artifact(
        Path(__file__),
        description="frozen evaluator",
        repo_root=ROOT,
        trusted_commit=context["evaluator_commit"],
        expected_sha256=context["evaluator_sha256"],
    )
    evaluator.pop("_payload", None)
    if args.checkpoint_training_provenance == "frozen_train_dev_only":
        if audit is None:
            raise ValueError("frozen_train_dev_only requires a provenance audit")
        validate_training_provenance_audit(
            audit["document"], dataset_hash=EXPECTED_FROZEN_TEST_SHA256
        )
    return selection, authorization, receipt, audit, evaluator, context


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_static_run_arguments(args)
    profile = get_enabled_profile(args.profile)
    (
        campaign_selection_artifact,
        campaign_authorization_artifact,
        campaign_consumption_receipt_artifact,
        training_provenance_audit_artifact,
        evaluator_artifact,
        campaign_context,
    ) = _load_campaign_preflight(args, profile=profile)
    run_dir = validate_run_directory(Path(args.run_dir))
    model_artifact, _ = audit_model_directory(
        args.model_dir,
        profile=profile,
        declared_revision=args.model_revision,
    )
    local_candidate = {
        "profile_key": args.profile,
        "model_id": model_artifact["source_model_id"],
        "revision": model_artifact["revision"],
        "weights_manifest_sha256": model_artifact["weights"]["manifest_sha256"],
        "model_tree_sha256": model_artifact["tree_manifest_sha256"],
        "tokenizer_tree_sha256": model_artifact["tokenizer_manifest_sha256"],
    }
    if local_candidate != _campaign_candidate(campaign_context):
        raise ValueError(
            "local model/tokenizer tree does not match the frozen campaign candidate"
        )
    claim_path = consume_campaign_slot(
        repo_root=ROOT,
        receipt_artifact=campaign_consumption_receipt_artifact,
        campaign_context=campaign_context,
    )

    # The first frozen-test read occurs only after the persistent single-use claim.
    records, dataset_artifact = prepare_frozen_file(
        args.test_file, profile=profile, limit=None
    )
    model, tokenizer, device, versions = load_hf_model_verified(
        args.model_dir,
        profile=profile,
        requested_device=args.device,
        declared_revision=args.model_revision,
        pre_load_artifact=model_artifact,
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
    staging = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}.staging-", dir=run_dir.parent)
    )
    predictions_staging = staging / "predictions.jsonl"
    summary_staging = staging / "summary.json"
    predictions_published = run_dir / "predictions.jsonl"
    write_predictions(predictions_staging, records)
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
        predictions_path=predictions_staging,
        records=records,
        metrics=metrics,
        runtime=runtime,
        checkpoint_training_provenance=args.checkpoint_training_provenance,
        training_provenance_audit_artifact=training_provenance_audit_artifact,
        campaign_selection_artifact=campaign_selection_artifact,
        campaign_authorization_artifact=campaign_authorization_artifact,
        campaign_consumption_receipt_artifact=campaign_consumption_receipt_artifact,
        campaign_claim=claim_path,
        evaluator_artifact=evaluator_artifact,
        campaign_context=campaign_context,
        published_predictions_path=predictions_published,
    )
    write_json(summary_staging, summary)
    _fsync_directory(staging)
    validate_run_directory(run_dir)
    os.rename(staging, run_dir)
    _fsync_directory(run_dir.parent)
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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--checkpoint-training-provenance",
        choices=CHECKPOINT_TRAINING_PROVENANCE_CHOICES,
        required=True,
    )
    parser.add_argument(
        "--training-provenance-audit",
        type=Path,
        help=(
            "JSON using esconv-checkpoint-training-provenance-audit-v1; required "
            "for leaderboard eligibility and must bind zero overlap to the test hash"
        ),
    )
    parser.add_argument("--training-provenance-audit-expected-sha256")
    parser.add_argument("--campaign-trusted-commit", required=True)
    parser.add_argument("--campaign-selection-manifest", type=Path, required=True)
    parser.add_argument("--campaign-selection-expected-sha256", required=True)
    parser.add_argument(
        "--campaign-authorization-manifest",
        type=Path,
        required=True,
        help=(
            "Committed one-candidate/one-prediction-run authorization"
        ),
    )
    parser.add_argument("--campaign-authorization-expected-sha256", required=True)
    parser.add_argument("--campaign-consumption-receipt", type=Path, required=True)
    parser.add_argument(
        "--campaign-consumption-receipt-expected-sha256", required=True
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
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
                "eligibility_status": summary["eligibility"]["status"],
                "eligible_for_frozen_leaderboard": summary[
                    "eligible_for_frozen_leaderboard"
                ],
                "summary": str((args.run_dir / "summary.json").resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
