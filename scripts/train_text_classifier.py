#!/usr/bin/env python3
"""Leakage-aware, task-specific text-classification trainer.

The command is a metadata-only preregistration by default and does not open the
declared external CSVs. ``--validate-data`` explicitly hashes and validates the
training and validation CSVs without loading a model. The standalone CLI never
trains; a validated benchmark adapter must call :func:`execute_training` with a
registry-bound config. Official test and holdout paths are deliberately
unsupported because model selection belongs on the declared validation split.

Runtime dependencies are intentionally not installed by this script. Training
uses the already provisioned PyTorch, Transformers, NumPy, and scikit-learn
packages; missing dependencies fail with an actionable message. ``--selftest``
uses only synthetic CSV rows and tensors and never downloads a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "text-classifier"
DEFAULT_MODEL = "FacebookAI/roberta-base"
DEFAULT_MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
DEFAULT_TOKENIZER = DEFAULT_MODEL
DEFAULT_TOKENIZER_REVISION = DEFAULT_MODEL_REVISION
DEFAULT_HEAD_TYPE = "independent_sequence_classification_head"
LOSS_NAMES = ("ce", "weighted-ce", "focal")
LABEL_MODES = ("exact", "imhi-response-prefix")
DEVICE_NAMES = ("auto", "cpu", "cuda", "mps")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_SELECTION_TOKEN_RE = re.compile(
    r"(?:^|[-_.])(test|testing|holdout)(?:$|[-_.])", re.IGNORECASE
)


@dataclass(frozen=True)
class Example:
    """One parsed classifier example kept only in process memory."""

    row_number: int
    text: str
    label: str
    label_id: int
    content_sha256: str
    cross_split_group_sha256: str
    example_sha256: str


@dataclass(frozen=True)
class LoadedSplit:
    """Parsed examples plus a text-free provenance manifest."""

    role: str
    examples: tuple[Example, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PreparedData:
    """Leakage-safe training/validation rows and their aggregate manifest."""

    train: tuple[Example, ...]
    valid: tuple[Example, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class TrainerConfig:
    """Frozen inputs that define one independently resumable classifier run."""

    task_id: str
    run_id: str
    train_csv: Path
    valid_csv: Path
    labels: tuple[str, ...]
    text_columns: tuple[str, ...] = ("post",)
    cross_split_leakage_columns: tuple[str, ...] = ("post",)
    label_column: str = "label"
    label_mode: str = "exact"
    model: str = DEFAULT_MODEL
    model_revision: str = DEFAULT_MODEL_REVISION
    tokenizer: str = DEFAULT_TOKENIZER
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION
    head_type: str = DEFAULT_HEAD_TYPE
    loss: str = "ce"
    seed: int = 42
    output_root: Path = DEFAULT_OUTPUT_ROOT
    device: str = "auto"
    max_length: int = 256
    train_batch: int = 8
    eval_batch: int = 16
    gradient_accumulation: int = 1
    epochs: float = 5.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    focal_gamma: float = 2.0
    early_stopping_patience: int = 2
    registry_path: str = ""
    registry_bundle_sha256: str = ""
    registry_index_sha256: str = ""
    registry_schema_sha256: str = ""
    task_spec_sha256: str = ""
    supervised_profile_id: str = ""
    supervised_profile_sha256: str = ""


def _dependency(name: str, purpose: str) -> Any:
    """Import a provisioned dependency or stop with a clear remediation."""

    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            f'missing required dependency "{name}" for {purpose}; '
            "use the repository's provisioned Python environment (this script "
            "does not install packages)"
        ) from error


def sha256_bytes(value: bytes) -> str:
    """Return the hexadecimal SHA-256 digest for bytes."""

    return hashlib.sha256(value).hexdigest()


def installed_version(distribution: str) -> str:
    """Return an installed dependency version without importing its runtime."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def canonical_json(value: Any) -> str:
    """Serialize an identity object deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def commitment(digests: Iterable[str]) -> str:
    """Commit to an order-sensitive list without exposing row content."""

    return sha256_bytes("\n".join(digests).encode("utf-8"))


def atomic_write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    """Atomically write a JSON artifact, replacing only the named file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    temporary.replace(path)


def git_identity() -> dict[str, Any]:
    """Return a text-free, content-sensitive commitment to repository state."""

    def run(*arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                ["git", "-C", str(ROOT), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("unable to resolve repository Git identity") from error
        return completed.stdout

    head = run("rev-parse", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise RuntimeError("Git HEAD is not a full lowercase commit SHA")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    entries = [line for line in status.splitlines() if line]
    tracked_diff = run("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked_output = run(
        "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked_files: list[dict[str, Any]] = []
    for raw_relative in sorted(value for value in untracked_output.split(b"\0") if value):
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
            kind = "symlink"
        elif path.is_file():
            payload = path.read_bytes()
            kind = "file"
        else:
            raise RuntimeError(f"untracked Git path is not hashable: {relative}")
        untracked_files.append(
            {
                "path": relative,
                "kind": kind,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return {
        "head": head,
        "dirty": bool(entries),
        "status_entry_count": len(entries),
        "status_sha256": sha256_bytes(status),
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "untracked_file_count": len(untracked_files),
        "untracked_tree_sha256": sha256_bytes(
            canonical_json(untracked_files).encode("utf-8")
        ),
        "status_paths_persisted": False,
    }


def training_schedule(config: TrainerConfig, train_rows: int) -> dict[str, Any]:
    """Compute the exact integer optimizer and warmup step schedule."""

    if train_rows < 1:
        raise ValueError("train_rows must be positive")
    batches_per_epoch = math.ceil(train_rows / config.train_batch)
    updates_per_epoch = math.ceil(
        batches_per_epoch / config.gradient_accumulation
    )
    total_update_steps = math.ceil(config.epochs * updates_per_epoch)
    warmup_steps = math.ceil(total_update_steps * config.warmup_ratio)
    return {
        "train_rows": train_rows,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "total_optimizer_update_steps": total_update_steps,
        "frozen_warmup_ratio": config.warmup_ratio,
        "warmup_steps": warmup_steps,
        "rounding": "ceil at batch, accumulation, total-update, and warmup stages",
    }


def artifact_tree_manifest(root: Path) -> dict[str, Any]:
    """Hash every regular file in an artifact tree without following symlinks."""

    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or root.is_symlink():
        raise ValueError(f"artifact root must be a regular directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"artifact tree contains a non-regular file: {path}")
        payload = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    if not files:
        raise ValueError(f"artifact tree contains no files: {root}")
    return {
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "checkpoint_sha256": sha256_bytes(
            canonical_json(files).encode("utf-8")
        ),
    }


def _validated_registry_path(value: str) -> Path:
    """Resolve a registry path and require it to remain inside this checkout."""

    candidate = Path(value)
    path = (candidate if candidate.is_absolute() else ROOT / candidate).resolve(
        strict=True
    )
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("registry path must remain inside the repository") from error
    if not path.is_file() or path.is_symlink():
        raise ValueError("registry path must be a regular non-symlink file")
    return path


def _registry_checker_output(registry_path: Path, *extra: str) -> str:
    completed = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts" / "check_benchmark_registry.py"),
            "--registry",
            str(registry_path),
            *extra,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"benchmark registry validation failed: {detail}")
    return completed.stdout


def _path_has_suffix(path: str, suffix: Path) -> bool:
    actual_parts = Path(path).resolve(strict=True).parts
    expected_parts = suffix.parts
    return (
        len(actual_parts) >= len(expected_parts)
        and actual_parts[-len(expected_parts):] == expected_parts
    )


def validate_registry_execution_proof(
    config: TrainerConfig,
    data: PreparedData,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Independently revalidate registry, profile, config, data, and plan."""

    validate_config(config)
    if not config.registry_path:
        raise ValueError("training requires a benchmark-registry binding")
    registry_path = _validated_registry_path(config.registry_path)
    pass_output = _registry_checker_output(registry_path)
    if "benchmark registry PASS:" not in pass_output:
        raise ValueError("benchmark registry checker did not emit its PASS marker")
    try:
        hashes = json.loads(
            _registry_checker_output(registry_path, "--print-hashes")
        )
        manifest = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("registry or validator hash output is invalid JSON") from error
    tasks: list[dict[str, Any]] = []
    for relative in manifest.get("task_files", []):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("registry contains an unsafe task file path")
        task_path = registry_path.parent / relative_path
        if not task_path.is_file() or task_path.is_symlink():
            raise ValueError("registry task file must be regular and non-symlink")
        records = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("registry task file must contain a JSON array")
        tasks.extend(records)
    observed_hashes = {
        "schema_sha256": sha256_bytes(
            (registry_path.parent / "schema.json").read_bytes()
        ),
        "registry_index_sha256": sha256_bytes(registry_path.read_bytes()),
        "registry_bundle_sha256": sha256_bytes(
            canonical_json({"manifest": manifest, "tasks": tasks}).encode("utf-8")
        ),
        "task_spec_sha256": {
            task["task_key"]: sha256_bytes(canonical_json(task).encode("utf-8"))
            for task in tasks
        },
    }
    if hashes != observed_hashes:
        raise ValueError("registry changed after validator authorization")
    by_key = {task["task_key"]: task for task in tasks}
    if config.task_id not in by_key:
        raise ValueError("registry does not contain the configured task")
    task = by_key[config.task_id]
    profile = task.get("supervised_classifier_profile")
    if not isinstance(profile, dict):
        raise ValueError("task has no supervised classifier profile")
    profile_sha = sha256_bytes(canonical_json(profile).encode("utf-8"))
    expected_binding = {
        "registry_path": config.registry_path,
        "registry_bundle_sha256": hashes["registry_bundle_sha256"],
        "registry_index_sha256": hashes["registry_index_sha256"],
        "registry_schema_sha256": hashes["schema_sha256"],
        "task_spec_sha256": hashes["task_spec_sha256"][config.task_id],
        "supervised_profile_id": profile["profile_id"],
        "supervised_profile_sha256": profile_sha,
    }
    configured_binding = {
        "registry_path": config.registry_path,
        "registry_bundle_sha256": config.registry_bundle_sha256,
        "registry_index_sha256": config.registry_index_sha256,
        "registry_schema_sha256": config.registry_schema_sha256,
        "task_spec_sha256": config.task_spec_sha256,
        "supervised_profile_id": config.supervised_profile_id,
        "supervised_profile_sha256": config.supervised_profile_sha256,
    }
    if configured_binding != expected_binding:
        raise ValueError("configured registry proof does not match the live registry")
    if plan.get("identity", {}).get("registry_binding") != expected_binding:
        raise ValueError("run plan registry proof does not match the live registry")
    if plan.get("identity_sha256") != sha256_bytes(
        canonical_json(plan.get("identity")).encode("utf-8")
    ):
        raise ValueError("run plan identity hash is invalid")
    rebuilt_plan = build_plan(
        config,
        data,
        repository_git=plan["identity"]["git"],
    )
    if (
        canonical_json(rebuilt_plan["identity"])
        != canonical_json(plan["identity"])
        or rebuilt_plan["identity_sha256"] != plan["identity_sha256"]
        or rebuilt_plan["output_dir"] != plan["output_dir"]
    ):
        raise ValueError("run plan differs from config, data, or implementation")

    model = profile["model"]
    if (
        profile.get("task_type") != "single_label_text_classification"
        or profile.get("input_parser")
        != "csv-dictreader-utf8-sig-newline-join-v1"
        or profile.get("data_cleaning_order")
        != [
            "same_label_content_keep_first",
            "conflicting_label_content_drop_group",
            "train_dev_content_overlap_drop_train_keep_dev",
        ]
    ):
        raise ValueError("registry parser/task/cleaning contract is incompatible")
    expected_config = {
        "labels": tuple(profile["labels"]),
        "text_columns": tuple(profile["text_columns"]),
        "cross_split_leakage_columns": tuple(
            profile["cross_split_leakage_columns"]
        ),
        "label_column": profile["label_column"],
        "label_mode": profile["label_parser"],
        "model": model["base_model_id"],
        "model_revision": model["base_model_revision"],
        "tokenizer": model["tokenizer_id"],
        "tokenizer_revision": model["tokenizer_revision"],
        "head_type": model["head_type"],
        "seed": profile["seed"],
    }
    for field, expected in expected_config.items():
        if getattr(config, field) != expected:
            raise ValueError(f"config field {field} differs from registry profile")
    if config.loss not in profile["loss_arms"]:
        raise ValueError("configured loss is not a registry loss arm")
    if profile["primary_metric"] != "weighted_f1" or profile["test_access"]:
        raise ValueError("registry selection/test policy is incompatible")
    protocol = profile["training_protocol"]
    expected_protocol = {
        "max_length": config.max_length,
        "train_batch_size": config.train_batch,
        "eval_batch_size": config.eval_batch,
        "gradient_accumulation_steps": config.gradient_accumulation,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "optimizer": "adamw_torch",
        "scheduler": "linear",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "early_stopping_patience": config.early_stopping_patience,
        "best_model_metric": "weighted_f1",
        "greater_is_better": True,
        "tie_break_rule": "first_strict_improvement_checkpoint",
        "focal_gamma": config.focal_gamma,
        "class_weighting": "balanced_inverse_frequency_after_cleaning",
        "max_grad_norm": 1.0,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": False,
    }
    if protocol != expected_protocol:
        raise ValueError("training protocol differs from registry profile")
    dataset_root = Path(task["dataset"]["relative_root"])
    if (
        str(config.train_csv.expanduser().resolve(strict=True))
        != data.manifest["train_source"]["path"]
        or str(config.valid_csv.expanduser().resolve(strict=True))
        != data.manifest["valid_source"]["path"]
    ):
        raise ValueError("prepared data paths differ from configured train/dev paths")
    if not _path_has_suffix(
        data.manifest["train_source"]["path"],
        dataset_root / profile["train_relative_path"],
    ) or not _path_has_suffix(
        data.manifest["valid_source"]["path"],
        dataset_root / profile["dev_relative_path"],
    ):
        raise ValueError("train/dev paths differ from registry profile")
    observed_rows = {
        "source_rows": {
            "train": data.manifest["train_source"]["rows"],
            "dev": data.manifest["valid_source"]["rows"],
        },
        "used_rows": {
            "train": data.manifest["used_train_rows"],
            "dev": data.manifest["used_valid_rows"],
        },
    }
    if observed_rows["source_rows"] != profile["source_rows"]:
        raise ValueError("source row counts differ from registry profile")
    if observed_rows["used_rows"] != profile["used_rows"]:
        raise ValueError("cleaned row counts differ from registry profile")
    commitments = profile["split_commitments"]
    observed_commitments = {
        "hash_algorithm": "sha256",
        "commitment_scope": "raw_file_bytes_before_parsing",
        "train_file_sha256": data.manifest["train_source"]["file_sha256"],
        "dev_file_sha256": data.manifest["valid_source"]["file_sha256"],
    }
    if commitments != observed_commitments:
        raise ValueError("split commitments differ from registry profile")
    return expected_binding


def validate_identifier(value: str, field: str) -> None:
    """Reject path traversal and unstable output identifiers."""

    if not SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{field} must start with an alphanumeric character and use only "
            "letters, digits, dot, underscore, or hyphen"
        )


def validate_config(config: TrainerConfig) -> None:
    """Validate a run configuration before touching model or output state."""

    validate_identifier(config.task_id, "task_id")
    validate_identifier(config.run_id, "run_id")
    if config.loss not in LOSS_NAMES:
        raise ValueError(f"unsupported loss {config.loss!r}; choose {LOSS_NAMES}")
    if config.label_mode not in LABEL_MODES:
        raise ValueError(
            f"unsupported label mode {config.label_mode!r}; choose {LABEL_MODES}"
        )
    if config.device not in DEVICE_NAMES:
        raise ValueError(f"unsupported device {config.device!r}; choose {DEVICE_NAMES}")
    if len(config.labels) < 2 or len(set(config.labels)) != len(config.labels):
        raise ValueError("labels must contain at least two unique values")
    if any(not label.strip() for label in config.labels):
        raise ValueError("labels cannot be empty")
    if len({label.casefold() for label in config.labels}) != len(config.labels):
        raise ValueError("labels must also be unique after case folding")
    if not config.text_columns or any(not column for column in config.text_columns):
        raise ValueError("at least one non-empty text column is required")
    if len(set(config.text_columns)) != len(config.text_columns):
        raise ValueError("text columns must be unique")
    if (
        not config.cross_split_leakage_columns
        or len(set(config.cross_split_leakage_columns))
        != len(config.cross_split_leakage_columns)
        or not set(config.cross_split_leakage_columns) <= set(config.text_columns)
    ):
        raise ValueError(
            "cross_split_leakage_columns must be a non-empty unique subset of "
            "text_columns"
        )
    if config.label_column in config.text_columns:
        raise ValueError("label column cannot also be a text column")
    if not config.model.strip() or not config.model_revision.strip():
        raise ValueError("model and model revision must be non-empty")
    if not config.tokenizer.strip() or not config.tokenizer_revision.strip():
        raise ValueError("tokenizer and tokenizer revision must be non-empty")
    if config.head_type != DEFAULT_HEAD_TYPE:
        raise ValueError(f"head_type must be {DEFAULT_HEAD_TYPE!r}")
    if config.max_length < 8:
        raise ValueError("max_length must be at least 8")
    if min(config.train_batch, config.eval_batch, config.gradient_accumulation) < 1:
        raise ValueError("batch sizes and gradient accumulation must be positive")
    if config.epochs <= 0 or config.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    if config.weight_decay < 0 or not 0 <= config.warmup_ratio < 1:
        raise ValueError("weight decay must be non-negative and warmup ratio in [0,1)")
    if config.focal_gamma < 0 or config.early_stopping_patience < 1:
        raise ValueError("focal gamma must be non-negative and patience positive")
    registry_values = {
        "registry_path": config.registry_path,
        "registry_bundle_sha256": config.registry_bundle_sha256,
        "registry_index_sha256": config.registry_index_sha256,
        "registry_schema_sha256": config.registry_schema_sha256,
        "task_spec_sha256": config.task_spec_sha256,
        "supervised_profile_id": config.supervised_profile_id,
        "supervised_profile_sha256": config.supervised_profile_sha256,
    }
    if any(registry_values.values()) and not all(registry_values.values()):
        raise ValueError("registry binding fields must be supplied together")
    if all(registry_values.values()):
        for field, value in registry_values.items():
            if field.endswith("_sha256") and not SHA256_RE.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        validate_identifier(config.supervised_profile_id, "supervised_profile_id")


def _path_tokens(path: Path) -> Iterable[str]:
    for component in path.parts:
        yield component
        yield Path(component).stem


def reject_selection_path(path: Path, role: str) -> Path:
    """Resolve and reject any test/holdout-like train or validation source."""

    if role not in {"train", "valid"}:
        raise ValueError(f"unsupported optimization role: {role}")
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{role} CSV must not be a symlink: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{role} CSV must be a regular non-symlink file: {resolved}")
    forbidden = [
        token for token in _path_tokens(resolved)
        if FORBIDDEN_SELECTION_TOKEN_RE.search(token)
    ]
    if forbidden:
        raise ValueError(
            f"{role} selection path contains forbidden test/holdout token(s): "
            f"{sorted(set(forbidden))}"
        )
    if resolved.suffix.lower() != ".csv":
        raise ValueError(f"{role} source must be a CSV file: {resolved}")
    return resolved


def _normalized_label_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip()).casefold()
    text = re.sub(r"^answer\s*:\s*", "", text)
    return text.strip(" \t\r\n\"'`.:,;!?")


def parse_label(raw: str, labels: Sequence[str], mode: str) -> str:
    """Parse an exact label or an IMHI response prefix without guessing."""

    candidate = raw
    if mode == "imhi-response-prefix":
        candidate = re.split(r"\breasoning\b", raw, maxsplit=1, flags=re.IGNORECASE)[0]
    normalized = _normalized_label_text(candidate)
    normalized_labels = {label: _normalized_label_text(label) for label in labels}
    exact = [label for label, value in normalized_labels.items() if normalized == value]
    if len(exact) == 1:
        return exact[0]
    if mode == "imhi-response-prefix":
        matches = []
        for label, value in normalized_labels.items():
            if normalized.startswith(value):
                tail = normalized[len(value):]
                if not tail or not tail[0].isalnum():
                    matches.append(label)
        if len(matches) == 1:
            return matches[0]
    raise ValueError("value does not resolve to exactly one declared label")


def _assemble_text(row: dict[str, str], text_columns: Sequence[str]) -> str:
    values = []
    for column in text_columns:
        value = str(row.get(column, "")).strip()
        if not value:
            raise ValueError(f"empty required text column {column!r}")
        if len(text_columns) == 1:
            values.append(value)
        else:
            values.append(f"{column}: {value}")
    return "\n\n".join(values)


def load_csv_split(
    path: Path,
    *,
    role: str,
    labels: Sequence[str],
    text_columns: Sequence[str],
    cross_split_leakage_columns: Sequence[str],
    label_column: str,
    label_mode: str,
) -> LoadedSplit:
    """Read one CSV split strictly and return rows plus a content-free manifest."""

    resolved = reject_selection_path(path, role)
    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} CSV must be UTF-8: {resolved}") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = reader.fieldnames or []
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError(f"{role} CSV has duplicate column names")
    required = [*text_columns, label_column]
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"{role} CSV is missing required columns: {missing}")

    label_to_id = {label: index for index, label in enumerate(labels)}
    examples: list[Example] = []
    label_counts: Counter[str] = Counter()
    content_labels: dict[str, set[str]] = defaultdict(set)
    content_counts: Counter[str] = Counter()
    for row_number, row in enumerate(reader, start=2):
        try:
            assembled = _assemble_text(row, text_columns)
            leakage_group = _assemble_text(row, cross_split_leakage_columns)
            label = parse_label(str(row.get(label_column, "")), labels, label_mode)
        except ValueError as error:
            raise ValueError(f"{role} CSV row {row_number}: {error}") from error
        content_digest = sha256_bytes(assembled.encode("utf-8"))
        leakage_digest = sha256_bytes(leakage_group.encode("utf-8"))
        example_digest = sha256_bytes(
            canonical_json({"text": assembled, "label": label}).encode("utf-8")
        )
        examples.append(
            Example(
                row_number=row_number,
                text=assembled,
                label=label,
                label_id=label_to_id[label],
                content_sha256=content_digest,
                cross_split_group_sha256=leakage_digest,
                example_sha256=example_digest,
            )
        )
        label_counts[label] += 1
        content_counts[content_digest] += 1
        content_labels[content_digest].add(label)
    if not examples:
        raise ValueError(f"{role} CSV contains no examples")
    missing_labels = [label for label in labels if label_counts[label] == 0]
    if missing_labels:
        raise ValueError(f"{role} CSV is missing declared labels: {missing_labels}")

    manifest = {
        "role": role,
        "path": str(resolved),
        "file_sha256": sha256_bytes(raw),
        "file_bytes": len(raw),
        "header": fieldnames,
        "text_columns": list(text_columns),
        "cross_split_leakage_columns": list(cross_split_leakage_columns),
        "label_column": label_column,
        "label_mode": label_mode,
        "rows": len(examples),
        "parsed_rows": len(examples),
        "invalid_label_rows": 0,
        "label_counts": {label: label_counts[label] for label in labels},
        "ordered_example_commitment_sha256": commitment(
            example.example_sha256 for example in examples
        ),
        "unique_content_rows": len(content_counts),
        "duplicate_content_rows": sum(count - 1 for count in content_counts.values()),
        "conflicting_label_content_groups": sum(
            1 for values in content_labels.values() if len(values) > 1
        ),
        "cross_split_group_commitment_sha256": commitment(
            example.cross_split_group_sha256 for example in examples
        ),
        "unique_cross_split_groups": len(
            {example.cross_split_group_sha256 for example in examples}
        ),
        "raw_text_or_row_ids_persisted": False,
    }
    return LoadedSplit(role=role, examples=tuple(examples), manifest=manifest)


def prepare_data(config: TrainerConfig) -> PreparedData:
    """Clean train duplicates, then remove validation-overlapping train rows."""

    validate_config(config)
    train_path = config.train_csv.expanduser().resolve(strict=True)
    valid_path = config.valid_csv.expanduser().resolve(strict=True)
    if train_path == valid_path:
        raise ValueError("train and valid CSV files must be distinct")
    train = load_csv_split(
        train_path,
        role="train",
        labels=config.labels,
        text_columns=config.text_columns,
        cross_split_leakage_columns=config.cross_split_leakage_columns,
        label_column=config.label_column,
        label_mode=config.label_mode,
    )
    valid = load_csv_split(
        valid_path,
        role="valid",
        labels=config.labels,
        text_columns=config.text_columns,
        cross_split_leakage_columns=config.cross_split_leakage_columns,
        label_column=config.label_column,
        label_mode=config.label_mode,
    )
    if valid.manifest["conflicting_label_content_groups"]:
        raise ValueError("valid CSV contains identical inputs with conflicting labels")
    if valid.manifest["duplicate_content_rows"]:
        raise ValueError(
            "valid CSV contains duplicate full inputs; validation weighting must "
            "be frozen explicitly instead of inferred"
        )
    train_groups: dict[str, list[Example]] = {}
    for example in train.examples:
        train_groups.setdefault(example.content_sha256, []).append(example)
    cleaned_train: list[Example] = []
    same_label_duplicates_dropped = 0
    conflicting_groups_dropped = 0
    conflicting_rows_dropped = 0
    for group in train_groups.values():
        group_labels = {example.label for example in group}
        if len(group_labels) > 1:
            conflicting_groups_dropped += 1
            conflicting_rows_dropped += len(group)
            continue
        cleaned_train.append(group[0])
        same_label_duplicates_dropped += len(group) - 1

    valid_content = {
        example.cross_split_group_sha256 for example in valid.examples
    }
    overlapping_train = [
        example for example in cleaned_train
        if example.cross_split_group_sha256 in valid_content
    ]
    filtered_train = tuple(
        example for example in cleaned_train
        if example.cross_split_group_sha256 not in valid_content
    )
    if not filtered_train:
        raise ValueError("no training examples remain after cross-split decontamination")
    filtered_counts = Counter(example.label for example in filtered_train)
    missing_after_filter = [
        label for label in config.labels if filtered_counts[label] == 0
    ]
    if missing_after_filter:
        raise ValueError(
            "cross-split decontamination removed all training examples for: "
            f"{missing_after_filter}"
        )
    overlap_hashes = sorted(
        {example.cross_split_group_sha256 for example in overlapping_train}
    )
    manifest = {
        "schema_version": 1,
        "train_source": train.manifest,
        "valid_source": valid.manifest,
        "within_train_duplicate_policy": (
            "keep first same-label content; drop all rows in conflicting-label "
            "content groups"
        ),
        "within_train_same_label_duplicate_rows_dropped": (
            same_label_duplicates_dropped
        ),
        "within_train_conflicting_content_groups_dropped": (
            conflicting_groups_dropped
        ),
        "within_train_conflicting_rows_dropped": conflicting_rows_dropped,
        "train_rows_after_within_split_cleanup": len(cleaned_train),
        "cross_split_policy": (
            "drop train rows sharing the frozen leakage-group key with valid; "
            "keep full valid"
        ),
        "cross_split_leakage_columns": list(
            config.cross_split_leakage_columns
        ),
        "cross_split_overlap_unique_contents": len(overlap_hashes),
        "cross_split_train_rows_dropped": len(overlapping_train),
        "cross_split_overlap_commitment_sha256": commitment(overlap_hashes),
        "used_train_rows": len(filtered_train),
        "used_valid_rows": len(valid.examples),
        "used_train_label_counts": {
            label: filtered_counts[label] for label in config.labels
        },
        "used_train_commitment_sha256": commitment(
            example.example_sha256 for example in filtered_train
        ),
        "used_valid_commitment_sha256": commitment(
            example.example_sha256 for example in valid.examples
        ),
        "selection_split": "valid",
        "test_or_holdout_access": False,
    }
    return PreparedData(
        train=filtered_train,
        valid=valid.examples,
        manifest=manifest,
    )


def resolve_device(requested: str) -> str:
    """Resolve an explicit CPU/CUDA/MPS device without silently falling back."""

    if requested not in DEVICE_NAMES:
        raise ValueError(f"unsupported device: {requested}")
    torch = _dependency("torch", "device selection")
    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    if requested == "mps" and not mps_available:
        raise RuntimeError("--device mps requested but torch MPS is unavailable")
    return requested


def balanced_class_weight_values(
    label_ids: Sequence[int], number_of_labels: int
) -> list[float]:
    """Compute train-only balanced class weights with mean weight one."""

    counts = Counter(int(value) for value in label_ids)
    missing = [index for index in range(number_of_labels) if counts[index] == 0]
    if missing:
        raise ValueError(f"training labels are missing ids: {missing}")
    raw = [len(label_ids) / (number_of_labels * counts[index]) for index in range(number_of_labels)]
    mean = sum(raw) / len(raw)
    return [float(value / mean) for value in raw]


def classifier_loss(
    logits: Any,
    labels: Any,
    *,
    loss_name: str,
    class_weights: Any,
    focal_gamma: float,
) -> Any:
    """Compute CE, balanced weighted CE, or balanced-alpha focal loss."""

    torch = _dependency("torch", "classifier loss")
    functional = torch.nn.functional
    if loss_name == "ce":
        return functional.cross_entropy(logits, labels)
    weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    if loss_name == "weighted-ce":
        return functional.cross_entropy(logits, labels, weight=weights)
    if loss_name == "focal":
        plain = functional.cross_entropy(logits, labels, reduction="none")
        modulation = (1.0 - torch.exp(-plain)).pow(float(focal_gamma))
        alpha = weights.gather(0, labels)
        return (alpha * modulation * plain).mean()
    raise ValueError(f"unsupported loss: {loss_name}")


def classification_metrics(
    gold: Sequence[int], predicted: Sequence[int], labels: Sequence[str]
) -> dict[str, Any]:
    """Return aggregate, per-class, and label-ordered confusion metrics."""

    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must be non-empty and equally sized")
    metrics = _dependency("sklearn.metrics", "classification metrics")
    label_ids = list(range(len(labels)))
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        gold,
        predicted,
        labels=label_ids,
        zero_division=0,
    )
    return {
        "accuracy": float(metrics.accuracy_score(gold, predicted)),
        "macro_f1": float(
            metrics.f1_score(
                gold, predicted, labels=label_ids, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            metrics.f1_score(
                gold, predicted, labels=label_ids, average="weighted", zero_division=0
            )
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": list(labels),
            "counts": [
                [int(value) for value in row]
                for row in metrics.confusion_matrix(
                    gold, predicted, labels=label_ids
                ).tolist()
            ],
            "rows": "gold",
            "columns": "predicted",
        },
    }


def run_directory(config: TrainerConfig) -> Path:
    """Return the isolated output directory for one task/loss/seed identity."""

    return (
        config.output_root.expanduser().resolve()
        / config.task_id
        / config.run_id
        / config.loss
        / f"seed-{config.seed}"
    )


def build_plan(
    config: TrainerConfig,
    data: PreparedData,
    *,
    repository_git: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frozen, hash-addressed plan used by dry-run and execution."""

    validate_config(config)
    resolved_device = resolve_device(config.device)
    output_dir = run_directory(config)
    implementation_path = Path(__file__).resolve()
    repository_identity = dict(repository_git or git_identity())
    schedule = training_schedule(config, len(data.train))
    label_mapping = {
        label: index for index, label in enumerate(config.labels)
    }
    registry_binding = None
    if config.registry_path:
        registry_binding = {
            "registry_path": config.registry_path,
            "registry_bundle_sha256": config.registry_bundle_sha256,
            "registry_index_sha256": config.registry_index_sha256,
            "registry_schema_sha256": config.registry_schema_sha256,
            "task_spec_sha256": config.task_spec_sha256,
            "supervised_profile_id": config.supervised_profile_id,
            "supervised_profile_sha256": config.supervised_profile_sha256,
        }
    identity = {
        "schema_version": 1,
        "git": repository_identity,
        "implementation": {
            "path": str(implementation_path.relative_to(ROOT)),
            "sha256": sha256_bytes(implementation_path.read_bytes()),
            "python": platform.python_version(),
            "dependency_versions": {
                name: installed_version(name)
                for name in ("accelerate", "numpy", "scikit-learn", "torch", "transformers")
            },
        },
        "task_id": config.task_id,
        "run_id": config.run_id,
        "model": config.model,
        "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer,
        "tokenizer_revision": config.tokenizer_revision,
        "head_type": config.head_type,
        "label_mapping": label_mapping,
        "label_mapping_sha256": sha256_bytes(
            canonical_json(label_mapping).encode("utf-8")
        ),
        "loss": config.loss,
        "seed": config.seed,
        "labels": list(config.labels),
        "text_columns": list(config.text_columns),
        "cross_split_leakage_columns": list(
            config.cross_split_leakage_columns
        ),
        "label_column": config.label_column,
        "label_mode": config.label_mode,
        "dataset": data.manifest,
        "device_requested": config.device,
        "device_resolved": resolved_device,
        "max_length": config.max_length,
        "train_batch": config.train_batch,
        "eval_batch": config.eval_batch,
        "gradient_accumulation": config.gradient_accumulation,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "schedule": schedule,
        "focal_gamma": config.focal_gamma,
        "early_stopping_patience": config.early_stopping_patience,
        "optimizer": "adamw_torch",
        "scheduler": "linear",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "max_grad_norm": 1.0,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": False,
        "class_weighting": "balanced_inverse_frequency_after_cleaning",
        "selection_metric": "weighted_f1",
        "selection_split": "valid",
        "selection_tie_break": (
            "earliest evaluation checkpoint among equal weighted_f1; only a "
            "strict improvement replaces the incumbent"
        ),
        "deterministic_execution": (
            "Python, NumPy, PyTorch, and Transformers seeds fixed; "
            "transformers.set_seed(deterministic=True); serial single-device run; "
            "checkpoint resume strictly saves/restores Apple-MPS RNG state"
        ),
        "train_manifest_sha256": sha256_bytes(
            canonical_json(data.manifest["train_source"]).encode("utf-8")
        ),
        "dev_manifest_sha256": sha256_bytes(
            canonical_json(data.manifest["valid_source"]).encode("utf-8")
        ),
        "registry_binding": registry_binding,
        "test_or_holdout_access": False,
    }
    identity_sha256 = sha256_bytes(canonical_json(identity).encode("utf-8"))
    return {
        "schema_version": 1,
        "dry_run_default": True,
        "status": "DATA_VALIDATED",
        "mode": "explicit_data_validation",
        "execution_interface": "registry_bound_adapter_only",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "output_dir": str(output_dir),
        "artifacts": {
            "identity": str(output_dir / "run-identity.json"),
            "result": str(output_dir / "result.json"),
            "best_model": str(output_dir / "best-model"),
        },
        "privacy_boundary": (
            "aggregate metrics, hashes, and label counts only; raw text, row ids, "
            "and row-level predictions are not persisted"
        ),
    }


def build_preregistration_plan(config: TrainerConfig) -> dict[str, Any]:
    """Freeze declared metadata without opening either external CSV."""

    validate_config(config)
    output_dir = run_directory(config)
    implementation_path = Path(__file__).resolve()
    identity = {
        "schema_version": 1,
        "mode": "metadata_only_preregistration",
        "external_dataset_read": False,
        "git": git_identity(),
        "implementation": {
            "path": str(implementation_path.relative_to(ROOT)),
            "sha256": sha256_bytes(implementation_path.read_bytes()),
            "python": platform.python_version(),
        },
        "task_id": config.task_id,
        "run_id": config.run_id,
        "declared_train_csv": str(config.train_csv),
        "declared_valid_csv": str(config.valid_csv),
        "labels": list(config.labels),
        "text_columns": list(config.text_columns),
        "cross_split_leakage_columns": list(
            config.cross_split_leakage_columns
        ),
        "label_column": config.label_column,
        "label_mode": config.label_mode,
        "model": config.model,
        "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer,
        "tokenizer_revision": config.tokenizer_revision,
        "head_type": config.head_type,
        "loss": config.loss,
        "seed": config.seed,
        "test_or_holdout_access": False,
    }
    identity_sha256 = sha256_bytes(canonical_json(identity).encode("utf-8"))
    return {
        "schema_version": 1,
        "status": "PREREGISTERED",
        "dry_run_default": True,
        "external_dataset_read": False,
        "data_validation_requires_flag": "--validate-data",
        "training_interface": "registry_bound_adapter_only",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "output_dir": str(output_dir),
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def validate_existing_identity(run_dir: Path, plan: dict[str, Any]) -> None:
    """Require byte-equivalent semantic identity before resume or reuse."""

    identity_path = run_dir / "run-identity.json"
    if not identity_path.is_file() or identity_path.is_symlink():
        raise ValueError("resume identity is missing or not a regular file")
    observed = _read_json(identity_path)
    if canonical_json(observed) != canonical_json(plan["identity"]):
        raise ValueError("resume identity mismatch")
    observed_sha = sha256_bytes(canonical_json(observed).encode("utf-8"))
    if observed_sha != plan["identity_sha256"]:
        raise ValueError("resume identity hash mismatch")


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def checkpoint_weight_path(checkpoint: Path) -> Path:
    """Require exactly one unsharded regular model-weight artifact."""

    choices = [
        path for name in ("model.safetensors", "pytorch_model.bin")
        if _regular_file(path := checkpoint / name)
    ]
    if len(choices) != 1:
        raise ValueError(
            f"checkpoint must contain exactly one model weight file: {checkpoint}"
        )
    return choices[0]


def mps_rng_state_path(checkpoint: Path) -> Path:
    return checkpoint / "mps_rng_state.pth"


def save_mps_rng_state(checkpoint: Path) -> Path:
    """Persist the current Apple-MPS RNG state as a strict checkpoint part."""

    torch = _dependency("torch", "MPS RNG checkpoint save")
    if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
        raise RuntimeError("MPS RNG state requested but MPS is unavailable")
    checkpoint.mkdir(parents=True, exist_ok=True)
    path = mps_rng_state_path(checkpoint)
    if path.is_symlink():
        raise ValueError("MPS RNG checkpoint path must not be a symlink")
    torch.save(torch.mps.get_rng_state().cpu(), path)
    return path


def load_mps_rng_state(checkpoint: Path, *, restore: bool) -> Any:
    """Strictly load, validate, and optionally restore an MPS RNG state."""

    torch = _dependency("torch", "MPS RNG checkpoint load")
    path = mps_rng_state_path(checkpoint)
    if not _regular_file(path):
        raise ValueError(f"checkpoint is missing regular MPS RNG state: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.uint8
        or state.ndim != 1
        or state.numel() == 0
    ):
        raise ValueError("checkpoint MPS RNG state has invalid tensor metadata")
    if restore:
        if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("cannot restore MPS RNG state because MPS is unavailable")
        torch.mps.set_rng_state(state)
    return state


def complete_checkpoint(
    checkpoint: Path, *, require_mps_rng: bool = False
) -> bool:
    """Return whether all artifacts required for exact Trainer resume exist."""

    if not checkpoint.is_dir() or checkpoint.is_symlink():
        return False
    required = (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
    )
    if not all(_regular_file(checkpoint / name) for name in required):
        return False
    rng_files = list(checkpoint.glob("rng_state*.pth"))
    if not rng_files or not all(_regular_file(path) for path in rng_files):
        return False
    try:
        state = _read_json(checkpoint / "trainer_state.json")
        if not isinstance(state, dict):
            return False
        checkpoint_weight_path(checkpoint)
        if require_mps_rng:
            load_mps_rng_state(checkpoint, restore=False)
    except Exception:
        # Candidate discovery must skip any malformed checkpoint rather than
        # allowing a higher corrupt directory to mask a lower valid resume.
        return False
    return True


def latest_checkpoint(
    run_dir: Path, *, require_mps_rng: bool = False
) -> Path | None:
    """Find the highest numbered fully resumable Transformers checkpoint."""

    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", path.name)
        if match and complete_checkpoint(path, require_mps_rng=require_mps_rng):
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def validate_checkpoint_model_state(checkpoint: Path, model: Any) -> None:
    """Reject checkpoint weights whose keys, shapes, or dtypes differ."""

    torch = _dependency("torch", "strict checkpoint validation")
    weight_path = checkpoint_weight_path(checkpoint)
    observed: dict[str, tuple[tuple[int, ...], str]] = {}
    if weight_path.name == "model.safetensors":
        safetensors = _dependency("safetensors", "strict checkpoint validation")
        with safetensors.safe_open(
            str(weight_path), framework="pt", device="cpu"
        ) as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                observed[key] = (tuple(tensor.shape), str(tensor.dtype))
    else:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint model state is not an object: {weight_path}")
        for key, tensor in state.items():
            if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
                raise ValueError(f"checkpoint state contains non-tensor key {key!r}")
            observed[str(key)] = (tuple(tensor.shape), str(tensor.dtype))
    expected = {
        key: (tuple(tensor.shape), str(tensor.dtype))
        for key, tensor in model.state_dict().items()
    }
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise ValueError(
            "checkpoint model-state key mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    mismatched = [
        key for key in expected if observed[key] != expected[key]
    ]
    if mismatched:
        key = sorted(mismatched)[0]
        raise ValueError(
            f"checkpoint tensor mismatch for {key!r}: "
            f"expected={expected[key]}, observed={observed[key]}"
        )


def earliest_strict_best_step(
    log_history: Sequence[dict[str, Any]],
    metric_key: str = "eval_weighted_f1",
) -> int | None:
    """Return the earliest evaluation step attaining the strict running max."""

    best_value = -math.inf
    best_step: int | None = None
    for entry in log_history:
        if metric_key not in entry or "step" not in entry:
            continue
        value = float(entry[metric_key])
        if math.isfinite(value) and value > best_value:
            best_value = value
            best_step = int(entry["step"])
    return best_step


def selection_checkpoint_provenance(
    log_history: Sequence[dict[str, Any]],
    *,
    best_global_step: int | None,
    best_model_checkpoint: str | None,
    optimizer_updates_per_epoch: int,
    metric_key: str = "eval_weighted_f1",
) -> dict[str, Any]:
    """Strictly derive the chosen checkpoint step and epoch from Trainer state."""

    if best_global_step is None or best_global_step < 1:
        raise ValueError("Trainer best_global_step must be positive")
    if not best_model_checkpoint:
        raise ValueError("Trainer best_model_checkpoint is missing")
    match = re.fullmatch(
        r"checkpoint-([1-9][0-9]*)", Path(best_model_checkpoint).name
    )
    if not match or int(match.group(1)) != best_global_step:
        raise ValueError("best checkpoint directory does not match best_global_step")
    if optimizer_updates_per_epoch < 1:
        raise ValueError("optimizer_updates_per_epoch must be positive")
    matching_epochs = {
        float(entry["epoch"])
        for entry in log_history
        if metric_key in entry
        and "epoch" in entry
        and int(entry.get("step", -1)) == best_global_step
        and math.isfinite(float(entry["epoch"]))
    }
    if len(matching_epochs) != 1:
        raise ValueError("best checkpoint must map to exactly one evaluation epoch")
    epoch = matching_epochs.pop()
    derived_epoch = best_global_step / optimizer_updates_per_epoch
    if not math.isclose(epoch, derived_epoch, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "logged best checkpoint epoch differs from the frozen update schedule"
        )
    return {
        "best_checkpoint": best_model_checkpoint,
        "best_global_step": best_global_step,
        "best_checkpoint_epoch": epoch,
        "epoch_derivation": (
            "best_global_step / optimizer_updates_per_epoch, cross-checked "
            "against the matching evaluation log entry"
        ),
    }


def prepare_run_directory(
    plan: dict[str, Any], *, resume: str | None
) -> Path | None:
    """Create a new identity or validate an incomplete identical resume."""

    run_dir = Path(plan["output_dir"])
    if run_dir.exists():
        if not resume:
            raise ValueError(f"refusing to overwrite existing run: {run_dir}")
        validate_existing_identity(run_dir, plan)
        if (run_dir / "result.json").exists():
            raise ValueError("completed runs cannot be resumed")
        require_mps_rng = plan["identity"]["device_resolved"] == "mps"
        checkpoint = latest_checkpoint(run_dir, require_mps_rng=require_mps_rng)
        if checkpoint is None:
            raise ValueError("no complete trainer checkpoint is available to resume")
        return checkpoint
    if resume:
        raise ValueError("--resume requires an existing identical run")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "run-identity.json", plan["identity"])
    return None


def completed_result(plan: dict[str, Any]) -> dict[str, Any] | None:
    """Return an already completed identical result for campaign-level reuse."""

    run_dir = Path(plan["output_dir"])
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        return None
    validate_existing_identity(run_dir, plan)
    result = _read_json(result_path)
    if result.get("identity_sha256") != plan["identity_sha256"]:
        raise ValueError("completed result identity mismatch")
    return result


class TokenizedTextDataset:
    """Minimal lazy-tokenized dataset accepted by Transformers Trainer."""

    def __init__(self, examples: Sequence[Example], tokenizer: Any, max_length: int):
        self.examples = tuple(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = example.label_id
        return encoded


def execute_training(
    config: TrainerConfig,
    data: PreparedData,
    plan: dict[str, Any],
    *,
    resume: str | None = None,
) -> dict[str, Any]:
    """Train and validate one run after an explicit caller authorization."""

    validate_registry_execution_proof(config, data, plan)
    torch = _dependency("torch", "classifier training")
    np = _dependency("numpy", "classifier training")
    transformers = _dependency("transformers", "classifier training")
    _dependency("accelerate", "Transformers Trainer execution")
    if git_identity() != plan["identity"]["git"]:
        raise ValueError("repository Git identity drifted after the run plan was built")
    resume_checkpoint = prepare_run_directory(plan, resume=resume)
    run_dir = Path(plan["output_dir"])
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    try:
        transformers.set_seed(config.seed, deterministic=True)
    except TypeError as error:
        raise RuntimeError(
            "installed Transformers cannot enforce deterministic set_seed"
        ) from error

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer, revision=config.tokenizer_revision
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model,
        revision=config.model_revision,
        num_labels=len(config.labels),
        id2label={index: label for index, label in enumerate(config.labels)},
        label2id={label: index for index, label in enumerate(config.labels)},
        ignore_mismatched_sizes=True,
    )
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if tokenizer_commit is not None and tokenizer_commit != config.tokenizer_revision:
        raise RuntimeError(
            "resolved tokenizer commit differs from the frozen tokenizer revision"
        )
    tokenizer_commit = tokenizer_commit or config.tokenizer_revision
    model_commit = getattr(model.config, "_commit_hash", None)
    if model_commit is not None and model_commit != config.model_revision:
        raise RuntimeError(
            "resolved model commit differs from the frozen model revision"
        )
    model_commit = model_commit or config.model_revision
    train_dataset = TokenizedTextDataset(data.train, tokenizer, config.max_length)
    valid_dataset = TokenizedTextDataset(data.valid, tokenizer, config.max_length)
    class_values = balanced_class_weight_values(
        [example.label_id for example in data.train], len(config.labels)
    )
    class_tensor = torch.tensor(class_values, dtype=torch.float32)

    class LossTrainer(transformers.Trainer):
        """Trainer whose only arm difference is the declared loss."""

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            # The custom reduction already returns the mean loss. Advertising
            # loss kwargs would make Trainer rescale it across accumulation.
            self.model_accepts_loss_kwargs = False

        def _save_rng_state(self, output_dir: str) -> None:
            super()._save_rng_state(output_dir)
            if resolved_device == "mps":
                save_mps_rng_state(Path(output_dir))

        def _load_rng_state(self, checkpoint: str | None) -> None:
            if checkpoint is None:
                return super()._load_rng_state(checkpoint)
            if resolved_device == "mps":
                # Validate before the framework mutates any other RNG stream.
                load_mps_rng_state(Path(checkpoint), restore=False)
            super()._load_rng_state(checkpoint)
            if resolved_device == "mps":
                load_mps_rng_state(Path(checkpoint), restore=True)

        def _load_from_checkpoint(
            self, resume_from_checkpoint: str, model: Any = None
        ) -> None:
            validate_checkpoint_model_state(
                Path(resume_from_checkpoint),
                model if model is not None else self.model,
            )
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        def _load_best_model(self) -> None:
            if not self.state.best_model_checkpoint:
                raise RuntimeError("Trainer did not record a best model checkpoint")
            validate_checkpoint_model_state(
                Path(self.state.best_model_checkpoint), self.model
            )
            return super()._load_best_model()

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = classifier_loss(
                outputs.logits,
                labels,
                loss_name=config.loss,
                class_weights=class_tensor,
                focal_gamma=config.focal_gamma,
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(evaluation: Any) -> dict[str, float]:
        predictions = np.argmax(evaluation.predictions, axis=-1)
        values = classification_metrics(
            evaluation.label_ids.tolist(), predictions.tolist(), config.labels
        )
        return {
            "accuracy": values["accuracy"],
            "macro_f1": values["macro_f1"],
            "weighted_f1": values["weighted_f1"],
        }

    resolved_device = plan["identity"]["device_resolved"]
    training_args = transformers.TrainingArguments(
        output_dir=str(run_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        save_total_limit=2,
        per_device_train_batch_size=config.train_batch,
        per_device_eval_batch_size=config.eval_batch,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.epochs,
        warmup_steps=plan["identity"]["schedule"]["warmup_steps"],
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        seed=config.seed,
        data_seed=config.seed,
        use_cpu=resolved_device == "cpu",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=resolved_device == "cuda",
        optim="adamw_torch",
        report_to="none",
    )
    trainer = LossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            transformers.EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )
    actual_device = str(trainer.args.device).split(":", maxsplit=1)[0]
    if actual_device != resolved_device:
        raise RuntimeError(
            f"actual Trainer device {trainer.args.device} does not match "
            f"resolved device {resolved_device}"
        )

    started = time.time()
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    earliest_best_step = earliest_strict_best_step(trainer.state.log_history)
    if earliest_best_step is None or earliest_best_step != trainer.state.best_global_step:
        raise RuntimeError(
            "best-checkpoint tie policy drift: expected earliest strict weighted-F1 "
            f"step {earliest_best_step}, Trainer recorded {trainer.state.best_global_step}"
        )
    selected_checkpoint = selection_checkpoint_provenance(
        trainer.state.log_history,
        best_global_step=trainer.state.best_global_step,
        best_model_checkpoint=trainer.state.best_model_checkpoint,
        optimizer_updates_per_epoch=plan["identity"]["schedule"]
        ["optimizer_updates_per_epoch"],
    )
    prediction = trainer.predict(valid_dataset, metric_key_prefix="valid")
    predicted = np.argmax(prediction.predictions, axis=-1).tolist()
    gold = prediction.label_ids.tolist()
    metrics = classification_metrics(gold, predicted, config.labels)
    best_model_dir = run_dir / "best-model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    best_model_artifact = artifact_tree_manifest(best_model_dir)
    result = {
        "schema_version": 1,
        "status": "complete",
        "identity_sha256": plan["identity_sha256"],
        "task_id": config.task_id,
        "run_id": config.run_id,
        "loss": config.loss,
        "seed": config.seed,
        "model": config.model,
        "requested_model_revision": config.model_revision,
        "resolved_model_commit": model_commit,
        "tokenizer_id": config.tokenizer,
        "tokenizer_revision": config.tokenizer_revision,
        "resolved_tokenizer_commit": tokenizer_commit,
        "head_type": config.head_type,
        "label_mapping_sha256": plan["identity"]["label_mapping_sha256"],
        "device": str(trainer.args.device),
        "train_rows": len(data.train),
        "valid_rows": len(data.valid),
        "train_runtime_seconds": float(
            train_result.metrics.get("train_runtime", time.time() - started)
        ),
        "train_loss": float(train_result.metrics.get("train_loss", math.nan)),
        "class_weights": {
            label: class_values[index]
            for index, label in enumerate(config.labels)
        },
        "loss_definition": (
            "plain cross entropy" if config.loss == "ce" else
            "balanced class-weighted cross entropy" if config.loss == "weighted-ce" else
            "balanced-class alpha focal loss"
        ),
        "metrics": metrics,
        "selection": {
            "split": "valid",
            "metric": "weighted_f1",
            **selected_checkpoint,
            "tie_break": plan["identity"]["selection_tie_break"],
            "test_or_holdout_access": False,
        },
        "deterministic_execution": plan["identity"]["deterministic_execution"],
        "git": plan["identity"]["git"],
        "registry_binding": plan["identity"]["registry_binding"],
        "train_manifest_sha256": plan["identity"]["train_manifest_sha256"],
        "dev_manifest_sha256": plan["identity"]["dev_manifest_sha256"],
        "checkpoint_sha256": best_model_artifact["checkpoint_sha256"],
        "dataset": data.manifest,
        "artifacts": {
            "best_model": str(best_model_dir),
            "best_model_manifest": best_model_artifact,
        },
        "row_level_material_persisted": False,
    }
    if not math.isfinite(result["train_loss"]):
        result["train_loss"] = None
    atomic_write_json(run_dir / "result.json", result)
    return result


def run_classifier(
    config: TrainerConfig,
    *,
    execute: bool,
    validate_data: bool = False,
    resume: str | None = None,
    manifest_out: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Prepare a plan and optionally execute its exactly identified run."""

    if execute and not config.registry_path:
        raise ValueError(
            "training execution requires a validated benchmark-registry binding"
        )
    if not execute and not validate_data:
        plan = build_preregistration_plan(config)
        if manifest_out is not None:
            atomic_write_json(manifest_out.expanduser().resolve(), plan)
        return plan, None
    data = prepare_data(config)
    plan = build_plan(config, data)
    if manifest_out is not None:
        atomic_write_json(manifest_out.expanduser().resolve(), plan)
    if not execute:
        return plan, None
    result = execute_training(config, data, plan, resume=resume)
    return plan, result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"selftest failed: {message}")


def selftest() -> None:
    """Exercise provenance, loss, metrics, and resume gates without a model."""

    torch = _dependency("torch", "synthetic selftest")
    with tempfile.TemporaryDirectory(prefix="text-classifier-selftest-") as temporary:
        root = Path(temporary)
        train_path = root / "train.csv"
        valid_path = root / "valid.csv"
        labels = ("no", "yes")

        def write_fixture(path: Path, rows: Sequence[tuple[str, str]]) -> None:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("post", "response"))
                writer.writeheader()
                for post, label in rows:
                    writer.writerow(
                        {
                            "post": post,
                            "response": f"{label}. Reasoning: synthetic fixture",
                        }
                    )

        write_fixture(
            train_path,
            [
                ("train no one", "no"),
                ("train no two", "no"),
                ("train yes one", "yes"),
                ("shared row", "yes"),
                ("train no one", "no"),
                ("conflicting row", "no"),
                ("conflicting row", "yes"),
            ],
        )
        write_fixture(
            valid_path,
            [
                ("valid no", "no"),
                ("valid yes", "yes"),
                ("shared row", "yes"),
            ],
        )
        config = TrainerConfig(
            task_id="synthetic-task",
            run_id="selftest",
            train_csv=train_path,
            valid_csv=valid_path,
            labels=labels,
            label_column="response",
            label_mode="imhi-response-prefix",
            output_root=root / "outputs",
            device="cpu",
        )
        data = prepare_data(config)
        _require(len(data.train) == 3, "validation overlap removed from train")
        _require(len(data.valid) == 3, "full validation retained")
        _require(
            data.manifest["within_train_same_label_duplicate_rows_dropped"] == 1,
            "same-label train duplicate removal",
        )
        _require(
            data.manifest["within_train_conflicting_content_groups_dropped"] == 1
            and data.manifest["within_train_conflicting_rows_dropped"] == 2,
            "conflicting-label train group removal",
        )
        _require(
            data.manifest["cross_split_train_rows_dropped"] == 1,
            "cross-split overlap manifest",
        )
        _require(
            data.manifest["train_source"]["file_sha256"]
            != data.manifest["valid_source"]["file_sha256"],
            "distinct source hashes",
        )
        plan = build_plan(config, data)
        _require(plan["identity"]["test_or_holdout_access"] is False, "test gate")
        _require(
            training_schedule(
                replace(
                    config,
                    train_batch=2,
                    gradient_accumulation=2,
                    epochs=2.5,
                    warmup_ratio=0.2,
                ),
                3,
            )
            == {
                "train_rows": 3,
                "batches_per_epoch": 2,
                "optimizer_updates_per_epoch": 1,
                "total_optimizer_update_steps": 3,
                "frozen_warmup_ratio": 0.2,
                "warmup_steps": 1,
                "rounding": (
                    "ceil at batch, accumulation, total-update, and warmup stages"
                ),
            },
            "integer warmup schedule",
        )
        tie_history = [
            {"step": 1, "epoch": 0.5, "eval_weighted_f1": 0.5},
            {"step": 2, "epoch": 1.0, "eval_weighted_f1": 0.7},
            {"step": 3, "epoch": 1.5, "eval_weighted_f1": 0.7},
        ]
        _require(
            earliest_strict_best_step(tie_history) == 2,
            "earliest checkpoint wins an exact metric tie",
        )
        selection_fixture = selection_checkpoint_provenance(
            tie_history,
            best_global_step=2,
            best_model_checkpoint=str(root / "checkpoint-2"),
            optimizer_updates_per_epoch=2,
        )
        _require(
            selection_fixture["best_checkpoint_epoch"] == 1.0
            and selection_fixture["best_global_step"] == 2,
            "chosen checkpoint epoch field and strict derivation",
        )

        balanced_values = balanced_class_weight_values([0, 0, 0, 1], 2)
        _require(
            all(
                math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
                for observed, expected in zip(balanced_values, (0.5, 1.5))
            ),
            "imbalanced train-only balanced class weights",
        )
        try:
            balanced_class_weight_values([0, 0], 2)
        except ValueError as error:
            _require("missing ids" in str(error), "empty-class rejection reason")
        else:
            raise RuntimeError("selftest failed: empty training class accepted")
        try:
            parse_label("not-a-declared-label", labels, "imhi-response-prefix")
        except ValueError:
            pass
        else:
            raise RuntimeError("selftest failed: invalid label was accepted")

        logits = torch.tensor(
            [[2.0, -0.5], [0.2, 1.7], [-0.4, 1.3]], requires_grad=True
        )
        gold_tensor = torch.tensor([0, 1, 0])
        weights = torch.tensor([0.75, 1.25])
        losses = {
            name: classifier_loss(
                logits,
                gold_tensor,
                loss_name=name,
                class_weights=weights,
                focal_gamma=2.0,
            )
            for name in LOSS_NAMES
        }
        _require(all(torch.isfinite(value) for value in losses.values()), "finite losses")
        _require(
            len({round(float(value.detach()), 7) for value in losses.values()}) == 3,
            "three distinct loss definitions",
        )
        sum(losses.values()).backward()
        metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], labels)
        repeated_metrics = classification_metrics(
            [0, 1, 1, 0], [0, 1, 0, 0], labels
        )
        _require(
            canonical_json(metrics) == canonical_json(repeated_metrics),
            "deterministic repeated binary metrics",
        )
        _require(abs(metrics["accuracy"] - 0.75) < 1e-12, "accuracy")
        _require(set(metrics["per_class"]) == set(labels), "per-class metrics")
        _require(
            metrics["confusion_matrix"]
            == {
                "labels": ["no", "yes"],
                "counts": [[2, 0], [1, 1]],
                "rows": "gold",
                "columns": "predicted",
            },
            "label-ordered aggregate confusion matrix",
        )

        multiclass_labels = ("alpha", "beta", "gamma")
        multiclass_gold = torch.tensor([0, 0, 0, 1, 2, 2])
        multiclass_logits = torch.tensor(
            [
                [2.0, 0.1, -0.4],
                [0.2, 1.4, -0.3],
                [1.3, 0.2, 0.1],
                [-0.2, 1.8, 0.3],
                [0.1, 0.4, 1.5],
                [0.5, 1.1, 0.7],
            ],
            requires_grad=True,
        )
        multiclass_weight_values = balanced_class_weight_values(
            multiclass_gold.tolist(), 3
        )
        multiclass_weights = torch.tensor(multiclass_weight_values)
        multiclass_losses = {
            name: classifier_loss(
                multiclass_logits,
                multiclass_gold,
                loss_name=name,
                class_weights=multiclass_weights,
                focal_gamma=2.0,
            )
            for name in LOSS_NAMES
        }
        _require(
            all(torch.isfinite(value) for value in multiclass_losses.values()),
            "finite three-class CE/weighted-CE/focal losses",
        )
        sum(multiclass_losses.values()).backward()
        multiclass_metrics = classification_metrics(
            [0, 0, 0, 1, 2, 2],
            [0, 1, 0, 1, 2, 1],
            multiclass_labels,
        )
        _require(
            set(multiclass_metrics["per_class"]) == set(multiclass_labels)
            and len(multiclass_metrics["confusion_matrix"]["counts"]) == 3,
            "complete three-class metrics and confusion matrix",
        )
        _require(
            canonical_json(multiclass_metrics)
            == canonical_json(
                classification_metrics(
                    [0, 0, 0, 1, 2, 2],
                    [0, 1, 0, 1, 2, 1],
                    multiclass_labels,
                )
            ),
            "deterministic repeated multiclass metrics",
        )

        prepare_run_directory(plan, resume=None)
        try:
            prepare_run_directory(plan, resume=None)
        except ValueError:
            pass
        else:
            raise RuntimeError("selftest failed: overwrite was accepted")
        checkpoint = Path(plan["output_dir"]) / "checkpoint-3"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")
        for name in ("optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"):
            (checkpoint / name).write_bytes(b"synthetic")
        linear = torch.nn.Linear(2, 2)
        torch.save(linear.state_dict(), checkpoint / "pytorch_model.bin")
        validate_checkpoint_model_state(checkpoint, linear)
        incomplete = Path(plan["output_dir"]) / "checkpoint-7"
        incomplete.mkdir()
        (incomplete / "trainer_state.json").write_text("{}\n", encoding="utf-8")
        _require(
            prepare_run_directory(plan, resume="latest") == checkpoint,
            "identical resume accepted and higher incomplete checkpoint skipped",
        )
        mismatched = Path(plan["output_dir"]) / "checkpoint-9"
        mismatched.mkdir()
        torch.save(torch.nn.Linear(3, 2).state_dict(), mismatched / "pytorch_model.bin")
        try:
            validate_checkpoint_model_state(mismatched, linear)
        except ValueError:
            pass
        else:
            raise RuntimeError("selftest failed: mismatched checkpoint state accepted")

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            mps_checkpoint = root / "mps-rng-checkpoint"
            torch.mps.manual_seed(731)
            saved_state = torch.mps.get_rng_state().clone()
            save_mps_rng_state(mps_checkpoint)
            expected_random = torch.rand(8, device="mps").cpu()
            torch.mps.manual_seed(991)
            load_mps_rng_state(mps_checkpoint, restore=True)
            observed_random = torch.rand(8, device="mps").cpu()
            _require(
                torch.equal(expected_random, observed_random),
                "MPS RNG checkpoint roundtrip continuity",
            )
            _require(
                torch.equal(
                    load_mps_rng_state(mps_checkpoint, restore=False),
                    saved_state,
                ),
                "MPS RNG state artifact is exact",
            )
            try:
                load_mps_rng_state(root / "missing-mps-rng", restore=False)
            except ValueError:
                pass
            else:
                raise RuntimeError("selftest failed: missing MPS RNG state accepted")

            def write_complete_checkpoint(path: Path) -> None:
                path.mkdir()
                (path / "trainer_state.json").write_text("{}\n", encoding="utf-8")
                for name in (
                    "optimizer.pt",
                    "scheduler.pt",
                    "training_args.bin",
                    "rng_state.pth",
                ):
                    (path / name).write_bytes(b"synthetic")
                torch.save(linear.state_dict(), path / "pytorch_model.bin")

            valid_mps = Path(plan["output_dir"]) / "checkpoint-5"
            write_complete_checkpoint(valid_mps)
            save_mps_rng_state(valid_mps)
            missing_mps = Path(plan["output_dir"]) / "checkpoint-6"
            write_complete_checkpoint(missing_mps)
            corrupt_mps = Path(plan["output_dir"]) / "checkpoint-8"
            write_complete_checkpoint(corrupt_mps)
            torch.save(
                torch.tensor([1.0], dtype=torch.float32),
                mps_rng_state_path(corrupt_mps),
            )
            _require(
                latest_checkpoint(
                    Path(plan["output_dir"]), require_mps_rng=True
                ) == valid_mps,
                "MPS resume skips higher missing/corrupt RNG checkpoints",
            )
        changed_plan = build_plan(replace(config, learning_rate=3e-5), data)
        changed_plan["output_dir"] = plan["output_dir"]
        try:
            prepare_run_directory(changed_plan, resume="latest")
        except ValueError as error:
            _require("identity mismatch" in str(error), "resume mismatch reason")
        else:
            raise RuntimeError("selftest failed: mismatched resume was accepted")

        grouped_train = root / "grouped-train.csv"
        grouped_valid = root / "grouped-valid.csv"

        def write_group_fixture(
            path: Path, rows: Sequence[tuple[str, str, str]]
        ) -> None:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("question", "post", "response")
                )
                writer.writeheader()
                for question, post, label in rows:
                    writer.writerow(
                        {
                            "question": question,
                            "post": post,
                            "response": f"{label}. Reasoning: synthetic fixture",
                        }
                    )

        write_group_fixture(
            grouped_train,
            [("q1", "train no", "no"), ("q2", "train yes", "yes"),
             ("train wording", "same source post", "no")],
        )
        write_group_fixture(
            grouped_valid,
            [("q3", "valid no", "no"), ("q4", "valid yes", "yes"),
             ("different wording", "same source post", "yes")],
        )
        grouped_config = replace(
            config,
            run_id="group-selftest",
            train_csv=grouped_train,
            valid_csv=grouped_valid,
            text_columns=("question", "post"),
            cross_split_leakage_columns=("post",),
        )
        grouped_data = prepare_data(grouped_config)
        _require(
            grouped_data.manifest["cross_split_train_rows_dropped"] == 1,
            "post-group leakage is removed despite different questions",
        )

        conflicting_valid = root / "conflicting-valid.csv"
        write_group_fixture(
            conflicting_valid,
            [("same q", "same post", "no"), ("same q", "same post", "yes")],
        )
        try:
            prepare_data(replace(grouped_config, valid_csv=conflicting_valid))
        except ValueError as error:
            _require("conflicting labels" in str(error), "valid conflict reason")
        else:
            raise RuntimeError("selftest failed: conflicting valid labels accepted")

        forbidden = root / "official-test.csv"
        write_fixture(forbidden, [("x", "no"), ("y", "yes")])
        try:
            reject_selection_path(forbidden, "valid")
        except ValueError:
            pass
        else:
            raise RuntimeError("selftest failed: test path was accepted for selection")

    print(
        "Generic text classifier selftest PASS: synthetic CSV hashes/manifests, "
        "train-side overlap removal, binary/multiclass CE/weighted-CE/focal and "
        "deterministic metrics, imbalanced weights, invalid/empty-class rejection, "
        "label-ordered confusion matrices, integer warmup, earliest-tie checkpoint "
        "epoch, complete checkpoint resume, strict tensor-state validation, Git "
        "identity, exact MPS RNG resume, post-group cross-split leakage, "
        "conflicting-valid rejection, and test-path rejection; no model download"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone trainer CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--run-id")
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--text-columns", default="post")
    parser.add_argument("--cross-split-leakage-columns", default="post")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="exact")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--head-type", default=DEFAULT_HEAD_TYPE)
    parser.add_argument("--loss", choices=LOSS_NAMES, default="ce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=DEVICE_NAMES, default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainerConfig:
    """Convert validated CLI values to the immutable run configuration."""

    required = {
        "--task-id": args.task_id,
        "--run-id": args.run_id,
        "--train-csv": args.train_csv,
        "--valid-csv": args.valid_csv,
        "--labels": args.labels,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    text_columns = tuple(
        value.strip() for value in args.text_columns.split(",") if value.strip()
    )
    leakage_columns = tuple(
        value.strip()
        for value in args.cross_split_leakage_columns.split(",")
        if value.strip()
    )
    return TrainerConfig(
        task_id=args.task_id,
        run_id=args.run_id,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        labels=tuple(args.labels),
        text_columns=text_columns,
        cross_split_leakage_columns=leakage_columns,
        label_column=args.label_column,
        label_mode=args.label_mode,
        model=args.model,
        model_revision=args.model_revision,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        head_type=args.head_type,
        loss=args.loss,
        seed=args.seed,
        output_root=args.output_root,
        device=args.device,
        max_length=args.max_length,
        train_batch=args.train_batch,
        eval_batch=args.eval_batch,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        focal_gamma=args.focal_gamma,
        early_stopping_patience=args.early_stopping_patience,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run a dry plan, an explicit execution, or the offline selftest."""

    args = parse_args(argv)
    if args.selftest:
        selftest()
        return
    try:
        config = config_from_args(args)
        plan, result = run_classifier(
            config,
            execute=False,
            validate_data=args.validate_data,
            resume=None,
            manifest_out=args.manifest_out,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.validate_data:
        print(
            "Metadata-only preregistration; no CSV opened. Add --validate-data "
            "to hash/validate declared train and valid data."
        )
    else:
        print(
            "Data-validation dry-run only. Formal execution is available only "
            "through a validated benchmark adapter."
        )


if __name__ == "__main__":
    main()
