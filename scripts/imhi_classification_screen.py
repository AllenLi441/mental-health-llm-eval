#!/usr/bin/env python3
"""Plan or execute the fixed four-task IMHI single-seed classifier screen.

The campaign is intentionally narrow: DR, Irf, SAD, and dreaddit; one declared
seed; and the CE, weighted-CE, and focal arms. It uses only each task's released
``train`` and ``valid`` CSV files. The default command only preregisters the
validated public registry metadata and does not open external datasets.
``--validate-data`` explicitly reads/hashes train and valid without training;
only ``--execute`` can start the twelve jobs.

No official test file is accepted or opened by this adapter. Model selection is
weighted F1 on validation, with accuracy, macro F1, and per-class diagnostics
reported by :mod:`train_text_classifier`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from train_text_classifier import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TOKENIZER,
    DEFAULT_TOKENIZER_REVISION,
    DEFAULT_HEAD_TYPE,
    DEVICE_NAMES,
    LOSS_NAMES,
    PreparedData,
    TrainerConfig,
    atomic_write_json,
    build_plan,
    canonical_json,
    completed_result,
    execute_training,
    git_identity,
    prepare_data,
    sha256_bytes,
    validate_identifier,
    validate_registry_execution_proof,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREEN_OUTPUT = ROOT / "results" / "imhi-classification-screen"
DEFAULT_REGISTRY = ROOT / "benchmark-specs" / "registry.json"
SCREEN_TASKS = ("DR", "Irf", "SAD", "dreaddit")
SCREEN_LOSSES = LOSS_NAMES
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SAD_LABELS = (
    "school",
    "financial problem",
    "family issues",
    "social relationships",
    "work",
    "health issues",
    "emotional turmoil",
    "everyday decision making",
    "other causes",
)


@dataclass(frozen=True)
class ImhiTask:
    """Immutable mapping from an IMHI task to released train/valid CSVs."""

    name: str
    task_id: str
    train_relative: Path
    valid_relative: Path
    labels: tuple[str, ...]
    expected_train_rows: int
    expected_valid_rows: int


IMHI_TASKS = (
    ImhiTask(
        name="DR",
        task_id="imhi-dr",
        train_relative=Path("DR/reddit_train.csv"),
        valid_relative=Path("DR/reddit_valid.csv"),
        labels=("yes", "no"),
        expected_train_rows=1003,
        expected_valid_rows=430,
    ),
    ImhiTask(
        name="Irf",
        task_id="imhi-irf",
        train_relative=Path("Irf/train.csv"),
        valid_relative=Path("Irf/val.csv"),
        labels=("yes", "no"),
        expected_train_rows=3943,
        expected_valid_rows=985,
    ),
    ImhiTask(
        name="SAD",
        task_id="imhi-sad",
        train_relative=Path("SAD/train.csv"),
        valid_relative=Path("SAD/val.csv"),
        labels=SAD_LABELS,
        expected_train_rows=5547,
        expected_valid_rows=616,
    ),
    ImhiTask(
        name="dreaddit",
        task_id="imhi-dreaddit",
        train_relative=Path("dreaddit/dreaddit-train.csv"),
        valid_relative=Path("dreaddit/val.csv"),
        labels=("yes", "no"),
        expected_train_rows=2837,
        expected_valid_rows=300,
    ),
)


@dataclass(frozen=True)
class ScreenJob:
    """One loss arm plus its already validated train/valid data."""

    task: ImhiTask
    config: TrainerConfig
    data: PreparedData
    plan: dict[str, Any]


@dataclass(frozen=True)
class RegistryBinding:
    """Canonical registry task/profile and their validated commitments."""

    task_spec: dict[str, Any]
    profile: dict[str, Any]
    task_spec_sha256: str
    profile_sha256: str


@dataclass(frozen=True)
class RegistrySnapshot:
    """One validator-approved registry bundle used by the whole campaign."""

    path: Path
    schema_sha256: str
    index_sha256: str
    bundle_sha256: str
    bindings: dict[str, RegistryBinding]


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp for campaign status only."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def attempted_execute_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Create a text-free attempt record before any execute preflight read."""

    return {
        "schema_version": 1,
        "screen_id": args.screen_id,
        "generated_at": utc_now(),
        "status": "ATTEMPTED",
        "phase": "PREFLIGHT",
        "mode": "execute",
        "execute_requested": True,
        "external_dataset_read": False,
        "protocol_request": {
            "tasks": list(SCREEN_TASKS),
            "losses": list(SCREEN_LOSSES),
            "seed": args.seed,
            "number_of_jobs": len(SCREEN_TASKS) * len(SCREEN_LOSSES),
            "model": args.model,
            "model_revision": args.model_revision,
            "tokenizer_id": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision,
        },
        "privacy_boundary": (
            "preflight artifact contains no source text, row ids, credentials, "
            "or exception message"
        ),
    }


def failed_execute_manifest(
    path: Path, fallback: dict[str, Any], error: Exception
) -> dict[str, Any]:
    """Persist an aggregate terminal failure without raw exception material."""

    try:
        current = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            current = dict(fallback)
    except (OSError, json.JSONDecodeError):
        current = dict(fallback)
    previous_status = current.get("status")
    execution_phase = (
        current.get("phase") == "EXECUTION"
        or previous_status == "RUNNING"
        or "started_at" in current
    )
    message = str(error)
    lowered = message.casefold()
    if "registry" in lowered:
        category = "REGISTRY_PREFLIGHT"
    elif any(token in lowered for token in ("dataset", "csv", "row", "split")):
        category = "DATA_PREFLIGHT"
    elif any(token in lowered for token in ("model", "tokenizer", "revision")):
        category = "PROTOCOL_PREFLIGHT"
    else:
        category = "EXECUTION" if execution_phase else "PREFLIGHT"
    current.update(
        {
            "status": "FAILED",
            "phase": "EXECUTION" if execution_phase else "PREFLIGHT",
            "failed_at": utc_now(),
            "error": {
                "category": category,
                "type": type(error).__name__,
                "message_sha256": sha256_bytes(message.encode("utf-8")),
                "message_persisted": False,
            },
        }
    )
    atomic_write_json(path, current)
    return current


def _registry_checker(registry_path: Path, *extra: str) -> str:
    """Run the repository validator as the sole registry authorization gate."""

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


def load_registry_snapshot(registry_path: Path) -> RegistrySnapshot:
    """Validate, hash, reload, and bind the four specialist profiles."""

    path = registry_path.expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"registry must be a regular non-symlink file: {path}")
    pass_output = _registry_checker(path)
    if "benchmark registry PASS:" not in pass_output:
        raise ValueError("benchmark registry checker did not emit its PASS marker")
    try:
        hashes = json.loads(_registry_checker(path, "--print-hashes"))
    except json.JSONDecodeError as error:
        raise ValueError("registry checker emitted invalid hash JSON") from error

    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for relative in manifest.get("task_files", []):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe registry task file path: {relative}")
        task_path = path.parent / relative_path
        if not task_path.is_file() or task_path.is_symlink():
            raise ValueError(f"registry task file is not regular: {task_path}")
        records = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"registry task file is not an array: {task_path}")
        tasks.extend(records)

    observed_hashes = {
        "schema_sha256": sha256_bytes((path.parent / "schema.json").read_bytes()),
        "registry_index_sha256": sha256_bytes(path.read_bytes()),
        "registry_bundle_sha256": sha256_bytes(
            canonical_json({"manifest": manifest, "tasks": tasks}).encode("utf-8")
        ),
        "task_spec_sha256": {
            task["task_key"]: sha256_bytes(canonical_json(task).encode("utf-8"))
            for task in tasks
        },
    }
    if hashes != observed_hashes:
        raise ValueError("registry changed after validator hash authorization")
    by_key = {task["task_key"]: task for task in tasks}
    bindings: dict[str, RegistryBinding] = {}
    for task in IMHI_TASKS:
        try:
            task_spec = by_key[task.task_id]
            profile = task_spec["supervised_classifier_profile"]
            task_hash = hashes["task_spec_sha256"][task.task_id]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"registry lacks supervised classifier profile for {task.task_id}"
            ) from error
        bindings[task.task_id] = RegistryBinding(
            task_spec=task_spec,
            profile=profile,
            task_spec_sha256=task_hash,
            profile_sha256=sha256_bytes(canonical_json(profile).encode("utf-8")),
        )
    return RegistrySnapshot(
        path=path,
        schema_sha256=hashes["schema_sha256"],
        index_sha256=hashes["registry_index_sha256"],
        bundle_sha256=hashes["registry_bundle_sha256"],
        bindings=bindings,
    )


def resolve_dataset_root(argument: Path | None) -> Path:
    """Resolve the external dataset root without embedding a private default."""

    value = argument
    if value is None:
        environment = os.environ.get("EVAL_DATASETS_DIR", "").strip()
        value = Path(environment) if environment else None
    if value is None:
        raise ValueError(
            "--dataset-root or EVAL_DATASETS_DIR is required; no licensed dataset "
            "path is embedded"
        )
    resolved = value.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"dataset root is not a directory: {resolved}")
    return resolved


def imhi_complete_data_root(dataset_root: Path) -> Path:
    """Return the released completion-format train/valid directory."""

    path = (
        dataset_root
        / "MentaLLaMA"
        / "train_data"
        / "complete_data"
    ).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"IMHI complete_data directory is missing: {path}")
    return path


def base_task_config(
    task: ImhiTask,
    *,
    registry: RegistrySnapshot,
    binding: RegistryBinding,
    data_root: Path,
    screen_id: str,
    output_root: Path,
    model: str,
    model_revision: str,
    tokenizer: str,
    tokenizer_revision: str,
    seed: int,
    device: str,
    max_length: int,
    train_batch: int,
    eval_batch: int,
    gradient_accumulation: int,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    focal_gamma: float,
    early_stopping_patience: int,
) -> TrainerConfig:
    """Construct the shared task identity before applying a loss arm."""

    profile = binding.profile
    protocol = profile["training_protocol"]
    requested_protocol = {
        "max_length": max_length,
        "train_batch_size": train_batch,
        "eval_batch_size": eval_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "optimizer": "adamw_torch",
        "scheduler": "linear",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "early_stopping_patience": early_stopping_patience,
        "best_model_metric": "weighted_f1",
        "greater_is_better": True,
        "tie_break_rule": "first_strict_improvement_checkpoint",
        "focal_gamma": focal_gamma,
        "class_weighting": "balanced_inverse_frequency_after_cleaning",
        "max_grad_norm": 1.0,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": False,
    }
    expected_profile = {
        "task_type": "single_label_text_classification",
        "labels": list(task.labels),
        "text_columns": ["question", "post"],
        "cross_split_leakage_columns": ["post"],
        "input_parser": "csv-dictreader-utf8-sig-newline-join-v1",
        "label_column": "response",
        "label_parser": "imhi-response-prefix",
        "train_relative_path": (
            Path("train_data/complete_data") / task.train_relative
        ).as_posix(),
        "dev_relative_path": (
            Path("train_data/complete_data") / task.valid_relative
        ).as_posix(),
        "test_access": False,
        "loss_arms": list(SCREEN_LOSSES),
        "seed": seed,
        "primary_metric": "weighted_f1",
        "model": {
            "base_model_id": model,
            "base_model_revision": model_revision,
            "tokenizer_id": tokenizer,
            "tokenizer_revision": tokenizer_revision,
            "head_type": DEFAULT_HEAD_TYPE,
        },
        "training_protocol": requested_protocol,
        "source_rows": {
            "train": task.expected_train_rows,
            "dev": task.expected_valid_rows,
        },
        "data_cleaning_order": [
            "same_label_content_keep_first",
            "conflicting_label_content_drop_group",
            "train_dev_content_overlap_drop_train_keep_dev",
        ],
    }
    for field, expected in expected_profile.items():
        if profile.get(field) != expected:
            raise ValueError(
                f"{task.task_id} registry profile {field} mismatch: "
                f"expected {expected!r}, got {profile.get(field)!r}"
            )
    model_profile = profile["model"]
    return TrainerConfig(
        task_id=task.task_id,
        run_id=screen_id,
        train_csv=data_root / task.train_relative,
        valid_csv=data_root / task.valid_relative,
        labels=task.labels,
        # Irf's question names the interpersonal-risk construct. Keeping the
        # question for every task gives one uniform input contract.
        text_columns=("question", "post"),
        cross_split_leakage_columns=tuple(
            profile["cross_split_leakage_columns"]
        ),
        label_column="response",
        label_mode="imhi-response-prefix",
        model=model_profile["base_model_id"],
        model_revision=model_profile["base_model_revision"],
        tokenizer=model_profile["tokenizer_id"],
        tokenizer_revision=model_profile["tokenizer_revision"],
        head_type=model_profile["head_type"],
        loss="ce",
        seed=seed,
        output_root=output_root,
        device=device,
        max_length=max_length,
        train_batch=train_batch,
        eval_batch=eval_batch,
        gradient_accumulation=gradient_accumulation,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        focal_gamma=focal_gamma,
        early_stopping_patience=early_stopping_patience,
        registry_path=str(registry.path.relative_to(ROOT)),
        registry_bundle_sha256=registry.bundle_sha256,
        registry_index_sha256=registry.index_sha256,
        registry_schema_sha256=registry.schema_sha256,
        task_spec_sha256=binding.task_spec_sha256,
        supervised_profile_id=profile["profile_id"],
        supervised_profile_sha256=binding.profile_sha256,
    )


def build_jobs(
    *,
    registry: RegistrySnapshot,
    dataset_root: Path,
    screen_id: str,
    output_root: Path,
    model: str,
    model_revision: str,
    tokenizer: str,
    tokenizer_revision: str,
    seed: int,
    device: str,
    max_length: int,
    train_batch: int,
    eval_batch: int,
    gradient_accumulation: int,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    focal_gamma: float,
    early_stopping_patience: int,
    repository_git: dict[str, Any] | None = None,
    enforce_expected_rows: bool = True,
) -> list[ScreenJob]:
    """Validate all four datasets and create exactly twelve single-seed jobs."""

    validate_identifier(screen_id, "screen_id")
    if tuple(task.name for task in IMHI_TASKS) != SCREEN_TASKS:
        raise ValueError("IMHI screen task registry drifted from the frozen four tasks")
    data_root = imhi_complete_data_root(dataset_root)
    frozen_git = dict(repository_git or git_identity())
    jobs: list[ScreenJob] = []
    for task in IMHI_TASKS:
        base = base_task_config(
            task,
            registry=registry,
            binding=registry.bindings[task.task_id],
            data_root=data_root,
            screen_id=screen_id,
            output_root=output_root,
            model=model,
            model_revision=model_revision,
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            device=device,
            max_length=max_length,
            train_batch=train_batch,
            eval_batch=eval_batch,
            gradient_accumulation=gradient_accumulation,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            focal_gamma=focal_gamma,
            early_stopping_patience=early_stopping_patience,
        )
        data = prepare_data(base)
        if enforce_expected_rows:
            profile = registry.bindings[task.task_id].profile
            observed_train = data.manifest["train_source"]["rows"]
            observed_valid = data.manifest["valid_source"]["rows"]
            if observed_train != task.expected_train_rows:
                raise ValueError(
                    f"{task.name} train row count drift: expected "
                    f"{task.expected_train_rows}, got {observed_train}"
                )
            if observed_valid != task.expected_valid_rows:
                raise ValueError(
                    f"{task.name} valid row count drift: expected "
                    f"{task.expected_valid_rows}, got {observed_valid}"
                )
            observed_used = {
                "train": data.manifest["used_train_rows"],
                "dev": data.manifest["used_valid_rows"],
            }
            if observed_used != profile["used_rows"]:
                raise ValueError(
                    f"{task.name} cleaned row count drift: expected "
                    f"{profile['used_rows']}, got {observed_used}"
                )
            split_commitments = profile["split_commitments"]
            expected_commitment_contract = {
                "hash_algorithm": "sha256",
                "commitment_scope": "raw_file_bytes_before_parsing",
            }
            for field, expected in expected_commitment_contract.items():
                if split_commitments.get(field) != expected:
                    raise ValueError(
                        f"{task.name} split commitment {field} drift: expected "
                        f"{expected!r}, got {split_commitments.get(field)!r}"
                    )
            observed_split_hashes = {
                "train_file_sha256": data.manifest["train_source"]["file_sha256"],
                "dev_file_sha256": data.manifest["valid_source"]["file_sha256"],
            }
            for field, observed in observed_split_hashes.items():
                if observed != split_commitments.get(field):
                    raise ValueError(
                        f"{task.name} {field} drift: registry="
                        f"{split_commitments.get(field)!r}, observed={observed!r}"
                    )
        for loss in SCREEN_LOSSES:
            config = replace(base, loss=loss)
            plan = build_plan(config, data, repository_git=frozen_git)
            if enforce_expected_rows:
                validate_registry_execution_proof(config, data, plan)
            jobs.append(
                ScreenJob(
                    task=task,
                    config=config,
                    data=data,
                    plan=plan,
                )
            )
    if len(jobs) != len(SCREEN_TASKS) * len(SCREEN_LOSSES):
        raise ValueError("screen must contain four tasks by three loss arms")
    if len({job.config.seed for job in jobs}) != 1:
        raise ValueError("screen must use exactly one seed")
    return jobs


def preregistration_manifest(
    *,
    registry: RegistrySnapshot,
    screen_id: str,
    output_root: Path,
    model: str,
    model_revision: str,
    tokenizer: str,
    tokenizer_revision: str,
    seed: int,
    device: str,
    max_length: int,
    train_batch: int,
    eval_batch: int,
    gradient_accumulation: int,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    focal_gamma: float,
    early_stopping_patience: int,
) -> dict[str, Any]:
    """Freeze the registry-backed campaign without opening external data."""

    validate_identifier(screen_id, "screen_id")
    configs: dict[str, TrainerConfig] = {}
    placeholder = Path("EXTERNAL_DATA_NOT_READ")
    for task in IMHI_TASKS:
        configs[task.task_id] = base_task_config(
            task,
            registry=registry,
            binding=registry.bindings[task.task_id],
            data_root=placeholder,
            screen_id=screen_id,
            output_root=output_root,
            model=model,
            model_revision=model_revision,
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            device=device,
            max_length=max_length,
            train_batch=train_batch,
            eval_batch=eval_batch,
            gradient_accumulation=gradient_accumulation,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            focal_gamma=focal_gamma,
            early_stopping_patience=early_stopping_patience,
        )
    protocol = registry.bindings[IMHI_TASKS[0].task_id].profile[
        "training_protocol"
    ]
    return {
        "schema_version": 1,
        "screen_id": screen_id,
        "generated_at": utc_now(),
        "status": "PREREGISTERED",
        "mode": "metadata_only_preregistration",
        "dry_run_default": True,
        "external_dataset_read": False,
        "data_validation_requires_flag": "--validate-data",
        "training_requires_flag": "--execute",
        "git": git_identity(),
        "registry": {
            "path": str(registry.path.relative_to(ROOT)),
            "schema_sha256": registry.schema_sha256,
            "index_sha256": registry.index_sha256,
            "bundle_sha256": registry.bundle_sha256,
            "validator": "scripts/check_benchmark_registry.py PASS",
        },
        "protocol": {
            "tasks": list(SCREEN_TASKS),
            "losses": list(SCREEN_LOSSES),
            "seed": seed,
            "number_of_seeds": 1,
            "number_of_jobs": len(SCREEN_TASKS) * len(SCREEN_LOSSES),
            "model": model,
            "model_revision": model_revision,
            "tokenizer_id": tokenizer,
            "tokenizer_revision": tokenizer_revision,
            "head_type": DEFAULT_HEAD_TYPE,
            "training_protocol": protocol,
            "selection_split": "valid",
            "selection_metric": "weighted_f1",
            "test_or_holdout_selection": "forbidden",
        },
        "jobs": [
            {
                "task": task.name,
                "task_id": task.task_id,
                "loss": loss,
                "seed": seed,
                "status": "AWAITING_EXPLICIT_DATA_VALIDATION",
                "output_dir": str(
                    output_root.resolve()
                    / task.task_id
                    / screen_id
                    / loss
                    / f"seed-{seed}"
                ),
                "task_spec_sha256": registry.bindings[
                    task.task_id
                ].task_spec_sha256,
                "supervised_profile_id": configs[
                    task.task_id
                ].supervised_profile_id,
                "supervised_profile_sha256": registry.bindings[
                    task.task_id
                ].profile_sha256,
            }
            for task in IMHI_TASKS
            for loss in SCREEN_LOSSES
        ],
    }


def campaign_manifest(
    *,
    registry: RegistrySnapshot,
    screen_id: str,
    dataset_root: Path,
    model: str,
    model_revision: str,
    tokenizer: str,
    tokenizer_revision: str,
    seed: int,
    jobs: Sequence[ScreenJob],
    execute: bool,
) -> dict[str, Any]:
    """Build the aggregate-only campaign manifest."""

    implementation_path = Path(__file__).resolve()
    representative = jobs[0].plan["identity"]
    return {
        "schema_version": 1,
        "screen_id": screen_id,
        "generated_at": utc_now(),
        "status": "READY_TO_EXECUTE" if execute else "DRY_RUN",
        "mode": "execute" if execute else "explicit_data_validation",
        "dry_run_default": True,
        "execute_requested": execute,
        "external_dataset_read": True,
        "data_validation_requested": True,
        "execute_requires_flag": "--execute",
        "implementation": {
            "path": str(implementation_path.relative_to(ROOT)),
            "sha256": sha256_bytes(implementation_path.read_bytes()),
        },
        "git": representative["git"],
        "registry": {
            "path": str(registry.path.relative_to(ROOT)),
            "schema_sha256": registry.schema_sha256,
            "index_sha256": registry.index_sha256,
            "bundle_sha256": registry.bundle_sha256,
            "validator": "scripts/check_benchmark_registry.py PASS",
            "task_spec_sha256": {
                task_id: binding.task_spec_sha256
                for task_id, binding in sorted(registry.bindings.items())
            },
            "supervised_profile_sha256": {
                task_id: binding.profile_sha256
                for task_id, binding in sorted(registry.bindings.items())
            },
        },
        "protocol": {
            "tasks": list(SCREEN_TASKS),
            "losses": list(SCREEN_LOSSES),
            "seed": seed,
            "number_of_seeds": 1,
            "number_of_jobs": len(jobs),
            "model": model,
            "model_revision": model_revision,
            "tokenizer_id": tokenizer,
            "tokenizer_revision": tokenizer_revision,
            "head_type": DEFAULT_HEAD_TYPE,
            "model_revision_immutable": bool(
                IMMUTABLE_REVISION_RE.fullmatch(model_revision)
            ),
            "device_requested": representative["device_requested"],
            "device_resolved": representative["device_resolved"],
            "hyperparameters": {
                "max_length": representative["max_length"],
                "train_batch": representative["train_batch"],
                "eval_batch": representative["eval_batch"],
                "gradient_accumulation": representative["gradient_accumulation"],
                "epochs": representative["epochs"],
                "learning_rate": representative["learning_rate"],
                "weight_decay": representative["weight_decay"],
                "warmup_ratio": representative["warmup_ratio"],
                "warmup_steps_by_task": {
                    job.task.name: job.plan["identity"]["schedule"]["warmup_steps"]
                    for job in jobs[::len(SCREEN_LOSSES)]
                },
                "focal_gamma": representative["focal_gamma"],
                "early_stopping_patience": representative[
                    "early_stopping_patience"
                ],
            },
            "selection_split": "valid",
            "selection_metric": "weighted_f1",
            "selection_tie_break": representative["selection_tie_break"],
            "deterministic_execution": representative["deterministic_execution"],
            "reported_metrics": [
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "per_class",
                "confusion_matrix",
            ],
            "test_or_holdout_selection": "forbidden and not accepted by the adapter",
            "cross_split_policy": (
                "drop train rows sharing the frozen post-only leakage key with "
                "valid; keep full valid"
            ),
            "within_train_duplicate_policy": (
                "keep first same-label content; drop conflicting-label content groups"
            ),
        },
        "dataset_root": str(dataset_root),
        "jobs": [
            {
                "task": job.task.name,
                "loss": job.config.loss,
                "seed": job.config.seed,
                "labels": list(job.config.labels),
                "text_columns": list(job.config.text_columns),
                "label_column": job.config.label_column,
                "label_mode": job.config.label_mode,
                "status": "PLANNED",
                "identity_sha256": job.plan["identity_sha256"],
                "registry_binding": job.plan["identity"]["registry_binding"],
                "output_dir": job.plan["output_dir"],
                "train_source": job.data.manifest["train_source"],
                "valid_source": job.data.manifest["valid_source"],
                "used_train_rows": job.data.manifest["used_train_rows"],
                "used_valid_rows": job.data.manifest["used_valid_rows"],
                "train_manifest_sha256": job.plan["identity"]
                ["train_manifest_sha256"],
                "dev_manifest_sha256": job.plan["identity"]["dev_manifest_sha256"],
                "cross_split_leakage_columns": list(
                    job.config.cross_split_leakage_columns
                ),
                "cross_split_train_rows_dropped": job.data.manifest[
                    "cross_split_train_rows_dropped"
                ],
                "within_train_same_label_duplicate_rows_dropped": job.data.manifest[
                    "within_train_same_label_duplicate_rows_dropped"
                ],
                "within_train_conflicting_content_groups_dropped": job.data.manifest[
                    "within_train_conflicting_content_groups_dropped"
                ],
                "within_train_conflicting_rows_dropped": job.data.manifest[
                    "within_train_conflicting_rows_dropped"
                ],
            }
            for job in jobs
        ],
        "privacy_boundary": (
            "manifest contains only paths, hashes, counts, identities, and aggregate "
            "metrics; no post text, row ids, or row-level predictions"
        ),
    }


def execute_campaign(
    jobs: Sequence[ScreenJob],
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    resume: str | None,
) -> dict[str, Any]:
    """Run twelve jobs serially while checkpointing aggregate campaign state."""

    manifest["status"] = "RUNNING"
    manifest["started_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    for index, job in enumerate(jobs):
        entry = manifest["jobs"][index]
        entry["status"] = "RUNNING"
        entry["started_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        try:
            reused = completed_result(job.plan) if resume else None
            if reused is not None:
                result = reused
                entry["resume_action"] = "reused identical completed result"
            else:
                run_dir_exists = Path(job.plan["output_dir"]).exists()
                job_resume = "latest" if resume and run_dir_exists else None
                result = execute_training(
                    job.config,
                    job.data,
                    job.plan,
                    resume=job_resume,
                )
            entry["status"] = "COMPLETE"
            entry["completed_at"] = utc_now()
            entry["metrics"] = result["metrics"]
            entry["result_path"] = job.plan["artifacts"]["result"]
        except Exception as error:
            entry["status"] = "FAILED"
            entry["failed_at"] = utc_now()
            entry["error"] = {
                "type": type(error).__name__,
                "message_sha256": sha256_bytes(str(error).encode("utf-8")),
                "message_persisted": False,
            }
            manifest["status"] = "FAILED"
            manifest["phase"] = "EXECUTION"
            manifest["failed_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            raise
        atomic_write_json(manifest_path, manifest)
    manifest["status"] = "COMPLETE"
    manifest["completed_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    return manifest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"selftest failed: {message}")


def selftest() -> None:
    """Build the fixed campaign from synthetic CSVs without loading a model."""

    with tempfile.TemporaryDirectory(prefix="imhi-screen-selftest-") as temporary:
        root = Path(temporary)
        data_root = root / "datasets"
        complete = data_root / "MentaLLaMA" / "train_data" / "complete_data"
        for task in IMHI_TASKS:
            train_path = complete / task.train_relative
            valid_path = complete / task.valid_relative
            train_path.parent.mkdir(parents=True, exist_ok=True)
            valid_path.parent.mkdir(parents=True, exist_ok=True)

            def write_rows(path: Path, split: str) -> None:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=("post", "question", "response")
                    )
                    writer.writeheader()
                    for index, label in enumerate(task.labels):
                        writer.writerow(
                            {
                                "post": f"synthetic {task.name} {split} {index}",
                                "question": f"synthetic question {task.name}",
                                "response": f"{label}. Reasoning: synthetic only",
                            }
                        )

            write_rows(train_path, "train")
            write_rows(valid_path, "valid")

        output_root = root / "outputs"
        registry = load_registry_snapshot(DEFAULT_REGISTRY)
        jobs = build_jobs(
            registry=registry,
            dataset_root=data_root,
            screen_id="synthetic-screen",
            output_root=output_root,
            model=DEFAULT_MODEL,
            model_revision=DEFAULT_MODEL_REVISION,
            tokenizer=DEFAULT_TOKENIZER,
            tokenizer_revision=DEFAULT_TOKENIZER_REVISION,
            seed=42,
            device="cpu",
            max_length=256,
            train_batch=8,
            eval_batch=16,
            gradient_accumulation=1,
            epochs=5.0,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            focal_gamma=2.0,
            early_stopping_patience=2,
            enforce_expected_rows=False,
        )
        _require(len(jobs) == 12, "four tasks by three losses")
        _require(
            {job.task.name for job in jobs} == set(SCREEN_TASKS),
            "only the frozen four tasks",
        )
        _require(
            {job.config.loss for job in jobs} == set(SCREEN_LOSSES),
            "all three loss arms",
        )
        _require({job.config.seed for job in jobs} == {42}, "single seed")
        _require(
            all(job.plan["identity"]["test_or_holdout_access"] is False for job in jobs),
            "no test or holdout selection",
        )
        manifest = campaign_manifest(
            registry=registry,
            screen_id="synthetic-screen",
            dataset_root=data_root,
            model=DEFAULT_MODEL,
            model_revision=DEFAULT_MODEL_REVISION,
            tokenizer=DEFAULT_TOKENIZER,
            tokenizer_revision=DEFAULT_TOKENIZER_REVISION,
            seed=42,
            jobs=jobs,
            execute=False,
        )
        manifest_path = output_root / "synthetic-screen" / "screen-manifest.json"
        atomic_write_json(manifest_path, manifest)
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(observed["status"] == "DRY_RUN", "dry-run default")
        _require(observed["protocol"]["number_of_jobs"] == 12, "output manifest")
        _require(
            observed["registry"]["bundle_sha256"] == registry.bundle_sha256,
            "validated registry bundle binding",
        )
        _require(
            all(
                job["registry_binding"]["supervised_profile_id"]
                for job in observed["jobs"]
            ),
            "per-job supervised profile binding",
        )
        failed_path = root / "failed-preflight.json"
        try:
            main(
                [
                    "--execute",
                    "--dataset-root",
                    str(root / "missing-dataset-root"),
                    "--device",
                    "cpu",
                    "--manifest-out",
                    str(failed_path),
                    "--output-root",
                    str(root / "failed-output"),
                ]
            )
        except SystemExit:
            pass
        else:
            raise RuntimeError("selftest failed: missing-root execute was accepted")
        failed = json.loads(failed_path.read_text(encoding="utf-8"))
        _require(failed["status"] == "FAILED", "terminal preflight status")
        _require(failed["phase"] == "PREFLIGHT", "preflight failure phase")
        _require(
            failed["error"]["message_persisted"] is False
            and "message" not in failed["error"],
            "preflight error remains aggregate-only",
        )
        try:
            main(["--resume", "latest"])
        except SystemExit as error:
            _require("requires --execute" in str(error), "resume mode rejection")
        else:
            raise RuntimeError("selftest failed: resume without execute was accepted")

    print(
        "IMHI classifier screen selftest PASS: synthetic DR/Irf/SAD/dreaddit, "
        "one seed, three loss arms, validator-approved task/profile/bundle hashes, "
        "frozen protocol, aggregate preflight failure artifact, strict resume mode, "
        "dry-run manifest, no test access, and no model download"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed campaign CLI; task and loss lists are not user-mutable."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--resume", choices=("latest",))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--screen-id", default="imhi-four-task-single-seed-v1")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SCREEN_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION
    )
    parser.add_argument("--seed", type=int, default=42)
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


def main(argv: Sequence[str] | None = None) -> None:
    """Write the dry-run manifest or explicitly execute the fixed campaign."""

    args = parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.resume and not args.execute:
        raise SystemExit("--resume requires --execute")
    manifest_path: Path | None = None
    attempt: dict[str, Any] | None = None
    try:
        output_root = args.output_root.expanduser().resolve()
        manifest_path = (
            args.manifest_out.expanduser().resolve()
            if args.manifest_out
            else output_root / args.screen_id / "screen-manifest.json"
        )
        if args.execute:
            attempt = attempted_execute_manifest(args)
            atomic_write_json(manifest_path, attempt)
        registry = load_registry_snapshot(args.registry)
        if args.execute and (
            not IMMUTABLE_REVISION_RE.fullmatch(args.model_revision)
            or not IMMUTABLE_REVISION_RE.fullmatch(args.tokenizer_revision)
        ):
            raise ValueError(
                "--execute requires an immutable 40- or 64-character hexadecimal "
                "model and tokenizer revision; aliases such as main are dry-run only"
            )
        if not args.validate_data and not args.execute:
            manifest = preregistration_manifest(
                registry=registry,
                screen_id=args.screen_id,
                output_root=output_root,
                model=args.model,
                model_revision=args.model_revision,
                tokenizer=args.tokenizer,
                tokenizer_revision=args.tokenizer_revision,
                seed=args.seed,
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
            jobs: list[ScreenJob] = []
        else:
            dataset_root = resolve_dataset_root(args.dataset_root)
            jobs = build_jobs(
                registry=registry,
                dataset_root=dataset_root,
                screen_id=args.screen_id,
                output_root=output_root,
                model=args.model,
                model_revision=args.model_revision,
                tokenizer=args.tokenizer,
                tokenizer_revision=args.tokenizer_revision,
                seed=args.seed,
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
            manifest = campaign_manifest(
                registry=registry,
                screen_id=args.screen_id,
                dataset_root=dataset_root,
                model=args.model,
                model_revision=args.model_revision,
                tokenizer=args.tokenizer,
                tokenizer_revision=args.tokenizer_revision,
                seed=args.seed,
                jobs=jobs,
                execute=args.execute,
            )
        atomic_write_json(manifest_path, manifest)
        if args.execute:
            manifest = execute_campaign(
                jobs,
                manifest,
                manifest_path,
                resume=args.resume,
            )
    except (OSError, RuntimeError, ValueError) as error:
        if args.execute and manifest_path is not None:
            failed_execute_manifest(
                manifest_path,
                attempt or attempted_execute_manifest(args),
                error,
            )
        raise SystemExit(str(error)) from error

    summary = {
        "status": manifest["status"],
        "screen_id": manifest["screen_id"],
        "tasks": manifest["protocol"]["tasks"],
        "losses": manifest["protocol"]["losses"],
        "seed": manifest["protocol"]["seed"],
        "jobs": manifest["protocol"]["number_of_jobs"],
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.execute and not args.validate_data:
        print(
            "Registry-only preregistration; no external dataset opened. Add "
            "--validate-data to hash/validate the four train/valid pairs."
        )
    elif not args.execute:
        print("Data-validation dry-run only. Add --execute to start training.")


if __name__ == "__main__":
    main()
