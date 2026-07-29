#!/usr/bin/env python3
"""Score the frozen PsySUICIDE train holdout exactly once.

The scorer is intentionally separate from the trainer. It defaults to a
holdout-closed dry run and requires the selected checkpoint, aggregate valid
report, scorer source, and partition commitment to be tracked, clean, committed,
and pushed before ``--execute`` can load holdout membership.

Only aggregate metrics are written under ``reports/``. The one-shot claim stays
under ignored ``results/`` and is created atomically before the licensed train
file is read, so a failed or interrupted attempt cannot be silently rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "reports" / "psysuicide-roberta-v1.prereg.json"
COMMITMENT_PATH = ROOT / "reports" / "psysuicide-v2-train-holdout.commitment.json"
BASE_MODEL_MANIFEST_PATH = (
    ROOT / "reports" / "psysuicide-roberta-v1-base-model-manifest.json"
)
DEFAULT_SELECTION = ROOT / "reports" / "psysuicide-roberta-v1-valid-selection.json"
DEFAULT_FREEZE = ROOT / "reports" / "psysuicide-roberta-v1-holdout-freeze.json"
DEFAULT_OUTPUT = ROOT / "reports" / "psysuicide-roberta-v1-holdout-result.json"
REPORTER_PATH = ROOT / "scripts" / "report_psysuicide_roberta.py"
RESULTS_ROOT = ROOT / "results" / "psysuicide-roberta"
DEFAULT_RUN_ID = "roberta-large-full-20260728"
PROTOCOL_ID = "psysuicide-roberta-v1-holdout-once"
EXPECTED_BRANCH = "codex/benchmark-model-optimization-v2"
EXPECTED_REMOTE = "origin"
PARTITION_SEED = "psysuicide-model-optimization-v2-2026-07-28"
LABELS = [
    "与自杀/自伤/攻击行为无关",
    "被动自杀意图",
    "主动自杀意图",
    "关于自杀的探索",
    "自杀计划",
    "自杀准备行为",
    "自杀未遂",
    "自伤意图",
    "自伤行为",
    "用户攻击行为",
    "他人攻击行为",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row(row: dict) -> str:
    return json.dumps(
        {
            "idx": str(row["idx"]),
            "labels": row["labels"],
            "text": str(row["text"]),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def commitment(digests: list[str]) -> str:
    return sha256_bytes("\n".join(sorted(digests)).encode())


def partition_holdout(rows: list[dict]) -> tuple[list[dict], dict]:
    """Rebuild only the precommitted holdout and aggregate partition manifest."""
    allowed = set(LABELS)
    retained = []
    dropped_multi = 0
    dropped_unknown = 0
    for row in rows:
        if not isinstance(row.get("labels"), list) or len(row["labels"]) != 1:
            dropped_multi += 1
            continue
        label = row["labels"][0]
        if label not in allowed:
            dropped_unknown += 1
            continue
        digest = sha256_bytes(canonical_row(row).encode())
        retained.append(
            {
                "row": row,
                "label": label,
                "row_digest": digest,
                "rank_digest": sha256_bytes(
                    f"{PARTITION_SEED}\0{digest}".encode()
                ),
            }
        )
    if len({entry["row_digest"] for entry in retained}) != len(retained):
        raise RuntimeError("duplicate canonical train rows")

    holdout = []
    holdout_digests = []
    optimization_digests = []
    per_label = {}
    for label in LABELS:
        group = sorted(
            (entry for entry in retained if entry["label"] == label),
            key=lambda entry: (entry["rank_digest"], entry["row_digest"]),
        )
        holdout_n = max(1, math.floor(len(group) * 0.2))
        holdout.extend(entry["row"] for entry in group[:holdout_n])
        holdout_digests.extend(entry["row_digest"] for entry in group[:holdout_n])
        optimization_digests.extend(
            entry["row_digest"] for entry in group[holdout_n:]
        )
        per_label[label] = {
            "retained": len(group),
            "optimization": len(group) - holdout_n,
            "holdout": holdout_n,
        }
    manifest = {
        "retained_single_label_rows": len(retained),
        "optimization_rows": len(optimization_digests),
        "holdout_rows": len(holdout),
        "dropped_multi_label_rows": dropped_multi,
        "dropped_unknown_label_rows": dropped_unknown,
        "per_label": per_label,
        "all_rows_commitment_sha256": commitment(
            [entry["row_digest"] for entry in retained]
        ),
        "optimization_commitment_sha256": commitment(optimization_digests),
        "holdout_commitment_sha256": commitment(holdout_digests),
    }
    return holdout, manifest


class HoldoutDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_ids = [LABELS.index(row["labels"][0]) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            str(self.rows[index]["text"]),
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = self.label_ids[index]
        return encoded


def metrics_from_predictions(gold: np.ndarray, predicted: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(
            f1_score(gold, predicted, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(gold, predicted, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def require_committed_and_pushed(
    paths: list[Path],
    expected_freeze_commit: str,
    execution_commit: str,
) -> tuple[str, str]:
    if (
        len(expected_freeze_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in expected_freeze_commit
        )
    ):
        raise RuntimeError("--expected-freeze-commit must be a full lowercase Git SHA")
    dirty = git_output("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError("tracked/untracked worktree changes remain; freeze commit required")
    head = git_output("rev-parse", "HEAD")
    if head != expected_freeze_commit:
        raise RuntimeError(
            f"HEAD differs from the expected freeze commit: {head} != {expected_freeze_commit}"
        )
    branch = git_output("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"unexpected freeze branch: {branch}")
    for path in paths:
        relative = str(path.resolve().relative_to(ROOT))
        subprocess.check_call(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    upstream = git_output(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    expected_upstream = f"{EXPECTED_REMOTE}/{EXPECTED_BRANCH}"
    if upstream != expected_upstream:
        raise RuntimeError(f"unexpected upstream: {upstream} != {expected_upstream}")
    upstream_head = git_output("rev-parse", "@{upstream}")
    if upstream_head != head:
        raise RuntimeError(
            f"freeze commit is not pushed: HEAD={head} {upstream}={upstream_head}"
        )
    remote_line = subprocess.check_output(
        [
            "git",
            "ls-remote",
            "--heads",
            EXPECTED_REMOTE,
            f"refs/heads/{EXPECTED_BRANCH}",
        ],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if not remote_line:
        raise RuntimeError("live remote freeze branch is missing")
    remote_head = remote_line.split()[0]
    if remote_head != head:
        raise RuntimeError(
            f"live remote SHA differs from freeze commit: {remote_head} != {head}"
        )
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", execution_commit, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "training execution commit is not an ancestor of the freeze commit"
        ) from error
    return head, upstream


def validate_selection(selection: dict, prereg: dict, frozen: dict) -> dict:
    if selection.get("schema_version") != 2:
        raise RuntimeError("unexpected selection schema_version")
    if (
        selection.get("scope")
        != "PsySUICIDE supervised v1 three-seed official-valid selection"
    ):
        raise RuntimeError("unexpected selection scope")
    if selection.get("preregistration_sha256") != sha256_file(PREREG_PATH):
        raise RuntimeError("selection is not bound to the current preregistration")
    identity = selection.get("execution_identity", {})
    if identity.get("execution_commit") != selection.get("execution_commit"):
        raise RuntimeError("selection execution identity mismatch")
    if (
        identity.get("execution_base_commit")
        != prereg["implementation"]["execution_base_commit"]
        or identity.get("training_script_sha256")
        != prereg["implementation"]["script_sha256"]
        or identity.get("preregistration_sha256") != sha256_file(PREREG_PATH)
        or identity.get("partition_commitment_artifact_sha256")
        != sha256_file(COMMITMENT_PATH)
    ):
        raise RuntimeError("selection execution identity differs from frozen protocol")
    base_identity = selection.get("base_model_identity", {})
    if (
        base_identity.get("base_model") != prereg["implementation"]["base_model"]
        or base_identity.get("base_model_revision")
        != prereg["implementation"]["base_model_revision"]
        or base_identity.get("local_cache_manifest_sha256")
        != sha256_file(BASE_MODEL_MANIFEST_PATH)
        or base_identity.get("verification_status")
        != "POST_RUN_LOCAL_ARTIFACT_MATCH"
    ):
        raise RuntimeError("selection base-model identity mismatch")
    partition = selection.get("partition", {})
    if partition.get("holdout_scored") is not False:
        raise RuntimeError("selection report says holdout was already scored")
    expected_partition = prereg["partition"]
    expected_commitment = prereg["partition"]["holdout_commitment_sha256"]
    if (
        partition.get("optimization_rows") != expected_partition["optimization_rows"]
        or partition.get("optimization_commitment_sha256")
        != expected_partition["optimization_commitment_sha256"]
        or partition.get("official_valid_rows")
        != expected_partition["official_valid_rows"]
        or partition.get("holdout_rows") != expected_partition["holdout_rows"]
        or partition.get("holdout_commitment_sha256") != expected_commitment
        or frozen["holdout_commitment_sha256"] != expected_commitment
    ):
        raise RuntimeError("selection partition differs from preregistration")
    configuration = selection.get("configuration", {})
    for field, expected in prereg["training"].items():
        if field in ("checkpoint_selection", "public_outputs"):
            continue
        if configuration.get(field) != expected:
            raise RuntimeError(f"selection training configuration mismatch: {field}")

    seed_results = selection.get("seed_results")
    if not isinstance(seed_results, list):
        raise RuntimeError("selection has no seed results")
    ordered = sorted(seed_results, key=lambda row: row.get("seed", -1))
    if [row.get("seed") for row in ordered] != prereg["training"]["seeds"]:
        raise RuntimeError("selection seed list differs from preregistration")
    metric_names = ("accuracy", "macro_f1", "weighted_f1")
    for row in ordered:
        for metric in metric_names:
            value = row.get(metric)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise RuntimeError(f"invalid selection seed metric: {metric}")
    means = {
        metric: float(np.mean([row[metric] for row in ordered]))
        for metric in metric_names
    }
    stds = {
        metric: float(np.std([row[metric] for row in ordered], ddof=1))
        for metric in metric_names
    }
    for metric in metric_names:
        if not math.isclose(
            selection.get("three_seed_mean", {}).get(metric, float("nan")),
            means[metric],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"selection mean metric mismatch: {metric}")
        if not math.isclose(
            selection.get("three_seed_sample_std", {}).get(
                metric, float("nan")
            ),
            stds[metric],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"selection metric std mismatch: {metric}")

    per_class = selection.get("per_class")
    if not isinstance(per_class, dict) or list(per_class) != LABELS:
        raise RuntimeError("selection per-class taxonomy mismatch")
    failures = []
    for label, reference_f1 in prereg["reference"][
        "per_class_f1_for_support_at_least_10"
    ].items():
        values = per_class.get(label, {})
        mean_f1 = values.get("mean_f1")
        if (
            not isinstance(mean_f1, (int, float))
            or not math.isfinite(mean_f1)
            or not 0.0 <= mean_f1 <= 1.0
        ):
            raise RuntimeError(f"invalid selection per-class mean F1: {label}")
        if values.get("reference_f1") != reference_f1:
            raise RuntimeError(f"selection reference F1 mismatch: {label}")
        delta = mean_f1 - reference_f1
        if not math.isclose(
            values.get("delta_vs_reference", float("nan")),
            delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"selection per-class delta mismatch: {label}")
        if delta < -0.10:
            failures.append(label)
    gate = selection.get("selection", {})
    primary_pass = means["macro_f1"] > prereg["reference"]["macro_f1"]
    regression_pass = not failures
    advances = primary_pass and regression_pass
    expected_decision = (
        "ADVANCE_SUPERVISED_V1_FREEZE_CHECKPOINT_BEFORE_HOLDOUT"
        if advances
        else "REJECT_SUPERVISED_V1"
    )
    if (
        gate.get("primary_reference_macro_f1")
        != prereg["reference"]["macro_f1"]
        or not math.isclose(
            gate.get("primary_delta", float("nan")),
            means["macro_f1"] - prereg["reference"]["macro_f1"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or gate.get("primary_pass") is not primary_pass
        or gate.get("regression_guard_failures") != failures
        or gate.get("regression_guard_pass") is not regression_pass
        or gate.get("candidate_advances") is not advances
        or gate.get("decision") != expected_decision
    ):
        raise RuntimeError("selection advancement gate is internally inconsistent")
    if not advances:
        raise RuntimeError("valid selection did not advance")

    best = max(ordered, key=lambda row: (row["macro_f1"], -row["seed"]))
    selected = selection.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise RuntimeError("selection report has no frozen checkpoint")
    if selected.get("seed") != best["seed"]:
        raise RuntimeError("selected checkpoint is not the preregistered best seed")
    if not math.isclose(
        selected.get("valid_macro_f1", float("nan")),
        best["macro_f1"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("selected checkpoint valid macro-F1 mismatch")
    weight_sha = selected.get("weight_sha256")
    if (
        not isinstance(weight_sha, str)
        or len(weight_sha) != 64
        or any(character not in "0123456789abcdef" for character in weight_sha)
    ):
        raise RuntimeError("selected checkpoint SHA-256 is invalid")
    if selected.get("label_order") != LABELS:
        raise RuntimeError("selected checkpoint label order mismatch")
    return selected


def validate_freeze_spec(
    spec: dict,
    selection: dict,
    selected: dict,
) -> dict:
    if spec.get("schema_version") != 1:
        raise RuntimeError("unexpected holdout-freeze schema_version")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("holdout-freeze protocol_id mismatch")
    if spec.get("run_id") != DEFAULT_RUN_ID:
        raise RuntimeError("holdout-freeze run_id mismatch")
    if spec.get("branch") != EXPECTED_BRANCH:
        raise RuntimeError("holdout-freeze branch mismatch")
    if spec.get("execution_commit") != selection.get("execution_commit"):
        raise RuntimeError("holdout-freeze execution commit mismatch")
    expected_artifacts = {
        "preregistration": PREREG_PATH,
        "partition_commitment": COMMITMENT_PATH,
        "base_model_manifest": BASE_MODEL_MANIFEST_PATH,
        "valid_selection": DEFAULT_SELECTION,
        "selection_reporter": REPORTER_PATH,
        "holdout_scorer": Path(__file__).resolve(),
    }
    declared_artifacts = spec.get("artifacts")
    if not isinstance(declared_artifacts, dict):
        raise RuntimeError("holdout-freeze artifact manifest missing")
    for name, path in expected_artifacts.items():
        expected_path = str(path.relative_to(ROOT))
        declared = declared_artifacts.get(name, {})
        if declared.get("path") != expected_path:
            raise RuntimeError(f"holdout-freeze artifact path mismatch: {name}")
        if declared.get("sha256") != sha256_file(path):
            raise RuntimeError(f"holdout-freeze artifact SHA-256 mismatch: {name}")
    checkpoint = spec.get("selected_checkpoint", {})
    expected_root = (
        f"results/psysuicide-roberta/{DEFAULT_RUN_ID}/"
        f"seed-{selected['seed']}/model"
    )
    if (
        checkpoint.get("seed") != selected["seed"]
        or checkpoint.get("root") != expected_root
        or checkpoint.get("files") != selected.get("artifact_files")
    ):
        raise RuntimeError("holdout-freeze selected checkpoint mismatch")
    inference = spec.get("inference", {})
    if (
        inference.get("label_order") != LABELS
        or inference.get("max_length") != 256
        or inference.get("eval_batch") != 8
        or inference.get("device") != "mps"
        or inference.get("dtype") != "float32"
        or inference.get("local_files_only") is not True
    ):
        raise RuntimeError("holdout-freeze inference configuration mismatch")
    outputs = spec.get("outputs", {})
    if (
        outputs.get("public_result")
        != str(DEFAULT_OUTPUT.relative_to(ROOT))
        or outputs.get("private_ledger_namespace") != PROTOCOL_ID
        or outputs.get("row_level_material") != "not persisted"
    ):
        raise RuntimeError("holdout-freeze output boundary mismatch")
    return checkpoint


def checkpoint_path(run_id: str, seed: int) -> Path:
    return RESULTS_ROOT / run_id / f"seed-{seed}" / "model"


def verify_checkpoint(path: Path, selected: dict, frozen_files: dict) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("selected checkpoint root must be a real local directory")
    weight_filename = selected.get("weight_filename")
    if (
        not isinstance(weight_filename, str)
        or Path(weight_filename).name != weight_filename
        or weight_filename not in ("model.safetensors", "pytorch_model.bin")
    ):
        raise RuntimeError("invalid selected checkpoint weight filename")
    if selected.get("artifact_files") != frozen_files:
        raise RuntimeError("selection and holdout-freeze checkpoint manifests differ")
    expected_names = set(frozen_files)
    actual_names = {
        candidate.name for candidate in path.iterdir() if candidate.is_file()
    }
    if actual_names != expected_names:
        raise RuntimeError(
            f"checkpoint file set mismatch: {sorted(actual_names)} != "
            f"{sorted(expected_names)}"
        )
    for filename, expected in frozen_files.items():
        if Path(filename).name != filename:
            raise RuntimeError("checkpoint manifest contains a path traversal")
        candidate = path / filename
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"checkpoint file must be a regular file: {filename}")
        if candidate.stat().st_size != expected.get("bytes"):
            raise RuntimeError(f"checkpoint byte count mismatch: {filename}")
        if sha256_file(candidate) != expected.get("sha256"):
            raise RuntimeError(f"checkpoint SHA-256 mismatch: {filename}")
    weight_path = path / weight_filename
    if frozen_files[weight_filename]["sha256"] != selected["weight_sha256"]:
        raise RuntimeError("selected weight SHA differs from checkpoint manifest")
    if frozen_files[weight_filename]["bytes"] != selected["weight_bytes"]:
        raise RuntimeError("selected weight byte count differs from checkpoint manifest")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    id2label = config.get("id2label", {})
    if (
        len(id2label) != len(LABELS)
        or [id2label.get(str(index)) for index in range(len(LABELS))] != LABELS
    ):
        raise RuntimeError("checkpoint label order mismatch")
    label2id = config.get("label2id", {})
    if (
        len(label2id) != len(LABELS)
        or [label2id.get(label) for label in LABELS]
        != list(range(len(LABELS)))
    ):
        raise RuntimeError("checkpoint reverse label map mismatch")
    return weight_path


def validate_partition(raw_train: bytes, rows: list[dict], frozen: dict) -> tuple[list[dict], dict]:
    if sha256_bytes(raw_train) != frozen["dataset_file_sha256"]:
        raise RuntimeError("licensed train file SHA-256 mismatch")
    holdout, manifest = partition_holdout(rows)
    for field in (
        "retained_single_label_rows",
        "optimization_rows",
        "holdout_rows",
        "all_rows_commitment_sha256",
        "optimization_commitment_sha256",
        "holdout_commitment_sha256",
    ):
        if manifest[field] != frozen[field]:
            raise RuntimeError(f"partition commitment mismatch for {field}")
    return holdout, manifest


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    fsync_directory(path.parent)


def write_all(descriptor: int, encoded: bytes) -> None:
    position = 0
    while position < len(encoded):
        written = os.write(descriptor, encoded[position:])
        if written <= 0:
            raise OSError("short write while persisting one-shot ledger")
        position += written


def durable_exclusive_json(path: Path, payload: dict) -> None:
    ensure_private_directory(path.parent)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def durable_atomic_json(path: Path, payload: dict, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        os.chmod(path, mode)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_checkpoint(
    checkpoint: Path,
    holdout: list[dict],
    max_length: int,
    eval_batch: int,
    expected_device: str,
) -> tuple[dict, str, float]:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    if expected_device == "mps":
        if not (
            torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            raise RuntimeError("frozen MPS inference device is unavailable")
        device = torch.device("mps")
    elif expected_device == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"unsupported frozen inference device: {expected_device}")
    model.to(device)
    model.eval()
    dataset = HoldoutDataset(holdout, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=eval_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    gold = []
    predicted = []
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            gold.extend(labels.cpu().numpy().tolist())
            predicted.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
    runtime_seconds = time.time() - started
    return (
        metrics_from_predictions(
            np.asarray(gold, dtype=np.int64),
            np.asarray(predicted, dtype=np.int64),
        ),
        str(device),
        runtime_seconds,
    )


def preflight_checkpoint(checkpoint: Path, expected_device: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    if expected_device == "mps":
        if not (
            torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            raise RuntimeError("frozen MPS inference device is unavailable")
        device = torch.device("mps")
    elif expected_device == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"unsupported frozen inference device: {expected_device}")
    model.to(device)
    model.eval()
    encoded = tokenizer(
        "合成预检文本",
        return_tensors="pt",
        truncation=True,
        max_length=32,
    )
    with torch.inference_mode():
        logits = model(
            **{key: value.to(device) for key, value in encoded.items()}
        ).logits
    if tuple(logits.shape) != (1, len(LABELS)):
        raise RuntimeError(f"checkpoint preflight shape mismatch: {tuple(logits.shape)}")
    del logits, encoded, model, tokenizer
    if expected_device == "mps":
        torch.mps.empty_cache()


def selftest() -> None:
    rows = []
    for label_index, label in enumerate(LABELS):
        for index in range(10):
            rows.append(
                {
                    "idx": f"{label_index}-{index}",
                    "labels": [label],
                    "text": f"synthetic-{label_index}-{index}",
                }
            )
    holdout, manifest = partition_holdout(rows)
    assert len(holdout) == 22
    assert manifest["optimization_rows"] == 88
    assert all(
        sum(row["labels"][0] == label for row in holdout) == 2
        for label in LABELS
    )
    gold = np.asarray([0, 1, 2, 3], dtype=np.int64)
    predicted = np.asarray([0, 1, 1, 3], dtype=np.int64)
    metrics = metrics_from_predictions(gold, predicted)
    assert metrics["accuracy"] == 0.75
    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        receipt = temporary_root / "ledger" / "claim.json"
        durable_exclusive_json(receipt, {"status": "CLAIMED"})
        try:
            durable_exclusive_json(receipt, {"status": "CLAIMED_AGAIN"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("second holdout claim was not rejected")

        checkpoint = temporary_root / "checkpoint"
        checkpoint.mkdir()
        contents = {
            "model.safetensors": b"fixture-weight",
            "config.json": (
                json.dumps(
                    {
                        "id2label": {
                            str(index): label
                            for index, label in enumerate(LABELS)
                        },
                        "label2id": {
                            label: index for index, label in enumerate(LABELS)
                        },
                    },
                    ensure_ascii=False,
                ).encode()
            ),
            "tokenizer.json": b"{}",
            "tokenizer_config.json": b"{}",
            "training_args.bin": b"fixture-args",
        }
        for filename, value in contents.items():
            (checkpoint / filename).write_bytes(value)
        files = {
            filename: {
                "bytes": len(value),
                "sha256": sha256_bytes(value),
            }
            for filename, value in contents.items()
        }
        selected = {
            "weight_filename": "model.safetensors",
            "weight_bytes": len(contents["model.safetensors"]),
            "weight_sha256": sha256_bytes(contents["model.safetensors"]),
            "artifact_files": files,
        }
        verify_checkpoint(checkpoint, selected, files)
        (checkpoint / "tokenizer_config.json").write_bytes(b'{"tampered":true}')
        try:
            verify_checkpoint(checkpoint, selected, files)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered checkpoint file was accepted")
    print(
        "PsySUICIDE RoBERTa holdout scorer selftest PASS: deterministic "
        "partition, aggregate metrics, full checkpoint manifest, durable "
        "exclusive one-shot claim"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--publish-existing", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--expected-freeze-commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return

    selection_bytes = DEFAULT_SELECTION.read_bytes()
    freeze_bytes = DEFAULT_FREEZE.read_bytes()
    selection = json.loads(selection_bytes)
    freeze_spec = json.loads(freeze_bytes)
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(COMMITMENT_PATH.read_text(encoding="utf-8"))
    selected = validate_selection(selection, prereg, frozen)
    frozen_checkpoint = validate_freeze_spec(freeze_spec, selection, selected)
    checkpoint = checkpoint_path(DEFAULT_RUN_ID, selected["seed"])
    weight_path = None
    if not args.publish_existing:
        weight_path = verify_checkpoint(
            checkpoint,
            selected,
            frozen_checkpoint["files"],
        )

    plan = {
        "scope": "one frozen PsySUICIDE train holdout confirmation",
        "protocol_id": PROTOCOL_ID,
        "run_id": DEFAULT_RUN_ID,
        "selected_seed": selected["seed"],
        "selected_checkpoint_sha256": selected["weight_sha256"],
        "selection_report_sha256": sha256_bytes(selection_bytes),
        "freeze_spec_sha256": sha256_bytes(freeze_bytes),
        "holdout_rows_committed": frozen["holdout_rows"],
        "holdout_commitment_sha256": frozen["holdout_commitment_sha256"],
        "holdout_loaded": False,
        "dry_run_default": True,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute and not args.publish_existing:
        print(
            "Dry-run only. Holdout membership remains unopened. "
            "The current local freeze spec and every local checkpoint byte match."
        )
        return
    if not args.dataset_root:
        raise SystemExit("--dataset-root is required for the private ledger")
    if not args.expected_freeze_commit:
        raise SystemExit(
            "--expected-freeze-commit is required for execute/publish-existing"
        )
    if DEFAULT_OUTPUT.exists():
        raise SystemExit(
            f"refusing to overwrite public holdout result: {DEFAULT_OUTPUT}"
        )

    freeze_commit, upstream = require_committed_and_pushed(
        [
            Path(__file__).resolve(),
            DEFAULT_FREEZE,
            DEFAULT_SELECTION,
            REPORTER_PATH,
            PREREG_PATH,
            COMMITMENT_PATH,
            BASE_MODEL_MANIFEST_PATH,
        ],
        args.expected_freeze_commit,
        selection["execution_commit"],
    )

    # Close the TOCTOU window after the Git/live-remote gate.
    if DEFAULT_SELECTION.read_bytes() != selection_bytes:
        raise SystemExit("selection report changed during the freeze check")
    if DEFAULT_FREEZE.read_bytes() != freeze_bytes:
        raise SystemExit("holdout-freeze spec changed during the freeze check")
    validate_freeze_spec(freeze_spec, selection, selected)

    ledger_root = (
        args.dataset_root.resolve()
        / ".eval-state"
        / "mental-health-llm-eval"
        / PROTOCOL_ID
    )
    claim_path = ledger_root / "access-started.json"
    completed_path = ledger_root / "score-completed.json"
    local_receipt = (
        RESULTS_ROOT
        / DEFAULT_RUN_ID
        / "holdout-one-shot.receipt.json"
    )

    if args.publish_existing:
        if not claim_path.is_file() or not completed_path.is_file():
            raise SystemExit("no completed private one-shot result is available")
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if (
            completed.get("protocol_id") != PROTOCOL_ID
            or completed.get("freeze_commit") != freeze_commit
            or not isinstance(completed.get("public_result"), dict)
        ):
            raise SystemExit("private completion receipt does not match the freeze")
        durable_atomic_json(DEFAULT_OUTPUT, completed["public_result"], mode=0o644)
        print(
            json.dumps(
                {
                    "status": "PUBLISHED_EXISTING_COMPLETION",
                    "output": str(DEFAULT_OUTPUT),
                    "output_sha256": sha256_file(DEFAULT_OUTPUT),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    weight_path = verify_checkpoint(
        checkpoint,
        selected,
        frozen_checkpoint["files"],
    )
    preflight_checkpoint(
        checkpoint,
        freeze_spec["inference"]["device"],
    )
    # Hash again after model preflight and immediately before claiming the attempt.
    verify_checkpoint(checkpoint, selected, frozen_checkpoint["files"])
    if completed_path.exists():
        raise SystemExit("private holdout completion already exists; refusing to rescore")

    scorer_sha = sha256_file(Path(__file__).resolve())
    claim = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "CLAIMED_BEFORE_HOLDOUT_LOAD",
        "claimed_at": utc_now(),
        "claim_id": uuid.uuid4().hex,
        "freeze_commit": freeze_commit,
        "upstream": upstream,
        "scorer_sha256": scorer_sha,
        "freeze_spec_sha256": sha256_bytes(freeze_bytes),
        "selection_report_sha256": sha256_bytes(selection_bytes),
        "selected_checkpoint_sha256": selected["weight_sha256"],
        "holdout_commitment_sha256": frozen["holdout_commitment_sha256"],
        "rerun_policy": "the claim permanently consumes this protocol attempt, including after failure",
        "publishing_boundary": "private ledger; contains no text, ids, labels, or row predictions",
    }
    try:
        durable_exclusive_json(claim_path, claim)
    except FileExistsError as error:
        raise SystemExit(
            f"one-shot holdout claim already exists; refusing to rescore: {claim_path}"
        ) from error
    durable_atomic_json(local_receipt, claim)

    try:
        # This is intentionally the first licensed train-file content read.
        train_path = (
            args.dataset_root.resolve()
            / "PsySUICIDE"
            / "repo"
            / "train.json"
        )
        raw_train = train_path.read_bytes()
        holdout, manifest = validate_partition(
            raw_train,
            json.loads(raw_train),
            frozen,
        )
        verify_checkpoint(checkpoint, selected, frozen_checkpoint["files"])
        metrics, device, runtime_seconds = evaluate_checkpoint(
            checkpoint=checkpoint,
            holdout=holdout,
            max_length=freeze_spec["inference"]["max_length"],
            eval_batch=freeze_spec["inference"]["eval_batch"],
            expected_device=freeze_spec["inference"]["device"],
        )
        verify_checkpoint(checkpoint, selected, frozen_checkpoint["files"])
        completed_at = utc_now()
        public_result = {
            "schema_version": 1,
            "scope": "one frozen PsySUICIDE train holdout confirmation",
            "status": "COMPLETED_FROM_EXCLUSIVE_CLAIM",
            "protocol_id": PROTOCOL_ID,
            "freeze_commit": freeze_commit,
            "upstream": upstream,
            "scored_at": completed_at,
            "claim_evidence": {
                "claim_id": claim["claim_id"],
                "claimed_at": claim["claimed_at"],
                "completed_from_exclusive_claim": True,
                "global_exactly_once_proven": False,
                "scope_note": "evidence applies to this scorer ledger namespace; local Git cannot prove no out-of-band data access",
            },
            "artifacts": freeze_spec["artifacts"],
            "selected_checkpoint": {
                "seed": selected["seed"],
                "weight_filename": weight_path.name,
                "weight_sha256": selected["weight_sha256"],
                "checkpoint_manifest": frozen_checkpoint["files"],
                "weights_published": False,
            },
            "partition": {
                "holdout_rows": manifest["holdout_rows"],
                "holdout_commitment_sha256": manifest[
                    "holdout_commitment_sha256"
                ],
            },
            "configuration": {
                "max_length": freeze_spec["inference"]["max_length"],
                "eval_batch": freeze_spec["inference"]["eval_batch"],
                "device": device,
                "dtype": freeze_spec["inference"]["dtype"],
                "predict_call_count": 1,
            },
            "runtime_seconds": runtime_seconds,
            "metrics": metrics,
            "inference_policy": {
                "status": "one-time internal confirmation only",
                "paper_test_substitute": False,
                "statistical_claims": "descriptive only; no paired test was preregistered",
            },
            "publishing_boundary": (
                "aggregate metrics and hashes only; licensed text, ids, labels, "
                "row predictions, logits, weights, and trainer state remain private"
            ),
        }
        completion = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "SCORE_COMPLETED",
            "completed_at": completed_at,
            "claim_id": claim["claim_id"],
            "freeze_commit": freeze_commit,
            "public_result": public_result,
        }
        durable_exclusive_json(completed_path, completion)
        durable_atomic_json(DEFAULT_OUTPUT, public_result, mode=0o644)
        result_sha = sha256_file(DEFAULT_OUTPUT)
        durable_atomic_json(
            local_receipt,
            {
                **claim,
                "status": "COMPLETED_FROM_EXCLUSIVE_CLAIM",
                "completed_at": completed_at,
                "public_result": str(DEFAULT_OUTPUT.relative_to(ROOT)),
                "public_result_sha256": result_sha,
                "holdout_rows_scored": manifest["holdout_rows"],
                "device": device,
                "runtime_seconds": runtime_seconds,
                "predict_call_count": 1,
            },
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETED_FROM_EXCLUSIVE_CLAIM",
                    "output": str(DEFAULT_OUTPUT),
                    "output_sha256": result_sha,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as error:
        durable_atomic_json(
            local_receipt,
            {
                **claim,
                "status": "FAILED_AFTER_ONE_SHOT_CLAIM",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "rerun_policy": "manual audit required; automatic rescore forbidden",
            },
        )
        raise


if __name__ == "__main__":
    main()
