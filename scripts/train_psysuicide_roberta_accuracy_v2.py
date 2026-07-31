#!/usr/bin/env python3
"""PsySUICIDE accuracy-first v2 trainer with a strict optimization-only boundary.

This entry point is dry-run by default.  It has no dataset-root option and
cannot reconstruct the optimization partition from the licensed full train
file.  Metric-producing execution requires a separately frozen, committed and
pushed preregistration plus a repository-external permission-0600 file
containing exactly the previously committed 9,342 optimization rows.

Official valid, official test and the consumed internal holdout are never
accepted by this program.  Checkpoints, row digests, logits and diagnostics
remain under ignored results/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import stat
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
RESULTS_ROOT = ROOT / "results" / "psysuicide-roberta-v2-accuracy-first"
DEFAULT_DRAFT = (
    ROOT / "reports" / "psysuicide-roberta-v2-accuracy-first.prereg.draft.json"
)
ALLOWED_PREREGISTRATIONS = {
    "reports/psysuicide-roberta-v2-accuracy-first.prereg.draft.json",
    "reports/psysuicide-roberta-v2-accuracy-first.prereg.json",
}
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
SPLIT_SEED = "psysuicide-accuracy-first-v2-inner-dev-2026-07-29"
OPTIMIZATION_ROWS = 9_342
TRAIN_ROWS = 7_479
INNER_DEV_ROWS = 1_863
OPTIMIZATION_COMMITMENT = (
    "0c3cbf98db02c609d500b4f9ef9bf95518b617bd4dffd1f4cde8a5c5e85a8dd9"
)
PROTOCOL_CONTRACT_SHA256 = (
    "7ee7688f630eca9339ea02be1d374784a434ff8a348740eb12811ebcb6dda4fa"
)
LABELS = (
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
)
EXPECTED_COUNTS = dict(
    zip(LABELS, (6883, 846, 844, 234, 93, 12, 65, 18, 62, 173, 112))
)
EXPECTED_DEV_COUNTS = dict(
    zip(LABELS, (1376, 169, 168, 46, 18, 2, 13, 3, 12, 34, 22))
)
CRITICAL_LABELS = (
    "主动自杀意图",
    "自杀计划",
    "自杀准备行为",
    "自杀未遂",
    "自伤意图",
    "自伤行为",
)
ARMS = {
    "A": {"weighted_sampler": False, "weighted_focal": False},
    "B": {"weighted_sampler": True, "weighted_focal": False},
    "C": {"weighted_sampler": False, "weighted_focal": True},
    "D": {"weighted_sampler": True, "weighted_focal": True},
}
FORBIDDEN_EXACT_NAMES = {
    "train.json",
    "train.jsonl",
    "valid.json",
    "valid.jsonl",
    "test.json",
    "test.jsonl",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row(row: dict[str, Any]) -> str:
    if set(row) != {"idx", "labels", "text"}:
        raise ValueError("rows must contain exactly idx, labels and text")
    labels = row.get("labels")
    if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in LABELS:
        raise ValueError("every row must contain exactly one declared label")
    if "idx" not in row or "text" not in row:
        raise ValueError("every row must contain idx, labels and text")
    return json.dumps(
        {"idx": str(row["idx"]), "labels": labels, "text": str(row["text"])},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def digest_commitment(digests: Iterable[str]) -> str:
    return sha256_bytes("\n".join(sorted(digests)).encode("utf-8"))


def protocol_contract_payload(prereg: dict[str, Any]) -> dict[str, Any]:
    implementation = prereg["implementation"]
    inner_split = {
        key: value
        for key, value in prereg["inner_split"].items()
        if key
        not in {
            "exact_train_commitment_sha256",
            "exact_inner_dev_commitment_sha256",
            "exact_inner_dev_per_label_commitment_sha256",
        }
    }
    return {
        "experiment_id": prereg["experiment_id"],
        "objective": prereg["objective"],
        "scope": prereg["scope"],
        "implementation": {
            key: implementation[key]
            for key in (
                "trainer_script",
                "analyzer_script",
                "base_model",
                "base_model_revision",
                "base_model_manifest",
                "base_model_manifest_sha256",
            )
        },
        "source_contract": prereg["source_contract"],
        "label_order": prereg["label_order"],
        "inner_split": inner_split,
        "fixed_training_protocol": prereg["fixed_training_protocol"],
        "arms": prereg["arms"],
        "phases": prereg["phases"],
        "metric_definitions": prereg["metric_definitions"],
        "decision_rules": prereg["decision_rules"],
        "paired_hierarchical_bootstrap": prereg["paired_hierarchical_bootstrap"],
        "privacy_and_publication": prereg["privacy_and_publication"],
        "later_work_boundary": prereg["later_work_boundary"],
    }


def protocol_contract_sha256(prereg: dict[str, Any]) -> str:
    canonical = json.dumps(
        protocol_contract_payload(prereg),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_bytes(canonical.encode("utf-8"))


def verify_protocol_contract(prereg: dict[str, Any]) -> None:
    observed = protocol_contract_sha256(prereg)
    if observed != PROTOCOL_CONTRACT_SHA256:
        raise ValueError(
            f"frozen protocol contract mismatch: {observed} != "
            f"{PROTOCOL_CONTRACT_SHA256}"
        )


def reject_forbidden_source_path(path: Path) -> Path:
    """Resolve and validate the private optimization-only source path."""
    if not path.is_absolute():
        raise ValueError("--optimization-file must be an absolute path")
    if path.is_symlink():
        raise ValueError("optimization source may not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("optimization source must stay outside the repository")
    lowered_parts = {part.lower() for part in resolved.parts}
    if resolved.name.lower() in FORBIDDEN_EXACT_NAMES:
        raise ValueError("full train, valid and test filenames are forbidden")
    path_tokens = {
        token
        for part in lowered_parts
        for token in re.split(r"[^a-z0-9]+", part)
        if token
    }
    if (
        path_tokens & {"valid", "validation", "test", "holdout"}
        or {"full", "train"} <= path_tokens
        or any("holdout" in part for part in lowered_parts)
    ):
        raise ValueError("valid, test and holdout paths are forbidden")
    return resolved


def resolve_preregistration_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else ROOT / path
    resolved = candidate.resolve(strict=True)
    if ROOT not in resolved.parents:
        raise ValueError("preregistration must stay inside this repository")
    if "prereg" not in resolved.name.lower() or resolved.suffix != ".json":
        raise ValueError("preregistration must be an explicit prereg JSON file")
    relative = str(resolved.relative_to(ROOT))
    if relative not in ALLOWED_PREREGISTRATIONS:
        raise ValueError("only the accuracy-first v2 preregistration is accepted")
    git_text("ls-files", "--error-unmatch", relative)
    return resolved


def read_private_bytes_0600(path: Path) -> bytes:
    resolved = reject_forbidden_source_path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("optimization source must be a regular file")
        if stat.S_IMODE(status.st_mode) != 0o600:
            raise ValueError("optimization source permissions must be exactly 0600")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def parse_rows(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("optimization source must be UTF-8") from error
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("optimization source is empty")
    if stripped.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON optimization source must be an array")
        return parsed
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        rows.append(row)
    return rows


def validate_optimization_rows(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int = OPTIMIZATION_ROWS,
    expected_counts: dict[str, int] = EXPECTED_COUNTS,
    expected_commitment: str = OPTIMIZATION_COMMITMENT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} optimization rows, got {len(rows)}")
    entries = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        canonical = canonical_row(row)
        row_digest = sha256_bytes(canonical.encode("utf-8"))
        if row_digest in seen:
            raise ValueError("duplicate canonical optimization row")
        seen.add(row_digest)
        label = row["labels"][0]
        counts[label] += 1
        normalized_row = {
            "idx": str(row["idx"]),
            "labels": [label],
            "text": str(row["text"]),
        }
        entries.append(
            {"row": normalized_row, "label": label, "row_digest": row_digest}
        )
    if dict(counts) != expected_counts:
        raise ValueError(f"optimization per-label counts mismatch: {dict(counts)}")
    observed_commitment = digest_commitment(seen)
    if observed_commitment != expected_commitment:
        raise ValueError("optimization commitment mismatch")
    manifest = {
        "optimization_rows": len(entries),
        "optimization_commitment_sha256": observed_commitment,
        "per_label": {label: counts[label] for label in LABELS},
        "unique_canonical_rows": len(seen),
    }
    return entries, manifest


def load_optimization_only(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return validate_optimization_rows(parse_rows(read_private_bytes_0600(path)))


def stratified_inner_split(
    entries: list[dict[str, Any]],
    *,
    split_seed: str = SPLIT_SEED,
    expected_dev_counts: dict[str, int] = EXPECTED_DEV_COUNTS,
    expected_train_rows: int = TRAIN_ROWS,
    expected_dev_rows: int = INNER_DEV_ROWS,
    expected_union_commitment: str = OPTIMIZATION_COMMITMENT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train: list[dict[str, Any]] = []
    inner_dev: list[dict[str, Any]] = []
    per_label: dict[str, dict[str, int]] = {}
    inner_dev_per_label_commitments: dict[str, str] = {}
    for label in LABELS:
        group = [entry for entry in entries if entry["label"] == label]
        ranked = sorted(
            group,
            key=lambda entry: (
                sha256_bytes(
                    f"{split_seed}\0{entry['row_digest']}".encode("utf-8")
                ),
                entry["row_digest"],
            ),
        )
        dev_count = max(1, math.floor(len(ranked) * 0.2))
        if expected_dev_counts.get(label) != dev_count:
            raise ValueError(f"unexpected inner-dev count for {label}: {dev_count}")
        inner_dev.extend(ranked[:dev_count])
        train.extend(ranked[dev_count:])
        per_label[label] = {
            "optimization": len(ranked),
            "train": len(ranked) - dev_count,
            "inner_dev": dev_count,
        }
        inner_dev_per_label_commitments[label] = digest_commitment(
            entry["row_digest"] for entry in ranked[:dev_count]
        )
    train_digests = {entry["row_digest"] for entry in train}
    dev_digests = {entry["row_digest"] for entry in inner_dev}
    if train_digests & dev_digests:
        raise ValueError("inner-train and inner-dev overlap")
    if len(train) != expected_train_rows or len(inner_dev) != expected_dev_rows:
        raise ValueError("unexpected inner split size")
    union_commitment = digest_commitment(train_digests | dev_digests)
    if union_commitment != expected_union_commitment:
        raise ValueError("inner split union does not match optimization commitment")
    manifest = {
        "schema_version": 1,
        "split_id": "psysuicide-optimization-inner-dev-v2",
        "split_seed": split_seed,
        "optimization_rows": len(entries),
        "train_rows": len(train),
        "inner_dev_rows": len(inner_dev),
        "optimization_commitment_sha256": union_commitment,
        "train_commitment_sha256": digest_commitment(train_digests),
        "inner_dev_commitment_sha256": digest_commitment(dev_digests),
        "inner_dev_per_label_commitment_sha256": inner_dev_per_label_commitments,
        "overlap_rows": 0,
        "per_label": per_label,
        "publishing_boundary": (
            "aggregate counts and commitments only; membership, row digests, "
            "text and source ids remain private"
        ),
    }
    return train, inner_dev, manifest


def atomic_write_json(path: Path, payload: dict[str, Any], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_checkpoint_completion(checkpoint_dir: Path, global_step: int) -> dict[str, Any]:
    required = {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    }
    present = {path.name for path in checkpoint_dir.iterdir() if path.is_file()}
    if not required <= present:
        raise ValueError(
            f"checkpoint is missing resume state: {sorted(required - present)}"
        )
    model_files = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or path.name.startswith("pytorch_model")
            and path.name.endswith(".bin")
        )
    )
    if not model_files:
        raise ValueError("checkpoint is missing model weights")
    files = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file() and path.name != "checkpoint-complete.json"
    )
    return {
        "schema_version": 1,
        "global_step": int(global_step),
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }


def checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    marker = checkpoint_dir / "checkpoint-complete.json"
    try:
        if checkpoint_dir.is_symlink() or marker.is_symlink():
            return False
        payload = json.loads(marker.read_text(encoding="utf-8"))
        match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", checkpoint_dir.name)
        if match is None:
            return False
        step = int(match.group(1))
        if (
            set(payload) != {"schema_version", "global_step", "files"}
            or not isinstance(payload.get("schema_version"), int)
            or isinstance(payload.get("schema_version"), bool)
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("global_step"), int)
            or isinstance(payload.get("global_step"), bool)
            or payload.get("global_step") != step
        ):
            return False
        files = payload.get("files")
        if not isinstance(files, dict) or not files:
            return False
        actual_children = list(checkpoint_dir.iterdir())
        if any(not child.is_file() or child.is_symlink() for child in actual_children):
            return False
        actual_files = {
            child.name for child in actual_children if child.name != marker.name
        }
        if set(files) != actual_files:
            return False
        required = {
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "training_args.bin",
        }
        if not required <= actual_files:
            return False
        if not any(
            filename.endswith(".safetensors")
            or (
                filename.startswith("pytorch_model")
                and filename.endswith(".bin")
            )
            for filename in actual_files
        ):
            return False
        for filename, expected in files.items():
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(expected, dict)
                or set(expected) != {"bytes", "sha256"}
                or not isinstance(expected.get("bytes"), int)
                or isinstance(expected.get("bytes"), bool)
                or expected["bytes"] < 0
                or not isinstance(expected.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None
            ):
                return False
            candidate = checkpoint_dir / filename
            if (
                not candidate.is_file()
                or candidate.is_symlink()
                or candidate.stat().st_size != expected["bytes"]
                or sha256_file(candidate) != expected["sha256"]
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def latest_complete_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if checkpoint_is_complete(path):
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return path.name


def installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def git_text(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def require_git_ignored(*paths: Path) -> None:
    for path in paths:
        resolved = path.resolve(strict=False)
        try:
            relative = str(resolved.relative_to(ROOT.resolve()))
        except ValueError as error:
            raise ValueError("result paths must stay inside the repository") from error
        if subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", "--", relative],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise ValueError(f"result path is not ignored by Git: {relative}")
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", relative],
            cwd=ROOT,
            text=True,
        ).strip()
        if tracked:
            raise ValueError(f"private result path is already tracked: {relative}")


def resolve_run_campaign(run_id: str) -> Path:
    if SAFE_RUN_ID.fullmatch(run_id) is None or run_id in {".", ".."}:
        raise ValueError(
            "--run-id must be 1-80 safe characters and begin with a letter or digit"
        )
    if RESULTS_ROOT.exists() and (
        RESULTS_ROOT.is_symlink() or not RESULTS_ROOT.is_dir()
    ):
        raise ValueError("results root must be a real directory")
    results_root = RESULTS_ROOT.resolve(strict=False)
    campaign = (results_root / run_id).resolve(strict=False)
    if campaign.parent != results_root:
        raise ValueError("run campaign escaped the fixed results root")
    require_git_ignored(ROOT / "results", RESULTS_ROOT, campaign)
    return campaign


def verify_base_model_dir(model_dir: Path, prereg: dict[str, Any]) -> dict[str, str]:
    """Verify that Transformers can only load the pinned general base bytes."""
    resolved = model_dir.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("--model-dir must be a local directory")
    implementation = prereg["implementation"]
    manifest_relative = implementation.get("base_model_manifest")
    if not isinstance(manifest_relative, str):
        raise ValueError("preregistration is missing the base-model manifest")
    manifest_path = (ROOT / manifest_relative).resolve(strict=True)
    if ROOT not in manifest_path.parents:
        raise ValueError("base-model manifest must stay inside the repository")
    expected_manifest_sha = implementation.get("base_model_manifest_sha256")
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("base-model manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_model") != implementation.get("base_model"):
        raise ValueError("base-model identity mismatch")
    if manifest.get("base_model_revision") != implementation.get(
        "base_model_revision"
    ):
        raise ValueError("base-model revision mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("base-model manifest has no files")
    for filename, expected in files.items():
        candidate = resolved / filename
        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"missing pinned base-model file: {filename}")
        if candidate.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"base-model byte-count mismatch: {filename}")
        if sha256_file(candidate) != expected["sha256"]:
            raise ValueError(f"base-model SHA-256 mismatch: {filename}")
    declared = set(files)
    loadable_weight_files = {
        path.name
        for path in resolved.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".bin")
            or path.name.endswith(".index.json")
        )
    }
    unexpected_weights = loadable_weight_files - declared
    if unexpected_weights:
        raise ValueError(
            f"undeclared loadable model weights are forbidden: {sorted(unexpected_weights)}"
        )
    return {
        "base_model": manifest["base_model"],
        "base_model_revision": manifest["base_model_revision"],
        "base_model_manifest_sha256": expected_manifest_sha,
    }


def verify_frozen_execution(
    prereg_path: Path,
    prereg: dict[str, Any],
    split_manifest: dict[str, Any],
) -> dict[str, str]:
    if prereg.get("status") != "FROZEN" or prereg.get("training_allowed") is not True:
        raise ValueError("training requires status=FROZEN and training_allowed=true")
    verify_protocol_contract(prereg)
    implementation = prereg["implementation"]
    trainer_sha = sha256_file(SCRIPT_PATH)
    if implementation.get("trainer_sha256") != trainer_sha:
        raise ValueError("trainer SHA-256 does not match frozen preregistration")
    inner = prereg["inner_split"]
    if inner.get("exact_train_commitment_sha256") != split_manifest[
        "train_commitment_sha256"
    ]:
        raise ValueError("inner-train commitment does not match preregistration")
    if inner.get("exact_inner_dev_commitment_sha256") != split_manifest[
        "inner_dev_commitment_sha256"
    ]:
        raise ValueError("inner-dev commitment does not match preregistration")
    per_label_commitments = inner.get(
        "exact_inner_dev_per_label_commitment_sha256"
    )
    if (
        not isinstance(per_label_commitments, dict)
        or tuple(per_label_commitments) != LABELS
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in per_label_commitments.values()
        )
        or per_label_commitments
        != split_manifest["inner_dev_per_label_commitment_sha256"]
    ):
        raise ValueError(
            "per-label inner-dev commitments do not match preregistration"
        )
    source = prereg["source_contract"]
    if source.get("required_optimization_commitment_sha256") != (
        split_manifest["optimization_commitment_sha256"]
    ):
        raise ValueError("optimization commitment does not match preregistration")
    prereg_resolved = prereg_path.resolve(strict=True)
    if ROOT not in prereg_resolved.parents:
        raise ValueError("frozen preregistration must be tracked in this repository")
    relative_prereg = str(prereg_resolved.relative_to(ROOT))
    relative_script = str(SCRIPT_PATH.relative_to(ROOT))
    analyzer_relative = implementation.get("analyzer_script")
    if not isinstance(analyzer_relative, str):
        raise ValueError("missing frozen analyzer path")
    analyzer_path = (ROOT / analyzer_relative).resolve(strict=True)
    if ROOT not in analyzer_path.parents:
        raise ValueError("analyzer must stay inside the repository")
    if sha256_file(analyzer_path) != implementation.get("analyzer_sha256"):
        raise ValueError("analyzer SHA-256 does not match frozen preregistration")
    git_text("ls-files", "--error-unmatch", relative_prereg)
    git_text("ls-files", "--error-unmatch", relative_script)
    git_text("ls-files", "--error-unmatch", analyzer_relative)
    if subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            relative_prereg,
            relative_script,
            analyzer_relative,
        ],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("trainer and preregistration must be clean at HEAD")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("all tracked files must be clean before execution")
    uncommitted = git_text("status", "--porcelain", "--untracked-files=all")
    if uncommitted:
        raise ValueError("execution requires a completely clean non-ignored worktree")
    head = git_text("rev-parse", "HEAD")
    upstream = git_text("rev-parse", "@{upstream}")
    if head != upstream:
        raise ValueError("local HEAD must equal the pushed upstream SHA")
    branch = git_text("branch", "--show-current")
    remote_line = git_text("ls-remote", "--heads", "origin", f"refs/heads/{branch}")
    remote_head = remote_line.split()[0] if remote_line else ""
    if remote_head != head:
        raise ValueError("live remote branch SHA must equal local HEAD")
    frozen_commit = implementation.get("execution_base_commit")
    if not isinstance(frozen_commit, str) or len(frozen_commit) != 40:
        raise ValueError("missing frozen protocol-and-code base commit")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_commit, head],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("frozen protocol-and-code base commit must be an ancestor")
    frozen_script = subprocess.check_output(
        ["git", "show", f"{frozen_commit}:{relative_script}"], cwd=ROOT
    )
    if sha256_bytes(frozen_script) != trainer_sha:
        raise ValueError("trainer bytes differ from the frozen execution base commit")
    frozen_analyzer = subprocess.check_output(
        ["git", "show", f"{frozen_commit}:{analyzer_relative}"], cwd=ROOT
    )
    if sha256_bytes(frozen_analyzer) != implementation["analyzer_sha256"]:
        raise ValueError("analyzer bytes differ from the frozen execution base commit")
    return {
        "head": head,
        "upstream": upstream,
        "live_remote_head": remote_head,
        "execution_base_commit": frozen_commit,
        "trainer_sha256": trainer_sha,
        "analyzer_sha256": implementation["analyzer_sha256"],
        "prereg_sha256": sha256_file(prereg_resolved),
    }


def validate_phase_identity(
    stage: str,
    arm: str,
    seed: int,
    selection_artifact: Path | None,
    prereg: dict[str, Any],
    prereg_sha: str,
    run_id: str,
) -> None:
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if stage == "smoke":
        if seed != 42:
            raise ValueError("smoke requires Seed 42")
        return
    if stage == "screen":
        if seed != 42:
            raise ValueError("screen requires Seed 42")
        return
    if stage != "confirm":
        raise ValueError("unknown metric-producing stage")
    if seed not in (43, 44, 45):
        raise ValueError("confirmation requires a fresh Seed 43, 44 or 45")
    if selection_artifact is None:
        raise ValueError("confirmation requires --selection-artifact")
    selection_path = selection_artifact.resolve(strict=True)
    if ROOT not in selection_path.parents:
        raise ValueError("selection artifact must be tracked in this repository")
    relative_selection = str(selection_path.relative_to(ROOT))
    git_text("ls-files", "--error-unmatch", relative_selection)
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative_selection],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("selection artifact must be clean at HEAD")
    analyzer_path = (
        ROOT / prereg["implementation"]["analyzer_script"]
    ).resolve(strict=True)
    specification = importlib.util.spec_from_file_location(
        "psysuicide_accuracy_v2_analyzer", analyzer_path
    )
    if specification is None or specification.loader is None:
        raise ValueError("unable to load the frozen selection analyzer")
    analyzer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(analyzer)
    selection = analyzer.strict_json_load(
        selection_path, "screen selection artifact"
    )
    if (
        selection.get("status") != "FROZEN"
        or selection.get("confirmation_allowed") is not True
    ):
        raise ValueError("selection artifact has not frozen confirmation")
    if selection.get("prereg_sha256") != prereg_sha:
        raise ValueError("selection artifact preregistration identity mismatch")
    if selection.get("screen_seed") != 42:
        raise ValueError("selection artifact screen seed mismatch")
    campaign_dir = resolve_run_campaign(run_id)
    recomputed = analyzer.analyze_screen(
        campaign_dir,
        prereg,
        prereg_sha,
    )
    if selection != recomputed:
        raise ValueError(
            "selection artifact does not match four frozen Seed-42 aggregates"
        )
    candidate = selection.get("selected_challenger")
    if candidate not in ("A", "B", "C"):
        raise ValueError("selection artifact has no eligible challenger")
    if arm not in (candidate, "D"):
        raise ValueError("confirmation permits only the selected challenger and D")


def deterministic_subset(
    entries: list[dict[str, Any]], *, seed: int, total: int, minimum: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_digests: set[str] = set()
    for label in LABELS:
        group = sorted(
            (entry for entry in entries if entry["label"] == label),
            key=lambda entry: sha256_bytes(
                f"{seed}\0{entry['row_digest']}".encode("utf-8")
            ),
        )
        if len(group) < minimum:
            raise ValueError(f"not enough rows for smoke class {label}")
        for entry in group[:minimum]:
            selected.append(entry)
            selected_digests.add(entry["row_digest"])
    remainder = sorted(
        (entry for entry in entries if entry["row_digest"] not in selected_digests),
        key=lambda entry: sha256_bytes(
            f"{seed}\0remainder\0{entry['row_digest']}".encode("utf-8")
        ),
    )
    selected.extend(remainder[: total - len(selected)])
    if len(selected) != total:
        raise ValueError("unable to construct smoke subset")
    return selected


def class_weight_values(label_ids: list[int]) -> tuple[list[float], list[float]]:
    import numpy as np

    counts = Counter(label_ids)
    if set(counts) != set(range(len(LABELS))):
        raise ValueError("training split must contain every label")
    values = np.array(
        [
            (len(label_ids) / (len(LABELS) * counts[index])) ** 0.25
            for index in range(len(LABELS))
        ],
        dtype=np.float32,
    )
    values = np.clip(values / values.mean(), 0.35, 4.0)
    class_values = [float(value) for value in values]
    sample_values = [class_values[label_id] for label_id in label_ids]
    return class_values, sample_values


def weighted_sample_indices(
    weights: list[float], *, base_seed: int, epoch: int
) -> list[int]:
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(base_seed) + int(epoch))
    return torch.multinomial(
        torch.as_tensor(weights, dtype=torch.double),
        len(weights),
        replacement=True,
        generator=generator,
    ).tolist()


def compute_arm_loss(
    logits: Any,
    labels: Any,
    class_values: list[float],
    *,
    arm: str,
    focal_gamma: float,
) -> Any:
    import torch
    import torch.nn.functional as functional

    if arm not in ARMS:
        raise ValueError("unknown loss arm")
    if not ARMS[arm]["weighted_focal"]:
        return functional.cross_entropy(logits, labels)
    weights = torch.as_tensor(
        class_values, dtype=torch.float32, device=logits.device
    )
    plain = functional.cross_entropy(logits, labels, reduction="none")
    weighted = functional.cross_entropy(
        logits, labels, weight=weights, reduction="none"
    )
    modulation = (1.0 - torch.exp(-plain)).pow(float(focal_gamma))
    return (modulation * weighted).mean()


def metrics_and_calibration(logits: Any, gold: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    logits = np.asarray(logits, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    critical_ids = {LABELS.index(label) for label in CRITICAL_LABELS}
    critical_mask = np.array([int(value) in critical_ids for value in gold])
    pooled_recall = float(
        (predicted[critical_mask] == gold[critical_mask]).mean()
    )
    metrics = {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(gold, predicted, average="weighted", zero_division=0)
        ),
        "pooled_critical_risk_recall": pooled_recall,
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
    confidence = probabilities.max(axis=1)
    correctness = (predicted == gold).astype(np.float64)
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 16)[:-1]:
        upper = lower + 1.0 / 15.0
        mask = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            mask = (confidence >= lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    clipped = np.clip(probabilities[np.arange(len(gold)), gold], 1e-12, 1.0)
    one_hot = np.eye(len(LABELS), dtype=np.float64)[gold]
    diagnostics = {
        "nll": float(-np.log(clipped).mean()),
        "multiclass_brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_equal_width": float(ece),
        "confusion_matrix": confusion_matrix(
            gold, predicted, labels=list(range(len(LABELS)))
        ).astype(np.int64),
        "probabilities": probabilities.astype(np.float32),
        "predicted": predicted.astype(np.int16),
    }
    return metrics, diagnostics


def write_private_diagnostics(
    path: Path,
    *,
    logits: Any,
    gold: Any,
    row_digests: list[str],
    diagnostics: dict[str, Any],
    allow_replace: bool = False,
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not allow_replace:
        raise ValueError(f"refusing to overwrite diagnostics: {path}")
    logits_array = np.asarray(logits, dtype=np.float32)
    probabilities = np.asarray(diagnostics["probabilities"], dtype=np.float32)
    gold_array = np.asarray(gold, dtype=np.int16)
    predicted = np.asarray(diagnostics["predicted"], dtype=np.int16)
    confusion = np.asarray(diagnostics["confusion_matrix"], dtype=np.int64)
    row_digest_array = np.asarray(row_digests)
    row_count = len(gold_array)
    if logits_array.shape != (row_count, len(LABELS)):
        raise ValueError("diagnostic logits shape mismatch")
    if probabilities.shape != (row_count, len(LABELS)):
        raise ValueError("diagnostic probability shape mismatch")
    if predicted.shape != (row_count,) or row_digest_array.shape != (row_count,):
        raise ValueError("diagnostic row arrays must have identical lengths")
    if confusion.shape != (len(LABELS), len(LABELS)):
        raise ValueError("diagnostic confusion shape mismatch")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in row_digests):
        raise ValueError("diagnostic row digests must be lowercase SHA-256")
    if not np.isfinite(logits_array).all() or not np.isfinite(probabilities).all():
        raise ValueError("diagnostic arrays must be finite")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("diagnostic probabilities must sum to one")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            np.savez_compressed(
                handle,
                logits=logits_array,
                probabilities=probabilities,
                gold=gold_array,
                predicted=predicted,
                row_digests=row_digest_array,
                confusion_matrix=confusion,
                nll=np.asarray([diagnostics["nll"]], dtype=np.float64),
                multiclass_brier=np.asarray(
                    [diagnostics["multiclass_brier"]], dtype=np.float64
                ),
                ece_15_equal_width=np.asarray(
                    [diagnostics["ece_15_equal_width"]], dtype=np.float64
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def training_argument_values(
    prereg: dict[str, Any],
    *,
    stage: str,
    seed: int,
    cpu: bool,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = prereg["fixed_training_protocol"]
    smoke = prereg["phases"]["code_smoke"]
    return {
        "output_dir": str(output_dir),
        "do_train": True,
        "do_eval": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 1 if stage == "smoke" else 50,
        "per_device_train_batch_size": int(protocol["per_device_train_batch"]),
        "per_device_eval_batch_size": int(protocol["per_device_eval_batch"]),
        "gradient_accumulation_steps": int(protocol["gradient_accumulation"]),
        "learning_rate": float(protocol["learning_rate"]),
        "weight_decay": float(protocol["weight_decay"]),
        "num_train_epochs": (
            float(smoke["epochs"]) if stage == "smoke" else float(protocol["epochs"])
        ),
        "max_steps": int(smoke["max_steps"]) if stage == "smoke" else -1,
        "warmup_ratio": float(protocol["warmup_ratio"]),
        "lr_scheduler_type": str(protocol["scheduler"]),
        "max_grad_norm": float(protocol["max_grad_norm"]),
        "seed": seed,
        "data_seed": seed,
        "use_cpu": cpu,
        "fp16": False,
        "bf16": False,
        "dataloader_num_workers": int(protocol["dataloader_num_workers"]),
        "dataloader_pin_memory": bool(protocol["dataloader_pin_memory"]),
        "gradient_checkpointing": bool(protocol["gradient_checkpointing"]),
        "optim": str(protocol["optimizer"]),
        "report_to": "none",
        "save_total_limit": int(protocol["save_total_limit"]),
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "full_determinism": bool(protocol["full_determinism"]),
    }


def execute_training(
    args: argparse.Namespace,
    prereg: dict[str, Any],
    train_entries: list[dict[str, Any]],
    dev_entries: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    git_identity: dict[str, str],
) -> None:
    import numpy as np
    import torch
    from torch.utils.data import Dataset, Sampler
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    class TextDataset(Dataset):
        def __init__(self, entries: list[dict[str, Any]], tokenizer: Any, max_length: int):
            self.entries = entries
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.label_ids = [LABELS.index(entry["label"]) for entry in entries]

        def __len__(self) -> int:
            return len(self.entries)

        def __getitem__(self, index: int) -> dict[str, Any]:
            encoded = self.tokenizer(
                str(self.entries[index]["row"]["text"]),
                truncation=True,
                max_length=self.max_length,
            )
            encoded["labels"] = self.label_ids[index]
            return encoded

    class EpochWeightedSampler(Sampler[int]):
        def __init__(self, weights: list[float], base_seed: int):
            self.weights = torch.as_tensor(weights, dtype=torch.double)
            self.base_seed = base_seed
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def __iter__(self):
            return iter(
                weighted_sample_indices(
                    self.weights.tolist(),
                    base_seed=self.base_seed,
                    epoch=self.epoch,
                )
            )

        def __len__(self) -> int:
            return len(self.weights)

    arm_config = ARMS[args.arm]
    protocol = prereg["fixed_training_protocol"]
    if args.stage == "smoke":
        smoke = prereg["phases"]["code_smoke"]
        train_entries = deterministic_subset(
            train_entries,
            seed=args.seed,
            total=int(smoke["train_rows"]),
            minimum=int(smoke["minimum_per_train_label"]),
        )
        dev_entries = deterministic_subset(
            dev_entries,
            seed=args.seed,
            total=int(smoke["inner_dev_rows"]),
            minimum=int(smoke["minimum_per_inner_dev_label"]),
        )
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_dir = args.model_dir.resolve(strict=True)
    base_model_identity = verify_base_model_dir(model_dir, prereg)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    train_dataset = TextDataset(train_entries, tokenizer, int(protocol["max_length"]))
    dev_dataset = TextDataset(dev_entries, tokenizer, int(protocol["max_length"]))
    class_values, sample_values = class_weight_values(train_dataset.label_ids)
    expected_device = (
        "cpu"
        if args.cpu
        else "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    environment_identity = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_forced": bool(args.cpu),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "expected_device": expected_device,
        "packages": {
            "torch": installed_version("torch"),
            "transformers": installed_version("transformers"),
            "accelerate": installed_version("accelerate"),
            "numpy": installed_version("numpy"),
            "scikit-learn": installed_version("scikit-learn"),
        },
    }

    class ArmTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            if not arm_config["weighted_sampler"]:
                return super()._get_train_sampler(train_dataset)
            return EpochWeightedSampler(sample_values, args.seed)

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = compute_arm_loss(
                outputs.logits,
                labels,
                class_values,
                arm=args.arm,
                focal_gamma=float(protocol["focal_gamma"]),
            )
            return (loss, outputs) if return_outputs else loss

    class CheckpointCompletionCallback(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            checkpoint_dir = (
                Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            )
            completion = build_checkpoint_completion(
                checkpoint_dir, state.global_step
            )
            atomic_write_json(
                checkpoint_dir / "checkpoint-complete.json",
                completion,
                0o600,
            )
            return control

    def compute_metrics(prediction: Any) -> dict[str, float]:
        metrics, _ = metrics_and_calibration(prediction.predictions, prediction.label_ids)
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "pooled_critical_risk_recall": metrics["pooled_critical_risk_recall"],
        }

    campaign_dir = resolve_run_campaign(args.run_id)
    run_dir = (
        campaign_dir / f"{args.stage}-{args.arm}-seed-{args.seed}"
    ).resolve(strict=False)
    if run_dir.parent != campaign_dir:
        raise ValueError("run directory escaped its fixed campaign directory")
    require_git_ignored(run_dir)
    identity = {
        "schema_version": 1,
        "stage": args.stage,
        "arm": args.arm,
        "seed": args.seed,
        **git_identity,
        "optimization_commitment_sha256": split_manifest[
            "optimization_commitment_sha256"
        ],
        "train_commitment_sha256": split_manifest["train_commitment_sha256"],
        "inner_dev_commitment_sha256": split_manifest["inner_dev_commitment_sha256"],
        **base_model_identity,
        "environment": environment_identity,
    }
    resume_checkpoint = None
    if run_dir.exists():
        if args.resume != "latest":
            raise ValueError(f"refusing to overwrite existing run: {run_dir}")
        identity_path = run_dir / "run-identity.json"
        if not identity_path.exists() or json.loads(
            identity_path.read_text(encoding="utf-8")
        ) != identity:
            raise ValueError("resume identity mismatch")
        if (run_dir / "aggregate.json").exists():
            raise ValueError("completed runs cannot be resumed")
        resume_checkpoint = latest_complete_checkpoint(run_dir)
        if resume_checkpoint is None:
            raise ValueError("no complete trainer checkpoint is available to resume")
    else:
        if args.resume:
            raise ValueError("--resume requires an existing identical run")
        run_dir.mkdir(parents=True)
        os.chmod(run_dir, 0o700)
        atomic_write_json(run_dir / "run-identity.json", identity, 0o600)
    training_values = training_argument_values(
        prereg,
        stage=args.stage,
        seed=args.seed,
        cpu=args.cpu,
        output_dir=run_dir,
    )
    epochs = training_values["num_train_epochs"]
    max_steps = training_values["max_steps"]
    training_args = TrainingArguments(**training_values)
    trainer = ArmTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[CheckpointCompletionCallback()],
    )
    if str(trainer.args.device) != expected_device:
        raise ValueError(
            f"actual trainer device {trainer.args.device} != frozen {expected_device}"
        )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    prediction = trainer.predict(dev_dataset)
    metrics, diagnostics = metrics_and_calibration(
        prediction.predictions, prediction.label_ids
    )
    if not trainer.state.best_model_checkpoint:
        raise ValueError("trainer did not record a best checkpoint")
    best_checkpoint_path = Path(trainer.state.best_model_checkpoint)
    if not checkpoint_is_complete(best_checkpoint_path):
        raise ValueError("best checkpoint has no valid completion marker")
    completion_path = best_checkpoint_path / "checkpoint-complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    weight_files = {
        filename: metadata["sha256"]
        for filename, metadata in completion["files"].items()
        if filename.endswith(".safetensors")
        or filename.startswith("pytorch_model")
        and filename.endswith(".bin")
    }
    best_checkpoint_manifest = {
        "schema_version": 1,
        "checkpoint": best_checkpoint_path.name,
        "global_step": completion["global_step"],
        "selection_metric": "inner-dev accuracy",
        "best_metric": trainer.state.best_metric,
        "checkpoint_completion_sha256": sha256_file(completion_path),
        "model_weight_sha256s": weight_files,
        "files": completion["files"],
        "publishing_boundary": (
            "hashes and byte counts only; checkpoint bytes remain ignored"
        ),
    }
    best_manifest_path = run_dir / "best-checkpoint-manifest.json"
    atomic_write_json(best_manifest_path, best_checkpoint_manifest, 0o600)
    diagnostics_path = run_dir / "private-diagnostics.npz"
    write_private_diagnostics(
        diagnostics_path,
        logits=prediction.predictions,
        gold=prediction.label_ids,
        row_digests=[entry["row_digest"] for entry in dev_entries],
        diagnostics=diagnostics,
        allow_replace=bool(args.resume),
    )
    aggregate = {
        "schema_version": 1,
        "experiment_id": prereg["experiment_id"],
        "stage": args.stage,
        "arm": args.arm,
        "seed": args.seed,
        "metric_claims_allowed": False if args.stage == "smoke" else True,
        "train_rows": len(train_dataset),
        "inner_dev_rows": len(dev_dataset),
        "epochs": epochs,
        "max_steps": max_steps,
        "best_checkpoint": best_checkpoint_path.name,
        "best_inner_dev_accuracy": trainer.state.best_metric,
        "best_checkpoint_manifest_sha256": sha256_file(best_manifest_path),
        "best_checkpoint_completion_sha256": sha256_file(completion_path),
        "best_model_weight_sha256s": weight_files,
        "metrics": metrics,
        "train_runtime_seconds": float(
            train_result.metrics.get("train_runtime", 0.0)
        ),
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "identity": identity,
        "split": split_manifest,
        "arm_definition": arm_config,
        "publishing_boundary": (
            "aggregate-only candidate; weights, trainer state, row digests, "
            "logits, gold labels and confusion remain local and ignored"
        ),
    }
    atomic_write_json(run_dir / "aggregate.json", aggregate, 0o600)
    print(
        json.dumps(
            {
                "status": "completed",
                "stage": args.stage,
                "arm": args.arm,
                "seed": args.seed,
                "metrics": metrics,
                "aggregate": str((run_dir / "aggregate.json").relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"selftest failed: {message}")


def selftest() -> None:
    synthetic: list[dict[str, Any]] = []
    synthetic_counts = {label: 10 for label in LABELS}
    for label_index, label in enumerate(LABELS):
        for row_index in range(10):
            synthetic.append(
                {
                    "idx": f"{label_index}-{row_index}",
                    "labels": [label],
                    "text": f"synthetic-{label_index}-{row_index}",
                }
            )
    synthetic_commitment = digest_commitment(
        sha256_bytes(canonical_row(row).encode("utf-8")) for row in synthetic
    )
    entries, manifest = validate_optimization_rows(
        synthetic,
        expected_rows=110,
        expected_counts=synthetic_counts,
        expected_commitment=synthetic_commitment,
    )
    require(manifest["unique_canonical_rows"] == 110, "unique row count")
    synthetic_dev = {label: 2 for label in LABELS}
    train, dev, split = stratified_inner_split(
        entries,
        expected_dev_counts=synthetic_dev,
        expected_train_rows=88,
        expected_dev_rows=22,
        expected_union_commitment=synthetic_commitment,
    )
    require(len(train) == 88 and len(dev) == 22, "synthetic split size")
    require(
        not (
            {entry["row_digest"] for entry in train}
            & {entry["row_digest"] for entry in dev}
        ),
        "synthetic split overlap",
    )
    require(split["overlap_rows"] == 0, "reported split overlap")
    _, _, reordered_split = stratified_inner_split(
        list(reversed(entries)),
        expected_dev_counts=synthetic_dev,
        expected_train_rows=88,
        expected_dev_rows=22,
        expected_union_commitment=synthetic_commitment,
    )
    require(
        reordered_split["train_commitment_sha256"]
        == split["train_commitment_sha256"],
        "train commitment order invariance",
    )
    require(
        reordered_split["inner_dev_commitment_sha256"]
        == split["inner_dev_commitment_sha256"],
        "inner-dev commitment order invariance",
    )
    require(
        reordered_split["inner_dev_per_label_commitment_sha256"]
        == split["inner_dev_per_label_commitment_sha256"],
        "per-label inner-dev commitment order invariance",
    )
    mutated = [dict(row) for row in synthetic]
    mutated[0] = {**mutated[0], "text": "mutated"}
    try:
        validate_optimization_rows(
            mutated,
            expected_rows=110,
            expected_counts=synthetic_counts,
            expected_commitment=synthetic_commitment,
        )
    except ValueError as error:
        require("commitment mismatch" in str(error), "mutation rejection reason")
    else:
        raise AssertionError("mutation must fail the commitment gate")
    extra_field = [dict(row) for row in synthetic]
    extra_field[0] = {**extra_field[0], "source_id": "forbidden"}
    try:
        validate_optimization_rows(
            extra_field,
            expected_rows=110,
            expected_counts=synthetic_counts,
            expected_commitment=synthetic_commitment,
        )
    except ValueError as error:
        require("exactly idx, labels and text" in str(error), "extra-field rejection")
    else:
        raise RuntimeError("selftest failed: extra source fields must be rejected")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for part in ("holdout", "valid", "test", "full-train"):
            forbidden = root / part / "optimization-only.jsonl"
            forbidden.parent.mkdir()
            forbidden.write_text("{}\n", encoding="utf-8")
            os.chmod(forbidden, 0o600)
            try:
                reject_forbidden_source_path(forbidden)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{part} path must be rejected")
        private = root / "optimization-only.jsonl"
        private.write_text("{}\n", encoding="utf-8")
        os.chmod(private, 0o644)
        try:
            read_private_bytes_0600(private)
        except ValueError as error:
            require("0600" in str(error), "permission rejection reason")
        else:
            raise AssertionError("non-0600 input must be rejected")
        output = root / "private.json"
        atomic_write_json(output, {"ok": True}, 0o600)
        require(
            stat.S_IMODE(output.stat().st_mode) == 0o600,
            "private JSON permissions",
        )
        checkpoint = root / "checkpoint-10"
        checkpoint.mkdir()
        for filename in (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "training_args.bin",
            "model.safetensors",
        ):
            (checkpoint / filename).write_bytes(f"fixture-{filename}".encode())
        completion = build_checkpoint_completion(checkpoint, 10)
        atomic_write_json(
            checkpoint / "checkpoint-complete.json", completion, 0o600
        )
        (root / "checkpoint-20").mkdir()
        require(checkpoint_is_complete(checkpoint), "complete checkpoint marker")
        require(
            latest_complete_checkpoint(root) == checkpoint,
            "skip incomplete latest checkpoint",
        )
        (checkpoint / "unrecorded.safetensors").write_bytes(b"unrecorded")
        require(
            not checkpoint_is_complete(checkpoint),
            "unrecorded checkpoint files must invalidate completion",
        )
        (checkpoint / "unrecorded.safetensors").unlink()
        (checkpoint / "optimizer.pt").write_bytes(b"corrupted")
        require(
            not checkpoint_is_complete(checkpoint),
            "checkpoint corruption rejection",
        )
        incomplete = root / "checkpoint-30"
        incomplete.mkdir()
        (incomplete / "note.txt").write_text("not a checkpoint", encoding="utf-8")
        atomic_write_json(
            incomplete / "checkpoint-complete.json",
            {
                "schema_version": 1,
                "global_step": 30,
                "files": {
                    "note.txt": {
                        "bytes": (incomplete / "note.txt").stat().st_size,
                        "sha256": sha256_file(incomplete / "note.txt"),
                    }
                },
            },
            0o600,
        )
        require(
            not checkpoint_is_complete(incomplete),
            "marker cannot bless a checkpoint without resume state and weights",
        )
    require(
        resolve_run_campaign("selftest-safe").parent == RESULTS_ROOT.resolve(),
        "safe run id remains under the fixed results root",
    )
    for unsafe_run_id in ("", ".", "..", ".hidden", "../escape", "a/b"):
        try:
            resolve_run_campaign(unsafe_run_id)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"selftest failed: unsafe run id accepted: {unsafe_run_id}")
    class_values, sample_values = class_weight_values(list(range(len(LABELS))))
    require(
        len(class_values) == len(LABELS) and len(sample_values) == len(LABELS),
        "class and sample weights",
    )
    weighted_0a = weighted_sample_indices(sample_values, base_seed=42, epoch=0)
    weighted_0b = weighted_sample_indices(sample_values, base_seed=42, epoch=0)
    weighted_1 = weighted_sample_indices(sample_values, base_seed=42, epoch=1)
    require(
        weighted_0a == weighted_0b and weighted_0a != weighted_1,
        "epoch sampler determinism",
    )
    logits = [[10.0 if row == column else 0.0 for column in range(len(LABELS))] for row in range(3)]
    metrics, diagnostics = metrics_and_calibration(logits, [0, 1, 2])
    require(metrics["accuracy"] == 1.0, "perfect synthetic accuracy")
    require(diagnostics["nll"] >= 0.0, "non-negative NLL")
    require(
        diagnostics["confusion_matrix"].shape == (len(LABELS), len(LABELS)),
        "confusion shape",
    )
    with tempfile.TemporaryDirectory() as directory:
        try:
            write_private_diagnostics(
                Path(directory) / "mismatch.npz",
                logits=logits,
                gold=[0, 1, 2],
                row_digests=["a" * 64, "b" * 64],
                diagnostics=diagnostics,
            )
        except ValueError as error:
            require(
                "identical lengths" in str(error),
                "diagnostic length rejection reason",
            )
        else:
            raise RuntimeError(
                "selftest failed: mismatched diagnostic lengths must be rejected"
            )
    import torch

    loss_logits = torch.tensor(
        [[2.0] + [0.0] * (len(LABELS) - 1), [0.0, 2.0] + [0.0] * (len(LABELS) - 2)]
    )
    loss_labels = torch.tensor([0, 1])
    loss_a = compute_arm_loss(
        loss_logits, loss_labels, class_values, arm="A", focal_gamma=1.5
    )
    loss_b = compute_arm_loss(
        loss_logits, loss_labels, class_values, arm="B", focal_gamma=1.5
    )
    loss_c = compute_arm_loss(
        loss_logits, loss_labels, class_values, arm="C", focal_gamma=1.5
    )
    loss_d = compute_arm_loss(
        loss_logits, loss_labels, class_values, arm="D", focal_gamma=1.5
    )
    require(torch.equal(loss_a, loss_b), "CE arm loss wiring")
    require(torch.equal(loss_c, loss_d), "focal arm loss wiring")
    require(not torch.equal(loss_a, loss_c), "CE versus focal distinction")
    draft = json.loads(DEFAULT_DRAFT.read_text(encoding="utf-8"))
    verify_protocol_contract(draft)
    changed = json.loads(json.dumps(draft, ensure_ascii=False))
    changed["fixed_training_protocol"]["epochs"] = 9.0
    try:
        verify_protocol_contract(changed)
    except ValueError:
        pass
    else:
        raise RuntimeError("selftest failed: protocol mutation must be rejected")
    screen_args = training_argument_values(
        draft,
        stage="screen",
        seed=42,
        cpu=False,
        output_dir=Path("/ignored/synthetic-screen"),
    )
    smoke_args = training_argument_values(
        draft,
        stage="smoke",
        seed=42,
        cpu=True,
        output_dir=Path("/ignored/synthetic-smoke"),
    )
    require(
        screen_args["num_train_epochs"] == 10.0
        and screen_args["eval_strategy"] == "epoch"
        and screen_args["save_strategy"] == "epoch"
        and screen_args["metric_for_best_model"] == "accuracy"
        and screen_args["save_total_limit"] == 2,
        "10-epoch accuracy-first Trainer arguments",
    )
    require(
        smoke_args["num_train_epochs"] == 1.0
        and smoke_args["max_steps"] == 2
        and smoke_args["use_cpu"] is True,
        "smoke Trainer arguments",
    )
    require(
        ARMS
        == {
            "A": {"weighted_sampler": False, "weighted_focal": False},
            "B": {"weighted_sampler": True, "weighted_focal": False},
            "C": {"weighted_sampler": False, "weighted_focal": True},
            "D": {"weighted_sampler": True, "weighted_focal": True},
        },
        "four-arm mapping",
    )
    print(
        "PsySUICIDE accuracy-first v2 trainer selftest PASS: "
        "optimization-only path/permission/commitment gates, deterministic "
        "stratified split, four frozen arms, no licensed data or model download"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stage", choices=("freeze", "smoke", "screen", "confirm"), default="freeze"
    )
    parser.add_argument("--optimization-file", type=Path)
    parser.add_argument("--split-manifest-out", type=Path)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--selection-artifact", type=Path)
    parser.add_argument("--resume", choices=("latest",))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    plan = {
        "schema_version": 1,
        "dry_run_default": True,
        "stage": args.stage,
        "prereg": display_path(args.prereg),
        "source_contract": {
            "optimization_only": True,
            "repository_external": True,
            "permissions": "0600",
            "rows": OPTIMIZATION_ROWS,
            "commitment_sha256": OPTIMIZATION_COMMITMENT,
            "forbidden": [
                "dataset root",
                "full train",
                "official valid",
                "official test",
                "frozen holdout",
            ],
        },
        "split": {"train": TRAIN_ROWS, "inner_dev": INNER_DEV_ROWS},
        "metric_run_requested": args.stage != "freeze",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Dry-run only. No private data was opened and no model was loaded.")
        return
    if args.optimization_file is None:
        raise SystemExit("--optimization-file is required with --execute")
    prereg_path = resolve_preregistration_path(args.prereg)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if args.stage != "freeze" and (
        prereg.get("status") != "FROZEN"
        or prereg.get("training_allowed") is not True
    ):
        raise SystemExit(
            "metric execution is blocked until the final preregistration is frozen"
        )
    entries, _source_manifest = load_optimization_only(args.optimization_file)
    train_entries, dev_entries, split_manifest = stratified_inner_split(entries)
    if args.stage == "freeze":
        if args.split_manifest_out is None:
            raise SystemExit("--split-manifest-out is required for freeze execution")
        if prereg.get("status") not in ("PARTITION_FREEZE_PENDING", "FROZEN"):
            raise SystemExit("freeze requires the declared draft or final preregistration")
        if args.split_manifest_out.exists():
            raise SystemExit("refusing to overwrite an existing split manifest")
        atomic_write_json(args.split_manifest_out, split_manifest, 0o644)
        print(json.dumps(split_manifest, ensure_ascii=False, indent=2))
        return
    try:
        resolve_run_campaign(args.run_id)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.model_dir is None or args.arm is None or args.seed is None:
        raise SystemExit("training requires --model-dir, --arm and --seed")
    git_identity = verify_frozen_execution(prereg_path, prereg, split_manifest)
    validate_phase_identity(
        args.stage,
        args.arm,
        args.seed,
        args.selection_artifact,
        prereg,
        git_identity["prereg_sha256"],
        args.run_id,
    )
    execute_training(
        args,
        prereg,
        train_entries,
        dev_entries,
        split_manifest,
        git_identity,
    )


if __name__ == "__main__":
    main()
