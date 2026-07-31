#!/usr/bin/env python3
"""One-time audited export of the frozen PsySUICIDE optimization partition.

This utility is deliberately separate from the v2 trainer. It may read the
licensed full train file only after explicit authorization, verifies the
already-published source and partition commitments, and writes only the 9,342
optimization rows. Holdout rows are never returned, printed, written,
tokenized, scored, or used for selection.

Dry-run is the default and does not open the licensed source. The row-level
output and execution receipt must remain outside the repository with mode 0600.
The only repository output is a fixed aggregate-only audit artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_BRANCH = "codex/psysuicide-accuracy-first-v2"
PUBLIC_AUDIT = (
    ROOT / "reports" / "psysuicide-roberta-v2-optimization-export.audit.json"
)
PARTITION_ID = "psysuicide-train-opt-holdout-v2"
PARTITION_SEED = "psysuicide-model-optimization-v2-2026-07-28"
SOURCE_FILE_SHA256 = (
    "6b84d36515c734963b7f43275756fe09a493b1c5cff16d15f2b27068f0c7979c"
)
ALL_ROWS_COMMITMENT = (
    "77e0e8d58aae55da70af8c5b6eb4b65047cf65749ba8ba895ed4abc87072033e"
)
OPTIMIZATION_COMMITMENT = (
    "0c3cbf98db02c609d500b4f9ef9bf95518b617bd4dffd1f4cde8a5c5e85a8dd9"
)
HOLDOUT_COMMITMENT = (
    "2691aca6db327d4e8c3b3cefe49b68716506d4181344883ae0f8a3e19a53fe30"
)
EXPECTED_RETAINED = 11_671
EXPECTED_OPTIMIZATION = 9_342
EXPECTED_HOLDOUT = 2_329
EXPECTED_DROPPED_MULTI = 164
EXPECTED_DROPPED_UNKNOWN = 0
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
EXPECTED_PER_LABEL = dict(
    zip(LABELS, (6883, 846, 844, 234, 93, 12, 65, 18, 62, 173, 112))
)
HEX40 = re.compile(r"[0-9a-f]{40}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "idx": str(row["idx"]),
            "labels": row["labels"],
            "text": str(row["text"]),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def digest_commitment(digests: Iterable[str]) -> str:
    return sha256_bytes("\n".join(sorted(digests)).encode("utf-8"))


def strict_json_array(raw: bytes) -> list[dict[str, Any]]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in licensed source")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number in licensed source")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("licensed source is not strict UTF-8 JSON") from error
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError("licensed source must be a JSON array of objects")
    return payload


def read_frozen_source(path: Path) -> tuple[bytes, dict[str, Any]]:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("source must be an absolute non-symlink path")
    resolved = path.resolve(strict=True)
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("licensed source must remain outside the repository")
    if tuple(part.lower() for part in resolved.parts[-3:]) != (
        "psysuicide",
        "repo",
        "train.json",
    ):
        raise ValueError("source must be the PsySUICIDE repo/train.json file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("licensed source must be a regular file")
        if stat.S_IMODE(before.st_mode) & 0o022:
            raise ValueError("licensed source may not be group/world writable")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("licensed source changed while being read")
    finally:
        os.close(descriptor)
    observed_sha = sha256_bytes(raw)
    if observed_sha != SOURCE_FILE_SHA256:
        raise ValueError("licensed source SHA-256 differs from the frozen artifact")
    return raw, {
        "basename": resolved.name,
        "bytes": len(raw),
        "sha256": observed_sha,
    }


def partition_optimization_only(
    rows: list[dict[str, Any]],
    *,
    labels: tuple[str, ...] = LABELS,
    partition_seed: str = PARTITION_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = set(labels)
    retained: list[dict[str, Any]] = []
    dropped_multi = 0
    dropped_unknown = 0
    for row in rows:
        row_labels = row.get("labels")
        if not isinstance(row_labels, list) or len(row_labels) != 1:
            dropped_multi += 1
            continue
        label = row_labels[0]
        if label not in allowed:
            dropped_unknown += 1
            continue
        if "idx" not in row or "text" not in row:
            raise ValueError("retained row is missing idx or text")
        normalized = {
            "idx": str(row["idx"]),
            "labels": [label],
            "text": str(row["text"]),
        }
        row_digest = sha256_bytes(canonical_row(normalized).encode("utf-8"))
        retained.append(
            {
                "row": normalized,
                "label": label,
                "row_digest": row_digest,
                "rank_digest": sha256_bytes(
                    f"{partition_seed}\0{row_digest}".encode("utf-8")
                ),
            }
        )
    if len({entry["row_digest"] for entry in retained}) != len(retained):
        raise ValueError("duplicate canonical retained rows")

    optimization_entries: list[dict[str, Any]] = []
    holdout_digests: list[str] = []
    per_label: dict[str, dict[str, int]] = {}
    for label in labels:
        group = sorted(
            (entry for entry in retained if entry["label"] == label),
            key=lambda entry: (entry["rank_digest"], entry["row_digest"]),
        )
        if not group:
            raise ValueError(f"retained label is empty: {label}")
        holdout_count = max(1, math.floor(len(group) * 0.2))
        holdout_digests.extend(
            entry["row_digest"] for entry in group[:holdout_count]
        )
        optimization_entries.extend(group[holdout_count:])
        per_label[label] = {
            "retained": len(group),
            "optimization": len(group) - holdout_count,
            "holdout": holdout_count,
        }

    all_digests = [entry["row_digest"] for entry in retained]
    optimization_digests = [
        entry["row_digest"] for entry in optimization_entries
    ]
    manifest = {
        "schema_version": 1,
        "partition_id": PARTITION_ID,
        "seed": partition_seed,
        "holdout_fraction": 0.2,
        "retained_single_label_rows": len(retained),
        "optimization_rows": len(optimization_entries),
        "holdout_rows": len(holdout_digests),
        "dropped_multi_label_rows": dropped_multi,
        "dropped_unknown_label_rows": dropped_unknown,
        "per_label": per_label,
        "all_rows_commitment_sha256": digest_commitment(all_digests),
        "optimization_commitment_sha256": digest_commitment(
            optimization_digests
        ),
        "holdout_commitment_sha256": digest_commitment(holdout_digests),
        "holdout_rows_separately_collected_or_written": False,
    }
    return [entry["row"] for entry in optimization_entries], manifest


def validate_frozen_partition(
    optimization_rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    exact = {
        "retained_single_label_rows": EXPECTED_RETAINED,
        "optimization_rows": EXPECTED_OPTIMIZATION,
        "holdout_rows": EXPECTED_HOLDOUT,
        "dropped_multi_label_rows": EXPECTED_DROPPED_MULTI,
        "dropped_unknown_label_rows": EXPECTED_DROPPED_UNKNOWN,
        "all_rows_commitment_sha256": ALL_ROWS_COMMITMENT,
        "optimization_commitment_sha256": OPTIMIZATION_COMMITMENT,
        "holdout_commitment_sha256": HOLDOUT_COMMITMENT,
        "holdout_rows_separately_collected_or_written": False,
    }
    for field, expected in exact.items():
        if manifest.get(field) != expected:
            raise ValueError(f"frozen partition mismatch for {field}")
    if len(optimization_rows) != EXPECTED_OPTIMIZATION:
        raise ValueError("unexpected optimization output length")
    counts = Counter(row["labels"][0] for row in optimization_rows)
    if dict(counts) != EXPECTED_PER_LABEL:
        raise ValueError("optimization per-label counts differ from frozen counts")
    if any(set(row) != {"idx", "labels", "text"} for row in optimization_rows):
        raise ValueError("optimization output contains undeclared fields")
    observed = digest_commitment(
        sha256_bytes(canonical_row(row).encode("utf-8"))
        for row in optimization_rows
    )
    if observed != OPTIMIZATION_COMMITMENT:
        raise ValueError("optimization rows do not reproduce the frozen commitment")


def git_text(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_pushed_exporter() -> dict[str, str]:
    relative = str(SCRIPT_PATH.relative_to(ROOT))
    git_text("ls-files", "--error-unmatch", relative)
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("exporter must be clean at HEAD")
    if git_text("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("one-time export requires a completely clean worktree")
    branch = git_text("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise ValueError("one-time export is restricted to the accuracy-first branch")
    head = git_text("rev-parse", "HEAD")
    upstream = git_text("rev-parse", "@{upstream}")
    if head != upstream or HEX40.fullmatch(head) is None:
        raise ValueError("local HEAD must equal the pushed upstream SHA")
    remote_line = git_text("ls-remote", "--heads", "origin", f"refs/heads/{branch}")
    remote = remote_line.split()[0] if remote_line else ""
    if remote != head:
        raise ValueError("live remote branch SHA must equal local HEAD")
    frozen_bytes = subprocess.check_output(
        ["git", "show", f"{head}:{relative}"], cwd=ROOT
    )
    current_sha = sha256_file(SCRIPT_PATH)
    if sha256_bytes(frozen_bytes) != current_sha:
        raise ValueError("executing exporter bytes differ from pushed HEAD")
    return {
        "branch": branch,
        "execution_commit": head,
        "exporter_sha256": current_sha,
    }


def validate_private_target(path: Path, suffix: str) -> Path:
    if not path.is_absolute() or path.suffix != suffix or path.is_symlink():
        raise ValueError(f"private target must be an absolute non-symlink {suffix} path")
    resolved_parent = path.parent.resolve(strict=True)
    if ROOT.resolve() == resolved_parent or ROOT.resolve() in resolved_parent.parents:
        raise ValueError("private targets must remain outside the repository")
    if stat.S_IMODE(resolved_parent.stat().st_mode) != 0o700:
        raise ValueError("private target directory permissions must be exactly 0700")
    target = resolved_parent / path.name
    if target.exists() or target.is_symlink():
        raise ValueError(f"refusing to overwrite private target: {target.name}")
    return target


def write_exclusive_bytes(path: Path, payload: bytes, mode: int) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, mode)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def selftest() -> None:
    labels = ("a", "b")
    rows: list[dict[str, Any]] = []
    for label in labels:
        for index in range(10):
            rows.append(
                {
                    "idx": f"{label}-{index}",
                    "labels": [label],
                    "text": f"synthetic-{label}-{index}",
                    "ignored_source_field": "must not be exported",
                }
            )
    rows.append({"idx": "multi", "labels": ["a", "b"], "text": "x"})
    first, first_manifest = partition_optimization_only(
        rows, labels=labels, partition_seed="synthetic-seed"
    )
    second, second_manifest = partition_optimization_only(
        list(reversed(rows)), labels=labels, partition_seed="synthetic-seed"
    )
    if len(first) != 16 or first_manifest["holdout_rows"] != 4:
        raise RuntimeError("selftest failed: synthetic partition counts")
    if first != second or first_manifest != second_manifest:
        raise RuntimeError("selftest failed: partition is source-order dependent")
    if any(set(row) != {"idx", "labels", "text"} for row in first):
        raise RuntimeError("selftest failed: undeclared source field was exported")
    if first_manifest["holdout_rows_separately_collected_or_written"] is not False:
        raise RuntimeError("selftest failed: holdout collection boundary")
    changed = json.loads(json.dumps(rows))
    changed[0]["text"] = "mutated"
    _, changed_manifest = partition_optimization_only(
        changed, labels=labels, partition_seed="synthetic-seed"
    )
    if (
        changed_manifest["optimization_commitment_sha256"]
        == first_manifest["optimization_commitment_sha256"]
        and changed_manifest["holdout_commitment_sha256"]
        == first_manifest["holdout_commitment_sha256"]
    ):
        raise RuntimeError("selftest failed: mutation did not change commitments")
    with tempfile.TemporaryDirectory(
        prefix="psysuicide-optimization-export-selftest-"
    ) as directory:
        target = Path(directory) / "claim.json"
        write_exclusive_bytes(target, json_bytes({"status": "CLAIMED"}), 0o600)
        if stat.S_IMODE(target.stat().st_mode) != 0o600:
            raise RuntimeError("selftest failed: private claim permissions")
        overwrite_blocked = False
        try:
            write_exclusive_bytes(
                target, json_bytes({"status": "SECOND_CLAIM"}), 0o600
            )
        except FileExistsError:
            overwrite_blocked = True
        if not overwrite_blocked:
            raise RuntimeError("selftest failed: second claim was not rejected")
    print(
        "PsySUICIDE one-time optimization exporter selftest PASS: deterministic "
        "partition, optimization-only rows, aggregate holdout verification"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-one-time-export", action="store_true")
    parser.add_argument("--source-train", type=Path)
    parser.add_argument("--optimization-out", type=Path)
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--public-audit-out", type=Path, default=PUBLIC_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    plan = {
        "schema_version": 1,
        "dry_run_default": True,
        "authorized_operation": "one-time optimization-only export",
        "expected_source_sha256": SOURCE_FILE_SHA256,
        "expected_optimization_rows": EXPECTED_OPTIMIZATION,
        "expected_optimization_commitment_sha256": OPTIMIZATION_COMMITMENT,
        "holdout_policy": (
            "aggregate count/commitment verification only; no holdout row output, "
            "printing, tokenization, scoring, or selection"
        ),
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("Dry-run only. The licensed source and all output paths were not opened.")
        return
    if not args.confirm_one_time_export:
        raise SystemExit("--confirm-one-time-export is required")
    if args.source_train is None or args.optimization_out is None or args.receipt_out is None:
        raise SystemExit("execution requires source, optimization output, and receipt paths")
    public_audit = args.public_audit_out.resolve(strict=False)
    if public_audit != PUBLIC_AUDIT or public_audit.exists():
        raise SystemExit("public audit path must be the fixed new report path")
    optimization_out = validate_private_target(args.optimization_out, ".jsonl")
    receipt_out = validate_private_target(args.receipt_out, ".json")
    if optimization_out.parent != receipt_out.parent:
        raise SystemExit("private output and receipt must share one 0700 directory")
    claim_out = validate_private_target(
        receipt_out.with_name(f"{receipt_out.stem}.claim.json"), ".json"
    )
    git_identity = verify_pushed_exporter()

    claim = {
        "schema_version": 1,
        "status": "CLAIMED_BEFORE_SOURCE_READ",
        "authorization_scope": (
            "one-time export of the precommitted optimization partition only"
        ),
        "claimed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_identity": git_identity,
        "source_basename": args.source_train.name,
        "optimization_output_basename": optimization_out.name,
        "receipt_basename": receipt_out.name,
        "local_absolute_paths_recorded": False,
    }
    write_exclusive_bytes(claim_out, json_bytes(claim), 0o600)
    claim_sha = sha256_file(claim_out)

    raw, source_identity = read_frozen_source(args.source_train)
    rows = strict_json_array(raw)
    del raw
    optimization_rows, manifest = partition_optimization_only(rows)
    del rows
    validate_frozen_partition(optimization_rows, manifest)
    output_payload = (
        "\n".join(canonical_row(row) for row in optimization_rows) + "\n"
    ).encode("utf-8")
    del optimization_rows
    write_exclusive_bytes(optimization_out, output_payload, 0o600)
    output_identity = {
        "basename": optimization_out.name,
        "bytes": len(output_payload),
        "sha256": sha256_bytes(output_payload),
        "permissions": "0600",
        "rows": EXPECTED_OPTIMIZATION,
    }
    del output_payload

    completed_at_utc = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    receipt = {
        "schema_version": 1,
        "status": "COMPLETED_ONCE",
        "completed_at_utc": completed_at_utc,
        "authorization_scope": (
            "one-time export of the precommitted optimization partition only"
        ),
        "private_claim_sha256": claim_sha,
        "git_identity": git_identity,
        "source": source_identity,
        "partition": manifest,
        "optimization_output": output_identity,
        "privacy": {
            "holdout_rows_written": False,
            "holdout_rows_printed": False,
            "holdout_rows_tokenized": False,
            "holdout_rows_scored": False,
            "holdout_used_for_selection": False,
            "local_absolute_paths_recorded": False,
        },
    }
    write_exclusive_bytes(receipt_out, json_bytes(receipt), 0o600)
    receipt_sha = sha256_file(receipt_out)
    audit = {
        "schema_version": 1,
        "status": "VERIFIED_OPTIMIZATION_ONLY_EXPORT",
        "completed_at_utc": completed_at_utc,
        "git_identity": git_identity,
        "source": source_identity,
        "partition": manifest,
        "optimization_output": output_identity,
        "private_claim_sha256": claim_sha,
        "private_receipt_sha256": receipt_sha,
        "publishing_boundary": (
            "aggregate identities only; no text, row ids, membership, local paths, "
            "secrets, predictions, diagnostics, or weights"
        ),
    }
    write_exclusive_bytes(public_audit, json_bytes(audit), 0o644)
    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
