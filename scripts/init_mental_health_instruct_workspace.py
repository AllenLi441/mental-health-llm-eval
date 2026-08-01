#!/usr/bin/env python3
"""Create an empty repository-external MentalHealth-Instruct v1 workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "dataset-design" / "mental-health-instruct-v1"
COPIED_SPECS = [
    "README.md",
    "annotation_guideline.md",
    "data_source_policy.md",
    "expert_review.md",
    "label_schema.json",
    "blank_annotation_card.schema.json",
    "pilot/blank-card.json",
]
DIRECTORIES = ["raw_data", "annotation_packets", "reviews", "manifests", "pilot"]
EMPTY_JSONL = ["train.jsonl", "validation.jsonl"]


class WorkspaceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkspaceError(message)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_payloads(created_at: str) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    spec_hashes: dict[str, str] = {}
    for relative in COPIED_SPECS:
        payload = (DESIGN / relative).read_bytes()
        destination = f"spec/{relative}"
        payloads[destination] = payload
        spec_hashes[destination] = sha256_bytes(payload)
    for relative in EMPTY_JSONL:
        payloads[relative] = b""
    payloads["raw_data/README.md"] = (
        "# Private raw-data staging\n\n"
        "Do not place benchmark valid/test, private client text, or unlicensed data here.\n"
        "Every candidate source must pass spec/data_source_policy.md before annotation.\n"
    ).encode("utf-8")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "mental-health-instruct-v1",
        "created_at": created_at,
        "status": "EMPTY_WORKSPACE",
        "training_allowed": False,
        "record_counts": {"train": 0, "validation": 0},
        "human_review": {"completed_records": 0, "expert_approved_records": 0},
        "public_spec_sha256": spec_hashes,
        "publishing_boundary": "private rows and reviews remain outside the public repository",
    }
    payloads["manifests/workspace.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return payloads


def validate_target(target: Path) -> Path:
    resolved = target.expanduser().resolve()
    require(resolved.is_absolute(), "output directory must resolve to an absolute path")
    require(not is_within(resolved, ROOT.resolve()), "output directory must be outside the public repository")
    require(resolved != Path("/"), "refusing filesystem root")
    return resolved


def initialize(target: Path, *, execute: bool, created_at: str | None = None) -> dict:
    resolved = validate_target(target)
    timestamp = created_at
    existing_manifest = resolved / "manifests/workspace.json"
    if timestamp is None and existing_manifest.is_file():
        try:
            existing_value = json.loads(existing_manifest.read_text(encoding="utf-8"))
            timestamp = existing_value["created_at"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise WorkspaceError("existing workspace manifest has no valid created_at") from exc
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    payloads = build_payloads(timestamp)
    plan = {
        "output_dir": str(resolved),
        "mode": "EXECUTE" if execute else "DRY_RUN",
        "directories": DIRECTORIES,
        "files": sorted(payloads),
        "training_allowed": False,
        "record_counts": {"train": 0, "validation": 0},
    }
    if not execute:
        return plan

    if resolved.exists():
        require(resolved.is_dir(), "output path exists and is not a directory")
    else:
        resolved.mkdir(parents=True)
    for relative in DIRECTORIES:
        (resolved / relative).mkdir(parents=True, exist_ok=True)

    for relative, payload in payloads.items():
        destination = resolved / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(destination.is_file(), f"destination exists and is not a file: {relative}")
            require(destination.read_bytes() == payload,
                    f"refusing to overwrite non-identical file: {relative}")
            continue
        destination.write_bytes(payload)
    return plan


def run_selftest() -> None:
    fixed_time = "2026-08-01T00:00:00+00:00"
    with tempfile.TemporaryDirectory(prefix="mental-health-instruct-v1-") as temporary:
        target = Path(temporary) / "workspace"
        dry = initialize(target, execute=False, created_at=fixed_time)
        require(not target.exists(), "dry-run created output")
        require(dry["record_counts"] == {"train": 0, "validation": 0}, "dry-run count drift")
        initialize(target, execute=True, created_at=fixed_time)
        require((target / "train.jsonl").read_bytes() == b"", "train JSONL must start empty")
        require((target / "validation.jsonl").read_bytes() == b"", "validation JSONL must start empty")
        blank = json.loads((target / "spec/pilot/blank-card.json").read_text(encoding="utf-8"))
        require(blank["status"] == "UNLABELED" and blank["export_to_training"] is False,
                "copied pilot must remain unlabeled")
        initialize(target, execute=True)
        (target / "train.jsonl").write_text("private-row\n", encoding="utf-8")
        try:
            initialize(target, execute=True, created_at=fixed_time)
        except WorkspaceError:
            pass
        else:
            raise WorkspaceError("initializer overwrote or accepted non-identical train.jsonl")

    try:
        initialize(ROOT / "forbidden-workspace", execute=False, created_at=fixed_time)
    except WorkspaceError:
        pass
    else:
        raise WorkspaceError("initializer accepted a path inside the public repository")
    print("MentalHealth-Instruct workspace selftest PASS: dry-run, external-only, empty JSONL, no overwrite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        require(args.output_dir is None and not args.execute,
                "--selftest cannot be combined with --output-dir or --execute")
        run_selftest()
        return
    require(args.output_dir is not None, "--output-dir is required unless --selftest is used")
    plan = initialize(args.output_dir, execute=args.execute)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
