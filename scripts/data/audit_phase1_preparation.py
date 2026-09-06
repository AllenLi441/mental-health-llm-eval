#!/usr/bin/env python3
"""Record local preparation evidence without reading credentials or changing frozen data."""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_a3_candidates import sha256


def describe(path, label):
    if not path.is_file():
        return {"path": label, "exists": False}
    return {"path": label, "exists": True, "sha256": sha256(path), "bytes": path.stat().st_size}


def count_jsonl(path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new audit directory")
    args.output.mkdir(parents=True)
    root = args.eval_root.resolve()
    datasets = args.datasets_root.resolve()
    research = datasets / "mental-health-instruct-research-v0"
    v2 = datasets / "mental-health-instruction-dataset-v2"
    protected = list((research / "exports").glob("*.jsonl"))
    protected += [research / "manifests/export_manifest.json", research / "manifests/split_manifest.json",
                  v2 / "spec/DATASET_FREEZE_V2.md", v2 / "manifests/training_readiness.json"]
    for line in (research / "server_training_bundle/BUNDLE_CONTENTS.sha256").read_text().splitlines():
        if line.strip():
            relative = line.split(maxsplit=1)[1].lstrip("*")
            path = research / "server_training_bundle" / relative
            path.resolve().relative_to(research.resolve())
            protected.append(path)
    before = {str(p.relative_to(datasets)): sha256(p) for p in sorted(set(protected))}
    checks = [
        ("research_contract", research, [sys.executable, "scripts/verify_bundle_contract.py"]),
        ("research_profiles", research / "server_training_bundle", [sys.executable, "scripts/test_model_profiles.py"]),
        ("research_checksums", research / "server_training_bundle", ["shasum", "-a", "256", "-c", "BUNDLE_CONTENTS.sha256"]),
        ("legacy_v2_preflight_selftest", v2, [sys.executable, "server_training_bundle/scripts/preflight.py", "--selftest"]),
        ("candidate_tests", root, [sys.executable, "-m", "unittest", "discover", "-s", "scripts/data", "-p", "test_*.py", "-v"]),
    ]
    results = {}
    for name, cwd, command in checks:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
        log = args.output / f"{name}.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        results[name] = {"command": command, "cwd": str(cwd.relative_to(root)) if cwd.is_relative_to(root) else str(cwd.relative_to(datasets)),
                         "exit_code": result.returncode, "log": log.name, "log_sha256": sha256(log)}
    candidate = json.loads(args.candidate_manifest.read_text())
    (args.output / "candidate_manifest.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n")
    readiness = json.loads((v2 / "manifests/training_readiness.json").read_text())
    after = {str(p.relative_to(datasets)): sha256(p) for p in sorted(set(protected))}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    report = {
        "schema_version": "phase1-lineage-inventory-1.0",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_base_commit": commit,
        "python_version": platform.python_version(),
        "audit_script_sha256": sha256(Path(__file__)),
        "checks": results,
        "protected_files_before_sha256": before,
        "protected_files_after_sha256": after,
        "protected_files_unchanged": before == after,
        "tracks": {
            "research_v0": {"content_state": "FROZEN_SMOKE_EXPORT", "actual_rows": {
                s: count_jsonl(research / f"exports/{s}.jsonl") for s in ["train", "development"]},
                "gpu_execution": "NOT_RUN", "product_use": "NOT_APPROVED"},
            "a3_b1": {"content_state": candidate["content_state"], "raw_records": candidate["raw_pool_records"],
                       "normalized_candidates": candidate["normalized_candidate_records"],
                       "raw_capacity_70_30": candidate["raw_70_30_capacity_estimate"],
                       "candidate_manifest": "candidate_manifest.json", "final_split": "NOT_BUILT"},
            "m1_v1": {"content_state": "HISTORICAL_CANDIDATE_SPLIT", "actual_rows": {
                s: count_jsonl(root / f"artifacts/m1-v1/data/{s}.jsonl") for s in ["train", "val", "test"]},
                "issues": ["license_scope_unresolved", "PII_review_incomplete", "target_truncation_reported", "not_final_independent_test"]},
            "legacy_human_v2": {"content_state": readiness.get("status"),
                "approved_file": describe(v2 / "production/canonical/approved_records.jsonl", "production/canonical/approved_records.jsonl"),
                "actual_approved_rows": count_jsonl(v2 / "production/canonical/approved_records.jsonl") or 0,
                "planned_rows": {"train": 6000, "development": 600, "human_gold": 200, "safety": 400},
                "scope": "This historical route only; not the exclusive project route"},
        },
        "evidence_files": [describe(root / p, p) for p in [
            "project-ledger/CURRENT_STATUS.md", "论文和代码/B1_STATUS.md", "论文和代码/licenses/license_manifest.tsv",
            "论文和代码/eval/instrument/preregistration_amendment_01.md", "论文和代码/eval/instrument/rubric_text.txt",
            "artifacts/m1-v1/audit/tokenization.json", "artifacts/m1-v1/audit/license_audit.json",
            "scripts/data/prepare_a3_candidates.py", "scripts/data/test_prepare_a3_candidates.py"]],
    }
    (args.output / "lineage_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    ok = before == after and all(r["exit_code"] == 0 for r in results.values())
    print(json.dumps({"static_checks_pass": ok, "protected_files_unchanged": before == after,
                      "protected_file_count": len(before), "inventory": str(args.output / "lineage_inventory.json")}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
