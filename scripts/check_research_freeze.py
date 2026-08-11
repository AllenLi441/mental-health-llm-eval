#!/usr/bin/env python3
"""Offline guard for benchmark freeze and MentalHealth-Instruct v1 design."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = ROOT / "benchmark-freezes"
DESIGN_DIR = ROOT / "dataset-design" / "mental-health-instruct-v1"
SUITE_PATH = FREEZE_DIR / "mental-health-benchmark-suite-v1.json"
BASELINE_PATH = FREEZE_DIR / "deepseek-historical-baseline-v1.json"
REGISTRY_PATH = ROOT / "benchmark-specs" / "registry.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_FAMILIES = {
    "cbtbench",
    "cpsyexam",
    "eatd",
    "emobench",
    "imhi",
    "mdd5k",
    "mentalmanip",
    "psysuicide",
}
EXPECTED_TASK_FAMILIES = {
    "knowledge_reasoning",
    "emotion_understanding",
    "risk_classification",
    "condition_classification",
    "cognitive_distortion",
    "manipulation_detection",
    "supportive_response",
}


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load JSON {path.relative_to(ROOT)}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be object: {path.relative_to(ROOT)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def live_registry_hashes() -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "check_benchmark_registry.py"),
        "--print-hashes",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    require(result.returncode == 0, f"registry checker failed: {result.stderr.strip()}")
    try:
        hashes = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError("registry checker did not emit JSON") from exc
    require(isinstance(hashes, dict), "registry hash output must be object")
    return hashes


def validate_suite(suite: dict, registry: dict, hashes: dict) -> None:
    require(suite.get("freeze_id") == "mental-health-benchmark-suite-v1", "wrong suite freeze ID")
    require(suite.get("status") == "FROZEN", "suite must be FROZEN")
    require(re.fullmatch(r"[0-9a-f]{40}", str(suite.get("source_commit", ""))) is not None,
            "suite source_commit must be a full Git SHA")
    expected_tasks = set(registry.get("expected_task_keys", []))
    frozen_tasks = set(suite.get("task_spec_sha256", {}))
    require(len(expected_tasks) == 19, "live registry must contain 19 tasks")
    require(frozen_tasks == expected_tasks, "suite task keys differ from live registry")
    require(set(suite.get("families", [])) == EXPECTED_FAMILIES, "suite families are incomplete")
    require(suite.get("registry", {}).get("task_count") == 19, "suite task_count must be 19")
    require(suite.get("registry", {}).get("family_count") == 8, "suite family_count must be 8")
    for key in ("schema_sha256", "registry_index_sha256", "registry_bundle_sha256"):
        require(suite.get("registry", {}).get(key) == hashes.get(key), f"suite {key} drifted")
    require(suite.get("task_spec_sha256") == hashes.get("task_spec_sha256"),
            "suite task-spec hashes drifted")
    boundary = json.dumps(suite.get("boundary", {}), ensure_ascii=False).lower()
    for marker in ("new test access", "training", "credentials", "model weights"):
        require(marker in boundary, f"suite boundary missing {marker!r}")


def validate_metric_values(metrics: object, task_key: str) -> None:
    require(isinstance(metrics, dict) and metrics, f"{task_key}: metrics must be nonempty object")
    for name, value in metrics.items():
        require(isinstance(name, str) and name, f"{task_key}: invalid metric name")
        if value is None:
            continue
        require(isinstance(value, (int, float)) and not isinstance(value, bool),
                f"{task_key}: metric {name} must be numeric or null")
        require(0.0 <= float(value) <= 1.0, f"{task_key}: metric {name} outside [0,1]")


def validate_baseline(baseline: dict, registry: dict) -> None:
    require(baseline.get("baseline_id") == "deepseek-historical-baseline-v1", "wrong baseline ID")
    require(baseline.get("paper_control_ready") is False, "mixed historical baseline cannot be paper ready")
    records = baseline.get("records")
    require(isinstance(records, list), "baseline records must be an array")
    require(len(records) == baseline.get("record_count") == 19, "baseline must contain 19 records")
    expected_tasks = set(registry.get("expected_task_keys", []))
    task_keys = [row.get("task_key") for row in records]
    require(len(task_keys) == len(set(task_keys)), "baseline contains duplicate task keys")
    require(set(task_keys) == expected_tasks, "baseline task keys differ from live registry")
    require({row.get("family") for row in records} == EXPECTED_FAMILIES,
            "baseline does not cover all 8 families")
    require(baseline.get("family_count") == 8, "baseline family_count must be 8")

    for row in records:
        task_key = str(row.get("task_key"))
        artifact_value = row.get("artifact")
        require(isinstance(artifact_value, str) and artifact_value, f"{task_key}: missing artifact")
        artifact = Path(artifact_value)
        require(not artifact.is_absolute() and ".." not in artifact.parts,
                f"{task_key}: artifact must remain repository-relative")
        artifact_path = ROOT / artifact
        require(artifact_path.is_file(), f"{task_key}: artifact does not exist: {artifact_value}")
        expected_sha = row.get("artifact_sha256")
        require(isinstance(expected_sha, str) and HEX64.fullmatch(expected_sha) is not None,
                f"{task_key}: invalid artifact SHA")
        require(sha256_file(artifact_path) == expected_sha, f"{task_key}: artifact SHA drifted")
        require(row.get("paper_control_eligible") is False,
                f"{task_key}: historical mixed row cannot be paper-control eligible")
        status = row.get("identity_status")
        require(status in {"exact", "response_exact_requested_legacy_alias", "legacy_alias_unresolved"},
                f"{task_key}: invalid identity status")
        if status == "legacy_alias_unresolved":
            require(row.get("model") == "deepseek-chat" and row.get("fingerprint") is None,
                    f"{task_key}: unresolved alias must not invent an exact fingerprint")
        else:
            require(str(row.get("model", "")).startswith("deepseek-v4-"),
                    f"{task_key}: exact identity must name a V4 model")
            require(isinstance(row.get("fingerprint"), str) and row.get("fingerprint"),
                    f"{task_key}: exact identity requires fingerprint")
        require(isinstance(row.get("n"), int) and row["n"] > 0, f"{task_key}: invalid n")
        validate_metric_values(row.get("metrics"), task_key)
        require(isinstance(row.get("limitation"), str) and row.get("limitation"),
                f"{task_key}: limitation is required")

    serialized = json.dumps(baseline, ensure_ascii=False).lower()
    require("do not average" in serialized, "baseline must prohibit mixed-protocol averaging")
    require("same examples" in serialized, "baseline must require same-example comparisons")
    require("/users/" not in serialized and "eval_api_key" not in serialized,
            "baseline contains local path or credential marker")


def validate_blank_card(blank: dict) -> None:
    require(blank.get("status") == "UNLABELED", "pilot card must remain UNLABELED")
    require(blank.get("instruction") == "" and blank.get("input") == "",
            "pilot card must not contain assistant-authored example text")
    require(blank.get("task_family") is None and blank.get("language") is None,
            "pilot card must not preselect a task or language")
    for field in ("author_private", "reviewer_a", "reviewer_b", "adjudication"):
        require(blank.get(field) is None, f"pilot card {field} must be blank")
    require(blank.get("export_to_training") is False, "blank pilot cannot export to training")
    source = blank.get("source", {})
    require(isinstance(source, dict) and all(value is None for value in source.values()),
            "blank pilot source must remain empty")


def validate_design() -> None:
    schema = read_json(DESIGN_DIR / "label_schema.json")
    blank_schema = read_json(DESIGN_DIR / "blank_annotation_card.schema.json")
    blank = read_json(DESIGN_DIR / "pilot" / "blank-card.json")
    validate_blank_card(blank)

    required = set(schema.get("required", []))
    require({"instruction", "input", "output", "safety", "provenance", "review"} <= required,
            "final schema missing core instruction fields")
    task_family = schema.get("properties", {}).get("task_family", {}).get("enum", [])
    require(set(task_family) == EXPECTED_TASK_FAMILIES, "final schema task families drifted")
    review = schema.get("properties", {}).get("review", {}).get("properties", {})
    require(review.get("model_generated_gold", {}).get("const") is False,
            "final schema must prohibit model-generated gold")
    require(review.get("status", {}).get("const") == "APPROVED",
            "final schema may export only APPROVED rows")
    provenance = schema.get("properties", {}).get("provenance", {}).get("properties", {})
    require(provenance.get("benchmark_test_excluded", {}).get("const") is True,
            "final schema must exclude benchmark test rows")
    require(provenance.get("pii_removed", {}).get("const") is True,
            "final schema must require PII removal")
    require(blank_schema.get("properties", {}).get("export_to_training", {}).get("const") is False,
            "blank-card schema must prohibit training export")
    require(not list(DESIGN_DIR.rglob("*.jsonl")), "public design directory must not contain JSONL data")

    documents = {
        "README.md": ["ZERO_RECORDS", "TRAINING_NOT_ALLOWED", "One human-authored pilot card"],
        "annotation_guideline.md": ["two real reviewers", "blind reviewer packet", "model_generated_gold=false"],
        "data_source_policy.md": ["benchmark valid/test", "Freeze train/validation membership by group", "AI-generated labels"],
        "expert_review.md": ["NOT STARTED", "Number of qualified reviewers | `0`", "Export to training | `NO`"],
    }
    for filename, markers in documents.items():
        text = (DESIGN_DIR / filename).read_text(encoding="utf-8")
        for marker in markers:
            require(marker in text, f"{filename} missing marker {marker!r}")


def expect_failure(callback, message: str) -> None:
    try:
        callback()
    except ValidationError:
        return
    raise ValidationError(f"selftest expected failure: {message}")


def selftest(suite: dict, baseline: dict, registry: dict, hashes: dict) -> None:
    broken_suite = copy.deepcopy(suite)
    broken_suite["registry"]["registry_bundle_sha256"] = "0" * 64
    expect_failure(lambda: validate_suite(broken_suite, registry, hashes), "registry drift")

    broken_baseline = copy.deepcopy(baseline)
    broken_baseline["paper_control_ready"] = True
    expect_failure(lambda: validate_baseline(broken_baseline, registry), "mixed baseline paper-ready claim")

    broken_alias = copy.deepcopy(baseline)
    legacy = next(row for row in broken_alias["records"] if row["identity_status"] == "legacy_alias_unresolved")
    legacy["fingerprint"] = "invented"
    expect_failure(lambda: validate_baseline(broken_alias, registry), "invented legacy fingerprint")

    blank = read_json(DESIGN_DIR / "pilot" / "blank-card.json")
    broken_blank = copy.deepcopy(blank)
    broken_blank["reviewer_a"] = {"label": "assistant-made"}
    expect_failure(lambda: validate_blank_card(broken_blank), "prelabeled pilot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    registry = read_json(REGISTRY_PATH)
    hashes = live_registry_hashes()
    suite = read_json(SUITE_PATH)
    baseline = read_json(BASELINE_PATH)
    validate_suite(suite, registry, hashes)
    validate_baseline(baseline, registry)
    validate_design()
    if args.selftest:
        selftest(suite, baseline, registry, hashes)
    print(
        "research freeze check PASS: 8 families, 19 tasks, 19 baseline records, "
        "zero public dataset rows, unlabeled pilot"
    )


if __name__ == "__main__":
    main()
