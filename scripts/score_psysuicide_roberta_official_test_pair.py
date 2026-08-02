#!/usr/bin/env python3
"""Pair the frozen supervised-v1 checkpoint with the historical official-test arm.

The historical DeepSeek V4-Pro taxonomy JSONL is treated as an immutable
reference. The script verifies its bytes and every normalized row identity,
scores the byte-frozen Seed 43 checkpoint on exactly the same retained official
test rows, and publishes aggregate-only paired evidence.

Default mode is a closed dry run. ``--execute`` additionally requires the
protocol and scorer to be committed and pushed at ``--expected-freeze-commit``.
Candidate row predictions are kept in memory and are never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from analyze_psysuicide_test_pair import paired_randomization_macro_f1
from analyze_psysuicide_valid_matrix import (
    LABELS,
    bootstrap_metric_deltas,
    exact_mcnemar,
    f1_metrics,
    top_confusions,
)
from score_psysuicide_roberta_holdout import (
    COMMITMENT_PATH,
    DEFAULT_RUN_ID,
    DEFAULT_SELECTION,
    HoldoutDataset,
    PREREG_PATH as V1_PREREG_PATH,
    checkpoint_path,
    preflight_checkpoint,
    validate_freeze_spec,
    validate_selection,
    verify_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
PAIR_PREREG_PATH = (
    ROOT / "reports" / "psysuicide-roberta-v1-official-test-paired.prereg.json"
)
PAIR_RESULT_PATH = (
    ROOT / "reports" / "psysuicide-roberta-v1-official-test-paired-result.json"
)
HOLDOUT_FREEZE_PATH = ROOT / "reports" / "psysuicide-roberta-v1-holdout-freeze.json"
REFERENCE_ANALYSIS_PATH = ROOT / "reports" / "psytest-20260728-confirmatory-analysis.json"
REFERENCE_JSONL_PATH = (
    ROOT
    / "results"
    / "psysuicide-test-taxonomy-deepseek-v4-pro-psytest-20260728-candidate.jsonl"
)
PRIVATE_ROOT = ROOT / "results" / "psysuicide-roberta" / DEFAULT_RUN_ID
CLAIM_PATH = PRIVATE_ROOT / "official-test-paired.claim.json"
COMPLETION_PATH = PRIVATE_ROOT / "official-test-paired.completion.json"
EXPECTED_N = 1464
PROTOCOL_ID = "psysuicide-roberta-v1-official-test-paired-v1"


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


def canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def require_full_sha(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} must be a full lowercase Git SHA")
    return value


def require_hash(path: Path, expected: str, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{name} must be a regular file")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA-256 mismatch: {actual} != {expected}")


def load_json(path: Path, name: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path.name}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(f"{path.name}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def verify_freeze_commit(expected_commit: str, prereg: dict) -> dict:
    expected_commit = require_full_sha(
        expected_commit, "--expected-freeze-commit"
    )
    head = git_output("rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(
            f"HEAD differs from freeze commit: {head} != {expected_commit}"
        )
    branch = git_output("branch", "--show-current")
    expected_branch = prereg["git"]["branch"]
    if branch != expected_branch:
        raise RuntimeError(f"unexpected branch: {branch} != {expected_branch}")
    for relative in prereg["git"]["required_tracked_paths"]:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{expected_commit}:{relative}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        changed = subprocess.run(
            ["git", "diff", "--quiet", expected_commit, "--", relative],
            cwd=ROOT,
            check=False,
        )
        if changed.returncode:
            raise RuntimeError(f"frozen path differs from commit: {relative}")
    remote = prereg["git"]["remote"]
    expected_upstream = f"{remote}/{expected_branch}"
    upstream = git_output(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    if upstream != expected_upstream:
        raise RuntimeError(f"unexpected upstream: {upstream} != {expected_upstream}")
    if git_output("rev-parse", "@{upstream}") != expected_commit:
        raise RuntimeError("freeze commit is not present at the local upstream")
    remote_line = git_output(
        "ls-remote", "--heads", remote, f"refs/heads/{expected_branch}"
    )
    remote_sha = remote_line.split()[0] if remote_line else ""
    if remote_sha != expected_commit:
        raise RuntimeError(
            f"live remote SHA differs from freeze commit: {remote_sha} != {expected_commit}"
        )
    return {
        "branch": branch,
        "freeze_commit": expected_commit,
        "upstream": upstream,
        "live_remote_sha": remote_sha,
    }


def validate_protocol(prereg: dict) -> None:
    if prereg.get("schema_version") != 1:
        raise RuntimeError("unexpected paired preregistration schema_version")
    if prereg.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("paired preregistration protocol_id mismatch")
    cohort = prereg.get("cohort", {})
    if (
        cohort.get("task") != "psysuicide"
        or cohort.get("split") != "test"
        or cohort.get("expected_rows") != EXPECTED_N
        or cohort.get("label_count") != len(LABELS)
    ):
        raise RuntimeError("paired preregistration cohort mismatch")
    inference = prereg.get("paired_inference", {})
    expected = {
        "primary_metric": "macro_f1",
        "alpha": 0.05,
        "paired_randomization_repetitions": 20000,
        "paired_randomization_seed": 20260802,
        "paired_bootstrap_repetitions": 20000,
        "paired_bootstrap_seed": 20260802,
        "secondary_metric": "accuracy",
        "secondary_test": "exact two-sided McNemar",
        "equivalence_tested": False,
    }
    for field, value in expected.items():
        if inference.get(field) != value:
            raise RuntimeError(f"paired inference freeze mismatch: {field}")


def verify_candidate(prereg: dict) -> tuple[dict, dict, Path]:
    candidate = prereg["candidate"]
    require_hash(
        ROOT / candidate["valid_selection"],
        candidate["valid_selection_sha256"],
        "valid selection",
    )
    require_hash(
        ROOT / candidate["holdout_freeze"],
        candidate["holdout_freeze_sha256"],
        "holdout freeze",
    )
    selection = load_json(DEFAULT_SELECTION, "valid selection")
    v1_prereg = load_json(V1_PREREG_PATH, "supervised-v1 preregistration")
    partition = load_json(COMMITMENT_PATH, "partition commitment")
    selected = validate_selection(selection, v1_prereg, partition)
    freeze = load_json(HOLDOUT_FREEZE_PATH, "holdout freeze")
    frozen_checkpoint = validate_freeze_spec(freeze, selection, selected)
    if (
        selected.get("seed") != candidate["seed"]
        or selected.get("weight_sha256")
        != candidate["checkpoint_weight_sha256"]
        or candidate["run_id"] != DEFAULT_RUN_ID
    ):
        raise RuntimeError("candidate identity differs from preregistration")
    checkpoint = checkpoint_path(candidate["run_id"], candidate["seed"])
    verify_checkpoint(checkpoint, selected, frozen_checkpoint["files"])
    if str(checkpoint.relative_to(ROOT)) != candidate["checkpoint_root"]:
        raise RuntimeError("candidate checkpoint root mismatch")
    return selected, frozen_checkpoint, checkpoint


def validate_reference(prereg: dict) -> tuple[dict[str, dict], dict]:
    reference = prereg["reference"]
    require_hash(
        REFERENCE_ANALYSIS_PATH,
        reference["source_analysis_sha256"],
        "historical reference analysis",
    )
    require_hash(
        REFERENCE_JSONL_PATH,
        reference["source_jsonl_sha256"],
        "historical reference JSONL",
    )
    analysis = load_json(REFERENCE_ANALYSIS_PATH, "historical reference analysis")
    pinned = analysis.get("candidate", {})
    if (
        pinned.get("file") != REFERENCE_JSONL_PATH.name
        or pinned.get("sha256") != reference["source_jsonl_sha256"]
        or pinned.get("model") != reference["model"]
        or pinned.get("prompt_profile") != reference["prompt_profile"]
        or pinned.get("dataset_manifest_sha256")
        != prereg["cohort"]["dataset_manifest_sha256"]
    ):
        raise RuntimeError("historical reference analysis identity mismatch")
    rows = load_jsonl(REFERENCE_JSONL_PATH)
    if len(rows) != EXPECTED_N:
        raise RuntimeError(
            f"historical reference expected {EXPECTED_N} rows, got {len(rows)}"
        )
    required = {
        "id",
        "gold",
        "predicted",
        "ok",
        "invalid",
        "error",
        "split",
        "prompt_profile",
        "requested_model",
        "response_model",
        "fingerprint",
        "case_sha256",
        "dataset_manifest_sha256",
        "prompt_template_sha256",
        "preregistration_sha256",
    }
    by_id = {}
    for index, row in enumerate(rows, 1):
        missing = sorted(required - row.keys())
        if missing:
            raise RuntimeError(f"reference row {index} missing fields: {missing}")
        row_id = str(row["id"])
        if row_id in by_id:
            raise RuntimeError(f"duplicate normalized reference id: {row_id}")
        expected = {
            "split": "test",
            "prompt_profile": reference["prompt_profile"],
            "requested_model": reference["model"],
            "response_model": reference["model"],
            "dataset_manifest_sha256": prereg["cohort"][
                "dataset_manifest_sha256"
            ],
            "prompt_template_sha256": reference["prompt_template_sha256"],
            "preregistration_sha256": reference["preregistration_sha256"],
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise RuntimeError(f"reference row {index} {field} mismatch")
        if (
            row["gold"] not in LABELS
            or row["predicted"] not in LABELS
            or row["error"] is not None
            or row["invalid"] is not False
            or bool(row["ok"]) != (row["gold"] == row["predicted"])
            or row["fingerprint"] not in reference["fingerprints"]
        ):
            raise RuntimeError(f"reference row {index} failed integrity checks")
        by_id[row_id] = row
    golds = [row["gold"] for row in rows]
    predictions = [row["predicted"] for row in rows]
    metrics = f1_metrics(golds, predictions)
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        expected = reference["expected_metrics"][metric]
        if not math.isclose(metrics[metric], expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"historical reference {metric} mismatch")
    return by_id, metrics


def load_test_entries(dataset_root: Path, prereg: dict, reference: dict[str, dict]):
    test_path = dataset_root / "PsySUICIDE" / "repo" / "test.json"
    raw = json.loads(test_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("licensed PsySUICIDE test file must be an array")
    entries = []
    for row in raw:
        if not isinstance(row.get("labels"), list) or len(row["labels"]) != 1:
            continue
        gold = row["labels"][0]
        if gold not in LABELS:
            continue
        normalized = {
            "id": f"test:{len(entries)}:{row['idx']}",
            "sourceId": str(row["idx"]),
            "gold": gold,
            "text": row["text"],
        }
        case_sha = sha256_bytes(canonical_json(normalized).encode("utf-8"))
        entries.append(
            {
                "id": normalized["id"],
                "gold": gold,
                "case_sha256": case_sha,
                "model_row": {"text": row["text"], "labels": [gold]},
            }
        )
    if len(entries) != EXPECTED_N:
        raise RuntimeError(f"official test expected {EXPECTED_N} rows, got {len(entries)}")
    manifest = sha256_bytes(
        "\n".join(entry["case_sha256"] for entry in entries).encode("ascii")
    )
    if manifest != prereg["cohort"]["dataset_manifest_sha256"]:
        raise RuntimeError("official test dataset manifest mismatch")
    if set(reference) != {entry["id"] for entry in entries}:
        raise RuntimeError("candidate/reference normalized ID sets differ")
    for entry in entries:
        frozen = reference[entry["id"]]
        if (
            frozen["gold"] != entry["gold"]
            or frozen["case_sha256"] != entry["case_sha256"]
        ):
            raise RuntimeError("candidate/reference row identity mismatch")
    return entries, manifest


def candidate_predictions(
    checkpoint: Path,
    rows: list[dict],
    max_length: int,
    eval_batch: int,
    expected_device: str,
) -> tuple[list[str], dict, str, float]:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    )
    if expected_device == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("frozen MPS inference device is unavailable")
        device = torch.device("mps")
    elif expected_device == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"unsupported inference device: {expected_device}")
    model.to(device)
    model.eval()
    dataset = HoldoutDataset(rows, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=eval_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    gold_ids = []
    predicted_ids = []
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(
                **{key: value.to(device) for key, value in batch.items()}
            ).logits
            gold_ids.extend(labels.cpu().numpy().tolist())
            predicted_ids.extend(
                torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            )
    runtime_seconds = time.time() - started
    golds = [LABELS[index] for index in gold_ids]
    predictions = [LABELS[index] for index in predicted_ids]
    metrics = f1_metrics(golds, predictions)
    del model, tokenizer, loader, dataset
    if expected_device == "mps":
        torch.mps.empty_cache()
    return predictions, metrics, str(device), runtime_seconds


def build_result(
    prereg: dict,
    git_identity: dict,
    selected: dict,
    manifest: str,
    entries: list[dict],
    reference_by_id: dict[str, dict],
    reference_metrics: dict,
    predictions: list[str],
    candidate_metrics: dict,
    device: str,
    runtime_seconds: float,
) -> dict:
    golds = [entry["gold"] for entry in entries]
    reference_predictions = [
        reference_by_id[entry["id"]]["predicted"] for entry in entries
    ]
    if len(predictions) != EXPECTED_N:
        raise RuntimeError("candidate prediction count mismatch")
    inference = prereg["paired_inference"]
    randomization = paired_randomization_macro_f1(
        golds,
        reference_predictions,
        predictions,
        inference["paired_randomization_repetitions"],
        inference["paired_randomization_seed"],
    )
    bootstrap = bootstrap_metric_deltas(
        golds,
        reference_predictions,
        predictions,
        inference["paired_bootstrap_repetitions"],
        inference["paired_bootstrap_seed"],
    )
    reference_ok = [
        gold == prediction
        for gold, prediction in zip(golds, reference_predictions)
    ]
    candidate_ok = [
        gold == prediction for gold, prediction in zip(golds, predictions)
    ]
    candidate_only = sum(
        candidate and not reference
        for reference, candidate in zip(reference_ok, candidate_ok)
    )
    reference_only = sum(
        reference and not candidate
        for reference, candidate in zip(reference_ok, candidate_ok)
    )
    macro_delta = candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
    if (
        macro_delta > 0
        and randomization["two_sided_p"] < inference["alpha"]
        and bootstrap["macro_f1_delta_ci95"][0] > 0
    ):
        claim = "CANDIDATE_SUPERIOR_ON_PREREGISTERED_MACRO_F1"
    elif (
        macro_delta < 0
        and randomization["two_sided_p"] < inference["alpha"]
        and bootstrap["macro_f1_delta_ci95"][1] < 0
    ):
        claim = "REFERENCE_SUPERIOR_ON_PREREGISTERED_MACRO_F1"
    else:
        claim = "NO_DETECTED_PRIMARY_DIFFERENCE_NOT_A_TIE_OR_EQUIVALENCE"
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "scope": prereg["scope"],
        "comparison_status": prereg["comparison_status"],
        "preregistration": {
            "path": str(PAIR_PREREG_PATH.relative_to(ROOT)),
            "sha256": sha256_file(PAIR_PREREG_PATH),
            "freeze_commit": git_identity["freeze_commit"],
            "live_remote_sha": git_identity["live_remote_sha"],
        },
        "cohort": {
            "task": "psysuicide",
            "split": "test",
            "rows": EXPECTED_N,
            "dataset_manifest_sha256": manifest,
            "pairing": "exact normalized ID, gold label, case commitment, and dataset manifest",
        },
        "reference": {
            "role": prereg["reference"]["role"],
            "model": prereg["reference"]["model"],
            "prompt_profile": prereg["reference"]["prompt_profile"],
            "source_jsonl_sha256": prereg["reference"][
                "source_jsonl_sha256"
            ],
            "fingerprints": prereg["reference"]["fingerprints"],
            "accuracy": reference_metrics["accuracy"],
            "macro_f1": reference_metrics["macro_f1"],
            "weighted_f1": reference_metrics["weighted_f1"],
            "per_class": reference_metrics["per_class"],
            "top_confusions": top_confusions(golds, reference_predictions),
        },
        "candidate": {
            "role": prereg["candidate"]["role"],
            "model": prereg["candidate"]["model"],
            "seed": prereg["candidate"]["seed"],
            "checkpoint_weight_sha256": selected["weight_sha256"],
            "device": device,
            "runtime_seconds": runtime_seconds,
            "accuracy": candidate_metrics["accuracy"],
            "macro_f1": candidate_metrics["macro_f1"],
            "weighted_f1": candidate_metrics["weighted_f1"],
            "per_class": candidate_metrics["per_class"],
            "top_confusions": top_confusions(golds, predictions),
        },
        "paired_inference": {
            "primary_metric": "macro_f1",
            "delta_candidate_minus_reference": {
                "accuracy": candidate_metrics["accuracy"]
                - reference_metrics["accuracy"],
                "macro_f1": macro_delta,
                "weighted_f1": candidate_metrics["weighted_f1"]
                - reference_metrics["weighted_f1"],
            },
            "paired_randomization": randomization,
            "paired_bootstrap": bootstrap,
            "accuracy_exact_mcnemar_secondary": {
                "candidate_only_correct": candidate_only,
                "reference_only_correct": reference_only,
                "two_sided_p": exact_mcnemar(candidate_only, reference_only),
            },
            "claim": claim,
            "equivalence_tested": False,
        },
        "limitations": [
            "The candidate checkpoint was frozen before this official-test inference, but the reference predictions are historical rather than rerun concurrently.",
            "The official test had already been used by the project for the historical LLM campaign, so this is not a first-use two-arm blind test.",
            "This task-specific classifier comparison does not prove clinical validity or deployed product improvement.",
        ],
        "publishing_boundary": prereg["outputs"]["publishing_boundary"],
    }


def private_exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("short write while persisting paired-score claim")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def private_atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def selftest() -> None:
    item = {"id": "test:0:示例", "sourceId": "示例", "gold": LABELS[0], "text": "文本"}
    assert canonical_json(item).startswith('{"gold":')
    assert len(sha256_bytes(canonical_json(item).encode("utf-8"))) == 64
    golds = LABELS * 2
    reference = golds.copy()
    candidate = golds.copy()
    reference[0] = LABELS[1]
    randomization = paired_randomization_macro_f1(
        golds, reference, candidate, repetitions=100, seed=1
    )
    bootstrap = bootstrap_metric_deltas(
        golds, reference, candidate, repetitions=100, seed=1
    )
    assert randomization["observed_delta"] > 0
    assert len(bootstrap["macro_f1_delta_ci95"]) == 2
    assert exact_mcnemar(3, 0) == 0.25
    print(
        "PsySUICIDE supervised-v1 official-test paired scorer selftest PASS: "
        "canonical identity, paired randomization/bootstrap, and McNemar"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--expected-freeze-commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    prereg = load_json(PAIR_PREREG_PATH, "paired preregistration")
    validate_protocol(prereg)
    selected, _frozen_checkpoint, checkpoint = verify_candidate(prereg)
    reference_by_id, reference_metrics = validate_reference(prereg)
    plan = {
        "status": "DRY_RUN_READY" if not args.execute else "EXECUTION_PREFLIGHT",
        "protocol_id": PROTOCOL_ID,
        "rows": EXPECTED_N,
        "reference_accuracy": reference_metrics["accuracy"],
        "candidate_seed": selected["seed"],
        "candidate_checkpoint_weight_sha256": selected["weight_sha256"],
        "public_result": str(PAIR_RESULT_PATH.relative_to(ROOT)),
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not args.dataset_root or not args.expected_freeze_commit:
        raise SystemExit(
            "--dataset-root and --expected-freeze-commit are required for --execute"
        )
    if PAIR_RESULT_PATH.exists() or CLAIM_PATH.exists() or COMPLETION_PATH.exists():
        raise RuntimeError("paired scorer output/claim already exists; refusing overwrite")
    git_identity = verify_freeze_commit(args.expected_freeze_commit, prereg)
    inference = prereg["inference"]
    preflight_checkpoint(checkpoint, inference["device"])
    verify_checkpoint(
        checkpoint,
        selected,
        load_json(HOLDOUT_FREEZE_PATH, "holdout freeze")["selected_checkpoint"]["files"],
    )
    entries, manifest = load_test_entries(
        args.dataset_root.resolve(), prereg, reference_by_id
    )
    private_exclusive_json(
        CLAIM_PATH,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "claimed_at": utc_now(),
            "freeze_commit": git_identity["freeze_commit"],
            "preregistration_sha256": sha256_file(PAIR_PREREG_PATH),
            "checkpoint_weight_sha256": selected["weight_sha256"],
            "expected_rows": EXPECTED_N,
        },
    )
    model_rows = [entry["model_row"] for entry in entries]
    predictions, candidate_metrics, device, runtime_seconds = candidate_predictions(
        checkpoint,
        model_rows,
        inference["max_length"],
        inference["eval_batch"],
        inference["device"],
    )
    result = build_result(
        prereg,
        git_identity,
        selected,
        manifest,
        entries,
        reference_by_id,
        reference_metrics,
        predictions,
        candidate_metrics,
        device,
        runtime_seconds,
    )
    PAIR_RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result_sha = sha256_file(PAIR_RESULT_PATH)
    private_atomic_json(
        COMPLETION_PATH,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "completed_at": utc_now(),
            "freeze_commit": git_identity["freeze_commit"],
            "rows": EXPECTED_N,
            "public_result": str(PAIR_RESULT_PATH.relative_to(ROOT)),
            "public_result_sha256": result_sha,
            "candidate_accuracy": candidate_metrics["accuracy"],
            "reference_accuracy": reference_metrics["accuracy"],
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "rows": EXPECTED_N,
                "reference_accuracy": reference_metrics["accuracy"],
                "candidate_accuracy": candidate_metrics["accuracy"],
                "accuracy_delta": result["paired_inference"][
                    "delta_candidate_minus_reference"
                ]["accuracy"],
                "macro_f1_delta": result["paired_inference"][
                    "delta_candidate_minus_reference"
                ]["macro_f1"],
                "claim": result["paired_inference"]["claim"],
                "public_result_sha256": result_sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
