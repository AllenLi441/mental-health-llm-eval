#!/usr/bin/env python3
"""Validate a local three-seed run and emit an aggregate-only public report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "reports" / "psysuicide-roberta-v1.prereg.json"
COMMITMENT_PATH = ROOT / "reports" / "psysuicide-v2-train-holdout.commitment.json"
BASE_MODEL_MANIFEST_PATH = (
    ROOT / "reports" / "psysuicide-roberta-v1-base-model-manifest.json"
)
RUN_ID = "roberta-large-full-20260728"
LOCAL_MODEL_PATH = "results/model-cache/hfl-chinese-roberta-wwm-ext-large"
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
SEED_SAMPLER = "weighted replacement; inverse fourth-root class frequency"
SEED_LOSS = "class-weighted focal loss; inverse fourth-root class frequency"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def assert_close(actual: float, expected: float, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field} mismatch: {actual} != {expected}")


def verify_execution_commit(execution_commit: str, prereg: dict) -> dict:
    if (
        not isinstance(execution_commit, str)
        or len(execution_commit) != 40
        or any(character not in "0123456789abcdef" for character in execution_commit)
    ):
        raise ValueError("execution_commit is not a full lowercase Git SHA")
    base_commit = prereg["implementation"]["execution_base_commit"]
    try:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{execution_commit}^{{commit}}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", base_commit, execution_commit],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tracked_bytes = {}
        for relative in (
            prereg["implementation"]["script"],
            str(PREREG_PATH.relative_to(ROOT)),
            str(COMMITMENT_PATH.relative_to(ROOT)),
        ):
            tracked_bytes[relative] = subprocess.check_output(
                ["git", "show", f"{execution_commit}:{relative}"],
                cwd=ROOT,
                stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as error:
        raise ValueError("execution commit or preregistered ancestry is invalid") from error
    script_sha = hashlib.sha256(
        tracked_bytes[prereg["implementation"]["script"]]
    ).hexdigest()
    if script_sha != prereg["implementation"]["script_sha256"]:
        raise ValueError("training script at execution commit differs from preregistration")
    prereg_sha = hashlib.sha256(
        tracked_bytes[str(PREREG_PATH.relative_to(ROOT))]
    ).hexdigest()
    commitment_sha = hashlib.sha256(
        tracked_bytes[str(COMMITMENT_PATH.relative_to(ROOT))]
    ).hexdigest()
    if prereg_sha != sha256_file(PREREG_PATH):
        raise ValueError("preregistration differs from the execution commit")
    if commitment_sha != sha256_file(COMMITMENT_PATH):
        raise ValueError("partition commitment differs from the execution commit")
    return {
        "execution_base_commit": base_commit,
        "execution_commit": execution_commit,
        "training_script": prereg["implementation"]["script"],
        "training_script_sha256": script_sha,
        "preregistration_sha256": prereg_sha,
        "partition_commitment_artifact_sha256": commitment_sha,
        "verification": "execution commit exists, descends from the preregistered base, and contains the preregistered trainer bytes",
        "limitation": "the trainer did not record whether its execution-time worktree contained uncommitted changes",
    }


def verify_base_model(
    plan: dict,
    seeds: list[dict],
    prereg: dict,
    manifest: dict,
    verify_files: bool,
) -> dict:
    implementation = prereg["implementation"]
    if manifest.get("schema_version") != 1:
        raise ValueError("unexpected base-model manifest schema_version")
    if plan.get("model") != LOCAL_MODEL_PATH:
        raise ValueError("full run did not use the frozen local base-model path")
    if any(row.get("model") != LOCAL_MODEL_PATH for row in seeds):
        raise ValueError("seed base-model path differs from the full-run plan")
    if any(row.get("model_revision") is not None for row in seeds):
        raise ValueError("unexpected seed model_revision for the local-files run")
    if manifest.get("base_model") != implementation["base_model"]:
        raise ValueError("base-model manifest repository mismatch")
    if manifest.get("base_model_revision") != implementation["base_model_revision"]:
        raise ValueError("base-model manifest revision mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or "pytorch_model.bin" not in files:
        raise ValueError("base-model manifest has no frozen model files")
    if manifest.get("official_revision_sha") != implementation["base_model_revision"]:
        raise ValueError("official base-model revision evidence mismatch")
    if (
        manifest.get("official_weight_lfs_sha256")
        != files["pytorch_model.bin"].get("sha256")
        or manifest.get("official_weight_bytes")
        != files["pytorch_model.bin"].get("bytes")
    ):
        raise ValueError("official LFS evidence differs from the local weight manifest")
    if verify_files:
        cache_root = (ROOT / LOCAL_MODEL_PATH).resolve()
        declared_revision = implementation["base_model_revision"]
        for filename, expected in files.items():
            path = cache_root / filename
            if not path.is_file():
                raise ValueError(f"base-model cache file missing: {filename}")
            if path.stat().st_size != expected["bytes"]:
                raise ValueError(f"base-model cache byte count mismatch: {filename}")
            if sha256_file(path) != expected["sha256"]:
                raise ValueError(f"base-model cache SHA-256 mismatch: {filename}")
            if filename != "pytorch_model.bin":
                if not path.is_symlink():
                    raise ValueError(
                        f"base-model metadata/tokenizer is not a snapshot symlink: {filename}"
                    )
                link_target = str(path.readlink())
                if f"/snapshots/{declared_revision}/" not in link_target:
                    raise ValueError(
                        f"base-model metadata/tokenizer is not linked to revision {declared_revision}: {filename}"
                    )
    return {
        "base_model": implementation["base_model"],
        "base_model_revision": implementation["base_model_revision"],
        "local_cache_manifest": str(BASE_MODEL_MANIFEST_PATH.relative_to(ROOT)),
        "local_cache_manifest_sha256": (
            sha256_file(BASE_MODEL_MANIFEST_PATH)
            if verify_files else manifest.get("fixture_sha256")
        ),
        "verification_status": (
            "POST_RUN_LOCAL_ARTIFACT_MATCH"
            if verify_files else "SELFTEST_FIXTURE"
        ),
        "limitation": (
            "the trainer used a local path and recorded model_revision=null; "
            "the pinned revision and exact cache bytes were verified after the run, "
            "not cryptographically recorded inside each seed result"
        ),
    }


def validate_seed_metrics(row: dict, expected_rows: int) -> None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("seed metrics missing")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict) or list(per_class) != LABELS:
        raise ValueError("seed per-class labels/order differ from the declared taxonomy")
    total_support = 0
    weighted_f1_numerator = 0.0
    true_positive_total = 0.0
    class_f1 = []
    for label in LABELS:
        values = per_class[label]
        support = values.get("support")
        if not isinstance(support, int) or isinstance(support, bool) or support <= 0:
            raise ValueError(f"invalid support for {label}")
        total_support += support
        for metric in ("precision", "recall", "f1"):
            value = values.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"non-finite {metric} for {label}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"out-of-range {metric} for {label}")
        class_f1.append(float(values["f1"]))
        weighted_f1_numerator += float(values["f1"]) * support
        raw_true_positive = float(values["recall"]) * support
        rounded_true_positive = round(raw_true_positive)
        if abs(raw_true_positive - rounded_true_positive) > 1e-9:
            raise ValueError(f"recall/support imply a non-integer TP for {label}")
        true_positive_total += rounded_true_positive
        precision = float(values["precision"])
        recall = float(values["recall"])
        expected_f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        assert_close(float(values["f1"]), expected_f1, f"class F1 for {label}")
        if precision > 0.0:
            raw_predicted = rounded_true_positive / precision
            rounded_predicted = round(raw_predicted)
            if abs(raw_predicted - rounded_predicted) > 1e-9:
                raise ValueError(
                    f"precision/TP imply a non-integer predicted count for {label}"
                )
            if rounded_predicted < rounded_true_positive:
                raise ValueError(f"predicted count is below TP for {label}")
    if total_support != expected_rows:
        raise ValueError(f"per-class support sums to {total_support}, expected {expected_rows}")
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid seed metric: {metric}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"out-of-range seed metric: {metric}")
    assert_close(
        float(metrics["macro_f1"]),
        mean(class_f1),
        "seed macro-F1 recomputation",
    )
    assert_close(
        float(metrics["weighted_f1"]),
        weighted_f1_numerator / total_support,
        "seed weighted-F1 recomputation",
    )
    assert_close(
        float(metrics["accuracy"]),
        true_positive_total / total_support,
        "seed accuracy recomputation",
    )


def validate_class_weights(row: dict, partition: dict) -> None:
    recorded = row.get("class_weights")
    if not isinstance(recorded, dict) or list(recorded) != LABELS:
        raise ValueError("seed class-weight labels/order differ from taxonomy")
    counts = np.asarray(
        [partition["per_label"][label]["optimization"] for label in LABELS],
        dtype=np.float32,
    )
    expected = np.asarray(
        [
            (
                partition["optimization_rows"]
                / (len(LABELS) * float(count))
            )
            ** 0.25
            for count in counts
        ],
        dtype=np.float32,
    )
    expected = np.clip(expected / expected.mean(), 0.35, 4.0)
    for index, label in enumerate(LABELS):
        value = recorded[label]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid class weight for {label}")
        if not math.isclose(
            float(value),
            float(expected[index]),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError(f"class weight differs from trainer formula for {label}")


def build_report(
    aggregate: dict,
    prereg: dict,
    frozen: dict,
    base_model_manifest: dict,
    weight_root: Path | None,
    *,
    verify_git: bool = True,
    verify_files: bool = True,
) -> dict:
    if aggregate.get("schema_version") != 1:
        raise ValueError("unexpected aggregate schema_version")
    if prereg.get("schema_version") != 1 or frozen.get("schema_version") != 1:
        raise ValueError("unexpected preregistration/partition schema_version")
    plan = aggregate["plan"]
    if plan.get("schema_version") != 1:
        raise ValueError("unexpected run-plan schema_version")
    expected_training = prereg["training"]
    expected_partition = prereg["partition"]
    if prereg["implementation"]["script"] != "scripts/train_psysuicide_roberta.py":
        raise ValueError("unexpected preregistered trainer path")
    if sha256_file(COMMITMENT_PATH) != frozen.get("artifact_sha256"):
        if verify_files:
            raise ValueError("partition commitment artifact SHA-256 mismatch")
    if plan["mode"] != "full":
        raise ValueError("public selection report requires a full run")
    if plan.get("run_id") != RUN_ID:
        raise ValueError("run_id differs from the frozen supervised-v1 run")
    if plan["seeds"] != expected_training["seeds"]:
        raise ValueError("seed list differs from preregistration")
    expected_plan = {
        "train_rows": expected_partition["optimization_rows"],
        "eval_rows": expected_partition["official_valid_rows"],
        "max_length": expected_training["max_length"],
        "train_batch": expected_training["per_device_train_batch"],
        "eval_batch": expected_training["per_device_eval_batch"],
        "gradient_accumulation": expected_training["gradient_accumulation"],
        "epochs": expected_training["epochs"],
        "max_steps": -1,
    }
    for field, expected in expected_plan.items():
        if plan[field] != expected:
            raise ValueError(f"plan mismatch for {field}: {plan[field]} != {expected}")
    if plan.get("dry_run_default") is not True:
        raise ValueError("run plan does not record the dry-run safety default")
    if plan.get("partition", {}).get("holdout_access") != (
        "partition hashes verified only; holdout rows not returned, tokenized, "
        "sampled, scored, or selected on"
    ):
        raise ValueError("run plan does not preserve the holdout-closed statement")
    partition_fields = (
        "dataset_file_sha256",
        "retained_single_label_rows",
        "optimization_rows",
        "holdout_rows",
        "all_rows_commitment_sha256",
        "optimization_commitment_sha256",
        "holdout_commitment_sha256",
        "per_label",
    )
    for field in partition_fields:
        if plan["partition"].get(field) != frozen.get(field):
            raise ValueError(f"partition mismatch for {field}")
    for field in ("optimization_commitment_sha256", "holdout_commitment_sha256"):
        if plan["partition"][field] != expected_partition[field]:
            raise ValueError(f"partition differs from preregistration for {field}")
    if plan["partition"].get("valid_rows") != expected_partition["official_valid_rows"]:
        raise ValueError("official-valid row count differs from preregistration")
    if aggregate.get("holdout_scored") is not False:
        raise ValueError("holdout must remain unscored during validation selection")

    seeds = sorted(aggregate["seeds"], key=lambda row: row["seed"])
    if [row["seed"] for row in seeds] != expected_training["seeds"]:
        raise ValueError("aggregate seed results differ from preregistration")
    for row in seeds:
        if row["mode"] != "full":
            raise ValueError("mixed run mode in seed results")
        expected_seed = {
            "train_rows": expected_partition["optimization_rows"],
            "eval_rows": expected_partition["official_valid_rows"],
            "max_length": expected_training["max_length"],
            "train_batch": expected_training["per_device_train_batch"],
            "eval_batch": expected_training["per_device_eval_batch"],
            "gradient_accumulation": expected_training["gradient_accumulation"],
            "epochs": expected_training["epochs"],
            "max_steps": -1,
            "learning_rate": expected_training["learning_rate"],
            "weight_decay": expected_training["weight_decay"],
            "focal_gamma": 1.5,
            "sampler": SEED_SAMPLER,
            "loss": SEED_LOSS,
            "checkpoint": (
                f"results/psysuicide-roberta/{RUN_ID}/seed-{row['seed']}/model"
            ),
        }
        for field, expected in expected_seed.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"seed {row['seed']} mismatch for {field}: "
                    f"{row.get(field)} != {expected}"
                )
        if row.get("partition") != plan["partition"]:
            raise ValueError(f"seed {row['seed']} partition differs from the run plan")
        validate_class_weights(row, plan["partition"])
        for field in ("train_runtime_seconds", "train_loss"):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"seed {row['seed']} has invalid {field}")
        validate_seed_metrics(row, expected_partition["official_valid_rows"])
        if weight_root is not None:
            seed_artifact = weight_root / f"seed-{row['seed']}" / "aggregate.json"
            if not seed_artifact.is_file():
                raise ValueError(f"seed aggregate artifact missing: {row['seed']}")
            if json.loads(seed_artifact.read_text(encoding="utf-8")) != row:
                raise ValueError(
                    f"top-level and per-seed aggregate differ for seed {row['seed']}"
                )

    execution_identity = (
        verify_execution_commit(aggregate.get("execution_commit"), prereg)
        if verify_git
        else {
            "execution_base_commit": prereg["implementation"]["execution_base_commit"],
            "execution_commit": aggregate.get("execution_commit"),
            "training_script": prereg["implementation"]["script"],
            "training_script_sha256": prereg["implementation"]["script_sha256"],
            "verification": "selftest fixture",
        }
    )
    base_model_identity = verify_base_model(
        plan,
        seeds,
        prereg,
        base_model_manifest,
        verify_files,
    )

    metric_names = ("accuracy", "macro_f1", "weighted_f1")
    means = {
        metric: mean([row["metrics"][metric] for row in seeds])
        for metric in metric_names
    }
    stds = {
        metric: float(np.std(
            [row["metrics"][metric] for row in seeds], ddof=1
        ))
        for metric in metric_names
    }
    assert_close(
        float(aggregate.get("mean_macro_f1")),
        means["macro_f1"],
        "top-level mean_macro_f1",
    )
    assert_close(
        float(aggregate.get("std_macro_f1")),
        stds["macro_f1"],
        "top-level std_macro_f1",
    )
    expected_best_seed = max(
        seeds,
        key=lambda row: (row["metrics"]["macro_f1"], -row["seed"]),
    )["seed"]
    if aggregate.get("best_seed") != expected_best_seed:
        raise ValueError("top-level best_seed differs from the preregistered tie-break")
    labels = list(seeds[0]["metrics"]["per_class"])
    per_class = {}
    for label in labels:
        supports = {row["metrics"]["per_class"][label]["support"] for row in seeds}
        if len(supports) != 1:
            raise ValueError(f"support differs across seeds for {label}")
        per_class[label] = {
            "support": supports.pop(),
            "mean_f1": mean([
                row["metrics"]["per_class"][label]["f1"] for row in seeds
            ]),
            "std_f1": float(np.std([
                row["metrics"]["per_class"][label]["f1"] for row in seeds
            ], ddof=1)),
        }

    reference = prereg["reference"]
    support_guard_labels = {
        label
        for label, values in seeds[0]["metrics"]["per_class"].items()
        if values["support"] >= 10
    }
    if support_guard_labels != set(
        reference["per_class_f1_for_support_at_least_10"]
    ):
        raise ValueError(
            "official-valid support>=10 labels differ from the preregistered guard"
        )
    failures = []
    for label, reference_f1 in reference["per_class_f1_for_support_at_least_10"].items():
        if label not in per_class:
            raise ValueError(f"regression-guard label missing from seed metrics: {label}")
        delta = per_class[label]["mean_f1"] - reference_f1
        per_class[label]["reference_f1"] = reference_f1
        per_class[label]["delta_vs_reference"] = delta
        if delta < -0.10:
            failures.append(label)
    primary_pass = means["macro_f1"] > reference["macro_f1"]
    regression_pass = not failures
    advances = primary_pass and regression_pass
    best = next(row for row in seeds if row["seed"] == expected_best_seed)

    selected_checkpoint = None
    if advances:
        if weight_root is None:
            raise ValueError("--weight-root is required when the candidate advances")
        candidates = [
            weight_root / f"seed-{best['seed']}" / "model" / "model.safetensors",
            weight_root / f"seed-{best['seed']}" / "model" / "pytorch_model.bin",
        ]
        weight_path = next((path for path in candidates if path.is_file()), None)
        if weight_path is None:
            raise ValueError("selected checkpoint weights not found")
        checkpoint_root = weight_path.parent
        required_inference_files = (
            weight_path.name,
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "training_args.bin",
        )
        artifact_files = {}
        for filename in required_inference_files:
            path = checkpoint_root / filename
            if not path.is_file():
                raise ValueError(f"selected checkpoint inference file missing: {filename}")
            artifact_files[filename] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        config = json.loads(
            (checkpoint_root / "config.json").read_text(encoding="utf-8")
        )
        id2label = config.get("id2label", {})
        if [id2label.get(str(index)) for index in range(len(LABELS))] != LABELS:
            raise ValueError("selected checkpoint label order differs from taxonomy")
        label2id = config.get("label2id", {})
        if [label2id.get(label) for label in LABELS] != list(range(len(LABELS))):
            raise ValueError("selected checkpoint reverse label map differs from taxonomy")
        selected_checkpoint = {
            "seed": best["seed"],
            "valid_macro_f1": best["metrics"]["macro_f1"],
            "weight_filename": weight_path.name,
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": sha256_file(weight_path),
            "artifact_files": artifact_files,
            "label_order": LABELS,
            "status": "frozen before holdout access; weights remain local and ignored",
        }

    return {
        "schema_version": 2,
        "scope": "PsySUICIDE supervised v1 three-seed official-valid selection",
        "execution_commit": aggregate["execution_commit"],
        "execution_identity": execution_identity,
        "preregistration": "reports/psysuicide-roberta-v1.prereg.json",
        "preregistration_sha256": sha256_file(PREREG_PATH),
        "base_model": prereg["implementation"]["base_model"],
        "base_model_revision": prereg["implementation"]["base_model_revision"],
        "base_model_identity": base_model_identity,
        "partition": {
            "optimization_rows": expected_partition["optimization_rows"],
            "optimization_commitment_sha256": expected_partition["optimization_commitment_sha256"],
            "official_valid_rows": expected_partition["official_valid_rows"],
            "official_valid_membership_verification": (
                "row count, exact label order, per-class support vector, and aggregate "
                "metric consistency only; the v1 trainer did not record a valid-file "
                "or valid-membership commitment"
            ),
            "holdout_rows": expected_partition["holdout_rows"],
            "holdout_commitment_sha256": expected_partition["holdout_commitment_sha256"],
            "holdout_scored": False,
        },
        "configuration": {
            key: value for key, value in expected_training.items()
            if key not in ("checkpoint_selection", "public_outputs")
        },
        "seed_results": [
            {
                "seed": row["seed"],
                "device": row["device"],
                "train_runtime_seconds": row["train_runtime_seconds"],
                "train_loss": row["train_loss"],
                **{metric: row["metrics"][metric] for metric in metric_names},
            }
            for row in seeds
        ],
        "three_seed_mean": means,
        "three_seed_sample_std": stds,
        "per_class": per_class,
        "selection": {
            "primary_reference_macro_f1": reference["macro_f1"],
            "primary_delta": means["macro_f1"] - reference["macro_f1"],
            "primary_pass": primary_pass,
            "regression_guard_failures": failures,
            "regression_guard_pass": regression_pass,
            "candidate_advances": advances,
            "decision": (
                "ADVANCE_SUPERVISED_V1_FREEZE_CHECKPOINT_BEFORE_HOLDOUT"
                if advances else "REJECT_SUPERVISED_V1"
            ),
        },
        "selected_checkpoint": selected_checkpoint,
        "integrity_limits": [
            "official-valid membership was not cryptographically committed by the v1 trainer",
            "base-model revision provenance is a post-run local-artifact verification because each seed recorded model_revision=null",
            "the frozen selected checkpoint and all files used for inference are byte-hashed before holdout access",
        ],
        "publishing_boundary": "aggregate metrics and hashes only; text, ids, row-level predictions, weights, and trainer state remain private and ignored",
    }


def selftest() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(COMMITMENT_PATH.read_text(encoding="utf-8"))
    frozen["artifact_sha256"] = "fixture"
    partition = {
        key: copy.deepcopy(frozen[key])
        for key in (
            "dataset_file_sha256",
            "retained_single_label_rows",
            "optimization_rows",
            "holdout_rows",
            "all_rows_commitment_sha256",
            "optimization_commitment_sha256",
            "holdout_commitment_sha256",
            "per_label",
        )
    }
    partition.update(
        {
            "valid_rows": prereg["partition"]["official_valid_rows"],
            "holdout_access": (
                "partition hashes verified only; holdout rows not returned, "
                "tokenized, sampled, scored, or selected on"
            ),
        }
    )
    supports = [1075, 132, 132, 36, 15, 2, 10, 3, 10, 27, 17]
    per_class = {
        label: {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": supports[index],
        }
        for index, label in enumerate(LABELS)
    }
    counts = np.asarray(
        [partition["per_label"][label]["optimization"] for label in LABELS],
        dtype=np.float32,
    )
    fixture_weights = np.asarray(
        [
            (partition["optimization_rows"] / (len(LABELS) * float(count))) ** 0.25
            for count in counts
        ],
        dtype=np.float32,
    )
    fixture_weights = np.clip(fixture_weights / fixture_weights.mean(), 0.35, 4.0)
    class_weights = {
        label: float(fixture_weights[index])
        for index, label in enumerate(LABELS)
    }
    seed_rows = [
        {
            "seed": seed,
            "mode": "full",
            "model": LOCAL_MODEL_PATH,
            "model_revision": None,
            "device": "cpu",
            "train_rows": prereg["partition"]["optimization_rows"],
            "eval_rows": prereg["partition"]["official_valid_rows"],
            "epochs": prereg["training"]["epochs"],
            "max_steps": -1,
            "max_length": prereg["training"]["max_length"],
            "train_batch": prereg["training"]["per_device_train_batch"],
            "eval_batch": prereg["training"]["per_device_eval_batch"],
            "gradient_accumulation": prereg["training"]["gradient_accumulation"],
            "learning_rate": prereg["training"]["learning_rate"],
            "weight_decay": prereg["training"]["weight_decay"],
            "focal_gamma": 1.5,
            "class_weights": copy.deepcopy(class_weights),
            "sampler": SEED_SAMPLER,
            "loss": SEED_LOSS,
            "train_runtime_seconds": 1.0,
            "train_loss": 1.0,
            "partition": copy.deepcopy(partition),
            "checkpoint": f"results/psysuicide-roberta/{RUN_ID}/seed-{seed}/model",
            "metrics": {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "per_class": copy.deepcopy(per_class),
            },
        }
        for seed in prereg["training"]["seeds"]
    ]
    plan = {
        "schema_version": 1,
        "mode": "full",
        "run_id": RUN_ID,
        "model": LOCAL_MODEL_PATH,
        "seeds": prereg["training"]["seeds"],
        "train_rows": prereg["partition"]["optimization_rows"],
        "eval_rows": prereg["partition"]["official_valid_rows"],
        "max_length": prereg["training"]["max_length"],
        "train_batch": prereg["training"]["per_device_train_batch"],
        "eval_batch": prereg["training"]["per_device_eval_batch"],
        "gradient_accumulation": prereg["training"]["gradient_accumulation"],
        "epochs": prereg["training"]["epochs"],
        "max_steps": -1,
        "partition": copy.deepcopy(partition),
        "dry_run_default": True,
    }
    aggregate = {
        "schema_version": 1,
        "execution_commit": "f" * 40,
        "plan": plan,
        "seeds": seed_rows,
        "mean_macro_f1": 0.0,
        "std_macro_f1": 0.0,
        "best_seed": 42,
        "holdout_scored": False,
    }
    base_manifest = {
        "schema_version": 1,
        "base_model": prereg["implementation"]["base_model"],
        "base_model_revision": prereg["implementation"]["base_model_revision"],
        "official_revision_sha": prereg["implementation"]["base_model_revision"],
        "official_weight_lfs_sha256": "0" * 64,
        "official_weight_bytes": 1,
        "files": {
            "pytorch_model.bin": {"bytes": 1, "sha256": "0" * 64},
        },
        "fixture_sha256": "fixture",
    }
    report = build_report(
        aggregate,
        prereg,
        frozen,
        base_manifest,
        None,
        verify_git=False,
        verify_files=False,
    )
    assert report["selection"]["decision"] == "REJECT_SUPERVISED_V1"
    assert report["partition"]["holdout_scored"] is False

    def must_reject(mutator) -> None:
        candidate = copy.deepcopy(aggregate)
        mutator(candidate)
        try:
            build_report(
                candidate,
                prereg,
                frozen,
                base_manifest,
                None,
                verify_git=False,
                verify_files=False,
            )
        except (KeyError, TypeError, ValueError):
            return
        raise AssertionError("tampered aggregate was accepted")

    must_reject(lambda value: value["plan"].__setitem__("model", "wrong/model"))
    must_reject(
        lambda value: value["seeds"][0].__setitem__("learning_rate", 9e-5)
    )
    must_reject(
        lambda value: value["seeds"][0]["partition"].__setitem__(
            "optimization_commitment_sha256", "0" * 64
        )
    )
    must_reject(
        lambda value: value["seeds"][0]["metrics"].__setitem__("macro_f1", 0.6)
    )
    must_reject(
        lambda value: value["seeds"][0]["metrics"]["per_class"][LABELS[0]].__setitem__(
            "support", 1074
        )
    )
    must_reject(
        lambda value: value["seeds"][0]["metrics"]["per_class"][LABELS[0]].__setitem__(
            "recall", float("nan")
        )
    )
    must_reject(
        lambda value: value["seeds"][0]["metrics"].__setitem__(
            "per_class",
            dict(
                reversed(
                    list(value["seeds"][0]["metrics"]["per_class"].items())
                )
            ),
        )
    )
    print(
        "PsySUICIDE RoBERTa report selftest PASS: execution/config/partition "
        "identity, metric recomputation, mutation rejection, holdout closed"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--weight-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.aggregate or not args.output:
        raise SystemExit("--aggregate and --output are required")
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(COMMITMENT_PATH.read_text(encoding="utf-8"))
    frozen["artifact_sha256"] = sha256_file(COMMITMENT_PATH)
    base_model_manifest = json.loads(
        BASE_MODEL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    report = build_report(
        aggregate,
        prereg,
        frozen,
        base_model_manifest,
        args.weight_root.resolve() if args.weight_root else None,
    )
    output = args.output.resolve()
    if output.parent != (ROOT / "reports").resolve():
        raise SystemExit("--output must be directly under reports/")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
