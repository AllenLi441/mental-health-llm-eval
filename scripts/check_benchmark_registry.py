#!/usr/bin/env python3
"""Deterministic, dependency-free checks for benchmark-specs/.

This intentionally implements the release-critical subset of schema.json with
the Python standard library.  It does not read licensed datasets or secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "benchmark-specs" / "registry.json"

KNOWN_METRICS = (
    "accuracy",
    "accuracy_by_language",
    "balanced_accuracy",
    "critical_class_recall",
    "depressed_class_f1",
    "exact_set_match",
    "expected_calibration_error",
    "group_accuracy",
    "invalid_rate",
    "macro_f1",
    "micro_f1",
    "multilabel_macro_f1",
    "multilabel_micro_f1",
    "per_class_f1",
    "per_class_recall",
    "positive_class_f1",
    "sensitivity",
    "specificity",
    "top1_hit_accuracy",
    "weighted_f1",
)

FAMILIES = {
    "emobench", "mdd5k", "psysuicide", "cbtbench", "mentalmanip",
    "imhi", "cpsyexam", "eatd",
}
REVISION_KINDS = {"git_commit", "dataset_release", "unversioned_local_copy"}
AVAILABILITY_STATUSES = {"present", "partial", "not_present"}
ACCESS_KINDS = {"authorized_local_only", "public"}
REDISTRIBUTION_KINDS = {"aggregate_only", "metadata_only", "public_allowed"}
SPLIT_AVAILABILITY = {"available", "not_registered", "derived_split_required"}
CONSUMPTION_STATUSES = {
    "untouched", "development_used", "exploratory_used", "confirmation_used",
    "not_applicable", "unknown",
}
ALLOWED_USES = {
    "training", "preprocessing_fit", "class_weight_estimation", "cross_validation",
    "model_selection", "early_stopping", "threshold_tuning", "prompt_selection",
    "parser_development", "exploratory_evaluation", "final_confirmation_once",
    "paper_replication", "internal_tracking_only", "historical_reporting",
    "protocol_correction_audit", "error_analysis_after_freeze",
}
TEST_SELECTION_USES = {
    "training", "preprocessing_fit", "class_weight_estimation", "cross_validation",
    "model_selection", "early_stopping", "threshold_tuning", "prompt_selection",
    "parser_development",
}
SAMPLE_MODES = {
    "full", "deterministic_sample", "paper_permutation_majority",
    "derived_patient_split", "prospective_fixed_split",
}
PAPER_STATUSES = {"compatible", "partial", "incompatible", "project_defined"}
WORKFLOW_STATES = {"active", "planned", "paused", "blocked"}
EVIDENCE_LEVELS = {"none", "selftest_only", "exploratory", "historical_frozen", "confirmatory"}

TASK_FIELDS = {
    "schema_version", "task_key", "family", "task_contract", "dataset", "use_policy", "metrics",
    "sample_protocol", "provenance_requirements", "paper_compatibility", "current_status",
}
OPTIONAL_TASK_FIELDS = {"supervised_classifier_profile"}
DATASET_FIELDS = {"root_env", "relative_root", "revision", "availability"}
TASK_CONTRACT_FIELDS = {
    "task_type", "output_representation", "canonical_output_labels", "label_order",
    "dynamic_options", "composition_rule", "scoring_method", "invalid_counts_as_incorrect",
}
SPLIT_FIELDS = {"source_split", "availability", "allowed_uses", "consumption_status", "notes"}
USE_POLICY_FIELDS = {"selection_role", "test_selection_prohibited", "train", "dev", "test"}
SAMPLE_FIELDS = {
    "protocol_id", "evaluation_role", "source_split", "selection_forbidden", "mode",
    "examples", "seed", "option_permutations", "repetitions_per_permutation",
    "api_calls_per_example", "expected_total_api_calls", "scoring_unit",
}
PROVENANCE_PART_FIELDS = {"required_fields", "frozen_before_test", "forbidden_aliases"}

MODEL_PROVENANCE_MINIMUM = {
    "requested_model", "response_model", "provider", "fingerprint", "temperature",
    "thinking", "reasoning_effort", "run_id",
}
PROMPT_PROVENANCE_MINIMUM = {"prompt_profile", "prompt_version", "prompt_template_sha256"}
PARSER_PROVENANCE_MINIMUM = {"parser_version", "parser_sha256", "invalid_policy"}
DATASET_PROVENANCE_MINIMUM = {
    "dataset_revision", "dataset_manifest_sha256", "split", "seed", "case_sha256",
}
SPECIALIST_SCREEN_TASKS = {"imhi-dr", "imhi-dreaddit", "imhi-irf", "imhi-sad"}
SCREEN_LABELS = {
    "imhi-dr": ["yes", "no"],
    "imhi-dreaddit": ["yes", "no"],
    "imhi-irf": ["yes", "no"],
    "imhi-sad": [
        "school", "financial problem", "family issues", "social relationships", "work",
        "health issues", "emotional turmoil", "everyday decision making", "other causes",
    ],
}
SCREEN_COUNTS = {
    "imhi-dr": ({"train": 1003, "dev": 430}, {"train": 1002, "dev": 430}),
    "imhi-dreaddit": ({"train": 2837, "dev": 300}, {"train": 2814, "dev": 300}),
    "imhi-irf": ({"train": 3943, "dev": 985}, {"train": 3941, "dev": 985}),
    "imhi-sad": ({"train": 5547, "dev": 616}, {"train": 5547, "dev": 616}),
}
SCREEN_SPLIT_COMMITMENTS = {
    "imhi-dr": {
        "hash_algorithm": "sha256",
        "commitment_scope": "raw_file_bytes_before_parsing",
        "train_file_sha256": "78fee2be7d39cd149d8105d4a7b80b23fb119c709392e988a3f41dab23eb64cb",
        "dev_file_sha256": "463928931041adc6b95aaf413edb4d0cece3358c84363a8fd1ea1ec5e52d9ec4",
    },
    "imhi-dreaddit": {
        "hash_algorithm": "sha256",
        "commitment_scope": "raw_file_bytes_before_parsing",
        "train_file_sha256": "4071523f933741d6ef416b5f61dd8ed4e73deb87f248a94c5b23033853d7f83d",
        "dev_file_sha256": "133317cb471c42704e8e43097159e95085c4f09822d09eee3111580a352133e4",
    },
    "imhi-irf": {
        "hash_algorithm": "sha256",
        "commitment_scope": "raw_file_bytes_before_parsing",
        "train_file_sha256": "8b36506b544909960b2800e6f933bfd9244b3c566c4084a985680442d08baa21",
        "dev_file_sha256": "720c72e9d297c9ff478cdf02cfedfe8f010868ec68655c76f6238b4059ee8670",
    },
    "imhi-sad": {
        "hash_algorithm": "sha256",
        "commitment_scope": "raw_file_bytes_before_parsing",
        "train_file_sha256": "e216e61596908b028eee9919654442c6aa35f9b0c4640532f8462980101f8640",
        "dev_file_sha256": "c9207d2cd809885e87495205722977ce89c75be6ce00aacf7721d448916635ee",
    },
}
EXPECTED_PRIMARY_METRICS = {
    "emobench-ea": ["accuracy_by_language"],
    "emobench-eu": ["accuracy_by_language"],
    "mdd5k-diagnosis": ["macro_f1"],
    "psysuicide": ["macro_f1"],
    "cbt-cd": ["multilabel_macro_f1"],
    "cbt-pc": ["multilabel_macro_f1"],
    "cbt-fc": ["multilabel_macro_f1"],
    "mentalmanip": ["positive_class_f1", "balanced_accuracy"],
    "imhi-dr": ["weighted_f1"],
    "imhi-dreaddit": ["weighted_f1"],
    "imhi-loneliness": ["weighted_f1"],
    "imhi-irf": ["weighted_f1"],
    "imhi-multiwd": ["weighted_f1"],
    "imhi-sad": ["weighted_f1"],
    "imhi-cams": ["weighted_f1"],
    "imhi-swmh": ["weighted_f1"],
    "imhi-t-sid": ["weighted_f1"],
    "cpsyexam": ["accuracy"],
    "eatd-depression": ["depressed_class_f1", "sensitivity"],
}


def task_contract(
    task_type: str,
    output_representation: str,
    labels: list[str],
    composition_rule: str,
    scoring_method: str,
    *,
    dynamic_options: bool = False,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "output_representation": output_representation,
        "canonical_output_labels": labels,
        "label_order": "declared",
        "dynamic_options": dynamic_options,
        "composition_rule": composition_rule,
        "scoring_method": scoring_method,
        "invalid_counts_as_incorrect": True,
    }


EXPECTED_TASK_CONTRACTS = {
    "emobench-ea": task_contract(
        "multiple_choice", "single_option_token", ["A", "B", "C", "D"],
        "single_token", "exact_match",
    ),
    "emobench-eu": task_contract(
        "multiple_choice", "compound_option_tokens", ["A", "B", "C", "D", "E", "F"],
        "emotion_token_plus_cause_token", "compound_exact_match", dynamic_options=True,
    ),
    "mdd5k-diagnosis": task_contract(
        "single_label_classification", "fixed_label",
        ["抑郁障碍", "焦虑障碍", "焦虑抑郁混合", "双相障碍", "其他精神障碍"],
        "single_label", "exact_match",
    ),
    "psysuicide": task_contract(
        "single_label_classification", "fixed_label",
        [
            "与自杀/自伤/攻击行为无关", "被动自杀意图", "主动自杀意图", "关于自杀的探索",
            "自杀计划", "自杀准备行为", "自杀未遂", "自伤意图", "自伤行为",
            "用户攻击行为", "他人攻击行为",
        ],
        "single_label", "exact_match",
    ),
    "cbt-cd": task_contract(
        "multilabel_classification", "label_set",
        [
            "all-or-nothing thinking", "overgeneralization", "mental filter", "should statements",
            "labeling", "personalization", "magnification", "emotional reasoning",
            "mind reading", "fortune-telling",
        ],
        "sorted_unique_label_set", "multilabel_f1",
    ),
    "cbt-pc": task_contract(
        "multilabel_classification", "label_set", ["helpless", "unlovable", "worthless"],
        "sorted_unique_label_set", "multilabel_f1",
    ),
    "cbt-fc": task_contract(
        "multilabel_classification", "label_set",
        [
            "I am incompetent", "I am helpless", "I am powerless, weak, vulnerable", "I am a victim",
            "I am needy", "I am trapped", "I am out of control", "I am a failure, loser",
            "I am defective", "I am unlovable", "I am unattractive", "I am undesirable, unwanted",
            "I am bound to be rejected", "I am bound to be abandoned", "I am bound to be alone",
            "I am worthless, waste", "I am immoral", "I am bad - dangerous, toxic, evil",
            "I don’t deserve to live",
        ],
        "sorted_unique_label_set", "multilabel_f1",
    ),
    "mentalmanip": task_contract(
        "single_label_classification", "fixed_label", ["yes", "no"],
        "single_label", "exact_match",
    ),
    "imhi-dr": task_contract(
        "single_label_classification", "fixed_label", ["yes", "no"],
        "single_label", "exact_match",
    ),
    "imhi-dreaddit": task_contract(
        "single_label_classification", "fixed_label", ["yes", "no"],
        "single_label", "exact_match",
    ),
    "imhi-loneliness": task_contract(
        "single_label_classification", "fixed_label", ["yes", "no"],
        "single_label", "exact_match",
    ),
    "imhi-irf": task_contract(
        "single_label_classification", "fixed_label", ["yes", "no"],
        "single_label", "exact_match",
    ),
    "imhi-multiwd": task_contract(
        "single_label_classification", "fixed_label", ["yes", "no"],
        "single_label", "exact_match",
    ),
    "imhi-sad": task_contract(
        "single_label_classification", "fixed_label",
        [
            "school", "financial problem", "family issues", "social relationships", "work",
            "health issues", "emotional turmoil", "everyday decision making", "other causes",
        ],
        "single_label", "exact_match",
    ),
    "imhi-cams": task_contract(
        "single_label_classification", "fixed_label",
        ["bias or abuse", "jobs and career", "medication", "relationship", "alienation", "none"],
        "single_label", "exact_match",
    ),
    "imhi-swmh": task_contract(
        "single_label_classification", "fixed_label",
        ["depression", "suicide", "anxiety", "bipolar disorder", "no mental disorders"],
        "single_label", "exact_match",
    ),
    "imhi-t-sid": task_contract(
        "single_label_classification", "fixed_label",
        ["depression", "suicide or self-harm tendency", "ptsd", "no mental disorders"],
        "single_label", "exact_match",
    ),
    "cpsyexam": task_contract(
        "multiple_choice", "option_token_set", ["A", "B", "C", "D", "E"],
        "sorted_unique_option_tokens", "exact_match", dynamic_options=True,
    ),
    "eatd-depression": task_contract(
        "single_label_classification", "fixed_label", ["depressed", "non-depressed"],
        "single_label", "exact_match",
    ),
}
PROFILE_FIELDS = {
    "profile_id", "task_type", "labels", "text_columns", "cross_split_leakage_columns", "input_parser",
    "label_column", "label_parser", "train_relative_path", "dev_relative_path", "split_commitments",
    "test_access", "loss_arms", "seed", "primary_metric", "model", "training_protocol", "source_rows",
    "used_rows", "data_cleaning_order", "provenance_fields",
}
FROZEN_TRAINING_PROTOCOL = {
    "max_length": 256,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "epochs": 5.0,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "optimizer": "adamw_torch",
    "scheduler": "linear",
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "early_stopping_patience": 2,
    "best_model_metric": "weighted_f1",
    "greater_is_better": True,
    "tie_break_rule": "first_strict_improvement_checkpoint",
    "focal_gamma": 2.0,
    "class_weighting": "balanced_inverse_frequency_after_cleaning",
    "max_grad_norm": 1.0,
    "save_total_limit": 2,
    "fp16": False,
    "bf16": False,
}
PROFILE_PROVENANCE_MINIMUM = {
    "model": {"model", "model_revision", "resolved_model_commit"},
    "tokenizer": {"tokenizer_id", "tokenizer_revision", "resolved_tokenizer_commit"},
    "head": {"task_id", "labels", "head_type", "label_mapping_sha256"},
    "loss": {"loss", "loss_definition", "class_weights", "focal_gamma"},
    "seed": {"seed"},
    "code": {"implementation_path", "implementation_sha256", "framework_versions"},
    "run": {
        "run_id", "identity_sha256", "train_manifest_sha256", "dev_manifest_sha256",
        "checkpoint_sha256",
    },
}

SECRET_OR_PRIVATE_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home)/", re.IGNORECASE),
    re.compile(r"(?:^|[/\\])\.env(?:$|[/\\])", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|secret|token)\s*[=:]", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
)
NONTEST_ROLE_FORBIDDEN_SOURCE_RE = re.compile(
    r"(?:^|[/_. -])(?:test|testing|holdout)(?:$|[/_. -])", re.IGNORECASE
)


class RegistryError(ValueError):
    """A deterministic registry validation failure."""


def fail(path: str, message: str) -> None:
    raise RegistryError(f"{path}: {message}")


def expect_object(value: Any, path: str, fields: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(path, "expected object")
    if fields is not None:
        missing = fields - set(value)
        unknown = set(value) - fields
        if missing:
            fail(path, f"missing fields: {', '.join(sorted(missing))}")
        if unknown:
            fail(path, f"unknown fields: {', '.join(sorted(unknown))}")
    return value


def expect_list(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        fail(path, "expected array")
    if nonempty and not value:
        fail(path, "must not be empty")
    return value


def expect_string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        fail(path, "expected non-empty string" if nonempty else "expected string")
    return value


def expect_unique_strings(value: Any, path: str, allowed: set[str] | None = None) -> list[str]:
    values = expect_list(value, path)
    for index, entry in enumerate(values):
        expect_string(entry, f"{path}[{index}]")
        if allowed is not None and entry not in allowed:
            fail(f"{path}[{index}]", f"unknown value {entry!r}")
    if len(values) != len(set(values)):
        fail(path, "duplicate values are not allowed")
    return values


def reject_private_or_secret_strings(value: Any, path: str = "registry") -> None:
    if isinstance(value, dict):
        for key, entry in value.items():
            reject_private_or_secret_strings(entry, f"{path}.{key}")
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            reject_private_or_secret_strings(entry, f"{path}[{index}]")
    elif isinstance(value, str):
        for pattern in SECRET_OR_PRIVATE_PATH_PATTERNS:
            if pattern.search(value):
                fail(path, "private absolute path or secret-like value is forbidden")


def validate_split_role(value: Any, path: str) -> dict[str, Any]:
    role = expect_object(value, path, SPLIT_FIELDS)
    source = role["source_split"]
    if source is not None:
        expect_string(source, f"{path}.source_split")
    if role["availability"] not in SPLIT_AVAILABILITY:
        fail(f"{path}.availability", f"unknown value {role['availability']!r}")
    uses = expect_unique_strings(role["allowed_uses"], f"{path}.allowed_uses", ALLOWED_USES)
    if role["availability"] == "available" and source is None:
        fail(f"{path}.source_split", "an available role must name its source split")
    if role["availability"] == "not_registered" and source is not None:
        fail(f"{path}.source_split", "a not_registered role must use null source_split")
    if role["availability"] == "not_registered" and uses:
        fail(f"{path}.allowed_uses", "a not_registered role cannot allow uses")
    if role["consumption_status"] not in CONSUMPTION_STATUSES:
        fail(f"{path}.consumption_status", f"unknown value {role['consumption_status']!r}")
    expect_string(role["notes"], f"{path}.notes", nonempty=False)
    return role


def validate_provenance_part(
    value: Any,
    path: str,
    minimum: set[str],
    *,
    require_frozen: bool,
) -> dict[str, Any]:
    part = expect_object(value, path)
    missing_keys = {"required_fields", "frozen_before_test"} - set(part)
    unknown_keys = set(part) - PROVENANCE_PART_FIELDS
    if missing_keys:
        fail(path, f"missing fields: {', '.join(sorted(missing_keys))}")
    if unknown_keys:
        fail(path, f"unknown fields: {', '.join(sorted(unknown_keys))}")
    fields = set(expect_unique_strings(part["required_fields"], f"{path}.required_fields"))
    if not minimum <= fields:
        fail(f"{path}.required_fields", f"missing required provenance names: {', '.join(sorted(minimum - fields))}")
    if not isinstance(part["frozen_before_test"], bool):
        fail(f"{path}.frozen_before_test", "expected boolean")
    if require_frozen and not part["frozen_before_test"]:
        fail(f"{path}.frozen_before_test", "must be true")
    if "forbidden_aliases" in part:
        expect_unique_strings(part["forbidden_aliases"], f"{path}.forbidden_aliases")
    return part


def validate_supervised_profile(value: Any, task: dict[str, Any], path: str) -> None:
    profile = expect_object(value, path, PROFILE_FIELDS)
    key = task["task_key"]
    expect_string(profile["profile_id"], f"{path}.profile_id")
    if profile["task_type"] != "single_label_text_classification":
        fail(f"{path}.task_type", "must equal single_label_text_classification")
    labels = expect_unique_strings(profile["labels"], f"{path}.labels")
    if labels != SCREEN_LABELS[key]:
        fail(f"{path}.labels", f"must preserve frozen order {SCREEN_LABELS[key]!r}")
    if profile["text_columns"] != ["question", "post"]:
        fail(f"{path}.text_columns", "must equal ['question', 'post'] in that order")
    if profile["cross_split_leakage_columns"] != ["post"]:
        fail(
            f"{path}.cross_split_leakage_columns",
            "must equal ['post']; within-train dedup still uses the full question+post input",
        )
    if profile["input_parser"] != "csv-dictreader-utf8-sig-newline-join-v1":
        fail(f"{path}.input_parser", "unexpected input parser")
    if profile["label_column"] != "response":
        fail(f"{path}.label_column", "must equal response")
    if profile["label_parser"] != "imhi-response-prefix":
        fail(f"{path}.label_parser", "must equal imhi-response-prefix")
    for field, role in (("train_relative_path", "train"), ("dev_relative_path", "dev")):
        relative = expect_string(profile[field], f"{path}.{field}")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            fail(f"{path}.{field}", "must be a safe path relative to EVAL_DATASETS_DIR")
        if relative != task["use_policy"][role]["source_split"]:
            fail(f"{path}.{field}", f"must match use_policy.{role}.source_split")
    commitments = expect_object(
        profile["split_commitments"],
        f"{path}.split_commitments",
        {"hash_algorithm", "commitment_scope", "train_file_sha256", "dev_file_sha256"},
    )
    for field in ("train_file_sha256", "dev_file_sha256"):
        digest = expect_string(commitments[field], f"{path}.split_commitments.{field}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"{path}.split_commitments.{field}", "must be a lowercase SHA-256 digest")
    if commitments != SCREEN_SPLIT_COMMITMENTS[key]:
        fail(
            f"{path}.split_commitments",
            "must equal the preregistered raw train/dev byte commitments",
        )
    if profile["test_access"] is not False:
        fail(f"{path}.test_access", "must be false")
    if profile["loss_arms"] != ["ce", "weighted-ce", "focal"]:
        fail(f"{path}.loss_arms", "must equal ['ce', 'weighted-ce', 'focal']")
    if profile["seed"] != 42:
        fail(f"{path}.seed", "must equal 42 for the frozen single-seed screen")
    if profile["primary_metric"] != "weighted_f1":
        fail(f"{path}.primary_metric", "must equal weighted_f1")
    if task["metrics"]["primary"][0] != profile["primary_metric"]:
        fail(f"{path}.primary_metric", "must match the task primary metric")
    model = expect_object(
        profile["model"],
        f"{path}.model",
        {"base_model_id", "base_model_revision", "tokenizer_id", "tokenizer_revision", "head_type"},
    )
    if model != {
        "base_model_id": "FacebookAI/roberta-base",
        "base_model_revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "tokenizer_id": "FacebookAI/roberta-base",
        "tokenizer_revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "head_type": "independent_sequence_classification_head",
    }:
        fail(f"{path}.model", "does not match the frozen RoBERTa screen identity")
    training_protocol = expect_object(
        profile["training_protocol"],
        f"{path}.training_protocol",
        set(FROZEN_TRAINING_PROTOCOL),
    )
    if training_protocol != FROZEN_TRAINING_PROTOCOL:
        fail(
            f"{path}.training_protocol",
            f"must equal the frozen screen protocol {FROZEN_TRAINING_PROTOCOL!r}",
        )
    for field, expected in zip(("source_rows", "used_rows"), SCREEN_COUNTS[key]):
        observed = expect_object(profile[field], f"{path}.{field}", {"train", "dev"})
        if observed != expected:
            fail(f"{path}.{field}", f"must equal {expected!r}")
    if profile["data_cleaning_order"] != [
        "same_label_content_keep_first",
        "conflicting_label_content_drop_group",
        "train_dev_content_overlap_drop_train_keep_dev",
    ]:
        fail(f"{path}.data_cleaning_order", "unexpected leakage-cleaning order")
    provenance = expect_object(
        profile["provenance_fields"], f"{path}.provenance_fields", set(PROFILE_PROVENANCE_MINIMUM)
    )
    for part, minimum in PROFILE_PROVENANCE_MINIMUM.items():
        fields = set(expect_unique_strings(provenance[part], f"{path}.provenance_fields.{part}"))
        if not minimum <= fields:
            fail(
                f"{path}.provenance_fields.{part}",
                f"missing fields: {', '.join(sorted(minimum - fields))}",
            )


def validate_task(task: Any, path: str = "task") -> dict[str, Any]:
    task = expect_object(task, path)
    missing_task_fields = TASK_FIELDS - set(task)
    unknown_task_fields = set(task) - TASK_FIELDS - OPTIONAL_TASK_FIELDS
    if missing_task_fields:
        fail(path, f"missing fields: {', '.join(sorted(missing_task_fields))}")
    if unknown_task_fields:
        fail(path, f"unknown fields: {', '.join(sorted(unknown_task_fields))}")
    if task["schema_version"] != "1.0":
        fail(f"{path}.schema_version", "must equal '1.0'")
    key = expect_string(task["task_key"], f"{path}.task_key")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", key):
        fail(f"{path}.task_key", "must match ^[a-z0-9][a-z0-9-]*$")
    if task["family"] not in FAMILIES:
        fail(f"{path}.family", f"unknown family {task['family']!r}")
    contract = expect_object(task["task_contract"], f"{path}.task_contract", TASK_CONTRACT_FIELDS)
    labels = expect_unique_strings(
        contract["canonical_output_labels"], f"{path}.task_contract.canonical_output_labels"
    )
    if len(labels) < 2:
        fail(f"{path}.task_contract.canonical_output_labels", "must contain at least two labels/tokens")
    if not isinstance(contract["dynamic_options"], bool):
        fail(f"{path}.task_contract.dynamic_options", "expected boolean")
    if contract["invalid_counts_as_incorrect"] is not True:
        fail(f"{path}.task_contract.invalid_counts_as_incorrect", "must be true")
    expected_contract = EXPECTED_TASK_CONTRACTS.get(key)
    if expected_contract is None:
        fail(f"{path}.task_key", "task has no frozen task/output contract")
    if contract != expected_contract:
        fail(
            f"{path}.task_contract",
            f"does not match frozen contract for {key}; expected {expected_contract!r}",
        )

    dataset = expect_object(task["dataset"], f"{path}.dataset", DATASET_FIELDS)
    if dataset["root_env"] != "EVAL_DATASETS_DIR":
        fail(f"{path}.dataset.root_env", "must equal EVAL_DATASETS_DIR")
    relative_root = expect_string(dataset["relative_root"], f"{path}.dataset.relative_root")
    relative_parts = Path(relative_root).parts
    if Path(relative_root).is_absolute() or ".." in relative_parts:
        fail(f"{path}.dataset.relative_root", "must be a safe path relative to EVAL_DATASETS_DIR")
    revision = expect_object(
        dataset["revision"], f"{path}.dataset.revision", {"kind", "value", "frozen"}
    )
    if revision["kind"] not in REVISION_KINDS:
        fail(f"{path}.dataset.revision.kind", f"unknown value {revision['kind']!r}")
    revision_value = expect_string(revision["value"], f"{path}.dataset.revision.value")
    if not isinstance(revision["frozen"], bool):
        fail(f"{path}.dataset.revision.frozen", "expected boolean")
    if revision["kind"] == "git_commit" and not re.fullmatch(r"[0-9a-f]{40}", revision_value):
        fail(f"{path}.dataset.revision.value", "git_commit must be a lowercase 40-hex SHA")
    if revision["kind"] == "unversioned_local_copy" and revision["frozen"]:
        fail(f"{path}.dataset.revision.frozen", "unversioned_local_copy cannot be marked frozen")
    availability = expect_object(
        dataset["availability"],
        f"{path}.dataset.availability",
        {"status", "access", "redistribution"},
    )
    if availability["status"] not in AVAILABILITY_STATUSES:
        fail(f"{path}.dataset.availability.status", f"unknown value {availability['status']!r}")
    if availability["access"] not in ACCESS_KINDS:
        fail(f"{path}.dataset.availability.access", f"unknown value {availability['access']!r}")
    if availability["redistribution"] not in REDISTRIBUTION_KINDS:
        fail(
            f"{path}.dataset.availability.redistribution",
            f"unknown value {availability['redistribution']!r}",
        )

    use_policy = expect_object(task["use_policy"], f"{path}.use_policy", USE_POLICY_FIELDS)
    if use_policy["selection_role"] not in {"train", "dev", "none"}:
        fail(f"{path}.use_policy.selection_role", "must be train, dev, or none")
    if use_policy["test_selection_prohibited"] is not True:
        fail(f"{path}.use_policy.test_selection_prohibited", "must be true")
    roles = {
        name: validate_split_role(use_policy[name], f"{path}.use_policy.{name}")
        for name in ("train", "dev", "test")
    }
    for role_name in ("train", "dev"):
        source = roles[role_name]["source_split"]
        if source is not None and NONTEST_ROLE_FORBIDDEN_SOURCE_RE.search(source):
            fail(
                f"{path}.use_policy.{role_name}.source_split",
                "train/dev source names may not resolve to a test or holdout split",
            )
    leaked_uses = TEST_SELECTION_USES & set(roles["test"]["allowed_uses"])
    if leaked_uses:
        fail(
            f"{path}.use_policy.test.allowed_uses",
            f"test selection/training is forbidden: {', '.join(sorted(leaked_uses))}",
        )
    if use_policy["selection_role"] != "none":
        selected_role = roles[use_policy["selection_role"]]
        if not ({"model_selection", "prompt_selection", "cross_validation"} & set(selected_role["allowed_uses"])):
            fail(
                f"{path}.use_policy.selection_role",
                "selected role must allow model_selection, prompt_selection, or cross_validation",
            )

    metrics = expect_object(task["metrics"], f"{path}.metrics", {"primary", "secondary"})
    primary = expect_unique_strings(metrics["primary"], f"{path}.metrics.primary", set(KNOWN_METRICS))
    secondary = expect_unique_strings(metrics["secondary"], f"{path}.metrics.secondary", set(KNOWN_METRICS))
    if not primary:
        fail(f"{path}.metrics.primary", "must not be empty")
    expected_primary = EXPECTED_PRIMARY_METRICS.get(key)
    if expected_primary is None:
        fail(f"{path}.task_key", "task has no frozen primary-metric contract")
    if primary != expected_primary:
        fail(
            f"{path}.metrics.primary",
            f"task {key} requires {expected_primary!r}, got {primary!r}",
        )
    overlap = set(primary) & set(secondary)
    if overlap:
        fail(f"{path}.metrics", f"primary and secondary overlap: {', '.join(sorted(overlap))}")

    sample = expect_object(task["sample_protocol"], f"{path}.sample_protocol", SAMPLE_FIELDS)
    expect_string(sample["protocol_id"], f"{path}.sample_protocol.protocol_id")
    if sample["evaluation_role"] not in {"train", "dev", "test"}:
        fail(f"{path}.sample_protocol.evaluation_role", "must be train, dev, or test")
    sample_source = expect_string(sample["source_split"], f"{path}.sample_protocol.source_split")
    if not isinstance(sample["selection_forbidden"], bool):
        fail(f"{path}.sample_protocol.selection_forbidden", "expected boolean")
    if sample["evaluation_role"] == "test" and not sample["selection_forbidden"]:
        fail(f"{path}.sample_protocol.selection_forbidden", "test sampling must forbid selection")
    sample_role = sample["evaluation_role"]
    if sample_role in {"train", "dev"} and NONTEST_ROLE_FORBIDDEN_SOURCE_RE.search(sample_source):
        fail(
            f"{path}.sample_protocol.source_split",
            "train/dev sample protocols may not resolve to a test or holdout split",
        )
    role_contract = roles[sample_role]
    if role_contract["availability"] == "not_registered":
        fail(
            f"{path}.sample_protocol.evaluation_role",
            f"cannot sample an unregistered {sample_role} role",
        )
    if role_contract["source_split"] is not None and sample_source != role_contract["source_split"]:
        fail(
            f"{path}.sample_protocol.source_split",
            f"must match use_policy.{sample_role}.source_split {role_contract['source_split']!r}",
        )
    if not sample["selection_forbidden"] and use_policy["selection_role"] != sample_role:
        fail(
            f"{path}.sample_protocol.selection_forbidden",
            "a selection-enabled sample must use the declared selection_role",
        )
    if sample["mode"] not in SAMPLE_MODES:
        fail(f"{path}.sample_protocol.mode", f"unknown mode {sample['mode']!r}")
    examples = expect_object(sample["examples"], f"{path}.sample_protocol.examples", {"kind", "value"})
    if examples["kind"] not in {"exact", "up_to", "derived"}:
        fail(f"{path}.sample_protocol.examples.kind", "must be exact, up_to, or derived")
    example_value = examples["value"]
    if examples["kind"] == "derived":
        if example_value is not None:
            fail(f"{path}.sample_protocol.examples.value", "derived sample size must be null")
    elif not isinstance(example_value, int) or isinstance(example_value, bool) or example_value <= 0:
        fail(f"{path}.sample_protocol.examples.value", "exact/up_to sample size must be a positive integer")
    if sample["seed"] is not None and (
        not isinstance(sample["seed"], int) or isinstance(sample["seed"], bool)
    ):
        fail(f"{path}.sample_protocol.seed", "expected integer or null")
    for field in (
        "option_permutations", "repetitions_per_permutation", "api_calls_per_example"
    ):
        if not isinstance(sample[field], int) or isinstance(sample[field], bool) or sample[field] <= 0:
            fail(f"{path}.sample_protocol.{field}", "must be a positive integer")
    expected_calls = sample["expected_total_api_calls"]
    if expected_calls is not None and (
        not isinstance(expected_calls, int) or isinstance(expected_calls, bool) or expected_calls <= 0
    ):
        fail(f"{path}.sample_protocol.expected_total_api_calls", "must be a positive integer or null")
    per_example = sample["option_permutations"] * sample["repetitions_per_permutation"]
    if sample["api_calls_per_example"] != per_example:
        fail(
            f"{path}.sample_protocol.api_calls_per_example",
            "must equal option_permutations * repetitions_per_permutation",
        )
    if examples["kind"] == "exact" and expected_calls != example_value * per_example:
        fail(
            f"{path}.sample_protocol.expected_total_api_calls",
            "must equal exact examples * api_calls_per_example",
        )
    expect_string(sample["scoring_unit"], f"{path}.sample_protocol.scoring_unit")

    provenance = expect_object(
        task["provenance_requirements"],
        f"{path}.provenance_requirements",
        {"model", "prompt", "parser", "dataset"},
    )
    model = validate_provenance_part(
        provenance["model"],
        f"{path}.provenance_requirements.model",
        MODEL_PROVENANCE_MINIMUM,
        require_frozen=True,
    )
    forbidden_aliases = set(model.get("forbidden_aliases", []))
    if not {"deepseek-chat", "deepseek-reasoner"} <= forbidden_aliases:
        fail(
            f"{path}.provenance_requirements.model.forbidden_aliases",
            "must forbid ambiguous deepseek-chat and deepseek-reasoner aliases",
        )
    validate_provenance_part(
        provenance["prompt"],
        f"{path}.provenance_requirements.prompt",
        PROMPT_PROVENANCE_MINIMUM,
        require_frozen=True,
    )
    validate_provenance_part(
        provenance["parser"],
        f"{path}.provenance_requirements.parser",
        PARSER_PROVENANCE_MINIMUM,
        require_frozen=True,
    )
    validate_provenance_part(
        provenance["dataset"],
        f"{path}.provenance_requirements.dataset",
        DATASET_PROVENANCE_MINIMUM,
        require_frozen=True,
    )
    if key in SPECIALIST_SCREEN_TASKS:
        if "supervised_classifier_profile" not in task:
            fail(f"{path}.supervised_classifier_profile", "required for this screen task")
        validate_supervised_profile(
            task["supervised_classifier_profile"], task, f"{path}.supervised_classifier_profile"
        )
    elif "supervised_classifier_profile" in task:
        fail(f"{path}.supervised_classifier_profile", "only the frozen four-task screen may declare this profile")

    paper = expect_object(
        task["paper_compatibility"],
        f"{path}.paper_compatibility",
        {"status", "reference_protocol", "paper_primary_metrics", "blocking_differences"},
    )
    if paper["status"] not in PAPER_STATUSES:
        fail(f"{path}.paper_compatibility.status", f"unknown value {paper['status']!r}")
    expect_string(paper["reference_protocol"], f"{path}.paper_compatibility.reference_protocol")
    expect_unique_strings(
        paper["paper_primary_metrics"],
        f"{path}.paper_compatibility.paper_primary_metrics",
        set(KNOWN_METRICS),
    )
    blockers = expect_unique_strings(
        paper["blocking_differences"], f"{path}.paper_compatibility.blocking_differences"
    )
    if paper["status"] == "compatible" and blockers:
        fail(f"{path}.paper_compatibility.blocking_differences", "compatible protocol cannot have blockers")
    if paper["status"] in {"partial", "incompatible"} and not blockers:
        fail(f"{path}.paper_compatibility.blocking_differences", "non-compatible protocol needs blockers")

    status = expect_object(
        task["current_status"],
        f"{path}.current_status",
        {"as_of", "workflow_state", "evidence_level", "summary", "next_action"},
    )
    if not isinstance(status["as_of"], str) or not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", status["as_of"]):
        fail(f"{path}.current_status.as_of", "must be YYYY-MM-DD")
    if status["workflow_state"] not in WORKFLOW_STATES:
        fail(f"{path}.current_status.workflow_state", f"unknown value {status['workflow_state']!r}")
    if status["evidence_level"] not in EVIDENCE_LEVELS:
        fail(f"{path}.current_status.evidence_level", f"unknown value {status['evidence_level']!r}")
    expect_string(status["summary"], f"{path}.current_status.summary")
    expect_string(status["next_action"], f"{path}.current_status.next_action")

    reject_private_or_secret_strings(task, path)
    return task


def validate_task_collection(tasks: list[Any], expected_keys: list[str] | None = None) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, task in enumerate(tasks):
        record = validate_task(task, f"tasks[{index}]")
        key = record["task_key"]
        if key in seen:
            fail(f"tasks[{index}].task_key", f"duplicate task key {key!r}")
        seen.add(key)
        validated.append(record)
    if expected_keys is not None:
        if len(expected_keys) != len(set(expected_keys)):
            fail("registry.expected_task_keys", "contains duplicates")
        missing = set(expected_keys) - seen
        extra = seen - set(expected_keys)
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing {', '.join(sorted(missing))}")
            if extra:
                detail.append(f"unexpected {', '.join(sorted(extra))}")
            fail("tasks", "; ".join(detail))
        if [task["task_key"] for task in validated] != expected_keys:
            fail("tasks", "task order must match registry.expected_task_keys")
    return validated


def validate_emobench_protocol(tasks: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> None:
    by_key = {task["task_key"]: task for task in tasks}
    keys = {"emobench-ea", "emobench-eu"}
    if not keys <= set(by_key):
        fail("tasks", "both emobench-ea and emobench-eu are required")
    total = 0
    for key in sorted(keys):
        sample = by_key[key]["sample_protocol"]
        expected = {
            "mode": "paper_permutation_majority",
            "examples": {"kind": "exact", "value": 400},
            "option_permutations": 4,
            "repetitions_per_permutation": 5,
            "api_calls_per_example": 20,
            "expected_total_api_calls": 8000,
        }
        for field, value in expected.items():
            if sample[field] != value:
                fail(
                    f"{key}.sample_protocol.{field}",
                    f"paper protocol requires {value!r}, got {sample[field]!r}",
                )
        total += sample["expected_total_api_calls"]
    if total != 16000:
        fail("emobench.sample_protocol", f"EA+EU must total 16000 calls, got {total}")
    if manifest is not None:
        expected_manifest = {
            "examples_per_task": 400,
            "option_permutations": 4,
            "repetitions_per_permutation": 5,
            "calls_per_task": 8000,
            "suite_calls": 16000,
        }
        if manifest.get("emobench_paper_protocol") != expected_manifest:
            fail("registry.emobench_paper_protocol", f"must equal {expected_manifest!r}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"invalid JSON {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc


def runner_task_keys() -> list[str]:
    completed = subprocess.run(
        ["node", str(ROOT / "run.mjs"), "list"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RegistryError(f"node run.mjs list failed: {completed.stderr.strip()}")
    keys = [line.split()[0] for line in completed.stdout.splitlines() if line.strip()]
    if not keys:
        raise RegistryError("node run.mjs list returned no task keys")
    return keys


def validate_registry(registry_path: Path, *, check_runner: bool = True) -> list[dict[str, Any]]:
    manifest = expect_object(
        load_json(registry_path),
        "registry",
        {
            "schema_version", "registry_id", "as_of", "expected_task_count",
            "expected_task_keys", "task_files", "metric_vocabulary", "emobench_paper_protocol",
        },
    )
    if manifest["schema_version"] != "1.0":
        fail("registry.schema_version", "must equal '1.0'")
    expect_string(manifest["registry_id"], "registry.registry_id")
    if not isinstance(manifest["as_of"], str) or not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", manifest["as_of"]):
        fail("registry.as_of", "must be YYYY-MM-DD")
    expected_count = manifest["expected_task_count"]
    if expected_count != 19:
        fail("registry.expected_task_count", "must equal 19")
    expected_keys = expect_unique_strings(manifest["expected_task_keys"], "registry.expected_task_keys")
    if len(expected_keys) != expected_count:
        fail("registry.expected_task_keys", f"expected {expected_count} keys, got {len(expected_keys)}")
    metric_vocabulary = expect_unique_strings(
        manifest["metric_vocabulary"], "registry.metric_vocabulary"
    )
    if metric_vocabulary != list(KNOWN_METRICS):
        fail("registry.metric_vocabulary", "must exactly match the canonical ordered metric vocabulary")
    task_files = expect_unique_strings(manifest["task_files"], "registry.task_files")
    tasks: list[Any] = []
    base = registry_path.parent
    for relative in task_files:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            fail("registry.task_files", f"unsafe relative path {relative!r}")
        records = load_json(base / path)
        if not isinstance(records, list):
            fail(str(base / path), "task file must contain a JSON array")
        tasks.extend(records)
    validated = validate_task_collection(tasks, expected_keys)
    validate_emobench_protocol(validated, manifest)
    if check_runner:
        live_keys = runner_task_keys()
        if live_keys != expected_keys:
            fail(
                "registry.expected_task_keys",
                f"does not match node run.mjs list; registry={expected_keys!r}, runner={live_keys!r}",
            )
    reject_private_or_secret_strings(manifest)
    return validated


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def hash_report(registry_path: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = load_json(registry_path)
    schema_path = registry_path.parent / "schema.json"
    return {
        "schema_sha256": sha256_bytes(schema_path.read_bytes()),
        "registry_index_sha256": sha256_bytes(registry_path.read_bytes()),
        "registry_bundle_sha256": canonical_sha256({"manifest": manifest, "tasks": tasks}),
        "task_spec_sha256": {
            task["task_key"]: canonical_sha256(task)
            for task in tasks
        },
    }


def fixture_task(key: str = "emobench-ea") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_key": key,
        "family": "emobench",
        "task_contract": copy_json(EXPECTED_TASK_CONTRACTS[key]),
        "dataset": {
            "root_env": "EVAL_DATASETS_DIR",
            "relative_root": "EmoBench/repo/data",
            "revision": {"kind": "git_commit", "value": "0" * 40, "frozen": True},
            "availability": {
                "status": "present", "access": "authorized_local_only", "redistribution": "aggregate_only"
            },
        },
        "use_policy": {
            "selection_role": "none",
            "test_selection_prohibited": True,
            "train": {
                "source_split": None, "availability": "not_registered", "allowed_uses": [],
                "consumption_status": "not_applicable", "notes": "",
            },
            "dev": {
                "source_split": None, "availability": "not_registered", "allowed_uses": [],
                "consumption_status": "not_applicable", "notes": "",
            },
            "test": {
                "source_split": "EA", "availability": "available", "allowed_uses": ["paper_replication"],
                "consumption_status": "exploratory_used", "notes": "",
            },
        },
        "metrics": {"primary": ["accuracy_by_language"], "secondary": ["accuracy"]},
        "sample_protocol": {
            "protocol_id": "emobench-paper-v1", "evaluation_role": "test", "source_split": "EA",
            "selection_forbidden": True, "mode": "paper_permutation_majority",
            "examples": {"kind": "exact", "value": 400}, "seed": None,
            "option_permutations": 4, "repetitions_per_permutation": 5,
            "api_calls_per_example": 20, "expected_total_api_calls": 8000,
            "scoring_unit": "question",
        },
        "provenance_requirements": {
            "model": {
                "required_fields": sorted(MODEL_PROVENANCE_MINIMUM), "frozen_before_test": True,
                "forbidden_aliases": ["deepseek-chat", "deepseek-reasoner"],
            },
            "prompt": {"required_fields": sorted(PROMPT_PROVENANCE_MINIMUM), "frozen_before_test": True},
            "parser": {"required_fields": sorted(PARSER_PROVENANCE_MINIMUM), "frozen_before_test": True},
            "dataset": {"required_fields": sorted(DATASET_PROVENANCE_MINIMUM), "frozen_before_test": True},
        },
        "paper_compatibility": {
            "status": "compatible", "reference_protocol": "synthetic selftest",
            "paper_primary_metrics": ["accuracy_by_language"], "blocking_differences": [],
        },
        "current_status": {
            "as_of": "2026-08-01", "workflow_state": "planned", "evidence_level": "selftest_only",
            "summary": "Synthetic fixture.", "next_action": "Run validation.",
        },
    }


def expect_rejected(name: str, mutate: Any) -> None:
    task = fixture_task()
    mutate(task)
    try:
        validate_task(task)
    except RegistryError:
        return
    raise AssertionError(f"selftest {name}: invalid fixture was accepted")


def run_selftest() -> None:
    checks = 0
    ea = fixture_task("emobench-ea")
    eu = fixture_task("emobench-eu")
    eu["sample_protocol"]["source_split"] = "EU"
    eu["use_policy"]["test"]["source_split"] = "EU"
    validate_task_collection([ea, eu])
    validate_emobench_protocol([ea, eu])
    checks += 1

    expect_rejected("missing-field", lambda task: task.pop("family"))
    checks += 1

    try:
        validate_task_collection([fixture_task(), fixture_task()])
    except RegistryError:
        checks += 1
    else:
        raise AssertionError("selftest duplicate-task: duplicate task key was accepted")

    expect_rejected(
        "unknown-metric", lambda task: task["metrics"]["primary"].append("made_up_score")
    )
    checks += 1

    expect_rejected(
        "test-selection",
        lambda task: task["use_policy"]["test"]["allowed_uses"].append("model_selection"),
    )
    checks += 1

    wrong = [fixture_task("emobench-ea"), fixture_task("emobench-eu")]
    wrong[1]["sample_protocol"]["source_split"] = "EU"
    wrong[1]["sample_protocol"]["repetitions_per_permutation"] = 4
    wrong[1]["sample_protocol"]["api_calls_per_example"] = 16
    wrong[1]["sample_protocol"]["expected_total_api_calls"] = 6400
    try:
        validate_task_collection(wrong)
        validate_emobench_protocol(wrong)
    except RegistryError:
        checks += 1
    else:
        raise AssertionError("selftest emobench-call-count: incorrect call count was accepted")

    imhi_records = load_json(ROOT / "benchmark-specs" / "tasks" / "imhi.json")
    screen_records = [
        task for task in imhi_records if task.get("task_key") in SPECIALIST_SCREEN_TASKS
    ]
    if len(screen_records) != 4:
        raise AssertionError("selftest supervised-profile: expected four screen records")
    for task in screen_records:
        validate_task(task)
    checks += 1

    bad_profile = copy_json(screen_records[0])
    bad_profile["supervised_classifier_profile"]["labels"].reverse()
    try:
        validate_task(bad_profile)
    except RegistryError:
        checks += 1
    else:
        raise AssertionError("selftest supervised-label-order: reversed labels were accepted")

    bad_deduplication = copy_json(screen_records[0])
    bad_deduplication["supervised_classifier_profile"]["cross_split_leakage_columns"] = [
        "question", "post"
    ]
    try:
        validate_task(bad_deduplication)
    except RegistryError:
        checks += 1
    else:
        raise AssertionError(
            "selftest supervised-cross-split-leakage: question+post identity was accepted"
        )

    bad_split_commitment = copy_json(screen_records[0])
    bad_split_commitment["supervised_classifier_profile"]["split_commitments"][
        "train_file_sha256"
    ] = "0" * 64
    try:
        validate_task(bad_split_commitment)
    except RegistryError:
        checks += 1
    else:
        raise AssertionError(
            "selftest supervised-split-commitment: changed train bytes were accepted"
        )

    def leak_dev_to_test(task: dict[str, Any]) -> None:
        task["use_policy"]["selection_role"] = "dev"
        task["use_policy"]["dev"] = {
            "source_split": "official-test.csv",
            "availability": "available",
            "allowed_uses": ["model_selection"],
            "consumption_status": "development_used",
            "notes": "synthetic negative fixture",
        }

    expect_rejected("dev-test-source", leak_dev_to_test)
    checks += 1

    def available_without_source(task: dict[str, Any]) -> None:
        task["use_policy"]["train"] = {
            "source_split": None,
            "availability": "available",
            "allowed_uses": ["training"],
            "consumption_status": "unknown",
            "notes": "synthetic negative fixture",
        }

    expect_rejected("available-null-source", available_without_source)
    checks += 1

    def wrong_task_metric(task: dict[str, Any]) -> None:
        task["metrics"]["primary"] = ["multilabel_macro_f1"]

    expect_rejected("task-metric-mismatch", wrong_task_metric)
    checks += 1

    def reverse_contract_labels(task: dict[str, Any]) -> None:
        task["task_contract"]["canonical_output_labels"].reverse()

    expect_rejected("task-contract-label-order", reverse_contract_labels)
    checks += 1

    def wrong_contract_type(task: dict[str, Any]) -> None:
        task["task_contract"]["task_type"] = "single_label_classification"

    expect_rejected("task-contract-type", wrong_contract_type)
    checks += 1

    def leak_sample_to_test(task: dict[str, Any]) -> None:
        task["use_policy"]["selection_role"] = "dev"
        task["use_policy"]["dev"] = {
            "source_split": "development.csv",
            "availability": "available",
            "allowed_uses": ["model_selection"],
            "consumption_status": "development_used",
            "notes": "synthetic negative fixture",
        }
        task["sample_protocol"]["evaluation_role"] = "dev"
        task["sample_protocol"]["source_split"] = "official-test.csv"
        task["sample_protocol"]["selection_forbidden"] = False

    expect_rejected("sample-dev-test-source", leak_sample_to_test)
    checks += 1

    def mismatch_sample_role_source(task: dict[str, Any]) -> None:
        task["use_policy"]["selection_role"] = "dev"
        task["use_policy"]["dev"] = {
            "source_split": "development.csv",
            "availability": "available",
            "allowed_uses": ["model_selection"],
            "consumption_status": "development_used",
            "notes": "synthetic negative fixture",
        }
        task["sample_protocol"]["evaluation_role"] = "dev"
        task["sample_protocol"]["source_split"] = "other-development.csv"
        task["sample_protocol"]["selection_forbidden"] = False

    expect_rejected("sample-role-source-mismatch", mismatch_sample_role_source)
    checks += 1

    if checks != 17:
        raise AssertionError(f"selftest accounting failure: expected 17 checks, got {checks}")
    print("benchmark registry selftest PASS: 17/17 deterministic checks")


def copy_json(value: Any) -> Any:
    """Return a JSON-semantic deep copy without importing non-stdlib helpers."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--selftest", action="store_true", help="run synthetic positive and negative checks")
    parser.add_argument(
        "--skip-runner-check",
        action="store_true",
        help="validate files without comparing against `node run.mjs list`",
    )
    parser.add_argument(
        "--print-hashes",
        action="store_true",
        help="after validation, print canonical registry and per-task SHA-256 commitments as JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.selftest:
            run_selftest()
            return 0
        tasks = validate_registry(args.registry.resolve(), check_runner=not args.skip_runner_check)
    except (RegistryError, AssertionError) as exc:
        print(f"benchmark registry FAIL: {exc}", file=sys.stderr)
        return 1
    if args.print_hashes:
        print(json.dumps(hash_report(args.registry.resolve(), tasks), ensure_ascii=False, indent=2))
    else:
        print(
            f"benchmark registry PASS: {len(tasks)}/19 tasks; schema=1.0; "
            "runner keys aligned; no dataset read or API call"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
