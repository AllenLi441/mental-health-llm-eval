#!/usr/bin/env python3
"""Offline, aggregate-only analysis for PsySUICIDE accuracy-first v2.

The analyzer has two prespecified phases:

* ``screen`` validates the four Seed-42 aggregate files and their private
  numeric diagnostics, ranks A/B/C/D by the frozen accuracy-first rule,
  applies the frozen delta guards against D, and freezes at most one non-D
  challenger.
* ``confirm`` validates the selected challenger and D on Seeds 43/44/45,
  verifies each private numeric diagnostics archive without exposing its
  rows, and runs the preregistered paired hierarchical bootstrap.

No dataset file, licensed text, source id, holdout artifact, model, checkpoint,
or network resource is opened.  Output is aggregate-only.  The program writes
nothing unless ``--execute`` is supplied, and it never overwrites an output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
TRAINER_PATH = ROOT / "scripts" / "train_psysuicide_roberta_accuracy_v2.py"
DEFAULT_PREREG = (
    ROOT / "reports" / "psysuicide-roberta-v2-accuracy-first.prereg.draft.json"
)
DEFAULT_RESULTS_ROOT = ROOT / "results" / "psysuicide-roberta-v2-accuracy-first"

EXPERIMENT_ID = "psysuicide-roberta-v2-accuracy-first"
SCREEN_SEED = 42
CONFIRM_SEEDS = (43, 44, 45)
ARMS = ("A", "B", "C", "D")
CONTROL_ARM = "D"
ARM_DEFINITIONS = {
    "A": {"weighted_sampler": False, "weighted_focal": False},
    "B": {"weighted_sampler": True, "weighted_focal": False},
    "C": {"weighted_sampler": False, "weighted_focal": True},
    "D": {"weighted_sampler": True, "weighted_focal": True},
}
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
CRITICAL_LABELS = (
    "主动自杀意图",
    "自杀计划",
    "自杀准备行为",
    "自杀未遂",
    "自伤意图",
    "自伤行为",
)
EXPECTED_OPTIMIZATION_ROWS = 9_342
EXPECTED_TRAIN_ROWS = 7_479
EXPECTED_INNER_DEV_ROWS = 1_863
EXPECTED_OPTIMIZATION_COMMITMENT = (
    "0c3cbf98db02c609d500b4f9ef9bf95518b617bd4dffd1f4cde8a5c5e85a8dd9"
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_072_945

SCREEN_THRESHOLDS = {
    "accuracy_delta": 0.002,
    "macro_f1_delta": -0.03,
    "pooled_critical_risk_recall_delta": -0.03,
    "stable_critical_class_recall_delta": -0.10,
}
CONFIRM_THRESHOLDS = {
    "mean_accuracy_delta": 0.005,
    "seed_wins": 2,
    "accuracy_ci95_lower_strict": 0.0,
    "macro_f1_ci95_lower": -0.02,
    "pooled_critical_risk_recall_ci95_lower": -0.02,
    "stable_critical_class_mean_recall_delta": -0.05,
}

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
FORBIDDEN_AGGREGATE_KEYS = {
    "text",
    "texts",
    "raw",
    "idx",
    "id",
    "source_id",
    "source_ids",
    "row_digest",
    "row_digests",
    "gold",
    "prediction",
    "predictions",
    "predicted",
    "logits",
    "probabilities",
    "confusion",
    "confusion_matrix",
    "class_weights",
    "weights",
    "optimizer_state",
    "trainer_state",
}
EXPECTED_DIAGNOSTIC_KEYS = {
    "logits",
    "probabilities",
    "gold",
    "predicted",
    "row_digests",
    "confusion_matrix",
    "nll",
    "multiclass_brier",
    "ece_15_equal_width",
}


class ValidationError(ValueError):
    """Raised when an input fails a frozen protocol or privacy gate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_commitment(digests: Iterable[str]) -> str:
    return sha256_bytes("\n".join(sorted(digests)).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_ignored_untracked_results_path(path: Path, context: str) -> Path:
    resolved = path.resolve(strict=True)
    results_root = (ROOT / "results").resolve(strict=True)
    require(
        resolved == results_root or results_root in resolved.parents,
        f"{context} must stay inside repository results/",
    )
    relative = str(resolved.relative_to(ROOT.resolve()))
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", relative],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    require(ignored, f"{context} is not ignored by Git")
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", relative],
        cwd=ROOT,
        text=True,
    ).strip()
    require(not tracked, f"{context} contains tracked private result files")
    return resolved


def strict_json_load(path: Path, description: str) -> dict[str, Any]:
    require(path.exists(), f"missing {description}")
    require(path.is_file() and not path.is_symlink(), f"invalid {description} file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate key in {description}")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValidationError(f"non-finite number in {description}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid UTF-8 JSON in {description}") from error
    require(isinstance(payload, dict), f"{description} must be a JSON object")
    return payload


def finite_number(value: Any, field: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{field} must be finite",
    )
    return float(value)


def exact_integer(value: Any, expected: int, field: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        f"{field} mismatch",
    )


def close(actual: float, expected: float, field: str, tolerance: float = 1e-10) -> None:
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{field} mismatch",
    )


def require_hex(value: Any, pattern: re.Pattern[str], field: str) -> str:
    require(isinstance(value, str) and bool(pattern.fullmatch(value)), f"invalid {field}")
    return value


def reject_row_payload(value: Any, context: str) -> None:
    """Reject row-level/sensitive keys before an aggregate can be reported."""
    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), f"invalid key in {context}")
            if key.lower() in FORBIDDEN_AGGREGATE_KEYS:
                raise ValidationError(f"row-level or private field forbidden in {context}")
            reject_row_payload(child, context)
    elif isinstance(value, list):
        for child in value:
            reject_row_payload(child, context)


def require_frozen_prereg(path: Path) -> tuple[dict[str, Any], str]:
    prereg = strict_json_load(path, "preregistration")
    exact_integer(prereg.get("schema_version"), 1, "preregistration schema_version")
    require(prereg.get("experiment_id") == EXPERIMENT_ID, "experiment identity mismatch")
    require(
        prereg.get("status") == "FROZEN" and prereg.get("training_allowed") is True,
        "analysis requires status=FROZEN and training_allowed=true",
    )
    require(tuple(prereg.get("label_order", ())) == LABELS, "label order mismatch")

    source = prereg.get("source_contract")
    require(isinstance(source, dict), "missing source contract")
    exact_integer(
        source.get("required_rows"),
        EXPECTED_OPTIMIZATION_ROWS,
        "optimization row count",
    )
    require(
        source.get("required_optimization_commitment_sha256")
        == EXPECTED_OPTIMIZATION_COMMITMENT,
        "optimization commitment mismatch",
    )

    inner = prereg.get("inner_split")
    require(isinstance(inner, dict), "missing inner split")
    expected_rows = inner.get("expected_rows")
    require(isinstance(expected_rows, dict), "missing inner split row counts")
    exact_integer(expected_rows.get("optimization"), EXPECTED_OPTIMIZATION_ROWS, "optimization rows")
    exact_integer(expected_rows.get("train"), EXPECTED_TRAIN_ROWS, "inner-train rows")
    exact_integer(expected_rows.get("inner_dev"), EXPECTED_INNER_DEV_ROWS, "inner-dev rows")
    require(inner.get("seed") == "psysuicide-accuracy-first-v2-inner-dev-2026-07-29", "inner split seed mismatch")
    require(inner.get("split_id") == "psysuicide-optimization-inner-dev-v2", "inner split id mismatch")
    require_hex(inner.get("exact_train_commitment_sha256"), HEX64, "inner-train commitment")
    require_hex(inner.get("exact_inner_dev_commitment_sha256"), HEX64, "inner-dev commitment")
    per_label_commitments = inner.get(
        "exact_inner_dev_per_label_commitment_sha256"
    )
    require(
        isinstance(per_label_commitments, dict)
        and tuple(per_label_commitments) == LABELS,
        "inner-dev per-label commitment map mismatch",
    )
    for label in LABELS:
        require_hex(
            per_label_commitments[label],
            HEX64,
            f"inner-dev commitment for {label}",
        )
    per_label = inner.get("expected_per_label")
    require(isinstance(per_label, dict) and tuple(per_label) == LABELS, "inner split label counts mismatch")
    for label in LABELS:
        counts = per_label[label]
        require(isinstance(counts, dict), "invalid per-label split counts")
        for partition in ("optimization", "train", "inner_dev"):
            require(
                isinstance(counts.get(partition), int)
                and not isinstance(counts.get(partition), bool)
                and counts[partition] > 0,
                "invalid per-label split count",
            )
        require(
            counts["optimization"] == counts["train"] + counts["inner_dev"],
            "per-label split count does not conserve rows",
        )
    require(
        sum(per_label[label]["optimization"] for label in LABELS)
        == EXPECTED_OPTIMIZATION_ROWS,
        "per-label optimization counts mismatch",
    )
    require(
        sum(per_label[label]["train"] for label in LABELS) == EXPECTED_TRAIN_ROWS,
        "per-label train counts mismatch",
    )
    require(
        sum(per_label[label]["inner_dev"] for label in LABELS)
        == EXPECTED_INNER_DEV_ROWS,
        "per-label inner-dev counts mismatch",
    )

    phases = prereg.get("phases")
    require(isinstance(phases, dict), "missing phase definitions")
    screen = phases.get("screen")
    confirmation = phases.get("confirmation")
    require(isinstance(screen, dict) and isinstance(confirmation, dict), "missing screen or confirmation phase")
    exact_integer(screen.get("seed"), SCREEN_SEED, "screen seed")
    require(tuple(screen.get("arms", ())) == ARMS, "screen arms mismatch")
    require(
        tuple(screen.get("ranking", ()))
        == (
            "highest inner-dev accuracy",
            "highest macro-F1",
            "highest pooled critical-risk recall",
            "fixed arm order A, B, C, D",
        ),
        "screen ranking rule mismatch",
    )
    require(tuple(confirmation.get("seeds", ())) == CONFIRM_SEEDS, "confirmation seeds mismatch")
    require(
        confirmation.get("screen_seed_reused_in_confirmatory_mean") is False,
        "screen seed must be excluded from confirmation",
    )

    definitions = prereg.get("metric_definitions")
    require(isinstance(definitions, dict), "missing metric definitions")
    require(
        tuple(definitions.get("critical_risk_labels", ())) == CRITICAL_LABELS,
        "critical-risk labels mismatch",
    )

    rules = prereg.get("decision_rules")
    require(isinstance(rules, dict), "missing decision rules")
    screen_rules = rules.get("screen")
    confirm_rules = rules.get("confirmation")
    require(isinstance(screen_rules, dict) and isinstance(confirm_rules, dict), "missing frozen decision rules")
    frozen_screen = {
        "minimum_accuracy_delta_vs_D": SCREEN_THRESHOLDS["accuracy_delta"],
        "minimum_macro_f1_delta_vs_D": SCREEN_THRESHOLDS["macro_f1_delta"],
        "minimum_pooled_critical_risk_recall_delta_vs_D": SCREEN_THRESHOLDS[
            "pooled_critical_risk_recall_delta"
        ],
        "minimum_each_stable_critical_class_recall_delta_vs_D": SCREEN_THRESHOLDS[
            "stable_critical_class_recall_delta"
        ],
    }
    for field, expected in frozen_screen.items():
        close(finite_number(screen_rules.get(field), field), expected, field)
    frozen_confirm = {
        "minimum_mean_accuracy_delta_vs_D": CONFIRM_THRESHOLDS[
            "mean_accuracy_delta"
        ],
        "accuracy_delta_ci95_lower_bound_must_be_greater_than": CONFIRM_THRESHOLDS[
            "accuracy_ci95_lower_strict"
        ],
        "macro_f1_delta_ci95_lower_bound_must_be_at_least": CONFIRM_THRESHOLDS[
            "macro_f1_ci95_lower"
        ],
        "pooled_critical_risk_recall_delta_ci95_lower_bound_must_be_at_least": CONFIRM_THRESHOLDS[
            "pooled_critical_risk_recall_ci95_lower"
        ],
        "minimum_each_stable_critical_class_mean_recall_delta_vs_D": CONFIRM_THRESHOLDS[
            "stable_critical_class_mean_recall_delta"
        ],
    }
    for field, expected in frozen_confirm.items():
        close(finite_number(confirm_rules.get(field), field), expected, field)
    exact_integer(
        confirm_rules.get("minimum_seed_wins_out_of_3"),
        CONFIRM_THRESHOLDS["seed_wins"],
        "minimum seed wins",
    )
    require(confirm_rules.get("all_conditions_required") is True, "confirmation must require every gate")

    bootstrap = prereg.get("paired_hierarchical_bootstrap")
    require(isinstance(bootstrap, dict), "missing paired hierarchical bootstrap")
    exact_integer(bootstrap.get("replicates"), BOOTSTRAP_REPLICATES, "bootstrap replicates")
    exact_integer(bootstrap.get("rng_seed"), BOOTSTRAP_SEED, "bootstrap RNG seed")

    protocol = prereg.get("fixed_training_protocol")
    require(isinstance(protocol, dict), "missing fixed training protocol")
    close(finite_number(protocol.get("epochs"), "epochs"), 10.0, "epochs")
    for field in (
        "evaluate_each_epoch",
        "save_each_epoch",
        "load_best_model_at_end",
        "greater_is_better",
    ):
        require(protocol.get(field) is True, f"fixed protocol {field} mismatch")
    require(protocol.get("checkpoint_selection_metric") == "inner-dev accuracy", "checkpoint selection metric mismatch")
    require(protocol.get("early_stopping") is False, "early stopping must remain disabled")
    frozen_integer_protocol = {
        "max_length": 256,
        "per_device_train_batch": 2,
        "per_device_eval_batch": 8,
        "gradient_accumulation": 8,
    }
    for field, expected in frozen_integer_protocol.items():
        exact_integer(protocol.get(field), expected, f"fixed protocol {field}")
    frozen_float_protocol = {
        "learning_rate": 0.00002,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "focal_gamma": 1.5,
    }
    for field, expected in frozen_float_protocol.items():
        close(
            finite_number(protocol.get(field), f"fixed protocol {field}"),
            expected,
            f"fixed protocol {field}",
        )
    require(protocol.get("scheduler") == "linear", "fixed scheduler mismatch")
    require(protocol.get("optimizer") == "adamw_torch", "fixed optimizer mismatch")
    require(
        protocol.get("class_weight_formula")
        == "inverse fourth-root train-class frequency, normalized to mean 1.0 and clipped to [0.35,4.0]",
        "fixed class-weight formula mismatch",
    )
    require(
        protocol.get("best_checkpoint_ties")
        == "on an exact unrounded accuracy tie, retain the earlier epoch; secondary metrics never select a within-run checkpoint",
        "fixed checkpoint tie rule mismatch",
    )

    arms = prereg.get("arms")
    require(isinstance(arms, dict) and tuple(arms) == ARMS, "frozen arm declarations mismatch")
    require(
        arms["A"]
        == {
            "sampling": "natural shuffled sampling without replacement",
            "loss": "ordinary cross-entropy",
        },
        "frozen Arm A mismatch",
    )
    require(
        arms["B"]
        == {
            "sampling": "weighted replacement sampling using the fixed class-weight formula",
            "loss": "ordinary cross-entropy",
        },
        "frozen Arm B mismatch",
    )
    require(
        arms["C"]
        == {
            "sampling": "natural shuffled sampling without replacement",
            "loss": "class-weighted focal loss using the fixed class-weight formula and gamma=1.5",
        },
        "frozen Arm C mismatch",
    )
    require(
        arms["D"]
        == {
            "sampling": "weighted replacement sampling using the fixed class-weight formula",
            "loss": "class-weighted focal loss using the fixed class-weight formula and gamma=1.5",
            "role": "v1-style double-imbalance control",
        },
        "frozen Arm D mismatch",
    )

    implementation = prereg.get("implementation")
    require(isinstance(implementation, dict), "missing implementation identity")
    require(implementation.get("trainer_script") == str(TRAINER_PATH.relative_to(ROOT)), "trainer path mismatch")
    trainer_sha = require_hex(implementation.get("trainer_sha256"), HEX64, "trainer SHA-256")
    require(TRAINER_PATH.is_file(), "frozen trainer is missing")
    require(sha256_file(TRAINER_PATH) == trainer_sha, "current trainer bytes differ from preregistration")
    require(
        implementation.get("analyzer_script") == str(SCRIPT_PATH.relative_to(ROOT)),
        "analyzer path mismatch",
    )
    analyzer_sha = require_hex(
        implementation.get("analyzer_sha256"), HEX64, "analyzer SHA-256"
    )
    require(
        sha256_file(SCRIPT_PATH) == analyzer_sha,
        "current analyzer bytes differ from preregistration",
    )
    require_hex(
        implementation.get("execution_base_commit"),
        HEX40,
        "execution base commit",
    )
    require_hex(
        implementation.get("base_model_manifest_sha256"),
        HEX64,
        "base-model manifest SHA-256",
    )
    require(isinstance(implementation.get("base_model"), str), "missing base-model identity")
    require_hex(implementation.get("base_model_revision"), HEX40, "base-model revision")
    return prereg, sha256_file(path)


def normalized_metric_block(
    metrics: Any,
    expected_supports: dict[str, int],
    context: str,
) -> dict[str, Any]:
    require(isinstance(metrics, dict), f"missing metrics in {context}")
    require(
        set(metrics)
        == {
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "pooled_critical_risk_recall",
            "per_class",
        },
        f"metric fields mismatch in {context}",
    )
    per_class = metrics.get("per_class")
    require(
        isinstance(per_class, dict) and tuple(per_class) == LABELS,
        f"per-class label order mismatch in {context}",
    )
    normalized_per_class: dict[str, dict[str, Any]] = {}
    true_positives: dict[str, int] = {}
    for label in LABELS:
        values = per_class[label]
        require(
            isinstance(values, dict)
            and set(values) == {"precision", "recall", "f1", "support"},
            f"per-class fields mismatch in {context}",
        )
        support = values.get("support")
        exact_integer(support, expected_supports[label], f"support in {context}")
        precision = finite_number(values.get("precision"), f"precision in {context}")
        recall = finite_number(values.get("recall"), f"recall in {context}")
        f1 = finite_number(values.get("f1"), f"F1 in {context}")
        require(
            all(0.0 <= value <= 1.0 for value in (precision, recall, f1)),
            f"out-of-range per-class metric in {context}",
        )
        expected_f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        close(f1, expected_f1, f"per-class F1 in {context}", tolerance=2e-9)
        raw_tp = recall * support
        tp = int(round(raw_tp))
        close(raw_tp, float(tp), f"recall/support count in {context}", tolerance=2e-8)
        require(0 <= tp <= support, f"invalid true-positive count in {context}")
        if precision > 0.0:
            predicted_count = tp / precision
            close(
                predicted_count,
                float(round(predicted_count)),
                f"precision/count in {context}",
                tolerance=2e-7,
            )
        true_positives[label] = tp
        normalized_per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = sum(expected_supports.values())
    accuracy = finite_number(metrics.get("accuracy"), f"accuracy in {context}")
    macro_f1 = finite_number(metrics.get("macro_f1"), f"macro-F1 in {context}")
    weighted_f1 = finite_number(metrics.get("weighted_f1"), f"weighted-F1 in {context}")
    pooled = finite_number(
        metrics.get("pooled_critical_risk_recall"),
        f"pooled critical-risk recall in {context}",
    )
    require(
        all(0.0 <= value <= 1.0 for value in (accuracy, macro_f1, weighted_f1, pooled)),
        f"out-of-range aggregate metric in {context}",
    )
    close(
        accuracy,
        sum(true_positives.values()) / total,
        f"accuracy recomputation in {context}",
        tolerance=2e-9,
    )
    close(
        macro_f1,
        sum(normalized_per_class[label]["f1"] for label in LABELS) / len(LABELS),
        f"macro-F1 recomputation in {context}",
        tolerance=2e-9,
    )
    close(
        weighted_f1,
        sum(
            normalized_per_class[label]["f1"] * expected_supports[label]
            for label in LABELS
        )
        / total,
        f"weighted-F1 recomputation in {context}",
        tolerance=2e-9,
    )
    critical_support = sum(expected_supports[label] for label in CRITICAL_LABELS)
    close(
        pooled,
        sum(true_positives[label] for label in CRITICAL_LABELS) / critical_support,
        f"pooled critical-risk recall recomputation in {context}",
        tolerance=2e-9,
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "pooled_critical_risk_recall": pooled,
        "per_class": normalized_per_class,
    }


def expected_supports(prereg: dict[str, Any]) -> dict[str, int]:
    per_label = prereg["inner_split"]["expected_per_label"]
    return {label: int(per_label[label]["inner_dev"]) for label in LABELS}


def expected_commitments(prereg: dict[str, Any]) -> dict[str, str]:
    return {
        "optimization_commitment_sha256": prereg["source_contract"]
        ["required_optimization_commitment_sha256"],
        "train_commitment_sha256": prereg["inner_split"]
        ["exact_train_commitment_sha256"],
        "inner_dev_commitment_sha256": prereg["inner_split"]
        ["exact_inner_dev_commitment_sha256"],
    }


def validate_split_manifest(split: Any, prereg: dict[str, Any], context: str) -> None:
    require(isinstance(split, dict), f"missing split manifest in {context}")
    require(
        set(split)
        == {
            "schema_version",
            "split_id",
            "split_seed",
            "optimization_rows",
            "train_rows",
            "inner_dev_rows",
            "optimization_commitment_sha256",
            "train_commitment_sha256",
            "inner_dev_commitment_sha256",
            "inner_dev_per_label_commitment_sha256",
            "overlap_rows",
            "per_label",
            "publishing_boundary",
        },
        f"split manifest fields mismatch in {context}",
    )
    exact_integer(split.get("schema_version"), 1, f"split schema in {context}")
    require(split.get("split_id") == prereg["inner_split"]["split_id"], f"split id mismatch in {context}")
    require(split.get("split_seed") == prereg["inner_split"]["seed"], f"split seed mismatch in {context}")
    exact_integer(split.get("optimization_rows"), EXPECTED_OPTIMIZATION_ROWS, f"optimization rows in {context}")
    exact_integer(split.get("train_rows"), EXPECTED_TRAIN_ROWS, f"train rows in {context}")
    exact_integer(split.get("inner_dev_rows"), EXPECTED_INNER_DEV_ROWS, f"inner-dev rows in {context}")
    exact_integer(split.get("overlap_rows"), 0, f"split overlap in {context}")
    commitments = expected_commitments(prereg)
    for field, expected in commitments.items():
        require(split.get(field) == expected, f"{field} mismatch in {context}")
    require(
        split.get("inner_dev_per_label_commitment_sha256")
        == prereg["inner_split"]["exact_inner_dev_per_label_commitment_sha256"],
        f"per-label inner-dev commitments mismatch in {context}",
    )
    declared = split.get("per_label")
    require(isinstance(declared, dict) and tuple(declared) == LABELS, f"split per-label order mismatch in {context}")
    for label in LABELS:
        require(
            declared[label] == prereg["inner_split"]["expected_per_label"][label],
            f"split per-label counts mismatch in {context}",
        )
    require(
        split.get("publishing_boundary")
        == (
            "aggregate counts and commitments only; membership, row digests, "
            "text and source ids remain private"
        ),
        f"split publishing boundary mismatch in {context}",
    )


def validate_identity(
    identity: Any,
    prereg: dict[str, Any],
    prereg_sha: str,
    stage: str,
    arm: str,
    seed: int,
    context: str,
) -> dict[str, Any]:
    require(isinstance(identity, dict), f"missing run identity in {context}")
    require(
        set(identity)
        == {
            "schema_version",
            "stage",
            "arm",
            "seed",
            "head",
            "upstream",
            "live_remote_head",
            "execution_base_commit",
            "trainer_sha256",
            "analyzer_sha256",
            "prereg_sha256",
            "optimization_commitment_sha256",
            "train_commitment_sha256",
            "inner_dev_commitment_sha256",
            "base_model",
            "base_model_revision",
            "base_model_manifest_sha256",
            "environment",
        },
        f"run identity fields mismatch in {context}",
    )
    exact_integer(identity.get("schema_version"), 1, f"identity schema in {context}")
    require(identity.get("stage") == stage, f"identity stage mismatch in {context}")
    require(identity.get("arm") == arm, f"identity arm mismatch in {context}")
    exact_integer(identity.get("seed"), seed, f"identity seed in {context}")
    implementation = prereg["implementation"]
    commitments = expected_commitments(prereg)
    expected = {
        "execution_base_commit": implementation["execution_base_commit"],
        "trainer_sha256": implementation["trainer_sha256"],
        "analyzer_sha256": implementation["analyzer_sha256"],
        "prereg_sha256": prereg_sha,
        **commitments,
        "base_model": implementation["base_model"],
        "base_model_revision": implementation["base_model_revision"],
        "base_model_manifest_sha256": implementation["base_model_manifest_sha256"],
    }
    for field, value in expected.items():
        require(identity.get(field) == value, f"run identity {field} mismatch in {context}")
    for field in ("head", "upstream", "live_remote_head"):
        require_hex(identity.get(field), HEX40, f"{field} in {context}")
    require(
        identity["head"] == identity["upstream"] == identity["live_remote_head"],
        f"local/upstream/remote identity mismatch in {context}",
    )
    environment = identity.get("environment")
    require(
        isinstance(environment, dict)
        and set(environment)
        == {
            "python",
            "platform",
            "cpu_forced",
            "mps_available",
            "cuda_available",
            "expected_device",
            "packages",
        },
        f"environment identity fields mismatch in {context}",
    )
    require(
        isinstance(environment["python"], str)
        and bool(environment["python"])
        and isinstance(environment["platform"], str)
        and bool(environment["platform"]),
        f"environment version identity missing in {context}",
    )
    for field in ("cpu_forced", "mps_available", "cuda_available"):
        require(isinstance(environment[field], bool), f"environment boolean mismatch in {context}")
    expected_device = (
        "cpu"
        if environment["cpu_forced"]
        else "mps"
        if environment["mps_available"]
        else "cuda"
        if environment["cuda_available"]
        else "cpu"
    )
    require(environment["expected_device"] == expected_device, f"environment device mismatch in {context}")
    packages = environment["packages"]
    require(
        isinstance(packages, dict)
        and set(packages)
        == {"torch", "transformers", "accelerate", "numpy", "scikit-learn"}
        and all(isinstance(value, str) and bool(value) for value in packages.values()),
        f"environment package identity mismatch in {context}",
    )
    return {
        **{field: str(value) for field, value in expected.items()},
        "head": identity["head"],
        "environment": copy.deepcopy(environment),
    }


def validate_aggregate(
    path: Path,
    prereg: dict[str, Any],
    prereg_sha: str,
    stage: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    context = f"{stage} aggregate {arm}/Seed-{seed}"
    aggregate = strict_json_load(path, context)
    reject_row_payload(aggregate, context)
    allowed_fields = {
        "schema_version",
        "experiment_id",
        "stage",
        "arm",
        "seed",
        "metric_claims_allowed",
        "train_rows",
        "inner_dev_rows",
        "epochs",
        "max_steps",
        "best_checkpoint",
        "best_inner_dev_accuracy",
        "best_checkpoint_manifest_sha256",
        "best_checkpoint_completion_sha256",
        "best_model_weight_sha256s",
        "metrics",
        "train_runtime_seconds",
        "train_loss",
        "identity",
        "split",
        "arm_definition",
        "publishing_boundary",
    }
    require(set(aggregate) == allowed_fields, f"aggregate fields mismatch in {context}")
    exact_integer(aggregate.get("schema_version"), 1, f"schema_version in {context}")
    require(aggregate.get("experiment_id") == EXPERIMENT_ID, f"experiment identity mismatch in {context}")
    require(aggregate.get("stage") == stage, f"stage mismatch in {context}")
    require(aggregate.get("arm") == arm, f"arm mismatch in {context}")
    exact_integer(aggregate.get("seed"), seed, f"seed in {context}")
    require(aggregate.get("metric_claims_allowed") is True, f"metric claims disabled in {context}")
    exact_integer(aggregate.get("train_rows"), EXPECTED_TRAIN_ROWS, f"train rows in {context}")
    exact_integer(aggregate.get("inner_dev_rows"), EXPECTED_INNER_DEV_ROWS, f"inner-dev rows in {context}")
    close(finite_number(aggregate.get("epochs"), f"epochs in {context}"), 10.0, f"epochs in {context}")
    exact_integer(aggregate.get("max_steps"), -1, f"max_steps in {context}")
    checkpoint = aggregate.get("best_checkpoint")
    require(
        isinstance(checkpoint, str) and bool(re.fullmatch(r"checkpoint-[1-9][0-9]*", checkpoint)),
        f"invalid best checkpoint in {context}",
    )
    best_checkpoint_manifest_sha = require_hex(
        aggregate.get("best_checkpoint_manifest_sha256"),
        HEX64,
        f"best-checkpoint manifest SHA-256 in {context}",
    )
    best_checkpoint_completion_sha = require_hex(
        aggregate.get("best_checkpoint_completion_sha256"),
        HEX64,
        f"checkpoint completion SHA-256 in {context}",
    )
    model_weight_sha256s = aggregate.get("best_model_weight_sha256s")
    require(
        isinstance(model_weight_sha256s, dict) and bool(model_weight_sha256s),
        f"missing best-model weight hashes in {context}",
    )
    for filename, digest in model_weight_sha256s.items():
        require(
            isinstance(filename, str)
            and Path(filename).name == filename
            and (
                filename.endswith(".safetensors")
                or (filename.startswith("pytorch_model") and filename.endswith(".bin"))
            ),
            f"invalid best-model weight filename in {context}",
        )
        require_hex(digest, HEX64, f"best-model weight SHA-256 in {context}")
    require(aggregate.get("arm_definition") == ARM_DEFINITIONS[arm], f"arm definition mismatch in {context}")
    require(
        aggregate.get("publishing_boundary")
        == (
            "aggregate-only candidate; weights, trainer state, row digests, "
            "logits, gold labels and confusion remain local and ignored"
        ),
        f"publishing boundary mismatch in {context}",
    )
    require(finite_number(aggregate.get("train_runtime_seconds"), f"runtime in {context}") >= 0.0, f"negative runtime in {context}")
    finite_number(aggregate.get("train_loss"), f"train loss in {context}")

    validate_split_manifest(aggregate.get("split"), prereg, context)
    identity = validate_identity(
        aggregate.get("identity"), prereg, prereg_sha, stage, arm, seed, context
    )
    metrics = normalized_metric_block(
        aggregate.get("metrics"), expected_supports(prereg), context
    )
    close(
        finite_number(aggregate.get("best_inner_dev_accuracy"), f"best accuracy in {context}"),
        metrics["accuracy"],
        f"best checkpoint accuracy in {context}",
        tolerance=2e-9,
    )
    return {
        "arm": arm,
        "seed": seed,
        "metrics": metrics,
        "aggregate_sha256": sha256_file(path),
        "identity": identity,
        "split": aggregate["split"],
        "best_checkpoint": checkpoint,
        "best_checkpoint_manifest_sha256": best_checkpoint_manifest_sha,
        "best_checkpoint_completion_sha256": best_checkpoint_completion_sha,
        "best_model_weight_sha256s": copy.deepcopy(model_weight_sha256s),
    }


def aggregate_path(run_dir: Path, stage: str, arm: str, seed: int) -> Path:
    return run_dir / f"{stage}-{arm}-seed-{seed}" / "aggregate.json"


def diagnostics_path(run_dir: Path, stage: str, arm: str, seed: int) -> Path:
    return run_dir / f"{stage}-{arm}-seed-{seed}" / "private-diagnostics.npz"


def require_common_phase_identity(records: Iterable[dict[str, Any]], context: str) -> None:
    records = list(records)
    require(bool(records), f"no records for {context}")
    invariant_fields = (
        "execution_base_commit",
        "trainer_sha256",
        "analyzer_sha256",
        "prereg_sha256",
        "optimization_commitment_sha256",
        "train_commitment_sha256",
        "inner_dev_commitment_sha256",
        "base_model",
        "base_model_revision",
        "base_model_manifest_sha256",
    )
    first = records[0]
    for record in records[1:]:
        for field in invariant_fields:
            require(
                record["identity"][field] == first["identity"][field],
                f"mixed {field} across {context}",
            )
        require(record["split"] == first["split"], f"mixed split manifest across {context}")
        require(
            record["identity"]["environment"]
            == first["identity"]["environment"],
            f"mixed execution environment across {context}",
        )


def stable_and_sparse_labels(supports: dict[str, int]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stable = tuple(label for label in CRITICAL_LABELS if supports[label] >= 10)
    sparse = tuple(label for label in CRITICAL_LABELS if supports[label] < 10)
    return stable, sparse


def analyze_screen(
    run_dir: Path,
    prereg: dict[str, Any],
    prereg_sha: str,
) -> dict[str, Any]:
    import numpy as np

    records = {
        arm: validate_aggregate(
            aggregate_path(run_dir, "screen", arm, SCREEN_SEED),
            prereg,
            prereg_sha,
            "screen",
            arm,
            SCREEN_SEED,
        )
        for arm in ARMS
    }
    diagnostics = {
        arm: validate_private_diagnostics(
            diagnostics_path(run_dir, "screen", arm, SCREEN_SEED),
            records[arm],
            prereg,
        )
        for arm in ARMS
    }
    reference = diagnostics[ARMS[0]]
    for arm in ARMS[1:]:
        require(
            np.array_equal(diagnostics[arm]["gold"], reference["gold"]),
            "screen diagnostics do not share identical gold labels",
        )
        require(
            np.array_equal(
                diagnostics[arm]["row_digests"], reference["row_digests"]
            ),
            "screen diagnostics do not share identical row digests",
        )
    require_common_phase_identity(records.values(), "screen aggregates")
    require(
        len({record["identity"]["head"] for record in records.values()}) == 1,
        "screen aggregates were not produced from one pushed HEAD",
    )

    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    ranking = sorted(
        ARMS,
        key=lambda arm: (
            -records[arm]["metrics"]["accuracy"],
            -records[arm]["metrics"]["macro_f1"],
            -records[arm]["metrics"]["pooled_critical_risk_recall"],
            arm_order[arm],
        ),
    )
    top_arm = ranking[0]
    candidate = top_arm if top_arm != CONTROL_ARM else None
    supports = expected_supports(prereg)
    stable, sparse = stable_and_sparse_labels(supports)
    deltas: dict[str, Any] | None = None
    gates: dict[str, Any]
    if candidate is None:
        gates = {
            "top_ranked_arm_is_non_D": False,
            "all_screen_guards_pass": False,
        }
    else:
        candidate_metrics = records[candidate]["metrics"]
        control_metrics = records[CONTROL_ARM]["metrics"]
        per_class_delta = {
            label: candidate_metrics["per_class"][label]["recall"]
            - control_metrics["per_class"][label]["recall"]
            for label in stable
        }
        deltas = {
            "candidate_minus_D": {
                "accuracy": candidate_metrics["accuracy"] - control_metrics["accuracy"],
                "macro_f1": candidate_metrics["macro_f1"] - control_metrics["macro_f1"],
                "pooled_critical_risk_recall": candidate_metrics[
                    "pooled_critical_risk_recall"
                ]
                - control_metrics["pooled_critical_risk_recall"],
                "stable_critical_class_recall": per_class_delta,
            }
        }
        observed = deltas["candidate_minus_D"]
        gate_results = {
            "accuracy_delta": observed["accuracy"]
            >= SCREEN_THRESHOLDS["accuracy_delta"],
            "macro_f1_delta": observed["macro_f1"]
            >= SCREEN_THRESHOLDS["macro_f1_delta"],
            "pooled_critical_risk_recall_delta": observed[
                "pooled_critical_risk_recall"
            ]
            >= SCREEN_THRESHOLDS["pooled_critical_risk_recall_delta"],
            "each_stable_critical_class_recall_delta": all(
                value
                >= SCREEN_THRESHOLDS["stable_critical_class_recall_delta"]
                for value in per_class_delta.values()
            ),
        }
        gates = {
            "top_ranked_arm_is_non_D": True,
            "thresholds": dict(SCREEN_THRESHOLDS),
            "results": gate_results,
            "all_screen_guards_pass": all(gate_results.values()),
        }
    confirmation_allowed = bool(
        candidate is not None and gates["all_screen_guards_pass"]
    )
    selected = candidate if confirmation_allowed else None
    decision = "ADVANCE_CONFIRMATION" if confirmation_allowed else "STOP_NO_SCREEN_EVIDENCE"
    public_metrics = {
        arm: {
            "aggregate_sha256": records[arm]["aggregate_sha256"],
            "private_diagnostics_sha256": diagnostics[arm][
                "diagnostics_sha256"
            ],
            "best_checkpoint": records[arm]["best_checkpoint"],
            "best_checkpoint_manifest_sha256": records[arm][
                "best_checkpoint_manifest_sha256"
            ],
            "best_checkpoint_completion_sha256": records[arm][
                "best_checkpoint_completion_sha256"
            ],
            "best_model_weight_sha256s": records[arm][
                "best_model_weight_sha256s"
            ],
            "accuracy": records[arm]["metrics"]["accuracy"],
            "macro_f1": records[arm]["metrics"]["macro_f1"],
            "weighted_f1": records[arm]["metrics"]["weighted_f1"],
            "pooled_critical_risk_recall": records[arm]["metrics"]
            ["pooled_critical_risk_recall"],
            "stable_critical_class_recall": {
                label: records[arm]["metrics"]["per_class"][label]["recall"]
                for label in stable
            },
        }
        for arm in ARMS
    }
    return {
        "schema_version": 1,
        "document_type": "psysuicide_roberta_accuracy_v2_screen_selection",
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN",
        "confirmation_allowed": confirmation_allowed,
        "prereg_sha256": prereg_sha,
        "analyzer_sha256": prereg["implementation"]["analyzer_sha256"],
        "screen_seed": SCREEN_SEED,
        "selected_challenger": selected,
        "control_arm": CONTROL_ARM,
        "decision": decision,
        "ranking": ranking,
        "ranking_rule": [
            "highest inner-dev accuracy",
            "highest macro-F1",
            "highest pooled critical-risk recall",
            "fixed arm order A, B, C, D",
        ],
        "commitments": expected_commitments(prereg),
        "screen_aggregate_sha256s": {
            arm: records[arm]["aggregate_sha256"] for arm in ARMS
        },
        "screen_private_diagnostics_sha256s": {
            arm: diagnostics[arm]["diagnostics_sha256"] for arm in ARMS
        },
        "screen_metrics": public_metrics,
        "deltas_vs_D": deltas,
        "guards": gates,
        "stable_critical_classes": {
            label: {"inner_dev_support": supports[label]} for label in stable
        },
        "sparse_critical_classes": {
            label: {
                "inner_dev_support": supports[label],
                "stable_per_class_claim_allowed": False,
            }
            for label in sparse
        },
        "screen_seed_reused_in_confirmation": False,
        "holdout_or_official_split_scored": False,
        "publishing_boundary": (
            "aggregate inner-dev selection only; no text, source ids, row digests, "
            "row-level labels or predictions, logits, confusion, weights, official "
            "valid/test, or frozen holdout"
        ),
    }


def validate_selection(
    path: Path,
    prereg_sha: str,
) -> tuple[dict[str, Any], str]:
    selection = strict_json_load(path, "screen selection artifact")
    reject_row_payload(selection, "screen selection artifact")
    exact_integer(selection.get("schema_version"), 1, "selection schema_version")
    require(selection.get("experiment_id") == EXPERIMENT_ID, "selection experiment mismatch")
    require(selection.get("status") == "FROZEN", "selection status is not FROZEN")
    require(selection.get("confirmation_allowed") is True, "selection does not allow confirmation")
    require(selection.get("prereg_sha256") == prereg_sha, "selection preregistration mismatch")
    require_hex(selection.get("analyzer_sha256"), HEX64, "selection analyzer SHA-256")
    exact_integer(selection.get("screen_seed"), SCREEN_SEED, "selection screen seed")
    challenger = selection.get("selected_challenger")
    require(challenger in ("A", "B", "C"), "selection has no eligible challenger")
    require(selection.get("control_arm") == CONTROL_ARM, "selection control arm mismatch")
    require(selection.get("decision") == "ADVANCE_CONFIRMATION", "selection decision mismatch")
    return selection, sha256_file(path)


def confusion_matrix(gold: Any, predicted: Any, label_count: int) -> Any:
    import numpy as np

    matrix = np.zeros((label_count, label_count), dtype=np.int64)
    np.add.at(matrix, (gold, predicted), 1)
    return matrix


def metrics_from_predictions(gold: Any, predicted: Any) -> dict[str, Any]:
    import numpy as np

    gold = np.asarray(gold, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    support = np.bincount(gold, minlength=len(LABELS)).astype(np.int64)
    predicted_count = np.bincount(predicted, minlength=len(LABELS)).astype(np.int64)
    true_positive = np.asarray(
        [np.sum((gold == index) & (predicted == index)) for index in range(len(LABELS))],
        dtype=np.int64,
    )
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(len(LABELS), dtype=np.float64),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(LABELS), dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * true_positive,
        support + predicted_count,
        out=np.zeros(len(LABELS), dtype=np.float64),
        where=(support + predicted_count) != 0,
    )
    critical_ids = [LABELS.index(label) for label in CRITICAL_LABELS]
    return {
        "accuracy": float(true_positive.sum() / len(gold)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.sum(f1 * support) / len(gold)),
        "pooled_critical_risk_recall": float(
            true_positive[critical_ids].sum() / support[critical_ids].sum()
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


def calibration_values(probabilities: Any, gold: Any, predicted: Any) -> dict[str, float]:
    import numpy as np

    probabilities = np.asarray(probabilities, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
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
    return {
        "nll": float(-np.log(clipped).mean()),
        "multiclass_brier": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "ece_15_equal_width": float(ece),
    }


def compare_metrics(
    aggregate_metrics: dict[str, Any],
    diagnostic_metrics: dict[str, Any],
    context: str,
) -> None:
    for metric in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "pooled_critical_risk_recall",
    ):
        close(
            aggregate_metrics[metric],
            diagnostic_metrics[metric],
            f"diagnostic {metric} in {context}",
            tolerance=2e-9,
        )
    for label in LABELS:
        for metric in ("precision", "recall", "f1"):
            close(
                aggregate_metrics["per_class"][label][metric],
                diagnostic_metrics["per_class"][label][metric],
                f"diagnostic per-class metric in {context}",
                tolerance=2e-9,
            )
        exact_integer(
            diagnostic_metrics["per_class"][label]["support"],
            aggregate_metrics["per_class"][label]["support"],
            f"diagnostic support in {context}",
        )


def validate_private_diagnostics(
    path: Path,
    aggregate_record: dict[str, Any],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    context = f"confirm diagnostics {aggregate_record['arm']}/Seed-{aggregate_record['seed']}"
    require(path.exists(), f"missing {context}")
    require(path.is_file() and not path.is_symlink(), f"invalid {context} file")
    require(
        stat.S_IMODE(path.stat().st_mode) == 0o600,
        f"{context} permissions must be exactly 0600",
    )
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValidationError(f"unable to open {context}") from error
    try:
        require(set(archive.files) == EXPECTED_DIAGNOSTIC_KEYS, f"diagnostic fields mismatch in {context}")
        logits = np.asarray(archive["logits"])
        probabilities = np.asarray(archive["probabilities"])
        gold = np.asarray(archive["gold"])
        predicted = np.asarray(archive["predicted"])
        row_digests = np.asarray(archive["row_digests"])
        recorded_confusion = np.asarray(archive["confusion_matrix"])
        nll = np.asarray(archive["nll"])
        brier = np.asarray(archive["multiclass_brier"])
        ece = np.asarray(archive["ece_15_equal_width"])
    except (KeyError, ValueError, TypeError) as error:
        raise ValidationError(f"invalid arrays in {context}") from error
    finally:
        archive.close()

    row_count = EXPECTED_INNER_DEV_ROWS
    class_count = len(LABELS)
    require(logits.shape == (row_count, class_count), f"logit shape mismatch in {context}")
    require(probabilities.shape == (row_count, class_count), f"probability shape mismatch in {context}")
    require(gold.shape == (row_count,), f"gold shape mismatch in {context}")
    require(predicted.shape == (row_count,), f"prediction shape mismatch in {context}")
    require(row_digests.shape == (row_count,), f"row-digest shape mismatch in {context}")
    require(recorded_confusion.shape == (class_count, class_count), f"confusion shape mismatch in {context}")
    require(nll.shape == brier.shape == ece.shape == (1,), f"calibration scalar shape mismatch in {context}")
    require(logits.dtype.kind == "f" and probabilities.dtype.kind == "f", f"diagnostic float dtype mismatch in {context}")
    require(gold.dtype.kind in "iu" and predicted.dtype.kind in "iu", f"diagnostic label dtype mismatch in {context}")
    require(recorded_confusion.dtype.kind in "iu", f"diagnostic confusion dtype mismatch in {context}")
    require(row_digests.dtype.kind in "SU", f"row-digest dtype mismatch in {context}")
    require(np.isfinite(logits).all(), f"non-finite logits in {context}")
    require(np.isfinite(probabilities).all(), f"non-finite probabilities in {context}")
    require(np.isfinite(nll).all() and np.isfinite(brier).all() and np.isfinite(ece).all(), f"non-finite calibration in {context}")
    require(((gold >= 0) & (gold < class_count)).all(), f"out-of-range gold labels in {context}")
    require(((predicted >= 0) & (predicted < class_count)).all(), f"out-of-range predictions in {context}")
    require((probabilities >= 0.0).all() and (probabilities <= 1.0).all(), f"out-of-range probabilities in {context}")
    require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6), f"probability rows do not sum to one in {context}")

    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    recomputed_probabilities = np.exp(shifted)
    recomputed_probabilities /= recomputed_probabilities.sum(axis=1, keepdims=True)
    require(
        np.allclose(probabilities, recomputed_probabilities, rtol=2e-5, atol=2e-6),
        f"probabilities do not match logits in {context}",
    )
    require(np.array_equal(predicted, logits.argmax(axis=1)), f"predictions do not match logits in {context}")
    require(
        np.array_equal(recorded_confusion, confusion_matrix(gold, predicted, class_count)),
        f"confusion matrix mismatch in {context}",
    )
    normalized_digest_values: list[str] = []
    for value in row_digests.tolist():
        if isinstance(value, bytes):
            try:
                normalized = value.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValidationError(f"invalid row digest in {context}") from error
        else:
            normalized = str(value)
        require(bool(HEX64.fullmatch(normalized)), f"invalid row digest in {context}")
        normalized_digest_values.append(normalized)
    normalized_digests = np.asarray(normalized_digest_values, dtype="U64")
    require(
        normalized_digests.shape == (row_count,),
        f"normalized row-digest shape mismatch in {context}",
    )
    require(len(np.unique(normalized_digests)) == row_count, f"duplicate row digest in {context}")
    require(
        digest_commitment(normalized_digest_values)
        == prereg["inner_split"]["exact_inner_dev_commitment_sha256"],
        f"row digests do not match the frozen inner-dev commitment in {context}",
    )
    expected_per_label_commitments = prereg["inner_split"][
        "exact_inner_dev_per_label_commitment_sha256"
    ]
    for label_id, label in enumerate(LABELS):
        label_digests = [
            normalized_digest_values[index]
            for index, gold_label in enumerate(gold.tolist())
            if int(gold_label) == label_id
        ]
        require(
            digest_commitment(label_digests)
            == expected_per_label_commitments[label],
            f"row-digest/gold-label mapping does not match the frozen commitment for {label} in {context}",
        )
    supports = expected_supports(prereg)
    observed_support = np.bincount(gold.astype(np.int64), minlength=class_count)
    require(
        all(int(observed_support[index]) == supports[label] for index, label in enumerate(LABELS)),
        f"diagnostic gold support mismatch in {context}",
    )
    diagnostic_metrics = metrics_from_predictions(gold, predicted)
    compare_metrics(aggregate_record["metrics"], diagnostic_metrics, context)
    calibration = calibration_values(probabilities, gold, predicted)
    close(float(nll[0]), calibration["nll"], f"NLL in {context}", tolerance=2e-6)
    close(float(brier[0]), calibration["multiclass_brier"], f"Brier score in {context}", tolerance=2e-6)
    close(float(ece[0]), calibration["ece_15_equal_width"], f"ECE in {context}", tolerance=2e-6)
    return {
        "gold": gold.astype(np.int16, copy=True),
        "predicted": predicted.astype(np.int16, copy=True),
        "row_digests": normalized_digests.copy(),
        "diagnostics_sha256": sha256_file(path),
    }


def paired_hierarchical_bootstrap(
    gold: Any,
    candidate_predictions: Any,
    control_predictions: Any,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Stratified paired row bootstrap nested inside seed resampling."""
    import numpy as np

    gold = np.asarray(gold, dtype=np.int64)
    candidate_predictions = np.asarray(candidate_predictions, dtype=np.int64)
    control_predictions = np.asarray(control_predictions, dtype=np.int64)
    require(gold.shape == (EXPECTED_INNER_DEV_ROWS,), "bootstrap gold shape mismatch")
    require(
        candidate_predictions.shape == control_predictions.shape
        == (len(CONFIRM_SEEDS), EXPECTED_INNER_DEV_ROWS),
        "bootstrap prediction shape mismatch",
    )
    require(replicates > 0 and batch_size > 0, "invalid bootstrap execution size")
    strata = [np.flatnonzero(gold == label_id) for label_id in range(len(LABELS))]
    support = np.asarray([len(indices) for indices in strata], dtype=np.float64)
    require((support > 0).all(), "bootstrap gold stratum is empty")
    critical_ids = np.asarray([LABELS.index(label) for label in CRITICAL_LABELS])
    critical_support = float(support[critical_ids].sum())
    rng = np.random.default_rng(rng_seed)
    seed_draws = rng.integers(0, len(CONFIRM_SEEDS), size=(replicates, len(CONFIRM_SEEDS)))
    accuracy_deltas = np.empty(replicates, dtype=np.float64)
    macro_f1_deltas = np.empty(replicates, dtype=np.float64)
    pooled_deltas = np.empty(replicates, dtype=np.float64)

    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        size = stop - start
        drawn = seed_draws[start:stop]
        accuracy_sum = np.zeros(size, dtype=np.float64)
        macro_sum = np.zeros(size, dtype=np.float64)
        pooled_sum = np.zeros(size, dtype=np.float64)
        rows = np.arange(size)
        for seed_slot in range(len(CONFIRM_SEEDS)):
            candidate_predicted_counts = np.zeros((size, len(LABELS)), dtype=np.int64)
            control_predicted_counts = np.zeros((size, len(LABELS)), dtype=np.int64)
            candidate_tp = np.zeros((size, len(LABELS)), dtype=np.int64)
            control_tp = np.zeros((size, len(LABELS)), dtype=np.int64)
            selected_seeds = drawn[:, seed_slot]
            for label_id, positions in enumerate(strata):
                sampled = positions[
                    rng.integers(0, len(positions), size=(size, len(positions)))
                ]
                candidate_values = candidate_predictions[
                    selected_seeds[:, None], sampled
                ]
                control_values = control_predictions[selected_seeds[:, None], sampled]
                repeated_rows = np.repeat(rows, len(positions))
                np.add.at(
                    candidate_predicted_counts,
                    (repeated_rows, candidate_values.ravel()),
                    1,
                )
                np.add.at(
                    control_predicted_counts,
                    (repeated_rows, control_values.ravel()),
                    1,
                )
                candidate_tp[:, label_id] = np.sum(
                    candidate_values == label_id, axis=1
                )
                control_tp[:, label_id] = np.sum(control_values == label_id, axis=1)
            accuracy_sum += (
                candidate_tp.sum(axis=1) - control_tp.sum(axis=1)
            ) / EXPECTED_INNER_DEV_ROWS
            candidate_f1 = np.divide(
                2.0 * candidate_tp,
                support[None, :] + candidate_predicted_counts,
                out=np.zeros_like(candidate_tp, dtype=np.float64),
                where=(support[None, :] + candidate_predicted_counts) != 0,
            )
            control_f1 = np.divide(
                2.0 * control_tp,
                support[None, :] + control_predicted_counts,
                out=np.zeros_like(control_tp, dtype=np.float64),
                where=(support[None, :] + control_predicted_counts) != 0,
            )
            macro_sum += candidate_f1.mean(axis=1) - control_f1.mean(axis=1)
            pooled_sum += (
                candidate_tp[:, critical_ids].sum(axis=1)
                - control_tp[:, critical_ids].sum(axis=1)
            ) / critical_support
        divisor = float(len(CONFIRM_SEEDS))
        accuracy_deltas[start:stop] = accuracy_sum / divisor
        macro_f1_deltas[start:stop] = macro_sum / divisor
        pooled_deltas[start:stop] = pooled_sum / divisor

    def interval(values: Any) -> list[float]:
        lower, upper = np.percentile(values, [2.5, 97.5], method="linear")
        return [float(lower), float(upper)]

    return {
        "replicates": replicates,
        "rng_seed": rng_seed,
        "confidence_interval": "two-sided percentile 95%",
        "percentile_method": "NumPy linear",
        "seed_resampling": "three confirmation seeds sampled with replacement",
        "row_resampling": (
            "paired positions sampled with replacement separately within each "
            "frozen gold-label stratum, preserving stratum size"
        ),
        "accuracy_delta_ci95": interval(accuracy_deltas),
        "macro_f1_delta_ci95": interval(macro_f1_deltas),
        "pooled_critical_risk_recall_delta_ci95": interval(pooled_deltas),
    }


def analyze_confirmation(
    run_dir: Path,
    prereg: dict[str, Any],
    prereg_sha: str,
    selection_path: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    import numpy as np

    selection, selection_sha = validate_selection(selection_path, prereg_sha)
    recomputed_selection = analyze_screen(run_dir, prereg, prereg_sha)
    require(
        selection == recomputed_selection,
        "screen selection artifact does not match the frozen screen aggregates",
    )
    candidate = selection["selected_challenger"]
    records: dict[str, dict[int, dict[str, Any]]] = {candidate: {}, CONTROL_ARM: {}}
    diagnostics: dict[str, dict[int, dict[str, Any]]] = {candidate: {}, CONTROL_ARM: {}}
    all_records: list[dict[str, Any]] = []
    reference_gold = None
    reference_digests = None
    for arm in (candidate, CONTROL_ARM):
        for seed in CONFIRM_SEEDS:
            record = validate_aggregate(
                aggregate_path(run_dir, "confirm", arm, seed),
                prereg,
                prereg_sha,
                "confirm",
                arm,
                seed,
            )
            private = validate_private_diagnostics(
                diagnostics_path(run_dir, "confirm", arm, seed), record, prereg
            )
            if reference_gold is None:
                reference_gold = private["gold"]
                reference_digests = private["row_digests"]
            else:
                require(np.array_equal(private["gold"], reference_gold), "confirmation diagnostics do not share identical gold labels")
                require(np.array_equal(private["row_digests"], reference_digests), "confirmation diagnostics do not share identical row digests")
            records[arm][seed] = record
            diagnostics[arm][seed] = private
            all_records.append(record)
    require_common_phase_identity(all_records, "confirmation aggregates")
    require(reference_gold is not None and reference_digests is not None, "confirmation diagnostics missing")

    candidate_predictions = np.stack(
        [diagnostics[candidate][seed]["predicted"] for seed in CONFIRM_SEEDS]
    )
    control_predictions = np.stack(
        [diagnostics[CONTROL_ARM][seed]["predicted"] for seed in CONFIRM_SEEDS]
    )
    bootstrap = paired_hierarchical_bootstrap(
        reference_gold,
        candidate_predictions,
        control_predictions,
        replicates=bootstrap_replicates,
        rng_seed=bootstrap_seed,
    )
    supports = expected_supports(prereg)
    stable, sparse = stable_and_sparse_labels(supports)

    def mean_metric(arm: str, metric: str) -> float:
        return float(
            np.mean([records[arm][seed]["metrics"][metric] for seed in CONFIRM_SEEDS])
        )

    candidate_mean = {
        metric: mean_metric(candidate, metric)
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "pooled_critical_risk_recall",
        )
    }
    control_mean = {
        metric: mean_metric(CONTROL_ARM, metric)
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "pooled_critical_risk_recall",
        )
    }
    point_delta = {
        metric: candidate_mean[metric] - control_mean[metric]
        for metric in candidate_mean
    }
    seed_accuracy_deltas = {
        str(seed): records[candidate][seed]["metrics"]["accuracy"]
        - records[CONTROL_ARM][seed]["metrics"]["accuracy"]
        for seed in CONFIRM_SEEDS
    }
    seed_wins = sum(value > 0.0 for value in seed_accuracy_deltas.values())
    stable_recall_delta = {
        label: float(
            np.mean(
                [
                    records[candidate][seed]["metrics"]["per_class"][label][
                        "recall"
                    ]
                    - records[CONTROL_ARM][seed]["metrics"]["per_class"][label][
                        "recall"
                    ]
                    for seed in CONFIRM_SEEDS
                ]
            )
        )
        for label in stable
    }
    gate_results = {
        "mean_accuracy_delta": point_delta["accuracy"]
        >= CONFIRM_THRESHOLDS["mean_accuracy_delta"],
        "seed_wins": seed_wins >= CONFIRM_THRESHOLDS["seed_wins"],
        "accuracy_ci95_lower_strictly_above_zero": bootstrap[
            "accuracy_delta_ci95"
        ][0]
        > CONFIRM_THRESHOLDS["accuracy_ci95_lower_strict"],
        "macro_f1_ci95_lower_noninferior": bootstrap["macro_f1_delta_ci95"][0]
        >= CONFIRM_THRESHOLDS["macro_f1_ci95_lower"],
        "pooled_critical_risk_recall_ci95_lower_noninferior": bootstrap[
            "pooled_critical_risk_recall_delta_ci95"
        ][0]
        >= CONFIRM_THRESHOLDS["pooled_critical_risk_recall_ci95_lower"],
        "each_stable_critical_class_mean_recall_noninferior": all(
            value
            >= CONFIRM_THRESHOLDS[
                "stable_critical_class_mean_recall_delta"
            ]
            for value in stable_recall_delta.values()
        ),
    }
    passed = all(gate_results.values())
    confirm_rules = prereg["decision_rules"]["confirmation"]
    decision = (
        confirm_rules["pass_label"] if passed else confirm_rules["failure_label"]
    )

    run_evidence = {
        arm: {
            str(seed): {
                "aggregate_sha256": records[arm][seed]["aggregate_sha256"],
                "private_diagnostics_sha256": diagnostics[arm][seed][
                    "diagnostics_sha256"
                ],
                "best_checkpoint": records[arm][seed]["best_checkpoint"],
                "best_checkpoint_manifest_sha256": records[arm][seed][
                    "best_checkpoint_manifest_sha256"
                ],
                "best_checkpoint_completion_sha256": records[arm][seed][
                    "best_checkpoint_completion_sha256"
                ],
                "best_model_weight_sha256s": records[arm][seed][
                    "best_model_weight_sha256s"
                ],
                "accuracy": records[arm][seed]["metrics"]["accuracy"],
                "macro_f1": records[arm][seed]["metrics"]["macro_f1"],
                "weighted_f1": records[arm][seed]["metrics"]["weighted_f1"],
                "pooled_critical_risk_recall": records[arm][seed]["metrics"]
                ["pooled_critical_risk_recall"],
            }
            for seed in CONFIRM_SEEDS
        }
        for arm in (candidate, CONTROL_ARM)
    }
    result = {
        "schema_version": 1,
        "document_type": "psysuicide_roberta_accuracy_v2_confirmation",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "prereg_sha256": prereg_sha,
        "analyzer_sha256": prereg["implementation"]["analyzer_sha256"],
        "selection_artifact_sha256": selection_sha,
        "selected_challenger": candidate,
        "control_arm": CONTROL_ARM,
        "screen_seed": SCREEN_SEED,
        "confirmation_seeds": list(CONFIRM_SEEDS),
        "screen_seed_reused_in_confirmatory_mean": False,
        "commitments": expected_commitments(prereg),
        "paired_identity_verified": {
            "same_row_digests_across_all_six_runs": True,
            "same_gold_labels_across_all_six_runs": True,
            "row_count": EXPECTED_INNER_DEV_ROWS,
            "row_level_values_published": False,
        },
        "runs": run_evidence,
        "three_seed_means": {
            "challenger": candidate_mean,
            "D": control_mean,
            "challenger_minus_D": point_delta,
        },
        "per_seed_accuracy_delta_challenger_minus_D": seed_accuracy_deltas,
        "challenger_accuracy_wins_out_of_3": seed_wins,
        "stable_critical_class_mean_recall_delta_challenger_minus_D": stable_recall_delta,
        "sparse_critical_classes": {
            label: {
                "inner_dev_support": supports[label],
                "stable_per_class_claim_allowed": False,
            }
            for label in sparse
        },
        "paired_hierarchical_bootstrap": bootstrap,
        "guards": {
            "thresholds": dict(CONFIRM_THRESHOLDS),
            "results": gate_results,
            "all_conditions_pass": passed,
        },
        "decision": decision,
        "claim": {
            "internal_inner_dev_improvement_allowed": passed,
            "equivalence_tested": False,
            "failure_is_equivalence": False,
            "external_or_holdout_claim_allowed": False,
        },
        "holdout_or_official_split_scored": False,
        "publishing_boundary": (
            "aggregate inner-dev confirmation only; private row digests, gold and "
            "prediction labels, logits, probabilities, confusion, calibration rows, "
            "text, source ids, checkpoints and weights are not included"
        ),
    }
    reject_row_payload(result, "confirmation output")
    return result


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), "refusing to overwrite output")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = -1
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if created and path.exists():
            path.unlink()
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def selftest_check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"selftest failure: {message}")


def synthetic_predictions(gold: Any, divisor: int, offset: int) -> Any:
    import numpy as np

    gold = np.asarray(gold, dtype=np.int16)
    predicted = gold.copy()
    for label_id in range(len(LABELS)):
        positions = np.flatnonzero(gold == label_id)
        wrong = positions[(np.arange(len(positions)) + offset) % divisor == 0]
        predicted[wrong] = (label_id + 1) % len(LABELS)
    return predicted


def synthetic_diagnostic_arrays(gold: Any, predicted: Any, digests: Any) -> dict[str, Any]:
    import numpy as np

    logits = np.full((len(gold), len(LABELS)), -4.0, dtype=np.float32)
    logits[np.arange(len(gold)), predicted] = 4.0
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities.astype(np.float32)
    calibration = calibration_values(probabilities, gold, predicted)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "gold": np.asarray(gold, dtype=np.int16),
        "predicted": np.asarray(predicted, dtype=np.int16),
        "row_digests": np.asarray(digests),
        "confusion_matrix": confusion_matrix(gold, predicted, len(LABELS)),
        "nll": np.asarray([calibration["nll"]], dtype=np.float64),
        "multiclass_brier": np.asarray(
            [calibration["multiclass_brier"]], dtype=np.float64
        ),
        "ece_15_equal_width": np.asarray(
            [calibration["ece_15_equal_width"]], dtype=np.float64
        ),
    }


def synthetic_split(prereg: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split_id": prereg["inner_split"]["split_id"],
        "split_seed": prereg["inner_split"]["seed"],
        "optimization_rows": EXPECTED_OPTIMIZATION_ROWS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "inner_dev_rows": EXPECTED_INNER_DEV_ROWS,
        **expected_commitments(prereg),
        "inner_dev_per_label_commitment_sha256": prereg["inner_split"]
        ["exact_inner_dev_per_label_commitment_sha256"],
        "overlap_rows": 0,
        "per_label": prereg["inner_split"]["expected_per_label"],
        "publishing_boundary": (
            "aggregate counts and commitments only; membership, row digests, "
            "text and source ids remain private"
        ),
    }


def synthetic_aggregate(
    prereg: dict[str, Any],
    prereg_sha: str,
    stage: str,
    arm: str,
    seed: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    implementation = prereg["implementation"]
    identity = {
        "schema_version": 1,
        "stage": stage,
        "arm": arm,
        "seed": seed,
        "head": "f" * 40,
        "upstream": "f" * 40,
        "live_remote_head": "f" * 40,
        "execution_base_commit": implementation["execution_base_commit"],
        "trainer_sha256": implementation["trainer_sha256"],
        "analyzer_sha256": implementation["analyzer_sha256"],
        "prereg_sha256": prereg_sha,
        **expected_commitments(prereg),
        "base_model": implementation["base_model"],
        "base_model_revision": implementation["base_model_revision"],
        "base_model_manifest_sha256": implementation[
            "base_model_manifest_sha256"
        ],
        "environment": {
            "python": "synthetic",
            "platform": "synthetic",
            "cpu_forced": True,
            "mps_available": False,
            "cuda_available": False,
            "expected_device": "cpu",
            "packages": {
                "torch": "synthetic",
                "transformers": "synthetic",
                "accelerate": "synthetic",
                "numpy": "synthetic",
                "scikit-learn": "synthetic",
            },
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "arm": arm,
        "seed": seed,
        "metric_claims_allowed": True,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "inner_dev_rows": EXPECTED_INNER_DEV_ROWS,
        "epochs": 10.0,
        "max_steps": -1,
        "best_checkpoint": "checkpoint-100",
        "best_inner_dev_accuracy": metrics["accuracy"],
        "best_checkpoint_manifest_sha256": "3" * 64,
        "best_checkpoint_completion_sha256": "4" * 64,
        "best_model_weight_sha256s": {"model.safetensors": "5" * 64},
        "metrics": metrics,
        "train_runtime_seconds": 1.0,
        "train_loss": 0.5,
        "identity": identity,
        "split": synthetic_split(prereg),
        "arm_definition": ARM_DEFINITIONS[arm],
        "publishing_boundary": (
            "aggregate-only candidate; weights, trainer state, row digests, "
            "logits, gold labels and confusion remain local and ignored"
        ),
    }


def selftest() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="psysuicide-v2-analyzer-selftest-") as directory:
        root = Path(directory)
        prereg = copy.deepcopy(
            strict_json_load(DEFAULT_PREREG, "public preregistration fixture")
        )
        prereg["status"] = "FROZEN"
        prereg["training_allowed"] = True
        prereg["implementation"]["trainer_sha256"] = sha256_file(TRAINER_PATH)
        prereg["implementation"]["analyzer_sha256"] = sha256_file(SCRIPT_PATH)
        prereg["implementation"]["execution_base_commit"] = "e" * 40
        supports = expected_supports(prereg)
        gold = np.concatenate(
            [np.full(supports[label], index, dtype=np.int16) for index, label in enumerate(LABELS)]
        )
        digests = np.asarray(
            [sha256_bytes(f"synthetic-row-{index}".encode()) for index in range(len(gold))]
        )
        prereg["inner_split"]["exact_train_commitment_sha256"] = "1" * 64
        prereg["inner_split"]["exact_inner_dev_commitment_sha256"] = digest_commitment(
            [str(value) for value in digests.tolist()]
        )
        prereg["inner_split"]["exact_inner_dev_per_label_commitment_sha256"] = {
            label: digest_commitment(
                [
                    str(digests[index])
                    for index, gold_label in enumerate(gold.tolist())
                    if int(gold_label) == label_id
                ]
            )
            for label_id, label in enumerate(LABELS)
        }
        prereg_path = root / "prereg.json"
        write_new_json(prereg_path, prereg)
        frozen, prereg_sha = require_frozen_prereg(prereg_path)

        run_dir = root / "synthetic-run"
        screen_divisors = {"A": 10, "B": 6, "C": 8, "D": 5}
        for arm in ARMS:
            predicted = synthetic_predictions(gold, screen_divisors[arm], 0)
            aggregate = synthetic_aggregate(
                frozen,
                prereg_sha,
                "screen",
                arm,
                SCREEN_SEED,
                metrics_from_predictions(gold, predicted),
            )
            write_new_json(
                aggregate_path(run_dir, "screen", arm, SCREEN_SEED), aggregate
            )
            private_path = diagnostics_path(
                run_dir, "screen", arm, SCREEN_SEED
            )
            private_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                private_path,
                **synthetic_diagnostic_arrays(gold, predicted, digests),
            )
            os.chmod(private_path, 0o600)
        selection = analyze_screen(run_dir, frozen, prereg_sha)
        selftest_check(selection["selected_challenger"] == "A", "screen must select A")
        selftest_check(selection["confirmation_allowed"] is True, "screen guards must pass")
        selftest_check(
            set(selection["screen_aggregate_sha256s"]) == set(ARMS),
            "selection must bind all four screen aggregate hashes",
        )
        selftest_check(
            set(selection["screen_private_diagnostics_sha256s"]) == set(ARMS),
            "selection must bind all four private screen diagnostics hashes",
        )
        selftest_check(
            selection["analyzer_sha256"] == sha256_file(SCRIPT_PATH),
            "selection must bind the frozen analyzer bytes",
        )
        selection_path = root / "selection.json"
        write_new_json(selection_path, selection)
        tampered_selection = copy.deepcopy(selection)
        tampered_selection["selected_challenger"] = "B"
        tampered_selection_path = root / "tampered-selection.json"
        write_new_json(tampered_selection_path, tampered_selection)
        tamper_blocked = False
        try:
            analyze_confirmation(
                run_dir,
                frozen,
                prereg_sha,
                tampered_selection_path,
                bootstrap_replicates=10,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
        except ValidationError:
            tamper_blocked = True
        selftest_check(
            tamper_blocked,
            "hand-edited challenger selection must be rejected before confirmation",
        )

        for arm in ("A", "D"):
            for seed in CONFIRM_SEEDS:
                divisor = 10 if arm == "A" else 5
                predicted = synthetic_predictions(gold, divisor, seed - CONFIRM_SEEDS[0])
                aggregate = synthetic_aggregate(
                    frozen,
                    prereg_sha,
                    "confirm",
                    arm,
                    seed,
                    metrics_from_predictions(gold, predicted),
                )
                write_new_json(aggregate_path(run_dir, "confirm", arm, seed), aggregate)
                private_path = diagnostics_path(run_dir, "confirm", arm, seed)
                private_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    private_path,
                    **synthetic_diagnostic_arrays(gold, predicted, digests),
                )
                os.chmod(private_path, 0o600)
        mismatched_prereg = copy.deepcopy(frozen)
        mismatched_prereg["inner_split"]["exact_inner_dev_commitment_sha256"] = (
            "2" * 64
        )
        first_record = validate_aggregate(
            aggregate_path(run_dir, "confirm", "A", CONFIRM_SEEDS[0]),
            frozen,
            prereg_sha,
            "confirm",
            "A",
            CONFIRM_SEEDS[0],
        )
        source_private_path = diagnostics_path(
            run_dir, "confirm", "A", CONFIRM_SEEDS[0]
        )
        with np.load(source_private_path, allow_pickle=False) as archive:
            remapped_arrays = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
        first_second_label_row = supports[LABELS[0]]
        remapped_arrays["row_digests"][[0, first_second_label_row]] = (
            remapped_arrays["row_digests"][[first_second_label_row, 0]]
        )
        remapped_path = root / "digest-gold-remap.npz"
        np.savez_compressed(remapped_path, **remapped_arrays)
        os.chmod(remapped_path, 0o600)
        gold_mapping_mismatch_blocked = False
        try:
            validate_private_diagnostics(remapped_path, first_record, frozen)
        except ValidationError:
            gold_mapping_mismatch_blocked = True
        selftest_check(
            gold_mapping_mismatch_blocked,
            "the frozen per-label commitments must reject digest/gold remapping",
        )
        commitment_mismatch_blocked = False
        try:
            validate_private_diagnostics(
                diagnostics_path(run_dir, "confirm", "A", CONFIRM_SEEDS[0]),
                first_record,
                mismatched_prereg,
            )
        except ValidationError:
            commitment_mismatch_blocked = True
        selftest_check(
            commitment_mismatch_blocked,
            "private row digests must match the frozen inner-dev commitment",
        )
        confirmation = analyze_confirmation(
            run_dir,
            frozen,
            prereg_sha,
            selection_path,
            bootstrap_replicates=300,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        selftest_check(
            confirmation["decision"]
            == frozen["decision_rules"]["confirmation"]["pass_label"],
            "synthetic confirmation must pass",
        )
        selftest_check(
            confirmation["paired_identity_verified"]
            ["same_row_digests_across_all_six_runs"]
            is True,
            "paired row identity must be verified",
        )
        serialized = json.dumps(confirmation, ensure_ascii=False)
        for forbidden in ('"row_digests"', '"gold"', '"logits"', '"predicted"', '"confusion_matrix"'):
            selftest_check(forbidden not in serialized, "confirmation leaked a private key")
        mismatch_path = diagnostics_path(
            run_dir, "confirm", "A", CONFIRM_SEEDS[0]
        )
        with np.load(mismatch_path, allow_pickle=False) as archive:
            mismatched_arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        mismatched_arrays["row_digests"][0] = sha256_bytes(
            b"synthetic-paired-identity-mismatch"
        )
        mismatch_path.unlink()
        np.savez_compressed(mismatch_path, **mismatched_arrays)
        os.chmod(mismatch_path, 0o600)
        pairing_blocked = False
        try:
            analyze_confirmation(
                run_dir,
                frozen,
                prereg_sha,
                selection_path,
                bootstrap_replicates=10,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
        except ValidationError:
            pairing_blocked = True
        selftest_check(
            pairing_blocked,
            "mismatched private row digests must block paired confirmation",
        )
        overwrite_blocked = False
        try:
            write_new_json(selection_path, selection)
        except ValidationError:
            overwrite_blocked = True
        selftest_check(overwrite_blocked, "output overwrite must be refused")
    print(
        "PsySUICIDE accuracy-first v2 analyzer selftest PASS: synthetic four-arm "
        "screen, frozen selection, paired six-run diagnostics identity, hierarchical "
        "bootstrap, aggregate-only output, no overwrite, no model/data/network access"
    )


def resolve_run_dir(args: argparse.Namespace) -> Path | None:
    if args.run_dir is not None and args.run_id is not None:
        raise ValidationError("use either --run-dir or --run-id, not both")
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve(strict=True)
    elif args.run_id is not None:
        require(bool(SAFE_RUN_ID.fullmatch(args.run_id)), "invalid --run-id")
        run_dir = (args.results_root / args.run_id).resolve(strict=True)
    else:
        return None
    require(run_dir.is_dir() and not run_dir.is_symlink(), "run directory is invalid")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline aggregate-only PsySUICIDE accuracy-first v2 analyzer"
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--selection-artifact", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
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
        "writes_requested": bool(args.execute),
        "input_boundary": (
            "aggregate.json plus private numeric diagnostics.npz for screen and "
            "confirmation; no dataset/model/checkpoint/network"
        ),
        "screen": {"arms": list(ARMS), "seed": SCREEN_SEED},
        "confirmation": {
            "arms": "frozen challenger and D only",
            "seeds": list(CONFIRM_SEEDS),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    run_dir = resolve_run_dir(args)
    if run_dir is None:
        if args.execute:
            raise SystemExit("--execute requires --run-dir or --run-id")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("Dry-run only. No result, private diagnostics, dataset, or model was opened.")
        return
    run_dir = require_ignored_untracked_results_path(run_dir, "run directory")
    if args.execute and args.out is None:
        raise SystemExit("--execute requires --out")
    repository_results = (ROOT / "results").resolve()
    require(
        repository_results in run_dir.parents,
        "run directory must stay inside the repository's ignored results directory",
    )
    prereg_path = args.prereg.resolve(strict=True)
    require(
        (ROOT / "reports").resolve() in prereg_path.parents,
        "preregistration must stay inside repository reports/",
    )
    prereg, prereg_sha = require_frozen_prereg(prereg_path)
    if args.stage == "screen":
        result = analyze_screen(run_dir, prereg, prereg_sha)
    else:
        if args.selection_artifact is None:
            raise SystemExit("confirmation requires --selection-artifact")
        selection_path = args.selection_artifact.resolve(strict=True)
        require(
            ROOT.resolve() in selection_path.parents,
            "selection artifact must stay inside the repository",
        )
        result = analyze_confirmation(
            run_dir,
            prereg,
            prereg_sha,
            selection_path,
        )
    if args.execute:
        write_new_json(args.out, result)
        summary = {
            "status": "written",
            "stage": args.stage,
            "output": args.out.name,
            "decision": result["decision"],
            "selected_challenger": result.get("selected_challenger"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"dry_run_only": True, "artifact": result}, ensure_ascii=False, indent=2, allow_nan=False))
        print("Dry-run analysis complete. No artifact was written; use --execute --out to freeze it.")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        raise SystemExit(str(error)) from error
