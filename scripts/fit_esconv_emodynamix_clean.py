#!/usr/bin/env python3
"""Train/dev-only optimizer primitives for clean EmoDynamiX retraining.

The command defaults to a read-only audit.  It intentionally exposes no test
input and no task-checkpoint initialization option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA_ROOT = ROOT / "tmp/official_benchmarks/esconv/codes/dataset"
DEFAULT_FEATURE_RUN_DIR = (
    ROOT / "tmp/emodynamix-clean-features/canonical-train-dev-v1"
)
DEFAULT_OUTPUT_DIR = ROOT / "tmp/emodynamix-clean-training"

CLASS_COUNTS = (706, 820, 1_523, 1_470, 1_392, 508, 524, 1_490)
EMODYNAMIX_INTERNAL_LABELS = (
    "Reflection of feelings",
    "Self-disclosure",
    "Question",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Restatement or Paraphrasing",
    "Information",
    "Others",
)
EMODYNAMIX_ID_TO_CANONICAL = (
    "Reflection of feelings",
    "Self-disclosure",
    "Questions",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Restatement or Paraphrasing",
    "Information",
    "Other",
)
LOSS_MODES = (
    "author_weighted_ce",
    "ce",
    "class_balanced",
    "logit_adjusted",
)


def _validated_counts(class_counts: Sequence[int], *, like):
    import torch

    if len(class_counts) != 8:
        raise ValueError("class_counts must contain exactly eight values")
    if any(int(count) <= 0 for count in class_counts):
        raise ValueError("class_counts must be positive")
    return torch.as_tensor(class_counts, dtype=like.dtype, device=like.device)


def _count_tensor(
    class_counts: Sequence[int], *, dtype=None, device=None
):
    import torch

    if len(class_counts) != 8:
        raise ValueError("class_counts must contain exactly eight values")
    if any(int(count) <= 0 for count in class_counts):
        raise ValueError("class_counts must be positive")
    return torch.as_tensor(class_counts, dtype=dtype, device=device)


def author_weighted_ce_weights(
    class_counts: Sequence[int],
    *,
    temperature: float = 1.75,
    dtype=None,
    device=None,
):
    import torch

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("author weight temperature must be positive and finite")
    counts = _count_tensor(class_counts, dtype=dtype, device=device)
    inverse_frequency = (counts.sum() / counts.numel()) / counts
    return torch.softmax(inverse_frequency / temperature, dim=0)


def class_balanced_weights(
    class_counts: Sequence[int],
    *,
    beta: float = 0.999,
    dtype=None,
    device=None,
):
    import torch

    if not 0.0 < beta < 1.0:
        raise ValueError("class balance beta must be between zero and one")
    counts = _count_tensor(class_counts, dtype=dtype, device=device)
    raw_weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    return raw_weights / raw_weights.mean()


def loss_components(
    logits,
    labels,
    *,
    mode: str,
    class_counts: Sequence[int],
    author_weight_temperature: float = 1.75,
    class_balance_beta: float = 0.999,
    logit_adjustment_tau: float = 1.0,
):
    """Return an additive numerator and denominator for one microbatch.

    Callers accumulate numerators with gradients and denominators as scalars
    across the complete optimizer window, then normalize gradients exactly
    once.  This makes odd final windows mathematically equivalent to a single
    full-window loss computation.
    """

    import torch
    import torch.nn.functional as functional

    if logits.ndim != 2 or logits.shape[1] != 8:
        raise ValueError("logits must have shape [batch, 8]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if labels.numel() == 0:
        raise ValueError("loss batch must not be empty")
    if mode not in LOSS_MODES:
        raise ValueError(f"unsupported loss mode: {mode}")
    counts = _validated_counts(class_counts, like=logits)

    if mode == "author_weighted_ce":
        weights = author_weighted_ce_weights(
            class_counts,
            temperature=author_weight_temperature,
            dtype=logits.dtype,
            device=logits.device,
        )
        per_row = functional.cross_entropy(logits, labels, reduction="none")
        target_weights = weights[labels]
        return (per_row * target_weights).sum(), target_weights.sum()

    if mode == "class_balanced":
        weights = class_balanced_weights(
            class_counts,
            beta=class_balance_beta,
            dtype=logits.dtype,
            device=logits.device,
        )
        numerator = functional.cross_entropy(
            logits, labels, weight=weights, reduction="sum"
        )
        return numerator, weights[labels].sum()

    if mode == "logit_adjusted":
        if not math.isfinite(logit_adjustment_tau) or logit_adjustment_tau < 0.0:
            raise ValueError("logit_adjustment_tau must be finite and non-negative")
        prior = counts / counts.sum()
        logits = logits + logit_adjustment_tau * torch.log(prior)

    numerator = functional.cross_entropy(logits, labels, reduction="sum")
    denominator = logits.new_tensor(float(labels.numel()))
    return numerator, denominator


def normalize_accumulated_gradients(
    parameters: Iterable[Any], denominator: float
) -> None:
    """Normalize a complete accumulation window's gradient in-place."""

    if not math.isfinite(float(denominator)) or float(denominator) <= 0.0:
        raise ValueError("gradient denominator must be positive and finite")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(float(denominator))


def is_better_dev(
    candidate: Mapping[str, float], incumbent: Mapping[str, float] | None
) -> bool:
    """Apply the frozen pilot tie-break: Macro-F1, accuracy, then lower CE."""

    if incumbent is None:
        return True
    candidate_key = (
        float(candidate["macro_f1"]),
        float(candidate["accuracy"]),
        -float(candidate["selection_loss"]),
    )
    incumbent_key = (
        float(incumbent["macro_f1"]),
        float(incumbent["accuracy"]),
        -float(incumbent["selection_loss"]),
    )
    return candidate_key > incumbent_key


def optimizer_parameter_groups(
    model, *, weight_decay: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create auditable AdamW decay/no-decay parameter groups."""

    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    decay_parameters = []
    no_decay_parameters = []
    decay_names = []
    no_decay_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = name.lower().replace("_", "")
        no_decay = name.endswith(".bias") or normalized_name.endswith(
            "layernorm.weight"
        )
        if no_decay:
            no_decay_parameters.append(parameter)
            no_decay_names.append(name)
        else:
            decay_parameters.append(parameter)
            decay_names.append(name)
    if not decay_parameters or not no_decay_parameters:
        raise ValueError("optimizer decay and no-decay groups must both be non-empty")
    groups = [
        {"params": decay_parameters, "weight_decay": float(weight_decay)},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    roster = [
        {
            "name": name,
            "numel": int(parameter.numel()),
            "group": (
                "no_decay" if name in no_decay_names else "decay"
            ),
        }
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad
    ]
    roster_sha256 = hashlib.sha256(
        json.dumps(
            roster,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return groups, {
        "rule": "all trainable parameters except bias and LayerNorm weights decay",
        "decay_parameter_names": sorted(decay_names),
        "no_decay_parameter_names": sorted(no_decay_names),
        "decay_parameter_count": len(decay_names),
        "no_decay_parameter_count": len(no_decay_names),
        "trainable_parameter_count": len(roster),
        "trainable_parameter_numel": sum(item["numel"] for item in roster),
        "parameter_roster_sha256": roster_sha256,
    }


def prediction_ids_from_raw_logits(logits) -> list[int]:
    if logits.ndim != 2 or logits.shape[1] != 8:
        raise ValueError("raw logits must have shape [batch, 8]")
    return [int(value) for value in logits.argmax(dim=-1).detach().cpu().tolist()]


def metrics_from_ids(
    gold_ids: Sequence[int],
    predicted_ids: Sequence[int],
    *,
    objective_loss: float,
    selection_loss: float,
) -> dict[str, Any]:
    from open_response_eval.esconv_metrics import compute_classification_metrics

    if len(gold_ids) != len(predicted_ids) or not gold_ids:
        raise ValueError("gold and prediction ids must have the same non-zero length")
    if any(
        isinstance(value, bool) or not 0 <= int(value) < len(EMODYNAMIX_ID_TO_CANONICAL)
        for value in [*gold_ids, *predicted_ids]
    ):
        raise ValueError("metric ids must be valid EmoDynamiX class ids")
    if any(
        not math.isfinite(float(value))
        for value in (objective_loss, selection_loss)
    ):
        raise ValueError("metric losses must be finite")
    records = [
        {
            "gold": EMODYNAMIX_ID_TO_CANONICAL[int(gold)],
            "prediction": EMODYNAMIX_ID_TO_CANONICAL[int(prediction)],
            "invalid": False,
        }
        for gold, prediction in zip(gold_ids, predicted_ids)
    ]
    metrics = compute_classification_metrics(
        records, labels=EMODYNAMIX_ID_TO_CANONICAL
    )
    metrics["objective_loss"] = float(objective_loss)
    metrics["selection_loss"] = float(selection_loss)
    return metrics


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_contract_digest(
    contract: Mapping[str, Any], field: str
) -> str:
    value = contract.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"preregistration {field} must be a lowercase SHA-256")
    return value


def load_verified_feature_run(
    run_dir: Path,
    *,
    prepared: dict[str, Any],
    preregistration_contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load one externally anchored, three-file verified feature bundle."""

    from scripts import train_esconv_emodynamix_clean as data_contract

    run_dir = Path(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("feature run directory is missing or symlinked")
    paths = {
        "summary": run_dir / "feature_run_summary.json",
        "manifest": run_dir / "generator_manifest.json",
        "features": run_dir / "features.jsonl",
    }
    for path in paths.values():
        if path.is_symlink():
            raise ValueError(f"feature bundle file must not be a symlink: {path.name}")
        if not path.is_file():
            raise ValueError(f"feature bundle file is missing: {path.name}")
    actual_names = {path.name for path in run_dir.iterdir()}
    expected_names = {path.name for path in paths.values()}
    if actual_names != expected_names:
        raise ValueError("feature run directory must contain exactly three bundle files")

    payloads = {name: path.read_bytes() for name, path in paths.items()}
    expected_summary_sha = _required_contract_digest(
        preregistration_contract, "feature_run_summary_sha256"
    )
    actual_summary_sha = _sha256_bytes(payloads["summary"])
    if actual_summary_sha != expected_summary_sha:
        raise ValueError(
            "feature run summary SHA mismatch: "
            f"expected={expected_summary_sha} actual={actual_summary_sha}"
        )
    try:
        summary = json.loads(payloads["summary"])
        manifest = json.loads(payloads["manifest"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("feature summary or manifest is invalid JSON") from error
    if not isinstance(summary, dict) or not isinstance(manifest, dict):
        raise ValueError("feature summary and manifest must be JSON objects")

    for field in (
        "generator_manifest_sha256",
        "features_jsonl_sha256",
        "feature_table_sha256",
    ):
        expected = _required_contract_digest(preregistration_contract, field)
        if summary.get(field) != expected:
            raise ValueError(f"feature summary {field} differs from preregistration")
    for field in (
        "training_records",
        "development_records",
        "feature_rows",
        "feature_batch_size",
    ):
        expected = preregistration_contract.get(field)
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
            raise ValueError(f"preregistration {field} must be a positive integer")
        if summary.get(field) != expected:
            raise ValueError(f"feature summary {field} differs from preregistration")

    if summary.get("training_records") != len(prepared["train_records"]):
        raise ValueError("feature summary training record count mismatch")
    if summary.get("development_records") != len(prepared["dev_records"]):
        raise ValueError("feature summary development record count mismatch")
    if summary.get("status") != "VERIFIED_FEATURES_REQUIRES_TRAINING_PREREGISTRATION":
        raise ValueError("feature summary status is not verified")
    if summary.get("frozen_test_accessed") is not False:
        raise ValueError("feature summary does not prove frozen-test isolation")
    if summary.get("target_or_label_fields_in_features") != []:
        raise ValueError("feature summary declares target or label fields")
    if summary.get("selectable_model_produced") is not False:
        raise ValueError("feature generation must not claim a selectable model")

    manifest_file_sha = _sha256_bytes(payloads["manifest"])
    if summary.get("generator_manifest_file_sha256") != manifest_file_sha:
        raise ValueError("generator manifest file SHA mismatch")
    manifest_sha = data_contract.canonical_json_sha256(manifest)
    if manifest_sha != summary["generator_manifest_sha256"]:
        raise ValueError("generator manifest canonical SHA mismatch")
    if manifest.get("feature_backend") != data_contract.VERIFIED_FEATURE_BACKEND:
        raise ValueError("feature manifest backend is not verified upstream SDDP/ERC")
    if manifest.get("target_or_label_fields") != []:
        raise ValueError("feature manifest includes target or label fields")
    if manifest.get("sddp_max_num_contexts") != 37:
        raise ValueError("feature manifest is not author-faithful SDDP=37")
    if manifest.get("feature_batch_size") != summary["feature_batch_size"]:
        raise ValueError("feature batch size differs between manifest and summary")
    if manifest.get("unique_input_count") != summary["feature_rows"]:
        raise ValueError("feature manifest unique input count mismatch")
    if manifest.get("source_audit") != prepared["source_audit"]:
        raise ValueError("feature manifest source audit differs from prepared data")
    if manifest.get("overlap_audit") != prepared["audit"]:
        raise ValueError("feature manifest overlap audit differs from prepared data")

    actual_features_sha = _sha256_bytes(payloads["features"])
    if actual_features_sha != summary["features_jsonl_sha256"]:
        raise ValueError(
            "features JSONL SHA mismatch: "
            f"expected={summary['features_jsonl_sha256']} actual={actual_features_sha}"
        )
    required_keys = {
        str(record["model_input_sha256"])
        for record in [*prepared["train_records"], *prepared["dev_records"]]
    }
    features, feature_audit = data_contract.load_feature_jsonl_bytes(
        payloads["features"],
        required_input_sha256=required_keys,
        expected_generator_manifest_sha256=manifest_sha,
        expected_table_sha256=summary["feature_table_sha256"],
        allow_fixture=False,
    )
    if len(features) != summary["feature_rows"]:
        raise ValueError("verified feature row count mismatch")
    audit = {
        **feature_audit,
        "feature_run_summary_sha256": actual_summary_sha,
        "generator_manifest_file_sha256": manifest_file_sha,
        "features_jsonl_sha256": actual_features_sha,
        "missing_feature_keys": [],
        "extra_feature_keys": [],
        "verified_upstream_features": True,
        "frozen_test_accessed": False,
    }
    return features, audit


def build_parser() -> argparse.ArgumentParser:
    from scripts import train_esconv_emodynamix_clean as data_contract

    parser = argparse.ArgumentParser(
        description="Audit or fit a clean train/dev-only EmoDynamiX strategy policy"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=(
            data_contract.DEFAULT_DATA_ROOT
            / data_contract.OFFICIAL_SPLITS["train"]["filename"]
        ),
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=(
            data_contract.DEFAULT_DATA_ROOT
            / data_contract.OFFICIAL_SPLITS["dev"]["filename"]
        ),
    )
    parser.add_argument("--feature-run-dir", type=Path, default=DEFAULT_FEATURE_RUN_DIR)
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("audit", "smoke", "pilot"), default="audit")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--loss", choices=LOSS_MODES, default="author_weighted_ce")
    parser.add_argument("--seed", type=int, default=114_514)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-updates", type=int, default=500)
    parser.add_argument("--max-updates", type=int, default=5_000)
    parser.add_argument("--author-weight-temperature", type=float, default=1.75)
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode != "audit" and not args.execute:
        raise SystemExit("smoke/pilot requires explicit --execute")
    print(
        {
            "status": "AUDIT_ONLY" if not args.execute else "NOT_YET_IMPLEMENTED",
            "mode": args.mode,
            "loss": args.loss,
            "frozen_test_accessed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
