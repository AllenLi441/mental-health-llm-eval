#!/usr/bin/env python3
"""Train/dev-only optimizer primitives for clean EmoDynamiX retraining.

The command defaults to a read-only audit.  It intentionally exposes no test
input and no task-checkpoint initialization option.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
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
DEFAULT_BASE_MODEL_DIR = ROOT / "tmp/official_benchmarks/roberta-base-e2da-materialized"
EXPECTED_BASE_MODEL_TREE_SHA256 = (
    "1d9faa93557a63a92292cd11dfbca3de8e336ffa60768745a71ecd1ed19aa91c"
)

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
        _validated_dev_selection_key(candidate)
        return True
    candidate_key = _validated_dev_selection_key(candidate)
    incumbent_key = _validated_dev_selection_key(incumbent)
    return candidate_key > incumbent_key


def _validated_dev_selection_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    try:
        macro_f1 = float(metrics["macro_f1"])
        accuracy = float(metrics["accuracy"])
        selection_loss = float(metrics["selection_loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dev metric selection fields are invalid") from error
    if (
        not all(math.isfinite(value) for value in (macro_f1, accuracy, selection_loss))
        or not 0.0 <= macro_f1 <= 1.0
        or not 0.0 <= accuracy <= 1.0
        or selection_loss < 0.0
    ):
        raise ValueError("dev metric values are nonfinite or outside valid bounds")
    candidate_key = (
        macro_f1,
        accuracy,
        -selection_loss,
    )
    return candidate_key


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
    metric_ids = [*gold_ids, *predicted_ids]
    if any(type(value) is not int for value in metric_ids):
        raise ValueError("metric ids must be strict integers")
    if any(not 0 <= value < len(EMODYNAMIX_ID_TO_CANONICAL) for value in metric_ids):
        raise ValueError("metric integer ids must be valid EmoDynamiX classes")
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


def derive_smoke_feature_contract(run_dir: Path) -> dict[str, Any]:
    """Self-anchor a feature summary for a non-selectable developmental smoke."""

    run_dir = Path(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("smoke feature run directory is missing or symlinked")
    summary_path = run_dir / "feature_run_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ValueError("smoke feature summary is missing or symlinked")
    payload = summary_path.read_bytes()
    try:
        summary = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("smoke feature summary is invalid JSON") from error
    if (
        not isinstance(summary, dict)
        or summary.get("status")
        != "VERIFIED_FEATURES_REQUIRES_TRAINING_PREREGISTRATION"
    ):
        raise ValueError("smoke feature summary status is not verified")
    contract = {
        "feature_run_summary_sha256": _sha256_bytes(payload),
        "training_records": summary.get("training_records"),
        "development_records": summary.get("development_records"),
        "feature_rows": summary.get("feature_rows"),
        "feature_batch_size": summary.get("feature_batch_size"),
        "generator_manifest_sha256": summary.get("generator_manifest_sha256"),
        "features_jsonl_sha256": summary.get("features_jsonl_sha256"),
        "feature_table_sha256": summary.get("feature_table_sha256"),
    }
    for field in (
        "generator_manifest_sha256",
        "features_jsonl_sha256",
        "feature_table_sha256",
    ):
        _required_contract_digest(contract, field)
    for field in (
        "training_records",
        "development_records",
        "feature_rows",
        "feature_batch_size",
    ):
        value = contract[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"smoke feature summary {field} is invalid")
    return contract


class CleanEmoDynamiXTrainingDataset:
    """Record-preserving many-to-one join from labels to causal features."""

    _ITEM_FIELDS = {
        "item_id",
        "model_input_sha256",
        "model_input",
        "feature",
        "label_id",
    }

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        features_by_input_sha256: Mapping[str, dict[str, Any]],
    ) -> None:
        from scripts import train_esconv_emodynamix_clean as data_contract

        if not records:
            raise ValueError("training dataset records must not be empty")
        items: list[dict[str, Any]] = []
        for record_index, record in enumerate(records):
            try:
                item_id = record["item_id"]
                model_input_sha256 = record["model_input_sha256"]
                model_input = record["model_input"]
                label_id = record["label_id"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"training record {record_index} lacks required fields"
                ) from error
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("training item_id must be a non-empty string")
            if not isinstance(model_input, dict):
                raise ValueError("training model_input must be an object")
            if data_contract.canonical_json_sha256(model_input) != model_input_sha256:
                raise ValueError("training model input SHA does not match input bytes")
            if isinstance(label_id, bool) or not isinstance(label_id, int):
                raise ValueError("training label_id must be an integer")
            if not 0 <= label_id < len(EMODYNAMIX_ID_TO_CANONICAL):
                raise ValueError("training label_id is outside the eight-class range")
            feature = features_by_input_sha256.get(model_input_sha256)
            if feature is None:
                raise ValueError(f"missing feature for training record: {item_id}")
            if feature.get("model_input_sha256") != model_input_sha256:
                raise ValueError("training feature SHA does not match record input SHA")
            data_contract.validate_feature_row(
                feature, expected_input_sha256=model_input_sha256
            )
            items.append(
                {
                    "item_id": item_id,
                    "model_input_sha256": model_input_sha256,
                    "model_input": dict(model_input),
                    "feature": feature,
                    "label_id": label_id,
                }
            )
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._items[index]


def make_training_collator(*, model_module: Any, device: Any):
    """Build a collator that keeps labels outside the exact model input."""

    import torch
    from scripts import train_esconv_emodynamix_clean as data_contract

    expected_item_fields = CleanEmoDynamiXTrainingDataset._ITEM_FIELDS
    expected_model_batch_fields = {
        "dialogue_history",
        "strategy_history",
        "speaker_turn",
        "parsed_dialogue",
        "erc_probabilities",
        "dialogue_sizes",
    }

    def collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("training batch must not be empty")
        model_inputs = []
        feature_rows = []
        labels = []
        item_ids = []
        for item in items:
            if set(item) != expected_item_fields:
                raise ValueError("training item fields do not match the clean contract")
            key = item["model_input_sha256"]
            model_input = item["model_input"]
            feature = item["feature"]
            if data_contract.canonical_json_sha256(model_input) != key:
                raise ValueError("training batch model input SHA mismatch")
            if feature.get("model_input_sha256") != key:
                raise ValueError("training batch feature SHA mismatch")
            label_id = item["label_id"]
            if isinstance(label_id, bool) or not isinstance(label_id, int):
                raise ValueError("training batch label must be an integer")
            if not 0 <= label_id < len(EMODYNAMIX_ID_TO_CANONICAL):
                raise ValueError("training batch label is outside the eight-class range")
            model_inputs.append(model_input)
            feature_rows.append(feature)
            labels.append(label_id)
            item_ids.append(item["item_id"])
        model_batch = model_module.collate_clean_model_batch(
            model_inputs, feature_rows, device=device
        )
        if set(model_batch) != expected_model_batch_fields:
            raise ValueError("model collator returned unexpected fields")
        return {
            "model_batch": model_batch,
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "item_ids": item_ids,
        }

    return collate


def evaluate_emodynamix(
    model,
    data_loader,
    *,
    device,
    loss_mode: str,
    class_counts: Sequence[int],
    author_weight_temperature: float,
    class_balance_beta: float,
    logit_adjustment_tau: float,
) -> dict[str, Any]:
    """Evaluate complete dev data with raw-logit predictions and common CE."""

    import torch
    import torch.nn.functional as functional

    objective_numerator = 0.0
    objective_denominator = 0.0
    selection_numerator = 0.0
    gold_ids: list[int] = []
    predicted_ids: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if set(batch) != {"model_batch", "labels", "item_ids"}:
                raise ValueError("development batch fields mismatch")
            labels = batch["labels"].to(device)
            model_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch["model_batch"].items()
            }
            outputs = model(model_batch)
            if not isinstance(outputs, dict) or "logits" not in outputs:
                raise ValueError("model output lacks raw logits")
            logits = outputs["logits"]
            if logits.ndim != 2 or logits.shape != (labels.numel(), 8):
                raise ValueError("model raw logits have the wrong shape")
            current_numerator, current_denominator = loss_components(
                logits,
                labels,
                mode=loss_mode,
                class_counts=class_counts,
                author_weight_temperature=author_weight_temperature,
                class_balance_beta=class_balance_beta,
                logit_adjustment_tau=logit_adjustment_tau,
            )
            objective_numerator += float(current_numerator.detach().cpu())
            objective_denominator += float(current_denominator.detach().cpu())
            selection_numerator += float(
                functional.cross_entropy(logits, labels, reduction="sum")
                .detach()
                .cpu()
            )
            gold_ids.extend(int(value) for value in labels.detach().cpu().tolist())
            predicted_ids.extend(prediction_ids_from_raw_logits(logits))
    if not gold_ids or objective_denominator <= 0.0:
        raise ValueError("development loader is empty")
    return metrics_from_ids(
        gold_ids,
        predicted_ids,
        objective_loss=objective_numerator / objective_denominator,
        selection_loss=selection_numerator / len(gold_ids),
    )


def run_training_epochs(
    model,
    train_loader,
    *,
    dev_loader,
    optimizer,
    scheduler,
    device,
    epochs: int,
    gradient_accumulation_steps: int,
    max_updates: int,
    max_grad_norm: float,
    loss_mode: str,
    class_counts: Sequence[int],
    author_weight_temperature: float,
    class_balance_beta: float,
    logit_adjustment_tau: float,
    checkpoint_callback,
) -> dict[str, Any]:
    """Run the train/dev loop with exact complete-window normalization."""

    import torch

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps < 1
    ):
        raise ValueError("gradient accumulation steps must be a positive integer")
    if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < 1:
        raise ValueError("max_updates must be a positive integer")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive and finite")
    try:
        total_batches = len(train_loader)
    except TypeError as error:
        raise ValueError("training loader must have a finite length") from error
    if total_batches < 1:
        raise ValueError("training loader is empty")

    actual_updates = 0
    actual_scheduler_steps = 0
    best_epoch: int | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stop_reason = "epochs_completed"

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        window_denominator = 0.0
        epoch_numerator = 0.0
        epoch_denominator = 0.0
        batches_seen = 0
        reached_max_updates = False
        for batch_index, batch in enumerate(train_loader):
            if set(batch) != {"model_batch", "labels", "item_ids"}:
                raise ValueError("training batch fields mismatch")
            labels = batch["labels"].to(device)
            model_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch["model_batch"].items()
            }
            outputs = model(model_batch)
            if not isinstance(outputs, dict) or "logits" not in outputs:
                raise ValueError("training model output lacks logits")
            numerator, denominator = loss_components(
                outputs["logits"],
                labels,
                mode=loss_mode,
                class_counts=class_counts,
                author_weight_temperature=author_weight_temperature,
                class_balance_beta=class_balance_beta,
                logit_adjustment_tau=logit_adjustment_tau,
            )
            numerator.backward()
            denominator_value = float(denominator.detach().cpu())
            if not math.isfinite(denominator_value) or denominator_value <= 0.0:
                raise ValueError("training loss denominator is invalid")
            window_denominator += denominator_value
            epoch_numerator += float(numerator.detach().cpu())
            epoch_denominator += denominator_value
            batches_seen += 1

            closes_window = (
                batches_seen % gradient_accumulation_steps == 0
                or batch_index + 1 == total_batches
            )
            if not closes_window:
                continue
            normalize_accumulated_gradients(model.parameters(), window_denominator)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            if not math.isfinite(float(gradient_norm.detach().cpu())):
                raise ValueError("training gradient norm is nonfinite")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                actual_scheduler_steps += 1
            optimizer.zero_grad(set_to_none=True)
            actual_updates += 1
            window_denominator = 0.0
            if actual_updates >= max_updates:
                reached_max_updates = True
                break

        if batches_seen < 1 or epoch_denominator <= 0.0:
            raise ValueError("training loader yielded no batches")
        dev_metrics = evaluate_emodynamix(
            model,
            dev_loader,
            device=device,
            loss_mode=loss_mode,
            class_counts=class_counts,
            author_weight_temperature=author_weight_temperature,
            class_balance_beta=class_balance_beta,
            logit_adjustment_tau=logit_adjustment_tau,
        )
        history.append(
            {
                "epoch": epoch,
                "actual_optimizer_updates": actual_updates,
                "train_objective_loss": epoch_numerator / epoch_denominator,
                "dev": dev_metrics,
            }
        )
        if is_better_dev(dev_metrics, best_metrics):
            best_epoch = epoch
            best_metrics = dict(dev_metrics)
            checkpoint_callback(
                epoch=epoch,
                metrics=dev_metrics,
                model=model,
                actual_updates=actual_updates,
            )
        if reached_max_updates:
            stop_reason = "max_updates_reached"
            break

    if best_epoch is None or best_metrics is None:
        raise ValueError("training completed without a selectable dev checkpoint")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_dev": best_metrics,
        "actual_optimizer_updates": actual_updates,
        "actual_scheduler_steps": actual_scheduler_steps,
        "stop_reason": stop_reason,
    }


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_training_artifacts(
    output_dir: Path,
    *,
    model_state_dict: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
    run_status: str,
) -> dict[str, Any]:
    """Atomically publish a safe state-dict checkpoint and hash-closed receipt."""

    import torch

    allowed_statuses = {
        "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE",
        "DEV_PILOT_CANDIDATE_NOT_FROZEN_TESTED",
    }
    if run_status not in allowed_statuses:
        raise ValueError("training artifact run status is unsupported")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        raise ValueError("training output directory must be absolute")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"training output directory already exists: {output_dir}")
    if manifest_base.get("task_checkpoint_sha256", "MISSING") is not None:
        raise ValueError("clean training manifest must forbid task checkpoint initialization")
    if not model_state_dict:
        raise ValueError("model state dict must not be empty")
    safe_state: dict[str, Any] = {}
    for name, tensor in model_state_dict.items():
        if not isinstance(name, str) or not name or not torch.is_tensor(tensor):
            raise ValueError("model state dict must map names to tensors")
        safe_state[name] = tensor.detach().cpu().clone()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    try:
        checkpoint_path = staging / "best_model.pt"
        manifest_path = staging / "training_manifest.json"
        receipt_path = staging / "artifact_receipt.json"
        checkpoint = {
            "checkpoint_schema_version": "clean-emodynamix-state-dict-v1",
            "label_order": list(EMODYNAMIX_ID_TO_CANONICAL),
            "model_state_dict": safe_state,
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha = _sha256_bytes(checkpoint_path.read_bytes())

        manifest = dict(manifest_base)
        manifest.update(
            {
                "schema_version": "clean-emodynamix-training-manifest-v1",
                "status": run_status,
                "checkpoint_filename": checkpoint_path.name,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_format": "pytorch_state_dict_weights_only_v1",
                "label_order": list(EMODYNAMIX_ID_TO_CANONICAL),
                "task_checkpoint_sha256": None,
                "selectable_model_produced": (
                    run_status == "DEV_PILOT_CANDIDATE_NOT_FROZEN_TESTED"
                ),
                "frozen_leaderboard_eligible": False,
                "frozen_test_accessed": False,
            }
        )
        manifest_payload = _pretty_json_bytes(manifest)
        manifest_path.write_bytes(manifest_payload)
        manifest_sha = _sha256_bytes(manifest_payload)
        receipt = {
            "schema_version": "clean-emodynamix-artifact-receipt-v1",
            "status": run_status,
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "training_manifest_filename": manifest_path.name,
            "training_manifest_sha256": manifest_sha,
            "frozen_test_accessed": False,
        }
        receipt_path.write_bytes(_pretty_json_bytes(receipt))
        if output_dir.exists():
            raise FileExistsError(
                f"training output directory appeared before publish: {output_dir}"
            )
        os.rename(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "output_dir": str(output_dir),
        "training_manifest": manifest,
        "artifact_receipt": receipt,
    }


def load_training_checkpoint_strict(
    path: Path, *, expected_sha256: str, model
) -> dict[str, Any]:
    """Load the exact audited bytes with weights-only and strict state matching."""

    import torch

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected checkpoint SHA-256 is invalid")
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("training checkpoint must be an absolute regular file")
    payload = path.read_bytes()
    actual_sha = _sha256_bytes(payload)
    if actual_sha != expected_sha256:
        raise ValueError(
            "checkpoint SHA mismatch: "
            f"expected={expected_sha256} actual={actual_sha}"
        )
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "checkpoint_schema_version",
        "label_order",
        "model_state_dict",
    }:
        raise ValueError("training checkpoint schema mismatch")
    if checkpoint["checkpoint_schema_version"] != "clean-emodynamix-state-dict-v1":
        raise ValueError("training checkpoint schema version mismatch")
    if checkpoint["label_order"] != list(EMODYNAMIX_ID_TO_CANONICAL):
        raise ValueError("training checkpoint label order mismatch")
    state = checkpoint["model_state_dict"]
    if (
        not isinstance(state, dict)
        or not state
        or any(not isinstance(name, str) or not torch.is_tensor(tensor) for name, tensor in state.items())
    ):
        raise ValueError("training checkpoint state dict is invalid")
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("training checkpoint does not strictly match the model") from error
    return {
        "checkpoint_sha256": actual_sha,
        "weights_only": True,
        "strict": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def class_counts_from_records(records: Sequence[dict[str, Any]]) -> tuple[int, ...]:
    counts = [0] * len(EMODYNAMIX_ID_TO_CANONICAL)
    for record in records:
        label_id = record.get("label_id")
        if type(label_id) is not int or not 0 <= label_id < len(counts):
            raise ValueError("training record has an invalid integer label_id")
        counts[label_id] += 1
    return tuple(counts)


def _select_smoke_records(
    records: Sequence[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("smoke row limit must be a positive integer")
    if len(records) <= limit:
        return list(records)
    selected_indices: list[int] = []
    seen_labels: set[int] = set()
    for index, record in enumerate(records):
        label_id = record.get("label_id")
        if type(label_id) is not int or not 0 <= label_id < 8:
            raise ValueError("smoke record has an invalid label_id")
        if label_id not in seen_labels:
            selected_indices.append(index)
            seen_labels.add(label_id)
            if len(selected_indices) == limit:
                break
    if len(selected_indices) < limit:
        selected = set(selected_indices)
        for index in range(len(records)):
            if index not in selected:
                selected_indices.append(index)
                if len(selected_indices) == limit:
                    break
    return [records[index] for index in sorted(selected_indices)]


def validate_training_execution(
    args,
    *,
    prepared: dict[str, Any],
    feature_run: dict[str, Any],
    execution_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Enforce the smoke/pilot boundary before model or optimizer creation."""

    mode = getattr(args, "mode", None)
    if mode not in {"smoke", "pilot"}:
        raise ValueError("training execution mode must be smoke or pilot")
    features = feature_run.get("features_by_input_sha256")
    audit = feature_run.get("audit")
    if not isinstance(features, dict) or not features or not isinstance(audit, dict):
        raise ValueError("training feature run is incomplete")
    required_keys = {
        str(record["model_input_sha256"])
        for record in [*prepared["train_records"], *prepared["dev_records"]]
    }
    if set(features) != required_keys:
        raise ValueError("training feature keys do not close over train/dev records")
    if mode == "pilot":
        if audit.get("verified_upstream_features") is not True:
            raise ValueError("pilot requires a verified feature bundle")
        if execution_provenance.get("execution_validated") is not True:
            raise ValueError("pilot requires a frozen preregistration execution gate")
        if len(prepared["train_records"]) != 8_433 or len(prepared["dev_records"]) != 2_985:
            raise ValueError("pilot requires the complete canonical train/dev records")
        if class_counts_from_records(prepared["train_records"]) != CLASS_COUNTS:
            raise ValueError("pilot clean-train class counts differ from preregistration")
    return {
        "mode": mode,
        "training_records": len(prepared["train_records"]),
        "development_records": len(prepared["dev_records"]),
        "unique_feature_keys": len(features),
        "verified_upstream_features": audit.get("verified_upstream_features") is True,
        "execution_validated": execution_provenance.get("execution_validated") is True,
        "frozen_test_accessed": False,
    }


def validate_pilot_preregistration_document(
    document: dict[str, Any],
    args,
    *,
    current_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate one frozen train/dev pilot arm against exact launch arguments."""

    protocol_id = "jingshi-esconv-emodynamix-clean-dev-pilot-v1"
    if not isinstance(document, dict):
        raise ValueError("pilot preregistration must be a JSON object")
    if (
        document.get("schema_version") != protocol_id
        or document.get("protocol_id") != protocol_id
        or document.get("parent_protocol_id") != "jingshi-esconv-first-v1"
    ):
        raise ValueError("pilot preregistration protocol identifiers mismatch")
    if document.get("status") != "FROZEN" or not isinstance(
        document.get("frozen_at"), str
    ) or not document["frozen_at"]:
        raise ValueError("pilot preregistration is not frozen")
    if document.get("frozen_test_policy") != "FORBIDDEN_DURING_TRAIN_DEV_PILOT":
        raise ValueError("pilot frozen-test policy mismatch")

    data_contract = document.get("data_contract")
    expected_data_contract = {
        "training_records": 8_433,
        "development_records": 2_985,
        "feature_rows": 11_366,
        "class_counts": list(CLASS_COUNTS),
    }
    if data_contract != expected_data_contract:
        raise ValueError("pilot data contract mismatch")
    feature_contract = document.get("feature_run_contract")
    if not isinstance(feature_contract, dict):
        raise ValueError("pilot feature run contract is missing")
    for field in (
        "feature_run_summary_sha256",
        "generator_manifest_sha256",
        "features_jsonl_sha256",
        "feature_table_sha256",
    ):
        _required_contract_digest(feature_contract, field)
    for field, expected in (
        ("training_records", 8_433),
        ("development_records", 2_985),
        ("feature_rows", 11_366),
        ("feature_batch_size", 8),
    ):
        if feature_contract.get(field) != expected:
            raise ValueError(f"pilot feature run contract {field} mismatch")

    if document.get("base_model_contract") != {
        "tree_sha256": args.expected_base_tree_sha256,
        "task_checkpoint_initialization": None,
    } or args.expected_base_tree_sha256 != EXPECTED_BASE_MODEL_TREE_SHA256:
        raise ValueError("pilot base model contract mismatch")
    expected_source_fields = {
        "trainer_sha256",
        "data_builder_sha256",
        "model_sha256",
        "metrics_sha256",
    }
    source_contract = document.get("source_code_contract")
    if (
        not isinstance(source_contract, dict)
        or set(source_contract) != expected_source_fields
        or dict(source_contract) != dict(current_source_hashes)
    ):
        raise ValueError("pilot source code contract mismatch")
    for field in expected_source_fields:
        _required_contract_digest(source_contract, field)

    shared = document.get("shared_training")
    expected_shared = {
        "device": args.device,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_updates": args.warmup_updates,
        "max_updates": args.max_updates,
        "max_grad_norm": args.max_grad_norm,
        "author_weight_temperature": args.author_weight_temperature,
        "class_balance_beta": args.class_balance_beta,
        "logit_adjustment_tau": args.logit_adjustment_tau,
    }
    if shared != expected_shared:
        raise ValueError("pilot shared training contract mismatch")
    arms = document.get("arms")
    if not isinstance(arms, dict) or args.arm_id not in arms:
        raise ValueError("pilot arm is not preregistered")
    selected_arm = arms[args.arm_id]
    if selected_arm != {
        "loss": args.loss,
        "seed": args.seed,
        "output_dir_name": Path(args.output_dir).name,
    }:
        if selected_arm.get("output_dir_name") != Path(args.output_dir).name:
            raise ValueError("pilot output directory differs from the authorized arm")
        raise ValueError("pilot arm hyperparameters differ from preregistration")
    if document.get("selection") != {
        "primary": "dev_macro_f1",
        "secondary": "dev_accuracy",
        "tertiary": "lower_raw_unweighted_ce",
        "prediction_source": "raw_logits",
    }:
        raise ValueError("pilot selection contract mismatch")
    return {
        "protocol_id": protocol_id,
        "arm_id": args.arm_id,
        "frozen_at": document["frozen_at"],
        "feature_run_contract": feature_contract,
        "source_code_contract": source_contract,
        "execution_validated": True,
        "frozen_test_accessed": False,
    }


def current_pilot_source_hashes() -> dict[str, str]:
    return {
        "trainer_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "data_builder_sha256": _sha256_bytes(
            (ROOT / "scripts/train_esconv_emodynamix_clean.py").read_bytes()
        ),
        "model_sha256": _sha256_bytes(
            (ROOT / "open_response_eval/esconv_emodynamix_model.py").read_bytes()
        ),
        "metrics_sha256": _sha256_bytes(
            (ROOT / "open_response_eval/esconv_metrics.py").read_bytes()
        ),
    }


def load_pilot_execution_provenance(args) -> dict[str, Any]:
    """Load an externally SHA-anchored, committed-clean pilot preregistration."""

    path = getattr(args, "preregistration", None)
    expected_sha = getattr(args, "preregistration_sha256", None)
    if path is None or expected_sha is None:
        raise ValueError("pilot requires preregistration path and SHA-256 anchor")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError("pilot preregistration SHA-256 anchor is invalid")
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("pilot preregistration must be an absolute regular file")
    try:
        relative_path = path.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("pilot preregistration must be inside the repository") from error
    payload = path.read_bytes()
    actual_sha = _sha256_bytes(payload)
    if actual_sha != expected_sha:
        raise ValueError(
            "pilot preregistration SHA mismatch: "
            f"expected={expected_sha} actual={actual_sha}"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("pilot preregistration is invalid JSON") from error
    source_hashes = current_pilot_source_hashes()
    validated = validate_pilot_preregistration_document(
        document, args, current_source_hashes=source_hashes
    )
    relevant_paths = [
        relative_path,
        "scripts/fit_esconv_emodynamix_clean.py",
        "scripts/train_esconv_emodynamix_clean.py",
        "open_response_eval/esconv_emodynamix_model.py",
        "open_response_eval/esconv_metrics.py",
    ]
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", *relevant_paths],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ValueError("pilot preregistration or source code is not git tracked")
    clean = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", *relevant_paths],
        check=False,
    )
    if clean.returncode != 0:
        raise ValueError("pilot preregistration or source code has uncommitted changes")
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    preregistration_commit = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", relative_path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not preregistration_commit:
        raise ValueError("pilot preregistration has no committed freeze point")
    return {
        **validated,
        "preregistration_path": str(path),
        "preregistration_sha256": actual_sha,
        "preregistration_commit": preregistration_commit,
        "launch_git_head": head,
        "source_code_contract": source_hashes,
    }


def _resolve_training_device(value: str):
    import torch

    if value == "auto":
        if torch.backends.mps.is_available():
            value = "mps"
        elif torch.cuda.is_available():
            value = "cuda"
        else:
            value = "cpu"
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("requested MPS training device is unavailable")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise ValueError("MPS training requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    elif value == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested CUDA training device is unavailable")
    elif value != "cpu":
        raise ValueError("training device must be auto, cpu, mps, or cuda")
    return torch.device(value)


def _seed_training(seed: int) -> None:
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _linear_warmup_scheduler(optimizer, *, warmup_updates: int, total_updates: int):
    import torch

    if warmup_updates < 0 or total_updates < 1 or warmup_updates > total_updates:
        raise ValueError("scheduler warmup/total updates are invalid")

    def multiplier(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return float(step + 1) / float(warmup_updates)
        decay_steps = max(total_updates - warmup_updates, 1)
        return max(0.0, float(total_updates - step) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def train_emodynamix(
    args,
    *,
    prepared: dict[str, Any],
    feature_run: dict[str, Any],
    execution_provenance: dict[str, Any],
    model_module=None,
) -> dict[str, Any]:
    """Orchestrate a clean smoke or preregistered full dev pilot."""

    import torch
    from torch.utils.data import DataLoader

    execution_audit = validate_training_execution(
        args,
        prepared=prepared,
        feature_run=feature_run,
        execution_provenance=execution_provenance,
    )
    launch_source_hashes = current_pilot_source_hashes()
    preregistered_source_hashes = execution_provenance.get("source_code_contract")
    if args.mode == "pilot" and preregistered_source_hashes != launch_source_hashes:
        raise ValueError("pilot source code changed after preregistration validation")
    _seed_training(args.seed)
    device = _resolve_training_device(args.device)
    if model_module is None:
        from open_response_eval import esconv_emodynamix_model as model_module

    features = feature_run["features_by_input_sha256"]
    if args.mode == "smoke":
        training_records = _select_smoke_records(
            prepared["train_records"], args.smoke_train_rows
        )
        development_records = _select_smoke_records(
            prepared["dev_records"], args.smoke_dev_rows
        )
    else:
        training_records = list(prepared["train_records"])
        development_records = list(prepared["dev_records"])
    execution_audit["executed_training_records"] = len(training_records)
    execution_audit["executed_development_records"] = len(development_records)
    train_dataset = CleanEmoDynamiXTrainingDataset(
        training_records, features
    )
    dev_dataset = CleanEmoDynamiXTrainingDataset(development_records, features)
    collator = make_training_collator(model_module=model_module, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    model, model_receipt = model_module.load_clean_emodynamix_model(
        base_model_path=Path(args.base_model_dir),
        expected_base_tree_sha256=args.expected_base_tree_sha256,
        seed=args.seed,
    )
    if model_receipt.get("task_checkpoint_sha256") is not None:
        raise ValueError("clean model was initialized from a task checkpoint")
    model.to(device)
    optimizer_groups, optimizer_audit = optimizer_parameter_groups(
        model, weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    planned_updates = min(args.max_updates, args.epochs * updates_per_epoch)
    scheduler = _linear_warmup_scheduler(
        optimizer,
        warmup_updates=min(args.warmup_updates, planned_updates),
        total_updates=planned_updates,
    )
    best_state: dict[str, Any] = {}

    def capture_best(**kwargs) -> None:
        del kwargs
        best_state.clear()
        best_state.update(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        )

    training_result = run_training_epochs(
        model,
        train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_updates=args.max_updates,
        max_grad_norm=args.max_grad_norm,
        loss_mode=args.loss,
        class_counts=CLASS_COUNTS,
        author_weight_temperature=args.author_weight_temperature,
        class_balance_beta=args.class_balance_beta,
        logit_adjustment_tau=args.logit_adjustment_tau,
        checkpoint_callback=capture_best,
    )
    if not best_state:
        raise ValueError("training did not capture a best dev state")
    if current_pilot_source_hashes() != launch_source_hashes:
        raise ValueError("training source code changed during execution")
    run_status = (
        "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE"
        if args.mode == "smoke"
        else "DEV_PILOT_CANDIDATE_NOT_FROZEN_TESTED"
    )
    manifest_base = {
        "protocol_id": execution_provenance.get("protocol_id"),
        "execution_provenance": execution_provenance,
        "execution_audit": execution_audit,
        "data_source_audit": prepared["source_audit"],
        "data_overlap_audit": prepared["audit"],
        "feature_run": feature_run["audit"],
        "model_initialization": model_receipt,
        "source_code_contract": launch_source_hashes,
        "task_checkpoint_sha256": None,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "parameter_groups": optimizer_audit,
        },
        "training_configuration": _json_safe(vars(args)),
        "training_result": training_result,
        "best_epoch": training_result["best_epoch"],
        "best_dev": training_result["best_dev"],
        "device": str(device),
        "determinism": {
            "mode": f"best_effort_{device.type}",
            "torch_deterministic_algorithms_warn_only": True,
            "bitwise_reproducible_not_guaranteed": True,
        },
        "class_counts": list(CLASS_COUNTS),
        "loss_mode": args.loss,
        "frozen_test_accessed": False,
    }
    artifacts = write_training_artifacts(
        Path(args.output_dir),
        model_state_dict=best_state,
        manifest_base=manifest_base,
        run_status=run_status,
    )
    return {"training": training_result, **artifacts}


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
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument(
        "--expected-base-tree-sha256",
        default=EXPECTED_BASE_MODEL_TREE_SHA256,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--preregistration-sha256")
    parser.add_argument("--arm-id")
    parser.add_argument("--mode", choices=("audit", "smoke", "pilot"), default="audit")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="mps")
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
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--smoke-train-rows", type=int, default=8)
    parser.add_argument("--smoke-dev-rows", type=int, default=8)
    parser.add_argument("--author-weight-temperature", type=float, default=1.75)
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from scripts import train_esconv_emodynamix_clean as data_contract

    prepared = data_contract.prepare_canonical_data(args.train_file, args.dev_file)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DATA_READY",
                    "mode": args.mode,
                    "training_records": len(prepared["train_records"]),
                    "development_records": len(prepared["dev_records"]),
                    "source_audit": prepared["source_audit"],
                    "overlap_audit": prepared["audit"],
                    "model_loaded": False,
                    "optimizer_created": False,
                    "artifact_written": False,
                    "frozen_test_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "audit":
        raise SystemExit("--execute requires --mode smoke or --mode pilot")
    if args.mode == "pilot":
        execution_provenance = load_pilot_execution_provenance(args)
        feature_contract = execution_provenance["feature_run_contract"]
    else:
        feature_contract = derive_smoke_feature_contract(args.feature_run_dir)
        execution_provenance = {
            "protocol_id": "developmental-smoke-only",
            "execution_validated": False,
            "feature_run_summary_sha256": feature_contract[
                "feature_run_summary_sha256"
            ],
            "frozen_test_accessed": False,
        }
    features, feature_audit = load_verified_feature_run(
        args.feature_run_dir,
        prepared=prepared,
        preregistration_contract=feature_contract,
    )
    result = train_emodynamix(
        args,
        prepared=prepared,
        feature_run={
            "features_by_input_sha256": features,
            "audit": feature_audit,
        },
        execution_provenance=execution_provenance,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
