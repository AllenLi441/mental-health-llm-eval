#!/usr/bin/env python3
"""Train/dev-only optimizer primitives for clean EmoDynamiX retraining.

The command defaults to a read-only audit.  It intentionally exposes no test
input and no task-checkpoint initialization option.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "tmp/official_benchmarks/esconv/codes/dataset"
DEFAULT_FEATURE_RUN_DIR = (
    ROOT / "tmp/emodynamix-clean-features/canonical-train-dev-v1"
)
DEFAULT_OUTPUT_DIR = ROOT / "tmp/emodynamix-clean-training"

CLASS_COUNTS = (706, 820, 1_523, 1_470, 1_392, 508, 524, 1_490)
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
        if not math.isfinite(author_weight_temperature) or author_weight_temperature <= 0:
            raise ValueError("author_weight_temperature must be positive and finite")
        inverse_frequency = (counts.sum() / counts.numel()) / counts
        weights = torch.softmax(inverse_frequency / author_weight_temperature, dim=0)
        per_row = functional.cross_entropy(logits, labels, reduction="none")
        target_weights = weights[labels]
        return (per_row * target_weights).sum(), target_weights.sum()

    if mode == "class_balanced":
        if not 0.0 < class_balance_beta < 1.0:
            raise ValueError("class_balance_beta must be between zero and one")
        raw_weights = (1.0 - class_balance_beta) / (
            1.0 - torch.pow(class_balance_beta, counts)
        )
        weights = raw_weights / raw_weights.mean()
        numerator = functional.cross_entropy(
            logits, labels, weight=weights, reduction="sum"
        )
        return numerator, weights[labels].sum()

    if mode == "logit_adjusted":
        if not math.isfinite(logit_adjustment_tau):
            raise ValueError("logit_adjustment_tau must be finite")
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
        -float(candidate["loss"]),
    )
    incumbent_key = (
        float(incumbent["macro_f1"]),
        float(incumbent["accuracy"]),
        -float(incumbent["loss"]),
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
    return groups, {
        "rule": "all trainable parameters except bias and LayerNorm weights decay",
        "decay_parameter_names": sorted(decay_names),
        "no_decay_parameter_names": sorted(no_decay_names),
        "decay_parameter_count": len(decay_names),
        "no_decay_parameter_count": len(no_decay_names),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or fit a clean train/dev-only EmoDynamiX strategy policy"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_DATA_ROOT / "trainWithStrategy_short.tsv",
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=DEFAULT_DATA_ROOT / "validWithStrategy_short.tsv",
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
