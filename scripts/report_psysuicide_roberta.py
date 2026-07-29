#!/usr/bin/env python3
"""Validate a local three-seed run and emit an aggregate-only public report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "reports" / "psysuicide-roberta-v1.prereg.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def build_report(aggregate: dict, prereg: dict, weight_root: Path | None) -> dict:
    plan = aggregate["plan"]
    expected_training = prereg["training"]
    expected_partition = prereg["partition"]
    if plan["mode"] != "full":
        raise ValueError("public selection report requires a full run")
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
    }
    for field, expected in expected_plan.items():
        if plan[field] != expected:
            raise ValueError(f"plan mismatch for {field}: {plan[field]} != {expected}")
    for field in ("optimization_commitment_sha256", "holdout_commitment_sha256"):
        if plan["partition"][field] != expected_partition[field]:
            raise ValueError(f"partition mismatch for {field}")
    if aggregate.get("holdout_scored") is not False:
        raise ValueError("holdout must remain unscored during validation selection")

    seeds = sorted(aggregate["seeds"], key=lambda row: row["seed"])
    if [row["seed"] for row in seeds] != expected_training["seeds"]:
        raise ValueError("aggregate seed results differ from preregistration")
    for row in seeds:
        if row["mode"] != "full":
            raise ValueError("mixed run mode in seed results")
        if row["train_rows"] != expected_partition["optimization_rows"]:
            raise ValueError("seed train row count mismatch")
        if row["eval_rows"] != expected_partition["official_valid_rows"]:
            raise ValueError("seed valid row count mismatch")
        if row["partition"]["holdout_commitment_sha256"] != expected_partition["holdout_commitment_sha256"]:
            raise ValueError("seed holdout commitment mismatch")

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
    failures = []
    for label, reference_f1 in reference["per_class_f1_for_support_at_least_10"].items():
        delta = per_class[label]["mean_f1"] - reference_f1
        per_class[label]["reference_f1"] = reference_f1
        per_class[label]["delta_vs_reference"] = delta
        if delta < -0.10:
            failures.append(label)
    primary_pass = means["macro_f1"] > reference["macro_f1"]
    regression_pass = not failures
    advances = primary_pass and regression_pass
    best = max(seeds, key=lambda row: (row["metrics"]["macro_f1"], -row["seed"]))

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
        selected_checkpoint = {
            "seed": best["seed"],
            "valid_macro_f1": best["metrics"]["macro_f1"],
            "weight_filename": weight_path.name,
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": sha256_file(weight_path),
            "status": "frozen before holdout access; weights remain local and ignored",
        }

    return {
        "schema_version": 1,
        "scope": "PsySUICIDE supervised v1 three-seed official-valid selection",
        "execution_commit": aggregate["execution_commit"],
        "preregistration": "reports/psysuicide-roberta-v1.prereg.json",
        "base_model": prereg["implementation"]["base_model"],
        "base_model_revision": prereg["implementation"]["base_model_revision"],
        "partition": {
            "optimization_rows": expected_partition["optimization_rows"],
            "optimization_commitment_sha256": expected_partition["optimization_commitment_sha256"],
            "official_valid_rows": expected_partition["official_valid_rows"],
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
        "publishing_boundary": "aggregate metrics and hashes only; text, ids, row-level predictions, weights, and trainer state remain private and ignored",
    }


def selftest() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    print("PsySUICIDE RoBERTa report selftest PASS: aggregate mean helper")


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
    report = build_report(
        aggregate,
        prereg,
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
