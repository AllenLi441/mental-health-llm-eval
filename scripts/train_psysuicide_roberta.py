#!/usr/bin/env python3
"""Leakage-safe supervised PsySUICIDE classifier training.

Dry-run by default. Full mode trains three declared seeds on only the committed
9,342-row optimization partition and evaluates on official valid. The frozen
2,329-row train holdout is never tokenized, sampled, scored, or used for model
selection. Model weights and trainer state stay under ignored results/.
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
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "psysuicide-roberta"
COMMITMENT = ROOT / "reports" / "psysuicide-v2-train-holdout.commitment.json"
DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext-large"
PARTITION_SEED = "psysuicide-model-optimization-v2-2026-07-28"
LABELS = [
    "与自杀/自伤/攻击行为无关", "被动自杀意图", "主动自杀意图", "关于自杀的探索",
    "自杀计划", "自杀准备行为", "自杀未遂", "自伤意图", "自伤行为",
    "用户攻击行为", "他人攻击行为",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_row(row: dict) -> str:
    return json.dumps(
        {"idx": str(row["idx"]), "labels": row["labels"], "text": str(row["text"])},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def commitment(digests: list[str]) -> str:
    return sha256_bytes("\n".join(sorted(digests)).encode())


def partition_train(rows: list[dict]) -> tuple[list[dict], dict]:
    allowed = set(LABELS)
    retained = []
    dropped_multi = 0
    dropped_unknown = 0
    for row in rows:
        if not isinstance(row.get("labels"), list) or len(row["labels"]) != 1:
            dropped_multi += 1
            continue
        label = row["labels"][0]
        if label not in allowed:
            dropped_unknown += 1
            continue
        digest = sha256_bytes(canonical_row(row).encode())
        retained.append({
            "row": row,
            "label": label,
            "row_digest": digest,
            "rank_digest": sha256_bytes(f"{PARTITION_SEED}\0{digest}".encode()),
        })
    if len({entry["row_digest"] for entry in retained}) != len(retained):
        raise SystemExit("duplicate canonical train rows")
    optimization = []
    holdout_digests = []
    per_label = {}
    for label in LABELS:
        group = sorted(
            (entry for entry in retained if entry["label"] == label),
            key=lambda entry: (entry["rank_digest"], entry["row_digest"]),
        )
        holdout_n = max(1, math.floor(len(group) * 0.2))
        holdout_digests.extend(entry["row_digest"] for entry in group[:holdout_n])
        optimization.extend(entry["row"] for entry in group[holdout_n:])
        per_label[label] = {
            "retained": len(group),
            "optimization": len(group) - holdout_n,
            "holdout": holdout_n,
        }
    manifest = {
        "retained_single_label_rows": len(retained),
        "optimization_rows": len(optimization),
        "holdout_rows": len(holdout_digests),
        "dropped_multi_label_rows": dropped_multi,
        "dropped_unknown_label_rows": dropped_unknown,
        "per_label": per_label,
        "all_rows_commitment_sha256": commitment(
            [entry["row_digest"] for entry in retained]
        ),
        "optimization_commitment_sha256": commitment(
            [
                sha256_bytes(canonical_row(row).encode())
                for row in optimization
            ]
        ),
        "holdout_commitment_sha256": commitment(holdout_digests),
    }
    return optimization, manifest


def load_data(dataset_root: Path) -> tuple[list[dict], list[dict], dict]:
    train_path = dataset_root / "PsySUICIDE" / "repo" / "train.json"
    valid_path = dataset_root / "PsySUICIDE" / "repo" / "valid.json"
    raw_train = train_path.read_bytes()
    train_rows = json.loads(raw_train)
    optimization, manifest = partition_train(train_rows)
    frozen = json.loads(COMMITMENT.read_text(encoding="utf-8"))
    checks = {
        "dataset_file_sha256": sha256_bytes(raw_train),
        "retained_single_label_rows": manifest["retained_single_label_rows"],
        "optimization_rows": manifest["optimization_rows"],
        "holdout_rows": manifest["holdout_rows"],
        "all_rows_commitment_sha256": manifest["all_rows_commitment_sha256"],
        "optimization_commitment_sha256": manifest["optimization_commitment_sha256"],
        "holdout_commitment_sha256": manifest["holdout_commitment_sha256"],
    }
    for field, value in checks.items():
        if frozen.get(field) != value:
            raise SystemExit(f"partition commitment mismatch for {field}")

    valid_rows = []
    for row in json.loads(valid_path.read_text(encoding="utf-8")):
        if (
            isinstance(row.get("labels"), list)
            and len(row["labels"]) == 1
            and row["labels"][0] in LABELS
        ):
            valid_rows.append(row)
    if len(valid_rows) != 1459:
        raise SystemExit(f"expected 1,459 official-valid rows, got {len(valid_rows)}")
    return optimization, valid_rows, {
        **checks,
        "per_label": manifest["per_label"],
        "valid_rows": len(valid_rows),
        "holdout_access": "partition hashes verified only; holdout rows not returned, tokenized, sampled, scored, or selected on",
    }


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_ids = [LABELS.index(row["labels"][0]) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            str(self.rows[index]["text"]),
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = self.label_ids[index]
        return encoded


class BalancedTrainer(Trainer):
    def __init__(self, *args, sample_weights, sampler_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
        self.sampler_seed = sampler_seed

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        generator = torch.Generator()
        generator.manual_seed(self.sampler_seed)
        return WeightedRandomSampler(
            self.sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )


def class_and_sample_weights(label_ids: list[int]) -> tuple[torch.Tensor, list[float]]:
    counts = Counter(label_ids)
    # Split imbalance correction between the loss and sampler. Each component
    # uses inverse fourth-root frequency, so their combined pressure is roughly
    # inverse square-root rather than the unstable full inverse frequency.
    raw = np.array(
        [(len(label_ids) / (len(LABELS) * counts[index])) ** 0.25 for index in range(len(LABELS))],
        dtype=np.float32,
    )
    raw = np.clip(raw / raw.mean(), 0.35, 4.0)
    sample_weights = [float(raw[label_id]) for label_id in label_ids]
    return torch.tensor(raw, dtype=torch.float32), sample_weights


def focal_loss(class_weights: torch.Tensor, gamma: float):
    def compute(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits
        weights = class_weights.to(logits.device)
        unweighted = F.cross_entropy(logits, labels, reduction="none")
        weighted = F.cross_entropy(logits, labels, weight=weights, reduction="none")
        modulation = (1.0 - torch.exp(-unweighted)).pow(gamma)
        return (modulation * weighted).mean()

    return compute


def metrics_from_predictions(gold: np.ndarray, predicted: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, predicted, average="weighted", zero_division=0)),
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


def train_seed(args, seed, optimization, valid_rows, manifest):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir = RESULTS / args.run_id / f"seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
    )
    model_commit = getattr(model.config, "_commit_hash", None)
    train_rows = optimization
    eval_rows = valid_rows
    max_steps = -1
    epochs = args.epochs
    if args.mode == "smoke":
        # Deterministic balanced-ish subset without exposing any row.
        train_rows = sorted(
            optimization,
            key=lambda row: sha256_bytes(f"{seed}\0{canonical_row(row)}".encode()),
        )[:128]
        eval_rows = sorted(
            valid_rows,
            key=lambda row: sha256_bytes(f"{seed}\0{canonical_row(row)}".encode()),
        )[:64]
        max_steps = 2
        epochs = 1.0
    train_dataset = TextDataset(train_rows, tokenizer, args.max_length)
    eval_dataset = TextDataset(eval_rows, tokenizer, args.max_length)
    weights, sample_weights = class_and_sample_weights(train_dataset.label_ids)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=False,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1 if args.mode == "smoke" else 50,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=epochs,
        max_steps=max_steps,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        seed=seed,
        data_seed=seed,
        use_cpu=args.cpu,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=args.mode == "full",
        optim="adamw_torch",
        report_to="none",
        disable_tqdm=False,
        full_determinism=False,
    )
    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_loss_func=focal_loss(weights, args.focal_gamma),
        sample_weights=sample_weights,
        sampler_seed=seed,
    )
    started = time.time()
    train_result = trainer.train()
    prediction = trainer.predict(eval_dataset)
    predicted = np.argmax(prediction.predictions, axis=-1)
    evaluated = metrics_from_predictions(prediction.label_ids, predicted)
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    result = {
        "seed": seed,
        "mode": args.mode,
        "model": args.model,
        "model_revision": model_commit,
        "device": str(trainer.args.device),
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "epochs": epochs,
        "max_steps": max_steps,
        "max_length": args.max_length,
        "train_batch": args.train_batch,
        "eval_batch": args.eval_batch,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "focal_gamma": args.focal_gamma,
        "class_weights": {
            LABELS[index]: float(value) for index, value in enumerate(weights.tolist())
        },
        "sampler": "weighted replacement; inverse fourth-root class frequency",
        "loss": "class-weighted focal loss; inverse fourth-root class frequency",
        "train_runtime_seconds": float(train_result.metrics.get("train_runtime", time.time() - started)),
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "metrics": evaluated,
        "partition": manifest,
        "checkpoint": str((output_dir / "model").relative_to(ROOT)),
        "publishing_boundary": "aggregate metrics and portable code only; weights, text, ids, predictions, and trainer state stay ignored",
    }
    (output_dir / "aggregate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del trainer, model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def selftest():
    rows = []
    for label_index, label in enumerate(LABELS):
        for index in range(10):
            rows.append({
                "idx": f"{label_index}-{index}",
                "labels": [label],
                "text": f"synthetic-{label_index}-{index}",
            })
    optimization, manifest = partition_train(rows)
    assert len(optimization) == 88
    assert manifest["holdout_rows"] == 22
    label_ids = [LABELS.index(row["labels"][0]) for row in optimization]
    weights, sample_weights = class_and_sample_weights(label_ids)
    assert len(weights) == len(LABELS)
    assert len(sample_weights) == len(optimization)
    logits = torch.randn(4, len(LABELS), requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss = focal_loss(weights, 1.5)(type("Output", (), {"logits": logits}), labels)
    assert torch.isfinite(loss)
    loss.backward()
    print("PsySUICIDE RoBERTa trainer selftest PASS: exact partition, balanced sampler weights, focal loss, no model download")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch", type=int, default=2)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if args.mode == "full" and seeds != [42, 43, 44]:
        raise SystemExit("full protocol requires exactly --seeds 42,43,44")
    if not args.run_id or not all(character.isalnum() or character in "._-" for character in args.run_id):
        raise SystemExit("--run-id is required and must use only letters, digits, dot, underscore, or hyphen")
    if not args.dataset_root:
        raise SystemExit("--dataset-root is required; no private dataset path is embedded")
    optimization, valid_rows, manifest = load_data(args.dataset_root.resolve())
    plan = {
        "schema_version": 1,
        "mode": args.mode,
        "run_id": args.run_id,
        "model": args.model,
        "seeds": seeds,
        "train_rows": 128 if args.mode == "smoke" else len(optimization),
        "eval_rows": 64 if args.mode == "smoke" else len(valid_rows),
        "max_length": args.max_length,
        "train_batch": args.train_batch,
        "eval_batch": args.eval_batch,
        "gradient_accumulation": args.gradient_accumulation,
        "epochs": 1 if args.mode == "smoke" else args.epochs,
        "max_steps": 2 if args.mode == "smoke" else -1,
        "partition": manifest,
        "device_gate": {
            "mps_available": torch.backends.mps.is_available(),
            "cpu_forced": args.cpu,
        },
        "dry_run_default": True,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Dry-run only. Add --execute to download/load the model and train.")
        return
    run_root = RESULTS / args.run_id
    if run_root.exists():
        raise SystemExit(f"refusing to overwrite run: {run_root}")
    run_root.mkdir(parents=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    results = []
    for seed in seeds:
        results.append(train_seed(args, seed, optimization, valid_rows, manifest))
    aggregate = {
        "schema_version": 1,
        "execution_commit": revision,
        "plan": plan,
        "seeds": results,
        "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in results])),
        "std_macro_f1": float(np.std([row["metrics"]["macro_f1"] for row in results], ddof=1)) if len(results) > 1 else 0.0,
        "best_seed": max(results, key=lambda row: row["metrics"]["macro_f1"])["seed"],
        "holdout_scored": False,
        "publishing_boundary": "aggregate validation evidence only; local weights and all row-level material stay ignored",
    }
    (run_root / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "mean_macro_f1": aggregate["mean_macro_f1"],
        "std_macro_f1": aggregate["std_macro_f1"],
        "best_seed": aggregate["best_seed"],
        "aggregate": str(run_root / "aggregate.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
