#!/usr/bin/env python3
"""Recompute 8-way ESConv strategy accuracy for the authors' Joint model.

The ACL 2021 paper did not publish strategy-selection accuracy. This script
loads the authors' released ESC_Blender_Strategy checkpoint and evaluates its
first decoder token on the paper's fixed 2,775-example test TSV.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    BlenderbotSmallConfig,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
)


STRATEGY_TOKENS = [
    "[Question]",
    "[Restatement or Paraphrasing]",
    "[Reflection of feelings]",
    "[Self-disclosure]",
    "[Affirmation and Reassurance]",
    "[Providing Suggestions]",
    "[Information]",
    "[Others]",
]

LABEL_ALIASES = {
    "Questions": "Question",
    "Other": "Others",
}

CHECKPOINT_REPO = "lsy641/ESC_Blender_Strategy"
CHECKPOINT_REVISION = "72c5d30a57217f7b44b7dae6d95230241335ec04"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    official_root = project_root / "tmp/official_benchmarks/esconv"
    default_checkpoint = (
        Path.home()
        / ".cache/huggingface/hub/models--lsy641--ESC_Blender_Strategy"
        / "snapshots"
        / CHECKPOINT_REVISION
        / "pytorch_model.bin"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-file",
        type=Path,
        default=official_root / "codes/dataset/testWithStrategy_short.tsv",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=official_root / "codes/blender-small",
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "reports/esconv_original_joint_official_test_20260810.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_label(raw: str) -> str:
    label = LABEL_ALIASES.get(raw, raw)
    token = f"[{label}]"
    if token not in STRATEGY_TOKENS:
        raise ValueError(f"Unknown strategy label: {raw!r}")
    return token


def parse_official_tsv(
    path: Path,
    tokenizer: BlenderbotSmallTokenizer,
    max_input_length: int,
    limit: int | None,
) -> tuple[list[list[int]], list[int]]:
    contexts: list[list[int]] = []
    labels: list[int] = []
    eos = tokenizer.eos_token_id

    with path.open(encoding="utf-8") as handle:
        for row_number, row in enumerate(handle, start=1):
            if limit is not None and len(contexts) >= limit:
                break
            parts = row.rstrip("\n").split("EOS")
            if len(parts) < 2:
                raise ValueError(f"Malformed row {row_number}: no EOS separator")

            input_ids: list[int] = []
            for utterance in parts[:-1]:
                fields = utterance.strip().split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed context in row {row_number}")
                text = " ".join(fields[3:])
                input_ids.extend(tokenizer.encode(text))
                input_ids.append(eos)

            # Match the original implementation: start at the first utterance
            # boundary found inside the final max_input_length-token window.
            if len(input_ids) > max_input_length:
                cut = input_ids.index(eos, -max_input_length)
                input_ids = input_ids[cut:]

            match = re.search(r"\[([^\]]+)\]", parts[-1])
            if match is None:
                raise ValueError(f"Missing gold strategy in row {row_number}")
            gold_token = normalize_label(match.group(1))

            contexts.append(input_ids)
            labels.append(STRATEGY_TOKENS.index(gold_token))

    return contexts, labels


def load_model(
    tokenizer_dir: Path,
    checkpoint: Path,
) -> tuple[BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, list[str]]:
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(tokenizer_dir)
    added = tokenizer.add_tokens(STRATEGY_TOKENS + ["[CLS]"])
    if added != 9 or len(tokenizer) != 54_953:
        raise RuntimeError(
            f"Unexpected tokenizer expansion: added={added}, size={len(tokenizer)}"
        )

    strategy_ids = tokenizer.convert_tokens_to_ids(STRATEGY_TOKENS)
    if strategy_ids != list(range(54_944, 54_952)):
        raise RuntimeError(f"Unexpected strategy token IDs: {strategy_ids}")

    config = BlenderbotSmallConfig.from_pretrained(tokenizer_dir)
    config.vocab_size = len(tokenizer)
    model = BlenderbotSmallForConditionalGeneration(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state_dict, strict=False)

    expected_unused = {
        "model.encoder.wre.weight",
        "model.encoder.wte.weight",
        "model.decoder.wre.weight",
        "model.decoder.wte.weight",
    }
    if incompatible.missing_keys or set(incompatible.unexpected_keys) != expected_unused:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    return tokenizer, model, incompatible.unexpected_keys


def predict(
    model: BlenderbotSmallForConditionalGeneration,
    tokenizer: BlenderbotSmallTokenizer,
    contexts: list[list[int]],
    batch_size: int,
    device: torch.device,
) -> list[int]:
    model.to(device)
    model.eval()
    predictions: list[int] = []
    strategy_start = tokenizer.convert_tokens_to_ids(STRATEGY_TOKENS[0])

    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            max_length = max(map(len, batch))
            input_ids = torch.full(
                (len(batch), max_length),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, ids in enumerate(batch):
                length = len(ids)
                input_ids[index, :length] = torch.tensor(ids, device=device)
                attention_mask[index, :length] = 1

            decoder_input_ids = torch.full(
                (len(batch), 1),
                tokenizer.bos_token_id,
                dtype=torch.long,
                device=device,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            )
            strategy_logits = outputs.logits[:, 0, strategy_start : strategy_start + 8]
            predictions.extend(strategy_logits.argmax(dim=-1).cpu().tolist())

            done = min(start + batch_size, len(contexts))
            print(f"\rEvaluated {done}/{len(contexts)}", end="", flush=True)
    print()
    return predictions


def main() -> None:
    args = parse_args()
    started = time.time()
    device = select_device(args.device)
    tokenizer, model, ignored_checkpoint_keys = load_model(
        args.tokenizer_dir, args.checkpoint
    )
    contexts, labels = parse_official_tsv(
        args.test_file,
        tokenizer,
        args.max_input_length,
        args.limit,
    )
    predictions = predict(model, tokenizer, contexts, args.batch_size, device)

    class_ids = list(range(len(STRATEGY_TOKENS)))
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    per_class = {}
    for class_id, token in enumerate(STRATEGY_TOKENS):
        gold_count = int(sum(label == class_id for label in labels))
        correct = int(matrix[class_id, class_id])
        per_class[token] = {
            "gold_count": gold_count,
            "predicted_count": int(sum(pred == class_id for pred in predictions)),
            "correct": correct,
            "recall": correct / gold_count if gold_count else None,
        }

    result = {
        "benchmark": "ESConv",
        "task": "8-way next-support-strategy prediction",
        "model": "ESC_Blender_Strategy (ACL 2021 Joint variant)",
        "checkpoint": {
            "repo": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
            "path": str(args.checkpoint),
            "ignored_unused_role_turn_embedding_keys": ignored_checkpoint_keys,
        },
        "protocol": {
            "test_file": str(args.test_file),
            "examples": len(labels),
            "paper_split": "fixed original 20% test TSV from the 1,053-dialogue release",
            "context": "all preceding utterances, truncated at an utterance boundary to <=512 tokens",
            "decision_rule": "argmax of the first decoder-step logits over the 8 strategy tokens",
            "note": "Post-hoc strategy evaluation; ACL 2021 did not publish strategy Accuracy.",
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "elapsed_seconds": round(time.time() - started, 3),
            "torch": torch.__version__,
        },
        "metrics": {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(
                f1_score(labels, predictions, labels=class_ids, average="macro")
            ),
            "weighted_f1": float(
                f1_score(labels, predictions, labels=class_ids, average="weighted")
            ),
            "correct": int(np.sum(np.asarray(labels) == np.asarray(predictions))),
            "total": len(labels),
        },
        "gold_distribution": {
            STRATEGY_TOKENS[index]: int(count)
            for index, count in sorted(Counter(labels).items())
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": STRATEGY_TOKENS,
            "rows_gold_columns_predicted": matrix.tolist(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
