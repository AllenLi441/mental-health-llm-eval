#!/usr/bin/env python3
"""Auditable ESConv Joint strategy evaluation with the corrected token map.

This evaluator is deliberately independent from label-list positions. It maps
each report label to its exact checkpoint token string, resolves token IDs from
the tokenizer, and gathers the eight first-step logits with ``index_select``.
"""

from __future__ import print_function

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import transformers
from transformers import (
    BlenderbotSmallConfig,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
)

from open_response_eval.esconv_metrics import compute_classification_metrics


OFFICIAL_REPO_COMMIT = "f262d062ad74cb39b17ea476facc81568ddcba24"

REPORT_LABELS = [
    "Questions",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Other",
]

TOKEN_BY_LABEL = {
    "Questions": "[Questions]",
    "Restatement or Paraphrasing": "[Restatement or Paraphrasing]",
    "Reflection of feelings": "[Reflection of feelings]",
    "Self-disclosure": "[Self-disclosure]",
    "Affirmation and Reassurance": "[Affirmation and Reassurance]",
    "Providing Suggestions": "[Providing Suggestions]",
    "Information": "[Information]",
    "Other": "[Other]",
}

TOKENS_IN_CHECKPOINT_ORDER = [
    "[Questions]",
    "[Reflection of feelings]",
    "[Information]",
    "[Restatement or Paraphrasing]",
    "[Other]",
    "[Self-disclosure]",
    "[Affirmation and Reassurance]",
    "[Providing Suggestions]",
    "[CLS]",
]

EXPECTED_TOKEN_IDS = {
    token: 54_944 + index for index, token in enumerate(TOKENS_IN_CHECKPOINT_ORDER)
}

EXPECTED_DATASET_SHA256 = {
    "trainWithStrategy_short.tsv": "0ecf37462f8e3fa7f1dc5bfbecd70733abb3501c3bf83b27a957bec5a926ec21",
    "devWithStrategy_short.tsv": "625b511f40cf9a9285582e808e6bbb4c2f35d0b0cd063f3709db430b8d0c3bdc",
    "testWithStrategy_short.tsv": "b85ae888bf747cefa54bba2a6c3e2f6ccb4c1005d4e0b6d1d3be3823cf040aef",
}

MODERN_UNUSED_CHECKPOINT_KEYS = {
    "model.encoder.wre.weight",
    "model.encoder.wte.weight",
    "model.decoder.wre.weight",
    "model.decoder.wte.weight",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inference_context():
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()


def load_torch_state(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.7 does not have weights_only.
        return torch.load(str(path), map_location="cpu")


def token_is_missing(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id is None or token_id == tokenizer.unk_token_id


def prepare_tokenizer(tokenizer_dir):
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(str(tokenizer_dir))
    missing = [
        token for token in TOKENS_IN_CHECKPOINT_ORDER if token_is_missing(tokenizer, token)
    ]
    if missing:
        if missing != TOKENS_IN_CHECKPOINT_ORDER:
            raise RuntimeError(
                "Tokenizer contains only part of the frozen token map: %s" % missing
            )
        tokenizer.add_tokens(TOKENS_IN_CHECKPOINT_ORDER[:-1])
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})

    actual_mapping = {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in TOKENS_IN_CHECKPOINT_ORDER
    }
    if actual_mapping != EXPECTED_TOKEN_IDS:
        raise RuntimeError(
            "Strategy token mapping mismatch. expected=%s actual=%s"
            % (EXPECTED_TOKEN_IDS, actual_mapping)
        )
    if token_is_missing(tokenizer, "[Questions]"):
        raise RuntimeError("[Questions] is missing")
    if not token_is_missing(tokenizer, "[Question]"):
        raise RuntimeError("Forbidden singular token [Question] is present")
    if not token_is_missing(tokenizer, "[Others]"):
        raise RuntimeError("Forbidden plural token [Others] is present")
    return tokenizer, actual_mapping


def resolve_weight_file(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        return checkpoint
    candidates = [
        checkpoint / "model.safetensors",
        checkpoint / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No checkpoint weight file found in %s" % checkpoint)


def load_model(checkpoint, config_dir, tokenizer, backend):
    checkpoint = Path(checkpoint)
    ignored_keys = []
    if checkpoint.is_dir():
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(str(checkpoint))
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            raise RuntimeError(
                "Checkpoint vocabulary size %d does not match tokenizer size %d"
                % (model.get_input_embeddings().weight.shape[0], len(tokenizer))
            )
        return model, ignored_keys

    config = BlenderbotSmallConfig.from_pretrained(str(config_dir))
    config.vocab_size = len(tokenizer)
    model = BlenderbotSmallForConditionalGeneration(config)
    state = load_torch_state(checkpoint)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing:
        raise RuntimeError("Missing checkpoint keys: %s" % missing)
    if unexpected:
        if backend == "modern" and set(unexpected) == MODERN_UNUSED_CHECKPOINT_KEYS:
            ignored_keys = unexpected
        else:
            raise RuntimeError("Unexpected checkpoint keys: %s" % unexpected)
    return model, ignored_keys


def parse_target_segment(segment):
    fields = segment.strip().split(None, 3)
    if len(fields) != 4:
        raise ValueError("Target segment does not contain three metadata fields")
    match = re.match(r"^\[([^\]]+)\](?:\s|$)", fields[3])
    if match is None:
        raise ValueError("Target does not begin with a strategy token")
    label = match.group(1)
    if label not in REPORT_LABELS:
        raise ValueError("Unknown gold strategy %r" % label)
    return label


def parse_context(parts, tokenizer, max_input_length):
    eos = tokenizer.eos_token_id
    input_ids = []
    for segment in parts:
        fields = segment.strip().split(None, 3)
        if len(fields) != 4:
            raise ValueError("Context segment does not contain three metadata fields")
        input_ids.extend(tokenizer.encode(fields[3]))
        input_ids.append(eos)
    if len(input_ids) > max_input_length:
        cut = input_ids.index(eos, -max_input_length)
        input_ids = input_ids[cut:]
    return input_ids


def load_examples(test_file, tokenizer, max_input_length, limit=None):
    examples = []
    invalid = []
    with Path(test_file).open(encoding="utf-8") as handle:
        for line_number, row in enumerate(handle, start=1):
            if limit is not None and len(examples) + len(invalid) >= limit:
                break
            item_id = line_number - 1
            try:
                parts = row.rstrip("\n").split("EOS")
                if len(parts) < 2:
                    raise ValueError("No EOS separator")
                gold = parse_target_segment(parts[-1])
                input_ids = parse_context(parts[:-1], tokenizer, max_input_length)
                examples.append(
                    {"item_id": item_id, "line_number": line_number, "gold": gold, "input_ids": input_ids}
                )
            except Exception as error:
                invalid.append(
                    {
                        "item_id": item_id,
                        "line_number": line_number,
                        "gold": None,
                        "prediction": None,
                        "correct": False,
                        "invalid": True,
                        "error": str(error),
                    }
                )
    return examples, invalid


def predict_examples(model, tokenizer, examples, batch_size, device):
    strategy_token_ids = torch.tensor(
        [
            tokenizer.convert_tokens_to_ids(TOKEN_BY_LABEL[label])
            for label in REPORT_LABELS
        ],
        dtype=torch.long,
        device=device,
    )
    model.to(device)
    model.eval()
    predictions = []
    with inference_context():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            max_length = max(len(item["input_ids"]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, item in enumerate(batch):
                values = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
                input_ids[row_index, : values.numel()] = values
                attention_mask[row_index, : values.numel()] = 1
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
            strategy_logits = outputs.logits[:, 0, :].index_select(
                dim=-1, index=strategy_token_ids
            )
            prediction_indices = strategy_logits.argmax(dim=-1).cpu().tolist()
            score_rows = strategy_logits.detach().cpu().tolist()
            for item, prediction_index, scores in zip(batch, prediction_indices, score_rows):
                prediction = REPORT_LABELS[prediction_index]
                predictions.append(
                    {
                        "item_id": item["item_id"],
                        "line_number": item["line_number"],
                        "gold": item["gold"],
                        "prediction": prediction,
                        "correct": prediction == item["gold"],
                        "invalid": False,
                        "token": TOKEN_BY_LABEL[prediction],
                        "token_id": int(strategy_token_ids[prediction_index].item()),
                        "strategy_logits": {
                            label: round(float(score), 8)
                            for label, score in zip(REPORT_LABELS, scores)
                        },
                    }
                )
            done = min(start + batch_size, len(examples))
            print("\rEvaluated %d/%d" % (done, len(examples)), end="", flush=True)
    if examples:
        print()
    return predictions


def compute_metrics(records):
    return compute_classification_metrics(records, REPORT_LABELS)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item["item_id"]):
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def update_manifest(path, run_name, summary, summary_output):
    if path is None:
        return
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.setdefault("evaluations", {})[run_name] = {
        "status": "COMPLETE",
        "summary": str(Path(summary_output).resolve()),
        "metrics": summary["metrics"],
        "checkpoint_sha256": summary["checkpoint"]["sha256"],
        "dataset_sha256": summary["dataset"]["sha256"],
        "runtime": summary["runtime"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    project = Path(__file__).resolve().parents[1]
    official = project / "tmp/official_benchmarks/esconv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=official / "codes/blender-small")
    parser.add_argument("--config-dir", type=Path, default=official / "codes/blender-small")
    parser.add_argument("--test-file", type=Path, default=official / "codes/dataset/testWithStrategy_short.tsv")
    parser.add_argument("--official-repo", type=Path, default=official)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--backend", choices=("modern", "legacy"), default="modern")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    commit = git_commit(args.official_repo)
    if commit != OFFICIAL_REPO_COMMIT:
        raise RuntimeError(
            "Official checkout commit mismatch: expected %s, got %s"
            % (OFFICIAL_REPO_COMMIT, commit)
        )
    dataset_hash = sha256_file(args.test_file)
    expected_hash = EXPECTED_DATASET_SHA256.get(args.test_file.name)
    if expected_hash and dataset_hash != expected_hash:
        raise RuntimeError(
            "Dataset hash mismatch: expected %s, got %s" % (expected_hash, dataset_hash)
        )

    tokenizer, token_mapping = prepare_tokenizer(args.tokenizer_dir)
    model, ignored_keys = load_model(args.checkpoint, args.config_dir, tokenizer, args.backend)
    device = select_device(args.device)
    examples, invalid = load_examples(
        args.test_file, tokenizer, args.max_input_length, args.limit
    )
    predicted = predict_examples(model, tokenizer, examples, args.batch_size, device)
    records = sorted(predicted + invalid, key=lambda item: item["item_id"])
    metrics = compute_metrics(records)
    write_jsonl(args.predictions_output, records)

    weight_file = resolve_weight_file(args.checkpoint)
    summary = {
        "audit_status": "COMPLETE",
        "run_name": args.run_name,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "paper": "Towards Emotional Support Dialog Systems",
        "task": "8-way next-support-strategy prediction",
        "protocol_note": "Post-hoc strategy evaluation; ACL 2021 did not publish strategy Accuracy.",
        "official_repo": {
            "path": str(args.official_repo.resolve()),
            "commit": commit,
        },
        "dataset": {
            "path": str(args.test_file.resolve()),
            "sha256": dataset_hash,
            "limit": args.limit,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "weight_file": str(weight_file.resolve()),
            "sha256": sha256_file(weight_file),
            "ignored_modern_only_keys": ignored_keys,
        },
        "tokenizer": {
            "path": str(args.tokenizer_dir.resolve()),
            "mapping": token_mapping,
            "report_label_to_token": TOKEN_BY_LABEL,
        },
        "decision_rule": "first decoder step; explicit token IDs; index_select; top-1 argmax",
        "metrics": metrics,
        "predictions": str(args.predictions_output.resolve()),
        "runtime": {
            "backend": args.backend,
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "batch_size": args.batch_size,
            "elapsed_seconds": round(time.time() - started, 3),
            "inference_context": "torch.inference_mode" if hasattr(torch, "inference_mode") else "torch.no_grad (PyTorch 1.7 compatibility)",
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    update_manifest(args.manifest, args.run_name, summary, args.summary_output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Summary: %s" % args.summary_output)
    print("Predictions: %s" % args.predictions_output)


if __name__ == "__main__":
    main()
