#!/usr/bin/env python3
"""Materialize a complete, quarantined research training export from A3 candidates.

The export is intentionally *not* accepted by the production/research trainer:
rights, human safety review, full contamination review and token-level masking are
still open. This command creates a reproducible train/development structure only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SYSTEM = {
    "zh": "你是非医疗心理支持助手。请尊重且不评判地回应，帮助用户澄清感受并考虑低风险的下一步。不要诊断、开药或声称替代专业帮助。",
    "en": "You are a non-medical emotional-support assistant. Respond with respect and no judgment, help clarify feelings, and consider a low-risk next step. Do not diagnose, prescribe, or claim to replace professional care.",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def materialize(candidate_root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output must be a new directory")
    inputs = sorted(candidate_root.glob("*.candidates.jsonl"))
    if not inputs:
        raise ValueError("candidate files not found")
    output.mkdir(parents=True)
    counts = Counter()
    per_source = {}
    groups = {"train": set(), "development": set()}
    input_hashes = {p.name: file_sha256(p) for p in inputs}
    with (output / "quarantine_index.jsonl").open("w", encoding="utf-8") as quarantine, \
         (output / "quarantine_records.jsonl").open("w", encoding="utf-8") as quarantine_records, \
         (output / "train.jsonl").open("w", encoding="utf-8") as train, \
         (output / "development.jsonl").open("w", encoding="utf-8") as development:
        for path in inputs:
            source = path.name.split(".candidates.jsonl", 1)[0]
            local = Counter()
            with path.open(encoding="utf-8") as source_handle:
                for line_no, line in enumerate(source_handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    counts["candidate_rows_seen"] += 1
                    local["candidate_rows_seen"] += 1
                    reason = None
                    if row.get("content_state") != "CANDIDATE_NOT_FROZEN":
                        reason = "STATE_NOT_CANDIDATE_NOT_FROZEN"
                    elif row.get("product_training_approved") is not False:
                        reason = "APPROVAL_FIELD_NOT_FALSE"
                    elif row.get("human_review_status") != "NOT_VERIFIED":
                        reason = "UNEXPECTED_REVIEW_STATUS"
                    elif row.get("review_flags"):
                        reason = "REVIEW_FLAGS_PRESENT"
                    elif not row.get("source_group_id"):
                        reason = "MISSING_VERIFIED_SOURCE_GROUP"
                    elif not isinstance(row.get("messages"), list) or row.get("target_message_index") != len(row["messages"]) - 1:
                        reason = "TARGET_OR_MESSAGES_INVALID"
                    if reason:
                        counts["quarantined"] += 1; local["quarantined"] += 1
                        write(quarantine, {"source": source, "source_row_1based": line_no, "record_id": row.get("record_id"), "reason": reason})
                        row_with_reason = dict(row); row_with_reason["quarantine_reason"] = reason
                        write(quarantine_records, row_with_reason)
                        continue
                    messages = row["messages"]
                    prompt_messages = messages[:-1]
                    if not prompt_messages or prompt_messages[-1].get("role") != "user":
                        reason = "PROMPT_DOES_NOT_END_IN_USER"
                        counts["quarantined"] += 1; local["quarantined"] += 1
                        write(quarantine, {"source": source, "source_row_1based": line_no, "record_id": row.get("record_id"), "reason": reason})
                        row_with_reason = dict(row); row_with_reason["quarantine_reason"] = reason
                        write(quarantine_records, row_with_reason)
                        continue
                    if prompt_messages[0].get("role") != "system":
                        prompt_messages = [{"role": "system", "content": SYSTEM[row["language"]]}] + prompt_messages
                    split = "development" if int(digest("a3-provisional-split-v1|" + row["source_group_id"]), 16) % 100 < 15 else "train"
                    if row["source_group_id"] in groups["development" if split == "train" else "train"]:
                        raise ValueError(f"group appears in both splits: {row['source_group_id']}")
                    groups[split].add(row["source_group_id"])
                    item = {
                        "schema_version": "mhi-a3-provisional-sft-1.0",
                        "record_id": row["record_id"], "source_id": source,
                        "source_revision": row.get("source_revision"), "source_group_id": row["source_group_id"],
                        "split": split, "language": row["language"],
                        "locale": "zh-CN" if row["language"] == "zh" else "en-US",
                        "task_family": "multi_turn_emotional_support",
                        "risk_level": "STATIC_FLAG_FREE_NOT_SAFETY_APPROVED", "gold_status": "NOT_GOLD",
                        "review_status": "NOT_VERIFIED_CANDIDATE", "rights_tier": "SOURCE_SCOPE_UNRESOLVED_CANDIDATE",
                        "source_license": "MUST_RECONCILE_BEFORE_TRAINING",
                        "source_metadata": {"source_row_1based": row["source_row_1based"], "target_selection": row["target_selection"], "synthetic_origin": row["synthetic_origin"]},
                        "prompt": prompt_messages, "completion": [{"role": "assistant", "content": messages[-1]["content"]}],
                        "chat_template_kwargs": {"enable_thinking": False}, "token_mask_validation": "NOT_RUN",
                        "product_training_approved": False,
                    }
                    write(train if split == "train" else development, item)
                    counts[split] += 1; local[split] += 1
            per_source[source] = dict(local)
    manifest = {
        "schema_version": "mhi-a3-provisional-export-1.0",
        "status": "PROVISIONAL_RESEARCH_TRAINING_EXPORT_NOT_RELEASED",
        "training_allowed": False,
        "product_training_allowed": False,
        "source_candidate_state_required": "CANDIDATE_NOT_FROZEN",
        "split_rule": "verified source_group_id hash; 85% train / 15% development; no SMILE records because group is missing",
        "input_files_sha256": input_hashes,
        "counts": {"candidate_rows_seen": counts["candidate_rows_seen"], "train": counts["train"], "development": counts["development"], "quarantined": counts["quarantined"]},
        "language_counts": {},
        "source_counts": per_source,
        "group_counts": {k: len(v) for k, v in groups.items()},
        "open_gates": ["source license/revision scope", "professional safety and PII review", "semantic and benchmark contamination", "tokenizer/chat-template/collator token mask", "independent held-out evaluation"],
        "not_claimed": ["clinical safety", "product permission", "final dataset", "Qwen3.8-27B token budget", "independent test validity"],
    }
    for split in ("train", "development"):
        lang = Counter()
        with (output / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): lang[json.loads(line)["language"]] += 1
        manifest["language_counts"][split] = dict(lang)
    for path in sorted(output.glob("*.jsonl")):
        manifest.setdefault("output_files", {})[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    (output / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.candidate_root, args.output)
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
