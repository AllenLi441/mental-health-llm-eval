#!/usr/bin/env python3
"""Rebuild four local A3 sources into quarantined, model-neutral candidates.

No network, training, final split, clinical approval, or redistribution. A single
raw record produces at most one target; historical turns remain context with
zero loss. PsyDT selects one reproducible turn per dialogue to avoid training
only closing replies. Message weights are a contract, not token masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


SOURCES = {
    "PsyDTCorpus": ("PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json", "zh"),
    "MeChat_smile": ("MeChat_smile/data.json", "zh"),
    "CounselChat": ("CounselChat/20220401_counsel_chat.csv", "en"),
    "EN_cand_CBT": ("EN_cand_CBT/CBT_dialogues.jsonl", "en"),
}
HELDOUT = "PsyDTCorpus/PsyDTCorpus_test_single_turn_split.json"
SCHEMA = "a3-candidate-1.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("EMPTY_OR_NONSTRING_TEXT")
    return value.replace("\r\n", "\n").strip()


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_rows(path: Path):
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("EXPECTED_JSON_ARRAY")
        yield from rows


def check_messages(messages: object) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("MISSING_MESSAGES")
    clean = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("UNKNOWN_ROLE_OR_MESSAGE")
        clean.append({"role": message["role"], "content": text(message.get("content"))})
    start = int(clean[0]["role"] == "system")
    body = clean[start:]
    if len(body) < 2 or len(body) % 2:
        raise ValueError("INCOMPLETE_USER_ASSISTANT_PAIRS")
    for index, message in enumerate(body):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("NONALTERNATING_ROLES")
    return clean


def convert(source: str, row: dict, index: int, input_hash: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError("EXPECTED_OBJECT")
    group = None
    group_basis = "MISSING_UPSTREAM_LINEAGE"
    if source == "PsyDTCorpus":
        messages = check_messages(row.get("messages"))
        if row.get("id") is not None:
            group = f"PsyDTCorpus:{row['id']}"
            group_basis = "UPSTREAM_DIALOGUE_ID"
    elif source == "MeChat_smile":
        messages = []
        instruction = row.get("instruction")
        if instruction is not None and instruction != "":
            messages.append({"role": "system", "content": text(instruction)})
        history = row.get("history")
        if not isinstance(history, list):
            raise ValueError("INVALID_HISTORY")
        for pair in history:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("INVALID_HISTORY_PAIR")
            messages.extend([{"role": "user", "content": text(pair[0])},
                             {"role": "assistant", "content": text(pair[1])}])
        messages.extend([{"role": "user", "content": text(row.get("input"))},
                         {"role": "assistant", "content": text(row.get("output"))}])
        messages = check_messages(messages)
    elif source == "CounselChat":
        messages = check_messages([{"role": "user", "content": row.get("questionText")},
                                   {"role": "assistant", "content": row.get("answerText")}])
        if row.get("questionID") not in (None, ""):
            group = f"CounselChat:{row['questionID']}"
            group_basis = "UPSTREAM_QUESTION_ID"
    elif source == "EN_cand_CBT":
        messages = check_messages([{"role": "user", "content": row.get("patient")},
                                   {"role": "assistant", "content": row.get("therapist")}])
        if row.get("profile_id"):
            group = f"EN_cand_CBT:{row['profile_id']}"
            group_basis = "UPSTREAM_SYNTHETIC_PROFILE_ID"
    else:
        raise ValueError("UNKNOWN_SOURCE")
    original_turns = sum(m["role"] == "assistant" for m in messages)
    selection = "source_provided_final_target"
    if source == "PsyDTCorpus":
        positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        key = group or digest(messages)
        position = positions[int(digest(["a3-turn-selection-v1", key]), 16) % len(positions)]
        messages = messages[:position + 1]
        selection = "deterministic_one_turn_per_dialogue_v1"
    first_user = next(m["content"] for m in messages if m["role"] == "user")
    return {
        "schema_version": SCHEMA,
        "record_id": f"{source}:{input_hash[:16]}:{index}",
        "source": source,
        "source_file_sha256": input_hash,
        "source_revision": None,
        "source_row_1based": index,
        "source_group_id": group,
        "group_basis": group_basis,
        "first_user_sha256": digest(normalized(first_user)),
        "first_user_group_is_verified_lineage": False,
        "language": SOURCES[source][1],
        "messages": messages,
        "loss_message_weights": [0] * (len(messages) - 1) + [1],
        "target_message_index": len(messages) - 1,
        "target_selection": selection,
        "original_assistant_turns": original_turns,
        "content_sha256": digest([{**m, "content": normalized(m["content"])} for m in messages]),
        "target_sha256": digest(normalized(messages[-1]["content"])),
        "content_state": "CANDIDATE_NOT_FROZEN",
        "split": None,
        "product_training_approved": False,
        "human_review_status": "NOT_VERIFIED",
        "token_mask_validation": "NOT_RUN",
        "synthetic_origin": "REPORTED_SYNTHETIC" if source != "CounselChat" else "PUBLIC_QA",
    }


PATTERNS = {
    "email_candidate": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "phone_candidate": re.compile(r"(?<!\d)(?:1[3-9]\d{9}|\d{3}[- .]\d{3}[- .]\d{4})(?!\d)"),
    "url_candidate": re.compile(r"https?://\S+", re.I),
    "acute_risk_cue": re.compile(r"自杀|自残|轻生|不想活|kill myself|suicid|self.harm|want to die", re.I),
    "somatic_exclusion_candidate": re.compile(r"不是生理问题|并非生理问题|排除.{0,6}生理|definitely not physical", re.I),
    "guaranteed_cure_candidate": re.compile(r"保证.{0,6}(治愈|康复)|一定.{0,5}(治愈|康复)|guarantee.{0,12}cur", re.I),
}


def flags(messages: list[dict]) -> list[str]:
    found = set()
    for message in messages:
        for name, pattern in PATTERNS.items():
            if pattern.search(message["content"]):
                found.add(f"{message['role']}:{name}")
    return sorted(found)


def write_line(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def build(raw_root: Path, output: Path) -> dict:
    paths = [raw_root / p for p, _ in SOURCES.values()] + [raw_root / HELDOUT]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"MISSING_OR_SYMLINK_INPUT: {path.name}")
    if output.exists():
        raise ValueError("OUTPUT_MUST_BE_NEW_DIRECTORY")
    before = {p.relative_to(raw_root).as_posix(): sha256(p) for p in paths}
    references = list(load_rows(raw_root / HELDOUT))
    heldout_groups = {f"PsyDTCorpus:{row['id']}" for row in references}
    heldout_first = {digest(normalized(next(m['content'] for m in check_messages(row['messages'])
                                          if m['role'] == 'user'))) for row in references}
    output.mkdir(parents=True)
    report = {
        "schema_version": SCHEMA,
        "content_state": "CANDIDATE_NOT_FROZEN",
        "policy": "one_target_per_raw_record; PsyDT deterministic turn; others source final target",
        "inputs_sha256": before,
        "script_sha256": sha256(Path(__file__)),
        "heldout": {"source": HELDOUT, "expanded_rows": len(references),
                    "dialogue_groups": len(heldout_groups), "excluded_from_candidates": True},
        "sources": {},
        "known_limits": [
            "No final split, sampling, license approval, professional safety approval, or tokenization.",
            "SMILE upstream report/question IDs absent; first-user hash is only a heuristic, not verified lineage.",
            "Exact message and first-user matches do not detect semantic paraphrases or synthetic descendants.",
            "Keyword flags are review candidates; absence of a flag is not evidence of clinical safety.",
            "Loss weights are message-level intent; trainer must prove actual token-level assistant-only masking.",
            "Only supplied PsyDT heldout checked; full project benchmark overlap audit remains required.",
            "Raw source record counts are not independent conversations or a final 70:30 split.",
        ],
    }
    seen_content = {}
    seen_targets = Counter()
    with (output / "rejections.jsonl").open("w", encoding="utf-8") as rejects:
        for source, (relative, language) in SOURCES.items():
            counts = Counter()
            groups = set()
            source_flags = Counter()
            with (output / f"{source}.candidates.jsonl").open("w", encoding="utf-8") as dest:
                for index, row in enumerate(load_rows(raw_root / relative), 1):
                    counts["raw_records"] += 1
                    try:
                        item = convert(source, row, index, before[relative])
                    except (ValueError, KeyError, TypeError, StopIteration) as error:
                        counts["rejected_schema"] += 1
                        # Exception class/code only; never copy malformed text into the public report.
                        write_line(rejects, {"source": source, "row_1based": index,
                                            "reason": str(error) if isinstance(error, ValueError) else type(error).__name__})
                        continue
                    if item["source_group_id"] in heldout_groups or item["first_user_sha256"] in heldout_first:
                        counts["quarantined_heldout_match"] += 1
                        write_line(rejects, {"record_id": item["record_id"], "reason": "HELDOUT_GROUP_OR_FIRST_USER_MATCH"})
                        continue
                    if item["content_sha256"] in seen_content:
                        counts["quarantined_exact_duplicate"] += 1
                        write_line(rejects, {"record_id": item["record_id"], "reason": "EXACT_DUPLICATE",
                                            "duplicate_of": seen_content[item["content_sha256"]]})
                        continue
                    seen_content[item["content_sha256"]] = item["record_id"]
                    seen_targets[item["target_sha256"]] += 1
                    item["review_flags"] = flags(item["messages"])
                    source_flags.update(item["review_flags"])
                    if item["review_flags"]:
                        counts["rows_with_review_flags"] += 1
                    if item["source_group_id"]:
                        groups.add(item["source_group_id"])
                    else:
                        counts["rows_missing_upstream_group"] += 1
                    counts["normalized_candidates"] += 1
                    counts["assistant_targets"] += 1
                    counts["historical_assistant_messages_masked"] += sum(m["role"] == "assistant" for m in item["messages"][:-1])
                    write_line(dest, item)
            assert counts["raw_records"] == counts["normalized_candidates"] + counts["rejected_schema"] + counts["quarantined_heldout_match"] + counts["quarantined_exact_duplicate"]
            report["sources"][source] = {"language": language, "counts": dict(counts),
                                           "known_source_groups": len(groups), "review_flags": dict(source_flags)}
    report["repeated_target_hashes"] = sum(n > 1 for n in seen_targets.values())
    report["repeated_target_excess_rows"] = sum(n - 1 for n in seen_targets.values())
    report["raw_pool_records"] = sum(s["counts"]["raw_records"] for s in report["sources"].values())
    report["normalized_candidate_records"] = sum(s["counts"].get("normalized_candidates", 0) for s in report["sources"].values())
    zh = sum(s["counts"]["raw_records"] for s in report["sources"].values() if s["language"] == "zh")
    en = sum(s["counts"]["raw_records"] for s in report["sources"].values() if s["language"] == "en")
    units = min(zh // 7, en // 3)
    report["raw_70_30_capacity_estimate"] = {"total": units * 10, "zh": units * 7, "en": units * 3,
                                              "status": "ARITHMETIC_ONLY_NOT_MATERIALIZED"}
    report["inputs_unchanged"] = all(sha256(p) == before[p.relative_to(raw_root).as_posix()] for p in paths)
    if not report["inputs_unchanged"]:
        raise ValueError("INPUT_CHANGED_DURING_BUILD")
    report["outputs"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                         for p in sorted(output.glob("*.jsonl"))}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.raw_root, args.output)
    print(json.dumps({"state": report["content_state"], "raw_records": report["raw_pool_records"],
                      "normalized_candidates": report["normalized_candidate_records"],
                      "manifest": str(args.output / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
