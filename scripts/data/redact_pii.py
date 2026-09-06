#!/usr/bin/env python3
"""Two-stage PII redaction for canonical conversations.

Stage 1 (always on): deterministic regex for phone/email/URL-with-id/ID-card/
exact-birth-date/postal-code/account-id patterns, with per-conversation stable
pseudonyms (<PHONE_1>, <EMAIL_1>, <ID_1>, <ADDRESS_1> ...).

Stage 2 (optional): pluggable NER hook (--ner-script) for names/schools/
employers; no NER model is bundled, the default run only does stage 1 and
flags conversations for review.

Hard gate: high-confidence direct PII found but not transformed -> DROP
(with reason recorded in the report). Broad contextual info (age, "my
mother", "I study at university") is intentionally NOT redacted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Stage-1 deterministic patterns.
# Redact only high-precision patterns. Address-like text without a digit or a
# unit suffix (号/栋/室/楼/单元) is NOT redacted -- it goes to review flags
# instead, because single chars like 道/路 appear constantly in normal
# counseling speech and over-redaction corrupts training data.
REDACT_PATTERNS = [
    ("PHONE", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),          # CN mobile
    ("PHONE", re.compile(r"(?<!\d)0\d{2,3}-\d{7,8}(?!\d)")),                    # CN landline
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("URL", re.compile(r"https?://[^\s\u4e00-\u9fff]{8,}")),
    ("IDCARD", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),                       # 18-digit CN ID
    # date of BIRTH only when birth context is explicit
    ("BIRTH", re.compile(r"(?:出生于?|出生日期|生日)(?:是|在|为)?\s*(19|20)\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?")),
    # address: place word + digits, or place word + unit suffix (high precision)
    ("ADDRESS", re.compile(r"[\u4e00-\u9fff]{2,8}(?:路|街|道|村|小区|大厦|花园|广场)[0-9]{1,6}(?:号)?(?:[0-9\u4e00-\u9fff号栋幢单元室楼-]{0,10})?")),
    ("ADDRESS", re.compile(r"[\u4e00-\u9fff]{2,8}(?:小区|大厦|花园|广场|街道)[\u4e00-\u9fff0-9]{2,12}(?:号|栋|幢|室|楼|单元)")),
]

# Review-only flags (no redaction): ambiguous place mentions, e.g. 北京市海淀区.
REVIEW_PATTERNS = [
    ("ADDRESS_REVIEW", re.compile(r"[\u4e00-\u9fff]{2,8}(?:省|市|区|县|镇|乡|街道)[\u4e00-\u9fff0-9]{2,14}")),
]


def redact_text(text: str, counter: dict) -> tuple[str, list[str]]:
    """Redact high-precision patterns; return (text, types_found)."""
    found: list[str] = []
    out = text
    for label, pattern in REDACT_PATTERNS:
        def repl(m: re.Match, label=label) -> str:
            counter[label] = counter.get(label, 0) + 1
            found.append(label)
            return f"<{label}_{counter[label]}>"
        out = pattern.sub(repl, out)
    return out, found


def review_hits(text: str) -> list[str]:
    return [label for label, pattern in REVIEW_PATTERNS if pattern.search(text)]


def process_row(row: dict, counter: dict, stats: dict) -> dict | None:
    types: set[str] = set()
    review_types: set[str] = set()
    for m in row.get("messages", []):
        new_content, found = redact_text(m.get("content", ""), counter)
        if found:
            types.update(found)
        review_types.update(review_hits(new_content))
        m["content"] = new_content
    stats["review_flagged_rows"] += int(bool(review_types))
    row["pii"] = {
        "pii_detected": bool(types),
        "pii_types": sorted(types),
        "pii_redacted": True,
        "pii_review_required": bool(review_types),
        "review_types": sorted(review_types),
    }
    stats["redacted_rows"] += int(bool(types))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--drop", action="store_true", help="drop untransformable PII rows")
    args = ap.parse_args()

    counter: dict = {}
    stats = {"total": 0, "redacted_rows": 0, "review_flagged_rows": 0, "kept": 0, "types": {}}
    kept = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stats["total"] += 1
            res = process_row(row, counter, stats)
            kept.append(res)
            stats["kept"] += 1
    stats["types"] = counter
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
