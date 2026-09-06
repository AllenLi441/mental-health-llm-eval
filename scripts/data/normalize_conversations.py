#!/usr/bin/env python3
"""Canonicalize M1 training sources into one conversation schema.

Output: canonical.jsonl lines like
{
  "sample_id": "cpsycound:000123",
  "source": "CPsyCounD",
  "source_revision": "...",
  "language": "zh",
  "group_id": "cpsycound-000123",
  "license_id": "CC-BY-4.0",
  "messages": [{"role": "user"|"assistant"|"system", "content": "...", "meta": {...}}],
  "metadata": {"topic": null, "strategy": null, "original_split": "train"}
}

Role mapping is deterministic: counselor->assistant, client/usr/user->user.
Original role labels are kept inside message "meta" for audit only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROLE_MAP = {
    "user": "user", "usr": "user", "client": "user", "seeker": "user",
    "assistant": "assistant", "counselor": "assistant", "sys": "system",
    "system": "system", "bot": "assistant",
}


def clean_text(s: str) -> str:
    s = s.replace("\u3000", " ").strip()
    return s


def canon_messages(msgs, keep_meta: bool = True) -> list[dict]:
    out = []
    for m in msgs:
        role = str(m.get("role", "")).strip().lower()
        content = clean_text(str(m.get("content", "")))
        if not content:
            continue
        canon_role = ROLE_MAP.get(role)
        if canon_role is None:
            print(f"WARN unknown role {role!r}, treating as user", file=sys.stderr)
            canon_role = "user"
        item = {"role": canon_role, "content": content}
        if keep_meta:
            meta = {}
            for k in ("strategy", "emotion", "intent"):
                if m.get(k) is not None:
                    meta[k] = m[k]
            if meta:
                item["meta"] = meta
        out.append(item)
    return out


def load_cpsycound(path: Path, revision: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, row in enumerate(data):
        msgs = []
        history = row.get("history") or []
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                msgs.append({"role": "user", "content": clean_text(str(turn[0]))})
                msgs.append({"role": "assistant", "content": clean_text(str(turn[1]))})
        instr = clean_text(str(row.get("instruction", "")))
        outp = clean_text(str(row.get("output", "")))
        if instr:
            msgs.append({"role": "user", "content": instr})
        if outp:
            msgs.append({"role": "assistant", "content": outp})
        out.append({
            "sample_id": f"cpsycound:{i:06d}",
            "source": "CPsyCounD",
            "source_revision": revision,
            "language": "zh",
            "group_id": f"cpsycound-{i:06d}",
            "license_id": "CC-BY-4.0",
            "messages": msgs,
            "metadata": {"topic": None, "strategy": None, "original_split": "train"},
        })
    return out


def load_psydial_d1(path: Path, revision: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, row in enumerate(data):
        msgs = canon_messages(row.get("messages") or [])
        out.append({
            "sample_id": f"psydiald1:{i:06d}",
            "source": "PsyDial-D1",
            "source_revision": revision,
            "language": "zh",
            "group_id": f"psydiald1-{i:06d}",
            "license_id": "apache-2.0",
            "messages": msgs,
            "metadata": {"topic": None, "strategy": None, "original_split": "train"},
        })
    return out


def load_cpsdd(path: Path, revision: str, lang_filter: str = "zh") -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if lang_filter and row.get("lang") != lang_filter:
                continue
            msgs = canon_messages(row.get("messages") or [])
            meta = {"topic": row.get("topic"), "strategy": None, "original_split": "train"}
            if row.get("client_profile"):
                meta["client_profile"] = row["client_profile"]
            out.append({
                "sample_id": f"cpsdd:{row.get('case_id')}",
                "source": "CPsDD",
                "source_revision": revision,
                "language": row.get("lang", "zh"),
                "group_id": f"cpsdd-{row.get('case_id')}",
                "license_id": "conditional-research",
                "messages": msgs,
                "metadata": meta,
            })
    return out


SOURCES = {
    "cpsycound": load_cpsycound,
    "psydial-d1": load_psydial_d1,
    "cpsdd-train": load_cpsdd,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpsycound", type=Path, default=Path("tmp/datasets/CAS-SIAT-XinHai__CPsyCoun/CPsyCounD.json"))
    ap.add_argument("--cpsycound-revision", default="8fe0fa0f77630b921932bdaa6bae7481bc611cf2")
    ap.add_argument("--psydial-d1", type=Path, default=Path("tmp/datasets/qiuhuachuan__PsyDial-D1/PsyDial-D1.json"))
    ap.add_argument("--psydial-d1-revision", default="0462554c9cdb12739d791566b75dd99ab2bb601f")
    ap.add_argument("--cpsdd-train", type=Path, default=Path("tmp/datasets/XuShihao6715__counseling-cpsdd/train.jsonl"))
    ap.add_argument("--cpsdd-revision", default="1ec603d5e7bfaac4d47aa219689c44a71bb71fdc")
    ap.add_argument("--output", type=Path, default=Path("artifacts/m1-v1/data/canonical.jsonl"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rows += load_cpsycound(args.cpsycound, args.cpsycound_revision)
    rows += load_psydial_d1(args.psydial_d1, args.psydial_d1_revision)
    rows += load_cpsdd(args.cpsdd_train, args.cpsdd_revision)

    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    counts = Counter(r["source"] for r in rows)
    print(json.dumps({"total": len(rows), "by_source": dict(counts), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
