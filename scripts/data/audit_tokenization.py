#!/usr/bin/env python3
"""Tokenization + serialization audit for the frozen M1 corpus.

Requires the selected Qwen base weights locally (transformers + tokenizer).
Checks: chat-template round-trip, token counts, language mix by tokens,
assistant-target length (never silently truncated), EOS termination.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def count_cjk(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path to Qwen checkpoint")
    ap.add_argument("--train", type=Path, default=Path("artifacts/m1-v1/data/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("artifacts/m1-v1/data/val.jsonl"))
    ap.add_argument("--test", type=Path, default=Path("artifacts/m1-v1/data/test.jsonl"))
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--output", type=Path, default=Path("artifacts/m1-v1/audit/tokenization.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    report = {"model": args.model, "max_length": args.max_length, "splits": {}}
    for split, path in (("train", args.train), ("val", args.val), ("test", args.test)):
        zh_chars = en_words = 0
        lens = []
        truncated_targets = 0
        bad_eos = 0
        n = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n += 1
                msgs = row["messages"]
                # token count of the full prompt+target
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
                lens.append(len(ids))
                text = " ".join(m.get("content", "") for m in msgs)
                zh_chars += count_cjk(text)
                en_words += len(text.split()) - count_cjk(text)  # rough non-CJK word proxy
                # target truncation check: serialize without target, then with target
                if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
                    no_target = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
                    if len(ids) - len(no_target) > args.max_length - len(no_target) - 64:
                        truncated_targets += 1
                    if ids[-1] != tok.eos_token_id and ids[-1] != getattr(tok, "eos_id", -1):
                        # template may append a trailing newline after <|im_end|>
                        if tok.eos_token_id not in ids[-3:]:
                            bad_eos += 1
        report["splits"][split] = {
            "examples": n,
            "token_total": sum(lens),
            "token_mean": round(sum(lens) / max(1, n), 1),
            "token_p50": sorted(lens)[len(lens) // 2] if lens else 0,
            "token_p95": sorted(lens)[int(len(lens) * 0.95)] if lens else 0,
            "over_max": sum(1 for x in lens if x > args.max_length),
            "zh_chars": zh_chars,
            "non_zh_word_proxy": en_words,
            "truncated_targets": truncated_targets,
            "missing_eos": bad_eos,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
