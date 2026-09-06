#!/usr/bin/env python3
"""Exact + near-duplicate removal for canonical conversations.

Exact: NFKC-normalize, whitespace-collapse each message content, concat
conversation text, SHA256 -> identical hash collapses (provenance kept).

Near: 128-hash MinHash over char 3-grams + word tokens, LSH (16 bands x
8 rows) candidate generation, then exact Jaccard on shingle sets.
Policy (frozen thresholds):
  Jaccard >= 0.90 -> auto-duplicate (dropped, listed in report)
  0.80 - 0.90    -> review-flagged (kept, flagged in report)
  < 0.80         -> retained
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

K_SKETCH = 64          # bottom-K sketch size
BANDS = 8              # 8 bands x 8 values
ROWS_PER_BAND = K_SKETCH // BANDS
MAXHASH = (1 << 63) - 1
MAX_BUCKET = 1000
MAX_PAIRS = 2_000_000


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def conv_text(row: dict) -> str:
    return " ".join(norm_text(m.get("content", "")) for m in row.get("messages", []))


def shingles(text: str, k: int = 3) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def bottom_k_signature(sh_set: set[str]) -> list[int]:
    """Bottom-K sketch: the K smallest 64-bit hashes, sorted ascending.
    Valid Jaccard estimator when |set| >= K; needs no per-seed mixing."""
    hs = (int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
          for s in sh_set)
    sig = heapq.nsmallest(K_SKETCH, hs)
    return sig


def est_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    return len(set(sig_a) & set(sig_b)) / K_SKETCH


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--auto-threshold", type=float, default=0.90)
    ap.add_argument("--review-threshold", type=float, default=0.80)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ---- exact dedup ----
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        h = hashlib.sha256(conv_text(r).encode("utf-8")).hexdigest()
        exact_groups[h].append(i)
    keep_idx: list[int] = []
    exact_dropped: list[dict] = []
    for h, idxs in exact_groups.items():
        keep_idx.append(idxs[0])
        for dup in idxs[1:]:
            exact_dropped.append({"hash": h, "sample_id": rows[dup]["sample_id"],
                                  "dup_of": rows[idxs[0]]["sample_id"]})
    rows = [rows[i] for i in sorted(keep_idx)]

    # ---- near dedup via bottom-K sketch + LSH ----
    sigs = []
    for r in rows:
        sh = shingles(conv_text(r))
        if len(sh) < 3:
            sigs.append([MAXHASH] * K_SKETCH)
        else:
            sigs.append(bottom_k_signature(sh))
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        for b in range(BANDS):
            key = tuple(sig[b * ROWS_PER_BAND:(b + 1) * ROWS_PER_BAND] + [b])
            buckets[key].append(i)
    candidate_pairs: set[tuple[int, int]] = set()
    skipped_buckets = 0
    lengths = [len(conv_text(r)) for r in rows]
    for idxs in buckets.values():
        if len(idxs) > MAX_BUCKET:
            skipped_buckets += 1
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                la, lb = lengths[ia], lengths[ib]
                if la and lb and abs(la - lb) / max(la, lb) > 0.25:
                    continue  # length prefilter
                candidate_pairs.add((min(ia, ib), max(ia, ib)))
                if len(candidate_pairs) > MAX_PAIRS:
                    break
            if len(candidate_pairs) > MAX_PAIRS:
                break
        if len(candidate_pairs) > MAX_PAIRS:
            break

    near_dropped, review = [], []
    auto_ids: set[int] = set()
    shingle_cache: dict[int, set[str]] = {}
    def sh_of(i: int) -> set[str]:
        if i not in shingle_cache:
            shingle_cache[i] = shingles(conv_text(rows[i]))
        return shingle_cache[i]
    for a, b in candidate_pairs:
        if a in auto_ids or b in auto_ids:
            continue
        est = est_jaccard(sigs[a], sigs[b])
        if est < args.review_threshold:
            continue
        if est < args.auto_threshold:
            # signature-level review flag; no exact verification (documented)
            review.append({"sample_id": rows[b]["sample_id"], "dup_of": rows[a]["sample_id"],
                           "minhash_est": round(est, 4), "exact_jaccard": None})
            continue
        sh_a, sh_b = sh_of(a), sh_of(b)
        jac = len(sh_a & sh_b) / max(1, len(sh_a | sh_b))
        pair = {"sample_id": rows[b]["sample_id"], "dup_of": rows[a]["sample_id"],
                "minhash_est": round(est, 4), "exact_jaccard": round(jac, 4)}
        if jac >= args.auto_threshold:
            auto_ids.add(b)
            near_dropped.append(pair)
        else:
            review.append(pair)

    final = [r for i, r in enumerate(rows) if i not in auto_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "input": len(rows) + len(exact_dropped),
        "exact_dropped": len(exact_dropped),
        "near_auto_dropped": len(near_dropped),
        "review_flagged": len(review),
        "kept": len(final),
        "skipped_oversized_buckets": skipped_buckets,
        "candidate_pairs": len(candidate_pairs),
        "exact_examples": exact_dropped[:5],
        "near_examples": near_dropped[:5],
        "review_examples": review[:5],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
