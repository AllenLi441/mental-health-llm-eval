#!/usr/bin/env python3
"""Benchmark contamination scan for M1 candidates.

Benchmark text sources (all test-only, never trainable):
  --cpcd-eval-root   tmp/official_benchmarks/psy_chronicle/eval_task_info
                     (srg/memory_recall/TCR + full_session, all strings)
  --esconv-test      tmp/official_benchmarks/esconv/codes/dataset/testWithStrategy_short.tsv
  --psydial-d101     tmp/datasets/qiuhuachuan__PsyDial-D101/PsyDial-D101.json

Checks per candidate:
  1. exact normalized full-conversation hash vs any benchmark item hash
  2. first-user-turn normalized hash
  3. MinHash-LSH candidates -> exact Jaccard >= 0.80 -> flag
  4. longest-common-substring length >= 40 chars -> alarm (on hash/Jaccard hits)

Output: filtered candidates + report. Any hit = the candidate must NOT enter M1.
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

K_SKETCH = 64
BANDS = 8
ROWS_PER_BAND = K_SKETCH // BANDS
MAXHASH = (1 << 63) - 1


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def sha256(s: str) -> str:
    return hashlib.sha256(norm(s).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 3) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def minhash(sh: set[str], seeds=None) -> list[int]:
    """Bottom-K sketch (seeds unused; kept for signature compatibility)."""
    hs = (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh)
    return heapq.nsmallest(K_SKETCH, hs)


def est_jaccard(a: list[int], b: list[int]) -> float:
    return len(set(a) & set(b)) / K_SKETCH


def lcs_len(s1: str, s2: str, cap: int = 80) -> int:
    if len(s1) > 2000 or len(s2) > 2000:
        return -1  # skip huge pairs
    prev = [0] * (len(s2) + 1)
    best = 0
    for i in range(1, len(s1) + 1):
        cur = [0] * (len(s2) + 1)
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
        if best >= cap:
            return cap
    return best


def iter_cpcd_texts(root: Path):
    for p in sorted(root.rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        yield p.name, json.dumps(d, ensure_ascii=False, sort_keys=True)


def iter_esconv_texts(path: Path):
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            segs = re.split(r"\s+EOS\s+", line.strip())
            yield f"esconv-line-{lineno}", " ".join(segs)


def iter_d101_texts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data:
        msgs = row.get("messages") or []
        golden = row.get("golden") or {}
        text = " ".join(m.get("content", "") for m in msgs)
        text += " " + str(golden.get("content", ""))
        yield f"d101-{row.get('idx')}", text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="deduped canonical.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--cpcd-eval-root", type=Path,
                    default=Path("tmp/official_benchmarks/psy_chronicle/eval_task_info"))
    ap.add_argument("--esconv-test", type=Path,
                    default=Path("tmp/official_benchmarks/esconv/codes/dataset/testWithStrategy_short.tsv"))
    ap.add_argument("--psydial-d101", type=Path,
                    default=Path("tmp/datasets/qiuhuachuan__PsyDial-D101/PsyDial-D101.json"))
    args = ap.parse_args()

    bench_items: list[tuple[str, str]] = []
    if args.cpcd_eval_root.exists():
        bench_items += list(iter_cpcd_texts(args.cpcd_eval_root))
    if args.esconv_test.exists():
        bench_items += list(iter_esconv_texts(args.esconv_test))
    if args.psydial_d101.exists():
        bench_items += list(iter_d101_texts(args.psydial_d101))
    print(f"benchmark text items: {len(bench_items)}", flush=True)

    bench_hashes = {sha256(t) for _, t in bench_items}
    first_turn_hashes = set()
    for _, t in bench_items:
        m = re.search(r'"content"\s*:\s*"([^"]{2,})"', t)
        if m:
            first_turn_hashes.add(sha256(m.group(1)))

    # Bottom-K sketch index over benchmark items (cap shingle work: 2000 chars)
    bench_sigs = []
    for _, t in bench_items:
        sh = shingles(t[:2000])
        bench_sigs.append(minhash(sh) if len(sh) >= 3 else [MAXHASH] * K_SKETCH)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, sig in enumerate(bench_sigs):
        for b in range(BANDS):
            buckets[tuple(sig[b * ROWS_PER_BAND:(b + 1) * ROWS_PER_BAND] + [b])].append(i)

    hits = []
    clean = []
    candidates = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    for row in candidates:
        msgs = row.get("messages", [])
        full = " ".join(norm(m.get("content", "")) for m in msgs)
        fh = sha256(full)
        first = norm(msgs[0]["content"]) if msgs else ""
        first_h = sha256(first) if first else None
        reason = None
        if fh in bench_hashes:
            reason = "exact-full-hash"
        elif first_h and first_h in first_turn_hashes:
            reason = "exact-first-turn-hash"
        else:
            sh = shingles(full[:2000])
            sig = minhash(sh) if len(sh) >= 3 else [MAXHASH] * K_SKETCH
            # band-bucket lookup only (no full sweep over all bench items)
            candidate_bench: set[int] = set()
            for b in range(BANDS):
                key = tuple(sig[b * ROWS_PER_BAND:(b + 1) * ROWS_PER_BAND] + [b])
                candidate_bench.update(buckets.get(key, ()))
            best_jac = 0.0
            best_bi = None
            for bi in candidate_bench:
                j = est_jaccard(sig, bench_sigs[bi])
                if j > best_jac:
                    best_jac, best_bi = j, bi
            if best_jac >= 0.55 and best_bi is not None:
                bt = bench_items[best_bi][1]
                ba, bb = shingles(full[:2000]), shingles(bt[:2000])
                jac = len(ba & bb) / max(1, len(ba | bb))
                if jac >= 0.80:
                    lcs = lcs_len(full[:2000], bt[:2000])
                    if lcs >= 40:
                        reason = f"near-dup-jaccard-{jac:.2f}-lcs-{lcs}"
                    elif lcs >= 0:
                        reason = f"near-dup-jaccard-{jac:.2f}-lcs-below-threshold"
        if reason:
            hits.append({"sample_id": row["sample_id"], "source": row["source"], "reason": reason})
        else:
            clean.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"benchmark_items": len(bench_items), "candidates": len(candidates),
              "hits": len(hits), "kept": len(clean), "hit_examples": hits[:20]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
