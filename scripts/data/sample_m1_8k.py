#!/usr/bin/env python3
"""Group-level stratified sampling for the frozen M1 corpus.

Quotas (v2 frozen, 2026-08-15):
  train 8,000 = CPsyCounD 2,800 + PsyDial-D1 2,000 + CPsDD 3,200
  validation 500  = 150 + 150 + 200
  internal test 500 = 150 + 150 + 200
All Chinese. One original dialogue = one group; a group never crosses splits.
CPsDD train is stratified by topic (metadata.topic) before quota sampling.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

QUOTAS = {
    "CPsyCounD": {"train": 2800, "val": 150, "test": 150},
    "PsyDial-D1": {"train": 2000, "val": 150, "test": 150},
    "CPsDD": {"train": 3200, "val": 200, "test": 200},
}
SEED = 20260815


def stratified_sample(groups: list[str], n: int, rng: random.Random, strata: dict[str, list[str]] | None):
    if strata is None:
        return rng.sample(groups, min(n, len(groups)))
    picked: list[str] = []
    strata_items = list(strata.items())
    rng.shuffle(strata_items)
    for _ in range(n):
        # pick a random stratum proportional to its remaining size, then a member
        total = sum(len(v) for _, v in strata_items if v)
        if total == 0:
            break
        r = rng.uniform(0, total)
        acc = 0
        for _, members in strata_items:
            if not members:
                continue
            acc += len(members)
            if r <= acc:
                picked.append(members.pop(rng.randrange(len(members))))
                break
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="post-overlap-scan canonical.jsonl")
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/m1-v1/data"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    report: dict = {"available": {}, "sampled": {}, "language": {}}
    for src, quota in QUOTAS.items():
        pool = by_source.get(src, [])
        report["available"][src] = len(pool)
        # stratum = topic for CPsDD, otherwise None (uniform)
        strata: dict[str, list[str]] | None = None
        if src == "CPsDD":
            strata = defaultdict(list)
            for r in pool:
                topic = (r.get("metadata") or {}).get("topic") or "unknown"
                strata[str(topic)].append(r["group_id"])
        groups = {r["group_id"]: r for r in pool}
        gids = list(groups)
        train_g = stratified_sample(gids[:], quota["train"], rng, {k: v[:] for k, v in strata.items()} if strata else None)
        rest = [g for g in gids if g not in set(train_g)]
        val_g = stratified_sample(rest[:], quota["val"], rng, None)
        test_g = [g for g in rest if g not in set(val_g)][: quota["test"]]
        for g in train_g:
            splits["train"].append(groups[g])
        for g in val_g:
            splits["val"].append(groups[g])
        for g in test_g:
            splits["test"].append(groups[g])
        report["sampled"][src] = {"train": len(train_g), "val": len(val_g), "test": len(test_g)}

    # assertions
    for s in ("train", "val", "test"):
        langs = Counter(r.get("language") for r in splits[s])
        report["language"][s] = dict(langs)
    train_ids = {r["group_id"] for r in splits["train"]}
    val_ids = {r["group_id"] for r in splits["val"]}
    test_ids = {r["group_id"] for r in splits["test"]}
    assert len(splits["train"]) == 8000, f"train {len(splits['train'])} != 8000"
    assert len(splits["val"]) == 500 and len(splits["test"]) == 500
    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)
    assert all(l == "zh" for l in report["language"]["train"]), "non-zh in train"
    for src, q in QUOTAS.items():
        n = sum(1 for r in splits["train"] if r["source"] == src)
        assert n == q["train"], f"{src} train {n} != {q['train']}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.output_dir / "sampling_report.json").write_text(
        json.dumps({"seed": args.seed, **report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"seed": args.seed, **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
