#!/usr/bin/env python3
"""
audit_results.py — 从逐条 JSONL 独立重算所有任务指标,与 summary 对账(固化口径)

回应复查审计建议 #1/#5:把"独立脚本重算 19/19 一致"落成可追踪脚本,并固化 suite overview。

对每个 results/*-v1.jsonl(以及 EmoBench 的独立结果):重算 n / correct / accuracy /
weighted-F1(如适用),与对应 *.summary.json 比对,输出 PASS/FAIL 表 + audit_recompute.json。

用法: python3 scripts/audit_results.py   (在 eval-suite/ 目录下)
"""
import json, os, glob, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
EMO = os.path.join(HERE, "..", "..", "EmoBench", "eval", "results")

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def f1(rows, lab):
    tp = sum(1 for r in rows if r["predicted"] == lab and r["gold"] == lab)
    fp = sum(1 for r in rows if r["predicted"] == lab and r["gold"] != lab)
    fn = sum(1 for r in rows if r["predicted"] != lab and r["gold"] == lab)
    return (2 * tp) / (2 * tp + fp + fn) if tp else 0.0

def main():
    report, fails = [], []
    for p in sorted(glob.glob(os.path.join(RES, "*-v1.jsonl"))):
        name = os.path.basename(p).replace("-v1.jsonl", "")
        rows = load(p)
        if not rows or "gold" not in rows[0]:
            continue
        n = len(rows); acc = sum(r["ok"] for r in rows) / n
        labels = sorted({r["gold"] for r in rows})
        wf1 = sum((sum(1 for r in rows if r["gold"] == lab) / n) * f1(rows, lab) for lab in labels)
        sp = p.replace(".jsonl", ".summary.json")
        s = json.load(open(sp)) if os.path.exists(sp) else {}
        ok = abs(acc - s.get("accuracy", -1)) < 1e-9 and \
             (s.get("weightedF1") is None or abs(wf1 - s["weightedF1"]) < 1e-6)
        report.append({"task": name, "n": n, "acc_recomputed": round(acc, 5),
                       "acc_summary": s.get("accuracy"), "wf1_recomputed": round(wf1, 5),
                       "wf1_summary": s.get("weightedF1"), "match": ok})
        if not ok: fails.append(name)
        print(f"[{'✓' if ok else '✗'}] {name:34s} n={n:5d} acc={acc*100:6.2f}%")
    # EmoBench (independent harness, by_language summaries)
    for p in sorted(glob.glob(os.path.join(EMO, "*.summary.json"))):
        s = json.load(open(p)); name = os.path.basename(p).replace(".summary.json", "")
        jl = p.replace(".summary.json", ".jsonl")
        if not os.path.exists(jl): continue
        rows = load(jl)
        by = collections.defaultdict(list)
        for r in rows: by[r["lang"]].append(r)
        ok = all(abs(sum(x["correct"] for x in v)/len(v) - s["by_language"][k]["accuracy"]) < 1e-9
                 for k, v in by.items())
        report.append({"task": "emobench/" + name, "n": len(rows), "match": ok,
                       "by_language": {k: round(sum(x["correct"] for x in v)/len(v), 4) for k, v in by.items()}})
        if not ok: fails.append(name)
        print(f"[{'✓' if ok else '✗'}] emobench/{name:24s} n={len(rows):5d}")
    outp = os.path.join(RES, "audit_recompute.json")
    json.dump({"all_match": not fails, "fails": fails, "tasks": report},
              open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{'全部一致 ✓' if not fails else '不一致: ' + str(fails)}  ({len(report)} 项)")
    print(f"saved -> {outp}")

if __name__ == "__main__":
    main()
