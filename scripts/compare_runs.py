#!/usr/bin/env python3
"""
compare_runs.py — 两次 run-id 之间的复跑稳定性对比 + suite overview 合成

用法: python3 scripts/compare_runs.py v1 v2
输出: results/all-<runB>.summary.json(与 run.mjs all 的 overview 同构:各任务 summary 数组)
      + stdout 对比表(acc/wF1 差异,标注 >1pp 的任务)
说明: temperature 0 下 DeepSeek 仍可能有微小非确定性;抽样任务种子 42 固定,题目完全相同。
"""
import json, os, glob, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

def load_summaries(run_id):
    out = {}
    for p in glob.glob(os.path.join(RES, f"*-{run_id}.summary.json")):
        base = os.path.basename(p).replace(f"-{run_id}.summary.json", "")
        # strip model suffix: task-model → keep full for uniqueness, task = summary's own field
        try:
            s = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(s, dict) and "task" in s:
            out[s["task"]] = s
    return out

def main():
    a, b = (sys.argv + ["v1", "v2"])[1:3]
    A, B = load_summaries(a), load_summaries(b)
    common = sorted(set(A) & set(B))
    print(f"{'task':18s} {'n':>5s} {a+'-acc':>8s} {b+'-acc':>8s} {'Δpp':>6s}   wF1差")
    big = []
    for t in common:
        sa, sb = A[t], B[t]
        da = (sb["accuracy"] - sa["accuracy"]) * 100
        dw = ""
        if sa.get("weightedF1") is not None and sb.get("weightedF1") is not None:
            dw = f"{(sb['weightedF1']-sa['weightedF1']):+.3f}"
        flag = " ⚠" if abs(da) > 1.0 else ""
        if abs(da) > 1.0: big.append(t)
        print(f"{t:18s} {sb['n']:5d} {sa['accuracy']*100:7.2f}% {sb['accuracy']*100:7.2f}% {da:+5.2f}{flag}   {dw}")
    missing_b = sorted(set(A) - set(B))
    if missing_b: print(f"\n{b} 缺少: {missing_b}")
    overview = [B[t] for t in sorted(B)]
    outp = os.path.join(RES, f"all-{b}.summary.json")
    json.dump(overview, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n共同任务 {len(common)};|Δacc|>1pp: {big or '无'}")
    print(f"overview({len(overview)} tasks) -> {outp}")

if __name__ == "__main__":
    main()
