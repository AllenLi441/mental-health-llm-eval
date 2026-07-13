#!/usr/bin/env python3
"""
paired_model_audit.py — Kimi vs DeepSeek 同题配对分析(固化口径,单一事实来源)

回应复查审计高优先级问题 #1:把碰撞剔除、错误剔除、配对规则和最终 n 固化成脚本,
并完整披露 HTTP 429 丢失量,避免"后处理口径不透明"。

口径(STRICT,报告采用):
  1. 两侧均剔除 error 非空的行(Kimi 侧 429 为时间性限速丢失,数量在输出中完整披露);
  2. 配对键 = (id, gold) 双键;
  3. 若该键在任一侧不唯一(数据集自带重复 id 所致),整组剔除(collision-dropped);
  4. 在配对子集上分别计算两模型 accuracy。
同时输出 KEEP-FIRST 变体(碰撞取首次出现,即外部复查所用口径)以展示结论对配对规则不敏感。

用法: python3 scripts/paired_model_audit.py   (在 eval-suite/ 目录下)
输出: results/paired-model-audit.json + stdout 表格
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

PAIRS = [
    ("cpsyexam",  "cpsyexam-deepseek-chat-v1.jsonl",  "cpsyexam-moonshot-v1-8k-kimi1.jsonl"),
    ("psysuicide","psysuicide-deepseek-chat-v1.jsonl","psysuicide-moonshot-v1-8k-kimi1.jsonl"),
]

def load(f):
    p = os.path.join(RES, f)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def audit(task, ds_file, kimi_file):
    ds_all, k_all = load(ds_file), load(kimi_file)
    k_err = [r for r in k_all if r.get("error")]
    err_types = collections.Counter((r["error"] or "")[:12] for r in k_err)
    ds = [r for r in ds_all if not r.get("error")]
    kv = [r for r in k_all if not r.get("error")]

    dmap, kmap = collections.defaultdict(list), collections.defaultdict(list)
    for r in ds: dmap[(r["id"], r["gold"])].append(r)
    for r in kv: kmap[(r["id"], r["gold"])].append(r)

    strict, dropped = [], 0
    for key, krows in kmap.items():
        drows = dmap.get(key, [])
        if len(krows) == 1 and len(drows) == 1:
            strict.append((krows[0], drows[0]))
        else:
            dropped += len(krows)
    keepfirst = [(krows[0], dmap[key][0]) for key, krows in kmap.items() if dmap.get(key)]

    def acc(pairs, idx): return round(sum(p[idx]["ok"] for p in pairs) / len(pairs) * 100, 2)
    out = {
        "task": task,
        "deepseek_file": ds_file, "kimi_file": kimi_file,
        "kimi_rows_total": len(k_all), "kimi_rows_error": len(k_err),
        "kimi_error_types": dict(err_types),
        "kimi_rows_valid": len(kv), "deepseek_rows_valid": len(ds),
        "strict": {"rule": "id+gold 双键, 两侧唯一, 碰撞整组剔除",
                   "n": len(strict), "collision_dropped": dropped,
                   "kimi_acc": acc(strict, 0), "deepseek_acc": acc(strict, 1)},
        "keep_first": {"rule": "id+gold 双键, 碰撞取首次出现(外部复查口径)",
                       "n": len(keepfirst),
                       "kimi_acc": acc(keepfirst, 0), "deepseek_acc": acc(keepfirst, 1)},
    }
    return out

def main():
    results = [audit(*p) for p in PAIRS]
    outp = os.path.join(RES, "paired-model-audit.json")
    json.dump(results, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'任务':12s} {'Kimi总行/429/有效':>20s} {'口径':10s} {'n':>5s} {'DeepSeek':>9s} {'Kimi':>7s}")
    for r in results:
        base = f"{r['kimi_rows_total']}/{r['kimi_rows_error']}/{r['kimi_rows_valid']}"
        for mode in ("strict", "keep_first"):
            m = r[mode]
            print(f"{r['task']:12s} {base:>20s} {mode:10s} {m['n']:5d} {m['deepseek_acc']:8.2f}% {m['kimi_acc']:6.2f}%")
    print(f"\nsaved -> {outp}")

if __name__ == "__main__":
    main()
