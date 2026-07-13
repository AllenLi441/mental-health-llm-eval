#!/usr/bin/env python3
"""
CPsyExam 数据污染探针 (guided-instruction memorization test)

假设:若 deepseek-chat 训练时见过 CPsyExam 测试集,给它题干应能凭"记忆"
逐字复现选项。做法:抽 100 道测试题,只给题干,要求模型写出该题的全部选项;
用 difflib 相似度把生成选项与真实选项对齐打分。

判读(保守口径):
- 单题"复现" = 真实选项与生成选项最佳配对的平均相似度 ≥0.85 且 ≥75% 选项相似度 ≥0.80
- 汇总报告整体复现率,并单独报告"长选项子集"(平均选项长度≥8字,短选项如"以上都是"
  容易被猜中,不作污染证据)
- 复现率接近 0 → 逐字记忆无法解释 83% 的成绩;复现率高 → 污染实锤,评测作废

用法: set -a; . ../../app/.env.local; set +a; python3 contamination_probe.py
输出: results/contamination-probe-deepseek-chat.json
"""
import json, os, re, glob, random, difflib, urllib.request, concurrent.futures, time

BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
KEY = os.environ["DEEPSEEK_API_KEY"]
MODEL = "deepseek-chat"
N = 100
SEED = 7

def load_items():
    d = os.path.join(os.path.dirname(__file__), "..", "CPsyExam", "data", "extracted_with_answer", "test")
    items = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        for q in json.load(open(f, encoding="utf-8")):
            opts = q.get("options")
            if isinstance(opts, dict):
                opts = [str(v) for k, v in sorted(opts.items())]
            elif isinstance(opts, list):
                opts = [str(v) for v in opts]
            else:
                continue
            opts = [o.strip() for o in opts if o and str(o).strip()]
            if len(opts) >= 3 and q.get("question", "").strip():
                items.append({"id": q["id"], "question": q["question"].strip(), "options": opts})
    return items

def probe_one(it):
    prompt = ("下面是一道中国心理学考试题的题干。这道题出自一个公开考试题库。\n"
              "请你凭记忆,逐字写出这道题原本的全部选项(每行一个,格式如 A) ...)。\n"
              "只写选项内容,不要回答问题,不要解释。如果不记得,请尽力回忆并写出最可能的原选项。\n\n"
              f"题干:{it['question']}")
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)["choices"][0]["message"]["content"]
            break
        except Exception:
            if attempt == 2: return {**it, "gen": None}
            time.sleep(2 * (attempt + 1))
    # split generated lines into candidate options
    lines = [re.sub(r"^\s*[A-Ha-h][)．.、:：]\s*", "", l).strip()
             for l in out.splitlines() if l.strip()]
    lines = [l for l in lines if l]
    # best-match similarity for each REAL option against generated lines
    sims = []
    for o in it["options"]:
        best = max((difflib.SequenceMatcher(None, o, g).ratio() for g in lines), default=0.0)
        sims.append(round(best, 3))
    mean = sum(sims) / len(sims)
    frac80 = sum(1 for s in sims if s >= 0.80) / len(sims)
    return {**it, "gen": lines, "sims": sims, "mean_sim": round(mean, 3),
            "reproduced": mean >= 0.85 and frac80 >= 0.75}

def main():
    items = load_items()
    random.Random(SEED).shuffle(items)
    sample = items[:N]
    print(f"pool={len(items)} probing n={len(sample)} model={MODEL}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(probe_one, sample))
    ok = [r for r in results if r.get("gen") is not None]
    rep = [r for r in ok if r["reproduced"]]
    long_sub = [r for r in ok if sum(len(o) for o in r["options"]) / len(r["options"]) >= 8]
    rep_long = [r for r in long_sub if r["reproduced"]]
    means = sorted(r["mean_sim"] for r in ok)
    summary = {
        "model": MODEL, "n": len(ok), "errors": len(results) - len(ok),
        "reproduced": len(rep), "reproduction_rate": round(len(rep) / len(ok), 4),
        "long_option_subset_n": len(long_sub), "long_reproduced": len(rep_long),
        "long_reproduction_rate": round(len(rep_long) / max(1, len(long_sub)), 4),
        "mean_sim_median": means[len(means)//2], "mean_sim_p90": means[int(len(means)*0.9)],
        "criterion": "mean best-match sim>=0.85 AND >=75% options sim>=0.80 (difflib)",
    }
    outdir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "contamination-probe-deepseek-chat.json")
    json.dump({"summary": summary, "items": results}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    # show the top-3 most-similar cases for eyeballing
    for r in sorted(ok, key=lambda x: -x["mean_sim"])[:3]:
        print(f"\n--- top sim {r['mean_sim']} id={r['id']}")
        print("  Q:", r["question"][:60])
        for o, s in zip(r["options"], r["sims"]):
            print(f"  real({s}): {o[:46]}")
        for g in (r["gen"] or [])[:6]:
            print(f"  gen : {g[:46]}")
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
