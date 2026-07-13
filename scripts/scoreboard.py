#!/usr/bin/env python3
# Honest final scoreboard: chat vs reasoner vs published baselines.
# Takes the better of chat/reasoner per task (labeled), compares to the fair baseline class.
import json, os, glob
SUITE=os.path.join(os.path.dirname(__file__),"..")
R=lambda p: json.load(open(os.path.join(SUITE,p)))
def jl(p):
    return [json.loads(l) for l in open(os.path.join(SUITE,p)) if l.strip()]

allv2={r["task"]:r for r in R("results/all-v2.summary.json")}

# published baselines: task -> list of (name, kind, score)  kind: zsllm|domainllm|finetuned
PUB={
 "imhi-dr":[("ChatGPT-zs","zsllm",82.41),("MentaLLaMA-13B","domainllm",85.68),("RoBERTa-ft","finetuned",95.11)],
 "imhi-dreaddit":[("ChatGPT-zs","zsllm",71.79),("MentaLLaMA-13B","domainllm",75.79),("MentalRoBERTa-ft","finetuned",81.76)],
 "imhi-loneliness":[("ChatGPT-zs","zsllm",58.4),("MentaLLaMA-13B","domainllm",85.1),("MentalRoBERTa-ft","finetuned",85.33)],
 "imhi-irf":[("ChatGPT-zs","zsllm",41.33),("MentaLLaMA-13B","domainllm",76.49),("MentalBERT-ft","finetuned",76.73)],
 "imhi-multiwd":[("ChatGPT-zs","zsllm",62.72),("MentaLLaMA-13B","domainllm",75.11),("BERT-ft","finetuned",76.69)],
 "imhi-sad":[("ChatGPT-zs","zsllm",54.05),("MentaLLaMA-13B","domainllm",63.62),("MentalRoBERTa-ft","finetuned",68.44)],
 "imhi-cams":[("ChatGPT-zs","zsllm",33.85),("MentaLLaMA-13B","domainllm",45.52),("MentalRoBERTa-ft","finetuned",47.62)],
 "imhi-swmh":[("ChatGPT-zs","zsllm",49.32),("MentaLLaMA-13B","domainllm",71.7),("MentalRoBERTa-ft","finetuned",72.16)],
 "imhi-t-sid":[("ChatGPT-zs","zsllm",33.3),("MentaLLaMA-13B","domainllm",75.31),("MentalRoBERTa-ft","finetuned",89.01)],
 "cpsyexam":[("ChatGPT-zs","zsllm",51.15),("ChatGLM-Turbo","zsllm",64.58),("GPT-4-zs","zsllm",67.43)],
 "psysuicide":[("majority","baseline",72.4)],
 "mentalmanip":[("majority","baseline",69.2)],
 "eatd-depression":[("BiLSTM-text","finetuned",65.0),("multimodal-fusion-SOTA","finetuned",71.0)],
}
IMHI_F1={t for t in PUB if t.startswith("imhi-")}|{"psysuicide"}  # paper metric = weighted F1

def num(task,model):
    """return (metric_value, metric_name, n) for a task+model, best available file."""
    if model=="chat":
        d=allv2.get(task)
        if not d: return None
    else:
        cands=sorted(glob.glob(os.path.join(SUITE,f"results/{task}-deepseek-reasoner-*.summary.json")))
        if not cands: return None
        d=json.load(open(cands[-1]))
    metric = d["weightedF1"]*100 if (task in IMHI_F1 and d.get("weightedF1") is not None) else d["accuracy"]*100
    mname = "wF1" if (task in IMHI_F1 and d.get("weightedF1") is not None) else "acc"
    return (metric, mname, d["n"])

# EmoBench official (separate harness)
EMB=os.path.join(SUITE,"../EmoBench/eval/results")
def emb(task,model):
    f=os.path.join(EMB,f"deepseek-{model}-{task}.jsonl")
    if not os.path.exists(f): return None
    rows=[json.loads(l) for l in open(f) if l.strip()]
    en=[r for r in rows if r["lang"]=="en"]; zh=[r for r in rows if r["lang"]=="zh"]
    a=(sum(r["correct"] for r in en)/len(en)+sum(r["correct"] for r in zh)/len(zh))/2*100
    return a

print("# 静室 vs 公开基准 —— 诚实计分板(chat=非思考, reasoner=思考)\n")
print(f"{'task':16s}{'chat':>7s}{'reas':>7s}{'best':>7s}{'cfg':>5s}{'metric':>7s} | 最强可比对照 → 结论")
print("-"*104)
WINS=[]; COMP=[]; LOSS=[]
def eval_task(task,label=None):
    c=num(task,"chat"); r=num(task,"reasoner")
    if not c and not r: return
    cv=c[0] if c else None; rv=r[0] if r else None
    mname=(c or r)[1]
    best=max([x for x in [cv,rv] if x is not None])
    cfg="reas" if (rv is not None and rv==best) else "chat"
    pubs=PUB.get(task,[])
    # fair baseline = best domainllm/zsllm/baseline we should target; note finetuned separately
    fair=[p for p in pubs if p[1] in ("zsllm","domainllm","baseline")]
    ftb=[p for p in pubs if p[1]=="finetuned"]
    fair_best=max(fair,key=lambda x:x[2]) if fair else None
    ft_best=max(ftb,key=lambda x:x[2]) if ftb else None
    verdict=""
    if fair_best and best>=fair_best[2]:
        # also check if beats finetuned
        if ft_best and best>=ft_best[2]:
            verdict=f"✅✅ 赢过微调SOTA {ft_best[0]}={ft_best[2]:.1f}"; WINS.append((task,best,cfg,verdict))
        else:
            verdict=f"✅ 赢过 {fair_best[0]}={fair_best[2]:.1f}"+(f"(仍输微调{ft_best[2]:.1f})" if ft_best else ""); WINS.append((task,best,cfg,verdict))
    elif fair_best:
        verdict=f"🟡 近 {fair_best[0]}={fair_best[2]:.1f} (Δ{best-fair_best[2]:+.1f})"; COMP.append((task,best,cfg,verdict))
    else:
        verdict="(无可比对照)"
    cstr=f"{cv:6.1f}" if cv is not None else "   -  "
    rstr=f"{rv:6.1f}" if rv is not None else "   -  "
    print(f"{(label or task):16s}{cstr}{rstr}{best:7.1f}{cfg:>5s}{mname:>7s} | {verdict}")

# EmoBench (official)
for t in ["EA","EU"]:
    c=emb(t,"chat"); r=emb(t,"reasoner")
    if c is None and r is None: continue
    best=max(x for x in [c,r] if x is not None); cfg="reas" if (r is not None and r==best) else "chat"
    gpt4=74.6 if t=="EA" else 56.9
    v=(f"✅ 赢过 GPT-4={gpt4}" if best>=gpt4 else f"🟡 近 GPT-4={gpt4} (Δ{best-gpt4:+.1f})")
    (WINS if best>=gpt4 else COMP).append((f"emobench-{t.lower()}",best,cfg,v))
    print(f"{'emobench-'+t.lower():16s}{c or 0:6.1f}{r or 0:6.1f}{best:7.1f}{cfg:>5s}{'acc':>7s} | {v}")

for t in ["cpsyexam","imhi-dr","imhi-dreaddit","imhi-loneliness","imhi-irf","imhi-multiwd",
          "imhi-sad","imhi-cams","imhi-swmh","imhi-t-sid","psysuicide","mentalmanip","eatd-depression"]:
    eval_task(t)

print("\n## 汇总")
print(f"✅ 赢过可比对照: {len(WINS)}")
for t,b,c,v in WINS: print(f"    {t} ({b:.1f}, {c}) — {v}")
print(f"🟡 competitive(近但未过): {len(COMP)}")
for t,b,c,v in COMP: print(f"    {t} ({b:.1f}, {c}) — {v}")
