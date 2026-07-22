#!/usr/bin/env python3
"""Honest scoreboard generated from aggregate summaries and reports/BASELINES.json."""
import argparse
import glob
import json
import os

SUITE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMMARY_DIR = os.path.join(SUITE, "results") if os.path.exists(os.path.join(SUITE, "results", "all-v2.summary.json")) else os.path.join(SUITE, "results-summary")

def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

registry = read_json(os.path.join(SUITE, "reports", "BASELINES.json"))
BASE = {row["id"]: row for row in registry["baselines"]}

def value(baseline_id):
    return BASE[baseline_id]["value"]

def build_published():
    imhi = BASE["imhi-weighted-f1"]["subtasks"]
    out = {}
    for subtask, row in imhi.items():
        key = "imhi-" + subtask.lower()
        out[key] = [
            ("ChatGPT-zs", "zsllm", row["chatgpt_zs"]),
            ("MentaLLaMA-13B", "domainllm", row["mentallama13b"]),
            (row["best_finetuned"]["name"] + "-ft", "finetuned", row["best_finetuned"]["value"]),
        ]
    out.update({
        "cpsyexam": [
            ("ChatGPT mixed-best-of", "mixed", value("cpsyexam-chatgpt-avg")),
            ("GPT-4 strict-zs derived", "zsllm", value("cpsyexam-gpt4-zeroshot-weighted")),
        ],
        "psysuicide": [
            ("majority", "baseline", value("psysuicide-majority")),
            ("GPT-4-preview-zs", "zsllm", value("psysuicide-gpt4-preview-acc")),
            ("RoBERTa-large-ft", "finetuned", value("psysuicide-roberta-large-acc")),
        ],
        "mentalmanip": [
            ("majority", "baseline", value("mentalmanip-majority")),
            ("GPT-4-Turbo-zs", "zsllm", value("mentalmanip-gpt4turbo")),
            ("RoBERTa-base-ft", "finetuned", value("mentalmanip-roberta-base")),
            ("Llama-2-13B-ft", "finetuned", value("mentalmanip-llama2-13b")),
        ],
        "eatd-depression": [],  # published F1 is not comparable to this script's accuracy
    })
    return out

REFERENCE_POINTS = build_published()
IMHI_F1 = {task for task in REFERENCE_POINTS if task.startswith("imhi-")} | {"psysuicide"}
allv2 = {row["task"]: row for row in read_json(os.path.join(SUMMARY_DIR, "all-v2.summary.json"))}

def numeric(task, model):
    if model == "chat":
        data = allv2.get(task)
    else:
        candidates = sorted(glob.glob(os.path.join(SUMMARY_DIR, f"{task}-deepseek-reasoner-*.summary.json")))
        data = read_json(candidates[-1]) if candidates else None
    if not data:
        return None
    metric_name = "wF1" if task in IMHI_F1 and data.get("weightedF1") is not None else "acc"
    metric = data["weightedF1"] * 100 if metric_name == "wF1" else data["accuracy"] * 100
    return metric, metric_name, data["n"]

def emo(task, model):
    path = os.path.join(SUMMARY_DIR, f"deepseek-{model}-{task}.summary.json")
    if not os.path.exists(path):
        return None
    return read_json(path)["combined"]["accuracy"] * 100

def selftest():
    required = {
        "cpsyexam-gpt4-zeroshot-weighted", "emobench-ea-gpt4-mean", "emobench-eu-gpt4-mean",
        "psysuicide-majority", "mentalmanip-majority", "imhi-weighted-f1",
    }
    missing = sorted(required - BASE.keys())
    if missing:
        raise SystemExit(f"missing baseline ids: {missing}")
    if len(allv2) != 19:
        raise SystemExit(f"expected 19 all-v2 summaries, got {len(allv2)}")
    print(f"scoreboard selftest PASS: baselines={len(BASE)}, summaries={len(allv2)}, source={SUMMARY_DIR}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    print("# 被测模型 vs 公开参考点（chat=非思考，reasoner=思考）\n")
    print("只比较已记录的点估计；不代表统计显著，采样、提示和协议差异须回到对应报告核对。\n")
    print(f"{'task':16s}{'chat':>7s}{'reas':>7s}{'best':>7s}{'cfg':>6s}{'metric':>7s} | 描述性参考差值")
    print("-" * 108)
    above, below = [], []

    def evaluate(task, label=None):
        chat, reasoner = numeric(task, "chat"), numeric(task, "reasoner")
        if not chat and not reasoner:
            return
        chat_value = chat[0] if chat else None
        reasoner_value = reasoner[0] if reasoner else None
        metric_name = (chat or reasoner)[1]
        best = max(value for value in (chat_value, reasoner_value) if value is not None)
        config = "reas" if reasoner_value is not None and reasoner_value == best else "chat"
        fair = [row for row in REFERENCE_POINTS.get(task, []) if row[1] in ("zsllm", "domainllm", "baseline")]
        finetuned = [row for row in REFERENCE_POINTS.get(task, []) if row[1] == "finetuned"]
        fair_best = max(fair, key=lambda row: row[2]) if fair else None
        ft_best = max(finetuned, key=lambda row: row[2]) if finetuned else None
        if fair_best and best >= fair_best[2]:
            verdict = f"点估计高于 {fair_best[0]}={fair_best[2]:.1f} (Δ{best-fair_best[2]:+.1f})"
            if ft_best and best < ft_best[2]: verdict += f"；仍低于微调 {ft_best[2]:.1f}"
            above.append(task)
        elif fair_best:
            verdict = f"点估计低于 {fair_best[0]}={fair_best[2]:.1f} (Δ{best-fair_best[2]:+.1f})"
            below.append(task)
        else:
            verdict = "无同指标公开参考，不判高低"
        cstr = f"{chat_value:6.1f}" if chat_value is not None else "   -  "
        rstr = f"{reasoner_value:6.1f}" if reasoner_value is not None else "   -  "
        print(f"{(label or task):16s}{cstr}{rstr}{best:7.1f}{config:>6s}{metric_name:>7s} | {verdict}")

    for task in ("EA", "EU"):
        chat, reasoner = emo(task, "chat"), emo(task, "reasoner")
        if chat is None and reasoner is None:
            continue
        best = max(value for value in (chat, reasoner) if value is not None)
        config = "reas" if reasoner is not None and reasoner == best else "chat"
        baseline_id = "emobench-ea-gpt4-mean" if task == "EA" else "emobench-eu-gpt4-mean"
        gpt4 = value(baseline_id)
        relation = "高于" if best >= gpt4 else "低于"
        verdict = f"点估计{relation} GPT-4论文中英均值={gpt4:.1f} (Δ{best-gpt4:+.1f})"
        (above if best >= gpt4 else below).append("emobench-" + task.lower())
        print(f"{'emobench-'+task.lower():16s}{chat or 0:6.1f}{reasoner or 0:6.1f}{best:7.1f}{config:>6s}{'acc':>7s} | {verdict}")

    for task in ["cpsyexam", "imhi-dr", "imhi-dreaddit", "imhi-loneliness", "imhi-irf", "imhi-multiwd",
                 "imhi-sad", "imhi-cams", "imhi-swmh", "imhi-t-sid", "psysuicide", "mentalmanip", "eatd-depression"]:
        evaluate(task)
    print(
        f"\n点估计高于参考：{len(above)}；低于参考：{len(below)}。"
        "这是描述性盘点，不是显著性检验或跨协议胜负。数值来源：reports/BASELINES.json"
    )

if __name__ == "__main__":
    main()
