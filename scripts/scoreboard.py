#!/usr/bin/env python3
"""Current, metric-matched scoreboard from pinned aggregate evidence.

This script intentionally does not scan for the lexicographically latest result.
Every source below is a named, reviewed protocol artifact. External paper rows
are descriptive point comparisons unless the report explicitly contains paired
predictions and a preregistered test.
"""
import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results-summary"
REPORTS = ROOT / "reports"


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


registry = read_json(REPORTS / "BASELINES.json")
BASE = {row["id"]: row for row in registry["baselines"]}


def baseline(baseline_id):
    return BASE[baseline_id]


def percent(value):
    return float(value) * 100


def add_comparison(rows, *, task, protocol, n, metric, score, baseline_id,
                   model, evidence, note=""):
    ref = baseline(baseline_id)
    normalized_result_metric = metric.lower().replace("accuracy", "acc").replace("-", "")
    normalized_reference_metric = ref["metric"].lower().replace("accuracy", "acc").replace("-", "")
    if normalized_result_metric not in normalized_reference_metric:
        raise ValueError(
            f"metric mismatch for {task}: result={metric!r}, "
            f"baseline={ref['metric']!r}"
        )
    if not isinstance(ref["value"], (int, float)):
        raise ValueError(f"baseline {baseline_id} is not a scalar point estimate")
    rows.append({
        "task": task,
        "protocol": protocol,
        "n": int(n),
        "metric": metric,
        "score": float(score),
        "model": model,
        "comparator": ref["method"],
        "comparator_score": float(ref["value"]),
        "comparator_scope": ref["scope"],
        "delta_pp": float(score) - float(ref["value"]),
        "evidence": evidence,
        "note": note,
    })


def current_rows():
    rows = []

    # CPsyExam: the full 3,902-item paired V4 artifact supersedes the 599-item pilot.
    cpsy = read_json(SUMMARY / "cpsyexam-v4-full-paired.summary.json")
    pro = cpsy["arms"]["v4_pro"]
    if pro["requested_models"] != ["deepseek-v4-pro"] or pro["rows"] != 3902:
        raise ValueError("unexpected CPsyExam V4 Pro identity or denominator")
    add_comparison(
        rows,
        task="cpsyexam",
        protocol="full-paired-v4",
        n=pro["rows"],
        metric="accuracy",
        score=100 * pro["correct"] / pro["rows"],
        baseline_id="cpsyexam-gpt4-zeroshot-weighted",
        model="deepseek-v4-pro",
        evidence="internal Pro-vs-Flash claim is paired; paper comparison is cross-protocol descriptive",
        note="599-item pilot excluded; GPT-4 value is a derived strict-zero-shot weighted mean",
    )

    # EmoBench: exact current V4 Pro deterministic proxy, not the older chat/reasoner files.
    emo = read_json(SUMMARY / "v4-pro-pilot-terminal.audit.json")
    if emo.get("model") != "deepseek-v4-pro":
        raise ValueError("unexpected EmoBench V4 Pro identity")
    for key, baseline_id, task in (
        ("emobench_ea", "emobench-ea-gpt4-mean", "emobench-ea"),
        ("emobench_eu", "emobench-eu-gpt4-mean", "emobench-eu"),
    ):
        result = emo[key]
        if result["n"] != 400:
            raise ValueError(f"{task} is not the full 400-item proxy")
        add_comparison(
            rows,
            task=task,
            protocol="temp0-single-proxy",
            n=result["n"],
            metric="accuracy",
            score=percent(result["accuracy"]),
            baseline_id=baseline_id,
            model="deepseek-v4-pro",
            evidence="cross-protocol descriptive",
            note="paper uses repeated sampling and option permutations; comparator is paper en/zh mean",
        )

    # PsySUICIDE: use the preregistered official-test candidate, with same-metric
    # paper references. Accuracy and macro-F1 are deliberately separate rows.
    psy = read_json(REPORTS / "psytest-20260728-confirmatory-analysis.json")
    candidate = psy["candidate"]
    if (
        psy["scope"] != "one frozen paired PsySUICIDE official test campaign"
        or candidate["model"] != "deepseek-v4-pro"
        or candidate["rows"] != 1464
    ):
        raise ValueError("unexpected PsySUICIDE confirmation identity or denominator")
    add_comparison(
        rows,
        task="psysuicide",
        protocol="official-test-taxonomy",
        n=candidate["rows"],
        metric="accuracy",
        score=percent(candidate["accuracy"]),
        baseline_id="psysuicide-gpt4-preview-acc",
        model=candidate["model"],
        evidence="cross-protocol descriptive",
        note="same metric; paper baseline has no row-level predictions for a paired test",
    )
    add_comparison(
        rows,
        task="psysuicide",
        protocol="official-test-taxonomy",
        n=candidate["rows"],
        metric="accuracy",
        score=percent(candidate["accuracy"]),
        baseline_id="psysuicide-roberta-large-acc",
        model=candidate["model"],
        evidence="cross-protocol descriptive",
        note="zero-shot prompted model versus supervised fine-tuned classifier",
    )
    add_comparison(
        rows,
        task="psysuicide",
        protocol="official-test-taxonomy",
        n=candidate["rows"],
        metric="macro-F1",
        score=percent(candidate["macro_f1"]),
        baseline_id="psysuicide-roberta-large-macrof1",
        model=candidate["model"],
        evidence="cross-protocol descriptive",
        note="primary metric matched; supervised fine-tuned comparator",
    )

    # IMHI: v3u is the only uniform, non-selective prompt protocol. Take the
    # reported best arm within that already-declared protocol, never v1/v2/v3.
    imhi_registry = baseline("imhi-weighted-f1")["subtasks"]
    for subtask, published in imhi_registry.items():
        task = f"imhi-{subtask.lower()}"
        candidates = []
        for path in sorted(glob.glob(str(SUMMARY / f"{task}-*-v3u.summary.json"))):
            data = read_json(path)
            api_models = data.get("api_models", [])
            if api_models != ["deepseek-v4-flash"]:
                raise ValueError(f"{path}: unverified IMHI response model identity")
            if data.get("errors") != 0:
                raise ValueError(f"{path}: IMHI v3u contains API errors")
            candidates.append((data["weightedF1"], data, Path(path).name))
        if not candidates:
            raise ValueError(f"missing uniform v3u result for {task}")
        _, result, source_name = max(candidates, key=lambda row: row[0])
        for method, scope, score in (
            ("ChatGPT", "zero-shot", published["chatgpt_zs"]),
            ("MentaLLaMA-chat-13B", "domain-llm", published["mentallama13b"]),
            (published["best_finetuned"]["name"], "fine-tuned",
             published["best_finetuned"]["value"]),
        ):
            rows.append({
                "task": task,
                "protocol": "uniform-v3u-reported-best-arm",
                "n": int(result["n"]),
                "metric": "weighted-F1",
                "score": percent(result["weightedF1"]),
                "model": "deepseek-v4-flash",
                "comparator": method,
                "comparator_score": float(score),
                "comparator_scope": scope,
                "delta_pp": percent(result["weightedF1"]) - float(score),
                "evidence": "cross-protocol descriptive",
                "note": f"9/10 paper test-set subset; source={source_name}",
            })

    return rows


def selftest():
    required = {
        "cpsyexam-gpt4-zeroshot-weighted",
        "emobench-ea-gpt4-mean",
        "emobench-eu-gpt4-mean",
        "psysuicide-gpt4-preview-acc",
        "psysuicide-roberta-large-acc",
        "psysuicide-roberta-large-macrof1",
        "imhi-weighted-f1",
    }
    missing = sorted(required - BASE.keys())
    if missing:
        raise SystemExit(f"missing baseline ids: {missing}")
    rows = current_rows()
    if len(rows) != 33:
        raise SystemExit(f"expected 33 metric-matched comparison rows, got {len(rows)}")
    if any(row["metric"] == "weighted-F1" and row["task"] == "psysuicide" for row in rows):
        raise SystemExit("PsySUICIDE weighted-F1 must not be compared with an accuracy baseline")
    imhi = [row for row in rows if row["task"].startswith("imhi-")]
    scopes = {row["comparator_scope"] for row in imhi}
    if scopes != {"zero-shot", "domain-llm", "fine-tuned"}:
        raise SystemExit(f"IMHI comparator scopes are incomplete: {scopes}")
    print(
        "scoreboard selftest PASS: pinned current artifacts, full denominators, "
        "same-metric comparisons, IMHI uniform-v3u only"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    rows = current_rows()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print("# 当前证据 scoreboard（固定聚合资产；同指标比较）\n")
    print("所有论文比较均为跨协议点估计；只有各自报告中明确预注册且有逐行配对的内部模型比较可以作显著性结论。\n")
    print(f"{'task':17s} {'protocol':27s} {'n':>5s} {'metric':>11s} {'ours':>7s} {'reference(scope)':34s} {'delta':>8s}")
    print("-" * 122)
    for row in rows:
        reference = f"{row['comparator']}({row['comparator_scope']})"
        print(
            f"{row['task']:17.17s} {row['protocol']:27.27s} {row['n']:5d} "
            f"{row['metric']:>11.11s} {row['score']:7.2f} "
            f"{reference:34.34s} {row['delta_pp']:+7.2f}pp"
        )

    imhi = [row for row in rows if row["task"].startswith("imhi-")]
    counts = {}
    for scope in ("zero-shot", "domain-llm", "fine-tuned"):
        subset = [row for row in imhi if row["comparator_scope"] == scope]
        counts[scope] = sum(row["delta_pp"] > 0 for row in subset)
    print(
        "\nIMHI uniform-v3u 9/10 子集的描述性点值："
        f"{counts['zero-shot']}/9 高于 ChatGPT zero-shot，"
        f"{counts['domain-llm']}/9 高于 MentaLLaMA-13B，"
        f"{counts['fine-tuned']}/9 高于 fine-tuned 判别式。"
    )
    print(
        "PsySUICIDE：accuracy 比 GPT-4-preview zero-shot 高，但 accuracy 与 macro-F1 "
        "仍低于 RoBERTa-large fine-tuned；未把 weighted-F1 错配到 accuracy。"
    )


if __name__ == "__main__":
    main()
