#!/usr/bin/env python3
"""Regression gate for the single baseline registry and known stale-value traps."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(ROOT, "reports", "BASELINES.json"), encoding="utf-8") as handle:
    rows = json.load(handle)["baselines"]
ids = [row["id"] for row in rows]
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate baseline ids")

required = {
    "cpsyexam-gpt4-zeroshot-weighted",
    "cpsyexam-chatgpt-avg",
    "emobench-eu-gpt4-mean",
    "mentalmanip-roberta-base",
    "mentalmanip-llama2-13b",
    "psysuicide-roberta-large-acc",
    "psysuicide-roberta-large-microf1",
    "psysuicide-roberta-large-macrof1",
    "psysuicide-majority",
    "imhi-weighted-f1",
}
missing = required - set(ids)
if missing:
    raise SystemExit(f"missing required baselines: {sorted(missing)}")
by_id = {row["id"]: row for row in rows}

for row in rows:
    for field in ("id", "dataset", "method", "value", "metric", "scope", "source", "notes"):
        if field not in row:
            raise SystemExit(f"{row.get('id', '<unknown>')}: missing baseline field {field}")

expected = {
    "mentalmanip-roberta-base": (76.6, "accuracy"),
    "mentalmanip-llama2-13b": (76.8, "accuracy"),
    "psysuicide-roberta-large-acc": (91.69, "accuracy"),
    "psysuicide-roberta-large-microf1": (92.77, "micro-F1(11 类)"),
    "psysuicide-roberta-large-macrof1": (69.76, "macro-F1(11 类)"),
}
for baseline_id, (value, metric) in expected.items():
    row = by_id[baseline_id]
    if row["value"] != value or row["metric"] != metric:
        raise SystemExit(f"{baseline_id}: expected value={value} metric={metric}, got {row['value']} / {row['metric']}")
if by_id["cpsyexam-chatgpt-avg"]["scope"] != "few-shot-mixed":
    raise SystemExit("CPsyExam ChatGPT 51.15 must be labeled mixed zero/few-shot best-of, not zero-shot")
if by_id["cpsyexam-chatglm-turbo-avg"]["scope"] != "few-shot-mixed":
    raise SystemExit("CPsyExam ChatGLM-Turbo 64.58 must be labeled mixed zero/few-shot best-of")
if "10 个 test sets" not in by_id["imhi-weighted-f1"]["notes"] or "CLP" not in by_id["imhi-weighted-f1"]["notes"]:
    raise SystemExit("IMHI baseline must disclose the paper's 10-test-set scope and the omitted CLP task")

with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as handle:
    readme = handle.read()
if "多数类 72.4" in readme:
    raise SystemExit("README contains stale PsySUICIDE majority=72.4")
with open(os.path.join(ROOT, "scripts", "scoreboard.py"), encoding="utf-8") as handle:
    scoreboard = handle.read()
if 'BASELINES.json' not in scoreboard or "PUB={" in scoreboard.replace(" ", "") or "PUB =" in scoreboard:
    raise SystemExit("scoreboard is not registry-driven")
if "赢过" in scoreboard or "同口径对照" in scoreboard:
    raise SystemExit("scoreboard contains an unsupported win/same-protocol claim")
if "ChatGPT-zs\", \"zsllm\", value(\"cpsyexam-chatgpt-avg\")" in scoreboard:
    raise SystemExit("scoreboard mislabels CPsyExam ChatGPT mixed-best-of as zero-shot")

uniform_report = open(os.path.join(ROOT, "reports", "imhi_uniform_v3u.md"), encoding="utf-8").read()
if "(平手" in uniform_report or "稳健超 GPT-4" in uniform_report or "污染对双方对称,不影响相对结论" in uniform_report:
    raise SystemExit("IMHI report contains an unsupported tie/robust-win/symmetric-contamination claim")

for filename in os.listdir(os.path.join(ROOT, "results-summary")):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join(ROOT, "results-summary", filename), encoding="utf-8") as handle:
        data = json.load(handle)
    stack = list(data) if isinstance(data, list) else [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            method = str(node.get("method", "")).lower()
            if "majority" in method and "自杀" in method and node.get("value") != 71.4:
                raise SystemExit(f"{filename}: stale PsySUICIDE majority comparison {node.get('value')}")
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)

print(f"baseline registry check PASS: {len(rows)} unique entries; known stale traps absent")
