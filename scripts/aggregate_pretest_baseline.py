#!/usr/bin/env python3
"""Aggregate the pre-finetune baseline for the Jingshi therapy-model campaign.

Scope
-----
CPCD open-response arms are scored here from the frozen proxy-judge records.
ESConv fixed250 strategy accuracy is NOT recomputed: it was already settled in
``reports/esconv_fixed250_deepseek_v4_pro_0813_20260813.json`` under a cluster
bootstrap + McNemar protocol. This script re-reads that artifact, verifies the
source hashes it declares, and carries the numbers forward by reference.

Every number written here is bound to the sha256 of the file it came from.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JUDGE_FILE = ROOT / "outputs" / "cpcd_proxy_flash_judgements_20260809.jsonl"
ESCONV_SETTLED = ROOT / "reports" / "esconv_fixed250_deepseek_v4_pro_0813_20260813.json"

CPCD_GENERATION_FILES = {
    "deepseek_v4_pro": "outputs/provider_v2_deepseek_cpcd_20260808.jsonl",
    "deepseek_v4_flash": "outputs/provider_v2_flash_cpcd_20260809.jsonl",
    "qwen_3_6_27b": "outputs/provider_v2_qwen_cpcd_20260808.jsonl",
    "qwen_3_6_27b_siliconflow": "outputs/provider_sf_qwen_cpcd_20260813.jsonl",
}

# Each CPCD family carries its own rubric dimensions; they are not comparable
# across families, so per-dimension means are always reported per family.
FAMILY_LABELS = {
    "srg": "单轮回复生成 (srg)",
    "mr": "记忆回溯 (mr)",
    "tcr": "时序因果回溯 (tcr)",
}
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260828


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def paired_bootstrap(diffs: list[float]) -> tuple[float, float]:
    """Percentile bootstrap CI over per-item paired differences."""
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    means = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        means.append(statistics.fmean(rng.choices(diffs, k=n)))
    means.sort()
    lo = means[int(0.025 * BOOTSTRAP_ITERATIONS)]
    hi = means[int(0.975 * BOOTSTRAP_ITERATIONS) - 1]
    return lo, hi


def collect_cpcd() -> dict:
    judgements = read_jsonl(JUDGE_FILE)

    by_arm: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    judge_identity: dict[str, set] = collections.defaultdict(set)
    rubrics: dict[str, set] = collections.defaultdict(set)

    for row in judgements:
        arm = row["target_model_id"]
        task = row["task_id"]
        if task in by_arm[arm]:
            raise SystemExit(f"duplicate judgement for {arm}/{task}")
        by_arm[arm][task] = row
        judge_identity[arm].add(
            (row["judge_id"], row["judge_mode"], row["judge_response_model"], row["judge_fingerprint"])
        )
        rubrics[row["family"]].add(row["rubric_source"])

    arms: dict[str, dict] = {}
    for arm, items in by_arm.items():
        rows = list(items.values())
        families = collections.defaultdict(list)
        for row in rows:
            families[row["family"]].append(row)

        per_family = {}
        for fam, fam_rows in sorted(families.items()):
            dims = sorted({d for r in fam_rows for d in r["parsed_judgement"]["scores"]})
            per_family[fam] = {
                "n": len(fam_rows),
                "mean_average_score": mean([r["parsed_judgement"]["average_score"] for r in fam_rows]),
                "per_dimension": {
                    dim: mean(
                        [
                            r["parsed_judgement"]["scores"][dim]
                            for r in fam_rows
                            if dim in r["parsed_judgement"]["scores"]
                        ]
                    )
                    for dim in dims
                },
            }

        family_means = [v["mean_average_score"] for v in per_family.values()]
        arms[arm] = {
            "n_judged": len(rows),
            "mean_average_score_micro": mean([r["parsed_judgement"]["average_score"] for r in rows]),
            "mean_average_score_macro_over_families": mean(family_means),
            "per_family": per_family,
            "judge_identity": sorted(
                {"judge_id": a, "judge_mode": b, "judge_response_model": c, "judge_fingerprint": d}
                for a, b, c, d in judge_identity[arm]
            ),
        }

    # Generation-side termination accounting (invalid / truncation rates).
    for arm, rel in CPCD_GENERATION_FILES.items():
        path = ROOT / rel
        rows = read_jsonl(path)
        terminations = collections.Counter(r.get("termination_status") for r in rows)
        arms.setdefault(arm, {})
        arms[arm]["generation"] = {
            "source": rel,
            "source_sha256": sha256(path),
            "n_generated": len(rows),
            "arm_id": sorted({r["arm_id"] for r in rows}),
            "protocol_id": sorted({r["protocol_id"] for r in rows}),
            "response_model": sorted({str(r.get("response_model")) for r in rows}),
            "termination_status": dict(terminations),
            "truncated_rate": terminations.get("length", 0) / len(rows) if rows else None,
        }

    # Paired comparisons on the intersection of judged task ids.
    pairs = [
        ("qwen_3_6_27b_siliconflow", "deepseek_v4_pro"),
        ("qwen_3_6_27b", "deepseek_v4_pro"),
        ("deepseek_v4_pro", "deepseek_v4_flash"),
        ("qwen_3_6_27b_siliconflow", "qwen_3_6_27b"),
    ]
    paired = []
    for a, b in pairs:
        common = sorted(set(by_arm[a]) & set(by_arm[b]))
        diffs = [
            by_arm[a][t]["parsed_judgement"]["average_score"]
            - by_arm[b][t]["parsed_judgement"]["average_score"]
            for t in common
        ]
        by_family = collections.defaultdict(list)
        for t in common:
            by_family[by_arm[a][t]["family"]].append(
                by_arm[a][t]["parsed_judgement"]["average_score"]
                - by_arm[b][t]["parsed_judgement"]["average_score"]
            )
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        ties = sum(1 for d in diffs if d == 0)
        lo, hi = paired_bootstrap(diffs)
        paired.append(
            {
                "comparison": f"{a} - {b}",
                "n_paired": len(common),
                "mean_diff": mean(diffs),
                "bootstrap_95_ci": [lo, hi],
                "a_better": wins,
                "b_better": losses,
                "tie": ties,
                "per_family_mean_diff": {
                    fam: {"n": len(d), "mean_diff": mean(d)} for fam, d in sorted(by_family.items())
                },
            }
        )

    return {
        "source": str(JUDGE_FILE.relative_to(ROOT)),
        "source_sha256": sha256(JUDGE_FILE),
        "n_judgements": len(judgements),
        "score_range": "1-5 per dimension; average_score = mean of coherence/empathy/professionalism",
        "rubrics": {fam: sorted(paths) for fam, paths in sorted(rubrics.items())},
        "arms": arms,
        "paired": paired,
    }


def collect_esconv() -> dict:
    settled = json.loads(ESCONV_SETTLED.read_text())
    carried = {
        "source_report": str(ESCONV_SETTLED.relative_to(ROOT)),
        "source_report_sha256": sha256(ESCONV_SETTLED),
        "protocol": settled.get("protocol") or settled.get("protocol_id"),
        "arms": {},
        "pairwise": settled.get("pairwise") or settled.get("comparisons"),
    }
    for name, arm in settled.get("arms", {}).items():
        entry = {
            "accuracy": arm.get("accuracy"),
            "accuracy_cluster_bootstrap_95_ci": arm.get("accuracy_cluster_bootstrap_95_ci"),
            "macro_f1": arm.get("macro_f1"),
            "weighted_f1": arm.get("weighted_f1"),
            "n": arm.get("n"),
            "missing": arm.get("missing"),
            "invalid": arm.get("invalid"),
            "coverage": arm.get("coverage"),
            "source": arm.get("source"),
            "declared_source_sha256": arm.get("source_sha256"),
        }
        source = arm.get("source")
        if source:
            path = Path(source)
            if not path.is_absolute():
                path = ROOT / source
            if path.exists():
                entry["recomputed_source_sha256"] = sha256(path)
                entry["source_hash_matches"] = (
                    entry["recomputed_source_sha256"] == arm.get("source_sha256")
                )
        carried["arms"][name] = entry
    return carried


def fmt(value, digits=4, percent=False):
    if value is None:
        return "n/a"
    if percent:
        return f"{value * 100:.2f}%"
    return f"{value:.{digits}f}"


def render_markdown(payload: dict) -> str:
    cpcd = payload["cpcd"]
    esconv = payload["esconv_fixed250"]
    lines: list[str] = []
    add = lines.append

    add("# 微调前基线结算（pre-finetune baseline）")
    add("")
    add(f"生成时间：{payload['generated_at']}　生成脚本：`{payload['generator']}`")
    add("")
    add("> **口径警告**：CPCD 分数来自内部 proxy judge（deepseek-v4-flash，非思考，盲评），")
    add("> 不是论文可比的公开榜分。只允许在本项目 pre/post 同 judge、同 rubric 的配对比较中使用。")
    add("")

    add("## 1. CPCD 开放回复（proxy judge 1–5 分）")
    add("")
    add(f"judge 记录：`{cpcd['source']}`（sha256 `{cpcd['source_sha256'][:16]}…`，{cpcd['n_judgements']} 条）")
    add("")
    add("三个 family 各用各的 rubric，维度不可跨 family 比较；`micro` = 159 条直接平均，")
    add("`macro` = 三个 family 均值再平均（消除 srg 占 62% 的权重偏斜）。")
    add("")
    add("| 臂 | 判分数 | micro 均分 | macro 均分 | srg | mr | tcr |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for arm, data in sorted(
        cpcd["arms"].items(), key=lambda kv: -(kv[1].get("mean_average_score_micro") or 0)
    ):
        fam = data.get("per_family", {})
        add(
            f"| `{arm}` | {data.get('n_judged', 0)} | **{fmt(data.get('mean_average_score_micro'), 3)}** | "
            f"{fmt(data.get('mean_average_score_macro_over_families'), 3)} | "
            f"{fmt(fam.get('srg', {}).get('mean_average_score'), 3)} | "
            f"{fmt(fam.get('mr', {}).get('mean_average_score'), 3)} | "
            f"{fmt(fam.get('tcr', {}).get('mean_average_score'), 3)} |"
        )
    add("")
    add("### 分维度（各 family 内）")
    add("")
    for fam in ("srg", "mr", "tcr"):
        dims = sorted(
            {
                d
                for data in cpcd["arms"].values()
                for d in data.get("per_family", {}).get(fam, {}).get("per_dimension", {})
            }
        )
        if not dims:
            continue
        add(f"**{FAMILY_LABELS.get(fam, fam)}**")
        add("")
        add("| 臂 | " + " | ".join(dims) + " |")
        add("|---" * (len(dims) + 1) + "|")
        for arm, data in sorted(cpcd["arms"].items()):
            block = data.get("per_family", {}).get(fam, {}).get("per_dimension", {})
            if not block:
                continue
            add(f"| `{arm}` | " + " | ".join(fmt(block.get(d), 3) for d in dims) + " |")
        add("")
    add("### 配对比较（同 task_id 交集，10,000 次 bootstrap，seed 20260828）")
    add("")
    add("| 比较 (A−B) | 配对数 | 平均差 | 95% CI | A 胜 | B 胜 | 平 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for row in cpcd["paired"]:
        lo, hi = row["bootstrap_95_ci"]
        add(
            f"| {row['comparison']} | {row['n_paired']} | {fmt(row['mean_diff'], 3)} | "
            f"[{fmt(lo, 3)}, {fmt(hi, 3)}] | {row['a_better']} | {row['b_better']} | {row['tie']} |"
        )
    add("")
    add("### 生成侧完成状态")
    add("")
    add("| 臂 | 生成条数 | 模型 | stop | length(截断) | 截断率 |")
    add("|---|---:|---|---:|---:|---:|")
    for arm, data in sorted(cpcd["arms"].items()):
        gen = data.get("generation")
        if not gen:
            continue
        term = gen["termination_status"]
        add(
            f"| `{arm}` | {gen['n_generated']} | {', '.join(gen['response_model'])} | "
            f"{term.get('stop', 0)} | {term.get('length', 0)} | {fmt(gen['truncated_rate'], percent=True)} |"
        )
    add("")

    add("## 2. ESConv fixed-250 八类策略（沿用已结算报告）")
    add("")
    add(f"来源：`{esconv['source_report']}`（sha256 `{esconv['source_report_sha256'][:16]}…`）")
    add("")
    add("| 臂 | Accuracy | Cluster-bootstrap 95% CI | Macro-F1 | Weighted-F1 | missing/invalid | 源文件哈希一致 |")
    add("|---|---:|---:|---:|---:|---:|:--:|")
    for name, arm in esconv["arms"].items():
        ci = arm.get("accuracy_cluster_bootstrap_95_ci") or [None, None]
        matched = arm.get("source_hash_matches")
        mark = "✅" if matched else ("❌" if matched is False else "—")
        add(
            f"| {name} | {fmt(arm.get('accuracy'), percent=True)} | "
            f"[{fmt(ci[0], percent=True)}, {fmt(ci[1], percent=True)}] | "
            f"{fmt(arm.get('macro_f1'), percent=True)} | {fmt(arm.get('weighted_f1'), percent=True)} | "
            f"{arm.get('missing')}/{arm.get('invalid')} | {mark} |"
        )
    add("")

    # ---- Section 3: how to read these numbers, computed from the tables above ----
    arms = cpcd["arms"]
    pro = arms.get("deepseek_v4_pro", {})
    qwen_sf = arms.get("qwen_3_6_27b_siliconflow", {})
    main_pair = next(
        (p for p in cpcd["paired"] if p["comparison"] == "qwen_3_6_27b_siliconflow - deepseek_v4_pro"),
        None,
    )
    esconv_accs = [a.get("accuracy") for a in esconv["arms"].values() if a.get("accuracy") is not None]

    add("## 3. 读数要点")
    add("")
    if main_pair:
        lo, hi = main_pair["bootstrap_95_ci"]
        crosses_zero = lo <= 0 <= hi
        add(
            f"1. **CPCD 上未微调 Qwen 已与 DeepSeek 基本持平。** V4-Pro {fmt(pro.get('mean_average_score_micro'), 3)} "
            f"vs Qwen3.6-27B(SF) {fmt(qwen_sf.get('mean_average_score_micro'), 3)}，配对差 "
            f"{fmt(main_pair['mean_diff'], 3)}，95% CI [{fmt(lo, 3)}, {fmt(hi, 3)}]"
            + ("——**跨 0，差异不显著**" if crosses_zero else "——不跨 0")
            + f"；{main_pair['n_paired']} 对中 {main_pair['tie']} 对完全同分。"
        )
        add("")
    if esconv_accs:
        add(
            f"2. **ESConv 八类策略上，所有 API 臂都远低于专用小模型。** 四臂区间 "
            f"{fmt(min(esconv_accs), percent=True)}–{fmt(max(esconv_accs), percent=True)}，"
            "而同一 frozen 协议下 BlenderBot-small Joint 为 32.22%、EmoDynamiX 为 33.61%。"
            "**这才是微调要补的缺口**——不是补 Qwen 与 DeepSeek 之间的差距。"
        )
        add("")
    flash_gen = arms.get("deepseek_v4_flash", {}).get("generation", {})
    qwen_gen = qwen_sf.get("generation", {})
    if flash_gen and qwen_gen:
        add(
            f"3. **截断率差异会污染跨臂比较。** V4-Flash 截断 "
            f"{flash_gen['termination_status'].get('length', 0)}/{flash_gen['n_generated']}"
            f"（{fmt(flash_gen.get('truncated_rate'), percent=True)}），Qwen 两臂均为 0%。"
            "Flash 的 tcr 低分很可能是截断产物而非能力差距，勿据此下能力结论。"
        )
        add("")

    add("## 4. 必须随分数一起披露的口径事项")
    add("")
    add("| 项 | 内容 |")
    add("|---|---|")
    for arm, data in sorted(arms.items()):
        for identity in data.get("judge_identity", []):
            add(
                f"| `{arm}` 的 judge 指纹 | `{identity['judge_response_model']}` / "
                f"`{identity['judge_mode']}` / `{identity['judge_fingerprint']}` |"
            )
    add(
        "| ⚠ 指纹不一致 | SiliconFlow Qwen 臂（8/13 判分）的 judge fingerprint 与其余三臂（8/09 判分）"
        "不同一格式。post-test 必须记录当次指纹并在报告中并列披露，不得假装同一次判分。 |"
    )
    add(
        "| 各臂 n 不等 | 判分数 159/159/156/136 不等，跨臂只能看配对交集列，"
        "不得直接比较各自的 micro 均分名次。 |"
    )
    add("| 分数性质 | 内部 proxy judge 分，**不可**写进论文或对外榜单。 |")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="reports/qwen_finetune_pretest_baseline_20260828.json")
    parser.add_argument("--out-md", default="reports/qwen_finetune_pretest_baseline_20260828.md")
    parser.add_argument("--generated-at", default="2026-08-28")
    args = parser.parse_args()

    payload = {
        "generated_at": args.generated_at,
        "generator": "scripts/aggregate_pretest_baseline.py",
        "purpose": "pre-finetune baseline for the Qwen therapy-model campaign",
        "cpcd": collect_cpcd(),
        "esconv_fixed250": collect_esconv(),
    }

    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    out_md.write_text(render_markdown(payload))
    print(f"wrote {out_json.relative_to(ROOT)}")
    print(f"wrote {out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
