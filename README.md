# eval-suite — 8 数据集 · 19 任务 · 零样本准确率评测

零依赖 Node.js（≥18）。被测模型 = 任意 OpenAI 兼容 chat API（默认 DeepSeek `deepseek-chat`）。

## 用法

```bash
echo "EVAL_API_KEY=sk-..." > .env        # 三选一：EVAL_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY
node run.mjs list                        # 任务清单
node run.mjs all --selftest              # 本地自检（不发请求）
node run.mjs psysuicide --sample 200     # 单任务，抽样覆盖默认值
node run.mjs all --run-id v1             # 全套顺序跑，汇总写 results/all-v1.summary.json
node run.mjs imhi-dr --resume results/imhi-dr-deepseek-chat-run.jsonl   # 断点续跑
```

通用协议：零样本、temperature 0、严格标签解析（无同义词映射，解析失败记 invalid 并算错）、
抽样用种子 42 确定性洗牌（`--seed` 可换）、逐条落盘 JSONL 可续跑、汇总含 accuracy + Wilson 95% CI +
weighted/macro-F1 + 逐类精度 + 混淆对。

## 任务与对比基准

| 任务 | 数据 | 量（默认抽样） | 指标 | 已发表参照 |
|---|---|---|---|---|
| emobench-ea | EmoBench EA | 400（全量） | acc（分中英） | GPT-4: 75.50 en / 73.75 zh |
| emobench-eu | EmoBench EU | 400（全量） | acc，情绪+原因全对 | GPT-4: 59.75 en / 54.12 zh |
| mdd5k-diagnosis | MDD-5k | 925（抽 400） | acc（5 类，ICD 归并，**本模块自定协议**） | 无统一榜单，横向比模型用 |
| psysuicide | PsySUICIDE test | 1,464（抽 500） | weighted-F1 / acc（11 类） | 多数类 72.4%；论文微调基线见 PDF |
| cbt-cd / pc / fc | CBT-Bench | 146/184/112（全量） | top-1 命中率（金标多标签） | 论文用 multi-label F1，口径不同 |
| mentalmanip | MentalManip con | 2,915（抽 500） | acc / F1 | 多数类 69.2%；GPT-4 基线见 PDF |
| imhi-dr 等 9 个 | IMHI test | 405~10,861（各抽 500） | weighted-F1 / acc | ChatGPT-ZS / 最强微调 / MentaLLaMA-13B 已写入各任务 comparisons |
| cpsyexam | CPsyExam test | 3,902（全量） | acc（KG/CA × 单选/多选分组） | GPT-4 零样本 76.56/10.76/60.33/13.00 |
| eatd-depression | EATD validation | 79（全量） | F1(抑郁) / acc | 论文文本 BiLSTM F1 0.65，融合 0.71 |

注意口径：我们是**零样本 LLM**，与「微调」参照比较时要注明设定差异；MDD-5k 和 CBT-Bench
的协议细节（ICD 归并 / top-1 命中）是本套件定义的，报告时须写明。

## 结构

```
eval-suite/
├── lib.mjs        共享：配置/.env、CSV 解析、API 重试、并发池、指标、任务运行器
├── run.mjs        入口（list / all / 单任务）
├── tasks/         emobench mdd5k psysuicide cbtbench mentalmanip imhi cpsyexam eatd
└── results/       <task>-<model>-<runid>.jsonl + .summary.json
```

## 审计与复现口径(2026-07-07 固化)

- **v1 主跑分 = 17 个非 EmoBench 任务**(`run-v1.log`,10,142 次调用);**v2(2026-07-07)= 全部 19 任务重跑**,
  17 主任务中 16 项与 v1 精确一致、cbt-fc -0.89pp(1 题输出波动),overview 固化于 `results/all-v2.summary.json`
  (由 `scripts/compare_runs.py v1 v2` 生成,含逐任务差异表)。
- ⚠ **EmoBench 口径**:报告中的 EmoBench 数字出自独立 harness `../EmoBench/eval/eval.mjs`(官方仓库提示词逐字复刻,
  可与论文 Table 1/2 对比)。套件内置 `emobench-ea/eu` 用的是简化提示词:EA 与官方版一致(71.5 vs 72.0),
  但 **EU 仅 39.3 vs 官方协议 57.8**——提示词差异对 EU 影响巨大,内置版数字不得与论文对比,仅作内部追踪。
- 独立重算对账:`python3 scripts/audit_results.py`(JSONL→指标重算,与 summary 逐项比对,输出 results/audit_recompute.json)。
- Kimi/DeepSeek 同题配对:`python3 scripts/paired_model_audit.py`(固化 id+gold 双键配对、碰撞剔除、429 披露;strict 与 keep-first 两口径)。
- ⚠ `--resume` 的 summary 只统计本次新跑的 items,不会自动合并旧记录;断点续跑后请用 `scripts/audit_results.py` 重算全量。
- ⚠ `.env` 含 API key,不得进入任何可分享包/提交范围。
- 数据许可与敏感性边界见 `../license_manifest.tsv`:本套件仅作研究评测;MentalManip/CBT-Bench 为 CC BY-NC(禁商用),
  PsySUICIDE 对营利组织需另签协议,EATD/IMHI 类含敏感文本不得外发,全部不用于产品训练。
