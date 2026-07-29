# eval-suite — 8 数据集 · 19 任务 · 零样本准确率评测

零依赖 Node.js（≥18）。被测模型 = 任意 OpenAI 兼容 chat API（示例默认显式使用 DeepSeek `deepseek-v4-flash`；runner 会拒绝已经退役/含义不明确的 `deepseek-chat` 和 `deepseek-reasoner` 别名）。公开仓库包含代码与聚合结果；规划中的 CPsyExam V4 Release 仅允许发布去敏的逐行承诺与成对正确性结果。受许可/敏感性约束的原始数据、题目、选项、标签、模型预测和原始输出不随仓库或 Release 发布。

## 用法

```bash
cp .env.example .env                     # 填 API key 和 EVAL_DATASETS_DIR
node run.mjs list                        # 任务清单
node run.mjs all --selftest              # 仓库内 19/19 prompt+parser 自检；不读外部数据、不发请求
node run.mjs all --data-check            # 读取 EVAL_DATASETS_DIR，验证授权数据布局；不发请求
node run.mjs psysuicide --split valid --prompt-profile baseline --model deepseek-v4-pro --run-id valid-baseline-v1
node run.mjs cpsyexam --model deepseek-v4-pro --thinking enabled --reasoning-effort high --run-id full
node run.mjs cpsyexam --model deepseek-v4-flash --thinking disabled --run-id paired-control
node run.mjs all --split valid --run-id v1  # PsySUICIDE 用 valid，其余任务仍用各自固定数据；顺序跑
node run.mjs imhi-dr --model deepseek-v4-flash --run-id v1 --resume results/imhi-dr-baseline-deepseek-v4-flash-v1.jsonl
python3 scripts/audit_results.py --selftest  # 检查已发布聚合结果，不冒充逐行重算
python3 scripts/scoreboard.py --selftest
python3 scripts/check_baselines.py
python3 scripts/check_harness_safety.py
node scripts/check_psysuicide_protocol.mjs
node scripts/run_psysuicide_valid_matrix.mjs --selftest
python3 scripts/analyze_psysuicide_valid_matrix.py --selftest
node scripts/run_psysuicide_test_pair.mjs --selftest
python3 scripts/analyze_psysuicide_test_pair.py --selftest
node emobench-official/eval.mjs --selftest # 无官方数据也可验证 prompt/parser；有数据时再验 400+400
node emobench-official/paper_protocol.mjs --selftest
node scripts/prepare_psysuicide_v2_partition.mjs --selftest
node scripts/run_psysuicide_v2_valid_matrix.mjs --selftest
python3 scripts/analyze_psysuicide_v2_valid.py --selftest
python3 scripts/train_psysuicide_roberta.py --selftest
python3 scripts/report_psysuicide_roberta.py --selftest
python3 scripts/score_psysuicide_roberta_holdout.py --selftest
node scripts/run_imhi_uniform_v4_matrix.mjs --selftest
```

通用协议：零样本、temperature 0、严格标签解析（无同义词映射，解析失败记 invalid 并算错）、
抽样用种子 42 确定性洗牌（`--seed` 可换）、逐条落盘 JSONL 可续跑、汇总含 accuracy + Wilson 95% CI +
weighted/macro-F1 + 逐类精度 + 混淆对。

DeepSeek V4 的思考配置必须显式传入：`--thinking enabled|disabled`；启用时可用
`--reasoning-effort high|max`。401/402 会立即中止且不会把该题写成“已完成”。逐行结果记录
requested/response model、provider、fingerprint、usage、UTC 时间、case/prompt/dataset 哈希与 run ID。

账户隔离：批量运行只读取专用的 `EVAL_API_KEY`，不会回退读取应用或 shell 中的
`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`。请为评测设置独立预算、告警和可撤销 key，避免跑分耗尽生产额度。

## PsySUICIDE 模型优化协议

PsySUICIDE 是本套件第一个强制切分隔离的优化模块。第一轮提示词开发只允许跑官方 `valid`；
官方 `test` 只能在验证集选出唯一配置后运行一次。第二轮监督开发在改动模型前把官方 `train`
按类确定性冻结为 9,342 条 optimization 与 2,329 条内部 holdout；optimization 可用于训练或
检索，但不作为模型选择分数，holdout 在预注册 valid 门槛通过前禁止读取或评分。
五种当前提示方案为：

- `baseline`：保留原始直接 11 类提示，作为冻结对照。
- `taxonomy`：加入 11 类工作定义，重点区分被动/主动/探索、计划/准备/未遂以及意图/行为。
- `hierarchical`：在 taxonomy 之上要求模型内部按类别族和细粒度层级判断，但仍只输出标签。
- `taxonomy-v2`：加入更细的相邻类别决策边界；完整 valid 结果证明它发生回退，已拒绝。
- `fewshot-balanced`：只从 optimization partition 检索每类示例；same-50 smoke 后因效果与成本
  门槛未通过而停止，没有进入完整 valid。

验证集由一个默认 dry-run 的编排器执行。下面第一条只打印 4 个 smoke 臂、调用量和保守预算，
不会调用 API：

```bash
node scripts/run_psysuicide_valid_matrix.mjs \
  --phase smoke --batch-id psyvalid-v1
```

实际执行必须同时提供 `--execute` 和批准预算。先跑四臂各 50 条；编排器会核对 response model、
错误、split、profile、thinking、run ID 和 usage 覆盖。每个臂结束后还会按固化价格快照计算
实际 token 费用；若已花费用加剩余臂预留超过批准预算，会在下一个臂开始前停止：

```bash
node scripts/run_psysuicide_valid_matrix.mjs \
  --phase smoke --batch-id psyvalid-v1 \
  --approved-budget-usd 1 --execute
```

人工看过 smoke 的 usage、invalid 和成本后才能启动四臂完整 1,459 条。full 会先读取并验收
同一 smoke batch 的四份 summary；任何一份缺失或模型路由错误都会拒跑：

```bash
node scripts/run_psysuicide_valid_matrix.mjs \
  --phase full --batch-id psyvalid-v1 --smoke-batch-id psyvalid-v1 \
  --approved-budget-usd 15 --execute

python3 scripts/analyze_psysuicide_valid_matrix.py \
  --batch-id psyvalid-v1 \
  --out reports/psyvalid-v1-valid-analysis.json
```

分析器强制同一完整 ID/gold/case-hash 集，输出四臂 macro-F1、weighted-F1、逐类指标、usage、
成本、三项预指定配对比较、exact McNemar、paired bootstrap 和 Holm 校正，并按冻结规则选出
唯一候选。输出仍是 valid 开发性结论，不允许写成 test 胜出或临床有效。

监督 RoBERTa v1 已完成三随机种子 official-valid 选优：accuracy 均值 `94.01%`，
macro-F1 `0.7380 ± 0.0386`，weighted-F1 `0.9404 ± 0.0022`（均值 ± sample SD）。
它通过预注册的 macro-F1 与逐类回退门槛，固定选择 Seed 43（valid macro-F1 `0.7616`）；
权重及 config/tokenizer 全部以 SHA-256 冻结在
`reports/psysuicide-roberta-v1-valid-selection.json` 与
`reports/psysuicide-roberta-v1-holdout-freeze.json`。在冻结资产提交并推送前，2,329 条
内部 holdout 仍未加载。holdout scorer 默认只做无数据 dry-run，正式评分需独占持久 claim；
结果只能作一次内部描述性确认，不代替论文 test，也没有预注册配对显著性检验。

审核并提交 valid 聚合分析后，生成一次冻结的成对 test campaign。它固定两个臂：
`A / V4-Flash / baseline` 作为当前控制，以及 valid 选出的唯一赢家作为候选。prepare 会为
两臂各生成一个完整 test 预注册，并把 valid 分析哈希、两个预注册哈希和主检验规则写入 campaign；
不会调用 API：

```bash
node scripts/run_psysuicide_test_pair.mjs \
  --mode prepare \
  --campaign-id psytest-v1 \
  --valid-analysis reports/psyvalid-v1-valid-analysis.json \
  --campaign reports/psytest-v1.campaign.json
```

提交 campaign 与两个 preregistration 后，先用 dry-run 查看精确两臂和预算；最昂贵的
hierarchical 赢家估算预留约 `$6.44`，因此示例批准 `$7`。实际执行必须显式确认 test pair：

```bash
node scripts/run_psysuicide_test_pair.mjs \
  --mode execute --campaign reports/psytest-v1.campaign.json

node scripts/run_psysuicide_test_pair.mjs \
  --mode execute --campaign reports/psytest-v1.campaign.json \
  --approved-budget-usd 7 --confirm-test-pair --execute

python3 scripts/analyze_psysuicide_test_pair.py \
  --campaign reports/psytest-v1.campaign.json \
  --out results/psytest-v1-confirmatory-analysis.json
```

控制和候选各在 1,464 条 test 上运行一次；若中断，只能 resume 同一冻结文件。主检验是
macro-F1 的成对随机化检验与 paired bootstrap CI；accuracy 的 exact McNemar 是次要指标。
没有两臂完整逐题结果就不能宣称显著提升。离线验收定义见
`.claude/evals/psysuicide-model-optimization.md`。

## PsySUICIDE 模型优化确认性结果（2026-07-28）

本次优化严格使用官方 `valid` 选择配置，再在任何 test 预测产生前提交一次冻结的双臂
campaign 和两份预注册。验证集从四个固定臂中选出 `DeepSeek V4-Pro + taxonomy`；确认性
test 将它与 `DeepSeek V4-Flash + baseline` 在同一批 1,464 个保留单标签样本上逐题配对：

| 冻结 test 臂 | Accuracy | Macro-F1 | Weighted-F1 | Invalid / Error |
|---|---:|---:|---:|---:|
| V4-Flash + baseline（reference） | 82.86% | 0.5494 | 0.8426 | 0 / 0 |
| V4-Pro + taxonomy（candidate） | **88.11%** | **0.6371** | **0.8796** | 0 / 0 |

预注册主检验的 macro-F1 差值为 `+0.0878`；20,000 次 paired randomization 的双侧
`p=0.04830`，20,000 次 paired bootstrap 的 95% CI 为 `[+0.00993, +0.16151]`。
次要 accuracy 差值为 `+5.26` 个百分点，exact McNemar `p≈6.48×10⁻⁹`。因此可以限定地
表述：**candidate 在这一次冻结的 PsySUICIDE 官方 test campaign 上，按预注册 macro-F1
标准显著优于 reference。**这不等于临床有效、跨数据集普遍优越或等价性结论。

可复核的聚合证据见
[`valid` 选优分析](reports/psyvalid-20260728-valid-analysis.json)、
[冻结 campaign](reports/psytest-20260728.campaign.json)和
[确认性分析](reports/psytest-20260728-confirmatory-analysis.json)。原始逐行 licensed
结果、咨询文本、ID、逐题 gold/prediction、原始模型输出和密钥均不公开。

## CPsyExam V4 全量确认性结果（2026-07-22）

本次比较在[不可变预注册 Release](https://github.com/AllenLi441/mental-health-llm-eval/releases/tag/cpsyexam-v4-prereg-2026-07-22)之后运行，并对同一批 3,902 道题进行严格配对：

- `deepseek-v4-pro`：3,307/3,902，**84.7514%**，invalid 17，API error 0；响应 fingerprint 为 `fp_9954b31ca7_prod0820_fp8_kvcache_20260402`。
- `deepseek-v4-flash`：3,252/3,902，**83.3419%**，invalid 0，API error 0；响应 fingerprint 为 `fp_8b330d02d0_prod0820_fp8_kvcache_20260402`。
- 配对列联：两者都对 3,088，只有 v4-pro 对 219，只有 v4-flash 对 164，两者都错 431；差值为 **+1.4095pp**。
- 预注册 exact McNemar 双侧检验 `p=0.00572255`；配对 normal 95% CI `[0.4274, 2.3917]pp`，20,000 次 bootstrap 95% CI `[0.4357, 2.4090]pp`。因此，本协议内的结论是 **v4-pro 显著优于 v4-flash**。
- ±2pp TOST 的 `p_lower=5.086e-12`、`p_upper=0.119332`，未同时通过；**未证明等价**，不得写成“统计平手”或“等价”。

公开聚合文件为 `results-summary/cpsyexam-v4-full-paired.summary.json`。完整去敏资产已发布到 [`cpsyexam-v4-full-2026-07-22`](https://github.com/AllenLi441/mental-health-llm-eval/releases/tag/cpsyexam-v4-full-2026-07-22)：GitHub 报告 `isImmutable=true`，Release attestation 与 6/6 个本地资产摘要均已通过 `gh release verify` / `verify-asset`。

复现定位：runner commit `3e6890374cb39631bb1cc8bca46ef4835df85446`；公开 case commitment manifest SHA-256 `1275ce7edeb55ad62500ac1692b82bef3800592decc2ece4d615fc8770232c9d`；数据 revision 见公开 summary 的 `dataset.revision`。

## 任务与对比基准

| 任务 | 数据 | 量（默认抽样） | 指标 | 已发表参照 |
|---|---|---|---|---|
| emobench-ea | EmoBench EA | 400（全量） | acc（分中英） | GPT-4: 75.50 en / 73.75 zh |
| emobench-eu | EmoBench EU | 400（全量） | acc，情绪+原因全对 | GPT-4: 59.75 en / 54.12 zh |
| mdd5k-diagnosis | MDD-5k | 925（抽 400） | acc（5 类，ICD 归并，**本模块自定协议**） | 无统一榜单，横向比模型用 |
| psysuicide | PsySUICIDE train/valid/test | 11,671 / 1,459 / 1,464（valid/test 全量） | macro-F1 主指标；weighted-F1 / acc 次指标（11 类） | 论文微调基线见 `reports/BASELINES.json`；只在冻结配置后比较 test |
| cbt-cd / pc / fc | CBT-Bench | 146/184/112（全量） | top-1 命中率（金标多标签） | 论文用 multi-label F1，口径不同 |
| mentalmanip | MentalManip con | 2,915（抽 500） | acc / F1 | 多数类 69.2%；GPT-4 基线见 PDF |
| imhi-dr 等 9 个 | IMHI test | 405~10,861（各抽 500） | weighted-F1 / acc | 原论文有 10 个 test sets；本 harness 未含 CLP，只能报告 9/10 子集 |
| cpsyexam | CPsyExam test | 3,902（全量） | acc（KG/CA × 单选/多选分组） | GPT-4 零样本 76.56/10.76/60.33/13.00（67.43 为论文含少样本均值,严格零样本加权 57.6） |
| eatd-depression | EATD validation | 79（全量） | F1(抑郁) / acc | 论文文本 BiLSTM F1 0.65，融合 0.71 |

注意口径：我们是**零样本 LLM**，与「微调」参照比较时要注明设定差异；MDD-5k 和 CBT-Bench
的协议细节（ICD 归并 / top-1 命中）是本套件定义的，报告时须写明。

## 结构

```
mental-health-llm-eval/
├── lib.mjs        共享：配置/.env、CSV 解析、API 重试、并发池、指标、任务运行器
├── lib/baselines.mjs  从 reports/BASELINES.json 读取对照值
├── run.mjs        入口（list / all / 单任务）
├── tasks/         emobench mdd5k psysuicide cbtbench mentalmanip imhi cpsyexam eatd
├── results-summary/  可公开的聚合结果
└── results/       本地运行后生成的逐行结果（不发布）
```

## 数据目录

真实评测需要从各数据集官方来源取得许可并放在同一个根目录，再把 `EVAL_DATASETS_DIR` 指向该目录。任务读取的顶层目录为：`EmoBench`、`MDD-5k`、`PsySUICIDE`、`CBT-Bench`、`MentalManip`、`MentaLLaMA`、`CPsyExam`、`EATD`。公开仓库的 `--selftest` 使用合成微型 fixture，只证明入口、prompt 和严格解析器可运行，不声称重算论文指标。

官方 EmoBench 复刻脚本单独读取 `EMOBENCH_DATA_DIR/{EA,EU}.jsonl`。未提供授权数据时，它的
`--selftest` 同样只运行仓库内合成 fixture；提供数据后会额外强制校验 EA/EU 各 400 条、中英各 200。

## 审计与复现口径(2026-07-07 固化)

- **v1 主跑分 = 17 个非 EmoBench 任务**(`run-v1.log`,10,142 次调用);**v2(2026-07-07)= 全部 19 任务重跑**,
  17 主任务中 16 项与 v1 精确一致、cbt-fc -0.89pp(1 题输出波动),公开 overview 固化于 `results-summary/all-v2.summary.json`
  (由 `scripts/compare_runs.py v1 v2` 生成,含逐任务差异表)。
- ⚠ **EmoBench 口径**:当前 scoreboard 的 EmoBench 数字仍出自独立 temp-0 单次 proxy，不能称完全同协议。`emobench-official/paper_protocol.mjs` 已实现论文所述的每题 5 次采样多数票 × 4 个选项排列取均值，并完成 160-call 完整性 smoke；8,000-call 全量在预算门禁处停止，因此没有用 smoke 数字替换 benchmark。套件内置 `emobench-ea/eu` 用的是简化提示词:EA 与独立 proxy 接近(71.5 vs 72.0),
  但 **EU 仅 39.3 vs 官方协议 57.8**——提示词差异对 EU 影响巨大,内置版数字不得与论文对比,仅作内部追踪。
- 独立重算对账：在持有授权逐行 JSONL 的本地环境运行 `python3 scripts/audit_results.py`；公开包没有原始行，`--selftest` 只验证聚合文件结构并明确标注边界。
- 授权环境可运行 `python3 scripts/audit_results.py --results-dir /authorized/results --manifest-out /review/authorized-run.manifest.json --audit-out /review/audit-recompute.json` 生成不含文本、输出和行 ID 的证据清单（文件 SHA-256、行数/唯一数/重复数、错误数、字段覆盖、模型/供应商聚合）。它能暴露续跑碰撞和 provenance 缺字段，但**不能替代获许可的逐行结果发布**。
- Kimi/DeepSeek 同题配对:`python3 scripts/paired_model_audit.py`(固化 id+gold 双键配对、碰撞剔除、429 披露;strict 与 keep-first 两口径)。
- CPsyExam V4 全量确认性比较以 `reports/cpsyexam_v4_full_preregistration.md` 为预注册口径；`scripts/cpsyexam_paired_inference.py` 固化 exact McNemar、paired CI、±2pp TOST 与去敏逐行 Release builder。2026-07-22 的 3,902×2 配对结果见 `results-summary/cpsyexam-v4-full-paired.summary.json`；旧 `n=599` 仅是历史 pilot，不进入确认性分析。
- `--resume` 会把既有 JSONL 行与本次新行合并后重建 summary；旧版本留下的中断运行必须标为 `incomplete_archived` 或由授权原始行重建，不能把半跑 summary 当完整结果。
- ⚠ `.env` 含 API key,不得进入任何可分享包/提交范围。
- 数据许可与敏感性边界见 `PUBLISHING_NOTE.md`：本套件仅作研究评测；各数据集许可必须分别遵守，原始敏感文本和逐行模型输出不随本仓库发布，也不用于产品训练。
