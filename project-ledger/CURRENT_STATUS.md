# 当前状态

更新时间：2026-08-01（America/Los_Angeles）

工作分支：`codex/cross-benchmark-accuracy-family-v1`

公开审核入口：[Draft PR #5](https://github.com/AllenLi441/mental-health-llm-eval/pull/5)，依赖前序 PR #4。

最高目标：分别提高有论文依据的各 benchmark 主指标，并用同切分、同协议、预注册证据证明；不是只优化 PsySUICIDE。

## 总判定

- 没有证据证明一个统一架构已经让所有 benchmark 同时提高。
- 当前有两项冻结 campaign 内的单任务提升：CPsyExam V4-Pro 相对 V4-Flash 同一 3,902 题 +1.4095pp；PsySUICIDE taxonomy 相对 Flash baseline 同一 official test accuracy +5.26pp、macro-F1 +0.0878。
- PsySUICIDE RoBERTa v1 是专用分类器证据；88.11% official test 与 93.43% internal holdout 不可相减。
- PsySUICIDE accuracy-first v2 已停止：A 完成，B 到 3,560/4,680，C/D 未开始；自动监控已删除；本轮 B/C/D 明确保持 `PAUSED`。
- Benchmark registry 本地验证为 19/19，17/17 synthetic selftest 在 normal Python 与 `python -O` 均通过；generic trainer 与 IMHI screen 的 normal/`-O` selftest 也通过。终审新增的 live-registry proof、binary/multiclass 与 imbalance/invalid/empty-class 实执行断言、checkpoint epoch、MPS exact resume、失败 preflight manifest/strict resume-mode 回归均已通过；最终代码审查为 0 blocker。
- 0731 Flash + taxonomy 已完成同一 1,459 条 valid：accuracy `87.80%`、macro-F1 `0.5318`，但 `16/1,459` invalid、关键类 recall guard 与 macro-F1 配对推断门槛失败，因此 `REJECTED_BY_GATE`，不进入 test。
- IMHI DR/Irf/SAD/dreaddit 已完成显式 data-validation dry-run 和 12-job manifest；模型训练尚未启动，状态是等待 clean、committed execution base，不能报告 screen 分数。
- 后续仍优先建立 IMHI、MDD、EATD、MentalManip 的“共享训练框架 + 每任务独立 head”，同时修复 CPsyExam/EmoBench/CBT 的协议问题。

## Benchmark 状态

| Benchmark | 最新可公开状态 | 下一步 |
|---|---|---|
| CPsyExam | Pro 84.7514%，Flash 83.3419%，旧冻结协议内 Pro +1.4095pp | 新 parser/profile 只在新 development 协议验证，不回改旧 Release |
| PsySUICIDE API | 历史 official test：Pro+taxonomy 88.11% / 0.6371；本轮 valid：0731 Flash+taxonomy 87.80% / 0.5318，但候选被门槛拒绝 | 本轮不碰 test；保留历史 campaign 边界，不与 internal holdout 混比 |
| PsySUICIDE RoBERTa v1 | valid 三 Seed均值 94.01% / 0.7380；internal holdout 93.43% / 0.7275 | 只称专用模型描述性泛化；极稀有类仍需真人数据 |
| PsySUICIDE v2 | A 93.29% / 0.6250；B 未完成；C/D 未开始 | B/C/D 本轮保持 `PAUSED`，不恢复、不选 winner |
| IMHI ×9 | 历史 zero-shot 仍为 8/9、1/9、0/9 描述性点值；新四任务 trainer/data manifest 已验证，0 个训练 job 启动 | clean commit 后启动 DR/Irf/SAD/dreaddit 的 12-job Seed-42 screen |
| EmoBench | EA 74.25%、EU 71.50% 是 temp-0 proxy | 完整协议 16,000 calls；预算门槛约 `$9.5702` |
| MDD | accuracy 57.25%，macro-F1 0.4051 | 以 macro-F1 为主训练专用 head |
| EATD | accuracy 83.54%，depressed F1 0.48 | 同时保护阳性类 F1 与 macro-F1 |
| MentalManip | accuracy 77.80%，positive F1 0.8510 | 单独训练并校准二分类 head |
| CBT | 当前为 top-1 proxy | 先恢复论文兼容多标签指标 |

## 当前公开修改

- 新增跨 benchmark 总复盘和 GPT‑Pro 独立复核 prompt。
- 修复 CPsyExam prospective parser，增加两个显式 prompt profiles 和合成回归。
- 更正 EmoBench full call count/预算，并把 16,000-call 不变量写入自检。
- 新增本 `project-ledger/`，以后每次更新必须同步记录。
- 新增 19-task benchmark spec registry 与通用分类 trainer；registry、trainer/screen selftest 及 IMHI 数据 dry-run 已通过。
- 终审修复 shared trainer 的独立 registry 重验证、Transformers 5 MPS RNG 精确恢复、preflight 失败留痕与 `--resume` 模式门禁。
- 0731 Flash taxonomy full valid 已完成但被 invalid、关键类 recall 与 macro-F1 inference 门槛拒绝；IMHI 训练尚未启动。
- 本轮命令、聚合结果和失败记录持续追加在[跨 benchmark 提分执行台账](updates/2026-08-01-accuracy-family-execution.md)与对应 eval log。

详细证据见 [`../reports/model_architecture_status_and_roadmap_20260801.md`](../reports/model_architecture_status_and_roadmap_20260801.md)。
