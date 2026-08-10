# 更新索引

本文件按时间倒序追加；每条必须链接到 `updates/` 的详细记录。

## 2026-08-01

- [`冻结 Benchmark/DeepSeek 历史基线并建立 MentalHealth-Instruct v1`](updates/2026-08-01-benchmark-freeze-instruct-v1.md)：冻结 8 families / 19 tasks 与 19 条 aggregate baseline evidence；明确当前不是统一 paper-ready control；建立 0-row、真人盲审、专家裁决、仓库外数据骨架，并记录 IMHI post-training checkpoint compatibility 失败。
- [`终审证据闭环与全仓回归`](updates/2026-08-01-accuracy-family-execution.md)：补齐 multiclass/imbalance/invalid/empty-class/deterministic-metric 实执行断言与 checkpoint epoch 记录；全仓回归和隐私扫描通过，终审为 0 blocker。
- [`终审执行门禁加固`](updates/2026-08-01-accuracy-family-execution.md)：shared trainer 现独立验证 live registry proof，MPS checkpoint 严格保存/恢复 RNG，preflight 失败生成去敏审计 manifest，`--resume` 模式误用 fail closed；normal/`-O` 回归通过，IMHI 仍未启动。
- [`提分执行结果：基础设施通过、Flash taxonomy 拒绝、IMHI 待启动`](updates/2026-08-01-accuracy-family-execution.md)：registry 19/19 与 trainer/screen 离线门禁通过；0731 Flash taxonomy 虽提高 valid 点值，但因 invalid、关键类 recall 和 macro-F1 inference 失败被拒绝；IMHI 仅完成 12-job data-validation dry-run，尚未训练。
- [`跨 benchmark 提分执行台账`](updates/2026-08-01-accuracy-family-execution.md)：冻结 registry、generic trainer、0731 Flash taxonomy valid 与 IMHI 四任务单 Seed screen 的执行顺序、验收门槛和追加式命令/错误记录；PsySUICIDE v2 B/C/D 继续暂停。
- [`跨 benchmark 目标重置、协议修正与公开同步`](updates/2026-08-01-cross-benchmark-reset.md)：停止未完成的 PsySUICIDE v2，承认跨 split 不可比；修正 CPsyExam parser 与 EmoBench 16,000-call 门禁；建立长期项目账本。

## 2026-07-31

- PsySUICIDE accuracy-first v2 冻结 optimization-only 7,479/1,863 inner split、四臂和 10 epochs 协议；后续 A 完成、B 中断，未形成四臂选择结论。历史报告：[`../reports/model_optimization_accuracy_first_v2_20260731.md`](../reports/model_optimization_accuracy_first_v2_20260731.md)。

## 2026-07-28 至 2026-07-29

- PsySUICIDE taxonomy-v2 在完整 valid 退步并被拒绝；IMHI contrastive-v4 same-10 统一候选退步并被拒绝。
- PsySUICIDE RoBERTa v1 完成三 Seed、checkpoint 冻结和一次性 internal holdout；它只适用于 PsySUICIDE。
- EmoBench 建立 5×4 论文兼容执行器并跑 160-call 完整性 smoke；当时全量调用数/预算被少算一半，已在 2026-08-01 更正。
- 汇总报告：[`../reports/model_optimization_v2_20260728.md`](../reports/model_optimization_v2_20260728.md)。

## 2026-07-27 至 2026-07-28

- PsySUICIDE V4-Pro taxonomy 与 V4-Flash baseline 完成同一 official test 配对确认；candidate 在该冻结 campaign 上提高 accuracy 5.26pp、macro-F1 0.0878。该结论不能与 RoBERTa internal holdout 混合。

## 2026-07-22

- CPsyExam 从 599 pilot 进入 3,902 全量 V4-Pro/V4-Flash 配对；发布预注册与不可变去敏 Release。旧 parser 风险在 2026-08-01 才被发现，旧 Release 不回写。
