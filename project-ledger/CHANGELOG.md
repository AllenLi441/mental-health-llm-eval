# 更新索引

本文件按时间倒序追加；每条必须链接到 `updates/` 的详细记录。

## 2026-08-01

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
