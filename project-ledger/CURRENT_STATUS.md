# 当前状态

更新时间：2026-08-01（America/Los_Angeles）

工作分支：`codex/cross-benchmark-accuracy-family-v1`

最高目标：分别提高有论文依据的各 benchmark 主指标，并用同切分、同协议、预注册证据证明；不是只优化 PsySUICIDE。

## 总判定

- 没有证据证明一个统一架构已经让所有 benchmark 同时提高。
- 当前有两项冻结 campaign 内的单任务提升：CPsyExam V4-Pro 相对 V4-Flash 同一 3,902 题 +1.4095pp；PsySUICIDE taxonomy 相对 Flash baseline 同一 official test accuracy +5.26pp、macro-F1 +0.0878。
- PsySUICIDE RoBERTa v1 是专用分类器证据；88.11% official test 与 93.43% internal holdout 不可相减。
- PsySUICIDE accuracy-first v2 已停止：A 完成，B 到 3,560/4,680，C/D 未开始；自动监控已删除。
- 接下来优先建立 IMHI、MDD、EATD、MentalManip 的“共享训练框架 + 每任务独立 head”，同时修复 CPsyExam/EmoBench/CBT 的协议问题。

## Benchmark 状态

| Benchmark | 最新可公开状态 | 下一步 |
|---|---|---|
| CPsyExam | Pro 84.7514%，Flash 83.3419%，旧冻结协议内 Pro +1.4095pp | 新 parser/profile 只在新 development 协议验证，不回改旧 Release |
| PsySUICIDE API | taxonomy 88.11% / 0.6371；Flash baseline 82.86% / 0.5494 | 保留同题配对提升结论，不与 internal holdout 混比 |
| PsySUICIDE RoBERTa v1 | valid 三 Seed均值 94.01% / 0.7380；internal holdout 93.43% / 0.7275 | 只称专用模型描述性泛化；极稀有类仍需真人数据 |
| PsySUICIDE v2 | A 93.29% / 0.6250；B 未完成；C/D 未开始 | 暂不续跑 |
| IMHI ×9 | 8/9 高于 ChatGPT zero-shot，1/9 高于 MentaLLaMA，0/9 高于 fine-tuned | 每任务独立监督 head，先 optimization/inner-dev |
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

详细证据见 [`../reports/model_architecture_status_and_roadmap_20260801.md`](../reports/model_architecture_status_and_roadmap_20260801.md)。
