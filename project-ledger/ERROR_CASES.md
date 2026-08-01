# 错误与失败用例账本

这里只记录公开安全的合成复现或聚合错误模式。状态含义：`OPEN`、`FIXED_PROSPECTIVE`、`HISTORICAL_LIMITATION`、`REJECTED_BY_GATE`、`INCOMPLETE`。

| ID | 类型 | 公开安全复现或证据 | 影响 | 状态 / 处理 |
|---|---|---|---|---|
| ERR-20260801-01 | 跨 split 误比 | 88.11% 来自 PsySUICIDE official test；93.43% 来自 frozen internal holdout | 错误写成 +5.32pp 会制造不存在的提升 | `FIXED_PROSPECTIVE`：报告明确禁止相减；未来只做同 split 对照 |
| ERR-20260801-02 | CPsyExam parser | 合成输出 `Answer: B` 被旧全局 `[A-E]` 扫描取到 `A` | 可产生“合法但错误”的预测；不能从公开聚合量化历史影响 | `HISTORICAL_LIMITATION`：旧 Release 不改；新 parser 合成回归要求输出 B |
| ERR-20260801-03 | EmoBench 计划计数 | EA/EU × 400 × 4 排列 × 5 次 = 16,000，不是 8,000 | 预算与完整运行门禁少算一半 | `FIXED_PROSPECTIVE`：自检锁定 16,000；预算更正约 `$9.5702` |
| ERR-20260801-04 | 未完成实验 | PsySUICIDE v2 A 完成；B 3,560/4,680；C/D 未开始 | 单臂不能选择赢家或声称 accuracy 提升 | `INCOMPLETE`：训练与自动监控已停止，保留 checkpoint |
| ERR-20260801-05 | 极稀有类失败 | v1 holdout “自杀准备行为”support=2、F1=0；v2 A 该类 support=2、F1=0 | 总体 accuracy 掩盖关键类失败 | `OPEN`：需要真人审核的稀有表达与硬负例；AI 不得冒充人工 gold |
| ERR-20260728-01 | Prompt 回退 | PsySUICIDE taxonomy-v2 valid accuracy -3.29pp、macro-F1 -0.0493 相对旧 taxonomy | 更复杂 prompt 没有自动提高模型 | `REJECTED_BY_GATE`：未进入 holdout |
| ERR-20260728-02 | 统一 prompt 回退 | IMHI contrastive-v4 same-10 九任务均值比 control 低 0.0702 | 统一 prompt 不能替代每任务训练 | `REJECTED_BY_GATE`：未启动完整运行 |
| ERR-20260801-06 | 指标不兼容 | CBT 当前 top-1 命中；论文报告多标签 F1 | 不能直接宣布超过或落后论文 | `OPEN`：先实现 paper-compatible multilabel scorer/head |

## 合成回归入口

- CPsyExam：`node scripts/check_cpsyexam_protocol.mjs`
- EmoBench：`node emobench-official/paper_protocol.mjs --selftest`
- 全任务 prompt/parser：`node run.mjs all --selftest`

发现真实逐行错误时，只把其抽象成不含原文、ID、gold/prediction 的类别边界或重新编写的合成 fixture，再放入本账本。
