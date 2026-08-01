# Project Ledger

这个目录是 `mental-health-llm-eval` 的长期项目账本，用来回答四个问题：

1. 最近到底做了什么；
2. 每个 benchmark 当前处于什么状态；
3. 哪些错误、失败实验和过度表述已经发现；
4. 下一步为什么这样排优先级。

## 固定文件

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)：只保留最新、可执行的状态。
- [`CHANGELOG.md`](CHANGELOG.md)：按时间追加的重要更新索引。
- [`ERROR_CASES.md`](ERROR_CASES.md)：公开安全的错误模式、合成复现和修复状态。
- [`updates/`](updates/)：每次重要代码、协议、实验或结论更新的详细记录。
- [`templates/UPDATE_TEMPLATE.md`](templates/UPDATE_TEMPLATE.md)：以后新增更新记录时复制的模板。

## 每次更新的强制规则

任何影响代码、数据切分、prompt、parser、模型、指标、报告、预算或发布状态的改动，都必须同时：

1. 在 `updates/YYYY-MM-DD-<slug>.md` 新增一条详细记录；
2. 在 `CHANGELOG.md` 加入链接和一句真实结论；
3. 若当前状态改变，更新 `CURRENT_STATUS.md`；
4. 若发现新的失败/错误模式，更新 `ERROR_CASES.md`；
5. 运行 `python3 scripts/check_project_ledger.py`；
6. 与代码一起 commit/push，不把“本地完成”写成“已部署”或“已发布”。

## 隐私边界

本目录只允许：聚合指标、公开代码位置、合成复现、commit/PR/Release、数据行数和 commitment。

本目录禁止：许可原始文本、题目/选项、用户对话、逐行 gold/prediction、原始模型输出、行 ID、private logits/diagnostics、API key、模型权重或可反推出敏感样本的细分。

“错误用例”默认使用合成文本，例如 `Answer: B`；真实数据只能记录聚合错误模式，例如“CA-MAQ accuracy 63.50%”，不能复制题目。
