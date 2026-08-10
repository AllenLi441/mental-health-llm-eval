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
| ERR-20260801-07 | 单看 accuracy 晋级 | 0731 Flash+taxonomy valid 相对同 fingerprint baseline accuracy `+0.054832`、macro-F1 `+0.051654`，但 invalid=`16/1,459`、关键类 recall guard 失败，macro randomization `p=0.06360` 且 bootstrap CI `[-0.00619,0.10574]` | 正向点估计不能覆盖 parser 完整性、关键类退步或不确定性失败 | `REJECTED_BY_GATE`：候选不进入 test；保留完整 valid 负结论 |
| ERR-20260801-08 | 把 screen 写成最终结果 | 合成场景：某一 loss 在单 Seed inner-dev 高于 CE control | 若据此声称 test/论文提升或在 test 选 arm，会产生选择偏差 | `OPEN`：单 Seed 只筛选；通过家族门槛后另行预注册三 Seed confirmation |
| ERR-20260801-09 | Transformers 5 参数兼容 | generic trainer 初始集成向 `TrainingArguments` 传入不受当前 Transformers 5 接受的 `save_safetensors`，触发 `TypeError` | 训练会在首个 job 创建前失败 | `FIXED_PROSPECTIVE`：移除不兼容参数，保留严格 checkpoint artifact 校验；normal/`-O` selftest 均通过 |
| ERR-20260801-10 | 跨 split leakage key 过窄 | 合成复现：相同 `post` 配不同 `question` 时，按完整拼接输入分组会漏掉跨 split 重复 post | 同一原帖可能同时进入 train 与 valid，造成信息泄漏 | `FIXED_PROSPECTIVE`：跨 split key 固定为 post-only；DR 与 dreaddit 各移除 1 条重叠 train row，valid 保持完整 |
| ERR-20260801-11 | Trainer registry trust boundary | 终审合成复现：shared `execute_training` 原先只要求 adapter 传入非空 registry proof，替代 caller 可伪造 bundle | 真实执行可能没有重新证明 live registry、task/profile/config/protocol 与 data/plan identity 一致 | `FIXED_PROSPECTIVE`：shared execute 独立重跑 checker/hash，并核对 task、profile、config、protocol、path、rows、split SHA 与 plan；伪造 bundle 回归被拒绝 |
| ERR-20260801-12 | MPS resume RNG 缺失 | Transformers 5 的 checkpoint 不保存 MPS RNG state | MPS 断点续跑无法保证与不中断执行逐步一致，且损坏的更高 step 可能遮住可恢复 checkpoint | `FIXED_PROSPECTIVE`：严格保存/恢复 `mps_rng_state.pth`；实际 MPS random-tensor roundtrip 一致；device-aware latest checkpoint 跳过缺失/损坏 RNG 的高 step |
| ERR-20260801-13 | Preflight 失败无审计 | `--execute` preflight 失败原先不留 manifest；`--resume` 未配 `--execute` 可静默落入 dry-run | 失败尝试无法聚合审计，CLI 误用可能被误解为已恢复 | `FIXED_PROSPECTIVE`：先写 `ATTEMPTED`，失败更新为 aggregate-only `FAILED/PREFLIGHT` 并 exit 1；不持久化 raw error message；resume misuse 同样 exit 1 |
| ERR-20260801-14 | Capability selftest 证据缺口 | 初版 checklist 勾选了 multiclass、imbalance、invalid/empty-class 与 deterministic metrics，但 selftest 尚未逐项执行这些断言 | 文档 PASS 可能强于实际测试证据 | `FIXED_PROSPECTIVE`：新增 binary/three-class 三 loss、balanced weights、invalid/empty-class rejection 与 repeated canonical metrics 回归；normal/`-O` 均实际通过 |
| ERR-20260801-15 | Checkpoint epoch 缺失 | 初版 result 只保存 best step/checkpoint，没有显式 chosen epoch | 无法完整满足 IMHI aggregate reporting contract | `FIXED_PROSPECTIVE`：严格交叉核对 checkpoint 名、best global step 与唯一 logged epoch，并保存 `best_checkpoint_epoch` 及推导规则 |
| ERR-20260801-16 | Transformers 5 checkpoint LayerNorm 键名兼容 | 首个 IMHI DR/CE job 在 epoch 4 / step 504 early-stop 后加载 best checkpoint-252；checkpoint 暴露 50 个 `gamma/beta` 键，live model 期望对应 `weight/bias`，其余共同参数 shape/dtype 一致；去敏 error SHA 与 manifest 相符 | 过严 literal-key 校验把可兼容 checkpoint 判为失败；manifest 又误标为 `PROTOCOL_PREFLIGHT`，可能把 post-training best-load 失败误写成数据/预检失败 | `OPEN / PAUSED`：0 complete / 1 failed / 11 planned；诊断 weighted-F1 `0.9085` 不计正式分数；当前不重启，先修复兼容映射、阶段分类和合成回归 |

## 合成回归入口

- CPsyExam：`node scripts/check_cpsyexam_protocol.mjs`
- EmoBench：`node emobench-official/paper_protocol.mjs --selftest`
- 全任务 prompt/parser：`node run.mjs all --selftest`

发现真实逐行错误时，只把其抽象成不含原文、ID、gold/prediction 的类别边界或重新编写的合成 fixture，再放入本账本。
