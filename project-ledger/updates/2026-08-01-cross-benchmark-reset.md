# 2026-08-01：跨 benchmark 目标重置、协议修正与公开同步

## 为什么更新

用户明确指出：项目目标是提高多个有论文依据的数据集上的模型架构表现，但最近计算几乎全部集中在 PsySUICIDE 专用分类器；并且 88.11% official test 与 93.43% internal holdout 曾被并排展示，容易被误解成 +5.32pp。

## 这段时间实际做了什么

1. 2026-07-22：完成 CPsyExam 3,902 题 V4-Pro/V4-Flash 全量配对、预注册和不可变 Release。
2. 2026-07-27 至 07-28：对 PsySUICIDE 做 prompt/模型同题比较，taxonomy 候选相对 Flash 在冻结 campaign 上提高。
3. 2026-07-28：建立固定 scoreboard、EmoBench 5×4 执行器、IMHI 统一候选与监督 RoBERTa 路线；负候选按门槛停止。
4. 2026-07-28 至 07-29：PsySUICIDE RoBERTa v1 跑三 Seed并一次性评分 internal holdout。
5. 2026-07-31 至 08-01：冻结 PsySUICIDE accuracy-first v2 四臂；A 完成，B 在 3,560/4,680 停止，C/D 未开始。
6. 2026-08-01：停止训练和自动监控，把最高优先级改回跨 benchmark 模型家族。

## 本次代码与文档修改

- `tasks/cpsyexam.mjs`
  - 保留 `subject_name`；
  - 增加 `legacy-zero-shot-v1` 与 `subject-json-v1` 候选；
  - 新 parser 只接受严格 JSON、完整锚定答案或纯字母；
  - 不再把 `Answer` 的 A 当答案。
- `scripts/check_cpsyexam_protocol.mjs`
  - 用纯合成 fixture 覆盖 `Answer: B → B`、多选排序、JSON、模糊解释拒绝。
- `emobench-official/paper_protocol.mjs`
  - 锁定 smoke=160、full all/all=16,000 的调用量不变量。
- `reports/emobench-paper-protocol-smoke-20260728.json`
  - 保留 smoke 观测，加入 8,000→16,000 与 `$4.7851`→`$9.5702` 的可审计更正。
- `reports/model_architecture_status_and_roadmap_20260801.md`
  - 按“已提升 / 描述性 / 不可比 / 未完成”重写总状态与路线图。
- `reports/chatgpt_pro_5_6_independent_review_prompt_20260801.md`
  - 给 GPT‑Pro 的独立复核 prompt；要求引用公开证据并禁止跨 split 过度声明。
- `project-ledger/`
  - 建立永久更新、现状与错误账本。

## 验证

- 全部 19/19 task 的 prompt/parser selftest 通过；CPsyExam 两个 profile 均通过。
- CPsyExam 合成协议回归通过。
- EmoBench proxy 与 5×4 protocol selftest 通过；full count 固定为 16,000。
- 76 个公开 summary 的 aggregate audit、scoreboard、baseline registry 与 harness safety 通过。
- PsySUICIDE/IMHI 的 runner、analyzer、trainer、reporter、scorer 离线 selftest 通过。
- 未读取许可原始数据，未调用付费 API，未访问 frozen holdout。

## 结论

这次没有产生新的论文 test 分数，也不声称跨 benchmark accuracy 已提升。它完成的是停止错误优先级、修正两个协议缺陷、整理真实证据，并为后续“共享框架 + 每任务专用 head/adapter”建立公开起点。

## GitHub 状态

- 分支：`codex/cross-benchmark-accuracy-family-v1`
- Draft PR：[AllenLi441/mental-health-llm-eval#5](https://github.com/AllenLi441/mental-health-llm-eval/pull/5)
- 依赖：PR #4；未直接合并 `main`
- 发布边界：仅代码、协议、aggregate、commitment 和公开文档；无 raw/secrets/weights/private diagnostics

## 下一步

1. 先对有合法训练标签的 IMHI 任务建立独立 spec/head/checkpoint；只用 train/inner-dev 筛选。
2. 复用训练框架到 MDD、EATD、MentalManip，但每个任务独立预注册和主指标。
3. CPsyExam 在新 development split 验证 parser/profile/检索，不能继续用已看过的 3,902 test 调参。
4. EmoBench 完整协议需 16,000 calls 和独立批准预算；CBT 先修多标签协议。
5. 最后再评估是否值得续跑 PsySUICIDE v2 的 B/C/D。
