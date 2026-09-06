# 2026-09-03：工作区状态与执行口径对齐

## 为什么更新

8 月底以后，模型准备、训练数据治理、量表、论文脚手架和 ESConv 干净重训产生了多批新产物，
但 `CURRENT_STATUS.md`、旧 `HANDOFF`、各阶段 `STATUS` 和实际工作区不再同步。继续把其中任一旧文件
当作当前事实，会重复已经完成的工作、混淆研究数据与商用数据，并把诊断分数误写成训练或部署门槛。

本记录冻结 **2026-09-03 工作区快照**。冲突时，以 `CURRENT_STATUS.md` 顶部和本记录为当前口径；
8 月历史记录不回写，只保留为当时证据。

## 修改范围

- 新增根目录 v3 HANDOFF，并把旧 handoff、周末总纲与阶段状态显式标为历史快照。
- 对齐 `论文和代码/` 的状态、许可说明、论文 Method 与 T1--T9 可再生表格源；补登记
  Kardia `chat_template.jinja` 这个 403 stub，使精选 manifest 38/38 自洽。
- 在仓库外 research-v0 bundle 中新增 final-adapter 选择 provenance 与 fail-closed 门，更新三份
  训练说明，并通过唯一 packager 完成第六次确定性重打。
- 不改冻结 500/50 内容，不调用付费 API，不下载完整权重，不上 GPU、不训练、不评分 frozen test，
  也不 push / commit。

## 权威工作区快照

### Git 与公开审核入口

- 当前分支：`codex/esconv-2021-faithful-reproduction`，本地 HEAD `e235dd0`。
- 相对其 tracking branch：本地 **ahead 91 / behind 0**。这些提交尚未推送，不能把本地状态写成
  远端已审核或已发布。
- 当前公开审核入口是 Draft PR #7；远端 head 仍停在 `b9cedae`。PR #5 属于历史跨 benchmark
  分支，不是当前分支的审核入口。

### 各工作流精确状态

| 工作流 | 2026-09-03 状态 | 尚未完成 / 不得外推 |
|---|---|---|
| Qwen3.8 pre-test | DashScope 的 ESConv fixed-250 已完成 250/250：Accuracy **13.60%**，0 missing / 0 invalid；六组两两比较 Holm 校正后均不显著 | 这是 development-sized 诊断，不是 full-test、微调后成绩或部署门槛；本次没有 Qwen3.8 CPCD 运行 |
| `research-v0` bundle | 冻结 train/dev **500/50**；第六次受控重打后内容 27/27、外层校验、确定性重建和归档↔磁盘 27/27 均通过；新增可核验 best-adapter 来源 | 数据含非商用来源，只允许 private noncommercial research / pipeline rehearsal；尚无 GPU run |
| B1 / A3 商用语料 | A3 已决定排除 NC、需另签协议和许可未知来源；候选池为中文 70,792 + 英文 11,775，70:30 约束可支撑 **39,250** 条 | 这是 source candidate pool，尚未完成统一规范化、最终冻结或训练就绪；不得与 `research-v0` 的 500/50 合并记账 |
| T0 | 27B 的 8 处硬编码已参数化；27B 是唯一 `VERIFIED` profile；Hub dry-run 已测得 32 文件、51.77 GiB，冷取门为 64.53 GiB；profile 与 adapter-selection 门合计 41/41 通过 | 14B / 7B 仍为 `UNVERIFIED_PLACEHOLDER`；真实 4-bit load 后的 LoRA 模块数、model/text config 三道门和 fast-path kernel 绑定仍需 GPU 实测 |
| T1.5 | 功效脚本、五维 rubric judge、agreement known-answer 验证和 crisis/style/rag 三套冻结 probes 已落盘 | 45-case 双判官 pilot、持久化区分度结果和真实 paired variance / ICC 未完成；`n=99` 来自 placeholder `sigma=1.2, ICC=0`，不是最终样本量 |
| T3 | 九张表、LaTeX 章节、TSV 转换器和构建脚手架结构完成；生成源已同步当前五维/判官/45-case/64-step 口径，9/9 表列宽一致 | T3/T4/T5/T6/T8 实证表、结果正文、摘要和结论仍等真实训练结果；本机无 `pdflatex`，未验证最终 PDF |
| EmoDynamiX clean | 干净特征 train/dev **8,433/2,985** 已验证；dev pilot 预注册与 CPU 修正案已冻结；CPU memcheck smoke 已完成 | smoke 标记 `DEVELOPMENTAL_SMOKE_NOT_SELECTABLE`，没有 selectable checkpoint；CE / class-balanced 两臂 pilot 尚未运行，frozen test 未评分 |
| 19-task benchmark 家族 | 仍以 2026-08-01 冻结状态为准，多数 `PAUSED` | 没有新的跨 benchmark 统一提升结论 |

## 指标口径对齐

- ESConv fixed-250 的 Qwen3.8 `13.60%` 只证明该 API/解析链路得到完整输出；不证明疗愈质量提升。
- Joint `32.22%` / EmoDynamiX `33.61%` 来自另一条监督分类工程轨，和生成式 fixed-250 的目标、
  训练方式及评测协议不同。尤其 `33.61%` **不再是 QLoRA 成功门槛**。
- 当前主要效果证据应来自预注册五维 rubric、输出侧 safety floor 与 RAG probe；但 T1.5 尚未完成
  双判官经验校准，因此现在也不能声称这套测量已经 paper-ready。
- CPCD 上既有 `−0.083`、95% CI `[−0.176, +0.010]` 是 Qwen3.6 与 DeepSeek 的内部描述性
  proxy；它不是 Qwen3.8 结果，也不是外部模型优劣结论。

## 训练数据边界

`research-v0` 与 A3/B1 是两件不同的交付物：

1. `research-v0` 的 500/50 用来证明 packaging、preflight、smoke 和训练闭环能走通；其许可边界为
   private noncommercial research。
2. A3/B1 的 39,250 是按 70:30 约束计算出的商用清洁**候选规模**；只有完成去重、规范化、
   许可复核、冻结和 manifest 后，才能称作训练集。

两者不得拼接、互相替代或共享“training-ready”结论。

## 当前测试基线

- `python3 -m unittest discover -s tests -p 'test_*.py' -v`：共 206，**194 pass / 7 fail /
  4 error / 1 skip**。已知不绿项分三组：ESConv strict-parser 测试仍假设旧严格行为；Responses API
  的测试字段仍读 `status` 而实现返回 `response_status`；CPCD fixtures 缺少新增 provenance / deployment
  字段。该基线不是全绿。
- README 离线矩阵：**27/29 pass**。两个 selftest 被 `论文和代码/` 内的 4 个未跟踪嵌套 Git
  仓库干扰，`git_identity()` 因目录不可唯一哈希而 fail closed。
- 静态完整性：Python AST 78/78、MJS syntax 30/30、shell syntax 4/4、JSON 306/306，均通过。
- Qwen3.8 addendum schema、250/250 行完整性、聚合块和源哈希均通过；但 addendum 声明
  fixed-250 且不含 CPCD，当前 CLI dry-run 却计划 ESConv 2,775/arm + CPCD 159/arm，共 5,868 次，
  仍是开放的计划口径失配。

## 工作区治理风险

`论文和代码/` 目前整体未跟踪，约 946 MiB；其中 191 个未被 ignore 的路径约 636.2 MiB，包含
原始语料类资产和 4 个嵌套 Git 仓库。它的 `MANIFEST.sha256` 只覆盖精选的 38 个文件，不能证明整个
目录已冻结或可发布；三份 Kardia 同哈希项是明确登记的 403 stub，不是有效配置。

在完成许可、隐私、嵌套仓库和发布边界审查前：

- 不得对仓库根执行无选择的 `git add .`；
- `论文和代码/` 应保持仓库外/显式忽略边界；
- 只有去敏、许可明确、体积受控的 manifest、聚合报告或脚本才能逐项纳入版本控制。

本次只记录风险，没有改 `.gitignore`，也没有把该目录任何内容加入 Git。

## 发现的文档漂移

- 旧 handoff 曾把已经完成的 Qwen3.8 provider pre-test 和 bundle 修复写成待办，并保留旧分支/PR
  入口；早期 T1.5 状态保留“2/4、blocked on T0”的历史计数。现已统一标为历史并指向 v3。
- 训练说明把 500 rows × 2 epochs ÷ effective batch 16 写成“约 62 steps”。按当前单卡 Trainer
  取整语义应为每 epoch `ceil(500/16)=32`，总计 **64 optimizer updates**。
- `save/eval_steps=20` 只形成 20/40/60 的定期候选；`checkpoint-60` 是最后一个定期 checkpoint，
  不是 step 64，也不必然是 best。研究配置启用 `load_best_model_at_end=True`，顶层 `final_adapter`
  应记录最小 `eval_loss` 对应的 checkpoint。

上述模式已新增到 `ERROR_CASES.md`；旧文件不在本次范围内，不静默回写历史。

## 验证

- `python3 scripts/check_project_ledger.py` 与 `git diff --check`：见本次最终核验。
- research bundle：41/41 本地门、内容合同、外层/内层哈希、连续确定性重建、28 个归档文件与
  bytecode/metadata 清洁检查均通过。
- 论文工作区：精选 manifest 38/38、agreement known-answer、T1--T9 9/9 表格定宽检查通过。
- 本记录只含聚合指标、文件级状态和合成/配置层错误，不含原始对话、逐行标签、密钥或权重。

## 下一步

1. 先收口公开/私有工作区边界，避免未跟踪大目录被误提交；再推送 91 个本地提交并刷新 PR #7。
2. 在计费训练前跑 T0 的 GPU-only 反向门、真实 4-bit smoke 与 adapter provenance 校验。
3. 完成 T1.5 双判官 pilot，以真实 sigma / ICC 重算样本量，并解决 addendum 与 CLI 计划失配。
4. 运行 EmoDynamiX 两臂 dev pilot；只有预注册选择规则通过后才产出 selectable checkpoint，仍不碰
   frozen test。
