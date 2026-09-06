# 交接书 v3：Qwen3.8-27B 疗愈模型（2026-09-03 对齐版）

状态权威日期：2026-09-03  
适用范围：`mental-health-llm-eval`、research-v0 训练包、`论文和代码/` 工作区  
替代文档：`HANDOFF-qwen-therapy-model-20260828.md`（仅保留为历史快照）

## 0. 一句话状态

**研究冒烟包已完成本机静态准备，但从未上 GPU；商用清洁语料池已完成来源决策，却还没有物化为正式训练集；评测工具已大体实现，但 45-case 双判官 pilot、实测功效参数、正式训练、GRPO 与部署验证都尚未完成。**

这三件事必须分开：

1. `mental-health-instruct-research-v0` 是 500/50 的**研究能力与流水线演练包**，含 NC / research-only 来源，不能训练产品可部署 adapter。
2. `论文和代码/` 中的 B1 是按 A3 政策筛出的**商用候选池**；目前只有审计与筛选结果，`corpus/normalized/`、`corpus/frozen/` 和 `weights/` 仍为空。
3. ESConv 八类 ACC 是诊断指标，不是疗愈回复质量或部署门；33.61% 来自另一模型、另一协议的监督分类实验，不能拿来要求 QLoRA 开放回复模型。

## 1. 当前仓库与发布边界

| 项目 | 2026-09-03 当前事实 |
|---|---|
| 本地分支 | `codex/esconv-2021-faithful-reproduction` |
| 本地 HEAD | `e235dd0ce43c740d012a23c47d9371d8dc36550e` |
| upstream | 同名 origin 分支，远端仍在 `b9cedae823aaf42a3f49790d0e4d8bfc5f8fc8da`；本地领先 91 commits |
| PR | Draft PR #7 对应该分支，但远端内容落后本地；旧文档所写 PR #5 不是当前入口 |
| 工作区 | 有既有修改与大量未跟踪研究资产；`论文和代码/` 整体约 946 MiB，含原始语料和 4 个嵌套 Git 仓库 |
| 发布红线 | 不执行 `git add .`，不把 raw 数据、模型副本、本机路径或嵌套仓库整体提交；提交前按 `PUBLISHING_NOTE.md` 做白名单选择 |

`论文和代码/` 的实际路径是：
`/Users/allenli/Desktop/mental-health-llm-eval/论文和代码/`。
旧交接中的 `~/Desktop/Esconv 改进论文和代码/` 与部分任务文档中的
`~/Desktop/论文和代码/` 都不是当前路径。

## 2. 已完成且可以依赖的事实

### 2.1 Qwen3.8 未微调基线已实跑

- 冻结 addendum：`open_response_eval/preregistration_dashscope_qwen38_esconv250_addendum.json`
- 提供方与模型：Alibaba DashScope `qwen3.8-27b`，`enable_thinking=false`
- ESConv fixed-250：ACC **13.60%**，cluster-bootstrap 95% CI **[9.20%, 18.62%]**，macro-F1 **9.60%**，weighted-F1 **11.82%**，missing/invalid **0/0**。
- 与 Qwen3.6、DeepSeek V4-Pro、V4-Flash 的所有 pairwise Holm 校正 p 值均为 **1**。
- CPCD 没有在这份 addendum 下运行；按 owner 决策延后到微调前后背靠背、同一 judge batch 执行。

因此 fixed-250 只保留为 diagnostic。旧交接中“provider smoke / Qwen3.8 pre-test 尚未完成”和“ESConv ≥33.61% 是部署门”两条均作废。

注意：通用 CLI 的 `dry-run` 尚不会自动读取这份 addendum 的窄范围，默认仍展示 CPCD 159 + ESConv 2,775。复现本次结果时必须显式使用 `--benchmark esconv --limit 250`；在 CLI 与协议完成机器可执行绑定前，不把默认 dry-run 当成该 addendum 的调用计划。

### 2.2 research-v0 训练包静态就绪

位置：`/Users/allenli/Desktop/静室/datasets/mental-health-instruct-research-v0/`

- 冻结数据：train 500（zh 300 / en 200）、development 50（zh 30 / en 20）。
- 来源：CPCD 300/30、AugESC 199/19、ESConv 1/1；冻结数据本轮未改。
- Qwen3.8-27B 唯一 VERIFIED profile；14B/7B 仍是 fail-closed 的 `UNVERIFIED_PLACEHOLDER`。
- 8 处模型规格硬编码已参数化；权重清单 dry-run 已测得 32 个文件、55,586,114,863 bytes（51.77 GiB），profile 容量门为 64.53 GiB。
- release 由 `scripts/package_bundle.py` 唯一生成；本次已完成第六次受控重打，连续两次构建字节一致。归档哈希只以 `releases/SHA256SUMS` 为准，不把手抄哈希当权威。
- 静态内容合同、包内 27/27 manifest、模型 profile 与 adapter-selection 反向门 41/41、归档 28 个文件/无 bytecode/归一化 metadata 检查均已通过并可本机重跑。
- `server_training_bundle/runs/` 与 `receipts/` 不存在：没有 GPU smoke、没有完整训练、没有 adapter 成品。

这批数据含 CC-BY-NC / research-only 来源。即使训练成功，产物也只能用于研究验证，**不能接入静室商业产品**。

### 2.3 research 模式的 adapter 选择语义

推荐 80GB 配置为 500 行、batch size 1、gradient accumulation 16、2 epochs：

- 每 epoch `ceil(500 / 16) = 32` 个 optimizer steps；总计**精确 64 steps**，不是旧文档估算的约 62。
- `eval_steps = save_steps = 20`，被评估的候选 checkpoint 是 **20 / 40 / 60**。
- `load_best_model_at_end=true`、`metric_for_best_model=eval_loss`、`greater_is_better=false`；顶层 `final_adapter/` 保存的是三者中最低 `eval_loss` 对应的 adapter，不一定是最后训练状态。
- 61–64 步没有新的评估点，因此不会参与 best-checkpoint 选择；`save_total_limit=2` 会保护 best 与 latest checkpoint。
- 48GB 实验配置为 `ceil(500 / 32) × 2 = 32 steps`，只有 checkpoint-20 被评估；它仍是实验路径，不能写成与 80GB 等价。
- `run_manifest.json` 必须写入 `adapter_selection`（source checkpoint、step、metric、adapter 文件名与 SHA-256）。最佳状态缺失、路径/step 不一致、metric 非有限或 checkpoint 缺 adapter 时，训练脚本在写顶层 adapter 前 fail closed。
- smoke 模式只做两步健康检查，记录 `last_trained_state_discard_only`，产物必须丢弃，不能冒充 best checkpoint。

### 2.4 商用语料 B1 已完成决策，但未形成训练 payload

最终口径是 **A3：严格排除 NC / 需另签 / 许可不明 + zh:en = 70:30**。

| 候选池 | 行数 | 组成 |
|---|---:|---|
| 中文 | 70,792 | PsyDTCorpus 4,760 + MeChat_smile 66,032 |
| 英文 | 11,775 | CounselChat 2,775 + EN_cand_CBT 9,000 |
| 合计 | 82,567 | 候选/原始池，不是冻结训练集 |
| 计划切片 | 39,250 | zh 27,475 + en 11,775；中文降采样，不重复扩充英文 |

EN_cand_CBT 已人工有条件放行；其 2,941 个机械风险命中是主题词语境错配，硬自伤/自杀 marker 为 0/9,000。必须同时披露：其 `safety_flags` 为空不能当标签、数据是六主题 LLM 合成数据、零危机场景不能训练或证明危机能力。

PsyQA 仍需上游协议，禁止进入 SFT；CounselChat 的两个换皮副本不重复计数；ESConv/AugESC/ED/KardiaBench 等 NC/ND 来源排除；CPCD/Psy-Chronicle 在 500/50 包中只承担 research smoke 角色，不属于 A3 商用 SFT。

### 2.5 T0 / T1.5 / T3 与 EmoDynamiX

| 工作流 | 当前状态 | 仍缺什么 |
|---|---|---|
| T0 模型参数化 | `complete-with-GPU-caveats`；本地 profile + adapter-selection 门 41/41 | 真实 4-bit load 后验证 LoRA=496、model/text config；GPU fast-path 绑定；14B/7B 若使用须实测建档 |
| B1 商用候选池 | `closed under A3/70:30` | 规范化、去重/切分、审计并冻结 39,250 行正式 payload |
| T1.5 评测工具 | 实现已大体完成 | 45-case 双判官实跑、区分效度、实测 σ/ICC、最终样本量；当前 `n=99` 只是 placeholder |
| 冻结 probes | crisis 20 + style 10 + RAG 10，hash 已登记 | 只能说明离线门；危机回复文本与英文 floor 漏检仍需独立验证 |
| T3 论文 | 9 节/9 表 scaffold | 结果、正式语料/训练、方法表口径同步与 LaTeX 编译环境 |
| EmoDynamiX | 只有 `DEVELOPMENTAL_SMOKE_NOT_SELECTABLE` CPU smoke | CE 与 class-balanced pilot 都没有落盘结果，也没有在跑的进程 |

判官五维的当前定义是 `relevance / empathy / safety / boundary / naturalness`；
Overall 是聚合而非第六维。主判官为 DeepSeek V4-Pro，副判官为 Qwen3-8B；Kimi 已因配额退出。Kardia-R1 权重未取得，现存三个“配置”文件其实是 155-byte 的 403 stub；rubric 方法来自论文，不依赖该权重。

## 3. 下一步执行顺序

1. **先固定发布边界。** 给 `论文和代码/` 建立白名单或迁移可公开文档/代码；不要把 raw、嵌套仓库和绝对路径 JSON 一并提交。冻结预注册在 Git 未跟踪时不具备仓库级不可变性。
2. **完成 T1.5 验证。** 持久化 judge discrimination probe；跑 45-case 双判官 pilot；用实测 σ / ICC 重算样本量，替换 placeholder `n=99`。
3. **完成 P2 商用训练集。** 从 B1 候选池规范化、按来源/场景/重复组切分并冻结 39,250 行；重新做许可、危机、PII、近重复和 train/eval 污染审计。不得把 research-v0 的 NC 数据混入。
4. **research-v0 仅做流水线演练。** 在 Linux x86_64、单卡 A100/H100 80GB 上按 `UPLOAD_AND_TRAIN.md` 依次执行校验、bootstrap、权重获取、强制 smoke、完整 research run；每一步保存 receipt/manifest。没有真实 GPU 与该栈验证前，不宣称训练包“可训练完成”。
5. **正式 SFT / 后续优化。** 商用 payload 冻结后再训练产品候选；之后才做 RAG 样本、偏好/GRPO 与消融。
6. **部署验证。** 用同一冻结五维 rubric 做 pre/post 配对；独立通过输出侧安全、危机回复、RAG 引用和静室 45-case 回归。ESConv fixed-250 只做诊断，不作 go/no-go。

任何付费 API 批跑、GPU 全量训练、生产部署或 Git push 都应在执行前单独确认范围与成本。

## 4. 当前验证底线

- Qwen3.8 addendum 校验、250 条输出 ID/协议/prompt/provenance/hash 与结果重算均通过。
- 最新 ESConv / EmoDynamiX 重点链路 16 个测试模块通过（1 个真实模型资产测试跳过）。
- 仓库全量 `unittest` 当前不是全绿：206 tests 中 7 failures、4 errors、1 skipped；主要是 strict parser 契约、Responses API 字段和 CPCD fixture 漂移。README 离线矩阵为 27/29，两个失败由未跟踪的嵌套 Git 目录触发 provenance 哈希门。
- 因此交付时只能声称相关新链路通过，不能声称整个仓库测试全绿。

## 5. 权威入口

- 项目当前状态：`project-ledger/CURRENT_STATUS.md`
- 本次对齐记录：`project-ledger/updates/2026-09-03-workspace-alignment.md`
- research-v0 训练说明：`/Users/allenli/Desktop/静室/datasets/mental-health-instruct-research-v0/UPLOAD_AND_TRAIN.md`
- research-v0 archive 哈希：同目录 `releases/SHA256SUMS`
- 商用语料状态：`论文和代码/B1_STATUS.md`
- 模型门状态：`论文和代码/STATUS_T0.md`
- 测量工具状态：`论文和代码/STATUS_T1.5_partial.md`
- Qwen3.8 基线：`reports/esconv_fixed250_qwen38_pretest_20260828.md`

若本文与带日期的旧计划、旧 HANDOFF 或历史 update 冲突，以本文件与同日项目台账为准；历史文件不回写成“当时就知道”。
