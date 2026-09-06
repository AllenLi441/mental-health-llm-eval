# ESConv-first 论文、模型与复现审计

核查日期：2026-08-20（America/Los_Angeles）

> **用途更正（2026-08-20）：** 本文前半部分是“论文数字与协议”的文献审计，不是可实施模型排名。没有真实源码/checkpoint 的 DPPLM、Causal-ESC、SAGE、CADSS 等不得用于工程基线选择；MultiESC 42.01 也不是 canonical ESConv 8-way 同协议成绩。实际的 code-first 选型、桌面包逐文件审计和当前改进结果见 [`esconv_code_first_model_audit_20260820.md`](esconv_code_first_model_audit_20260820.md)。

## 决策摘要

这次不再把不同数据版本、标签空间和评测任务的 Accuracy 混成一张榜。当前应同时保留四个互斥结论：

| 轨道 | 当前代表结论 | 结果 | 可以如何表述 |
|---|---|---:|---|
| `synthetic_expanded_protocol` | AFlow | ACC **65.10%**；F1 **63.50%** | AFlow 自建的 MCTS-expanded validation 协议最高；不是原始 ESConv gold test |
| `paper_reported_esconv_related_protocols` / headline | DPPLM | Strategy ACC **58.03%** | 本次找到的 ESConv 相关论文 headline 最高；完整 split、代码和 checkpoint 未公开，尚不可复现 |
| `paper_reported_esconv_related_protocols` / full table | Causal-ESC | Strategy ACC **53.53%** | 完整正式论文表中最高；训练/评测划分仍不够清楚，也没有官方权重 |
| `public_runnable_native_protocols` | TinyLlama AugESC classifier | ACC **41.58%**；Macro-F1 **24.53%** | 本清单中模型卡自报 Accuracy 最高的可下载八分类模型；指标来自 AugESC 原生协议，各条目不可互排，也不是 frozen-2775 |
| `frozen_2775_reproduced` | BlenderBot-small Joint | ACC **32.2162%**；Macro-F1 **21.8272%** | 本仓库在同一 frozen 2,775 条上已经重跑并可审计的当前基线 |

在本次核查到的公开、可下载且确实是八分类 next-strategy head 中，`heegyu/TinyLlama-augesc-context` 的模型卡自报 Accuracy 最高。作者模型卡在 **AugESC 自己的 test** 上自报 ACC **41.58%**、Macro-F1 **24.53%**；各公开 checkpoint 的原生协议不同，这不是 frozen-2775 成绩，也不能提前写成统一榜 SOTA。

结论因此不是“最高就是 65.10%”或“最高就是 58.03%”，而是：

> AFlow 65.10% 是新合成协议；DPPLM 58.03% 是本次找到的 ESConv 相关 headline 最高；Causal-ESC 53.53% 是完整论文表最高；本仓库公平 frozen 榜目前仍只有 BlenderBot-small Joint 32.2162%。

## Papers with Code 核查

用户指定的 `paperswithcode.co` 可以用于发现论文和代码，但它没有建立 ESConv Strategy Accuracy leaderboard：

- [原始 ESConv 页面](https://paperswithcode.co/paper/2106.01144)记录了论文和官方仓库，但 `leaderboard_ranks` 为空。
- 它能检索到 MISC、MultiESC、PAL、TransESC、SRA、CARE 等相关工作。
- 它明显漏掉 AFlow、DPPLM、Causal-ESC、SAGE、CADSS、EmoDynamiX、D2RCU、CauESC、PRCCF 等重要条目或关联。
- 因此，Papers with Code 只作为发现入口；数字最终以原论文表、正式出版页面、作者代码和模型卡交叉核对。

截至核查日，Papers with Code 收录的最新相关工作是：

1. [ESC-Skills](https://paperswithcode.co/paper/2605.27908)，2026-05-27。它研究可执行支持技能和自演化，不是原始八类单标签 next-strategy Accuracy。
2. [Modeling Multiple Support Strategies within a Single Turn](https://paperswithcode.co/paper/2604.17972)，2026-04-20。一个回复可以包含多个 strategy-response pair，输出空间已经改变，不能进入经典八分类榜。

所以“最新论文”和“经典 Accuracy 最高论文”不是同一个问题。

## 论文自报结果：协议特定参考榜

下表只用于文献综述，不表示同一测试集上的排名。

| 模型 | 年份 | Strategy ACC | 关键协议差异 | 代码 / 权重状态 |
|---|---:|---:|---|---|
| DPPLM | 2025 online | **58.03%** | 官方机构摘要可核实；完整 split 与 test 数未公开 | 未找到代码或 checkpoint |
| Causal-ESC | 2026 | **53.53%** | offline causal policy；论文对随机 20% 数据的用途表述不够清楚 | 未找到作者明确链接的代码或 checkpoint |
| SAGE | 2026 | **46.80%** | 1,300 dialogues；70/10/20；self-retrieval + COMET/HEAL | 无代码/权重；数据称可申请 |
| CADSS | 2026 | **46.26%** | 8:1:1；多智能体规划 | 仓库仍写 code forthcoming；无权重 |
| MultiESC | 2022 | **42.01%** | 8:1:1；合并 Suggestion/Information，并重定义 Greetings/Others | 代码公开；无可直接统一补测 checkpoint |
| PRCCF | 2026 preprint | **40.72%** | 1,300 dialogues；80/10/10 | 论文所列仓库不可用 |
| GREEN | 2025 | **38.80%** | 1,300-dialogue 快照；检索与外部知识 | 未找到权重 |
| DKPE / Hao–Kong | 2025 | **35.51%** | 动态知识过滤与 persona | 未找到权重 |
| D²RCU | 2024 | **35.32%** | 1,300 dialogues；80/10/10 | 未找到官方权重 |
| TransESC | 2023 | **34.71%** | 生成式策略控制 | 代码公开；无 checkpoint |
| PAL | 2023 | **34.51%** | PESConv persona augmentation；7:2:1 | 不属于原始 frozen 协议 |
| CauESC | 2024 | **33.33%** | 因果知识增强 | 代码公开；需要重新训练 |
| FADO | 2023 | **32.90%** | 论文协议特定 | 未找到可直接统一补测权重 |
| MISC | 2022 | **31.63%** | 后续表格存在 31.34/31.61/31.63 的转录差异 | 代码不完整；无权重 |
| CARE | 2025 | **30.29%** | Llama-3.1-8B SFT+RL；910/195/195 | 未找到独立 checkpoint |

需要排除的错误比较：

- AFlow 65.10 不能与上述数字直接排，因为它用从 500 个 seed 对话扩展出来的 validation dialogues。
- KEMP 39.31 来自 EmpatheticDialogues 的情绪分类，不是 ESConv 八类策略 Accuracy。
- Stage Accuracy、SR@10、strategy adherence、检索 R@K、BLEU/ROUGE/BERTScore 都不是 next-strategy Accuracy。
- Multi-strategy generation 与经典单标签八分类的输出空间不同。

## AFlow：最高新数字，但暂不租机复现

[AFlow 论文](https://arxiv.org/abs/2602.08826)表 1 报告 ESConv ACC 65.10、F1 63.50；[代码仓库](https://github.com/chz2025/AffectiveFlow)公开，但没有训练权重。

这项结果必须单列，原因是：

- 它从 500 个 ESConv seed 出发，用 MCTS 生成扩展对话树，再形成统一 taxonomy 下的 expanded validation dialogues。
- 它不是本仓库的原始 2,775-row frozen gold test。
- 论文训练配置使用 8×A100。
- 论文复现附录仍残留 `{seed}`、`{K}` 等占位符，仓库配置也需要外部 API/model 参数。

因此 AFlow 适合作为改进思想来源，不适合作为第一笔租 GPU 的复现对象。应先移植其“policy/value 分离、前缀一致性、收益估计”思想到低成本受控实验，再决定是否复现完整 MCTS。

## 公开可下载模型清单

### 已有可审计结果

| 模型 | 任务头 | 权重 | 当前证据 |
|---|---|---:|---|
| `lsy641/ESC_Blender_Strategy` | Joint generation，首 token 映射为八策略 | 约 90M | frozen-2775：894/2,775，ACC 32.2162%，Macro-F1 21.8272%，Weighted-F1 28.5626% |

### 优先核查与原生协议补测

| 模型 | 规模 | 已知指标 | 评测状态 / 风险 |
|---|---:|---:|---|
| [`heegyu/TinyLlama-augesc-context`](https://huggingface.co/heegyu/TinyLlama-augesc-context) | 1.034B，FP32 约 4.14 GB | AugESC ACC 41.58%；Macro-F1 24.53% | 正确 `LlamaForSequenceClassification` 八分类头；下载未完成，本地哈希尚未验证；只允许先复核 AugESC 原生协议，唯一候选冻结前禁止新建 frozen-test prediction；checkpoint 与 AugESC 权利链未核清 |
| [`cw-wan/EmoDynamiX-v2`](https://github.com/cw-wan/EmoDynamiX-v2) | RoBERTa-base + heterogeneous graph | 论文只报 Macro-F1 27.70%、Weighted-F1 32.71% | 官方 checkpoint 已实测：作者 test.pkl ACC **33.6097%**、Macro-F1 **27.7040%**、Weighted-F1 **32.7087%**；frozen 全量因训练重叠只能作污染诊断；产品使用权利未核清 |
| `heegyu/esconv-xlm-roberta-base` | 278M，约 1.11 GB | 未报告 | 虽是八分类头，但 checkpoint 训练来源、license 与不可变输入模板证据均缺失；profile 先禁用，只能在 dev 固定输入契约后作诊断，禁止直接预测 frozen test |
| `heegyu/esconv-xlm-roberta-large` | 560M，约 2.24 GB | 未报告 | 同上；未知来源时不具 frozen 榜资格，也不应为其先租机 |
| 三个 `thanaphatt1/ModernBERT-base-esconv-*` | 约 150M，约 598 MB | 未报告 | 数据版本、label map、单/多轮模板和 license 不清楚；输入契约与映射未固定前 runner 必须拒跑 |

`heegyu/TinyLlama-augesc-context-strategy` 虽有模型卡数字，但实际上传配置是 32,000-vocab `LlamaForCausalLM`，不是八分类 head，不能把它当分类器上榜。

以下模型属于回复生成或给定策略后的 adherence，不应进入 next-strategy 榜：`thu-coai/blenderbot-400M-esconv`、`thu-coai/blenderbot-1B-augesc`、SRA Llama-3 adapters、`heegyu/esconv-tinyllama` generation checkpoint。

### 本次实际补出的缺失 Accuracy

EmoDynamiX 论文没有给 ESConv Strategy Accuracy。本次用作者仓库 commit `c9213d7`、官方 `checkpoint-2600.pth`（SHA-256 `6cb5b273…aa4e7`）和作者 released `test.pkl` 的 2,895 条预处理样本完成了原生重跑：

| 指标 | 本次实测 |
|---|---:|
| Correct / total | 973 / 2,895 |
| Strategy ACC | **33.6097%** |
| Macro-F1 | **27.7040%** |
| Weighted-F1 | **32.7087%** |
| Invalid | 0 |

Macro-F1 和 Weighted-F1 与论文的 27.70 / 32.71 对齐到四舍五入精度，说明 checkpoint、label order 和 scorer 接通正确。完整哈希与逐类结果见 `reports/esconv_emodynamix_author_native_20260820.json`。该结果状态是 `AUTHOR_TEST_REPRODUCTION_ONLY`，不是 frozen-2775 榜成绩。

## 本次新增的污染审计

EmoDynamiX 作者把当前 1,300-dialogue ESConv 快照按 seed 13 重新切分。将 frozen-2775 所属 210 个旧对话与作者 split 做规范化文本匹配后：

| 作者 split | frozen 对话 | frozen strategy rows |
|---|---:|---:|
| train | **132** | **1,734** |
| valid | 29 | 380 |
| test | 20 | 272 |
| unmatched | 29 | 389 |
| ambiguous | 0 | 0 |

因此 released EmoDynamiX checkpoint 在 frozen-2775 上的任何全量结果都必须强制标为 `DIAGNOSTIC_ONLY_TRAIN_CONTAMINATED`，并设置 `frozen_leaderboard_eligible=false`。不能用这个结果宣称超越 32.2162% 的公平 frozen 基线。

### 官方 train/dev 的小规模样本重复

训练器的第二轮只读审计还发现：官方固定 train（632 个对话）与 dev（211 个对话）之间没有整段对话的 exact/prefix overlap，但有 **2** 种完整 TSV 行、**12** 种实际模型输入上下文跨 split 完全相同；后者在 train 中对应 **129** 行。为保证“dev 不进入梯度”的内容级含义，首次训练前已预注册一个不可关闭的过滤器：移除所有 `input_sha256` 出现在 dev 的 train 行，派生训练集固定为 **8,433** 行，records commitment 为 `55c098b7…b3480`。这属于小规模 exact-example contamination，不应误写成整段对话 split 泄漏。

## 租 GPU 决策

现在不要为 DPPLM、Causal-ESC、SAGE、CADSS 或 AFlow 直接租机：前四者没有可核验权重或完整训练配方；AFlow 虽有代码，但没有权重、成本高且复现参数未闭合。租机不能补回缺失的 artifact。

推荐顺序：

1. 本机 M4 24 GB 已完成 EmoDynamiX 作者协议；下一步只复核 TinyLlama 的 AugESC 原生协议，并用 train/dev 固定自有 RoBERTa policy。XLM-R 只能在来源证据与输入契约固定后作 dev 诊断，ModernBERT 继续阻断。
2. 只有 train/dev 实验明确显示 encoder / 1B classifier 值得扩大后，才考虑租 1×24 GB 或 1×48 GB GPU；租机不授权额外 frozen-test 运行。
3. 策略条件生成器使用本地已有 Qwen3.8-27B；QLoRA 建议 1×80 GB，或 2×48/80 GB 以留出长上下文与评测余量。
4. 只有当 AFlow 作者补齐配置、训练权重或可运行 manifest 后，再评估 8×A100 级复现。

## 改进模型：Jingshi-ESConv-Policy-v1

不直接复制一个不可复现的论文模型。第一版采用“策略 policy + 策略条件回复生成”两级架构：

```text
conversation history
        │
        ├─ emotion / discourse state encoder
        ├─ train-only strategy prototype retrieval
        └─ calibrated 8-class policy + lightweight value head
                              │
                              ▼
                 selected strategy + confidence
                              │
                     safety / escalation gate
                              │
                              ▼
                 Qwen3.8-27B strategy-conditioned LoRA
```

策略层整合近期工作的有效部分：

- EmoDynamiX / DPPLM：混合情绪、变化趋势和篇章状态，而不是只看最后一句。
- SAGE：只从 train split 建 strategy prototypes / self-retrieval，禁止 test 示例进入索引。
- AFlow / MultiESC：加入轻量 value head 和 prefix-consistency 辅助损失；第一轮不做昂贵 MCTS。
- 类别不平衡：class-balanced 或 logit-adjusted loss；以 dev Macro-F1 选模型，Accuracy 只作 tie-breaker。
- 概率可用性：在 dev 上做 temperature calibration，并为低置信度、危机和越界请求保留升级路由。

训练与确认规则：

- frozen train 先执行已冻结的 dev-exact-input 去重，再只用于训练与检索；frozen dev 只用于模型选择、模板固定与校准，绝不进入训练或检索索引。
- frozen test 在唯一候选冻结后只运行一次。
- 主选择指标为 dev Macro-F1，tie-breaker 为 Accuracy。
- test 同时报告 ACC、Macro/Weighted-F1、逐类结果、invalid、conversation-cluster bootstrap 和 paired McNemar。
- 任何单类显著回退、污染、未知 label map 或 target response 泄漏都会使结果失去 leaderboard 资格。

## 不能直接替换“静室”核心 API

ESConv 官方仓库把数据/代码限定为学术研究用途，公开数据卡也标注 CC BY-NC 4.0。用 ESConv 训练出来的权重，不能仅凭模型代码 license 推断可用于商业产品。TinyLlama、XLM-R、ModernBERT 等候选的 checkpoint、本体模型与 AugESC/其他辅助数据也必须分别完成 license 和来源审查。投入生产前需要取得相关权利人的明确商业许可，或改用权利链清晰的自有/授权数据重新训练。这不是法律意见。

此外，ESConv 只验证支持策略选择和回复质量，不验证诊断、治疗有效性、危机处置、自伤风险、药物建议或临床安全。即使策略 Accuracy 提升，也只能先作为 shadow-mode policy/router 候选；完成危机安全集、人工专家复核、拒答/升级机制、隐私与在线回滚测试后，才讨论替换 DeepSeek API。

## 仓库实现

- `scripts/eval_esconv_emodynamix.py`：prepare / infer / score / contamination audit，强制 target exclusion 和污染资格判定；审计产物为 `reports/esconv_emodynamix_contamination_audit_20260820.json`。
- `scripts/eval_esconv_hf_classifier.py`：通用 Hugging Face 八分类 checkpoint 评测器，拒绝 CausalLM、未知 label mapping 与未提交授权；正式 frozen 运行强制唯一 committed campaign、完整 2,775 行、模型/Tokenizer 整树复验和永久单次消费凭据。
- `scripts/train_esconv_policy.py`：train/dev-only RoBERTa policy trainer；固定数据、派生去重集、基模整树、loss、dev 选优和 checkpoint manifest，默认只读审计。
- `open_response_eval/esconv_metrics.py`：统一 ACC、Macro/Weighted-F1、逐类指标和混淆矩阵。
- `open_response_eval/preregistration_esconv_first_v1.json`：ESConv-first 模型开发冻结协议。
- `reports/esconv_roberta_base_materialization_20260820.json`：RoBERTa-base 不可变 revision 到本地 regular-file tree 的逐字节来源凭据。
- `reports/esconv_model_registry_20260820.json`：机器可读四轨 registry。

## 主要来源

- [原始 ESConv 论文](https://aclanthology.org/2021.acl-long.269/)；[官方仓库](https://github.com/thu-coai/Emotional-Support-Conversation)
- [DPPLM 官方机构记录](https://researchoutput.ncku.edu.tw/en/publications/applying-emotion-cause-entailment-for-help-seeker-guidance-in-emo/)
- [Causal-ESC](https://aclanthology.org/2026.acl-long.886/)
- [SAGE](https://www.sciencedirect.com/science/article/pii/S0957417426004379)
- [CADSS](https://ojs.aaai.org/index.php/AAAI/article/view/38825)
- [AFlow](https://arxiv.org/abs/2602.08826)；[代码](https://github.com/chz2025/AffectiveFlow)
- [EmoDynamiX](https://aclanthology.org/2025.naacl-long.81/)；[代码与 checkpoint](https://github.com/cw-wan/EmoDynamiX-v2)
- [TinyLlama AugESC classifier](https://huggingface.co/heegyu/TinyLlama-augesc-context)
