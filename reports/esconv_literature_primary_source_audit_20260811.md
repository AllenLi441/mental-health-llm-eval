# ESConv 后续方法原文核查

核查日期：2026-08-11

这份记录只讨论明确使用 ESConv 的工作。数字只采用论文原文、出版社页面或作者机构页面。没有拿到全文的论文，不把摘要中的数字当作已经完成协议复核的结果。

## 一、先说结论

1. 2021 年 ESConv 原论文没有报告“策略预测 Accuracy”。原论文主要比较回复生成质量和人工互动效果，所以不能把某个后续论文的 Accuracy 倒写成 2021 baseline。
2. 目前本地已取得并读取 6 篇全文：ESConv 原论文、MultiESC、EmoDynamiX、GREEN、SAGE、Causal-ESC。
3. DPPLM 只有付费摘要，无法核对实验协议，已从本报告和成绩对比中删除。
4. 后续论文报告的 `42.01%`、`38.8%`、`46.8%`、`53.53%` 不是一组天然可横向排序的成绩。它们的数据版本、标签处理、切分方式或预测目标存在差异。
5. 截至本次核查，没有一篇后续论文已经被证明与我们当前 fixed250 的样本、标签映射、输入窗口和评分脚本完全相同。因此这些成绩只能写成 `paper-reported, protocol-specific`，不能直接写成“我们的统一 baseline”。

## 二、统一核查标准

每篇论文检查以下项目：

| 项目 | 要核对的内容 |
|---|---|
| 数据 | 是否明确使用 ESConv；使用 1,053 对话还是后续 1,300 对话版本 |
| 任务 | 回复生成、下一策略预测、策略规划，还是其他任务 |
| 标签 | 是否保留原始 8 类；是否合并、删除或新增标签 |
| 输入 | 使用多少轮历史；是否加入情绪标签、知识图谱、检索结果或未来信息 |
| 切分 | 是否使用 2021 原论文的 6:2:2；是否重新随机切分 |
| 指标 | Accuracy、Macro-F1、Weighted-F1，或 BLEU/ROUGE 等生成指标 |
| 证据 | 数字来自正文表格、摘要，还是出版社宣传文字 |
| 可比性 | 能否直接和我们的 fixed250 Accuracy 比较 |

## 三、2021 ESConv 原论文

论文：[Towards Emotional Support Dialog Systems](https://aclanthology.org/2021.acl-long.269/)

### 数据和任务

- 原论文筛选后保留 1,053 段合格对话。
- 对话支持过程分为 Exploration、Comforting、Action 三个阶段。
- supporter 使用 8 类策略：Question、Restatement or Paraphrasing、Reflection of Feelings、Self-disclosure、Affirmation and Reassurance、Providing Suggestions、Information、Others。
- 论文中的策略标注总数为 14,855。
- 原始数据切分为 6:2:2。
- 每个训练片段包含目标 supporter 回复和它之前的 4 条 utterance。

### 原论文方法

- 基础模型是 DialoGPT-small 和 BlenderBot-small。
- Vanilla 只生成回复。
- Joint 先生成策略 token，再生成回复。
- Oracle 在生成时直接提供真实策略，用来观察正确策略对回复质量的帮助。
- Adam 学习率为 `5e-5`，训练 5 epochs，按验证集最低 PPL 选择 checkpoint。
- 推理采用 `top-p=0.9`、`top-k=30`、`temperature=0.7`、`repetition penalty=1.03`。

### 原论文报告的数据

原论文没有给 Joint 模型报告策略 Accuracy。表 4 报告的是生成指标：

| 模型 | PPL | BLEU-2 | ROUGE-L | Extrema |
|---|---:|---:|---:|---:|
| DialoGPT Vanilla | 15.51 | 5.13 | 15.26 | 49.80 |
| DialoGPT Joint | 未报告 | 5.00 | 15.09 | 49.97 |
| DialoGPT Oracle | 15.19 | 5.52 | 15.82 | 50.18 |
| BlenderBot Vanilla | 16.23 | 5.45 | 15.43 | 50.49 |
| BlenderBot Joint | 未报告 | 5.35 | 15.46 | 50.27 |
| BlenderBot Oracle | 16.03 | 6.31 | 17.90 | 51.65 |

### 对我们实验的含义

2021 原论文不能提供一个论文原生的策略 Accuracy baseline。我们目前的 `32.22%`（full test）和 `37.60%`（fixed250）是作者 Joint checkpoint 经修正 token 映射后的本地重放结果，不是原论文表格中的成绩。正式写法必须区分：

> 2021 论文未报告策略 Accuracy；本项目重放作者 Joint checkpoint，在本项目评测脚本下得到 full-test 32.22%、fixed250 37.60%。

## 四、MultiESC（2022）

论文：[Improving Multi-turn Emotional Support Dialogue Generation with Lookahead Strategy Planning](https://aclanthology.org/2022.emnlp-main.195/)

### 是否使用 ESConv

是。论文直接在 ESConv 上训练和评测。

### 方法和创新点

- 使用 BART-small 建模多轮情感支持对话。
- 从对话历史和 VAD 情绪特征估计动态用户状态。
- 先生成一串候选未来策略，而不是只预测下一步策略。
- 用用户反馈预测器估计不同策略序列可能带来的情绪改善。
- 在 beam search 中加入类似 A* 的前瞻评分，选择未来收益更高的策略序列。
- 最后根据选定策略生成回复。

### 论文报告的数据

| 模型 | Strategy Accuracy | Weighted-F1 | Feedback |
|---|---:|---:|---:|
| DialoGPT-Joint | 26.03 | 23.86 | 2.87 |
| BlenderBot-Joint | 29.92 | 29.56 | 3.05 |
| MISC | 31.61 | 未报告 | 未报告 |
| MultiESC | **42.01** | **34.01** | **3.85** |
| MultiESC w/o lookahead | 38.76 | 30.21 | 3.36 |

前瞻规划在论文自己的协议中带来 `42.01 - 38.76 = 3.25` 个百分点的 Accuracy 增益。

### 与 2021 原论文没有对齐的地方

- MultiESC 把 Providing Suggestions 和 Information 合并为一类。
- 它从 Others 中单独提取 Greetings。
- 剩余 Others 不参与策略规划训练。
- 附录写的数据切分是 8:1:1，而 2021 原论文是 6:2:2。

因此 `42.01%` 是真实存在于论文表格中的结果，但不是原始 8 类、原始 6:2:2 协议下可直接复用的 baseline。可以借鉴“用户状态建模 + lookahead”，不能直接拿 42.01% 和 fixed250 做严格差值。

## 五、EmoDynamiX（2025）

论文：[Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics](https://aclanthology.org/2025.naacl-long.81/)

### 是否使用 ESConv

是。论文在 ESConv 和 AnnoMI 上做下一策略预测。ESConv 部分保留原始 8 类策略。

### 方法和创新点

- 情绪识别器不只输出一个硬标签，而是保留混合情绪概率分布。
- 用温度调整突出主要情绪，同时保留次要情绪信息。
- 使用预训练 discourse parser 建立 utterance 之间的篇章关系。
- 构造异构图，把情绪节点、策略节点和篇章关系一起建模。
- 使用关系图注意力和全局虚拟节点聚合多轮信息。
- 采用与类别频率成反比的 weighted cross-entropy，降低类别不平衡影响。

### 论文报告的数据

论文认为 ESConv 类别不平衡，主指标采用 Macro-F1、Weighted-F1 和 Preference Bias，而不是 Accuracy。

| 模型 | Macro-F1 | Weighted-F1 | Preference Bias |
|---|---:|---:|---:|
| LLaMA3-70B | 15.36 | 18.45 | 1.03 |
| LLaMA3-70B + 2-shot | 17.70 | 21.47 | 1.29 |
| ChatGPT | 18.14 | 20.27 | 0.88 |
| RoBERTa | 25.04 | 27.94 | 0.68 |
| BART | 25.66 | 29.08 | 0.64 |
| LLaMA3-8B | 25.91 | 29.82 | 0.83 |
| MISC | 20.91 | 24.93 | 0.89 |
| MultiESC | 25.73 | 29.31 | 0.61 |
| KEMI | 24.69 | 26.80 | 0.86 |
| TransESC | 26.28 | 31.33 | 0.73 |
| EmoDynamiX | **27.70** | **32.71** | **0.45** |

### 与原论文对齐情况

- 优点：明确使用 ESConv，保留原始 8 类策略，任务也是下一策略预测。
- 差异：它把数据转换为五条历史 utterance 的滑动窗口，共得到 18,376 个样本；论文关注独立策略分类，不是原论文的“策略 token + 回复联合生成”。
- 论文没有报告 ESConv Accuracy。不能把其 DailyDialog 情绪识别器的 `82.26%` 或其他情绪 Accuracy 写成 ESConv 策略 Accuracy。

这是目前最适合直接借鉴的分类方法之一，但正式复现时仍要强制使用我们的冻结 split、原始标签映射和同一评分脚本。

## 六、GREEN（2025）

论文：[Generative Retrieval-Enhanced Emotional Support Conversations](https://journals.sagepub.com/doi/10.1177/21582440251395922)

### 是否使用 ESConv

是。论文只在 ESConv 上评测。不过它写的是 1,300 段对话、38,365 条 utterance，与 2021 原论文筛选后的 1,053 段对话不一致。这说明它使用了后续数据快照，或至少没有证明与 2021 论文表格使用的是同一快照。

### 方法和创新点

- 不直接依赖向量最近邻，而是让模型生成离散检索标识。
- 使用 residual quantization VAE 生成层级 ResID，编码语境、情绪状态、策略和回复类型。
- 使用语义保持匹配、自适应 margin 和修改后的 Sinkhorn-Knopp 平衡 codebook。
- 使用 COMET 补充事件的影响、意图、需求和愿望。
- 使用 HEAL distress-management graph 补充支持性知识。
- 通过多知识 cross-attention 融合对话、COMET、HEAL 和检索样本后生成回复。

### 论文报告的数据

| 模型 | Strategy Accuracy | PPL | BLEU-4 | Distinct-1 | Distinct-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| BlenderBot-Joint | 27.72 | 18.11 | 1.66 | 3.27 | 20.87 | 15.13 |
| PAL | 34.51 | 15.92 | 2.66 | 5.00 | 30.27 | 18.06 |
| D2RCU | 35.32 | 15.43 | 2.31 | 4.97 | 26.21 | 18.71 |
| GREEN | **38.8** | 15.87 | 2.88 | 6.24 | 32.86 | 18.51 |

出版社 HTML 表格中 GREEN 的 BLEU-2 单元格排列不清楚，因此这里不抄写该值。

论文所说的 Accuracy 提升约 `9.8%` 是相对提升：

`(38.8 - 35.32) / 35.32 = 9.85%`

绝对提升是 `3.48` 个百分点，不是增加 9.8 个百分点。

### 与原论文对齐情况

- 使用 ESConv，但数据规模已经与 2021 论文不一致。
- 引入外部知识和检索，因此更接近带 RAG 的系统评测，不是纯模型策略分类。
- 全文提到沿用 original split，但没有消除 1,053 与 1,300 数据版本差异。

`38.8%` 可以作为 GREEN 论文自己的成绩，不能直接作为我们 fixed250 的同协议 baseline。

## 七、SAGE（2026）

论文：[SAGE: Self-retrieval-augmented generative LLM for emotional support conversation](https://www.sciencedirect.com/science/article/pii/S0957417426004379)

### 是否使用 ESConv

是。出版社页面明确说明模型在 ESConv 的 8 类策略任务上评测。

### 方法和创新点

- 在一个 LLM 中统一索引、检索和自评估，而不是单独使用外部 embedding 检索器。
- 记忆 utterance-response-strategy 三元组。
- 使用层级 ResID 编码 strategy、affect 和 context。
- 根据当前情绪状态预测策略。
- 用 strategy-specific trie 约束检索标识解码，避免检索到不符合目标策略的样本。
- 重排同时考虑模型自评估、COMET cognitive alignment 和 HEAL therapeutic appropriateness。
- 使用 cross-attention 融合上下文、COMET、HEAL 和检索结果后生成回复。

### 本地全文确认的数据

本地 PDF 的表 1 和表 2 给出：

| 模型 | Strategy Accuracy | Macro-F1 |
|---|---:|---:|
| GPT-4o | 41.28% | 32.27% |
| Mistral-7B-Instruct | 38.91% | 31.39% |
| LLaMA-3.1-8B-Instruct | 37.45% | 32.89% |
| Qwen-2.5-7B-Instruct | 36.89% | 29.99% |
| SAGE | **46.80%** | **38.95%** |

SAGE 还报告 PPL `14.32`、BLEU-2 `11.87`、BLEU-4 `3.88`、Distinct-1 `9.24`、Distinct-2 `39.86`、ROUGE-L `20.51`。

论文所谓比 GPT-4o 提升 `13.4%` 是相对提升：`(46.80 - 41.28) / 41.28 = 13.37%`。绝对提升是 `5.52` 个百分点。

### 与原论文对齐情况

- SAGE 使用 1,300 段对话、38,365 条 utterance 的后续 ESConv 数据版本。
- 论文附录明确采用 70/10/20：训练 910 段、验证 130 段、测试 260 段。
- 这不是 2021 原论文写明的 1,053 段合格对话和 6:2:2 协议。
- SAGE 还加入自检索、COMET 和 HEAL，因此属于检索增强系统，不是纯模型策略分类。

所以 `46.80%` 已经由本地全文确认，但仍不能直接和我们的 fixed250 纯 API 成绩相减。

## 八、Causal-ESC（2026）

论文：[Causal-ESC](https://aclanthology.org/2026.acl-long.886/)

### 是否使用 ESConv

是。论文使用 ESConv 和 EmpatheticDialogues；ESConv 部分采用 8 类支持策略。

### 方法和创新点

- 第一阶段学习策略 policy，第二阶段生成具有指定策略风格的回复。
- 状态包含对话上下文、问题描述、初始情绪、归一化轮次、当前 sentiment 和上一策略。
- 奖励定义为采用某策略前后求助者 sentiment 的变化。
- 同时训练 propensity model 和 reward model。
- 使用 Doubly Robust 估计，把 direct method 与 inverse propensity weighting 结合，并对 propensity 做 clipping。
- 回复阶段先生成与上下文相关的草稿，再检索同策略示例进行 style rewrite。
- 论文实验使用 GPT-4o 和 DeepSeek API 作为生成器。

### 论文报告的数据

| 模型 | Strategy Accuracy |
|---|---:|
| MISC | 31.61% |
| KEMP | 39.31% |
| MultiESC | 42.01% |
| Causal-ESC | **53.53%** |

Causal-ESC 的生成结果包括：

| BLEU-2 | BLEU-4 | Distinct-1 | Distinct-2 | ROUGE-L | METEOR |
|---:|---:|---:|---:|---:|---:|
| 9.54 | 4.07 | 8.25 | 40.86 | 22.15 | 10.23 |

### 53.53% 到底是不是“瞎吹”

不是凭空编出来的数字：`53.53%` 确实出现在论文结果表中。但目前不能把它认定为与 2021 原始协议完全一致的最高 baseline，原因是：

- 论文没有清楚给出与 2021 原论文 6:2:2 对应的完整 split 复现说明。
- 附录写到随机选择 ESConv 的 20% 用于 policy training 和 Style Demonstration Pool；正文又写 style pool 来自 original training data 的 20%，表述需要代码或作者说明才能排除歧义。
- 结果表沿用了 MultiESC 的 `42.01%`，而 MultiESC 已修改标签和切分。
- 生成阶段还使用检索示例和外部 API，不能与纯 API 下一策略分类混为一项。

因此准确结论是：

> 53.53% 是 Causal-ESC 的论文报告值，不是虚构值；但论文现有协议描述不足以证明它与我们的 fixed250 或 2021 原始标签/切分严格同协议，所以只能作为待复现的文献目标。

## 九、六篇论文放在同一张表里

| 论文 | 明确使用 ESConv | 任务 | ESConv 结果 | 与 2021 原协议的主要差异 | 能否直接和 fixed250 比 |
|---|---|---|---:|---|---|
| ESConv 2021 | 是 | 策略条件回复生成 | 未报告策略 Accuracy | 原始协议本身 | 否；只能用本地 checkpoint 重放建立 Accuracy |
| MultiESC 2022 | 是 | 策略规划＋回复生成 | Acc 42.01 | 合并/删除/新增策略；8:1:1 | 否 |
| EmoDynamiX 2025 | 是 | 下一策略分类 | Macro-F1 27.70；Weighted-F1 32.71 | 滑动窗口分类；未报告 Acc | 不能比 Accuracy；方法可同协议复现 |
| GREEN 2025 | 是 | 检索增强策略与回复生成 | Acc 38.8 | 1,300 对话版本；外部知识/RAG | 否 |
| SAGE 2026 | 是 | 自检索增强策略与回复生成 | Acc 46.8 | 完整 split/表格待 PDF | 暂时不能 |
| Causal-ESC 2026 | 是 | 因果策略学习＋回复重写 | Acc 53.53 | 20% 数据描述含糊；沿用 MultiESC 比较 | 暂时不能 |

## 十、哪些创新值得我们采用

### 第一优先：在冻结的原始 8 类协议中直接实验

1. EmoDynamiX 的类别加权损失、混合情绪分布和篇章图。这些改动直接服务于下一策略预测，最容易在不改变测试集的前提下做消融。
2. MultiESC 的 user-state 和 lookahead。保留方法，但恢复原始 8 类，不合并 Suggestions/Information，不删除 Others，不增加 Greetings。
3. Causal-ESC 的 Doubly Robust 学习。需要先冻结 reward、propensity、训练集边界，禁止测试样本或 style pool 泄漏。

### 第二优先：作为完整静室产品实验，不放进 MODEL_ONLY 主榜

1. SAGE 的自检索、策略 trie 约束、COMET/HEAL 重排。
2. GREEN 的 ResID 检索和多知识融合。

这两类方法会引入检索与外部知识，适合 `JINGSHI_FULL`，但开启后不能把成绩写成纯模型能力。

## 十一、建议冻结的正式 ESConv 评测协议

1. 固定同一个 ESConv 数据快照并记录文件哈希。
2. 固定 2021 原始 8 类标签，不合并、不删除、不新增。
3. 固定 train/dev/test 样本 ID；fixed250 只能作为开发小榜，full test 才是正式成绩。
4. 输入只允许当前样本规定的历史对话，禁止未来 utterance。
5. 主指标同时报告 Accuracy、Macro-F1、Weighted-F1、每类 F1、混淆矩阵和无效输出率。
6. 所有模型使用相同 prompt、温度、最大输出、解析和重试规则。
7. `MODEL_ONLY` 禁止 RAG、联网、检索示例和输出改写。
8. `JINGSHI_FULL` 单独报告危险检测、RAG、联网和过滤全部开启后的产品成绩。
9. 后续论文数字放在“文献参考”栏；只有在同一冻结协议重跑成功后，才进入“可直接比较”栏。

按这套规则，我们当前可以诚实使用的同协议历史参照仍是本地 Joint checkpoint 重放：fixed250 `37.60%`。DeepSeek V4-Pro `14.80%`，相差 22.80 个百分点；未微调 Qwen3.6-27B `14.00%`，相差 23.60 个百分点。`53.53%` 暂时只作为待复现的论文目标。

## 十二、PDF 下载情况

| 论文 | 本地状态 | 说明 |
|---|---|---|
| ESConv 2021 | 已下载官方 PDF | ACL Anthology |
| MultiESC 2022 | 已下载官方 PDF | ACL Anthology |
| EmoDynamiX 2025 | 已下载官方 PDF | ACL Anthology |
| Causal-ESC 2026 | 已下载官方 PDF | ACL Anthology |
| GREEN 2025 | 已读取本地原始 PDF | SAGE Open 出版版本 |
| SAGE 2026 | 已读取本地原始 PDF | Expert Systems With Applications 出版版本 |

本次六篇均以用户本地下载的出版版本或 ACL Anthology 官方版本完成核查。DPPLM 因无全文，完全移出结果表。
