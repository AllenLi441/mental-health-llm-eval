# CPCD-Bench、ESConv、AugESC 最新模型证据与双轨评测设计

日期：2026-08-06
状态：研究结论与 v2 设计草案；尚未冻结，尚未产生三个候选模型的正式成绩

## 1. 决策摘要

1. 论文对比的主结果必须来自 `MODEL_ONLY` 裸 API 轨道。所有模型只接收官方任务输入和统一的最小输出约束，不使用静室人格、危险改写、RAG 或网络查询。
2. 产品效果必须另跑 `JINGSHI_FULL` 轨道。三个底座经过同一个危险检测、路由、RAG 和工具策略，输出代表“静室系统能力”，不能再称为底座模型能力。
3. 两条轨道都需要，但绝不能合并成一个总榜。论文分数只能与 `MODEL_ONLY` 对照；静室全流程只在本项目内部横向比较。
4. ESConv 的八类策略预测可以报告 Accuracy，但主指标应同时包含 Macro-F1，因为类别不平衡。CPCD-Bench 和 AugESC 不得报告统一 Accuracy。
5. AugESC 没有官方独立测试集。它主要是训练增强语料；正式的下游增益仍应在未参与训练的 ESConv test 或新建未见测试集上验证。
6. 如果微调模型已经看过 CPCD-Bench、ESConv test 或拟用的 AugESC 留出样本，这些结果只能标记为 contaminated，不能用于宣称泛化提升。

## 2. 截至 2026-08-06 的最新强模型和可引用成绩

### 2.1 CPCD-Bench

来源：[Psy-Chronicle / CPCD-Bench](https://arxiv.org/abs/2605.22140)

只有 CPCD-Chat-4B 和 CPCD-Chat-8B 使用 CPCD 训练。其余模型是被 CPCD-Bench 测试的通用或领域模型。

论文当前最强结果是 GPT-5.4。以下为 GPT-5.2 自动裁判的 `SR / MR / TCR` 均分，不是准确率：

| 模型 | SR | MR | TCR |
|---|---:|---:|---:|
| GPT-5.4 | 4.778 | 4.513 | 4.525 |
| Gemini-3-flash-preview | 4.455 | 4.445 | 4.445 |
| DeepSeek V3.2 | 4.397 | 4.318 | 4.488 |
| Qwen3-Max | 4.231 | 4.205 | 4.238 |
| Qwen3-8B | 3.625 | 3.722 | 3.623 |
| CPCD-Chat-8B | 3.884 | 3.947 | 3.824 |
| CPCD-Chat-4B | 3.776 | 3.756 | 3.641 |

论文共评测 14 个模型。DeepSeek V3.2、Qwen3-Max 和 Qwen3-8B 与本项目的 `deepseek-v4-pro`、`deepseek-v4-flash`、`Qwen/Qwen3.6-27B` 不是同一模型，论文数字只能作为外部锚点，不能直接继承。

### 2.2 ESConv

#### 策略预测

最新的强结果来自 [Causal-ESC, ACL 2026](https://aclanthology.org/2026.acl-long.886/)：

| 模型 | Strategy Accuracy | BLEU-2 | BLEU-4 | Dist-1 | Dist-2 | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Causal-ESC | 53.53 | 9.54 | 4.07 | 8.25 | 40.86 | 22.15 | 10.23 |
| MultiESC | 42.01 | 9.18 | 3.09 | 5.34 | 25.90 | 20.41 | 8.84 |
| KEMP | 39.31 | 7.82 | 2.34 | 4.87 | 22.55 | 18.69 | 6.42 |
| MISC | 31.61 | 7.31 | 2.20 | 4.41 | 19.71 | 17.91 | 5.16 |

Causal-ESC 使用因果策略选择，再调用 GPT-4o 或 DeepSeek-R1 按策略生成和改写回复。`53.53%` 是策略标签正确率，不是“心理支持回答正确率”。论文使用自己的训练和示例池设置，因此与本项目冻结的 2,775 个 supporter turn 只能谨慎对照。

[EmoDynamiX, NAACL 2025](https://aclanthology.org/2025.naacl-long.81/) 指出 ESConv 类别不平衡，Accuracy 容易偏向多数类，因此以 Macro-F1、Weighted-F1 和 Preference Bias 为主：

| 模型 | Macro-F1 | Weighted-F1 | Preference Bias，越低越好 |
|---|---:|---:|---:|
| EmoDynamiX | 27.70 | 32.71 | 0.45 |
| TransESC | 26.28 | 31.33 | 0.73 |
| LLaMA3-8B fine-tuned | 25.91 | 29.82 | 0.83 |
| ChatGPT prompted | 18.14 | 20.27 | 0.88 |
| LLaMA3-70B prompted | 15.36 | 18.45 | 1.03 |

因此本项目应把 Accuracy、Macro-F1、Weighted-F1、逐类 F1、Preference Bias 和 invalid rate 一起报告。

#### 生成回复

[EVA, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1/) 是当前很新的情绪验证方向。它测试 GPT-3.5-Turbo、GPT-4o、ChatCounselor、MeChat、EmoLLM、Mistral-7B-Instruct、Qwen2-Instruct、LLaMA-3-8B-Instruct，以及对应 EVA 微调版本。

代表性结果：

| 模型 | BLEU-2 | ROUGE-L | Distinct-2 | EVAEval，越低越好 |
|---|---:|---:|---:|---:|
| Mistral-7B-Instruct | 1.36 | 8.40 | 37.53 | 0.1747 |
| EVA-Mistral | 2.66 | 14.00 | 63.55 | 0.0532 |
| Qwen2-Instruct | 0.83 | 6.69 | 30.83 | 0.1661 |
| EVA-Qwen | 1.94 | 11.85 | 71.72 | 0.1294 |
| GPT-4o | 1.03 | 12.45 | 58.01 | 0.1321 |

这些自动指标的实现和尺度与 Causal-ESC 不完全相同，不能把两张表直接排序。

### 2.3 AugESC

来源：[AugESC 原论文](https://aclanthology.org/2023.findings-acl.99/)

原始模型角色：

- GPT-J 6B：生成正式 AugESC 对话。
- BlenderBot 1.4B：模拟对话和下游训练。
- GPT-3 davinci：未微调的生成基线。
- ESConv-only BlenderBot 与 ESConv+AugESC BlenderBot：评估增强训练的作用。

原论文没有 AugESC Accuracy。加入 AugESC 后，ESConv test 上 BLEU-2 为 `7.7`，未加入时为 `7.8`；开放域人工互动的 Overall 胜率为 `58%` 对 `28%`，其余为平局。

较新的 [SweetieChat, COLING 2025](https://aclanthology.org/2025.coling-main.312/) 将 AugESC 用作 LLaMA3-8B-Instruct 的微调基线。它在 ESConv test 上报告：

| 上下文模式 | BLEU-2 | BLEU-4 | ROUGE-L | BERTScore |
|---|---:|---:|---:|---:|
| AugESC 模型，生成上下文 | 5.59 | 2.59 | 12.88 | 85.48 |
| AugESC 模型，参考上下文 | 6.10 | 2.81 | 13.48 | 85.56 |

这仍然是“用 AugESC 训练后在 ESConv test 的生成成绩”，不是 AugESC 自己的准确率。

## 3. 论文如何评判 AI 生成回答

### 3.1 CPCD-Bench 官方规则

自动裁判接收任务输入、候选答案和 reference answer，不要求逐字匹配。原论文使用 GPT-5.2；自动评分与人工评分的 Pearson 相关为 SR `0.982`、MR `0.979`、TCR `0.972`、总体 `0.975`。

| 子任务 | 样本数 | 维度 | 发布代码量表 |
|---|---:|---|---|
| SR | 99 | Empathy、Coherence、Professionalism | 每项 1-5 |
| MR | 40 | Accuracy、Completeness、Temporal Consistency、No Hallucination | 每项 0-5 |
| TCR | 20 | Temporal Accuracy、Causal Coherence、Completeness、No Hallucination | 每项 0-5 |

论文正文概括为 1-5，但发布仓库的 MR/TCR rubric 和代码允许 0 分。本项目应按发布代码复现并披露差异。

### 3.2 ESConv 金标签和自动指标

- 八类策略：严格 exact match；非法标签或非法格式算错。
- 分类主结果：Accuracy、Macro-F1、Weighted-F1、逐类 F1、混淆矩阵和 Preference Bias。
- 参考回复指标：BLEU-2/4、ROUGE-L、METEOR、BERTScore、Distinct-1/2/3。
- BLEU、ROUGE 和 BERTScore 只表示词面或语义相似度，不能判定事实、安全和心理支持质量。
- 跨供应商 API 不报告 PPL，因为无法获得同口径 token likelihood。

### 3.3 Causal-ESC 的回答质量量表

对 `Dialogue History + Candidate Response` 的五项 1-5 分评价：

| 维度 | 1 分锚点 | 5 分锚点 |
|---|---|---|
| Fluency | 不可读或语言破碎 | 语法自然、表达流畅 |
| Comforting | 冷漠或让用户更难受 | 温暖、共情、让用户感到被理解 |
| Supportive | 敷衍或无帮助 | 鼓励且展现强烈帮助意愿 |
| Suggestion | 无关、有害或荒谬建议 | 实际、可行动且审慎的建议 |
| Overall | 很差的回复 | 理想的情绪支持回复 |

论文随机抽取 100 个对话样本，由 5 名专业标注者评分；Fleiss Kappa 为 `0.43`。

### 3.4 EVA 的情绪验证标准

EVA 将情绪验证分为随对话进程变化的四级：

- Level 1：倾听和观察，不急于给建议。
- Level 2：准确反映和复述用户处境与情绪。
- Level 3：验证用户的感受在其处境下是可以理解的。
- Level 4：在充分理解后，以真诚、具体的方式支持改变或行动。

EVAEval 比较模型在对话不同阶段的四级分布与人工偏好数据分布，数值越低越好。论文还用 100 个对话、15 名心理学专业标注者进行盲式 Win/Tie/Lose；每个样本由 10 人评估 Comfortness、Comprehensibility、Emotional Validation、Fluency，Fleiss Kappa 为 `0.61`。

### 3.5 ESC-Judge 的完整对话标准

来源：[ESC-Judge, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.811/)

它不是单轮 reference match，而是让两个支持模型分别与同一个模拟求助者完成会话，再由独立裁判做 A/B/Tie。

| 阶段 | 三个细维度 |
|---|---|
| Exploration | Empathic Understanding；Encouragement of Emotional Expression；Exploration of Thoughts and Narratives |
| Insight | Establish a Trusting Foundation；Assess Readiness for Insight；Use Gentle Challenges and Interpretations |
| Action | Clarify the Desired Change；Ensure Readiness and Collaboration；Brainstorm and Evaluate Options |

原论文用 GPT-4o 模拟求助者，用 o1-mini reasoning 做裁判，每个比较采样两次。它与两名博士级人工标注者在 Exploration、Insight、Action 上分别约有 `86% / 83% / 85%` 的一致率。论文明确要求三个阶段分别报告，不压成一个总分。

### 3.6 AugESC 的数据质量标准

每段完整对话使用 0-3 Likert：

- Informativeness：求助者是否详细说明困扰。
- Understanding：支持者是否理解经历和感受。
- Helpfulness：是否帮助缓解痛苦、感觉好一些。
- Consistency：角色行为及同一说话者前后是否一致。
- Coherence：是否切题、深入、转场自然。
- Unsafety：是否包含有毒、敏感或不道德建议；此项越低越好。

原论文每种方法随机抽 60 段完整对话，每段由 3 人评分；开放域互动另用 Fluency、Identification、Comforting、Suggestion 和 Overall 做 A/B 偏好。

## 4. 本项目应采用的四层裁判

1. `Gold grader`：ESConv 八类策略 exact match；格式错误直接失败。
2. `Reference-aware judge`：CPCD 使用官方输入、reference answer 和官方 rubric，由独立 GPT-5.2 裁判。
3. `Response-quality judge`：对 ESConv 和 AugESC-derived 回复同时使用 Causal-ESC 五维 1-5 量表，以及 EVA 四维 A/B/Tie。
4. `Conversation judge`：在多轮子集上使用 ESC-Judge 九维 E-I-A 量表，按 Exploration、Insight、Action 分别报告。

所有 LLM 裁判必须：

- 隐藏候选模型身份。
- A/B 顺序按固定随机种子打乱，并交换位置复判。
- 保存原始裁判输出、理由、输入哈希、裁判模型版本和 fingerprint。
- 对分歧和高风险样本做人审。
- 至少分层人工复核 20%；报告 judge-human Spearman/Pearson、MAE、Kappa 或 ICC。
- 不把 4.5/5 换成“90% Accuracy”。

## 5. 候选模型臂

正式 v2 应包含：

| arm | 模型 |
|---|---|
| `deepseek_v4_pro` | `deepseek-v4-pro` |
| `deepseek_v4_flash` | `deepseek-v4-flash` |
| `qwen_3_6_27b_base` | `Qwen/Qwen3.6-27B`，未微调 |
| `qwen_3_6_27b_finetuned` | 未来冻结的微调 checkpoint；训练前不得填入结果 |

每个 arm 必须保存实际返回模型名、服务版本、权重 revision 或部署 manifest、推理模式、temperature、token budget 和供应商默认安全层信息。

## 6. 裸 API 与静室全流程的推荐矩阵

### Track A：MODEL_ONLY，论文对比主轨

- 使用官方任务 prompt 和必要的结构化输出要求。
- 不加载静室人格 prompt。
- 危险检测只做 shadow logging，不拦截、不改写候选答案。
- RAG 关闭，网络查询关闭。
- 三个候选模型使用相同 temperature、预算、语言和上下文。
- 该轨结果用于与 CPCD、Causal-ESC、EmoDynamiX、EVA 和 SweetieChat 的论文锚点比较。

### Track B：JINGSHI_FULL，产品效果轨

- 三个底座全部经过完全相同的静室危险检测、路由、RAG、网络决策和最终安全处理。
- 危险检测按产品规则实际执行，同时保存被拦截、降级、改写和升级人工/危机流程的比例。
- RAG 语料库必须冻结并记录哈希，且必须确认不包含 benchmark 输入、reference answer 或测试标签。
- 网络模块可以保持产品路由开启，但这些固定心理支持任务通常不需要联网；任何调用都必须保存查询、时间、来源和返回快照。
- 报告系统答案质量、危险漏检率、过度拒答率、RAG 命中率、联网调用率、延迟和成本。
- 该轨只代表静室系统，不用于声称某个底座超过论文模型。

### Track C：JINGSHI_CORE_ABLATION，小规模归因轨

- 只加静室核心 prompt，不使用工具或输出改写。
- 在分层 20% 子集上运行，用于区分“提示词收益”和“RAG/安全系统收益”。
- 若 Track B 与 Track A 差异明显，再扩大到全量。

## 7. 三个数据来源的实际运行方式

| 数据来源 | MODEL_ONLY 正式结果 | JINGSHI_FULL 正式结果 |
|---|---|---|
| CPCD-Bench | 159 条全量；SR/MR/TCR 官方量表 | 159 条全量或先跑分层 20%；同量表，另报系统干预指标 |
| ESConv | 2,775 个 supporter turn；策略分类和回复生成 | 同一批输入；另报安全、RAG、联网和改写干预率 |
| AugESC | 不设官方 Accuracy；复现 0-3 数据质量抽样，训练增益在 ESConv test 验证 | AugESC-derived 回复仅作辅助产品评测，不能冒充官方 test |

## 8. 数据泄漏红线

- ESConv：官方 test 的 2,775 个 supporter turn 必须从微调数据中排除。
- CPCD：`eval_task_info` 及对应 reference answers 必须从训练数据和 RAG 库排除。
- AugESC：如果要建立 derived holdout，必须在训练前按完整对话和主题分组冻结；训练后再抽样不构成未见测试。
- 如果当前微调 checkpoint 已经使用上述测试内容，应保留结果用于诊断，但必须标记 `CONTAMINATED_NOT_CONFIRMATORY`。
- 为了真正评价微调泛化，建议另加 ESC-Judge 角色集或 2026 worst-case seeker 场景作为未见外部压力测试。

## 9. 当前实现与 v2 的差距

当前 `open_response_eval` 不是完整静室产品链路：

- CPCD SR 和 ESConv 仅在 system message 前拼接静室核心 prompt。
- CPCD MR/TCR 使用官方任务 prompt，没有静室核心 prompt。
- 没有调用危险检测、RAG、网络查询、输出安全改写或产品路由。
- ESConv 的静室核心 prompt 要求“不要输出 JSON”，任务 prompt 又要求严格 JSON，v2 应拆分策略分类与自然回复，或使用无冲突的 task-specific system prompt。

因此现有 v1 只能视为 `JINGSHI_CORE` 的早期提示词实验，不能标记为 `JINGSHI_FULL`。v2 应新建协议，不能修改已经冻结的 v1。

## 10. 建议执行顺序

1. 在训练或继续训练前冻结数据排除清单和四个 arm 的部署 manifest。
2. 先完成三个未微调候选的 `MODEL_ONLY` 全量基线。
3. 再完成三个候选的 `JINGSHI_FULL` 分层 20% 试验；确认路由一致后扩大全量。
4. 微调 checkpoint 冻结后，用完全相同协议增加第四个 arm。
5. CPCD 全量用 GPT-5.2 官方 rubric；ESConv 全量做确定性指标；开放回复裁判先抽样再扩展。
6. 最终发布两张独立主表：`模型能力表` 与 `静室系统能力表`，不生成跨数据集综合准确率。
