# ESConv Benchmark：Strategy Accuracy、F1 与生成指标原始证据审计

**核查日期：2026-08-19**  
**仓库：** `AllenLi441/mental-health-llm-eval`  
**范围：** 明确使用 ESConv 或其衍生版本、并报告下一支持策略预测或策略控制生成结果的论文。原始论文表、出版社页面和本仓库机器可读结果优先于后续论文转引。

## 一、最终结论

1. **截至 2026-08-19，在本次原始证据审计找到并核实的最高论文自报 ESConv Strategy Accuracy 是 DPPLM：58.03%。** DOI/online-first 记录在 2025 年，最终卷期元数据为 IEEE Transactions on Computational Social Systems 13(2), 2060–2072（2026）。官方机构/出版社摘要明确给出 58.03%，但完整实验表、精确 split、代码和 checkpoint 未取得，因此证据等级为 **B**：数字有官方出处，但尚不是可独立复现的统一 SOTA。
2. **目前完整原论文表中可直接核到的最高值是 Causal-ESC：53.53%。** 该数字确实出现在 ACL 2026 Table 2，并且论文把 ACC 定义为策略选择正确率；但它使用论文特定的 offline causal-policy + LLM rewriting 协议，不能直接继承为本仓库 frozen 2,775-turn 榜首。
3. **SAGE 为 46.80% ACC、38.95% Macro-F1；CADSS 为 46.26% ACC。** 两篇都是 2026 年正式发表工作，但它们的数据版本、切分和系统输入都与 2021 原始 ESConv 或本仓库冻结测试不同。
4. **2021 年原始 ESConv 论文没有发表 Strategy Accuracy。** 本仓库当前机器可读 replay 文件记录的是作者 BlenderBot-small Joint checkpoint 在 2,775 条冻结测试上的 **32.22%（894/2,775）**、Macro-F1 **21.83%**、Weighted-F1 **28.56%**。这是本项目后验复算，不是 ACL 2021 论文成绩。
5. **不能制作一个不带协议列的“统一排行榜”。** 58.03、53.53、46.80、46.26、42.01 等均是正确存在的论文数字，但不是同一 test IDs、同一标签空间、同一数据版本和同一模型输入下的成绩。

## 二、这里的“准确率”到底指什么

- **Strategy ACC**：在该论文自己的评测协议下，预测策略与金标策略完全相同的样本比例。
- **Macro-F1**：先逐类算 F1，再对各策略等权平均；对少数类更敏感。
- **Weighted-F1**：按各策略样本量加权的 F1；更接近总体分布。
- **Emotion ACC**：情绪标签准确率，不能当 Strategy ACC。
- **Stage ACC**：Exploration / Comforting / Action 三阶段预测准确率，不能当 8 类策略准确率。
- **PPL、BLEU、ROUGE、Distinct、METEOR、BERTScore**：回复生成指标，不是策略准确率。

## 三、证据等级

| 等级 | 标准 | 可怎么写 |
|---|---|---|
| **A** | 已查看原论文完整结果表，并核对任务/指标定义 | “论文报告 ACC = X%，协议见该论文” |
| **B** | 官方出版社或作者机构记录明确给出 ACC，但完整表/协议不可访问 | “官方摘要报告 X%，尚待全文与代码复核” |
| **X** | 数字属于别的任务、原论文没有该指标，或仅有无法核实的二手转引 | 不进入 Strategy-ACC 排名 |

## 四、论文自报 Strategy Accuracy 排名

> 这张表按数值排序，只表示“文献中报告过的 headline”；**不代表统一 benchmark 排名**。

| 排名 | 模型 | 年份 | 期刊/会议 | Strategy ACC | F1 / 其他主要数据 | 数据/协议关键差异 | 证据 |
|---:|---|---:|---|---:|---|---|:---:|
| 1 | **DPPLM** | 2025 online / 2026 issue | IEEE Transactions on Computational Social Systems, 13(2), 2060–2072 | **58.03%** | Emotion ACC 65.42; D-1 4.63; D-2 32.24; R-L 12.10; BERTScore 0.8451 | not disclosed in the accessible official abstract；paper-reported headline only; not proven identical to the frozen 2,775-turn track | B |
| 2 | **Causal-ESC** | 2026 | ACL 2026, Long Papers | **53.53%** | B-2 9.54; B-4 4.07; D-1 8.25; D-2 40.86; R-L 22.15; METEOR 10.23 | paper-specific offline-policy setup; appendix mentions a random 20% selection for policy training/style demonstrations；not identical to the frozen 2,775-turn track; exact test IDs and a reproducible checkpoint were not released | A |
| 3 | **SAGE** | 2026 | Expert Systems with Applications, Vol. 313, Article 131524 | **46.80%** | Macro-F1 38.95; PPL 14.32; B-2 11.87; B-4 3.88; D-1 9.24; D-2 39.86; R-L 20.51 | 1,300-dialogue ESConv snapshot; 70/10/20 split (910/130/260 dialogues)；not identical to the original 1,053-dialogue 6:2:2 snapshot or the frozen 2,775-turn track | A |
| 4 | **CADSS** | 2026 | AAAI 2026, Vol. 40 No. 21 | **46.26%** | PPL 30.14; B-2 15.06; B-4 5.64; D-1 8.88; D-2 32.49; R-L 16.57 | paper reports an 8:1:1 split；paper-specific multi-agent system; not a frozen-turn matched comparison | A |
| 5 | **MultiESC** | 2022 | EMNLP 2022 | **42.01%** | Weighted-F1 34.01; Feedback 3.85 | 8:1:1 split；not directly comparable to original eight-class 6:2:2 or frozen 2,775-turn exact-match | A |
| 6 | **PRCCF** | 2026 | arXiv preprint; submitted to TACL | **40.72%** | PPL 13.10; B-2 10.71; B-4 3.55; D-1 6.17; D-2 31.16; R-L 19.78 | paper-specific ESConv setup; exact frozen IDs not released；not independently reproduced; preprint status | A |
| 7 | **GREEN** | 2025 | SAGE Open, Vol. 15 No. 4 | **38.80%** | PPL 15.87; B-4 2.88; D-1 6.24; D-2 32.86; R-L 18.51 | paper describes 1,300 dialogues and 38,365 utterances and says it follows the original split；data snapshot differs from the original 1,053-dialogue paper snapshot | A |
| 8 | **Hao–Kong framework**（后续表格别名 DKPE/CKPI） | 2025 | COLING 2025 | **35.51%** | PPL 14.88; B-2 9.27; B-4 2.92; D-1 4.88; D-2 25.95; R-L 18.87 | 80/10/10 split；paper-specific system; no frozen task-ID match | A |
| 9 | **D²RCU** | 2024 | SIGIR 2024 | **35.32%** | PPL 15.43; B-2 9.01; D-1 4.97; D-2 26.21; R-L 18.71; B-4 2.94* | paper-specific setup using ESConv/PESConv-derived retrieval information；not a frozen task-ID match | A |
| 10 | **TransESC** | 2023 | Findings of ACL 2023 | **34.71%** | PPL 15.85; B-2 7.64; B-4 2.43; D-1 4.73; D-2 20.48; R-L 17.51 | paper-specific ESConv setup；not a frozen task-ID match | A |
| 11 | **PAL** | 2023 | Findings of ACL 2023 | **34.51%** | PPL 15.92; B-2 8.75; B-4 2.66; D-1 5.00; D-2 30.27; R-L 18.06 | PESConv, 7:2:1 split；not the original ESConv split; persona-augmented dataset | A |
| 12 | **CauESC** | 2024 | arXiv preprint | **33.33%** | PPL 15.30; B-2 8.17; B-4 2.82; D-1 4.70; D-2 19.85; R-L 18.20 | 8:1:1; 1,040/130/130 dialogues; averaged over three runs；paper-specific preprint setup | A |
| 13 | **FADO** | 2023 | Knowledge-Based Systems, Vol. 264, Article 110340 | **32.90%** | PPL 15.72; B-4 2.32; D-2 21.84; R-L 17.53 | reports both the official ESConv split and the MISC re-split；the two scores must not be merged; 32.90 is the official-split row | A |
| 14 | **MISC** | 2022 | ACL 2022, Long Papers | **31.63%** | PPL 16.16; B-2 7.31; B-4 2.20; D-1 4.41; D-2 19.71; R-L 17.91; METEOR 11.05 | paper re-split differs from the original official split；not a frozen task-ID match | A |
| 15 | **CARE (SFT-RL)** | 2025 | arXiv preprint（under review） | **30.29%** | B-1 15.01; B-2 6.03; R-L 16.79; METEOR 14.56; BERTScore 16.75; D-1 4.73; D-2 27.80 | 910 train / 195 test；LLaMA-3.1-8B SFT + GRPO；preprint-specific protocol | A |

### 排名怎么读

- **DPPLM 58.03%**：截至核查日，本审计找到的最高官方论文摘要 headline；2025 为 DOI/online-first 年份，最终卷期为 2026。因为精确实验协议不可见，所以不能标成“已复现统一 SOTA”。
- **Causal-ESC 53.53%**：原论文表中真实存在，证据比 DPPLM 更完整；但属于 offline policy learning + strategy-style rewriting 的论文专用协议。
- **SAGE 46.80% / CADSS 46.26%**：均为正式发表的 2026 结果；SAGE 使用 1,300 对话版本和 70/10/20，CADSS 报告 8:1:1。
- **MultiESC 42.01%**：合并 Suggestions 与 Information、提取 Greetings，并改变规划目标；不能和原始 8 类单 turn exact-match 直接相减。
- **PAL 34.51%**：使用 persona-augmented PESConv、7:2:1，而非原始 ESConv split。
- **FADO 32.90%**：这是 official split 行；同一论文另报 MISC re-split 32.41%，两者不能混成一个值。

## 五、逐项证明 ACC 来自哪里

| 模型 | ACC | 原始证据 | 为什么判定“数字正确” |
|---|---:|---|---|
| DPPLM | 58.03% | [官方机构出版记录](https://researchoutput.ncku.edu.tw/en/publications/applying-emotion-cause-entailment-for-help-seeker-guidance-in-emo/) · [DOI](https://doi.org/10.1109/TCSS.2025.3605971) | 官方机构/出版社摘要明确写出 Strategy Accuracy 58.03%。DOI 于 2025 年建立，最终卷期元数据为 2026；完整实验表、精确 split、代码和 checkpoint 未取得，因此只能判为 B 级论文 headline。 |
| Causal-ESC | 53.53% | [原论文/出版社](https://aclanthology.org/2026.acl-long.886/) | ACC = 53.53 appears in the original ACL 2026 result table; the paper also explicitly defines Accuracy as correct strategy selection. |
| SAGE | 46.80% | [原论文/出版社](https://www.sciencedirect.com/science/article/pii/S0957417426004379) | The Elsevier article highlights explicitly state 46.8% ACC and 38.95% Macro-F1; the full paper table supplies the generation metrics. |
| CADSS | 46.26% | [原论文/出版社](https://ojs.aaai.org/index.php/AAAI/article/view/38825) | The AAAI paper result table and official repository README report ESConv ACC = 46.26. |
| MultiESC | 42.01% | [原论文/出版社](https://aclanthology.org/2022.emnlp-main.195/) | ACC = 42.01 and Weighted-F1 = 34.01 are in the original EMNLP paper table. |
| PRCCF | 40.72% | [原论文/出版社](https://arxiv.org/abs/2604.01671) | ACC = 40.72 appears in the original arXiv paper result table. |
| GREEN | 38.80% | [原论文/出版社](https://journals.sagepub.com/doi/10.1177/21582440251395922) | The publisher full-text results table reports ACC = 38.8. |
| Hao–Kong framework | 35.51% | [原论文/出版社](https://aclanthology.org/2025.coling-main.214/) | 原论文 Table 2 报 ACC = 35.51，并明确 ACC 是八类 ESConv 策略预测准确率。论文自身没有稳定使用 CKPI/DKPE 缩写；后续表格别名不作为官方模型名。 |
| D²RCU | 35.32% | [原论文/出版社](https://arxiv.org/abs/2404.02505) | The original SIGIR paper table reports ACC = 35.32. |
| TransESC | 34.71% | [原论文/出版社](https://aclanthology.org/2023.findings-acl.420/) | The original Findings ACL paper table reports ACC = 34.71. |
| PAL | 34.51% | [原论文/出版社](https://aclanthology.org/2023.findings-acl.34/) | The original Findings ACL paper table reports ACC = 34.51. |
| CauESC | 33.33% | [原论文/出版社](https://arxiv.org/abs/2401.17755) | The original arXiv PDF result table reports ACC = 33.33. |
| FADO | 32.90% | [原论文/出版社](https://www.sciencedirect.com/science/article/abs/pii/S0950705123000904) | The original paper explicitly separates official-split ACC = 32.90 from MISC-re-split ACC = 32.41. |
| MISC | 31.63% | [原论文/出版社](https://aclanthology.org/2022.acl-long.25/) | The original ACL paper table reports MISC ACC = 31.63; its BlenderBot-Joint baseline is 28.57. |
| CARE (SFT-RL) | 30.29% | [原论文](https://arxiv.org/abs/2510.05122) | 原始 preprint Table 1 报 `ACC_Stra. = 30.29`，正文定义为预测支持策略与金标策略匹配的比例；属于 under-review preprint，而非正式会议/期刊结果。 |

## 六、本仓库冻结 track：必须与论文 headline 分开

| 项目 | 数值 |
|---|---:|
| 模型 | BlenderBot-small Joint，作者 checkpoint 后验重放 |
| Test | 2,775 supporter turns |
| Correct | 894 |
| Strategy ACC | **32.2162%** |
| Macro-F1 | **21.8272%** |
| Weighted-F1 | **28.5626%** |
| Invalid | 0 |
| Test SHA-256 | `b85ae888bf747cefa54bba2a6c3e2f6ccb4c1005d4e0b6d1d3be3823cf040aef` |
| Checkpoint SHA-256 | `c972037e051773afb505c14746daafaadc827e36d57588269b86fccbc56ee7c6` |
| 机器可读证据 | `reports/esconv_joint_official_corrected_full_20260811.json` |

### 关于此前的 19.60%：已经撤回，不是“尚待解释”

`reports/esconv_original_joint_official_test_20260810.json` 已明确标记 `audit_status = RETRACTED`。撤回原因是：评测器使用了错误的策略 token 拼写/顺序，并用错误的标签映射解释 checkpoint embedding rows；其中 Accuracy、Macro-F1、Weighted-F1、逐类指标和混淆矩阵全部失效。替代评测器为 `scripts/eval_esconv_joint_corrected.py`。

因此正式口径只有一个：

- 撤回值：`544 / 2775 = 19.6036%`；不得继续引用。
- 修正替代值：`894 / 2775 = 32.2162%`。
- 修正 Macro-F1：`21.8272%`。
- 修正 Weighted-F1：`28.5626%`。

这仍然是本仓库对作者 checkpoint 的后验重放，不是 ACL 2021 论文原生公布的 Strategy ACC。

## 七、不能放入 8 类 Strategy-ACC 排名的数字

### 1. KEMP 39.31%：任务被误配

Causal-ESC Table 2 把 KEMP 39.31 写进 ACC 对照，但 KEMP 原论文 `Knowledge Bridging for Empathetic Dialogue Generation` 的 39.31 是 **EmpatheticDialogues 情绪分类准确率**。原论文不是 ESConv 下一策略预测论文，也没有提供一个公开的 ESConv 8 类重跑。因此本 benchmark 不把 KEMP 39.31 排入榜单。

### 2. 三阶段 86.4%：不是 8 类策略

`Improving Next Stage Prediction in Multi-turn Emotional Support Conversations via Multi-task Learning` 的 86.4% 是 Exploration / Comforting / Action 三阶段分类。类别数、目标和难度均不同，不可写成 ESConv Strategy ACC。

### 3. EmoDynamiX：没有报告 Accuracy

EmoDynamiX 在 ESConv 上报告 Macro-F1 **27.70**、Weighted-F1 **32.71**、Preference Bias **0.45**，没有报告 Strategy ACC。任何把 DailyDialog 情绪识别 Accuracy 或自行猜测值填给 EmoDynamiX 的做法都不正确。

### 4. ACL 2021 原论文：没有 Strategy ACC

原论文只报告生成指标，例如 BlenderBot Joint 的 BLEU-2 5.35、ROUGE-L 15.46、BOW Extrema 50.27；Strategy ACC 是本项目后来用 checkpoint 复算的，不应倒写回 2021 Table 4。

### 5. ESCA 的 47.44%：是 SR@10，不是 Strategy ACC

AAAI 2026 的 ESCA 报告交互式任务目标成功率 `SR@10 = 47.44%` 和平均轮数 `AT = 8.96`。这衡量的是最多 10 轮内是否完成情绪支持目标及交互效率，不是单个 supporter turn 的八类策略 exact-match Accuracy，因此不能插入 58.03/53.53/46.80 的 Strategy-ACC 排名。

### 6. 多策略单轮任务：目标结构已经改变

2026 年的 multi-strategy formulation 允许一个 supporter turn 中连续出现多个 strategy–response pairs。它预测的是策略序列/片段，而不是经典单标签下一策略，必须另建 track。

## 八、Google Scholar 链接

Google Scholar 的 `Cited by` 数会变化，而且本次自动访问触发限流，所以没有拿 Semantic Scholar、Scopus、PWC 或其他网站的数字冒充 Scholar cited-by。下面全部是**精确题名查询链接**；它们用于定位论文和 citation graph，**不是 ACC 数字本身的证据**。ACC 的证据仍然是原论文表或官方出版社摘要。需要引用数时，应手动打开并记录日期、结果条目和 `Cited by` URL。

| 模型/论文 | Google Scholar 精确题名查询 | 当前 cited-by 字段 |
|---|---|---|
| DPPLM | [Google Scholar](https://scholar.google.com/scholar?q=%22Applying+Emotion-Cause+Entailment+for+Help-Seeker+Guidance+in+Emotional+Support+Conversations%22) | `null`（动态，2026-08-19 未冻结） |
| Causal-ESC | [Google Scholar](https://scholar.google.com/scholar?q=%22Causal-ESC%3A+Reliable+Policy+Learning+for+Emotional+Support+Conversation+via+Causal+Inference%22) | `null`（动态，2026-08-19 未冻结） |
| SAGE | [Google Scholar](https://scholar.google.com/scholar?q=%22SAGE%3A+Self-retrieval-augmented+generative+LLM+for+emotional+support+conversation%22) | `null`（动态，2026-08-19 未冻结） |
| CADSS | [Google Scholar](https://scholar.google.com/scholar?q=%22Simulating+Human-Like+Counseling%3A+A+Path-+and+Scenario-Guided+Framework+for+Psychological+Support+Dialogue%22) | `null`（动态，2026-08-19 未冻结） |
| MultiESC | [Google Scholar](https://scholar.google.com/scholar?q=%22Improving+Multi-turn+Emotional+Support+Dialogue+Generation+with+Lookahead+Strategy+Planning%22) | `null`（动态，2026-08-19 未冻结） |
| PRCCF | [Google Scholar](https://scholar.google.com/scholar?q=%22PRCCF%3A+A+Persona-guided+Retrieval+and+Causal-aware+Cognitive+Filtering+Framework+for+Emotional+Support+Conversation%22) | `null`（动态，2026-08-19 未冻结） |
| GREEN | [Google Scholar](https://scholar.google.com/scholar?q=%22GREEN%3A+Generative+Retrieval-Enhanced+Emotional+Support+Conversations%22) | `null`（动态，2026-08-19 未冻结） |
| Hao–Kong framework | [Google Scholar](https://scholar.google.com/scholar?q=%22Enhancing+Emotional+Support+Conversations%3A+A+Framework+for+Dynamic+Knowledge+Filtering+and+Persona+Extraction%22) | `null`（动态，2026-08-19 未冻结） |
| D²RCU | [Google Scholar](https://scholar.google.com/scholar?q=%22Dynamic+Demonstration+Retrieval+and+Cognitive+Understanding+for+Emotional+Support+Conversation%22) | `null`（动态，2026-08-19 未冻结） |
| TransESC | [Google Scholar](https://scholar.google.com/scholar?q=%22TransESC%3A+Smoothing+Emotional+Support+Conversation+via+Turn-Level+State+Transition%22) | `null`（动态，2026-08-19 未冻结） |
| PAL | [Google Scholar](https://scholar.google.com/scholar?q=%22PAL%3A+Persona-Augmented+Emotional+Support+Conversation+Generation%22) | `null`（动态，2026-08-19 未冻结） |
| CauESC | [Google Scholar](https://scholar.google.com/scholar?q=%22CauESC%3A+A+Causal+Aware+Model+for+Emotional+Support+Conversation%22) | `null`（动态，2026-08-19 未冻结） |
| FADO | [Google Scholar](https://scholar.google.com/scholar?q=%22FADO%3A+Feedback-Aware+Double+COntrolling+Network+for+Emotional+Support+Conversation%22) | `null`（动态，2026-08-19 未冻结） |
| MISC | [Google Scholar](https://scholar.google.com/scholar?q=%22MISC%3A+A+Mixed+Strategy-Aware+Model+integrating+COMET+for+Emotional+Support+Conversation%22) | `null`（动态，2026-08-19 未冻结） |
| CARE | [Google Scholar](https://scholar.google.com/scholar?q=%22CARE%3A+Cognitive-reasoning+Augmented+Reinforcement+for+Emotional+Support+Conversation%22) | `null`（动态，2026-08-19 未冻结） |
| ESCA（非 ACC track） | [Google Scholar](https://scholar.google.com/scholar?q=%22ESCA%3A+An+Emotional+Support+Conversation+Agent+for+Enhancing+Reasonable+Strategy+Planning+and+Effective+Expression%22) | `null`（动态，2026-08-19 未冻结） |
| BlenderBot-small Joint (author checkpoint replay in this repository) | [Google Scholar](https://scholar.google.com/scholar?q=%22Towards+Emotional+Support+Dialog+Systems%22) | `null`（动态，2026-08-19 未冻结） |
| EmoDynamiX | [Google Scholar](https://scholar.google.com/scholar?q=%22EmoDynamiX%3A+Emotional+Support+Dialogue+Strategy+Prediction+by+Modelling+MiXed+Emotions+and+Discourse+Dynamics%22) | `null`（动态，2026-08-19 未冻结） |

## 九、Papers With Code 链接与限制

这里必须区分两个不同站点：

1. **原始 `paperswithcode.com`**：原来的 Meta Papers With Code benchmark/leaderboard 服务在 2025 年 7 月停止运行，后来旧域名转向其他页面。它留下的论文页与数据归档只能作为历史索引。
2. **用户给出的 `paperswithcode.co`**：截至 2026-08-19 目前可以打开，首页和 paper archive 会收录 2026 论文；但本次检查到的页面没有说明它是原 Meta Papers With Code 的官方继承者，也没有找到可核验的 ESConv Strategy-ACC leaderboard 或上述 ESConv 论文的精确条目。因此不能拿 `.co` 的存在替代原论文 ACC 证据。

当前可核查入口：

- [`paperswithcode.co` 首页](https://paperswithcode.co/)
- [`paperswithcode.co` Paper Archive](https://paperswithcode.co/papers/archive)
- [原始 Papers With Code 历史数据归档](https://github.com/paperswithcode/paperswithcode-data)

结论：**Papers With Code 类站点用于找论文/代码；ACC 是否正确仍必须回到原论文表或官方出版社记录。** 本次 `.co` 检索没有发现能证明 DPPLM 58.03、Causal-ESC 53.53 或 SAGE 46.80 的独立 leaderboard 页面，所以 registry 对这些条目的 `paperswithcode_co_url` 保持 `null`。

| 论文 | 原始 `.com` 历史 PWC 证据 | 当前 `.co` 核查 |
|---|---|---|
| DPPLM | —（发布时间接近原服务停止） | 未找到精确条目/ESConv leaderboard |
| Causal-ESC | —（2026 发表） | 未找到精确条目/ESConv leaderboard |
| SAGE | —（2026 发表） | 未找到精确条目/ESConv leaderboard |
| CADSS | —（2026 发表） | 未找到精确条目/ESConv leaderboard |
| MultiESC | [历史作者索引](https://paperswithcode.com/author/yefeng-zheng) | 未找到可核验的精确条目 |
| PRCCF | —（2026 preprint） | 未找到精确条目/ESConv leaderboard |
| GREEN | — | 未找到精确条目/ESConv leaderboard |
| Hao–Kong framework | — | 未找到精确条目/ESConv leaderboard |
| D²RCU | — | 未找到精确条目/ESConv leaderboard |
| TransESC | [历史作者索引](https://paperswithcode.com/author/weixiang-zhao) | 未找到可核验的精确条目 |
| PAL | [历史作者索引](https://paperswithcode.com/author/minlie-huang) | 未找到可核验的精确条目 |
| CauESC | — | 未找到精确条目/ESConv leaderboard |
| FADO | [历史论文页](https://paperswithcode.com/paper/fado-feedback-aware-double-controlling) | 未找到可核验的精确条目 |
| MISC | [历史作者索引](https://paperswithcode.com/author/quan-tu) | 未找到可核验的精确条目 |
| CARE | —（原服务停止后提交） | 未找到精确条目/ESConv leaderboard |
| ESConv 2021 | [历史论文页](https://paperswithcode.com/paper/towards-emotional-support-dialog-systems) | 未找到可核验的精确条目；历史页本身也没有 Strategy-ACC 结果 |

## 十、代码和 checkpoint 可用性

| 模型 | 代码状态 |
|---|---|
| DPPLM | not found |
| Causal-ESC | [https://github.com/zze-00/Causal-ESC](https://github.com/zze-00/Causal-ESC) — public repository found, but release identity/checkpoint equivalence to the ACL paper is not sufficiently documented |
| SAGE | not found; publisher says data available on request |
| CADSS | [https://github.com/FakerBoom/CPsDD](https://github.com/FakerBoom/CPsDD) — repository public; README says CADSS/PGSim code will be released later |
| MultiESC | [https://github.com/lwgkzl/MultiESC](https://github.com/lwgkzl/MultiESC) — public code/data; no released trained checkpoint found |
| PRCCF | [https://github.com/YancyLyx/PRCCF](https://github.com/YancyLyx/PRCCF) — paper claims public code; repository returned 404 during the 2026-08-19 audit |
| GREEN | not found |
| Hao–Kong framework | not found |
| D²RCU | [https://github.com/Bat-Reality/DDRCU](https://github.com/Bat-Reality/DDRCU) — public code link stated in the paper |
| TransESC | paper says source code would be released; no checkpoint used in this audit |
| PAL | [https://github.com/chengjl19/PAL](https://github.com/chengjl19/PAL) — public code and data |
| CauESC | not found |
| FADO | [https://github.com/Thedatababbler/FADO](https://github.com/Thedatababbler/FADO) — paper states code is available; historical PWC page lists no implementation entry |
| MISC | [https://github.com/morecry/MISC](https://github.com/morecry/MISC) — public code |
| CARE | no verified official code release used in this audit |
| ESCA | [official AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/38807); its SR@10 belongs to a separate interactive track |
| EmoDynamiX | [https://github.com/cw-wan/EmoDynamiX-v2](https://github.com/cw-wan/EmoDynamiX-v2) — 代码、ESConv checkpoint 下载项和测试脚本公开；需要在 frozen 2,775 上自行测 ACC |

## 十一、正式 benchmark 应该怎么保存

本项目以后应固定为两个榜，禁止混用：

### A. `paper_reported_protocol_specific`

保存论文自报值、年份、venue、表格位置、数据版本、标签空间、split、代码/权重状态和证据等级。该榜用于文献综述，不能宣称是本项目统一 SOTA。

### B. `frozen_2775_reproduced`

只有满足以下条件才能进入：

1. 完全相同的 2,775 个 task IDs；
2. 原始 8 类标签与冻结映射；
3. 当前 target strategy/response 不进入输入；
4. 同一 exact-match scorer；
5. 保存逐条预测、checkpoint hash、代码 commit 和环境；
6. 报 ACC、Macro-F1、Weighted-F1、逐类 F1、混淆矩阵；
7. 按 dialogue cluster 做 bootstrap 95% CI；
8. 模型间做配对 McNemar，而不是只比较两个百分数。

## 十二、现在可以下的结论

- **截至 2026-08-19，本次审计找到的最高论文自报 headline：DPPLM 58.03%，证据 B。**
- **最高完整原论文表核实值：Causal-ESC 53.53%，证据 A，但协议特殊。**
- **最高明确 8 类且同时给 Macro-F1 的正式期刊结果：SAGE 46.80% ACC / 38.95% Macro-F1。**
- **当前本仓库机器可读 frozen replay：32.22% ACC，不是论文成绩；旧 19.60% 已明确撤回。**
- **KEMP 39.31、三阶段 86.4、EmoDynamiX 的非 ACC 指标均不得混入 8 类 Strategy-ACC 排名。**

机器可读版本：`reports/esconv_benchmark_registry_20260819.json`。该 JSON 对每项保存年份、venue、ACC/F1/生成指标、原始证据 URL、Google Scholar 精确题名查询、历史 PWC 状态、代码状态和协议可比性。
