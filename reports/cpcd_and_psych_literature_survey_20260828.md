# CPCD 生态普查 + 心理学相关论文/数据集普查

日期：2026-08-28（loop tick 3）
方法：paperswithcode.co `/api/v1/papers/search` API（5 组查询去重 118 篇）+ GitHub 搜索 API 交叉验证。

---

## 1. 结论先行：CPCD 没有 ESConv 那样的生态，一点都没有

用户问"CPCD 有没有像 ESConv 这么大体量的改进论文和代码"。**答案是否定的，而且差距是数量级的。**

| | ESConv | CPCD（Psy-Chronicle） |
|---|---|---|
| 原论文引用 | **482** | — |
| GitHub 星标 | thu-coai 官方仓库，生态活跃 | **★4，0 fork** |
| 检索到的相关论文 | 53 篇（179 去重后过滤） | **标题含 Psy-Chronicle / CPCD 的：0 篇** |
| 有官方实现的改进工作 | 26 篇 | **0** |
| 可辨认的方法谱系 | MISC → GLHG → MultiESC → TransESC → PAL → CauESC → EmoDynamiX → … | 无 |

**Psy-Chronicle 是一个 2026 年的新合成基准（校园心理咨询长程对话），目前还没有任何人在它上面做改进工作。**

### 这对我们的策略意味着什么

- **"找到最好的模型然后改进它"这条路只在 ESConv 上成立。** CPCD 上没有可改进的对象，它只能作为**评测**用途（我们已经在这么用）。
- CPCD 的价值在于它测的是 ESConv 测不了的东西：`mr`（记忆回溯）和 `tcr`（时序因果回溯）—— 长程记忆能力。**这块的可借鉴工作不在"CPCD 改进论文"里，而在通用长程对话记忆文献里**（见 §4）。
- 别在 CPCD 上找"SOTA 对手"，找不到。

---

## 2. 【最高优先级】两篇必读，与静室直接相关

### 2.1 Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers

**227 引，2025-04-25，有官方实现**，arXiv 2504.18412

这是本次普查里引用最高的心理健康专项论文，而且论点直指静室在做的事：**LLM 会表达病耻感（stigma）并给出不当回应，因此不能安全替代人类心理服务提供者**。

**必须读，而且必须在部署门里体现。** 我们现有的部署门有 `mental-health-ai-safety-engine` 的 10,000 案例，但那是我们自己造的；这篇是同行评议的、被引 227 次的失效模式清单。用它做我们安全门的外部校准。

### 2.2 SoulChat: Improving LLMs' Empathy, Listening, and Comfort Abilities through Fine-tuning with Multi-turn Empathy Conversations

**185 引，2023-11-01**，arXiv 2311.00273

**中文**心理健康 LLM，做法就是我们要做的事：用多轮共情对话微调出共情/倾听/安慰能力。静室是中英双语、中文为主，而我们的 research-v0 训练集有 300 条中文 CPCD——**SoulChat 是最贴近的中文先例，必须对照它的数据构造与评测方式。**

### 2.3 The opportunities and risks of large language models in mental health

239 引，2024-03-21，arXiv 2403.14814 —— 综述，用作风险清单的框架。

---

## 3. 心理咨询数据集（潜在训练数据）

| 数据集/方法 | 日期 | 引用 | 代码 | 备注 |
|---|---|---:|:--:|---|
| **KokoroChat** 日语心理咨询对话数据集 | 2025-06-02 | 10 | — | 真人收集，非合成；语言不匹配但构造方法可借鉴 |
| **OnCoCo 1.0** 在线咨询细粒度消息分类公开数据集 | 2025-12-10 | 1 | — | 有标注体系，可能补我们的策略标签 |
| **AnnaAgent** 多会话记忆的动态演化 agent | 2025-05-31 | 23 | ✅ | 生成真实求助者模拟 |
| **MCTSr-Zero** 自反思心理咨询对话生成 | 2025-05-29 | 4 | ✅ | MCTS 合成，注意合成数据的风险 |
| **Graph2Counsel** 临床接地的合成咨询对话生成 | 2026-04-22 | 2 | — | 声称"临床接地"，需核实其接地方式 |
| **MAGneT** 多智能体协同生成多轮心理健康对话 | 2025-09-04 | 7 | — | 合成 |
| Roleplaying with Structure 结构化治疗师-来访者对话生成 | 2025-10-29 | 4 | — | 合成 |
| **AntEngage** 共情对话数据集（HF） | 2026-08-04 | 0 | ✅ | 待查许可与规模 |

⚠ **合成数据警告**：上表半数是 LLM 合成的。我们自己的红线已经写死——`mental-health-instruction-dataset-v2` 要求真人金标，`MentalChat16K` 合成数据已因危机回复安全抽查失败被隔离。**这些合成数据集在采纳前必须走同一道危机安全抽查**，不能因为"是论文发的"就免检。

---

## 4. 长程记忆文献（对应 CPCD 的 mr / tcr，也对应静室的记忆需求）

CPCD 没有改进生态，但它测的记忆能力有成熟的独立文献，且几乎全都有代码：

| 论文 | 日期 | 引用 | 代码 |
|---|---|---:|:--:|
| **LoCoMo**: Evaluating Very Long-Term Conversational Memory of LLM Agents | 2024-02-27 | **756** | ✅ |
| **MemoryBank**: Enhancing LLMs with Long-Term Memory | 2023-05-17 | **622** | ✅ |
| **LongMemEval**: Benchmarking Chat Assistants on Long-Term Interactive Memory | 2024-10-14 | **543** | ✅ |
| SGMem: Sentence Graph Memory | 2025-09-25 | 28 | — |
| Hindsight is 20/20: Agent Memory that Retains, Recalls, Reflects | 2025-12-14 | 24 | — |
| REMem: Reasoning with Episodic Memory | 2026-02-13 | 17 | ✅ |
| A Simple Yet Strong Baseline for Long-Term Conversational Memory | 2025-11-21 | 15 | ✅ |
| EviMem / APEX-MEM / AssoMem / MemUse / Locomo-Plus | 2025–2026 | 各 2–9 | 部分 ✅ |

**这三篇高引（LoCoMo / MemoryBank / LongMemEval）都有代码，是提升 CPCD mr/tcr 分数的正确入口**——比在 CPCD 上瞎找对手有效得多。

---

## 5. 心理健康评测与安全基准

| 基准 | 日期 | 引用 | 关注点 |
|---|---|---:|---|
| **CounselBench** 大规模专家评测 + 对抗基准 | 2025-06-10 | 19 | 专家评测，可校准我们的 proxy judge |
| **MHSafeEval** 角色感知的交互级心理健康安全评测 | 2026-04-20 | 2 | **与静室危机管线直接同构** |
| **PsychEval** 多会话多疗法高真实度基准 | 2026-01-05 | 1 | 多疗法覆盖 |
| **Ψ-Arena** LLM 心理咨询师交互式评估与优化 | 2025-05-06 | 8 | 三方反馈 |
| Measuring What Matters 治疗原则评估 | 2026-04-07 | 0 | 治疗原则而非表面共情 |
| When AI Takes the Couch 心理测量越狱 | 2025-12-02 | 3 | 攻击面 |
| **TheraMind** 长程心理咨询的策略自适应 agent | 2025-10-29 | 8 | 方法参考 |
| Beyond Empathy 诊断+治疗推理 | 2025-05-21 | 22 | 方法参考 |
| **Psyche-R1** 统一共情+专业+推理的心理 LLM | 2025-08-14 | 22 | 方法参考，与 Kardia-R1 同代 |

---

## 6. ESConv 侧本轮补充：代码仓库定位结果

| 论文 | 仓库 | 星标 | 状态 |
|---|---|---:|---|
| **Kardia-R1**（rubric-as-judge RL） | `JhCircle/Kardia-R1` | ★60 | ✅ 已定位，[WWW] 会议 |
| **TEA-Bench** | `XingYuSSS/TEA-Bench` | ★12 | ✅ 已定位，[ACL 2026] |
| **ESC-Eval** | `haidequanbu/ESC-Eval` | ★27 | ✅ 已定位，[EMNLP 2024] |
| **MISC**（157 引） | `morecry/MISC` | ★39 | ✅ 已定位，ACL 2022 |
| MultiESC | `lwgkzl/MultiESC` | ★51 | 已在收藏中 |
| MentalBench | `HoyunS/MentalBench` ★22 / `abeerbadawi/MentalBench-Align` ★10 | | ⚠ 需确认哪个对应 arXiv 2510.19032 |
| GLHG（114 引） | — | | ❌ 关键词搜不到，下轮换作者名搜 |
| KEMI（82 引） | — | | ❌ 同上（作者 Deng et al.） |
| ESC-Skills / 单轮多策略 | — | | ❌ 未找到；arXiv abs 页抓到的 `aliyun/qwen-dianjin` 是侧栏误报，已排除 |

---

## 7. 下一步

1. 读 **stigma 论文（227 引）**，把它的失效模式并入部署安全门
2. 读 **SoulChat**，对照其中文数据构造与评测
3. 取 **Kardia-R1** 代码，评估其 rubric-as-judge RL 能否直接复用我们现有的 proxy-judge 管线做奖励模型
4. GLHG / KEMI / ESC-Skills / 单轮多策略 换作者名再搜一轮
5. 查 **AntEngage** 与 **OnCoCo** 的许可与规模
6. **LoCoMo / MemoryBank / LongMemEval** 作为 CPCD mr/tcr 的改进入口评估
