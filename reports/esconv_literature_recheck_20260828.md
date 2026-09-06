# ESConv 文献重查（第 1 轮）

日期：2026-08-28
目标：在 `~/Desktop/Esconv 改进论文和代码/` 现有 7 篇之外，找出遗漏的改进论文与可用代码，重排最高准确率榜。
状态：**进行中**。本轮为搜索引擎轨；paperswithcode.co 正面抓取被挡，下轮用浏览器工具重试。

---

## 0. 本轮同时完成的实测（不是文献，是新数字）

**Qwen3.8-27B 零样本 ESConv fixed-250 = 13.60%**（0 missing / 0 invalid）

| 臂 | Accuracy | 95% CI | Macro-F1 | missing/invalid |
|---|---:|---:|---:|---:|
| DeepSeek V4-Flash | 16.40% | [12.08%, 21.08%] | 12.26% | 3/4 |
| DeepSeek V4-Pro-0813 | 16.00% | [12.64%, 19.92%] | 9.64% | 10/12 |
| Qwen3.6-27B（未微调） | 14.00% | [9.66%, 19.15%] | 10.44% | **0/0** |
| **Qwen3.8-27B（未微调）** | **13.60%** | [9.20%, 18.62%] | 9.60% | **0/0** |

六组两两比较 **Holm p 全为 1**，即四臂之间无一显著。协议：`jingshi-esconv-250-20260828-dashscope-qwen38-addendum-v1`；报告：`reports/esconv_fixed250_qwen38_pretest_20260828.{md,json}`。

**这坐实了立项判断**：换哪个 API 模型都在 13–16%，而工程轨专用小模型是 32–34%。差距不在模型大小，在**有没有针对这个任务训练过**。

---

## 1. paperswithcode.co 抓取状态 —— 【第 2 轮已攻克，结论确定】

**403 的原因是 UA 拦截**，不是需要登录。带浏览器 UA 的 curl 直接 200。站点是重建版 PwC，前端渲染，但有 JSON API：

```
GET https://paperswithcode.co/api/v1/papers/search?q=<query>&page=<n>
    → {next_page, results:[{id,title,arxiv_id,published,citation_count,
                            has_official_implementation,code_repository_count,url_abs}]}
GET https://paperswithcode.co/api/v1/tasks/<slug>
GET https://paperswithcode.co/api/v1/papers/<id>     # 只回元数据，不含仓库地址
```

### ✅ 确认：站上没有 ESConv 排行榜

探测 5 个候选 slug —— `emotional-support-conversation` / `emotional-support-dialogue` / `esconv` / `emotional-support` / `dialogue-generation` —— **全部 404**。

**所以"去 PwC 抄榜"这条路在方法上是走不通的，不是我们没找到。** `~/Desktop/数据集准确率对比_修正版_20260821.docx` 的判断得到独立复核。可用排名只能靠：逐篇读原文抽口径 + 自己复跑。这也再次印证现有 README 的原则——**论文报的分不算数**。

### 用 API 做的系统普查

5 组查询（ESConv / emotional support conversation / emotional support dialogue / emotional support dialog / support strategy prediction）去重后 **179 篇**，按标题关键词过滤出 **53 篇相关**，其中 **26 篇有官方实现**。原始数据：`tmp/pwc_esconv_papers.json`。

---

## 1bis. 【最重要的发现】这个领域已经分成两个时代，你的收藏集中在旧的那个

| | 第一代（2021–2023） | 第二代（2025–2026） |
|---|---|---|
| 做法 | 小模型（BlenderBot / BART / DialoGPT）+ 策略分类头 | LLM + **RL，用 rubric 判官当奖励** |
| 报的指标 | 八类策略 ACC（32–34%） | rubric 分 / 判官偏好 / 人评，**基本不报八类 ACC** |
| 代表 | ESConv Joint、MISC、GLHG、MultiESC、TransESC、PAL、KEMI | **Kardia-R1**、CARE、Psyche-R1、Value Reinforcement、开放式 RL |
| 你现有收藏 | ✅ 覆盖得不错 | ❌ **一篇都没有** |

**这件事直接影响我们的目标选型。** 静室要替换的是 DeepSeek API，做的是**回复生成**，不是策略分类——八类 ACC 只是代理指标。第二代那条线才是我们真正要做的东西。

而且有个现成的杠杆：**Kardia-R1 的 "rubric-as-judge RL" 与我们已经建好的 proxy-judge 评测管线是同一套结构**。现有 `open_response_eval/` 的盲评判官稍加改造就能当 RL 的奖励模型，不用另起炉灶。

---

## 1ter. 26 篇有官方实现的相关论文（可复跑候选池）

### 第二代 / LLM+RL（你的收藏里全部缺失，优先补）

| 论文 | 日期 | 引用 | arXiv |
|---|---|---:|---|
| **Kardia-R1**: Rubric-as-Judge 强化学习做共情推理 | 2025-12-01 | 24 | 2512.01282 |
| **ESC-Skills**: 自演化技能发现 | 2026-05-27 | 0 | 2605.27908 |
| **Modeling Multiple Support Strategies within a Single Turn** | 2026-04-20 | 0 | 2604.17972 |
| **TEA-Bench**: 工具增强 ESC 智能体评测 | 2026-01-26 | 3 | 2601.18700 |
| **When Can We Trust LLMs in Mental Health?** 大规模可靠性基准 | 2025-10-21 | 28 | 2510.19032 |
| **Dialogue Systems for ESC via Value Reinforcement** | 2025-01-25 | 13 | 2501.17182 |
| **ESC-Eval**: LLM 的 ESC 能力评测 | 2024-06-21 | 20 | 2406.14952 |
| **ESCoT**: 可解释 ESC（CoT） | 2024-06-16 | 70 | 2406.10960 |
| **FEEL**: 用 LLM 评估情感支持能力 | 2024-03-23 | 5 | 2403.15699 |
| Steering Conversational LLMs for Long ESC | 2024-02-16 | 10 | 2402.10453 |

### 第一代经典（你的收藏也缺这几篇高引的）

| 论文 | 日期 | 引用 | arXiv |
|---|---|---:|---|
| **MISC**: 混合策略感知 + COMET | 2022-03-25 | **157** | 2203.13560 |
| **GLHG**（Control Globally, Understand Locally）全局到局部层次图 | 2022-04-27 | **114** | 2204.12749 |
| **KEMI**: 知识增强混合主动 | 2023-05-17 | 82 | 2305.10172 |
| MultiESC（Lookahead Strategy Planning） | 2022-10-09 | 94 | 2210.04242 |
| TransESC | 2023-05-05 | 53 | 2305.03296 |
| PAL | 2022-12-19 | 67 | 2212.09235 |
| ESConv 原论文 | 2021-06-02 | **482** | 2106.01144 |
| AugESC | 2022-02-26 | 125 | 2202.13047 |

### 无官方实现但 2025-06 后的新工作（只能读方法）

CARE（认知推理增强 RL，2025-09-30）、**Psyche-R1**（统一共情+专业+推理，2025-08-14，22 引）、Towards Open-Ended ESC via RL（2025-08-18）、STRIDE-ED（2026-04-08）、ES-MemEval（2026-02-02）、EmoHarbor（2026-01-04）、User-Aware Active Knowledge Acquisition（2026-05-28）、Discourse Diversity in Multi-Turn Empathic Dialogue（2026-04-13）、EMPA（2026-02-28）、检测情绪动态轨迹评测框架（2025-11-12）。

### 潜在训练数据

- **AntEngage Empathy Conversation Dataset**（2026-08-04，HF `antengage/empathy-conversations`）—— 新共情对话数据集，需查许可与规模
- AugESC（已在用）、EmpatheticDialogues（已审查未采用）

---

## 2. 本轮新发现（现有收藏中**没有**的）

| 论文 | 出处 | 报告的策略指标 | 代码 | 为什么值得看 |
|---|---|---|---|---|
| **MAGO**：Multi-Knowledge Aware and Global Strategy Sequence Optimizing Network | Neurocomputing 2025, vol 618, 128888 | 策略准确率 **"提升 18%"（相对值，非绝对值）** | 未找到公开仓库 | **方法上最相关**：用 Strategy-Constrained CRF 做**全局策略序列**估计，而非逐轮独立分类；同时融合 commonsense + conceptual facts。同组的 FADO 是开源的 |
| **FADO**：Feedback-Aware Double Controlling Network | Knowledge-Based Systems 2023, vol 264, 110340 | 自报 ESConv 策略选择 SOTA | ✅ **有公开代码** | MAGO 同组前作，可作为 MAGO 思路的可运行替身 |
| Modeling Multiple Support Strategies within a Single Turn | arXiv 2604.17972 | — | 未查 | **可能动摇整个八类协议**：指出 ESConv 中同一轮连续使用多个策略很常见（最多观察到 7 个），单标签设定本身有问题 |
| Dynamic Demonstration Retrieval and Cognitive Understanding (DDRCU) | arXiv 2404.02505 | — | 未查 | 检索式示范 + 认知理解，与 RAG 路线可复用 |
| Memory-Enhanced ESC with Motivation-Driven Strategy Inference | ECML-PKDD 2024 | — | 未查 | 记忆机制，与静室的长期记忆需求同构 |
| Multi-level semantics-aware & multi-granularity knowledge-infused | Information Fusion 2025, S1566253525010309 | — | 未查 | 多粒度知识注入 |
| Affective Flow Language Model | arXiv 2602.08826 | — | 未查 | 情感流建模 |
| Knowledge-enhanced Memory Model | arXiv 2310.07700 | — | 未查 | 早期记忆增强 |

### ⚠ 一个必须挡住的陷阱

搜索结果里有一篇多任务 DialoGPT 报 **86.4% accuracy / weighted-F1 0.86**，看起来碾压一切。**它做的是"下一阶段(stage)预测"，不是八类策略分类**——stage 是更粗的标签体系。这与 README 里 MultiESC 42.01%（实为约 7 类）是同一类问题：**不同输出空间的数字不能进同一张榜**。已排除。

---

## 3. 与现有收藏的合并视图（文献轨，非同协议排名）

| 模型 | 出处 | 自报 ACC | 代码 | 现状 |
|---|---|---:|---|---|
| AFlow | arXiv 2026 | 65.10% | 有码无权重 | ⚠ 合成扩展协议，非原始 test；已排除出主榜 |
| DPPLM | 2025 NCKU | 58.03% | 无 | 不可复跑 |
| Causal-ESC | ACL 2026 Long | 53.53% | 无 | 不可复跑 |
| **SAGE** | ESWA 2026 | **46.80%** | 无 | 不可复跑；self-retrieval + trie 约束解码 |
| CADSS | AAAI 2026 | 46.26% | 只有 PDF | 不可复跑 |
| MultiESC | EMNLP 2022 | 42.01%* | 有码无 ckpt | *约 7 类，口径不符 |
| PRCCF | arXiv 2026 | 40.72% | 仓库不可用 | 不可复跑 |
| GREEN | SAGE Open 2025 | 38.80% | 无权重 | 不可复跑 |
| Hao–Kong (DKPE) | COLING 2025 | 35.51% | 无权重 | 不可复跑 |
| TransESC | ACL Findings 2023 | 34.71% | ✅ 有码 | **待复跑·第 2 优先** |
| PAL | ACL Findings 2023 | 34.51% | ✅ 有码 | 用 PESConv 增强，协议不符 |
| **EmoDynamiX** | NAACL 2025 | 论文未报 | ✅ 有码 | ✅ **本机实测 33.61%**，当前改进基座 |
| CauESC | arXiv 2024 | 33.33% | ✅ 有码 | **待复跑·第 1 优先** |
| ESConv Joint | ACL 2021 | — | ✅ 有码 | ✅ **本机实测 32.22%** |
| **MAGO** | Neurocomputing 2025 | "+18%"相对 | ❌ 未找到 | 🆕 本轮新增，方法最相关 |
| **FADO** | KBS 2023 | 自报 SOTA | ✅ **有码** | 🆕 本轮新增，MAGO 同组前作 |

**可实证的最高分仍然是本机复现的 EmoDynamiX 33.61%。** 46%–65% 那一档全部没有可运行实现，属于"论文说的"，不是"我们验证过的"。

---

## 4. 下一步（按价值排序，第 2 轮后更新）

1. **补齐第二代论文的代码**——PwC 的 detail API 不给仓库地址，直接从 GitHub 找。优先 **Kardia-R1**（rubric-as-judge RL，与我们现有判官管线同构）、**Modeling Multiple Support Strategies**（单轮多策略）、**ESC-Skills**
2. **MISC / GLHG / KEMI 三篇高引第一代补进收藏**（157/114/82 引，都有官方实现，且都是八类协议）——现有复跑计划只排了 CauESC 和 TransESC，漏了这三篇
3. 逐篇核新论文的**输出空间与 test 切分**（是否真八类、是否原始 test），并入总榜
4. 按现有 README 推进 CauESC → TransESC 复跑
5. **重新审视八类单标签假设**（2604.17972）——若同一轮多策略确实普遍，改进方向可能应是多标签
6. 查 **AntEngage** 数据集的许可与规模，评估能否作训练数据

## 5. 待办（尚未开始）

- CPCD 的改进论文与代码普查（用同一套 PwC API 方法）
- 其他心理学数据集/论文普查
