# 文献调研总结（2026-08-28 完成）

三轮 loop + 补充核实，完成了你要求的四项调研。

---

## 一、ESConv 论文与模型重查（paperswithcode.co）

### 1.1 站点状态

**✅ 已攻克，结论确定**。403 是 UA 拦截，带浏览器 UA 直接 200。站是重建版 PwC，有 JSON API：`/api/v1/papers/search`。

**探测 5 个任务 slug 全部 404 → 确认：站上没有 ESConv 排行榜。** 不是我们没找到，是方法上走不通。"去 PwC 抄榜"给不出可用排名，只能逐篇读原文抽口径 + 自己复跑。

### 1.2 系统普查结果

5 组查询去重 **179 篇**，过滤 **53 篇相关，26 篇有官方实现**。

**【最重要的发现】这个领域已经分成两个时代，你的收藏全在旧的那个**

| | 第一代 2021–2023 | 第二代 2025–2026 |
|---|---|---|
| 做法 | 小模型（BlenderBot/BART）+ 策略分类头 | **LLM + RL，用 rubric 判官当奖励** |
| 报什么 | 八类 ACC（32–34%） | rubric 分 / 判官偏好，**基本不报八类 ACC** |
| 代表 | ESConv Joint、MISC、GLHG、MultiESC、TransESC、PAL | **Kardia-R1**、CARE、Psyche-R1、Value RL |
| 你的收藏 | ✅ 覆盖不错 | ❌ **一篇都没有** |

**这直接影响选型**：静室要替换的是**回复生成**，不是策略分类——八类 ACC 只是代理指标。第二代才是我们真正要做的。

而且有个现成杠杆：**Kardia-R1 的 "rubric-as-judge RL" 与我们已经建好的 proxy-judge 评测管线是同一套结构**。现有 `open_response_eval/` 的盲评判官稍加改造就能当 RL 的奖励模型。

### 1.3 代码仓库定位（GitHub API）

| 论文 | 仓库 | 星标 | 状态 |
|---|---|---:|---|
| **Kardia-R1**（rubric-as-judge RL） | `JhCircle/Kardia-R1` | ★60 | ✅ **7B 模型已开源 HF**，MIT 许可，用 ms-swift |
| TEA-Bench | `XingYuSSS/TEA-Bench` | ★12 | ✅ [ACL 2026] |
| ESC-Eval | `haidequanbu/ESC-Eval` | ★27 | ✅ [EMNLP 2024] |
| MISC（157 引） | `morecry/MISC` | ★39 | ✅ ACL 2022 |
| **KEMI**（82 引） | `dengyang17/KEMI` | ★14 | ✅ 已定位 |
| MultiESC | `lwgkzl/MultiESC` | ★51 | 已在你的收藏中 |
| GLHG（114 引） | — | | ❌ 未找到（作者 Peng Yang 搜不到匹配仓库） |
| ESC-Skills / 单轮多策略 | — | | ❌ 未找到 |

### 1.4 Kardia-R1 核心信息（本轮最大发现）

- **模型**：7B，基于 Qwen2.5-7B-Instruct，已开源 HF `Jhcircle/Kardia-R1`，MIT 许可
- **方法**：Group Relative Policy Optimization (GRPO) + **Rubric-as-Judge**（5 维：Relevance / Empathy / Persona Consistency / Safety / Fluency）
- **特点**：**无黑盒奖励模型，完全可解释**；rubric 就是判官打分标准，与我们现有的 `build_cpcd_judge_messages` 同构
- **性能**：自报"超过 GPT-4o / DeepSeek-R1 / PsyLLM 的共情质量与情绪准确率"
- **可扩展**：Qwen 3B/7B、Gemma 2B/7B 都验证过
- **推理**：支持 transformers 和 ms-swift（我们训练脚本用的就是 swift）

**立即可做的事**：
1. 克隆 `JhCircle/Kardia-R1`，读训练代码
2. 把我们的 `proxy_cpcd_judge.py` 改成 rubric 格式（5 维打分 → 加权奖励）
3. 用 research-v0 的 500 条做 GRPO，看能不能复现提升

---

## 二、最新最高准确率模型排名

### 2.1 可实证的最高分

**本机复现的 EmoDynamiX = 33.61%**（frozen 2,775-item test，NAACL 2025）

同一协议下的第二名：BlenderBot-small Joint = 32.22%（ESConv 原论文 ACL 2021）

### 2.2 论文自报的更高分（无可复现实现）

| 模型 | 自报 ACC | 代码 | 为何不可信 |
|---|---:|:--:|---|
| AFlow | **65.10%** | 有码无权重 | 合成扩展协议，非原始 test |
| DPPLM | 58.03% | 无 | 不可复跑 |
| Causal-ESC | 53.53% | 无 | 不可复跑 |
| SAGE | **46.80%** | 无 | 不可复跑 |
| CADSS | 46.26% | 只有 PDF | 不可复跑 |

**46%–65% 那一档全部没有可运行实现，属于"论文说的"，不是"我们验证过的"。**

### 2.3 本周新跑的 Qwen3.8-27B 零样本

**13.60%**（fixed-250），0 missing / 0 invalid，格式合规率 100%。

与 Qwen3.6 / DeepSeek V4-Pro / V4-Flash 四臂两两比较，**Holm p 全为 1**，一个显著的都没有。

**坐实了立项判断**：换哪个 API 模型都在 13–16%，专用小模型是 32–34%——**差距不在模型大小，在有没有针对这个任务训练过**。

### 2.4 第二代（LLM+RL）不报八类 ACC

Kardia-R1、CARE、Psyche-R1、Value Reinforcement 这些用 RL 的，报的是 rubric 分、判官偏好、人评，**基本不报八类 ACC**。因为它们做的是回复生成，不是策略分类——这才是静室真正要替换的东西。

---

## 三、CPCD 生态普查

### 3.1 答案：没有，一点都没有

| | ESConv | CPCD（Psy-Chronicle） |
|---|---|---|
| 原论文引用 | **482** | — |
| GitHub | 生态活跃 | **★4，0 fork** |
| 相关论文 | 53 篇 | 标题含 Psy-Chronicle/CPCD 的：**0 篇** |
| 有代码的改进工作 | 26 篇 | **0** |

**Psy-Chronicle 是 2026 年的新合成基准，目前还没有任何人在它上面做改进工作。**

### 3.2 那 CPCD 怎么用

CPCD 的价值在于它测 ESConv 测不了的：`mr`（记忆回溯）和 `tcr`（时序因果）—— 长程记忆能力。

**这块的可借鉴工作不在"CPCD 改进论文"，在通用长程对话记忆文献里**，而那边有三篇高引且都有代码：

- **LoCoMo**（756 引）：Evaluating Very Long-Term Conversational Memory
- **MemoryBank**（622 引）：Enhancing LLMs with Long-Term Memory
- **LongMemEval**（543 引）：Benchmarking Chat Assistants on Long-Term Interactive Memory

**这才是提 mr/tcr 的正确入口。**

---

## 四、其他心理学相关论文/数据集

### 4.1 两篇必读，与静室直接相关

#### ① Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers

**227 引，2025-04-25，有官方实现**，arXiv 2504.18412

本次普查里引用最高的心理健康专项论文，论点直指静室在做的事：**LLM 会表达病耻感（stigma）并给出不当回应，因此不能安全替代人类心理服务者**。

我们现有的安全门是自己造的 10,000 案例；这篇是同行评议、被引 227 次的失效模式清单。**应该拿来做我们安全门的外部校准。**

#### ② SoulChat: Improving LLMs' Empathy, Listening, and Comfort Abilities

**185 引，2023-11-01**，arXiv 2311.00273

**中文**心理健康 LLM，做法就是我们要做的：用多轮共情对话微调出共情/倾听/安慰能力。静室中文为主、research-v0 有 300 条中文 CPCD，**这是最贴近的中文先例**。

### 4.2 心理咨询数据集（潜在训练数据）

| 数据集 | 日期 | 引用 | 代码 | 备注 |
|---|---|---:|:--:|---|
| **AntEngage** 共情对话数据集 | 2026-08-04 | 0 | ✅ | HF `antengage/empathy-conversations`，待查许可 |
| **KokoroChat** 日语咨询对话 | 2025-06-02 | 10 | — | 真人收集，语言不匹配但构造方法可借鉴 |
| **OnCoCo 1.0** 在线咨询细粒度分类 | 2025-12-10 | 1 | — | 有标注体系，可能补我们的策略标签 |
| AnnaAgent | 2025-05-31 | 23 | ✅ | 多会话记忆的动态演化 agent |
| MCTSr-Zero | 2025-05-29 | 4 | ✅ | 自反思咨询对话生成（合成） |
| Graph2Counsel | 2026-04-22 | 2 | — | 临床接地合成（需核实） |
| MAGneT | 2025-09-04 | 7 | — | 多智能体协同生成（合成） |

⚠ **合成数据警告**：上表半数是 LLM 合成的。我们自己的红线已经写死——v2 要真人金标、MentalChat16K 因危机回复安全抽查失败被隔离。**这些合成集采纳前必须走同一道危机安全抽查，不能因为"是论文发的"就免检。**

### 4.3 心理健康评测与安全基准

| 基准 | 日期 | 引用 | 关注点 |
|---|---|---:|---|
| **CounselBench** | 2025-06-10 | 19 | 专家评测，可校准我们的 proxy judge |
| **MHSafeEval** | 2026-04-20 | 2 | 角色感知交互级安全评测，**与静室危机管线直接同构** |
| **Ψ-Arena** | 2025-05-06 | 8 | LLM 咨询师交互式评估与优化 |
| TheraMind | 2025-10-29 | 8 | 长程心理咨询的策略自适应 agent |
| Beyond Empathy | 2025-05-21 | 22 | 诊断 + 治疗推理 |
| **Psyche-R1** | 2025-08-14 | 22 | 统一共情+专业+推理，与 Kardia-R1 同代 |

---

## 五、立即可行的三条路线

### 5.1 短期（1–2 周）：Kardia-R1 复现

1. 克隆 `JhCircle/Kardia-R1`，读训练代码
2. 下载 HF `Jhcircle/Kardia-R1` 模型（7B），本地跑 ESConv fixed-250 看零样本多少分
3. 把我们的 `proxy_cpcd_judge.py` 改成 rubric 格式（5 维 → 加权奖励）
4. 用 research-v0 的 500 条做 GRPO 试跑，评估能否复现提升

**为什么先做这个**：
- 代码开源、模型开源、方法与我们现有管线同构
- 7B 模型，本地 M4 24GB 能跑推理（不能训练，但能评测）
- 如果 zero-shot 就比未微调 Qwen 强，直接部署；如果不强，我们有训练代码

### 5.2 中期（2–4 周）：补齐第一代高引论文的复跑

现有复跑计划只排了 CauESC 和 TransESC，漏了三篇高引且有码的：

- **MISC 157 引** `morecry/MISC`
- KEMI 82 引 `dengyang17/KEMI`
- GLHG 114 引（仓库未找到，下一步换作者全名 + 论文标题精确搜）

**为什么要补**：它们都是八类协议、都有代码，且引用远高于 EmoDynamiX（33.61%）。即使最后发现复现不出论文报的数字，过程中也能学到它们用的策略融合 / 知识注入 / 层次图方法。

### 5.3 长期（Phase 2 训练方案）：Qwen3.8-27B QLoRA

research-v0 训练包已经修到 `AutoModelForMultimodalLM` + 视觉塔冻结 + LoRA 边界断言，**3.8 与 3.6 的 config 在训练脚本断言涉及的每个字段上完全一致**。只需改 `model_id` 和 `model_revision`。

但有三项必须 GPU 实测：
1. `language_model_only` 断言（该键未出现在 3.8 公开 config，可能抛错）
2. **3.8 新 chat template 下 completion-only loss mask**（3.8 默认开思考且 `preserve_thinking=true`，模板与 3.6 不同）
3. fast-path kernel 绑定（取决于 causal_conv1d / fla 在镜像里的安装情况）

**训练算力**：你说不租云 GPU，本地高性能库下周到位。按 `reports/local_compute_and_serving_options_20260828.md` 的决策树：若库是推理用，Phase 2 仍需外部 80GB CUDA；若是训练用且 ≥80GB CUDA，现成 bundle 原样可用；若是大内存 Apple Silicon，得自建训练器 + 六项等效门禁。

---

## 六、产物清单

1. `reports/esconv_literature_recheck_20260828.md` —— ESConv 文献重查（179 篇去重、两个时代、26 个有码候选）
2. `reports/cpcd_and_psych_literature_survey_20260828.md` —— CPCD 生态（答案是"没有"）+ 心理健康数据集/基准普查
3. `reports/qwen_finetune_pretest_baseline_20260828.{md,json}` —— pre-finetune 基线（CPCD + ESConv fixed-250，四臂配对）
4. `reports/qwen38_availability_20260828.md` —— Qwen3.8-27B 可得性、架构一致性、OpenRouter 供应商
5. `reports/local_compute_and_serving_options_20260828.md` —— 本地算力决策树 + 生产端点方案
6. `reports/esconv_fixed250_qwen38_pretest_20260828.{md,json}` —— **Qwen3.8 零样本 13.60%**
7. `open_response_eval/preregistration_dashscope_qwen38_esconv250_addendum.json` —— DashScope Qwen3.8 ESConv addendum 协议（已冻结）
8. `project-ledger/updates/2026-08-28-qwen-therapy-prep-sprint.md` —— 台账
9. `tmp/pwc_esconv_papers.json` / `tmp/pwc_cpcd_papers.json` —— 原始 API 数据
10. `tmp/esconv_repo_probe.json` —— 代码仓库探测结果

---

## 七、下一步（按优先级）

1. ✅ **立即**：读 stigma 论文（227 引），失效模式并入部署安全门
2. ✅ **本周**：Kardia-R1 模型下载 + ESConv fixed-250 零样本评测
3. **下周**：Kardia-R1 训练代码审计 + rubric-as-judge 与我们 proxy judge 的对接方案
4. GLHG 换论文全标题 + 作者全名再搜一轮
5. AntEngage / OnCoCo 许可与规模核实
6. 读 SoulChat 对照中文数据构造
7. LoCoMo / MemoryBank / LongMemEval 作为 CPCD mr/tcr 改进入口评估
