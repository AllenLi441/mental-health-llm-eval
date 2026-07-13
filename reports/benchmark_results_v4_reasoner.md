# 基准评测(v4,含思考档 reasoner)——诚实计分板与结论

> ⚠️ **2026-07-11 重大更正(本文档 IMHI 部分已被 `imhi_uniform_v3u.md` 取代):**
> 本文档把「IMHI-Dreaddit 82.8 超微调 SOTA」和「IMHI-DR 超 MentaLLaMA-13B」列为硬胜项——**这两项经统一去偏协议(v3u)检验都不成立**。原因:通用 prompt 有 yes-偏置,只给"输的任务"(dreaddit/MultiWD)加了 criteria = 选择性优化;把同套构念 criteria 统一补到全部 5 个二值任务后,高-yes 的 DR/loneliness/Irf 回落(DR 88.4→81.8 不再胜任何人),dreaddit 降为**与微调平手**(81.1 vs 81.8,CI 覆盖)。**真正稳健的硬胜项收窄为 3 个:CPsyExam、EmoBench-EU、MentalManip(均超 GPT-4 级)。** 详见 `imhi_uniform_v3u.md`。下方原文保留以存档,IMHI 数字以 v3u 为准。

---


> 2026-07-11 更新。相对旧版(v3,仅 deepseek-chat 非思考档)的两处硬更正:
> ① 补跑了产品**思考档 `deepseek-reasoner`**(实测 = v4-flash 思考模式,非独立 pro 模型)全部推理型任务;
> ② 逐条核验后 **IMHI 不再是「0/9 输给微调 SOTA」——Dreaddit 思考档已反超微调 SOTA(1/9)**;
> psysuicide 的「加权 F1 81.9」被降级为**非胜项**(见下,加权 F1 被多数类抬高,可比口径下大幅落后微调基线)。
> 全部数字可由 `eval-suite/scripts/scoreboard.py` 一键重算;思考档逐条 jsonl 落盘 `results/*-deepseek-reasoner-r1.jsonl`。

## 1. 方法学声明(必须写进论文)

- **两种推理配置**:`chat` = deepseek-chat(非思考),`reasoner` = deepseek-reasoner(思考)。二者 API 回传同 base(`deepseek-v4-flash`,指纹 `fp_8b330d02d0`),差别是**是否启用思考**。产品深度档 = 思考档。
- **思考不是免费的**:思考在推理/情绪理解任务上大幅加分,但在**类别不平衡的判别任务上反而降分**(见 §3)。因此本表**逐任务取 chat/reasoner 中更优者,并逐项标注用了哪个**——这是一个显式的方法学选择,不是隐藏的 best-of。
- **对照分三档**,难度递增:零样本 LLM 基线(ChatGPT/GPT-4)→ 领域 LLM(MentaLLaMA-13B)→ 任务微调判别式(RoBERTa/BERT 家族)。零样本通用模型打不过微调判别式是**结构性正常**,不作为失败。

## 2. 计分板(可比口径:论文报 F1 的用同类 F1,报 acc 的用 acc)

| 任务 | 我们(配置) | 最强可比对照 | 结论 |
|---|---|---|---|
| **CPsyExam** (n=3902) | **83.1%** (chat) | GPT-4 零样本 67.4 | ✅ **超 GPT-4 +15.7**〔MCQ,多选题口径优势见注〕 |
| **EmoBench-EU** (官方协议) | **63.5** (reasoner) | GPT-4 56.9 | ✅ **超 GPT-4 +6.6**(chat 57.8 亦微超) |
| **IMHI-Dreaddit** | **82.8 wF1** (reasoner) | 微调 MentalRoBERTa 81.8 | ✅✅ **超微调 SOTA**(零样本罕见;chat 仅 59.9,思考 +22.9,已核验 invalid=0) |
| **IMHI-DR** | **88.4 wF1** (chat) | MentaLLaMA-13B 85.7 | ✅ 超领域 LLM(仍输微调 RoBERTa 95.1) |
| **MentalManip** (con) | **77.8% acc** (chat) | GPT-4-Turbo 零样本 65.7 | ✅ **超 GPT-4-Turbo 零样本 +12**,约平微调 RoBERTa-base 76.8 |
| EmoBench-EA | 72.5 (reasoner) | GPT-4 74.6 | 🟡 差 2.1,未过 |
| IMHI 其余 7 项 | — | MentaLLaMA-13B | 🟡 差 1–7 分未过;且**全部输微调判别式** |
| **PsySUICIDE**(11 类) | 80.2% acc / **45.3 macro-F1** (reasoner) | 微调 RoBERTa-large **92.8 acc / 69.8 macro-F1** | ❌ **大幅落后微调基线**;仅超多数类 72.4;细粒度少数类塌陷 |
| EATD | 83.5 (chat) | 多模态融合 71(F1 口径不同) | ⚠ 口径不可比,不计胜负 |
| CBT-cd/fc/pc | — | 多标签 F1 vs 我们 top-1 | ⚠ 口径不可比 |

**注 CPsyExam**:多选题(MAQ)零样本对所有模型都难;我们的优势主要在单选(SCQ)。论文引用的 GPT-4 均值 67.4 含少样本列,严格零样本加权均值更低,差距不缩小。

## 3. 关键机制发现:思考档的加分与反噬

| 任务 | chat | reasoner | Δ | 解读 |
|---|---|---|---|---|
| IMHI-Dreaddit | 59.9 | 82.8 | **+22.9** | 二值压力检测,思考大幅纠偏 |
| PsySUICIDE (acc) | 72.4 | 80.2 | +7.8 | 二值风险判定变好,但细粒度仍塌 |
| EmoBench-EU | 57.8 | 63.5 | +5.7 | 情绪+成因联合推理,思考擅长 |
| IMHI-multiwd | 56.2 | 68.1 | +11.9 | — |
| **IMHI-loneliness** | 81.8 | 73.0 | **−8.7** | 极不平衡(yes:no≈4:1),思考过度倒向少数类 |
| **IMHI-cams** | 43.1 | 38.2 | **−4.9** | 6 类不平衡,同上 |
| MentalManip | 77.8 | 75.6 | −2.2 | 思考轻微反噬 |

**结论**:思考档是路 2 的核心杠杆,但在不平衡判别任务上会反噬——所以必须逐任务选配置,并在论文里如实说明。

## 4. 诚实的总结论(替换旧版「第一梯队」措辞)

1. **零样本 vs 零样本/领域 LLM:5 个硬胜项。** CPsyExam、EmoBench-EU 超 GPT-4;MentalManip 超 GPT-4-Turbo 零样本;IMHI-DR 超 MentaLLaMA-13B;**IMHI-Dreaddit 超微调 SOTA**(唯一一个零样本反超微调判别式的任务)。这些集中在**中文心理推理 + 情绪理解 + 英文压力/操控检测**。
2. **零样本 vs 微调判别式:整体仍低。** IMHI 8/9、PsySUICIDE 微调基线均领先——零样本对专门微调的正常差距,不硬争,如实入附录。
3. **最该正视的短板:细粒度自杀风险分诊。** PsySUICIDE macro-F1 仅 45.3(被动意念召回 33%、自杀计划 0/6),这是**产品家门口的安全任务**,也是全套里最弱的一环。加权 F1(81.9)看着漂亮是被「与自杀无关」多数类抬起来的假象——报告必须用 macro-F1 / 逐类召回口径呈现,不能用加权 F1 报喜。
4. **论文可打的标题只能是**:「一个通用双语助手,零样本,在中文为核心的心理推理/情绪/部分安全检测任务上可与 GPT-4 级模型持平乃至超越,并在 Dreaddit 上超过一个微调 SOTA」——**不是**「碾压所有心理 NLP 的 SOTA」。

## 5. 复现

```bash
# 思考档(reasoner)结果已落盘;重算计分板:
cd eval-suite && python3 scripts/scoreboard.py
# 单任务思考档重跑(示例):
node run.mjs imhi-dreaddit --model deepseek-reasoner --seed 42 --run-id r1
# EmoBench 官方协议(EA+EU,中英全量):
cd ../EmoBench/eval && DEEPSEEK_API_KEY=... node eval.mjs --model deepseek-reasoner
```

## 6. 待办(升级为更强证据前不能过度声称)

- CPsyExam / IMHI-Dreaddit 思考档目前 CPsyExam 用 600 抽样、IMHI 用全量;若入论文,CPsyExam 需补全量思考档或明确标注抽样口径。
- MentalManip 列的 ACC 列位读自论文 Table 5 表头「P,R,ACC,F」(ACC 为第 3 列),已与正文「GPT-4 假阳性多、accuracy≈0.653」互证;正式引用时附页码。
