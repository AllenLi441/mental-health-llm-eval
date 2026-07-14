# 检测臂 vs 纯 DeepSeek 基线 — 端到端安全与干预路由(2026-07-13)

金标 = 数据集种子 label(4 类,多轮取 per_turn);单元 = id#turn。
baseline_deepseek = 与标注员 A 同一协议的纯 deepseek-chat 零样本(复用其输出)。

**判官归属:** 判官臂运行当日 Kimi 账户欠费停用,判官调用全部由 DeepSeek 兜底档应答(判官=DeepSeek 兜底档);同一账户状态下的归属探针 10/10 由 DeepSeek 兜底(证据 `results/judge_attrib_probe.jsonl`)。

**方法学修正(同行评审):** 管线臂改为 (1) **全局串行**执行,(2) 决策日志路由按 `sessionHash+turnIndex` **唯一认领**(修掉并发下取最新日志造成的数据串线),(3) 多轮样例用 `runConversation` **真实回放**(逐轮送入 + 回灌 app 回复 + 危机状态粘滞),不再只拼接用户消息。
路由认领审计:pipeline_fast: route 唯一相关 343/344 = 99.7% · pipeline_deep: route 唯一相关 344/344 = 100.0%

## 表 6-A · 端到端安全(全臂)

| 臂 | 二值风险召回 | 二值误报率 | crisis 类召回 | passive 单元任意干预率 |
|---|---|---|---|---|
| 全管线(快速档) | 65.9% | 17.5% | 83.0% | 30.0% |
| 全管线(深度档) | 53.9% | 10.7% | 61.7% | 25.0% |
| 语义判官(单臂) | 97.0% | 16.9% | 83.0% | 100.0% |
| 词表(单臂) | 24.6% | 9.0% | 8.5% | 10.0% |
| 纯 DeepSeek 基线 | 97.0% | 7.3% | 78.7% | 98.3% |

**表注:**「passive 单元任意干预率」= passive 金标单元中 4 类投影预测 ≠ none 的比例(即触发了 crisis / suspected / gentle_check 任一干预分支;medication / diagnosis / medical_redflag / normal 等**边界分支投影为 none,不计入干预**)。**该列须与「二值误报率」对照读**:干预率高、误报率也高,只是「宁可错杀」的激进阈值,并非越高越好。管线臂该列偏低的成因见文末〈附录:管线 4 类投影〉的两半机制脚注。

## 表 6-B · 干预路由质量(仅管线)

正确目标 = `expected_branch`(7 类,含 `acceptable_branches` 可接受分支口径);此表衡量路由系统本身,不做 4 类压缩投影。

| expected_branch | 全管线(快速档) | 全管线(深度档) |
|---|---|---|
| **总分支命中率** | **170/344(49.4%)** | **186/344(54.1%)** |
| crisis | 39/47(83.0%) | 29/47(61.7%) |
| suspected | 33/90(36.7%) | 31/90(34.4%) |
| gentle_check | 4/30(13.3%) | 3/30(10.0%) |
| normal | 75/105(71.4%) | 88/105(83.8%) |
| diagnosis | 2/24(8.3%) | 2/24(8.3%) |
| medication | 11/24(45.8%) | 10/24(41.7%) |
| medical_redflag | 6/24(25.0%) | 23/24(95.8%) |

## 单臂 4 类 P/R/F1(判官 / 词表 / 基线)

### 语义判官(单臂)(judge,n=344)

| 类别 | precision | recall | F1 | support |
|---|---|---|---|---|
| none | 96.7 | 83.1 | 89.4 | 177 |
| passive_ideation | 62.8 | 90.0 | 74.0 | 60 |
| active_ideation | 86.7 | 65.0 | 74.3 | 60 |
| crisis | 63.9 | 83.0 | 72.2 | 47 |
| **acc** | 81.1 | **macro-F1** | 77.5 | wF1 81.7 |

二值化(none vs 风险):召回 97.0%,误报 16.9%;passive_ideation 召回 90.0%

混淆矩阵(行=gold,列=pred):

| gold\pred | none | passive_ideation | active_ideation | crisis | null |
|---|---|---|---|---|---|
| none | 147 | 15 | 2 | 13 | 0 |
| passive_ideation | 0 | 54 | 3 | 3 | 0 |
| active_ideation | 1 | 14 | 39 | 6 | 0 |
| crisis | 4 | 3 | 1 | 39 | 0 |

### 词表(单臂)(lexicon,n=344)

| 类别 | precision | recall | F1 | support |
|---|---|---|---|---|
| none | 56.1 | 91.0 | 69.4 | 177 |
| passive_ideation | 26.3 | 8.3 | 12.7 | 60 |
| active_ideation | 0.0 | 0.0 | 0.0 | 60 |
| crisis | 10.5 | 8.5 | 9.4 | 47 |
| **acc** | 49.4 | **macro-F1** | 22.9 | wF1 39.2 |

二值化(none vs 风险):召回 24.6%,误报 9.0%;passive_ideation 召回 8.3%

混淆矩阵(行=gold,列=pred):

| gold\pred | none | passive_ideation | active_ideation | crisis | null |
|---|---|---|---|---|---|
| none | 161 | 2 | 0 | 14 | 0 |
| passive_ideation | 54 | 5 | 0 | 1 | 0 |
| active_ideation | 36 | 5 | 0 | 19 | 0 |
| crisis | 36 | 7 | 0 | 4 | 0 |

### 纯 DeepSeek 基线(baseline_deepseek,n=344)

| 类别 | precision | recall | F1 | support |
|---|---|---|---|---|
| none | 97.0 | 92.7 | 94.8 | 177 |
| passive_ideation | 64.6 | 88.3 | 74.6 | 60 |
| active_ideation | 80.0 | 53.3 | 64.0 | 60 |
| crisis | 69.8 | 78.7 | 74.0 | 47 |
| **acc** | 83.1 | **macro-F1** | 76.9 | wF1 83.1 |

二值化(none vs 风险):召回 97.0%,误报 7.3%;passive_ideation 召回 88.3%

混淆矩阵(行=gold,列=pred):

| gold\pred | none | passive_ideation | active_ideation | crisis | null |
|---|---|---|---|---|---|
| none | 164 | 3 | 3 | 7 | 0 |
| passive_ideation | 1 | 53 | 3 | 3 | 0 |
| active_ideation | 1 | 21 | 32 | 6 | 0 |
| crisis | 3 | 5 | 2 | 37 | 0 |

---

## 附录:管线 4 类投影(参考)

**脚注 · passive 投影的两半机制(为何管线 passive 行 P/R/F1 近 0 不等于全数漏检):**

passive 金标共 60 条,按数据集 `expected_branch` 恰好分两半 —— 30 条期望 `suspected`、30 条期望 `gentle_check`。

1. **结构假象半(30 条期望 suspected):** 管线正确路由到 `suspected` 时,4 类投影按 `labelFromBranch(suspected)=active_ideation` 记为 active,于是在 passive 行被计为「未召回」—— 这是投影缺 suspected→passive 槽位造成的**结构假象**,并非放行(端到端安全应看表 6-A 的任意干预率)。
2. **真实漏检半(30 条期望 gentle_check):** 管线几乎不触发 `gentle_check` 分支,这些单元被投影为 none —— 属**真实漏检(放行)**。

两半叠加 → 快速档 passive 行 none=42(约 70% 放行),故该行 precision/recall/F1 近 0;拆半后仅约半数为真实漏检,另一半是投影结构假象。

### 全管线(快速档)(pipeline_fast,n=344)

| 类别 | precision | recall | F1 | support |
|---|---|---|---|---|
| none | 71.9 | 82.5 | 76.8 | 177 |
| passive_ideation | 0.0 | 0.0 | 0.0 | 60 |
| active_ideation | 52.1 | 41.7 | 46.3 | 60 |
| crisis | 42.4 | 83.0 | 56.1 | 47 |
| **acc** | 61.0 | **macro-F1** | 44.8 | wF1 55.3 |

二值化(none vs 风险):召回 65.9%,误报 17.5%;passive_ideation 召回 0.0%

混淆矩阵(行=gold,列=pred):

| gold\pred | none | passive_ideation | active_ideation | crisis | null |
|---|---|---|---|---|---|
| none | 146 | 1 | 5 | 25 | 0 |
| passive_ideation | 42 | 0 | 16 | 2 | 0 |
| active_ideation | 9 | 0 | 25 | 26 | 0 |
| crisis | 6 | 0 | 2 | 39 | 0 |

### 全管线(深度档)(pipeline_deep,n=344)

| 类别 | precision | recall | F1 | support |
|---|---|---|---|---|
| none | 67.2 | 89.3 | 76.7 | 177 |
| passive_ideation | 0.0 | 0.0 | 0.0 | 60 |
| active_ideation | 53.7 | 36.7 | 43.6 | 60 |
| crisis | 43.3 | 61.7 | 50.9 | 47 |
| **acc** | 60.8 | **macro-F1** | 42.8 | wF1 54.0 |

二值化(none vs 风险):召回 53.9%,误报 10.7%;passive_ideation 召回 0.0%

混淆矩阵(行=gold,列=pred):

| gold\pred | none | passive_ideation | active_ideation | crisis | null |
|---|---|---|---|---|---|
| none | 158 | 1 | 2 | 16 | 0 |
| passive_ideation | 45 | 0 | 15 | 0 | 0 |
| active_ideation | 16 | 0 | 22 | 22 | 0 |
| crisis | 16 | 0 | 2 | 29 | 0 |
