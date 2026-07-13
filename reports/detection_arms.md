# 三臂检测 vs 纯 DeepSeek 基线 — 分类别 P/R/F1

金标 = 数据集种子 label(4 类,多轮取 per_turn);单元 = id#turn。
baseline_deepseek = 与标注员 A 同一协议的纯 deepseek-chat 零样本(复用其输出)。

## judge(n=344)

| 类别 | precision | recall | F1 | support |
|---|---|---|---|---|
| none | 96.1 | 83.6 | 89.4 | 177 |
| passive_ideation | 64.7 | 91.7 | 75.9 | 60 |
| active_ideation | 87.8 | 60.0 | 71.3 | 60 |
| crisis | 60.9 | 83.0 | 70.3 | 47 |
| **acc** | 80.8 | **macro-F1** | 76.7 | wF1 81.3 |

二值化(none vs 风险):召回 96.4%,误报 16.4%;passive_ideation 召回 91.7%

混淆矩阵(行=gold,列=pred):

| gold\pred | none | passive_ideation | active_ideation | crisis | null |
|---|---|---|---|---|---|
| none | 148 | 14 | 1 | 14 | 0 |
| passive_ideation | 0 | 55 | 2 | 3 | 0 |
| active_ideation | 2 | 14 | 36 | 8 | 0 |
| crisis | 4 | 2 | 2 | 39 | 0 |

## lexicon(n=344)

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

## pipeline_deep(n=344)

| 类别 | precision | recall | F1 | support |
|---|---|---|---|---|
| none | 70.5 | 75.7 | 73.0 | 177 |
| passive_ideation | 0.0 | 0.0 | 0.0 | 60 |
| active_ideation | 32.7 | 30.0 | 31.3 | 60 |
| crisis | 28.9 | 59.6 | 38.9 | 47 |
| **acc** | 52.3 | **macro-F1** | 35.8 | wF1 48.3 |

二值化(none vs 风险):召回 66.5%,误报 24.3%;passive_ideation 召回 0.0%

混淆矩阵(行=gold,列=pred):

| gold\pred | none | passive_ideation | active_ideation | crisis | null |
|---|---|---|---|---|---|
| none | 134 | 2 | 9 | 32 | 0 |
| passive_ideation | 29 | 0 | 24 | 7 | 0 |
| active_ideation | 12 | 0 | 18 | 30 | 0 |
| crisis | 15 | 0 | 4 | 28 | 0 |

## pipeline_fast(n=344)

| 类别 | precision | recall | F1 | support |
|---|---|---|---|---|
| none | 64.9 | 75.1 | 69.6 | 177 |
| passive_ideation | 0.0 | 0.0 | 0.0 | 60 |
| active_ideation | 45.7 | 35.0 | 39.6 | 60 |
| crisis | 32.3 | 63.8 | 42.9 | 47 |
| **acc** | 53.5 | **macro-F1** | 38.0 | wF1 48.6 |

二值化(none vs 风险):召回 56.9%,误报 24.9%;passive_ideation 召回 0.0%

混淆矩阵(行=gold,列=pred):

| gold\pred | none | passive_ideation | active_ideation | crisis | null |
|---|---|---|---|---|---|
| none | 133 | 0 | 7 | 37 | 0 |
| passive_ideation | 40 | 0 | 16 | 4 | 0 |
| active_ideation | 17 | 0 | 21 | 22 | 0 |
| crisis | 15 | 0 | 2 | 30 | 0 |

## baseline_deepseek(n=344)

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

## pipeline_fast — 分支级评分(正确目标:expected_branch,含可接受分支)

分支命中率:43.3%(n=344)

| expected_branch | 命中/总数 | 命中率 |
|---|---|---|
| crisis | 30/47 | 63.8% |
| diagnosis | 4/24 | 16.7% |
| gentle_check | 4/30 | 13.3% |
| medical_redflag | 6/24 | 25.0% |
| medication | 12/24 | 50.0% |
| normal | 65/105 | 61.9% |
| suspected | 28/90 | 31.1% |

## pipeline_deep — 分支级评分(正确目标:expected_branch,含可接受分支)

分支命中率:46.2%(n=344)

| expected_branch | 命中/总数 | 命中率 |
|---|---|---|
| crisis | 28/47 | 59.6% |
| diagnosis | 1/24 | 4.2% |
| gentle_check | 4/30 | 13.3% |
| medical_redflag | 23/24 | 95.8% |
| medication | 8/24 | 33.3% |
| normal | 64/105 | 61.0% |
| suspected | 31/90 | 34.4% |
