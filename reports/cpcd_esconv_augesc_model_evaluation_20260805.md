# CPCD、ESConv、AugESC 模型证据与静室评测报告

日期：2026-08-05
状态：协议和评测器已完成；双模型正式成绩尚未完成
目标模型：`deepseek-v4-pro`（静室冻结提示词）与 `Qwen/Qwen3.6-27B`

## 先给结论

这三个数据集不能各写一个“准确率”：

- CPCD 是开放回答评测，正式结果是 SR、MR、TCR 的质量均分。换算成百分比时只能叫“归一化质量分”，不能叫 Accuracy。
- ESConv 可以计算 8 类支持策略 exact-match Accuracy、Macro-F1 和 Weighted-F1；回复文本另报 BLEU、ROUGE-L、Distinct。
- AugESC 是训练增强数据，没有独立 test accuracy。论文是在 ESConv test 和人工互动中验证它是否改善下游模型。

因此，本项目最终只会出现一张 CPCD 质量分表和一张 ESConv 策略准确率/回复指标表，不会伪造“AugESC 准确率”。

## 对此前网页研究报告的纠正

此前报告第 4 节给出的以下区间没有可核查的逐条输出、运行清单、模型指纹或评分文件，不能作为实验结果：

- DeepSeek ESConv Accuracy `34.20% - 35.80%`
- Qwen3.6-27B ESConv Accuracy `31.50% - 33.20%`
- DeepSeek CPCD SR/MR/TCR `4.400 - 4.620 / 4.300 - 4.450 / 4.400 - 4.550`
- Qwen3.6-27B CPCD SR/MR/TCR `4.500 - 4.680 / 4.350 - 4.500 / 4.180 - 4.350`

这些数字是推测，不是实测。本报告取代这些区间；在完整 JSONL 和评分摘要生成前，结果栏必须保持“待测”。

此前报告还把 CPCD 三项笼统写成统一 1-5 分。论文正文确实这样描述，但固定官方仓库的 rubric 实际为：SR 1-5，MR/TCR 0-5。正式复现采用发布仓库 rubric，并披露这一论文-代码差异。

## 1. CPCD 被哪些模型使用，成绩是多少

官方来源：

- 论文：[Psy-Chronicle](https://arxiv.org/abs/2605.22140)
- 仓库：[Psy-Chronicle `ff812c9`](https://github.com/EdwinUSTB/Psy-Chronicle/tree/ff812c9084b606631dac3a8c01f7be0d5cbc8c8d)

只有 `CPCD-Chat-4B` 和 `CPCD-Chat-8B` 使用 CPCD 训练，底座分别是 Qwen3-4B 和 Qwen3-8B。其他模型只是参加 CPCD-Bench 评测。

论文 Table 3：

| 模型 | Judge SR | Judge MR | Judge TCR | Human SR | Human MR | Human TCR |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4 | 4.778 | 4.513 | 4.525 | 4.626 | 4.448 | 4.569 |
| Gemini-3-flash-preview | 4.455 | 4.445 | 4.445 | 4.510 | 4.334 | 4.412 |
| Gemini-2.5-flash | 3.967 | 4.296 | 4.211 | 3.912 | 4.211 | 4.189 |
| DeepSeek V3.2 | 4.397 | 4.318 | 4.488 | 4.312 | 4.208 | 4.413 |
| MiniMax M2.7 | 4.173 | 4.102 | 4.212 | 4.017 | 4.012 | 4.188 |
| GLM-5.1 | 4.198 | 4.202 | 4.311 | 4.186 | 4.189 | 4.288 |
| Kimi-k2.5 | 4.212 | 4.103 | 4.228 | 4.225 | 4.125 | 4.189 |
| Qwen3-Max | 4.231 | 4.205 | 4.238 | 4.123 | 4.285 | 4.123 |
| Qwen3-8B | 3.625 | 3.722 | 3.623 | 3.558 | 3.677 | 3.789 |
| CBT-LLM (Baichuan-7B) | 3.714 | 3.811 | 3.733 | 3.678 | 3.774 | 3.658 |
| Camel (LLaMA-8B) | 3.778 | 3.814 | 3.712 | 3.812 | 3.789 | 3.755 |
| PsyLLM (Qwen3-8B) | 3.825 | 3.922 | 3.823 | 3.768 | 3.899 | 3.965 |
| CPCD-Chat-4B | 3.776 | 3.756 | 3.641 | 3.712 | 3.706 | 3.612 |
| CPCD-Chat-8B | 3.884 | 3.947 | 3.824 | 3.846 | 3.944 | 3.812 |

这些是开放回答均分，不是 Accuracy。DeepSeek V3.2 和 Qwen3-8B 的论文数字也不能代替本次 `deepseek-v4-pro` 与 `Qwen3.6-27B` 的结果。

### CPCD 官方评分规则

| 任务 | 样本 | 发布 rubric | 正式报告方式 |
|---|---:|---|---|
| SR（仓库目录 `srg`） | 99 | Empathy、Coherence、Professionalism，各 1-5 | 三维均分及总体均分 |
| MR | 40 | Accuracy、Completeness、Temporal Consistency、No Hallucination，各 0-5 | 四维均分及总体均分 |
| TCR | 20 | Temporal Accuracy、Causal Coherence、Completeness、No Hallucination，各 0-5 | 四维均分及总体均分 |

论文以 `GPT-5.2` 作自动 judge，并传入任务输入、候选答案和 reference answer。自动评分与人工评分的 Pearson 相关为 SR `0.982`、MR `0.979`、TCR `0.972`，总体 `0.975`。

## 2. ESConv 被哪些模型使用，成绩是多少

官方来源：

- 原论文：[Towards Emotional Support Dialog Systems](https://aclanthology.org/2021.acl-long.269/)
- 仓库：[Emotional-Support-Conversation `f262d06`](https://github.com/thu-coai/Emotional-Support-Conversation/tree/f262d062ad74cb39b17ea476facc81568ddcba24)
- 本次固定 test：2,775 个 supporter turn；SHA-256 `b85ae888bf747cefa54bba2a6c3e2f6ccb4c1005d4e0b6d1d3be3823cf040aef`

ESConv 2021 原论文没有报告策略 Accuracy，只报告回复生成指标：

| Backbone | Variant | PPL | BLEU-2 | ROUGE-L | BOW Extrema |
|---|---|---:|---:|---:|---:|
| DialoGPT | Vanilla | 15.51 | 5.13 | 15.26 | 49.80 |
| DialoGPT | Joint | - | 5.00 | 15.09 | 49.97 |
| DialoGPT | Oracle | 15.19 | 5.52 | 15.82 | 50.18 |
| BlenderBot | Vanilla | 16.23 | 5.45 | 15.43 | 50.49 |
| BlenderBot | Joint | - | 5.35 | 15.46 | 50.27 |
| BlenderBot | Oracle | 16.03 | 6.31 | 17.90 | 51.65 |

后续论文才报告“下一轮支持策略分类 Accuracy”：

| 论文 | 模型 | Strategy Accuracy |
|---|---|---:|
| [MISC](https://aclanthology.org/2022.acl-long.25/) | BlenderBot-Joint / MISC | 28.57% / 31.63% |
| [MultiESC](https://aclanthology.org/2022.emnlp-main.195/) | DialoGPT-Joint / BlenderBot-Joint / MISC / MultiESC | 26.03% / 29.92% / 31.61% / 42.01% |
| [TransESC](https://aclanthology.org/2023.findings-acl.420/) | BlenderBot-Joint / MISC / TransESC | 17.69% / 31.67% / 34.71% |
| [PAL](https://aclanthology.org/2023.findings-acl.34/) | BlenderBot-Joint / MISC / Hard Prompt / PAL | 27.72% / 31.34% / 34.24% / 34.51% |

这些数字不能合并成统一排行榜：各论文使用的切分、预处理、测试样本数和标签空间不同；MultiESC 实际还把 Suggestions 与 Information 合并并加入 Greetings。本次固定 2,775 行、原始 8 类 test 上得到的两模型结果可以彼此公平比较，但不能直接声称超过 MultiESC 的 42.01%。

本次 ESConv 正式指标：

- 8 类 exact-match Accuracy
- Macro-F1、Weighted-F1、逐类 F1、混淆矩阵
- 非法 JSON/非法标签率；非法输出直接算错
- BLEU-1/2/3/4、ROUGE-L、Distinct-1/2/3
- 不报跨供应商 PPL，因为 API 不提供同口径 token likelihood

## 3. AugESC 使用了什么模型，成绩是多少

官方来源：

- 论文：[AugESC](https://aclanthology.org/2023.findings-acl.99/)
- 仓库：[AugESC `7c9da4e`](https://github.com/thu-coai/AugESC/tree/7c9da4e30099e363a9fdfe142c2673f71b5a9d59)
- 发布模型：[blenderbot-1B-augesc](https://huggingface.co/thu-coai/blenderbot-1B-augesc)

模型角色：

- `GPT-J 6B` 用于生成 AugESC 数据。
- 下游实验使用两个 `BlenderBot 1.4B`。两者先在 ESConv 1,100 个对话上训练，其中一个再在 AugESC 上 post-train 1 epoch。

论文 Table 7 在同一个 ESConv 200-session test 上报告：

| AugESC post-train | PPL | B-2 | B-4 | R-L | D-2 | D-3 |
|---|---:|---:|---:|---:|---:|---:|
| No | 11.2 | 7.8 | 2.4 | 16.9 | 23.8 | 48.0 |
| Yes | 11.5 | 7.7 | 2.4 | 16.7 | 24.3 | 49.4 |

开放域人工互动中，“训练 AugESC / 未训练”的胜率列为：Fluency `47/13`、Identification `68/22`、Comforting `55/22`、Suggestion `58/15`、Overall `58/28`，其余为平局。

AugESC 没有独立 test split 和 Accuracy，所以本次不重复把它当第三个测试集。它未来用于训练；训练后的模型仍在冻结 ESConv test 或另建的未见数据上评测。

## 4. 本次静室与 Qwen 的统一评测协议

冻结文件：

- `open_response_eval/preregistration.json`
- `open_response_eval/esconv_method_v2.json`
- `open_response_eval/prompts/jingshi_core_zh.txt`
- `open_response_eval/prompts/jingshi_core_en.txt`

公平条件：

- 两个目标臂使用同一任务、同一静室核心提示词、同一输出格式和同一预算。
- ESConv 历史 supporter turn 保留官方策略标签，但绝不暴露当前 target label。
- Qwen thinking 关闭；DeepSeek 记录其供应商默认 reasoning。这个比较代表两个预定部署臂，不等价于“同推理模式的纯底座能力比较”。
- CPCD 用独立 `openai/gpt-5.2` judge，模型名盲化，并按固定 seed `20260805` 稳定打乱两臂顺序。
- 每条结果保存请求模型、返回模型、fingerprint、prompt hash、usage 和原始输出；可续跑。
- 当前训练数据含 CPCD、ESConv、AugESC，因此未来 adapter 训练完成后，这三者不能再声称为“完全未见测试”。本次必须在训练前完成 base-model 冻结评测。

## 5. 当前真实运行状态

截至本报告写入时：

| 环节 | 状态 | 可计入正式结果 |
|---|---|---|
| DeepSeek CPCD generation | 140/159：SR 99/99、MR 39/40、TCR 2/20 | 否，覆盖不完整且部分旧预算触顶 |
| DeepSeek ESConv legacy smoke | 6/2,775 | 否，旧输入漏带历史策略标签 |
| DeepSeek ESConv v2 | 0/2,775 | 否 |
| Qwen3.6-27B CPCD/ESConv | 0/2,934 | 否 |
| GPT-5.2 CPCD judge | 0/318 | 否 |

DeepSeek 已保存的 CPCD 140 条共使用：

- prompt tokens：2,382,611
- completion tokens：67,677，其中 reasoning tokens 49,740
- total tokens：2,450,288

随后专用 DeepSeek 评测账户返回 `HTTP 402 Insufficient Balance`。这不是模型分数，也不是程序解析失败；在补充余额前无法继续调用。

另一个有效性发现是旧输出预算过低：SR 有 5 条 completion 正好达到 800 tokens，最短正文只有 6-17 个字符；旧 ESConv 也多次达到 501/508 of 512 tokens。正式运行必须使用对两个模型统一提高后的冻结预算，并排除这些旧 partial，不得把截断输出算作模型能力。

## 6. 尚缺的运行条件

本机可调用 DeepSeek，但当前余额不足。Qwen3.6-27B 官方模型真实存在，冻结 revision 为 `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`；官方 BF16 权重约 51.75 GiB，而本机只有 24 GiB 统一内存，无法本地公平加载。

完成最终表需要：

1. 为专用 DeepSeek 评测 key 补充余额。
2. 提供远程 OpenAI-compatible Qwen3.6-27B endpoint，并设置 `QWEN_EVAL_BASE_URL`、`QWEN_EVAL_API_KEY`。
3. 设置独立 `JUDGE_API_KEY`，用于 `openai/gpt-5.2` CPCD 盲评。
4. 若 Qwen 为自托管，保存启动命令、权重 revision 和服务 manifest；若使用供应商托管 API，结果必须标为“provider-managed，未验证精确权重 revision”。

密钥只写本地 `.env`，不写报告、不提交 Git。完整名义调用量为两臂 generation `5,868` 次，加 CPCD judge `318` 次，共 `6,186` 次；格式修正和传输重试另计。

## 7. 最终结果表模板

在覆盖率、模型身份和 judge 身份全部验证前，以下结果保持空白：

| 模型臂 | CPCD SR | CPCD MR | CPCD TCR | ESConv Accuracy | Macro-F1 | Invalid rate |
|---|---:|---:|---:|---:|---:|---:|
| 静室 DeepSeek V4-Pro | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Qwen3.6-27B | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

只有当两臂 CPCD 各 159 条、ESConv 各 2,775 条、CPCD judge 各 159 条均完整，且模型身份验证通过后，才允许把“待测”替换成正式数字。
