# 给 ChatGPT 5.6 Pro Deep Research 的独立复核 Prompt

下面内容可以整段复制。请优先把仓库分支、PR #1–#4 和本文提到的公开聚合文件一起交给它；不要上传本地 `results/`、许可数据、模型权重、`.env`、逐行输出或 private diagnostics。

---

你是一名严格的机器学习评测审稿人、心理健康 NLP 研究员和可复现性审计员。请对下面这个公开仓库做一次**独立、批判性、逐证据复核**，不要顺着我的结论说，也不要为了安慰我虚构提升。

仓库：

- https://github.com/AllenLi441/mental-health-llm-eval
- 当前跨 benchmark 汇总分支：https://github.com/AllenLi441/mental-health-llm-eval/tree/codex/cross-benchmark-accuracy-family-v1
- Draft PR 链：https://github.com/AllenLi441/mental-health-llm-eval/pull/1 、/pull/2 、/pull/3 、/pull/4

我的最高目标不是只优化 PsySUICIDE，而是：

> 设计一套可复用的模型架构与验证纪律，让它在 CPsyExam、EmoBench、PsySUICIDE、IMHI 九任务、MDD-5k、EATD、MentalManip、CBT-Bench 等有论文依据的数据集上，分别得到更好的、可复现的准确率或论文主指标。

请接受以下硬约束：

1. 不得把不同数据切分的分数相减后宣称提升。
2. 不得使用 official test、frozen holdout 调参；只有预注册后的一次性确认才可看。
3. 不得把一个 PsySUICIDE 11 类分类头的提升推断成其他 benchmark 提升。
4. 选择题、单标签分类、多标签分类必须用各自正确的任务头和评分协议。
5. 没有同一批样本上的对照预测与配对检验，就只能作描述性报告。
6. 只根据公开代码、聚合结果、commitment、PR/Release 证据判断；不要要求公开许可原始文本、逐行敏感输出、API key 或模型权重。

已知事实，请逐项复核，发现不一致时以仓库证据为准：

## 1. CPsyExam

- V4-Pro：3,307 / 3,902，accuracy 84.7514%。
- V4-Flash：3,252 / 3,902，accuracy 83.3419%。
- 同题配对差值：+1.4095pp；exact McNemar p=0.0057226；paired bootstrap 95% CI=[0.4357pp, 2.4090pp]。
- 这是当前少数可以严谨宣称“同一批题上提升”的结果。
- 但未来协议审计发现旧 parser 可能把 `Answer: B` 中单词 `Answer` 的 A 误读成答案。现有不可变 Release 不能被事后改写，也无法仅凭聚合结果量化此 bug 影响了多少题。
- V4-Pro 分组：KG-SCQ 90.69%、KG-MAQ 75.16%、CA-SCQ 81.33%、CA-MAQ 63.50%；多选和案例分析仍明显更弱。

请判断：这个 +1.4095pp 的统计结论在旧协议内是否成立；parser 风险如何限制外部效度；下一版应该如何在不重复用 test 调参的前提下修复、建立开发集、做检索或知识增强。

## 2. PsySUICIDE

- DeepSeek V4-Pro + taxonomy 在 official test：accuracy 88.11%，macro-F1 0.6371。
- 同一 official test 的 V4-Flash + baseline：accuracy 82.86%，macro-F1 0.5494；冻结 campaign 内 taxonomy 分别提高 +5.26pp 与 +0.0878。
- 论文 RoBERTa-large 参考点值：accuracy 91.69%，macro-F1 0.6976；协议/模型类型并不完全相同。
- 专用 RoBERTa v1 三 Seed在 official-valid：accuracy 均值 94.01%，macro-F1 均值 0.7380。
- 选中 Seed 43 在另一个 frozen internal holdout：accuracy 93.43%，macro-F1 0.7275。
- **88.11% 来自 official test；93.43% 来自 frozen internal holdout，不能宣称提升 5.32pp，也没有同一 holdout 上的 LLM 对照。**
- v1 holdout 的极稀有“自杀准备行为”只有 2 条，F1=0，说明整体 accuracy 高不等于所有风险类别解决。
- accuracy-first v2 已暂停：A 臂只完成一个 Seed-42 inner-dev 结果（accuracy 93.29%，macro-F1 0.6250）；B 臂停在 3,560/4,680；C、D 未开始。A 与 v1 的 valid/holdout 也不是同一切分，所以 v2 不能宣称提高或下降。

请判断：v1 真实证明了什么、没有证明什么；v2 是否值得以后续跑；类别极不平衡下 accuracy、macro-F1、关键风险 recall 应怎样共同设门槛。

## 3. IMHI 九任务

- 统一 zero-shot 候选当前 weighted-F1：DR 81.82、dreaddit 81.10、loneliness 73.24、IRF 63.87、MultiWD 71.15、SAD 55.15、CAMS 43.08、swmh 64.69、T-SID 74.40。
- 描述性比较：8/9 高于论文 ChatGPT zero-shot 点值，1/9 高于 MentaLLaMA-13B，0/9 高于各任务 fine-tuned 判别式。
- contrastive-v4 same-10 统一候选整体退步，已按门槛拒绝，不能只挑局部上涨的任务。

请重点分析为什么 0/9 仍未超过 fine-tuned 判别器；给出“共享 encoder/训练框架 + 每任务独立 head”的实现优先级、拆分纪律、类别不平衡策略、三 Seed 验证和停止规则。不要建议把 PsySUICIDE 11 类 head 直接复用。

## 4. EmoBench

- 当前 scoreboard 仍是 temperature-0 单次 proxy：EA 74.25%，EU 71.50%，不能冒充论文完整协议。
- 论文兼容脚本实现：EA 与 EU 各 400 题，每题 4 个选项排列，每排列 5 次随机采样多数票。
- 正确完整调用量是 **16,000 calls = 2×400×4×5**。早先 8,000 calls / $4.7851 的预算估计漏计一个任务；按 160-call smoke 校准并加 1.5 倍预留，更正为约 $9.5702。
- 完整付费复刻尚未运行，因此不能说已经超过或落后论文。

请审计：实现与论文还有哪些假设差异；应先做协议复现还是 prompt/模型优化；如何把开发集优化与最终论文 test 严格隔离。

## 5. 其他任务

- MDD-5k：当前 accuracy 57.25%，macro-F1 0.4051。
- EATD：accuracy 83.54%，macro-F1 0.6911，但 depressed-class F1 只有 0.48。
- MentalManip：accuracy 77.80%，positive-class F1 0.8510。
- CBT-Bench 当前是 top-1 命中代理：CD 58.22%、PC 89.13%、FC 69.64%，不等同论文多标签 F1。

请判断哪些任务能复用分类训练底座，哪些必须重建论文协议；不要把不兼容指标直接排名。

## 要求输出

请用中文给出以下 8 部分，并给每个判断附公开文件、commit、PR、Release 或论文链接：

1. 一页执行摘要：当前模型体系到底是什么，不是什么。
2. 按 benchmark 的证据表：当前分数、对照、是否同切分/同协议、结论等级（已证明提升 / 描述性更好 / 未提升 / 不可比 / 未完成）。
3. 已经解决的问题：协议、解析、配对、泄漏门禁、可复现、成本或模型效果分别列出。
4. 尚未解决的问题：尤其是错误预测集中在哪里；只能从聚合证据推断的地方必须标“推断”，不能编造逐行原因。
5. 对此前工作是否偏离“跨 benchmark 提升”目标的评价；PsySUICIDE 长训练哪些有价值、哪些机会成本过高。
6. 推荐的模型架构：至少区分分类任务家族、MCQ/检索家族、多标签家族、共同评测控制层；说明能共享什么、不能共享什么。
7. 未来 30 天分阶段实施计划：先做什么、每步需要什么数据、训练量、预算、成功门槛、失败停止规则；优先真正可能提高多个论文任务的工作。
8. 列出所有必须撤回或禁止的表述，尤其是“88.11%→93.43%=+5.32pp”“一套 RoBERTa 通吃所有论文”“EmoBench 全量只需 8,000 calls”。

最后请给出明确总判定：目前有没有证据证明“整体模型架构在所有论文 benchmark 上提高了”；哪些单项已有可信提升；下一笔计算时间和 API 预算应该花在哪里，为什么。

---
