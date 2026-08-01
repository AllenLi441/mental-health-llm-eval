# mental-health-llm-eval 模型架构现状、工作复盘与路线图

日期：2026-08-01

范围：公开评测仓库的代码、预注册、聚合结果与 Git 证据；不读取或发布许可原始文本、逐行敏感输出、私有 logits、密钥或模型权重。

## 一句话结论

目前**没有证据证明一套统一模型架构已经让所有论文 benchmark 一起提高**。已经有两项冻结 campaign 内的可信单任务提升：CPsyExam 中 V4-Pro 在同一 3,902 题上比 V4-Flash 高 1.4095pp；PsySUICIDE 中 V4-Pro taxonomy 相对 V4-Flash baseline 的 official-test accuracy 高 5.26pp、macro-F1 高 0.0878。PsySUICIDE 的专用 RoBERTa 另行证明了“这个任务可训练出高分专用分类器”，但它既不能直接迁移到其他任务，也不能把 official-test 的 88.11% 与 internal-holdout 的 93.43% 相减成 +5.32pp。

真正应该复用的是训练框架、数据隔离、预注册、配对验证和发布边界；每个任务仍需要兼容的输入适配器、输出头与主指标。

## 当前真实状态

| Benchmark | 当前公开点值 | 证据等级 | 真实结论 |
|---|---:|---|---|
| CPsyExam | V4-Pro 84.7514%；V4-Flash 83.3419% | 同一 3,902 题、预注册、配对 | Pro +1.4095pp，McNemar p=0.00572；旧 parser 风险限制外部效度 |
| PsySUICIDE API | taxonomy 88.11% / 0.6371；Flash baseline 82.86% / 0.5494 | 同一 official test、预注册、配对 | taxonomy accuracy +5.26pp、macro-F1 +0.0878；仍低于论文 fine-tuned RoBERTa 点值 |
| PsySUICIDE RoBERTa v1 | valid 三 Seed 94.01% / 0.7380；internal holdout 93.43% / 0.7275 | 专用监督模型；不同于 official test | 支持未见内部样本上的描述性泛化；没有同一 holdout 的 LLM 对照，不能算 +5.32pp |
| PsySUICIDE v2 | A: 93.29% / 0.6250；B 3,560/4,680；C/D 未开始 | 未完成 screen | 不能选择训练臂，不能宣称提高；已暂停且不会自动重启 |
| IMHI 九任务 | weighted-F1 43.08%–81.82% | 统一 zero-shot 点值 | 8/9 高于 ChatGPT zero-shot 点值，1/9 高于 MentaLLaMA，0/9 高于 fine-tuned 判别器 |
| EmoBench | EA 74.25%；EU 71.50% | temp-0 单次 proxy | 不是论文 5×4 协议；完整 16,000-call 尚未运行 |
| MDD-5k | accuracy 57.25%；macro-F1 0.4051 | 当前套件协议 | 明显有改进空间，不能只看多数类 accuracy |
| EATD | accuracy 83.54%；macro-F1 0.6911；depressed F1 0.48 | 当前套件协议 | 总体 accuracy 掩盖阳性类偏弱 |
| MentalManip | accuracy 77.80%；positive F1 0.8510 | 当前套件协议 | 可作为专用二分类训练候选，尚无同切分新模型提升证据 |
| CBT-Bench | CD 58.22%；PC 89.13%；FC 69.64% | top-1 命中 proxy | 与论文多标签 F1 不兼容，先修协议再谈提升 |

这里的 `/` 对 PsySUICIDE 表示 `accuracy / macro-F1`。跨论文点值只用于定位短板，不能替代同样本、同协议的配对比较。

## 已经解决了什么

1. **CPsyExam 从 599 pilot 扩展到 3,902 全量配对。** V4-Pro 与 V4-Flash 在同一题目集合上运行，建立了预注册、不可变 Release、配对 bootstrap、exact McNemar 和等价性边界。
2. **PsySUICIDE API 模型比较进入同协议确认流程。** taxonomy 候选相对 Flash 的比较使用同一 official test，并保留 paired 分析；被否决的 taxonomy-v2 与 IMHI contrastive-v4 没有因局部好看而替换赢家。
3. **PsySUICIDE 专用监督训练可运行。** v1 完成三 Seed、checkpoint 冻结、valid 门槛和一次性 internal holdout；这证明专用分类器路线有可行性。
4. **数据泄漏与发布边界更清楚。** optimization、inner-dev、frozen holdout、official valid/test 的用途被区分；公开仓库只放聚合指标、commitment、代码与协议。
5. **统一 scoreboard 不再随意混用最新文件。** 它固定读取具名聚合资产，并区分 accuracy、macro-F1、weighted-F1 与 proxy。
6. **发现并更正 EmoBench 成本门禁错误。** 完整协议是 16,000 calls，不是 8,000；160-call smoke 观测不变，预算预留由 `$4.7851` 更正为 `$9.5702`。
7. **发现 CPsyExam 旧解析器风险。** 全局扫描 A–E 可能把 `Answer` 的 A 当答案。新解析器只接受严格 JSON、锚定答案字段或纯字母；历史不可变 Release 不被事后重写。

## 没有解决什么

### 没有跨 benchmark 的统一提升证据

过去两天的大部分计算集中在 PsySUICIDE 专用 RoBERTa。它的 11 类输出头只适用于该风险分类任务，不能用于 CPsyExam、EmoBench，也不能直接用于标签不同的 IMHI、MDD、EATD 或 MentalManip。

### PsySUICIDE 极稀有类仍失败

v1 internal holdout 的“自杀准备行为”support=2、F1=0。v2 A 的 inner-dev 中“自杀准备行为”support=2、F1=0，“自伤意图”support=3、F1=0。大类 accuracy 约 93% 并不意味着关键小类被解决。需要真人审核的稀有表达与硬负例扩充，不能靠重复采样创造新语言模式，也不能让 AI 冒充人工金标。

### CPsyExam 错误原因尚不能从公开聚合精确分解

V4-Pro 的分组 accuracy 为：KG-SCQ 90.69%、KG-MAQ 75.16%、CA-SCQ 81.33%、CA-MAQ 63.50%。多选与案例分析更弱是事实；但知识缺失、推理错误、位置偏差与 parser bug 各占多少，公开聚合不足以回答。旧 run 的 17 个 invalid 也不包含“被误解析成合法 A”的情况。

### IMHI 仍输给每任务 fine-tuned 判别器

当前统一 prompt 0/9 超过论文 fine-tuned 模型。最弱点值包括 CAMS 43.08%、SAD 55.15%、IRF 63.87%。这说明继续统一改 prompt 的预期收益有限；更合理的是每任务独立训练 head，并在只来自训练区的 inner-dev 上处理阈值和类别失衡。

### EmoBench 与 CBT 仍未同论文协议闭环

EmoBench scoreboard 仍是 temp-0 proxy，完整 5×4 需要 16,000 calls；CBT 当前 top-1 命中不等同多标签 F1。协议不一致时，所谓“超过论文”不成立。

## 为什么此前会长期跑 PsySUICIDE

此前阶段顺序写成“先完成 PsySUICIDE Stage A，再做 accuracy-first Stage B，之后才盘点其他 benchmark”。自动任务忠实执行了这个顺序，因而 A/B/C/D 四臂各 10 epochs 的顺序训练占用了大量时间。

这段工作不是完全无用：它建立了监督训练、checkpoint 完整性、三 Seed、数据边界与长尾诊断底座。但执行优先级偏离了最高目标。到暂停时，v2 只有 A 完成，B 到 3,560/4,680，C/D 未开始；因此计算成本已经发生，却尚未形成 v2 选择结论。

## 正确的模型家族

```text
共同评测控制层
├─ 数据许可 / split registry / commitment / 去泄漏
├─ 预注册 / 同样本 baseline / 配对统计 / 三 Seed
└─ provenance / 成本 / 敏感性审计 / aggregate-only 发布

任务家族 A：判别式分类
├─ 共享中文 encoder 与训练器
├─ IMHI 每任务独立 head
├─ MDD、EATD、MentalManip 各自独立 head
└─ PsySUICIDE 11 类专用 head

任务家族 B：选择题与知识推理
├─ CPsyExam：subject-aware prompt + 严格结构化答案 + 合法知识检索
└─ EmoBench：官方 prompt + 5×4 聚合 + 选项排列鲁棒性

任务家族 C：多标签
└─ CBT-Bench：sigmoid/BCE 或集合生成 + paper-compatible multilabel 指标
```

可以共享的是 encoder 初始化、训练器、采样/损失消融、日志与验证纪律；不能共享的是 PsySUICIDE 分类头、标签空间、prompt、决策阈值和论文主指标。

## 接下来怎么做

### 阶段 0：先把事实与协议锁死

- 保留 PsySUICIDE v2 checkpoint，但不续跑、不看 holdout。
- 提交 CPsyExam parser 合成回归与新 prompt profile；历史 Release 保持不可变。
- 将 EmoBench 16,000-call 和 `$9.5702` 门禁写入代码自检与报告。
- 为每个 benchmark 建立一行 spec：合法训练数据、optimization/inner-dev/test、主指标、论文 baseline、现有结果、是否可配对。

成功标准：全套 selftest、聚合审计、敏感性扫描通过；没有 raw/secrets/weights 进入 Git。

### 阶段 1：优先 IMHI 九任务分类家族

- 只对有合法本地训练标签的任务，从各自 train 构建 stratified inner-dev。
- 共享 RoBERTa encoder 代码，但每任务单独 head、checkpoint 和预注册。
- 先跑小型单 Seed 消融：自然 CE、加权 CE、focal；不碰 official test。
- 过门槛后再跑 3 Seeds，并与当前 uniform-v3u 在同一确认集做配对。

建议门槛：每任务 primary weighted-F1 提高；关键少数类 recall/F1 不显著退化；九任务必须全报，不能只挑赢家。若 3 个优先任务都不能在 inner-dev 超过当前点值，停止扩展而重新审计标签/协议。

### 阶段 2：EATD、MentalManip、MDD

- 复用阶段 1 的训练框架，不复用输出头。
- EATD 以 depressed-class F1 与 macro-F1 为共同门槛，防止 83.54% accuracy 掩盖阳性类。
- MDD 以 macro-F1 为主；MentalManip 同时看 positive F1、balanced accuracy 和校准。
- 每个任务独立预注册、三 Seed、同切分 baseline；一个任务上涨不能推断另一个上涨。

### 阶段 3：CPsyExam v5

- 不再用已经看过的 3,902 official test 调 prompt。
- 在训练/开发材料中建立新 development split；比较 legacy prompt、subject-aware strict JSON、合法检索增强。
- 单选与多选分开报告；重点优化 KG-MAQ 与 CA-MAQ。
- 冻结 parser、retrieval corpus hash、prompt 与模型后，才申请下一次独立确认集。

### 阶段 4：EmoBench 与 CBT

- EmoBench 先完成 16,000-call 论文协议复现，预算门槛至少 `$9.58`；复现结果与优化实验分开。
- 任何 prompt/模型优化只用 development split，冻结后再跑一次 test。
- CBT 先恢复 paper-compatible 多标签输出与指标，再训练多标签 head；旧 top-1 只能保留为内部 proxy。

### 阶段 5：最后再决定是否续跑 PsySUICIDE v2

只有当跨 benchmark 分类框架已证明可复用、且 v2 剩余计算仍能回答“采样/损失哪一个提高 accuracy”时，才从相同 B checkpoint identity 安全续跑。A 单臂不能参与结论，C/D 未完成前不能选方案。

## 禁止继续使用的表述

- “PsySUICIDE 从 88.11% 提升到 93.43%，提高 5.32pp。”——不同 split，无同样本对照。
- “RoBERTa 分词器提高了全部论文测试。”——训练的是完整 PsySUICIDE 专用分类器，不是通用分词器。
- “一套 RoBERTa 可以直接通吃所有任务。”——最多共享底座与训练器，必须有任务专用 head。
- “EmoBench 完整重算是 8,000 calls / `$4.79`。”——正确是 16,000 calls / 约 `$9.57` 预留。
- “EmoBench 已经超过论文。”——当前只有 proxy，完整协议未运行。
- “CBT 89.13% 超过论文多标签 F1。”——top-1 proxy 与多标签 F1 不可比。
- “PsySUICIDE v2 已经提高准确率。”——四臂 screen 未完成。

## 最终判定

- **整体架构是否已提高所有 benchmark：否。**
- **可信的单项提升：CPsyExam V4-Pro 相对 V4-Flash，在旧冻结协议内 +1.4095pp；PsySUICIDE API taxonomy 相对 Flash 也有同题比较，但不能与专用 RoBERTa 的不同 split 分数相减。**
- **监督模型路线最值得跨任务迁移的部分：训练框架、验证纪律和独立 head 设计，不是 PsySUICIDE 权重。**
- **下一笔主要计算应投向 IMHI 每任务分类器的低成本、同切分开发验证；下一笔 API 预算应先完成 EmoBench 协议复现或 CPsyExam 独立开发集实验，而不是继续无限扩大 PsySUICIDE 单任务训练。**

## 证据入口

- `results-summary/cpsyexam-v4-full-paired.summary.json`
- `reports/model_optimization_v2_20260728.md`
- `reports/psysuicide-roberta-v1-valid-selection.json`
- `reports/psysuicide-roberta-v1-holdout-result.json`
- `reports/model_optimization_accuracy_first_v2_20260731.md`
- `reports/imhi-uniform-v4-smoke-20260728.json`
- `reports/emobench-paper-protocol-smoke-20260728.json`
- `scripts/scoreboard.py`
- `PUBLISHING_NOTE.md`
