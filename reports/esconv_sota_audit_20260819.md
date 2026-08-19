# ESConv 论文、模型与 Strategy Accuracy 审计

日期：2026-08-19  
状态：完成第一轮论文与代码可用性审计；Google Scholar 被引数等待人工直连复核  
目标：判断 `53.53%` 是否是可用于本仓库的最佳 ESConv Accuracy，并确定下一批真正值得租 GPU 重跑的模型。

## 结论先说

`53.53%` 是真实的论文数字，来源是 ACL 2026 的 **Causal-ESC**。但它不能继续在本项目里显示为“ESConv 统一最高准确率”，因为论文采用了自己的离线策略学习协议：随机抽取 ESConv 的 20% 用于 policy training 和 style demonstration pool，未公布与本仓库冻结 2,775 个 supporter turn 完全一致的 test IDs、随机种子和评测脚本。它属于“论文特定 policy-learning 结果”，不是本仓库冻结 benchmark 的可替代基线。

本轮检索到的最高已发表 headline 是 **DPPLM 的 58.03% strategy accuracy**。然而目前只能从论文/机构元数据确认数字，拿不到完整协议、代码、权重和评测脚本，因此它只能叫“最高已发表 headline”，不能叫“已验证 SOTA”，也不值得现在就租服务器盲目重构。

真正可执行的候选分成两类：

1. **立即能跑的 strategy planner：EmoDynamiX。** 代码、ESConv checkpoint 下载项和测试脚本都公开。论文主指标是 Macro-F1 27.70、Weighted-F1 32.71，并没有报告 Accuracy，因此我们必须把 checkpoint 接到冻结 2,775 行上自行测。
2. **端到端标准 ESConv 候选：PRCCF。** 论文报告 40.72% Accuracy，并描述标准 80/10/10 ESConv split；但论文给出的 GitHub 仓库在 2026-08-19 返回 404，当前无法下载代码或权重，暂时阻塞。

所以目前最正确的状态不是把 `53.53` 换成 `58.03`，而是：

- 冻结榜继续保留已复算的 **BlenderBot Joint 19.60% / 2,775**；
- 新建协议分层 registry；
- 第一优先租一张约 24 GB VRAM 的 CUDA GPU，重跑 **EmoDynamiX checkpoint → frozen 2,775**；
- DPPLM、SAGE、Causal-ESC 和 PRCCF 在关键材料释放前不租卡盲跑；
- 未来“静室”核心的替代决策必须同时通过策略、回复质量、安全、延迟、成本和统计显著性门禁。

## 1. 为什么不存在一个可以直接排序的“ESConv Accuracy 榜”

同样写着 ESConv 和 Accuracy，实际可能至少有以下差异：

- 原始 8 类还是合并/新增过的策略标签；
- 单个 supporter turn，还是未来策略序列；
- 官方 80/10/10 split，还是随机抽样的 policy-learning 数据；
- 完整 2,775 个 supporter turn，还是论文自己重构的样本；
- exact-match Accuracy，还是 Macro-F1 / Weighted-F1；
- 模型直接预测策略，还是由 API 生成回复；
- 是否把当前 target response 或 target strategy 泄露进输入。

因此，本审计禁止维护一个跨协议 global leader。只有使用本仓库冻结文件、同一 2,775 个 task IDs、原始 8 类标签和同一 target construction 的运行，才允许进入正式榜。

## 2. 当前论文结果分层

| 层级 | 论文/模型 | 论文数字 | 现在能否当本项目 SOTA | 代码/权重状态 |
|---|---|---:|---|---|
| 最高 headline、协议未知 | DPPLM | Strategy ACC 58.03 | 否 | 未找到代码、权重、完整评测协议 |
| 论文特定 offline-policy | Causal-ESC, ACL 2026 | ACC 53.53 | 否 | 未找到对应 ACL 2026 代码；生成还依赖 GPT-4o/DeepSeek API |
| 近期 8 类 ESConv 论文 | SAGE, 2026 | ACC 46.80；Macro-F1 38.95 | 不能直接进入冻结榜 | 未找到代码/权重；数据 on request |
| 修改标签/规划协议 | MultiESC, EMNLP 2022 | ACC 42.01 | 否 | 代码与数据公开；无训练后 checkpoint，需要多阶段训练 |
| 标准 split 的端到端候选 | PRCCF, 2026 | ACC 40.72 | 复现后才可能 | 论文声称公开，但仓库当前 404 |
| 可立即运行的 planner | EmoDynamiX, NAACL 2025 | Macro-F1 27.70；Weighted-F1 32.71 | 必须自行测 Accuracy | 代码、checkpoint 与测试脚本公开 |
| 外部策略测试 | CSO, Findings EMNLP 2025 | 主表 LoRA CSO-DPO：Macro-F1 35.77；Weighted-F1 52.34 | 否 | 数据构造代码公开；无训练权重；测试不是 frozen ESConv |
| 本仓库正式冻结结果 | BlenderBot Joint, 2021 checkpoint | ACC 19.60 / 2,775 | 是，当前正式 baseline | 已复现 |

### 2.1 DPPLM：58.03 是“最高 headline”，不是已验证冠军

论文为 *Applying Emotion-Cause Entailment for Help-Seeker Guidance in Emotional Support Conversations*，模型称 DPPLM。可访问元数据给出的结果为：

- Strategy Accuracy 58.03%
- Emotion Accuracy 65.42%
- BERTScore 0.8451
- ROUGE-L 12.10
- Distinct-1 4.63
- Distinct-2 32.24

阻塞点：

- 未取得完整可执行代码和权重；
- 无法确认 exact split、测试条数、标签映射和 target unit；
- 无法把 58.03 与 53.53、46.80 或冻结 19.60 做配对统计；
- 目前不应为它租 GPU，因为没有可运行对象。

登记状态：`unknown_protocol_headline`。

### 2.2 Causal-ESC：53.53 的来源已确认，但协议不同

ACL 2026 Table 2 报告：

| 模型 | ACC | BLEU-2 | BLEU-4 | Dist-1 | Dist-2 | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Causal-ESC | 53.53 | 9.54 | 4.07 | 8.25 | 40.86 | 22.15 | 10.23 |

论文附录进一步说明：

- 使用 GPT-4o 和 DeepSeek API 做 strategy-styled response generation；
- 对 ESConv 随机抽取 20% 用于 policy training 和 style demonstration pool；
- 未提供与本仓库 frozen test 完全一致的 test ID list、seed、checkpoint 和评测脚本。

因此 `53.53` 可以写成：

> Causal-ESC 论文特定 offline-policy 协议下的 strategy accuracy = 53.53%。

不能写成：

> ESConv 当前最高 Accuracy = 53.53%。

登记状态：`paper_specific_policy_learning`。

### 2.3 SAGE：46.80 很强，但当前不可下载复现

SAGE 报告 Strategy Accuracy 46.80%、Macro-F1 38.95%、Distinct-2 39.86，并使用 emotion-state strategy prediction、strategy-specific trie-constrained decoding、COMET 与 HEAL knowledge fusion。它对改进路线很有价值，但出版页面目前仅写数据可按需提供，未找到公开代码和 checkpoint。

登记状态：`paper_reported_standard_esconv_8class`，复现状态：blocked。

### 2.4 PRCCF：标准 split 候选，但“公开代码”当前失效

PRCCF 论文报告：

- Accuracy 40.72
- PPL 13.10
- BLEU-4 3.55
- ROUGE-L 19.78

论文描述标准 ESConv 80/10/10 split，并写明单张 NVIDIA 3090 Ti；从协议角度看，它比 53.53 更接近本仓库要复现的端到端 baseline。

但论文给出的 `YancyLyx/PRCCF` 在审计当天返回 404。不能因为摘要写着 code public 就把它登记成可运行。

登记状态：`blocked_repository_unavailable`。

### 2.5 EmoDynamiX：本轮最值得先实际跑的模型

EmoDynamiX 是独立 strategy planner，而不是回复生成器。其公开仓库提供：

- ESConv checkpoint 下载项；
- pretrained submodules；
- `test_roberta_hg_esconv.sh`；
- `train_roberta_hg_esconv.sh`；
- Python quick-start 接口。

论文刻意使用 Macro-F1 / Weighted-F1 来处理类别不平衡，报告 27.70 / 32.71，没有给 Accuracy。正因为没有 Accuracy，它正好符合用户提出的第 3 项：下载模型并在统一数据上自行跑。

本分支新增 `scripts/run_emodynamix_frozen_esconv.py`，把这一步从“计划”推进为可审计执行入口。脚本固定外部 commit
`c9213d718a9684a5e05ce5daa947f9cbbfb7b927`，严格复刻作者的输入构造：

- 在对话开头加入作者代码中的 `<START>` 伪 turn；
- 只保留 target 前最后 5 个 turn；
- 历史 supporter strategy 可以输入，但当前 target strategy 和 target response 永不输入；
- 只允许两项显式标签拼写映射：`Questions -> Question`、`Other -> Others`；
- checkpoint、frozen test、adapter 和逐条输入都记录 SHA-256；
- 支持续跑，但发现 checkpoint、commit 或 adapter protocol 改变时拒绝混写；
- 输出与现有 paired analyzer 兼容的 JSONL。

正式运行形式：

```bash
python scripts/run_emodynamix_frozen_esconv.py \
  --emodynamix-repo /path/to/EmoDynamiX-v2 \
  --checkpoint /path/to/released-esconv-checkpoint \
  --test-file /path/to/frozen/esconv/test.txt \
  --output results/esconv/emodynamix_frozen2775.jsonl
```

然后使用现有 paired analyzer，并将 `--n` 设为 `2775`，与已冻结 baseline prediction arm 做逐条比较。

第一轮正式 GPU 任务应当是：

1. checkout 上述外部 commit；
2. 下载 checkpoint，计算并保存文件 SHA-256；
3. 先用少量 `--limit` smoke test 检查依赖和输出；
4. 用本仓库 frozen 2,775 task IDs 完整推理；
5. 计算 Accuracy、Macro-F1、Weighted-F1、逐类 F1、混淆矩阵；
6. 按 dialogue cluster 做 bootstrap 95% CI；
7. 与 BlenderBot Joint、DeepSeek/Qwen arms 做配对 McNemar；
8. 保存 GPU 型号、运行时、环境锁、完整 JSONL 和 manifest。

登记状态：`adapter_ready_checkpoint_download_pending`。

### 2.6 MultiESC 和 CSO 为什么不能直接当冠军

MultiESC 的 42.01 来自 lookahead strategy planning 管线；其标签处理和预测对象不等同于冻结 8 类单 turn exact-match。代码和数据公开，但仓库没有训练后权重，多阶段命令还显式使用两张 GPU。

CSO 公开的是 MCTS、偏好数据构造和 ESC-Pro 数据入口；论文主表的 LoRA CSO-DPO 结果为 Macro-F1 35.77、Weighted-F1 52.34，附录还报告了不同优化变体，但这些数字都来自 ExTES 派生策略测试，不是 frozen ESConv Accuracy。它适合以后做 preference optimization，不适合做当前 Accuracy baseline。

## 3. Google Scholar cited-by 审计状态

用户要求论文与 cited-by 数全部以 Google Scholar 为准。本次环境无法直接打开 Google Scholar 结果页，因此：

- 所有 `cited_by_count` 保持 `null`；
- 没有拿 Semantic Scholar、Scopus、ResearchGate 或出版商引用数冒充 Google Scholar；
- 每篇论文的精确 Scholar 查询字符串已写入 JSON registry；
- 待人工直连 Google Scholar 后填写 `count + result URL + verified_at`；
- cited-by 数只用于了解影响力，不用于判断模型成绩是否可比较。

这是有意的“不填”，不是漏项。错误地填一个非 Scholar 数字，会破坏整个证据链。

## 4. GPU 复现队列

### P1：EmoDynamiX frozen-2775

这是唯一已经同时满足代码、checkpoint 下载项、公开测试入口的近期 planner。

必须产物：

- 外部 commit、环境锁、checkpoint SHA-256；
- 2,775 条逐条 prediction JSONL；
- 显式 8 类映射表；
- Accuracy / Macro-F1 / Weighted-F1 / confusion matrix；
- dialogue-cluster bootstrap CI；
- 完整运行日志与 GPU/耗时记录。

### P2：MultiESC paper-native reproduction

先复现其自己的 42.01，再讨论是否能映射回原始 8 类。禁止边改标签边声称复现论文。因为没有 checkpoint，需要训练；作者命令的若干阶段使用两个 CUDA device。

### P3：PRCCF

先等待官方仓库恢复或作者提供归档。代码不可用前不租卡。

### P4：CSO preference stage

只作为训练改进，不作为 ESConv Accuracy arm。先审计 ESC-Pro 与冻结 benchmarks 的污染风险。

### 暂停：DPPLM / SAGE / Causal-ESC

缺少代码、权重或 exact protocol。继续租卡只会变成“猜论文实现”，无法形成可审计证据。

## 5. 基于最好思想改进，而不是照抄一个不可复现模型

建议的新“静室”开源核心采用 **planner + generator + safety gate**：

### A. 显式策略规划器

起点：EmoDynamiX。

加入：

- 类别平衡 loss；
- 概率校准和 abstention；
- Causal-ESC 启发的 propensity / outcome 双模型，但只在 train split 上训练；
- emotion-cause 与 emotion trajectory 特征；
- 预测完整 8 类概率，不只输出 argmax。

### B. 开源生成器

使用可本地部署的 instruction-tuned 开源模型，输入：

- 对话上下文；
- planner 概率和最终策略；
- 检索证据；
- 安全约束。

先做 SFT，再做 preference optimization；不能先拿冻结 test 调 prompt 或训练。

### C. 检索与知识模块

吸收 SAGE / PRCCF 的可解释思想：

- persona-aware retrieval；
- emotion-trajectory retrieval；
- causality-aware knowledge filtering；
- strategy-specific retrieval；
- 只从 train-side 索引检索，避免 test response 泄漏。

### D. CSO 式偏好优化

在 planner + SFT generator 固定后，再使用 MCTS/偏好对训练。CSO 的论文分数不能继承给新模型，必须重新跑全部冻结 benchmark。

### E. 安全 fallback

模型部署在心理支持场景，不能因为策略 Accuracy 上升就自动取代 API。必须有：

- 风险分类与安全模板；
- 低置信度回退；
- 工具/事实检索边界；
- 失败日志和人工审计；
- 可一键切回当前 API 的 rollback。

## 6. 取代 DeepSeek API 的门禁

新模型至少同时满足：

1. frozen 2,775 strategy Accuracy、Macro-F1、Weighted-F1 全部报告；
2. 对当前 DeepSeek arm 做逐条配对比较，而不是只比两个百分数；
3. Accuracy 差值的 dialogue-cluster bootstrap 95% CI；
4. exact McNemar + Holm correction；
5. 未见场景的盲评回复质量；
6. 安全任务不得退步；
7. invalid output、延迟、吞吐、成本、宕机恢复满足部署要求；
8. 所有训练数据与 benchmark 去污染记录可审计；
9. 保留 API fallback 和 rollback。

任何一个论文 Accuracy 都不能单独触发生产替换。

## 7. 本分支新增的审计资产

- `reports/esconv_sota_registry_20260819.json`  
  机器可读论文/模型/协议/指标/代码权重状态；强制按 track 分层。
- `scripts/check_esconv_sota_registry.py`  
  阻止跨协议 global leader、伪造 Scholar 数字和非法百分比。
- `scripts/run_emodynamix_frozen_esconv.py`  
  已完成的 frozen 2,775 适配器，固定外部 commit、输入窗口、标签映射、resume 和哈希证据。
- `configs/esconv_reproduction_queue_20260819.json`  
  GPU 复现优先级、产物和阻塞条件。
- 本报告  
  给出研究结论、改进路线和替代门禁。

## 8. 当前最终判定

- **58.03%**：最高已发表 headline；不可复现，协议未核实。
- **53.53%**：真实论文值；不是 frozen 2,775 协议。
- **46.80%**：SAGE 强结果；无代码/权重。
- **42.01%**：MultiESC 修改协议；有代码、无权重。
- **40.72%**：PRCCF 标准 split 候选；仓库当前 404。
- **EmoDynamiX**：本轮唯一“代码 + checkpoint + 测试脚本”明确可执行的近期策略规划器。
- **19.60%**：当前唯一已在本仓库 frozen full test 上完成复算的正式基线。

下一步不是改榜单数字，而是完成 `EmoDynamiX → frozen 2,775` 的第一次外部 checkpoint 统一重跑。
