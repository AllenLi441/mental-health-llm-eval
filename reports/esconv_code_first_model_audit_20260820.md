# ESConv code-first 模型审计与改进决定

核查日期：2026-08-20（America/Los_Angeles）

## 先纠正旧榜单

旧榜单回答的是“论文曾经报过多高的数字”，不是“今天能不能下载、重跑和继续开发”。这两个问题不能混为一谈。本文只为工程选型服务；DPPLM、Causal-ESC、SAGE、CADSS 等没有可执行实现或权重的高分论文不参与基线选择。

一个项目只有同时满足以下条件才算“有实现代码”：

1. 仓库里存在与论文方法对应的模型源码，而不只是 README、PDF、prompt 或 `code forthcoming`；
2. 至少存在一个具体训练或推理入口；
3. 代码、论文、checkpoint、预处理数据、运行命令、依赖和许可证分别标记，不能用“有 GitHub 链接”代替这些证据；
4. 不同任务、标签空间和 split 的 Accuracy 不横排。

## 结论

当前最适合继续改进的 **ESConv 八类 next-strategy 工程基座是 EmoDynamiX**，不是论文 Accuracy 最高但没有代码的模型，也不是 MultiESC：

- [EmoDynamiX](https://github.com/cw-wan/EmoDynamiX-v2)有真实训练/测试源码和作者 checkpoint。本仓库已在作者原生 2,895-row test 上复现到 ACC 33.6097%、Macro-F1 27.7040%、Weighted-F1 32.7087%，后两项与[论文](https://aclanthology.org/2025.naacl-long.81/)对齐。
- 但作者 released checkpoint 的 train 与本仓库 frozen test 有 132 个对话、1,734 行重叠，因此只能用于作者原生复现和开发，不能把它在 frozen test 上写成公平成绩。正式候选必须从干净 train/dev 重新训练。
- [MultiESC](https://github.com/lwgkzl/MultiESC)有核心源码，适合贡献 lookahead/transition 思路；但没有论文 checkpoint，修改了标签体系，发布流程还有确定性错误，所以不能直接作为可复现基线，也不能把论文的 42.01 与 canonical 8-way Accuracy 比较。

唯一当前公平、可审计的 frozen-2,775 基线仍是原始 ESConv BlenderBot-small Joint：ACC 32.2162%、Macro-F1 21.8272%。它用于最终公平对照，不代表它是最好的现代开发架构。

## 真正带源码的候选

| 项目 | 真实源码 / 入口 | 论文 checkpoint | 任务口径 | 工程决定 |
|---|---|---|---|---|
| [论文](https://aclanthology.org/2025.naacl-long.81/) · [EmoDynamiX 代码](https://github.com/cw-wan/EmoDynamiX-v2) | `main.py`、train/test shell、图网络与预处理完整 | **有；已本地原生复现** | 原始 8 类 next-strategy；作者 70/15/15 split | **A：当前改进基座**；released 权重不得上 frozen 榜，需干净重训 |
| [论文](https://aclanthology.org/2021.acl-long.269/) · [原始 ESConv 代码](https://github.com/thu-coai/Emotional-Support-Conversation) | 数据、joint/vanilla 训练和生成代码 | 有公开模型；本仓库已 frozen 重跑 | joint generation 的首策略 token | **A：公平 frozen 基线** |
| [论文](https://aclanthology.org/2025.sicon-1.9/) · [ESConv-SRA 代码](https://github.com/navidmdn/esconv-sra) · [LoRA](https://huggingface.co/navidmadani/esconv_sra_llama3_8b) | 数据处理、LoRA 训练、分类器、CLI chat 和比较脚本 | **有 1.36 GB Llama-3.1-8B LoRA** | 扩展为 15 类的策略条件长对话生成/策略遵循，不是自动 canonical 8-way next-strategy | **A1-GEN：未来 generator 方法/权重供体**；不可进入八分类榜 |
| [论文](https://aclanthology.org/2022.emnlp-main.195/) · [MultiESC 代码](https://github.com/lwgkzl/MultiESC) | 五阶段策略序列、feedback、lookahead、生成源码 | **无** | 合并 Suggestion/Information、抽出 Greeting、排除剩余 Others；GT 实质 7 类 | **B：方法供体**；先 clean-room 移植，不按旧栈直接重训 |
| [论文](https://aclanthology.org/2023.findings-acl.420/) · [TransESC 代码](https://github.com/circle-hit/TransESC) | `main.py`、训练与 `--test` 入口、数据/metric/旧 Transformers 源码 | 无 | 状态转移控制的回复生成 | **B：可训练源码**；不是现成 checkpoint |
| [论文](https://aclanthology.org/2023.findings-acl.34/) · [PAL 代码](https://github.com/chengjl19/PAL) | `codes/`、persona extractor 和数据 | 无 | PESConv persona-conditioned generation | **B：不同数据/任务的方法供体** |
| [论文](https://aclanthology.org/2025.findings-emnlp.1209/) · [DecoupledESC 代码](https://github.com/Zc0812/DecoupledESC) | 有数据构造、评测和 API 推理片段；**没有 SFT/DPO 训练代码或 requirements** | 无 release | 原始 8 类 strategy planner + response generation | **C：部分源码/数据**；三个 vLLM shell 还有语法错误，当前不可训练复现 |
| [论文](https://arxiv.org/abs/2602.08826) · [AFlow 代码](https://github.com/chz2025/AffectiveFlow) | MCTS、路径抽取和 AFPO 源码很多，但当前训练脚本有 `IndentationError`，pipeline 引用缺失文件 | **无** | 从 ESConv seed 扩展并重新映射 taxonomy 的 synthetic validation 协议 | **C：ideas-only**；配置、轨迹和权重均未闭合，不租 8×A100 |

“有源码”不等于“已复现”。B 级条目必须从头训练，不能沿用论文数字当本地结果。

### 命令、依赖和许可状态

| 项目 | 作者公布/经核验的入口 | 依赖与资产状态 | 许可状态 |
|---|---|---|---|
| EmoDynamiX | `bash test_roberta_hg_esconv.sh`；`bash train_roberta_hg_esconv.sh` | checkpoint 已校验；桌面 ZIP 缺 RoBERTa/SDDP/ERC 资产；requirements 与 environment 版本冲突，需隔离环境 | 源码 MIT；ESConv 非商用研究；checkpoint 产品权利未核 |
| 原始 ESConv | 本仓库固定 wrapper：`scripts/setup_esconv_legacy_env.sh`、`scripts/train_esconv_2021_joint.sh`、`scripts/generate_esconv_2021_joint.sh` | 旧版 BlenderBot/PyTorch 环境，已有 frozen reproduction receipt | 上游明确 academic research only |
| ESConv-SRA | README 首条路径/参数有误；可用入口为 `PYTHONPATH=. python training/cli_chat.py --model_path navidmadani/esconv_sra_llama3_8b` | 需取得 gated Llama-3.1-8B base；作者实验 1×A100 80GB | 仓库无 LICENSE；adapter 为 OpenRAIL，另受 Meta Llama license 约束 |
| MultiESC | README 给出五阶段命令；第一步以 `python generate_strategy_norm_train.py ...` 开始，第二步原样必因 beam7/beam8 错误中断 | CUDA 10.1、PyTorch 1.8、Transformers 4.16.2；无 checkpoint/metric package | 仓库无整体 LICENSE；数据沿用 ESConv 限制 |
| TransESC | `python main.py`；`python main.py --test` | Python 3.7、PyTorch 1.8.2、Transformers 4.12.3、CUDA 11.1；数据需外部下载，无 checkpoint | 仓库许可未核清 |
| PAL | `codes/train.py`、`codes/infer.py` 是实际入口；作者未给一个闭合的顶层精确命令 | 需补本地模型与配置，无 checkpoint | 仓库许可未核清；PESConv 权利链另审 |
| DecoupledESC | 三个 `vllm_model_*.sh` 均含非法的带空格变量赋值，不能原样执行 | 无 requirements、训练代码或 checkpoint；论文训练称 4×RTX 4090 24GB | 仓库无 LICENSE |
| AFlow | pipeline/launcher 存在，但 `scripts/train_afpo.py` 当前有 `IndentationError`，且引用缺失脚本 | 无 checkpoint/trajectory；配置仍有 model/API placeholder；论文称 8×A100 | MIT |

## 伪“有代码”与必须排除的条目

| 项目 | 仓库实际内容 | 结论 |
|---|---|---|
| [CADSS / CPsDD](https://github.com/FakerBoom/CPsDD/tree/main/code) | `code/` 只有 `Prompts and Appendix.pdf` 和 `readme`；根 README 明写 CADSS/PGSim code 将来发布 | **不是代码候选** |
| [AugESC](https://github.com/thu-coai/AugESC) | 归档仓库只有 `LICENCE` 和 `README.md`，README 指向数据和微调生成模型 | **不是论文实现源码**；第三方策略分类 checkpoint 另作来源不明资产处理 |
| [MISC](https://github.com/morecry/MISC) | GitHub 只有 README；作者称未整理代码，另给外部 Drive | **仓库不可运行**；外部包取得并逐文件核验前不算通过 |
| [KEMI](https://github.com/dengyang17/KEMI) | GitHub 只有 README；模型/代码放在外部 Drive | **仓库不可运行**；同上 |
| TinyLlama / XLM-R / ModernBERT 第三方 HF 分类器 | 有或可能有八分类权重，但训练 split、输入模板、论文实现关系和 license 不闭合 | **checkpoint-only / unverified**；不能用模型卡数字决定工程基线 |
| DPPLM、Causal-ESC、SAGE | 论文数字存在，但没有可核验官方训练代码和 checkpoint | **paper-only**；只保留为思想来源 |

这也确认了用户对 CPsDD 的判断：目录名字叫 `code` 不代表里面有程序。

## 两个桌面项目的逐文件审计

### EmoDynamiX-v2-master

- 桌面 ZIP 没有 `.git`，但除 `.DS_Store` 外 83 个文件与官方 commit `c9213d718a9684a5e05ce5daa947f9cbbfb7b927` 逐路径、逐字节一致。
- 本地排序文件清单+内容的 SHA-256 commitment：`f326fc97d5a44d1e7bcd294303ebac44766370ff5b46b4c45d72e821694fd3d9`。
- 桌面 ZIP 本身没有 checkpoint、本地 RoBERTa 和在线 discourse/ERC 预训练资产；工作区已取得并校验 ESConv checkpoint，SHA-256 为 `6cb5b27356b566a9db1d071fa6cb969aba6fda8a3073632003b3c6b797daa4e7`。
- 作者 split 是 train 910 dialogues / 12,759 rows、valid 195 / 2,722、test 195 / 2,895，标签保持原始八类。
- 结论：`OFFICIAL_SOURCE_SNAPSHOT_VERIFIED`；源码真实，桌面 ZIP 不是单独开箱即跑包。

### MultiESC-main

- 除 `.DS_Store` 外 30 个文件与官方 HEAD `46834e134320253e51920a2f5d34f166335bd965` 逐字节一致；本地清单 commitment 为 `af20e319b7736cb1006fd1ddd350b896ad8479438a4483920abc2617e245692c`。
- 没有 `.bin/.pt/.pth/.ckpt/.safetensors`；`MODEL/` 与 `metric/` 只是下载说明。
- `generate_strategy_test.py` 先生成 beam 7，却马上读取 `*_beam8.pk`，README 的第 2 步原样会 `FileNotFoundError`。
- 发布代码固定老旧 CUDA/PyTorch/Transformers 私有 API；当前 Transformers 5 导入失败。策略验证 `acc1` 分母还从 1 开始，造成 N+1 偏差。
- 发布数据实际为 910/195/195（70/15/15），而非论文文字的 8:1:1；策略标签也不是 canonical 8-way。
- 结论：`OFFICIAL_SOURCE_SNAPSHOT_VERIFIED_BUT_NO_CHECKPOINT`；只移植方法，不接受 42.01 为统一榜成绩。

## 已完成的第一轮改进

第一轮没有盲目重训大模型，而是在已复现的 EmoDynamiX logits 上验证 MultiESC 启发的 train-only 策略转移先验和类别不平衡校正：

```text
adjusted_logit(y)
  = base_logit(y)
  + transition_weight × log P_train(y | strategy_history)
  - tau × log P_train(y)
```

所有先验只从 author train 12,759 行拟合；54 个固定配置只在 author valid 2,722 行选择；接口没有 test split，也不允许把当前 gold strategy 传入 reranker。完整机器凭据见 `reports/esconv_emodynamix_code_first_dev_20260820.json`。

| author-valid 结果 | Accuracy | Macro-F1 | Weighted-F1 | 状态 |
|---|---:|---:|---:|---|
| 原 EmoDynamiX logits | 33.6517% | 26.5243% | 32.1032% | base control |
| 只加 transition prior | 33.6517% | 26.5243% | 32.1032% | **无提升，拒绝**；最优回到 weight=0 |
| 正式 Macro-F1 选优：weight=0.1, tau=0.4 | 31.9618% | **27.6500%** | 32.4309% | Macro-F1 +1.1257 pp；Accuracy -1.6899 pp |
| 探索性 Pareto 点：weight=0.4, tau=0.4 | **33.7252%** | **27.1669%** | **32.6160%** | 三项均高于 base；未按主规则选中，需预注册复验 |

这不是“新 SOTA”，也不是最终模型：它只证明类别先验校正和轻量历史规划值得进入干净重训。尤其是只加转移先验的实验失败了，报告保留了这个负结果，没有删掉或改口径。

## 接下来的正确改进路线

1. 以 EmoDynamiX 的 emotion/discourse encoder 为架构基座，在 canonical ESConv train/dev 上从头训练干净 checkpoint；released checkpoint 只作实现验证。
2. 冻结两条训练 arm：原 loss 与 train-only class-balanced/logit-adjusted loss；dev Macro-F1 主选，Accuracy 次选。把本轮 transition/prior 层作为可关闭模块，而不是把 MultiESC 旧代码整包接进来。
3. 在 clean-room、现代 Transformers/PyTorch 实现里移植 MultiESC 的 lookahead/value 思想，保持原始八类，不合并 Information/Suggestion，不删除 Others。
4. 唯一候选和全部字节冻结后，才消费一次 frozen test，与 BlenderBot-small Joint 做同一 2,775-row evaluator 对比。
5. 策略 policy 通过后，再接一个本地、策略条件的 generator；SRA 可提供生成控制思路，但其 LoRA 不能直接当自动策略预测器。

现在不为 MultiESC 或无代码论文租 GPU。租机不能补出缺失权重、预处理或许可。完成现代化 smoke 和 dev 配置冻结后，EmoDynamiX 干净全量重训可先评估 1×A100 40GB 或 1×L40S 48GB；作者 SRA 的 8B 实验则明确使用 1×A100 80GB，属于后续 generator 阶段。

## 为什么还不能替换“静室”的 DeepSeek API

当前改进的是八类策略 policy，不是回复生成器。真正的替换路径必须是：

```text
conversation history
  -> improved ESConv policy + confidence
  -> crisis / boundary / low-confidence safety gate
  -> locally hosted strategy-conditioned generator
  -> output safety checker and escalation route
```

此外，[原始 ESConv 仓库](https://github.com/thu-coai/Emotional-Support-Conversation)明确写明数据和代码只供 academic research，公开数据又带非商用限制；EmoDynamiX checkpoint、MultiESC 和第三方模型的权利链也没有完成商业核准。ESConv Accuracy 更不等于危机处置、临床有效性或生产安全。许可证、授权数据、危机安全集、人工专家复核、隐私和回滚测试全部通过前，只能做 research/shadow-mode，不能直接替换线上 API。

## 可复跑实现

- `scripts/rerank_esconv_emodynamix.py`：train/valid-only prepare 与固定网格选择；没有 test 参数。
- `tests/test_esconv_emodynamix_planner.py`：split 拒绝、target isolation、先验数学、选优规则、CLI 和输出哈希测试。
- `.claude/evals/esconv-code-first-improvement-v1.md`：本轮硬门槛。
- `reports/esconv_emodynamix_author_native_20260820.json`：作者原生 checkpoint 复现。
- `reports/esconv_emodynamix_contamination_audit_20260820.json`：released checkpoint 与 frozen split 的污染证据。
