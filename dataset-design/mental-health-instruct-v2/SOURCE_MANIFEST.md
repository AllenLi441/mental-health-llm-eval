# 训练数据溯源清单（仓库可见副本）

> 本文件是 `tmp/datasets/MANIFEST.md`（gitignored）的元数据副本，只含来源/许可/规模登记，不含任何对话原文。原文与权重不入库。

# 训练数据下载登记（2026-08-15，许可裁决复核版）

全部数据在 `tmp/datasets/`（已 gitignore，不进入版本库）。机器可读版：`MANIFEST.json`。

## M1 许可裁决（2026-08-15 复核，以本地/上游证据为准）

| 数据 | 裁决 | 证据 |
|---|---|---|
| **CPsyCounD** | ✅ M1 核心 | 上游 LICENSE 已核：**CC BY 4.0**（"Attribution 4.0 International"）。HF 元数据 `cc-by-sa-4.0` 作废，以仓库 LICENSE 为准 |
| **CPsDD train** | ✅ M1 核心（≤40%，分层抽样） | license `other`；研究用条件性使用，非研究用途需核上游 |
| **PsyDial-D1** | ✅ M1 核心 | apache-2.0 标注，按实际下载版本逐条复核 |
| **PsyDial-D101** | 🔒 **EVAL ONLY，禁止训练** | 本地核验：101 段专业咨询对话展开为 1,278 个 response-level 评测用例，含 golden 答案 |
| **PsyQA (mirror)** | ⛔ HOLD | 官方 thu-coai/PsyQA 要求签署用户协议并邮件授权；mirror 的 MIT 标注不足以授权训练 |
| **MindChat-R0-FT-Data** | ⛔ HOLD | 本地 README 正文 "private research dataset (non-commercial use); redistribution restricted" 与 metadata `MIT` 冲突 |
| **EmoLLM mirror 整包** | 🟡 RESERVE | 必须拆源→provenance→exact/near dedup 后逐源裁决；`multi_turn_dataset_2` 与 CPsyCounD 重复，禁止整包训练（隐式加权风险） |
| **CPCD conversation** | 🟡 ABLATION ONLY | 不进 M1-core，只作 M1+CPCD 消融臂。本地规模核验：100 profiles / 1,489 sessions / **90,823 utterances** / 11.45M chars（官方口径 100 profiles / ~90,000 dialogue units） |
| **AugESC** | 🟡 ABLATION ONLY | 不作 benchmark；M1 不用 |

## 已下载

| 数据 | 位置 | 大小 | 规模 | 性质 | 许可（裁决后） |
|---|---|---|---:|---|---|---|
| CPsyCounD | `CAS-SIAT-XinHai__CPsyCoun/` | 8 MB | 3,134 对话 | 报告重建（GPT-4） | **CC BY 4.0** |
| CPsDD (processed) | `XuShihao6715__counseling-cpsdd/` | 809 MB | train 54,508 / dev 6,813 / test 6,815 | 专家引导 LLM 合成 | other（研究用条件性） |
| PsyDial-D1 | `qiuhuachuan__PsyDial-D1/` | 32 MB | 2,382 对话 | 心理热线改编（RMRR 重建） | apache-2.0 |
| PsyDial-D101 | `qiuhuachuan__PsyDial-D101/` | 4 MB | **1,278 评测用例（101 对话）** | 专业咨询书改编，评测专用 | EVAL ONLY |
| PsyQA (mirror) | `liuzj288__PsyQA/` | 318 MB | 22,341 QA | 平台问答 | HOLD（需官方协议） |
| MindChat-R0-FT-Data | `dongSHE__MindChat-R0-FT-Data/` | 52 MB | 4,000+500 | 混合 | HOLD（许可冲突） |
| EmoLLM 镜像 | `SmartFlowAI__EmoLLM_mirror/` | 133 MB | 多来源聚合 | 混合 | RESERVE（拆源后裁决） |

## 未能下载（原因 + 状态）

| 数据 | 原因 | 状态 |
|---|---|---|
| SoulChatCorpus | HF 仓库为空壳；ModelScope 链接 404 | 暂缺（广域层已有替代） |
| thu-coai/PsyQA 官方版 | 需签署用户协议 | HOLD |
| smile_chat_55k 官方版 | gated | 已有 EmoLLM 镜像子集（36k 多轮） |
| PsyDTCorpus | hyper.ai 需账号 | 第一轮非必需 |
| Psyche-R1 数据 | 模型 gated | 可选，后续申请 |
| **PsyCrisis-Bench** | 待授权/许可确认（GitHub: mental-health-llm-safety-eval/psycrisis-bench；arXiv 2508.08236） | 安全表候选，待下载冻结 |

## 使用规则

1. M1 = 8,000 条纯中文（CPsyCounD 2,800 + PsyDial-D1 2,000 + CPsDD 3,200），开发集 ~1,000 条同源分组切分。
2. 训练黑名单：CPCD-Bench 159、PsyDial-D101、ESConv test、PsyCrisis、AugESC（M1）、CPCD conversation（M1）。
3. 污染筛查：exact hash + MinHash 近重复 + starter/对话簇排除，跑在切分之前。
4. CPsDD 只用 train.jsonl 且按 group×problem×cause×focus×strategy 分层抽样。
