# Qwen3.8-27B 训练量与静室核心模型判定（2026-09-06）

## 判定

当前数据**不能作为 Qwen3.8-27B 的完整训练量直接开训**。A3/B1 本地候选重建是 82,401 条，状态仍为 `CANDIDATE_NOT_FROZEN`；它不是批准训练集、不是可信切分，也没有完成许可、人工安全、近重复、跨来源派生和 token 级 loss-mask 验证。

| 路线 | 当前可计数内容 | 能做什么 | 不能做什么 |
|---|---:|---|---|
| `research-v0` | 500 train / 50 development | 已固定的研究 pipeline smoke | 不能代表正式效果或产品数据 |
| `a3-b1-candidates` | 82,567 raw → 82,401 candidate | 继续本地规范化、审计和迁移准备 | 不能进入训练目录或生成训练 release |
| `m1-v1` | 8,000 / 500 / 500 | 历史研究候选、另立消融 | 不能改称最终测试集；PII、目标截断和权利仍有问题 |
| `formal-v2` | 0 approved / 6,600 planned | 继续真人写作和审核 | 不能正式训练；0/6,600 只描述这条历史路线 |

A3/B1 候选构成为：PsyDT 4,758 中文（125 行复核标记）、MeChat_smile 66,031 中文（66,031 行缺可靠来源组，2,180 行复核标记）、CounselChat 2,612 英文（163 行被格式门拒绝，211 行复核标记）、EN CBT 9,000 英文（合成、3,000 profile×3、六主题窄覆盖）。另有 53 个重复 target hash、77 个重复余量。候选语言约为 70,789 中文 / 11,612 英文，不能直接当作计划中的 70:30。

`39,250`（27,475 中文、11,775 英文）仅是原始 A3/B1 报告中的容量算术，当前 manifest 明确标为 `ARITHMETIC_ONLY_NOT_MATERIALIZED`；它不是训练行数、批准量或冻结规模。

按当前候选文本字符做的容量估计约为 5988 万“粗略 token”（上下文总量），其中监督目标约 612 万“粗略 token”；中文按 1 字符、英文按 4 字符换算，仅用于估算磁盘/吞吐，不能代替 Qwen tokenizer、chat template 和截断后的真实统计。

## 可以如何用于周一上机

先把候选目录复制到独立的 `candidate-not-frozen` 目录，复制前后核对 manifest 的文件列表、大小和 SHA-256，并绑定代码提交。不要把它接入默认训练目录。之后按来源分别补齐：用途许可和 revision、来源组、近重复/benchmark 污染、PII 与专业安全复核、正式 split、实际 tokenizer/chat template/collator 的 token 级 assistant-only mask。只有这些证据对应的导出版本才可以另立 Qwen SFT run。

本轮已经生成完整的本地候选训练包：`artifacts/phase1-a3-provisional-training-20260906-final-v2/`，含 `train.jsonl` 13,702 条、`development.jsonl` 2,332 条，以及带原文的 `quarantine_records.jsonl` 66,367 条和索引。后者包括全部被标记、缺来源组或未通过结构门的记录，便于后续逐条处理；包的 `export_manifest.json` 仍明确 `training_allowed=false`。这是一份**结构完整的待审训练包**，不是已经获得训练授权的成品。

## 静室能否使用 Qwen 作为核心模型

网站可以继续作为 Qwen 的前端、安全壳和 RAG 层，但当前不能把 Qwen 直接换成线上核心：

* `app/src/app/api/chat/route.ts` 的正常请求固定调用 DeepSeek；深度模式为 `deepseek-v4-pro`，快速模式为 `deepseek-v4-flash`。
* Kimi-K2.5（SiliconFlow）是独立的风险判官，确定性危机路由、危机 UI、RAG embedding/rerank 不属于底座模型权重。
* `/api/health` 目前只能报告配置状态，不能证明实际响应的模型身份。
* Qwen 替换需要自托管或 GPU endpoint、OpenAI-compatible provider、stream/timeout/thinking 兼容、tokenizer/chat template 固定、runtime model fingerprint 和延迟/错误率记录；安全判官和确定性安全层应继续保留。

因此，Qwen SFT 只有在“Qwen base vs Qwen SFT vs 当前 DeepSeek 产品基线”的同一冻结场景比较通过后，才有资格讨论替换。当前网站线上页面本轮未能独立读取，结论依据本地 app 代码和 README；没有把线上环境变量推断成事实。

## 对比表是否已经准备好

**表格骨架和部分基线已经有，完整验收表还没有。**已存在的可复用表包括：

* ESConv fixed-250：DeepSeek V4-Pro、V4-Flash、Qwen3.6-27B 和 Qwen3.8-27B 未微调基线的策略 Accuracy/Macro-F1/Weighted-F1/invalid，以及回复 BLEU/ROUGE/Distinct。
* CPCD：SR/MR/TCR 的 proxy judge 分数和 coverage；它们是内部代理分数，不能与论文的 GPT-5.2 分数混排。
* AugESC：ESConv 固定测试上的回复指标；AugESC 没有独立 Accuracy。
* 评测草案：五维 relevance、empathy、safety、boundary、naturalness，等权 aggregate，已有六项 Holm 家族定义。

尚缺：Qwen3.8 SFT 的真实输出和 token-mask 证据、当前 DeepSeek 运行 model ID/fingerprint、独立 Human Gold/Safety 冻结证据、pilot 方差和评分噪声、`delta_safety`/`tau_critical` 阈值、入口负向测试，以及完整来源组污染报告。因此不能说“对比表和验收标准都已完成”，只能说研究基线表和评测协议草案已整理。

## 当前可发布状态

本报告与脚本、manifest、合同一起进入 GitHub；原始候选文本保留在本机，不作为公开数据发布。最终状态是：候选重建完成，训练尚未放行，网站核心模型尚未替换，Qwen SFT 结果尚不存在。
