# MentalHealth-Instruct v2 — 静室 Qwen3.6/3.8-27B 微调方案（冻结稿 v2）

Status: `DESIGN_FROZEN_2026-08-15 / TRAINING_NOT_STARTED / LICENSE_GATES_PARTIAL`

## 0. v1 → v2 变更摘要

1. **M1 配方锁定为 8,000 条纯中文**（CPsyCounD 2,800 + PsyDial-D1 2,000 + CPsDD 3,200），不做 70/30 双语混合——保因果归因干净，且避免 AugESC→ESConv 的 in-domain 污染。
2. **第三 benchmark 由 AugESC 换成 PsyDial-D101**（已下载、评测专用、中文真实风格）；AugESC 降为训练消融表。
3. **新增安全表候选 PsyCrisis-Bench**（待授权下载）。
4. **CPCD conversation 从核心降为消融臂**（M1+CPCD），不进 M1-core。
5. **许可裁决落地**：CPsyCounD → CC BY 4.0（上游 LICENSE 已核）；PsyQA → HOLD；MindChat → HOLD；EmoLLM → RESERVE（拆源后逐源裁决）；PsyDial-D101 → 训练黑名单。
6. **基座选型更新**：Qwen3.8-27B 官方权重已发布（Apache 2.0，2026-08-14，HF `Qwen/Qwen3.8-27B`），本地 3.6/3.8 A/B 立即可以做；Qwen Cloud 托管 API 仍 coming soon。
7. **非思考协议补全**：`enable_thinking=False` 且 `preserve_thinking=False`（3.8 两者默认均开启）。

## 1. 评测与训练解耦（三张 benchmark 表 + 一张消融 + 一张安全）

| 表 | 内容 | 地位 |
|---|---|---|
| **CPCD-Bench**（CPCD_Benchmark.xlsx） | 159 任务：SRG 99 / MR 40 / TCR 20；代理裁判内部基线 | 主 Benchmark（中文长程） |
| **PsyDial-D101**（PsyDial_D101_Benchmark.xlsx） | 101 段专业咨询对话 → 1,278 response-level 评测用例 | 第二 Benchmark（中文真实风格咨询质量） |
| **ESConv**（ESConv_Benchmark.xlsx） | fixed250 开发回归 / full 2,775 最终冻结 | 辅助结构诊断（英文策略） |
| **PsyCrisis-Bench**（待授权） | 中文高风险心理危机场景安全评测（LLM-as-Judge） | 安全门 |
| **AugESC**（AugESC_Benchmark.xlsx，已改标题） | M0/M1/M2/M3 训练消融 | 消融表，不是 benchmark |

训练数据与评测解耦：三张 benchmark 表口径不变；训练语料不限这三个来源，但受黑名单约束。

## 2. M1 训练配方（8,000 条中文，先小后大）

| 来源 | 条数 | 占比 | 许可状态 | 作用 |
|---|---:|---:|---|---|
| CPsyCounD | 2,800 | 35% | CC BY 4.0 ✓ | 专业咨询结构/流派/完整多轮 |
| PsyDial-D1 | 2,000 | 25% | apache-2.0（按版本复核） | 中文长程真实风格 |
| CPsDD train | 3,200 | 40% | conditional（研究用；分层抽样） | 人群/问题/策略覆盖 |

原则：真实/重建高质量 ≥60%，合成补覆盖 ≤40%，来源不明 = 0，评测数据 = 0。

- **开发集**：额外 ~1,000 条（CPsyCounD ~300 / PsyDial-D1 ~300 / CPsDD ~400），按 dialogue/group 分组切分，同源对话不得跨 train/val/test。
- **训练黑名单**：CPCD-Bench 159 任务、PsyDial-D101、ESConv test、PsyCrisis、AugESC（M1 不用）、CPCD conversation（M1 不用，仅消融臂）。
- **训练参数**：QLoRA 4-bit NF4 / BF16；r=32 α=64 dropout=0.05；只打语言模型 Linear 层（冻结视觉编码器）；1 epoch；max_seq 4096；global batch 16；LR 1e-4 cosine；warmup 3%；assistant-only loss；seed 42（复现 seed 43/44）。无 RL/DPO，无 thinking。

## 3. 预处理流水线（审计顺序固定）

license gate → 规范化 schema（user/assistant，保留原始角色于 metadata）→ PII 两段式（regex + NER，对话内稳定化名，高置信未脱敏即 DROP）→ exact hash → MinHash 近重复（Jaccard≥0.90 自动去重）→ benchmark 污染扫描（CPCD 159 / ESConv fixed250+full / PsyDial-D101 / PsyCrisis；AugESC 另做 starter-group 排除）→ group 级切分 → 分层抽样 8k → chat 模板序列化 → tokenizer 长度审计（目标回复不静默截断）→ SHA256 冻结。

## 4. 严格前后对照（不变）

训练前本地推理环境（固定引擎/量化/chat 模板）跑 CPCD 159 + ESConv fixed250 + PsyDial-D101 → 冻结本地基座；训练后同环境重测；差值 = 训练增益。SiliconFlow/DeepSeek API 基线仅作审计参照。

## 5. 基座选型（2026-08-15 更新）

- Qwen3.8-27B：Apache 2.0；27B；原生 262,144 → YaRN 1M；基于 Qwen3.5 架构；thinking **默认开**（`enable_thinking` 与 `preserve_thinking` 默认 on）；官方支持 Transformers/vLLM/SGLang/TokenSpeed；**Qwen Cloud hosted 版 coming soon**。
- Qwen3.6-27B：Apache 2.0，27B，非思考模式明确支持。
- **本地权重 A/B 现在即可**：同一 item ID / 系统提示 / 语言 / 输出预算 / 解析器 / 裁判 / thinking=off / 解码协议（硬件尽量一致）。
- **决策门（全过才切 3.8）**：
  - Gate A：CPCD 汇总 3.8 ≥ 3.6；
  - Gate B：PsyDial-D101 无明显咨询质量退化；
  - Gate C：ESConv 无明显策略能力退化；
  - Gate D：安全门（PsyCrisis 或等价安全套件）不退化；
  - Gate E：训练栈 smoke 全链路通过（load tokenizer → chat template → `enable_thinking=False` 前向 → 4bit 加载 → LoRA 注入 → 一步优化 → save/reload → 合并副本 → 数值容差一致）。
  - 平局偏向 3.6（避免为无验证收益引入新架构/序列化栈）。
- 3.6 的 tokenized 缓存不得迁移到 3.8；从 canonical `messages` 重新序列化。
- 只用官方权重 `Qwen/Qwen3.8-27B`；不用第三方魔改版。

## 6. 评估统计口径

- ESConv：Acc / Macro-F1 / Weighted-F1 / 逐类 F1 / 混淆矩阵 / 无效率；**对话簇 bootstrap** 95% CI；M0 vs M1 用配对 **McNemar**（b/c 与 CI 同报）。
- CPCD / PsyDial-D101：连续裁判分按 case 聚类 bootstrap，不用 McNemar。
- fixed250 只用于训练后快速回归；**禁止用任何 benchmark 分数调超参**，只用 internal validation。

## 7. 开放项（开训前必须闭环）

- [ ] PsyCrisis-Bench 许可确认 + 下载冻结（[GitHub](https://github.com/mental-health-llm-safety-eval/psycrisis-bench) / arXiv 2508.08236）
- [ ] PsyDial-D1 实际版本许可复核；CPsDD 上游条款存档
- [ ] 训练环境选型（QLoRA 80GB，M1 预算 ~8–24 GPU-h）+ 依赖版本冻结
- [x] 数据清洗脚本（normalize / pii / dedup / overlap / group-split / sample-8k / tokenize-audit）——`scripts/data/`，已在真实数据跑通
- [x] M1 语料冻结：`artifacts/m1-v1/data/`（train 8,000 / val 500 / test 500，全中文，组间零交叉，benchmark 污染扫描 0 命中）
- [ ] 32 条过拟合冒烟 + LoRA save/reload 冒烟（需 GPU 训练环境）
- [ ] 本地基座冻结（选中基座后：CPCD + fixed250 + PsyDial-D101）

## 8. 数据流水线执行记录（2026-08-15/16）

| 步骤 | 结果 |
|---|---|
| normalize | 60,024 条 canonical（CPsyCounD 3,134 / PsyDial-D1 2,382 / CPsDD 54,508） |
| PII | 733 条脱敏（URL 992 / EMAIL 8 / PHONE 23 / ADDRESS 7 / IDCARD 1）；8,059 条仅标记复核（地址类模糊表述，不破坏文本） |
| dedup | exact 10 去重 + near 27 自动去重 + 30 复核标记；保留 59,987 |
| benchmark 重叠 | 4,133 条评测文本（CPCD 159 文件 + ESConv 2,775 + PsyDial-D101 1,278），**0 命中** |
| 采样 | train 8,000（2,800/2,000/3,200）/ val 500 / test 500；全 zh；组级不相交 ✓ |
| manifest 审计 | PASS |
| Qwen3.8-27B 权重 | ✅ 已下载（ModelScope 镜像，52GB，18 分片 + tokenizer 全齐；HF 直连 TLS 不稳） |
| tokenizer 审计 | ✅ train 8,000：token 均值 1,062.6 / p95 2,785 / 超 4,096 仅 21 条(0.26%) / 目标截断保护 31 条 / EOS 全部正确（`<|im_end|>`=248046）；val/test 同样健康 |

注意事项（已修复的坑，后续复现时勿踩）：
1. PII 地址正则必须锚定数字/单元后缀，单字"道/路/街"会造成大规模误脱敏。
2. MinHash 的 seed 混合同需用通用哈希 h1+i·h2；XOR 小 seed 会使签名退化（est=1.0 但真实 Jaccard 0.05）。现用 bottom-K 草图（每 shingle 单次 blake2b）。
3. LSH 桶需设上限（>1000 跳过），否则 CPsDD 模板开场白导致候选对爆炸。
