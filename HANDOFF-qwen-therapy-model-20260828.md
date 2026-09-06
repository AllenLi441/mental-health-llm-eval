# 交接书 v2：Qwen-27B 疗愈模型微调 → 替换静室 DeepSeek API

> **历史快照，已于 2026-09-03 被替代。** 本文保留 2026-08-28 当时的判断与执行记录，
> 其中 provider/pre-test/训练包阻塞、分支与 PR、目录路径、EmoDynamiX 状态，以及把
> ESConv 33.61% 当部署门的说法均已过时。继续执行前请以
> [`HANDOFF-qwen-therapy-model-20260903.md`](HANDOFF-qwen-therapy-model-20260903.md)
> 为唯一当前交接入口；不要从本文恢复任务状态。

写给：执行窗口的 Opus 5（Claude Code）
来自：规划窗口
版本：v2（2026-08-28 全桌面整合。v1 只覆盖 `mental-health-llm-eval` 一个仓库，**漏掉了真正可直接训练的那份数据包**，见 §1.2 与 §7 更正表）
最终目标：把 Qwen 27B 微调成 therapy 模型，替换 https://ai-therapy-room.vercel.app/（静室）后端的 DeepSeek API。

**执行纪律：本机 24GB M4 无风扇。全程遵守 ~/.claude/CLAUDE.md 资源规则——单一重型进程、不本地训练、后台进程记 PID 必杀、不全盘扫描。训练一律在租用的 Linux GPU 上做。**

---

## 0. 决策点（2026-08-28 用户已拍板，不再询问）

**0.1 模型版本：已定 B —— Qwen3.8-27B。**（2026-08-14 发布，Apache 2.0；HF `Qwen/Qwen3.8-27B` + 官方 FP8 版；多模态稠密 27B、262K 上下文、带 `reasoning_effort` 档位。）连带工作全部进本周冲刺（§0bis）：

- **SiliconFlow 尚未上架 3.8-27B**（目录停在 3.6）→ 基线 API 臂改用 OpenRouter（12 家供应商，含 Alibaba Cloud Int. / Cloudflare）或 Cloudflare Workers AI；prereg addendum 里冻结实际 provider + base_url + 价格。
- prereg addendum v2：新 arm `qwen_3_8_27b_<provider>`，**必须冻结 reasoning/thinking 档**——3.6 臂是 `enable_thinking:false`，3.8 对应用最低/关闭的 `reasoning_effort`，写死进协议；prompt 与 proxy judge 完全复用。
- 重跑 pre-test：3.8 臂 CPCD 159 + ESConv fixed-250 + 同一盲评 judge（预计几美元，跑前报价）。
- research-v0 训练包重打：**先核 `requirements.lock.txt` 的 pinned transformers 是否支持 Qwen3.8 架构**——bundle 是 2026-08-04 打的，3.8（新多模态架构）当时还没发布，几乎必然要升版重 pin，bootstrap 的装前装后比对基线也要跟着更新；然后改 config model id → 重打 tar.gz + SHA256SUMS。
- 路线 R 脚本不用改文件：`QWEN_MODEL=Qwen/Qwen3.8-27B` env 覆盖即可。

**0.2 训练路线：维持 P 先 R 后**（见 §1.2），两条不并行。

**0.3 算力：不租云 GPU。** 用户找到一个本地高性能运行库。**到位时间：用户 2026-08-30 告知当天弄不成，要等到下一周才能用**（最早 2026-08-31 那一周，具体哪天未定，以用户确认为准——原文写的「下周到位」已不可再按 8/28 那天去读）。**在机器到位前 = 准备冲刺，不启动任何训练。** 待用户补充确认（见 §0bis 第 7 条）：该库做训练还是只做推理、跑在哪台机器、叫什么名字。已知约束先写在这：若指本机 24GB M4——4-bit 27B **推理**约需 16–19GB 统一内存（Ollama 已有 `qwen3.8:27b-mlx`），可行但要关其他大应用；**本地训练仍被本机资源纪律禁止**；且 80GB bundle 是 CUDA 专用，换非 CUDA 栈必须另写适配层并保留等效的两步 smoke 门。

---

## 0bis. 本周准备冲刺清单（按序执行，全部本机可做）

1. ✅ **2026-08-28 完成** — **Phase 0 基线结算**：`scripts/aggregate_pretest_baseline.py` → `reports/qwen_finetune_pretest_baseline_20260828.{md,json}`；`project-ledger/CURRENT_STATUS.md` 已补到当前并加"三条线"总表；台账 `updates/2026-08-28-qwen-therapy-prep-sprint.md`。**关键结论见 §0ter。**
2. ✅ **2026-08-28 完成** — **3.8 可得性核实落盘**：`reports/qwen38_availability_20260828.md`。存在、Apache-2.0、非 gated、27.78B/55.6GB、main sha `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`、发布 2026-08-14；OpenRouter 11 家供应商 $0.35–0.48 in / $2.55–3.40 out；**SiliconFlow 确认未上架**。
3. ⏸ **阻塞在 provider smoke** — **prereg addendum v2**：价格不构成判据（各家都在 $3 上下），真判据是**供应商是否真正遵守 thinking 关闭**。建议首选 Alibaba Cloud Int.、备选 Cloudflare；**先跑 3–5 条 smoke 确认返回体无思考内容再冻结**。
4. 💰 **待用户确认** — **pre-test 3.8 臂重跑**：按各臂实测 token 算，CPCD 159 + ESConv 250 = 409 条，**含判官合计 $2.47–2.99**（思考开启约 $2.8–3.4）。确认即可执行。
5. ⛔ **被 ERR-20260828-01 阻塞** — **训练包 3.8 适配**：好消息是 **3.8 与 3.6 的 config 在训练脚本断言涉及的每个字段上完全一致**（`qwen3_5` / `qwen3_5_text` / 64 层 / 48 GDN + 16 GA / vocab 248,320），496 这个模块数正是该布局的算术结果 → **架构断言不用改**，只改 `model_id` + `model_revision`。仍须 GPU 实测三项：`language_model_only` 断言、**3.8 新 chat template 下的 completion-only loss mask**、fast-path 绑定。
6. ⛔ **未通过** — **数据终验**：exports 全对（500/50、zh:en=300:200 与 30:20、来源 cpcd/augesc/esconv 分布一致），但**三层完整性校验全部不一致**，见 §0ter 的 ERR-20260828-01。
7. 📋 **已写条件预案**（用户尚未定机器）— `reports/local_compute_and_serving_options_20260828.md` 第一部分：决策树 + 路线 I/C/C'/M 对比 + 机器到位后要问的五个问题。
8. ✅ **2026-08-28 完成** — 同一报告第二部分：本地机器不能背生产流量的硬约束 + 7a/7b/7c 三条端点路线对比（推荐序 7a → 7c → 7b）。

---

## 0ter. 本周冲刺已出的两个硬结论

### A. 基线数字推翻了"微调空间巨大"的原论据

v2 引用的"DeepSeek 16.67% vs Qwen 6.94%"来自 8/08 的 **n=72 pilot**，已被 8/12–8/13 的 fixed-250 结算报告取代：

- **CPCD**：未微调 Qwen3.6-27B(SF) 4.198 vs DeepSeek V4-Pro 4.281，配对差 **−0.083，95% CI [−0.176, +0.010] 跨 0 不显著**，159 对中 106 对完全同分。
- **ESConv fixed-250**：四臂 14.00%–16.40%，两两差异全不显著（原报告 Holm p 全 1）。
- 同协议工程轨基线：Joint **32.22%**、EmoDynamiX **33.61%**。

**修正后的立项理由**：换后端在回复质量上是低风险的（Qwen 零样本已与 DeepSeek 持平，且截断率 0% vs Flash 的 21.15%）；**微调要补的是 API 模型 14–16% 与专用小模型 32–34% 之间的策略能力缺口**，不是补 Qwen 与 DeepSeek 之间的差距。Phase 3 部署门第 2 条（ACC ≥ 33.61%）因此仍然是正确的门槛。

### B. ERR-20260828-01：research-v0 训练包完整性冻结已破（阻塞 ⑤⑥）

1. 外层 `releases/SHA256SUMS` **过期**：记录 `f91385cb…`，实算 `d2f8c2d7…`；压缩包（07:32）比校验和文件（07:29）晚 3 分钟生成，校验和未重算。
2. 压缩包**内容落后于磁盘源且缺 `licenses/` 整个目录**（AugESC / ESConv 均 CC-BY-NC，随包分发许可证是权利边界要求）；压缩包自身 18/18 内部自洽。
3. **磁盘源也不通过自己的 manifest**：8 项 FAILED，清单列了不存在的 `licenses/EmpatheticDialogues-*.txt`，却漏列实际存在的 AugESC / CPCD / ESConv 三份。

磁盘版的改动看起来是有意改进（bootstrap 增加 `sha256sum --check` 强制校验；train_qlora.py 迁到 `AutoModelForMultimodalLM` + 视觉塔冻结 + LoRA 边界断言；eval/save steps 100→20），只是改完没重算 manifest、没重打包。

**未自行修复**——重算冻结数据集的完整性清单属数据治理动作，需 owner 批准。且 3.8 迁移本就要重打包，**两件事应合并成一次受控重打**：以磁盘源为准 → 补齐 licenses 清单 → 改 model_id/revision → 重算 `BUNDLE_CONTENTS.sha256` → 重打 tar.gz → 重算 `SHA256SUMS` → 更新 `UPLOAD_AND_TRAIN.md`。

---

## 1. 已有资产总盘点（散在 6 个位置，勿重复造）

### 1.1 评测与协议 —— `~/Desktop/mental-health-llm-eval/`

| 资产 | 路径 | 状态 |
|---|---|---|
| 预注册：DeepSeek V4-Pro vs Qwen3.6-27B(SiliconFlow)，CPCD+ESConv | `open_response_eval/preregistration_siliconflow_qwen_cpcd_v1.json` | FROZEN 2026-08-13 |
| Judge 偏差备案：协议写 GPT-5.2，实跑是 deepseek-v4-flash 盲评代理 | `..._flash_addendum.json` | FROZEN；**post-test 必须复用同一 proxy judge、temp=0、非思考、同 blind_id 方案** |
| 冻结提示词（SHA256 已入协议） | `open_response_eval/prompts/` | 勿改 |
| pre-test 各臂生成 | `outputs/provider_sf_qwen_cpcd_20260813.jsonl`、`provider_v2_deepseek_cpcd_20260808.jsonl`、`provider_v2_flash_cpcd_20260809.jsonl`、`provider_v2_flash_esconv_250_20260809.jsonl`、`paired226_qwen_esconv_20260808.jsonl` 等 12 份 | 已存在 |
| 盲评 judge 记录 | `outputs/cpcd_proxy_flash_judgements_20260809.jsonl` | 已存在 |
| ESConv 工程基线（frozen 2,775 test） | Joint(BlenderBot-small) **32.22%**；EmoDynamiX 作者原生 **33.61%** | `reports/esconv_code_first_model_audit_20260820.md` |
| 零样本 pilot（说明为何必须微调） | DeepSeek V4-Pro 16.67% / Qwen3.6-27B **6.94%**（72 配对，八类） | `reports/pilot_esconv_deepseek_qwen_20260808.json` |
| 19 任务 benchmark suite v1 + DeepSeek 历史基线 | `benchmark-freezes/` | FROZEN，`paper_control_ready=false`，勿跨协议平均 |
| 项目台账（每次更新必须同步） | `project-ledger/CURRENT_STATUS.md` + `updates/` | 最后一条 2026-08-01，**已落后于代码，需要补** |

⚠️ **仓库有在飞任务，别踩**：分支 `codex/cross-benchmark-accuracy-family-v1`，Draft PR #5。最近提交（8/21–8/24）是 **EmoDynamiX 干净重训**：特征已生成并验证（`tmp/emodynamix-clean-features/canonical-train-dev-v1`，干净切分 8,433/2,985），dev pilot 预注册已冻结（ce + class_balanced 两臂，8/23）并于 8/24 改为 CPU device，8/28 00:59 跑过一次 `smoke_cpu_memcheck`。这条线属于论文轨，与本交接书的疗愈模型是**并行的两件事**——不要为了训练 Qwen 去改这些文件或回滚这些提交。

### 1.2 训练资产 —— 两套，必须分清

#### 路线 P：静室双语 QLoRA 包（`~/Desktop/静室/datasets/mental-health-instruct-research-v0/`）

**这是 v1 交接书完全漏掉的东西，也是最接近"疗愈模型"的那份数据。**

- 状态：`RESEARCH_V0_READY_FOR_CONTROLLED_TRAINING_WITH_MANDATORY_SMOKE_STATIC_ONLY`（2026-08-04 冻结，**从未训练过**——`server_training_bundle/runs/` 不存在）
- 规模：train 500（中 300 / 英 200）、development 50（中 30 / 英 20），严格 60:40
- 来源：CPCD 中文 300/30；AugESC 英文 199/19；ESConv 英文 1/1。EmpatheticDialogues、MentalChat16K、ProsocialDialog 等**均已审查并拒绝**，原因在 `manifests/source_registry.json`
- 数据合同：`prompt`(system+历史，必须以 user 结束) / `completion`(仅最后一条 assistant) / `gold_status=NOT_GOLD` / `risk_level=R0_STATIC_SCREEN_ONLY`。**训练必须开 completion-only loss，对历史 assistant token 做 loss mask**
- 已过门禁：来源 revision+SHA-256 固定、profile 不跨 split、全局精确重复为零、中英近重复硬门、静态 PII/危机/医疗/诊断/建议/承诺/依赖/角色翻转筛查、字节级可重现
- 训练包：`server_training_bundle/`（`bootstrap.sh` / `run_smoke_80gb.sh` / `run_train_80gb.sh` / `train_qlora.py` / `requirements.lock.txt` / `configs/{smoke,recommended}-80gb.json` + 48GB 实验档 / `BUNDLE_CONTENTS.sha256`）
- 上传物**只有两个文件**：`releases/mental-health-instruct-research-v0-qwen3.8-27b-20260828.tar.gz`（0.28 MiB）+ `releases/SHA256SUMS`。**归档 sha 只从 `releases/SHA256SUMS` 读，本文件不抄**（手抄即 ERR-20260828-01 同类）
- 机型硬要求（见 `UPLOAD_AND_TRAIN.md`）：**1×A100/H100 80GB**、Linux x86_64、Python 3.11、torch `2.7.1+cu128`、torchvision `0.22.1+cu128`、Triton 3.3.1、CXX11 ABI enabled。磁盘**按容量买，不按门要的空闲买**：缓存卷与训练包异卷 **≥120 GiB**、同卷 **≥220 GiB**（门每次 preflight 重量一遍空闲，而快照下完后常驻 51.77 GiB；只按 160 配盘会在 smoke 起手被拦，ERR-20260830-02）。48GB 是需显式确认的实验路径，24GB 明确不适用
- 强制流程：`sha256sum -c` → `bootstrap.sh`（装前装后比对 Python/Torch/CUDA/Triton/ABI，有变化即停）→ **`fetch_weights.py --dry-run` 再 `fetch_weights.py`（取 51.77 GiB 快照，不可跳过）** → `run_smoke_80gb.sh`（两步 smoke，跑最长两条，NaN/Inf 即失败，产 SHA-256 收据）→ `run_train_80gb.sh`（无有效 smoke 收据拒绝启动）
- 取权重那一步没有门守着：`bootstrap.sh` 与两个 runner 都不下载，`train_qlora.py` 的 `from_pretrained()` 不带 `local_files_only`，所以**跳过它不报错**，只把下载挪进 smoke，等 GPU 计费后才暴露网络/磁盘/gated 问题
- 原始快照在隔壁 `mental-health-training-sources-research-v0/`（raw/licenses/audits/reviewed_not_used），**不上传、不进 Git**

#### 路线 R：eval 仓库 ESConv LoRA（`~/Desktop/mental-health-llm-eval/`）

| 资产 | 路径 | 说明 |
|---|---|---|
| SFT 数据（已构建 8/12） | `artifacts/esconv/data/esconv_{strategy,generation}_{train,dev}.jsonl` + `modern_sft_manifest.json` + `dataset.sha256` + `test_item_dialog_ids.jsonl` | 用前按 sha256 校验；测试集只落 ID 不落文本，防泄漏 |
| 构建脚本 | `scripts/prepare_esconv_modern_sft.py` | 内置 EXPECTED_SHA256 |
| LoRA 脚本（**ms-swift**） | `scripts/run_qwen36_esconv_lora.sh` | 硬性只跑 Linux x86-64 CUDA（自检拒绝 mac、拒绝无 nvidia-smi、拒绝无 swift）。strategy: 3ep/5e-5/len1024；generation: 2ep/2e-5/len2048；`ESCONV_SMOKE=1`；默认 `CUDA_VISIBLE_DEVICES=0,1` 两卡 |

#### 不可用于训练（红线）

- `~/Desktop/静室/datasets/mental-health-instruction-dataset-v2/`：产品级真人金标线。目标 6,000 train + 600 dev + 200 Human Gold Test + 400 Safety Eval，**当前真人批准 0 行、`training_allowed=false`**，卡在心理专业负责人签核 + 伦理/IRB 判断 + 标注员知情同意。已生成的只是 25 条 A/B 校准包（`export_to_training=false`）和 AnnoMI 600 条空白撰写包。**Qwen 可以生成 `ai_draft`，但绝不能给自己造金标——不要用 AI 填答案绕过这道门。**唯一允许的服务器动作是 `mhi-v2-smoke-only.tar.gz` 的丢弃式 GPU smoke（不含训练数据）。
- `mental-health-llm-eval/dataset-design/mental-health-instruct-v{1,2}/`：同一条线在 eval 仓库里的公开 schema 副本，0 行。
- `mental-health-training-sources-v1/quarantine_raw/`：MentalChat16K 合成 9,774 条，**危机回复安全抽查失败已隔离，禁止作正例**。

### 1.3 文献与复现 —— `~/Desktop/Esconv 改进论文和代码/`（1.1G）

7 篇论文各带 `论文/`(PDF) + `代码/`(GitHub clone)，README 有完整可运行性审计。核心原则写在里面：**论文报的分不算数，有代码的自己跑一遍**。

- 已本机复现：`01_ESConv原论文_ACL2021` Joint **32.22%**；`06_EmoDynamiX_NAACL2025` **33.61%**（当前改进基座）
- 待复跑优先级：`05_CauESC`（第 1，协议干净规模小）→ `03_TransESC`（第 2，自报 34.71%，需先核切分）
- 不进榜：`02_MultiESC`（42.01% 实为约 7 类）、`04_PAL`（用 PESConv 增强）、`09_ESConv-SRA`（15 类另一套输出空间，留作未来回复生成器供体）
- `98_代码不可运行/`：DecoupledESC（无训练代码）、AFlow（IndentationError，65.10% 是 MCTS 合成协议）
- 配套文献榜：`~/Desktop/数据集准确率对比_修正版_20260821.docx` —— 含关键更正：Papers with Code 上**没有** ESConv 统一排行榜；DPPLM 58.03% / Causal-ESC 53.53% / SAGE 46.80% / CADSS 46.26% 等高分论文**全部无可用官方实现**，不可复跑

### 1.4 安全引擎 —— `~/Desktop/OH-WorkSpace/mental-health-ai-safety-engine/`

独立 TS/npm 包（有 LICENSE / SECURITY.md / CONTRIBUTING.md / .github / vitest）。`evals/` 里有 `safety-cases-10000.json` + `safety-eval-10000-{summary,findings}` + `SAFETY_EVAL_PROTOCOL.md` + 公开挑战集。**这是 Phase 3 安全门和 Phase 5 上线守卫的现成工具，别另造。**

### 1.5 静室生产接入面 —— `~/Desktop/静室/app/`

- 模型走 OpenAI 兼容 env：`DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` / `DEEPSEEK_API_KEY`
- **注意**：chat 的 deep/fast 档位在 `src/lib/model-options.ts` 里按会话节奏硬选 `v4-pro`/`v4-flash`，不是只读 env——换后端必须改这里的映射
- SiliconFlow 账户已在生产使用（Kimi 判官 + Qwen3-Embedding-0.6B + Reranker），网关兼容性已验证
- 回归网：`jingshi-eval/`（CPsyCounE 45 例双判官 replay，`node eval.mjs N`）
- **安全红线（不可协商）**：危机词表绝不自动改；`X-Crisis-Triggered` / `X-Crisis-Mode` header 契约不变；危机快照测试禁 `-u`；上线 = push main（Vercel 自动部署）+ bump `APP_VERSION` 并报版本号；交付物中不得出现临床评审者姓名

### 1.6 其他相关材料

- `~/Desktop/心理学相关调查问卷/`：GAD-7、DASS-21、WHO-5 原始量表 PDF
- `~/Desktop/AugESC_Data_Audit.xlsx.inspect.ndjson`：AugESC train 审计（65,077 对话 / 1,738,446 utterance / 完全重复 0，但 **99.998% 落在重复起始场景组 → 必须按场景组切分，不能随机切**）。research-v0 已遵守此约束
- `~/Desktop/项目成果汇总_2026-06-05/`：6 月的历史交付（论文统一指标 PDF、静室产品提升计划、各项安全审计报告），仅作参考

---

## 2. 执行计划

### Phase 0 — 基线结算（本机，零 API 成本，先做）

把 8/08–8/13 已生成的 CPCD/ESConv 各臂输出 + proxy judge 记录聚合成 **pre-finetune 基线报告**：每臂 judged 分数、配对差、invalid 率。

- 产出：`reports/qwen_finetune_pretest_baseline_20260828.md` + 同名 json
- 验证：数字能从 `outputs/*.jsonl` 逐条重算对上；报告注明"proxy judge 内部分数，不得当论文榜"
- 顺手把 `project-ledger/CURRENT_STATUS.md`（停在 8/01）补到当前状态

### Phase 1 — 训练数据定稿（本机）

**路线 P**：`sha256sum -c releases/SHA256SUMS`；核对 `manifests/export_manifest.json` 与 `split_manifest.json`；确认 500/50 行数与 60:40 语言比。构建器可字节级重现，如需重建跑 `scripts/build_research_v0.py`。

**路线 R**：校验 `artifacts/esconv/data/dataset.sha256`；确认 train∩test 对话 ID = ∅。

（v1 交接书写"中文能力靠 base 模型自带"——**作废**，路线 P 本身就有 330 条中文 CPCD。）

### Phase 2 — 训练（**方案变更：不租云 GPU，等用户的本地算力方案；该机器 8/30 起推迟，最早 2026-08-31 那一周可用**；以下租用流程保留作后备）

**路线 P（推荐先跑）**：1×A100/H100 80GB，按 §1.2 的 pinned 环境，磁盘按容量配（异卷 ≥120 GiB / 同卷 ≥220 GiB）。只上传 tar.gz + SHA256SUMS → `sha256sum -c` → `bootstrap.sh` → **`fetch_weights.py --dry-run` 再 `fetch_weights.py`（51.77 GiB，不可跳过）** → `run_smoke_80gb.sh`（两步强制 smoke）→ `run_train_80gb.sh` → 下载 `runs/research-80gb-<UTC>/` 的 adapter。500 条 QLoRA 很快，按小时租即可。

**路线 R**：2×A100 80G（脚本默认两卡；单卡需 `NPROC_PER_NODE=1`）+ pinned ms-swift → 上传 `scripts/` + `artifacts/esconv/data/` → `ESCONV_SMOKE=1` 冒烟 → 全量 strategy → 全量 generation。

- 验证：loss 曲线存档；smoke 收据 SHA-256 归档；路线 R 另需 dev 集八类 ACC ≥ 40%（低于 33.61% 即无意义，调参重跑一次仍不过就停下汇报）
- **两条路线不并行**；预算预计 <$30，先报数字

### Phase 3 — Post-test（严格按冻结协议）

微调后模型用 vLLM 临时起 OpenAI 兼容端口，重跑 CPCD 159 条 + ESConv fixed-250，**同一冻结 prompt、同一 proxy judge（deepseek-v4-flash、temp=0、非思考、同 blind_id sha256 方案）**。路线 R 另跑 frozen-2,775 八类 ACC。

部署门（全满足才进 Phase 4）：

1. CPCD judged 分 ≥ DeepSeek V4-Pro 臂（配对比较）
2. ESConv 八类 ACC ≥ 33.61%（仅路线 R）
3. 静室 `jingshi-eval` 45 例 replay 无安全回退
4. **`mental-health-ai-safety-engine` 10,000 案例安全评测不低于 DeepSeek 现状**（§1.4）
5. invalid 输出率 ≤ 基线

产出：`reports/qwen_finetune_posttest_20260XXX.md`（pre/post 配对表）

### Phase 4 — 上线服务（二选一，先问用户）

- **4a 托管**：SiliconFlow 若支持该 base 的 LoRA 托管/自部署实例 → 与现有账户同网关，静室零新依赖（**先查这个**）
- **4b 自托管**：merge LoRA → 量化(AWQ/FP8) → serverless GPU 挂 vLLM OpenAI 兼容端点。有常驻成本，报价后再选

### Phase 5 — 静室切换（渐进，可回滚）

1. 新增 env：`THERAPY_BASE_URL` / `THERAPY_MODEL` / `THERAPY_API_KEY`，未设置时行为完全不变
2. 改 `src/lib/model-options.ts`：deep/fast 档位映射到新模型（fast 档留 v4-flash 还是同模型低 max_tokens——问用户）
3. DeepSeek 保留为超时/报错 fallback
4. 危机管线一行不动；全量测试 + 危机快照（禁 `-u`）绿 → 本地 `npm run build` 过 → push main → Vercel → 线上冒烟（中英双语 + 危机 header 验证）→ bump `APP_VERSION` 并报版本号

---

## 3. 汇报约定

- 每完成一个 Phase：`project-ledger/updates/` 追加记录（照现有格式）+ 同步 `CURRENT_STATUS.md`
- 花钱的步骤（GPU、租用、API 批量）先报数字等确认
- 卡住 >30 分钟：写明已试方案 + 卡点回报，不要空转

---

## 4. 磁盘警告（512GB SSD，先告知用户再动手）

`~/Desktop/mental-health-llm-eval/` 已占 **100G**：

| 目录 | 大小 | 说明 |
|---|---:|---|
| `tmp/weights` | 52G | HF 模型权重缓存 |
| `results/psysuicide-roberta-v2-accuracy-first` | 33G | **v2 campaign 已 PAUSED**（A 完成、B 停在 3,560/4,680、C/D 未开始）→ 最大的一块可回收空间 |
| `tmp/official_benchmarks` | 5.5G | 官方基准数据 |
| `results/psysuicide-roberta` | 4.9G | v1 checkpoint（已出成绩，可能仍需保留） |
| `tmp/datasets` | 1.3G | |
| `results/model-cache` | 1.2G | |

租 GPU 前如果本机空间紧张，**先问用户**再删；这些是实验凭据，不是缓存。

---

## 5. 会话遗留物（已处理）

`mental-health-llm-eval/.claude-bg.pids` 记录的 29606 / 29610 均已不存在（上一会话未回收）。本次已删除该文件。

---

## 6. 术语澄清

- **"oxalpha"** = opencode 之前用过的一个模型名，不是本机的工具或目录，无需寻找。
- 本项目实际有**三条并行线**，别混：① 疗愈模型微调（本交接书）② ESConv 八类策略论文轨（EmoDynamiX 干净重训，在飞）③ 19 任务 benchmark 家族（8/01 起多数 PAUSED）。

---

## 7. v1 → v2 更正表

| v1 的说法 | v2 更正 |
|---|---|
| 训练资产只有 eval 仓库的 ms-swift ESConv LoRA | **漏了静室 `mental-health-instruct-research-v0`**——双语 500/50 QLoRA 包，含完整 server bundle 与 release 压缩包，更贴近产品目标 |
| Phase 2 默认 2×A100 | 路线 P 是 **1×80GB**，且环境版本被 `bootstrap.sh` 硬校验 |
| "中文能力靠 base 模型自带" | 作废，research-v0 有 330 条中文 CPCD |
| "MentalHealth-Instruct v1 …本次不用" | 真正的产品金标线是静室 **v2**（0 行、`training_allowed=false`、卡真人签核），红线更明确 |
| 未提及 | eval 仓库有**在飞**的 EmoDynamiX 干净重训（8/21–8/24 提交，PR #5），勿踩 |
| 未提及 | 现成安全引擎（10,000 案例）应作为 Phase 3 部署门第 4 条 |
| 未提及 | 文献/复现库 `Esconv 改进论文和代码/`（7 篇论文+代码+可运行性审计） |
| 未提及 | 仓库占 100G，33G 属已暂停 campaign |
