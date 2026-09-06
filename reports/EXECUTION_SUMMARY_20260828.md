# 文献调研与基线实测执行总结（2026-08-28）

> **历史执行快照（2026-09-03 对齐注记）：** 本文记录 8/28 当日完成的调研与基线，
> 不再是当前待办清单。之后已完成 B1 A3/70:30 收口、T0 参数化、T1.5 工具/40 条 probe
> 冻结、research-v0 第六次受控重打，以及 Qwen3.8 DashScope fixed-250 实际结果。
> 其中 ESConv 32–34% 监督分类数字与生成式 fixed-250 跨协议，现为 diagnostic-only，
> 不作 QLoRA 或部署门；当前执行口径见根目录 `HANDOFF-qwen-therapy-model-20260903.md`。

执行时间：2026-08-28 03:30 – 05:00（约 1.5 小时，4 轮 loop + 补充核实）

执行方式：loop 自动节奏 + 人工中断后补充核查

---

## 一、完成的工作

### 1.1 ESConv 文献重查（Papers with Code API）

- ✅ 确认 paperswithcode.co 无 ESConv 排行榜（5 个 slug 全 404）
- ✅ 系统普查 179 篇去重、53 篇相关、26 篇有官方实现
- ✅ 识别出两个时代：第一代（小模型+策略分类，32–34% ACC）vs 第二代（LLM+RL，rubric 分）
- ✅ 定位 Kardia-R1（★60）、TEA-Bench、ESC-Eval（★27）、MISC（★39）、KEMI（★14）的 GitHub 仓库
- ❌ GLHG 未找到公开代码（多种搜索词均 404）

### 1.2 Qwen3.8-27B pre-test 基线实测

- ✅ 冻结 DashScope Qwen3.8 ESConv-250 addendum 协议
- ✅ 跑完 250 条 ESConv（135 秒，免费额度内）
- ✅ 结算：**Qwen3.8-27B 零样本 = 13.60%**（0 missing / 0 invalid，格式合规率 100%）
- ✅ 四臂配对：与 Qwen3.6 / DeepSeek V4-Pro / V4-Flash **全部不显著**（Holm p 全为 1）
- ✅ 坐实立项判断：API 模型 13–16%，专用小模型 32–34%，差距在"有没有训练过"

### 1.3 CPCD 生态普查

- ✅ 确认 CPCD（Psy-Chronicle）**没有改进生态**（★4、0 fork、零篇衍生论文）
- ✅ 识别长程记忆文献作为替代：LoCoMo（756 引）、MemoryBank（622 引）、LongMemEval（543 引）

### 1.4 心理健康文献普查

- ✅ 识别两篇必读：stigma 论文（227 引）、SoulChat（185 引，中文先例）
- ✅ 识别 8 个潜在训练数据集（AntEngage、KokoroChat、OnCoCo、SoulChat corpus 等）
- ✅ 识别安全基准（MHSafeEval、CounselBench、Ψ-Arena）与方法参考（TheraMind、Psyche-R1）

### 1.5 深度核查

- ✅ 提取 stigma 论文三大失效模式（Don't Stigmatize / Don't Collude with Delusions / Don't Enable Suicidal Ideation）
- ✅ 定位 stigma 论文 GitHub 仓库 `jlcmoore/llms-as-therapists`（有评测代码）
- ✅ 提取 SoulChat 方法（三项能力：Empathy / Listening / Comfort，VAR-Safe 安全机制）
- ✅ 定位 SoulChat2.0 仓库（★252）
- ⏸ 数据集许可核查（AntEngage / OnCoCo / SoulChat corpus 待访问 HF / 克隆仓库）
- ❌ Kardia-R1 模型下载失败（网络问题）

---

## 二、交付物清单（10 个报告 + 1 个协议 + 3 个数据文件）

### 核心报告（6 个）

1. **`reports/LITERATURE_SURVEY_SUMMARY_20260828.md`** —— 总纲，包含：
   - ESConv 两个时代的发现
   - CPCD 生态判断
   - Kardia-R1 核心信息
   - 立即可行的三条路线

2. **`reports/esconv_literature_recheck_20260828.md`** —— ESConv 文献重查详细版
   - 179 篇去重、53 篇相关、26 篇有码
   - 两个时代对比表
   - 代码仓库定位结果

3. **`reports/cpcd_and_psych_literature_survey_20260828.md`** —— CPCD + 心理健康普查
   - CPCD 无生态的证据
   - 8 个心理咨询数据集
   - 长程记忆文献（LoCoMo / MemoryBank / LongMemEval）
   - 安全基准与评测

4. **`reports/stigma_paper_findings_20260828.md`** —— stigma 论文核心发现
   - 三大失效模式
   - 治疗联盟的根本性障碍
   - 对静室安全门的启示

5. **`reports/soulchat_method_extraction_20260828.md`** —— SoulChat 方法提取
   - 三项能力（Empathy / Listening / Comfort）
   - VAR-Safe 安全机制
   - 与我们现有评测的对应

6. **`reports/dataset_license_check_20260828.md`** —— 数据集许可核查
   - 8 个数据集的待核实清单
   - 合成数据警告
   - 下一步优先级

### 基线报告（3 个）

7. **`reports/qwen_finetune_pretest_baseline_20260828.{md,json}`** —— pre-finetune 基线
   - CPCD 四臂配对（159/159/156/136）
   - ESConv fixed-250 四臂（已结算 8/13 报告，本次回验一致）
   - Qwen3.8 新臂：13.60% [9.20%, 18.62%]

8. **`reports/esconv_fixed250_qwen38_pretest_20260828.{md,json}`** —— Qwen3.8 ESConv 专项
   - 四臂两两比较 Holm p 全为 1

9. **`reports/qwen38_availability_20260828.md`** —— Qwen3.8 可得性核查
   - 架构与 3.6 逐字段一致
   - DashScope 第一方已上架
   - thinking 开关实测通过
   - system_fingerprint 缺失的证据降级

### 其他报告（1 个）

10. **`reports/local_compute_and_serving_options_20260828.md`** —— 本地算力决策树 + 生产端点方案

### 协议与数据

11. **`open_response_eval/preregistration_dashscope_qwen38_esconv250_addendum.json`** —— 冻结协议
12. **`tmp/pwc_esconv_papers.json`** —— ESConv 179 篇原始数据
13. **`tmp/pwc_cpcd_papers.json`** —— CPCD 118 篇原始数据
14. **`tmp/esconv_repo_probe.json`** —— 代码仓库探测结果

### 台账

15. **`project-ledger/updates/2026-08-28-qwen-therapy-prep-sprint.md`** —— 工作台账

---

## 三、核心发现（3 条）

### 🔴 发现 1：ESConv 已分成两个时代，你的收藏全在旧的

| | 第一代 2021–2023 | 第二代 2025–2026 |
|---|---|---|
| 做法 | 小模型 + 策略分类头 | **LLM + RL，用 rubric 判官当奖励** |
| 报什么 | 八类 ACC（32–34%） | rubric 分，**基本不报八类 ACC** |
| 你的收藏 | ✅ 覆盖不错 | ❌ **一篇都没有** |

**Kardia-R1 是本轮最大发现**：7B 模型已开源（MIT），用 rubric-as-judge GRPO，**与我们现有 proxy-judge 管线同构**。

### 🔴 发现 2：CPCD 没有改进生态，差距是数量级的

| | ESConv | CPCD |
|---|---|---|
| 原论文引用 | **482** | — |
| GitHub | 生态活跃 | **★4，0 fork** |
| 衍生论文 | 53 篇 | **0 篇** |

但 CPCD 测的长程记忆能力有独立文献（LoCoMo / MemoryBank / LongMemEval），三篇都有代码。

### 🔴 发现 3：Qwen3.8 零样本与 Qwen3.6 / DeepSeek 全部不显著

**13.60%** vs 14.00% / 16.00% / 16.40%，Holm p 全为 1。

**坐实立项判断**：换哪个 API 模型都在 13–16%，专用小模型是 32–34%——**微调要补的是这个 20 个百分点的缺口，不是 Qwen 与 DeepSeek 之间的 2 个百分点。**

---

## 四、立即可做（按优先级）

### 🟢 本周（Week 1）

1. **访问 AntEngage HF 页面**，核实许可、规模、gated 状态
2. **克隆 SoulChat2.0**，找 corpus 公开情况与许可
3. **克隆 `jlcmoore/llms-as-therapists`**，读 stigma 检测代码
4. **下载 Kardia-R1 模型**（网络问题解决后），跑 ESConv fixed-250 零样本评测

### 🟡 下周（Week 2）

5. **Kardia-R1 训练代码审计** + rubric-as-judge 与我们 proxy judge 的对接方案
6. 读 **KokoroChat 论文**，提取真人收集方法
7. 读 **SoulChat 论文**，对照中文数据构造
8. **stigma 三大失效模式并入部署安全门**

### 🔵 两周后（Week 3–4）

9. 补齐第一代高引论文复跑（MISC / KEMI / CauESC / TransESC）
10. LoCoMo / MemoryBank / LongMemEval 作为 CPCD mr/tcr 改进入口评估

---

## 五、未完成的项（原因与补救）

### ❌ Kardia-R1 模型下载

**原因**：HuggingFace 网络连接失败（"cannot find the appropriate snapshot folder"）

**补救**：
- 方法 1：换网络环境（VPN / 镜像站）
- 方法 2：用 `huggingface-cli download` 而非 `snapshot_download`
- 方法 3：直接在 HF 页面手动下载权重文件

### ❌ GLHG 代码仓库

**原因**：多种搜索词均未找到公开仓库（404）

**判断**：GLHG（114 引，ACL 2022）可能没有公开代码，或者代码在作者个人页面 / 机构内网

**补救**：
- 方法 1：直接邮件联系作者 Peng Yang 请求代码
- 方法 2：按论文描述自己实现（层次图 + 全局到局部）

### ⏸ 数据集许可详细核查

**原因**：HF API 网络问题，部分数据集托管位置未找到

**补救**：已写成独立报告 `dataset_license_check_20260828.md`，列出待核实清单与优先级

---

## 六、给 Fable 审阅的重点

### 需要 Fable 确认的三个判断

1. **"ESConv 分成两个时代"的划分是否合理？**
   - 第一代：小模型 + 策略分类（32–34% ACC）
   - 第二代：LLM + RL（rubric 分，不报八类 ACC）
   - 这个划分是否遗漏了重要的中间形态？

2. **"CPCD 没有改进生态"的结论是否过于绝对？**
   - 证据：★4、0 fork、零篇标题含 Psy-Chronicle/CPCD 的论文
   - 是否有可能存在隐性工作（如企业内部、未公开的硕博士论文）？

3. **Kardia-R1 作为"最大发现"的优先级是否合理？**
   - 优势：开源、同框架、方法同构
   - 风险：7B 模型性能是否足够、零样本表现未知、训练成本未评估

### 需要 Fable 补充的角度

1. **方法论审查**：Papers with Code API 的使用是否遗漏了重要来源（如 Google Scholar / Semantic Scholar / DBLP）？
2. **文献质量**：是否应该加入发表级别筛选（顶会 vs arxiv）？
3. **时间优先级**：在 Kardia-R1 / 补齐高引论文 / 读 stigma 安全 / 数据集许可核查 这四条线中，哪条应该最优先？

---

## 七、后续计划（Phase 2 准备）

### 短期（1–2 周）：Kardia-R1 复现

1. 下载模型 → ESConv fixed-250 零样本评测
2. 读训练代码 → 评估 GRPO 在 research-v0 上的可行性
3. rubric-as-judge 适配 → 改造 `proxy_cpcd_judge.py`

### 中期（2–4 周）：第一代论文补齐

- MISC（157 引）
- KEMI（82 引）
- CauESC
- TransESC

### 长期（Phase 2 正式训练）

- 等本地算力到位 → 按 `local_compute_and_serving_options_20260828.md` 决策树选路
- 3.8 训练包重打（合并 ERR-20260828-01 的完整性修复）
- GPU 实测三项：`language_model_only` 断言、3.8 chat template loss mask、fast-path 绑定
