# 数据集许可与可用性核查（2026-08-28）

> **2026-09-03 状态注记：** 本文保留 8/28 的候选核查过程，不能覆盖后续结论。
> B1 已按 A3 + zh:en=70:30 收口：`EN_cand_CBT` 9,000 条经人工复核后带条件放行，
> 商用候选池为 zh 70,792 + en 11,775（计划切片 39,250）；PsyQA 仍需协议，NC/
> research-only 来源仍排除。权威当前状态见 `论文和代码/B1_STATUS.md`。

调研过程中发现的潜在训练数据，逐个核实许可、规模、来源。

---

## 1. AntEngage Empathy Conversations

**HuggingFace**：`antengage/empathy-conversations`

**状态**：HF API 查询失败（网络问题），但从 PwC 搜索结果确认：

- 发布日期：2026-08-04
- 引用：0（太新）
- 有官方实现标记：✅

**待核实**：
- 许可证（License）
- 数据规模（多少条对话、平均轮数）
- 语言（英文 / 中文 / 多语）
- 来源（真人收集 / 合成 / 爬取）
- **是否 gated**（需申请访问）

**下一步**：
```bash
# 方法 1：直接访问 HF 页面
open https://huggingface.co/datasets/antengage/empathy-conversations

# 方法 2：用 datasets 库本地加载（如果不 gated）
from datasets import load_dataset
ds = load_dataset("antengage/empathy-conversations")
print(ds)
```

**预判**：AntEngage 是一个组织名（ant + engage），可能是企业或研究机构。如果是企业数据集，许可可能有限制（如仅研究用途）。

---

## 2. OnCoCo 1.0

**全名**：在线咨询细粒度消息分类公开数据集

**来源**：论文搜索结果提到，发布日期 2025-12-10，引用 1

**状态**：GitHub 搜索 `OnCoCo counseling` 未找到匹配仓库（404）

**待核实**：
- 论文 arXiv ID 或 DOI
- 数据集托管位置（HF / GitHub / Zenodo / 作者个人页面）
- 许可证
- "细粒度消息分类"的具体标注体系

**下一步**：
```bash
# 搜索论文
gh api search/repositories -f q="OnCoCo online counseling" 
# 或直接 Google Scholar 搜 "OnCoCo counseling dataset"
```

**预判**：1 引且 2025-12 发布，可能是刚被接收的会议论文，数据集可能还没公开或在审查期。

---

## 3. KokoroChat

**全名**：A Japanese Psychological Counseling Dialogue Dataset Collected via...

**状态**：
- 发布日期：2025-06-02
- 引用：10
- 无官方实现标记（可能是纯数据集，无模型）

**语言**：**日语**

**来源**：真人收集（"Collected via" 表明不是合成）

**用途**：
- **直接用作训练数据**：语言不匹配（我们需要中英文）
- **参考构造方法**：✅ 他们如何收集真人日语心理咨询对话的流程可以借鉴

**待核实**：
- 许可证
- 收集方法（在线咨询平台？志愿者？付费参与者？）
- 伦理审查与隐私保护措施（我们如果做中文收集，必须有同等保护）

**下一步**：读论文，重点看 Methodology 和 Ethics Statement 部分。

---

## 4. SoulChat Corpus

**来源**：SoulChat 论文（arXiv 2311.00273）与 `scutcyr/SoulChat2.0` 仓库

**状态**：仓库 ★252，活跃维护中

**语言**：中文

**待核实**：
- **语料是否公开**（仓库里有没有 `data/` 目录或下载链接）
- 许可证
- 数据规模
- 真人 vs 合成的比例

**关键问题**：
- 如果 SoulChat corpus 不公开，我们能否联系作者申请研究用途访问
- 如果是合成数据为主，按我们的红线必须走危机安全抽查

**下一步**：
```bash
git clone https://github.com/scutcyr/SoulChat2.0
cd SoulChat2.0
ls -R | grep -i data
cat README.md | grep -i license -A5
cat README.md | grep -i corpus -A10
```

---

## 5. MAGneT / Graph2Counsel / MCTSr-Zero / AnnaAgent

**共同特征**：全部是 **LLM 合成数据**

**状态**：
- MAGneT：多智能体协同生成，7 引
- Graph2Counsel：声称"临床接地"（clinically grounded），需核实其接地方式
- MCTSr-Zero：MCTS 自反思生成，有代码 ✅
- AnnaAgent：多会话记忆生成，23 引，有代码 ✅

**我们的红线**：
> `mental-health-instruction-dataset-v2` 要求真人金标。`MentalChat16K` 合成数据已因危机回复安全抽查失败被隔离。**这些合成数据集在采纳前必须走同一道危机安全抽查**，不能因为"是论文发的"就免检。

**用途**：
- **不能直接用作训练数据**（未经我们的危机安全抽查）
- **可以参考生成方法**：如果我们要做数据增强，可以借鉴他们的 prompt / agent 设计
- **Graph2Counsel 的"临床接地"值得深究**：如果他们有一套验证合成数据临床合理性的流程，我们可以复用

---

## 6. EmpatheticDialogues

**状态**：已审查，未采用（见 research-v0 文档）

**原因**：
- 许可：Facebook Research 的数据集，许可允许研究使用
- 未采用的原因：（需回查 research-v0 的决策记录）

---

## 总结表

| 数据集 | 语言 | 来源 | 许可 | 规模 | 状态 |
|---|---|---|---|---|---|
| **AntEngage** | 待查 | 待查 | 待查 | 待查 | ⏳ 需访问 HF 页面 |
| **OnCoCo 1.0** | 中文？ | 真人？ | 待查 | 待查 | ⏳ 未找到托管位置 |
| **KokoroChat** | 日语 | 真人收集 | 待查 | 待查 | 📖 可借鉴收集方法 |
| **SoulChat corpus** | 中文 | 真人+合成？ | 待查 | 待查 | ⏳ 需克隆仓库核实 |
| MAGneT | 中英 | LLM 合成 | 待查 | 待查 | ⚠️ 需危机抽查 |
| Graph2Counsel | 英文 | LLM 合成（声称临床接地） | 待查 | 待查 | ⚠️ 需核实接地方式 + 危机抽查 |
| MCTSr-Zero | 中文 | LLM 合成 | 待查 | 待查 | ⚠️ 需危机抽查，有代码 |
| AnnaAgent | 多语？ | LLM 合成 | 待查 | 待查 | ⚠️ 需危机抽查，有代码 |

---

## 立即可做（优先级排序）

### 高优先级（本周）

1. **访问 AntEngage HF 页面**，核实许可、规模、gated 状态
2. **克隆 SoulChat2.0**，找 corpus 的公开情况与许可

### 中优先级（下周）

3. 读 **KokoroChat 论文**，提取真人收集方法与伦理审查流程
4. 搜索 **OnCoCo** 论文，找数据集托管位置

### 低优先级（作为备选）

5. 如果上述真人数据集都不可用或不够，再考虑合成数据集
6. 合成数据集必须先读 **Graph2Counsel 的临床接地验证方法**
7. 对任何合成数据运行 **危机安全抽查**（复用 `MentalChat16K` 被隔离时的检测脚本）
