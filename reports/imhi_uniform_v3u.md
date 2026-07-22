# IMHI 统一协议评测(v3u)——去除选择性优化后的诚实结论

- 日期:2026-07-11
- 动机:Codex 的 v3 只给 **2/9** 个 IMHI 任务(dreaddit、MultiWD)加了 decision criteria,而这两个恰是唯二输给 ChatGPT 基线的任务。通用 prompt 有 **yes-偏置**(dreaddit v1:144 条 no→yes),该偏置在**高-yes 任务上抬分、低-yes 任务上压分**——只修低-yes 的两项 = 系统性利好自己。
- 本次:把**同一套构念定义 criteria 统一补到全部 5 个二值任务**(DR/loneliness/Irf 新增,dreaddit/MultiWD 沿用 Codex),多分类任务不变;chat + reasoner 两个模型都在统一协议下重跑(run-id `v3u`)。criteria 取自各构念的标准定义(抑郁 DSM、孤独主观困扰、Irf=人际自杀理论的 thwarted belongingness / perceived burdensomeness),**不看测试标签、不针对分数调**。

## 完整性核验(全过)
- 9 个任务 **gold 改变 = 0**(与 v2 逐 id 比对);**0 API error**;invalid 仅 CAMS 22 / t-sid 1,均为既有的模型不合规输出,解析器**未放松**(遵守 Codex 的"不改 parser"原则)。
- 一致性:dreaddit/MultiWD 的 `v3u` 与 Codex 的 `v3` **逐位一致**(79.30 / 71.15),证明 harness 确定性、我的复跑忠实。
- 4 个多分类任务 Δ=0.0,证明只动了二值任务,无附带改动。

## 结果:去偏后二值任务的真实分数(weighted-F1)
| 任务 | 金标 yes% | 通用prompt(旧) | 统一criteria(v3u, chat/reas 取优) | 变化 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| dreaddit | 51% | 59.9 | **81.1** | +21 | 低-yes,偏置本在害它 → 修对了 |
| MultiWD | 36% | 56.2 | **71.1** | +15 | 同上 |
| DR | 76% | 88.4 | **81.8** | **−7** | 高-yes,旧分是 yes-偏置**虚高** |
| loneliness | 79% | 81.8 | **73.2** | **−9** | 同上,虚高 |
| Irf | 65% | 68.8 | **63.9** | −5 | 同上 |

预测 yes 占比从"远高于金标"回落到"接近/略低于金标",偏置确实被去掉(去偏在低-yes 上略微反向、高-yes 上明显下拉,属构念定义的正常副作用,**未再调 prompt 去追分**)。

## 对"胜负"的诚实修订(相对我之前的说法)
1. **DR 不再是胜项。** 旧称"88.4 胜 MentaLLaMA-13B(85.7)"——统一去偏后 **81.8,谁都不胜**(连 ChatGPT-ZS 82.4 都略低)。那个胜是 yes-偏置假象。
2. **dreaddit 不再声称“超越微调 SOTA”。** 统一协议 reasoner **81.1**，MentalRoBERTa 论文点值 **81.8**，描述性差 −0.7pp。缺少对照逐行预测，不能做配对显著性检验，也不能把本模型单侧 CI 覆盖对照点值写成“统计平手”。之前的 82.8 是非统一旧 prompt，不作数。
3. **统一协议下，本 harness 覆盖论文 10 个 test sets 中的 9 个（未包含 CLP）：8/9 的点估计高于 ChatGPT 零样本（DR 略低），仅 dreaddit 的点估计高于 MentaLLaMA-13B，0/9 的点估计高于微调判别式 SOTA。** 这些是跨设置的描述性点值，不是胜负或显著性结论。

## 全基准里的描述性领先项
- ✓ **CPsyExam 83.1% > GPT-4 67.4(含少样本;严格零样本加权 57.6)**(n=3902；描述性比较，设置与污染 caveat 见下)
- ✓ **EmoBench-EU 63.5 > GPT-4 56.9**(官方数据、prompt 与评分兼容的 temp-0 单次 proxy；论文使用 5 次采样多数票 × 4 个选项排列，故不完全同协议；56.9 为论文 en/zh 分数均值,我们计算)
- ✓ **MentalManip 77.8% > GPT-4-Turbo 零样本 65.7**
- ~ **dreaddit 81.1 vs 微调 81.8**（描述性接近且略低；不能称“统计平手”。点估计高于 ChatGPT-ZS 与 MentaLLaMA-13B）
- ✗ 其余 IMHI 与 PsySUICIDE 细粒度均低于微调 SOTA(零样本对微调的正常差距)

## 元结论(这才是最该写进论文的一句)
零样本 LLM 在这些二值临床分类上的分数**对 prompt 框架和类别基率高度敏感**:模型天然有"看着像临床就答 yes"的倾向,通用 prompt 让它在高基率任务上"蒙对"、低基率任务上"蒙错"。**只有统一、非挑选、构念定义式的协议下的数字才可作证据。** 当前 CPsyExam 与 EmoBench 的本模型点估计相对较高，但污染、采样和论文协议差异仍在；它们只能支持后续复现实验优先级，不能直接写成对 GPT-4 的稳健胜出。

## 本模型区间与比较边界(bootstrap 95%CI,seed=42,B=2000)

下表区间只来自**本模型**逐行结果。公开基线没有逐行配对预测或其不确定性，因此这些区间不能作为两模型差值的显著性检验；“高于/低于”均是描述性点值比较。

| 比较 | 我们 | 95%CI | 对照 | 判定 |
| --- | ---: | :---: | ---: | --- |
| CPsyExam vs GPT-4 | 83.1 acc | [81.9, 84.3] | 67.4(含少样本;零样本 57.6) | 描述性高于；设置不同，不称显著 |
| MentalManip vs GPT-4-Turbo | 77.8 acc | [74.2, 81.2] | 65.7 | 描述性高于；缺基线逐行预测 |
| EmoBench-EU vs GPT-4 | 63.5 acc | [58.8, 68.5] | 56.9 | 描述性高于；56.9 是论文分语言均值 |
| dreaddit vs 微调 MentalRoBERTa | 81.1 wF1 | [77.3, 85.0] | 81.8 | 点值接近且略低；不作显著性判断 |
| dreaddit vs MentaLLaMA-13B | 81.1 wF1 | [77.3, 85.0] | 75.8 | 描述性高于 |
| DR vs MentaLLaMA-13B | 81.8 wF1 | [78.2, 85.3] | 85.7 | 描述性低于 |
| loneliness/Irf/MultiWD vs 微调 | — | — | — | 描述性低于 |

## 三个较高点估计的证据强弱（不作胜项计数）
1. **EmoBench-EU** —— 情境为人工编写，数据与 prompt/评分实现接近官方；但当前运行是 temp-0 单次 proxy，而论文是每题 5 次采样多数票并做 4 个选项排列后取均值。只能报告描述性差值，不能称完全同协议复现。
2. **MentalManip** —— 使用同一数据资源，但本项目是自定义 prompt 与抽样运行，论文对照也没有逐行预测；不得假设污染对双方对称，也不得把点值差写成显著优势。
3. **CPsyExam** —— margin 较大但污染与抽样风险最高。污染探针未发现逐字复现，仍不能排除同源题库弱污染；v4-pro 终端结果又只是 n=599 抽样，引用时必须同时披露样本与协议差异。

## 复现
```bash
cd datasets/eval-suite
# 统一协议全套(chat)
for t in imhi-dr imhi-dreaddit imhi-loneliness imhi-irf imhi-multiwd imhi-sad imhi-cams imhi-swmh imhi-t-sid; do
  node run.mjs $t --model deepseek-chat --seed 42 --run-id v3u; done
# 二值任务 reasoner
for t in imhi-dr imhi-dreaddit imhi-loneliness imhi-irf imhi-multiwd; do
  node run.mjs $t --model deepseek-reasoner --seed 42 --run-id v3u; done
```
criteria 定义见 `tasks/imhi.mjs`(5 个二值任务的 `decisionGuide`);parser/labels/gold 加载全程未改。
