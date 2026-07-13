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
2. **dreaddit 从"超越微调 SOTA"降级为"平手"。** 统一协议 reasoner **81.1,95%CI[77.1,84.6] 覆盖微调 MentalRoBERTa 81.8** → 统计上平手,**不能声称超越微调**。之前的 82.8 是非统一的旧 prompt 下取得,不作数。
3. **统一协议下,IMHI 9 项:8/9 胜 ChatGPT 零样本(DR 现略低),仅 dreaddit 胜 MentaLLaMA-13B,0/9 明确胜微调判别式 SOTA。**

## 全基准里真正稳健、经得起统一协议/复现的胜项
- ✓ **CPsyExam 83.1% > GPT-4 零样本 67.4**(n=3902,非 prompt 脆弱)
- ✓ **EmoBench-EU 63.5 > GPT-4 56.9**(官方协议 harness)
- ✓ **MentalManip 77.8% > GPT-4-Turbo 零样本 65.7**
- ~ **dreaddit 81.1 ≈ 微调 81.8**(平手,胜 ChatGPT-ZS + MentaLLaMA-13B)
- ✗ 其余 IMHI 与 PsySUICIDE 细粒度均低于微调 SOTA(零样本对微调的正常差距)

## 元结论(这才是最该写进论文的一句)
零样本 LLM 在这些二值临床分类上的分数**对 prompt 框架和类别基率高度敏感**:模型天然有"看着像临床就答 yes"的倾向,通用 prompt 让它在高基率任务上"蒙对"、低基率任务上"蒙错"。**只有统一、非挑选、构念定义式的协议下的数字才可作证据。** 据此,静室零样本的真实强项是**中文心理推理 + 情绪理解**(CPsyExam、EmoBench 稳健超 GPT-4),而非英文临床单标签分类——后者整体仍是微调判别式的天下。

## 统计稳健性(bootstrap 95%CI,seed=42,B=2000)
每个胜负判定都带区间,不靠点值:

| 比较 | 我们 | 95%CI | 对照 | 判定 |
| --- | ---: | :---: | ---: | --- |
| CPsyExam vs GPT-4 | 83.1 acc | [81.9, 84.3] | 67.4 | ✅ 显著胜(CI 全在基线上) |
| MentalManip vs GPT-4-Turbo | 77.8 acc | [74.2, 81.2] | 65.7 | ✅ 显著胜 |
| EmoBench-EU vs GPT-4 | 63.5 acc | [58.8, 68.5] | 56.9 | ✅ 显著胜(下界 58.8 > 56.9) |
| dreaddit vs 微调 MentalRoBERTa | 81.1 wF1 | [77.3, 85.0] | 81.8 | ~ 平手(CI 覆盖 81.8) |
| dreaddit vs MentaLLaMA-13B | 81.1 wF1 | [77.3, 85.0] | 75.8 | ✅ 胜(下界 77.3 > 75.8) |
| DR vs MentaLLaMA-13B | 81.8 wF1 | [78.2, 85.3] | 85.7 | ✗ 输(上界 85.3 < 85.7) |
| loneliness/Irf/MultiWD vs 微调 | — | — | — | ✗ 均显著输 |

## 三个稳健胜项按「抗质疑度」排序(审稿人第一刀=数据污染)
1. **EmoBench-EU 最硬** —— 情境全人工手写、作者专门做了防污染设计,且用**论文官方协议 harness**(非我调的 prompt);显著超 GPT-4。这是全套里最经得起推敲的一条。
2. **MentalManip 次之** —— 康奈尔电影对白(理论上可能进训练集),但**在同一数据上超过 GPT-4-Turbo**,污染对双方对称,不影响相对结论。
3. **CPsyExam margin 最大但污染风险最高** —— 中文考试题库,模型可能背过题。已有污染探针(长选项 0/24 逐字复现、相似度中位数 0.538 ⇒ 未发现逐字背题,但同源题库弱污染不能排除);**引用时必须带此caveat,不可当"纯知识"胜利**。

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
