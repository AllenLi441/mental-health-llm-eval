# 模型优化第二轮结案报告（2026-07-28）

## 结论先行

当前仍有部分任务低于原论文 benchmark，而且差距主要集中在“零样本 LLM 对比任务微调判别式”
这一组，不是简单改一版 prompt 就能消除：

- PsySUICIDE 当前确认性 `V4-Pro + taxonomy` 为 accuracy `88.11%`、macro-F1 `0.6371`；
  比论文 RoBERTa-large 分别低 `3.58pp` 和 `0.0605`。它高于 GPT-4-preview zero-shot
  accuracy，但这不是同协议模型对比。
- 新训练的 PsySUICIDE RoBERTa v1 在三 Seed official-valid 上为 accuracy `94.01%`、
  macro-F1 `0.7380`（均值），选中的 Seed 43 在预先冻结的 2,329 条内部 holdout 上为
  `93.43% / 0.7275`。这表明该专用分类器在未见内部样本上仍维持较高分数，但它不是
  PsySUICIDE 论文 test，也没有同一 holdout 上的 LLM 对照，不能据此量化相对提升或作
  预注册配对检验结论。
- IMHI 的统一 v3u 在 9 个可用任务中有 8/9 的点值高于 ChatGPT zero-shot，却只有 1/9
  高于 MentaLLaMA-13B，0/9 高于各任务 fine-tuned 判别式。
- EmoBench 当前可进入 scoreboard 的仍是 temp-0 单次 proxy：EA 比论文 GPT-4 中英均值低
  `0.35pp`，EU 高 `14.60pp`。论文 5 次采样多数票 × 4 个选项排列的完整 8,000-call
  重算尚未发生，不能把 proxy 差值写成完全同协议胜负。

本轮真正完成的是把“继续凭感觉改 prompt”改造成可证伪、会自动停止的优化流程。两个新
LLM prompt 候选都被真实数据否决，没有用局部好看的分数替换当前赢家；监督 RoBERTa
候选按三随机种子预注册运行，最终状态见下方监督实验小节。

## 1. 当前总表已纠正

`scripts/scoreboard.py` 不再扫描“最新文件”或混入旧 pilot，而是固定读取具名的当前聚合资产：

- CPsyExam：3,902 全量配对，不再使用旧 `n=599`。
- PsySUICIDE：accuracy 对 accuracy、macro-F1 对 macro-F1，不再把 weighted-F1 错配为
  accuracy。
- IMHI：只使用统一 v3u，不把曾经只优化 2/9 任务的选择性结果当最终证据。
- EmoBench：明确标成 temp-0 single proxy，不冒充论文采样协议。

总表是跨论文协议的描述性点估计；只有各自报告中预注册且保留逐行配对的内部比较才能作
显著性结论。

## 2. PsySUICIDE 数据隔离

在任何第二轮 prompt 或监督训练发生前，官方 train 的 11,671 条单标签记录被按类、按承诺
哈希确定性切分为：

| 分区 | 行数 | 用途 |
|---|---:|---|
| optimization | 9,342 | 训练或 few-shot 检索 |
| frozen holdout | 2,329 | 只有 valid 预注册门槛通过后才允许一次性内部确认 |

holdout commitment 为
`2691aca6db327d4e8c3b3cefe49b68716506d4181344883ae0f8a3e19a53fe30`。
公开仓库只保留计数与 commitment，不保留成员、文本、ID、标签或逐行预测。

## 3. PsySUICIDE 新 prompt 候选

### Same-50 smoke

三臂使用同一确定性 50 条 official valid 样本，全部 150 次请求为 0 error / 0 invalid：

| 候选 | Accuracy | Macro-F1 | 估算费用 |
|---|---:|---:|---:|
| 旧 taxonomy | 86.00% | 0.2083 | $0.00457 |
| taxonomy-v2 | 88.00% | 0.3201 | $0.00757 |
| balanced few-shot | 84.00% | 0.2398 | $0.01512 |

balanced few-shot 每类只从 optimization partition 检索一个示例，排除目标行并把 holdout
commitment 纳入检索指纹；它在 smoke 同时输给 taxonomy-v2 且成本约为旧 taxonomy 的
3.3 倍，因此按 futility/cost 门槛停止。

### Taxonomy-v2 完整 valid

taxonomy-v2 完成 1,459 条 official valid，全程 0 error / 0 invalid，费用 `$0.18508`：

| 配置 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| 旧 taxonomy | 88.55% | 0.5315 | 0.8842 |
| taxonomy-v2 | 85.26% | 0.4822 | 0.8519 |
| 差值 | -3.29pp | -0.0493 | -0.0323 |

探索性配对分析的 macro-F1 randomization `p=0.01905`、bootstrap 95% CI
`[-0.09405, -0.00940]`；因为重采样参数没有预注册，这里只作探索性证据。被动意图、
主动意图、计划等类别明显退步，最终决定为 `REJECT_TAXONOMY_V2`，holdout 未评分。

## 4. EmoBench 论文协议

`emobench-official/paper_protocol.mjs` 已实现论文描述的 temperature `0.6`、每个排列 5 次
采样多数票、原排列加 3 个选项重排、4 个排列准确率取均值。官方发布代码没有实现论文中的
聚合与排列，因此额外冻结了确定性排列种子、EU 分字段投票和 tie-break 假设。

8 个唯一题目的完整性 smoke 实际发出 160 次请求：0 error、usage 160/160、响应模型和
fingerprint 均通过，费用 `$0.06380`。全量需要 8,000 次调用；按 smoke 校准并加 1.5 倍
预留需要 `$4.7851`，当时余额为 `¥6.81`，所以状态为 `BLOCKED_BY_PROVIDER_BALANCE`。
没有把 8 题 smoke 当成 benchmark。

## 5. IMHI 九任务统一候选

contrastive-v4 对全部 9 个任务使用同一 evidence-first 框架与事先写定的标签边界，不允许
逐任务选择“哪个 prompt 更好看”。same-10 矩阵共 180 次请求，0 error / 0 invalid：

| 统一聚合（九任务 weighted-F1 等权均值） | 分数 |
|---|---:|
| uniform-v3u control | 0.6391 |
| contrastive-v4 candidate | 0.5689 |
| 差值 | -0.0702 |

候选在 DR、dreaddit、loneliness、MultiWD 四项退步超过 2pp；虽然 CAMS、swmh 有小幅改善，
统一规则禁止只留下局部赢家。决定为 `REJECT_CONTRASTIVE_V4`，没有启动 8,638-call 全量。
即使候选通过，按 smoke 校准的 `$5.5178` 预留也超过当时 `¥6.32` 余额。

## 6. 监督 RoBERTa 三随机种子

监督候选的配置与门槛已先于训练提交：
`hfl/chinese-roberta-wwm-ext-large`、seeds 42/43/44、3 epochs、加权替换采样、
inverse-fourth-root 类别权重与 focal loss。主门槛是三种子 official-valid macro-F1
均值严格高于 `0.5314999909`；同时 official-valid support ≥10 的任何类别，三种子均值
F1 不得比旧 taxonomy 低超过 `0.10`。

<!-- SUPERVISED_RESULT_BEGIN -->
三种子均正常完成，报告器重新核验了执行 commit 中 trainer 字节、预注册配置、partition
commitment、类别权重、逐类指标代数、三种子聚合和 checkpoint 标签映射：

| Seed | Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 42 | 93.83% | 0.6935 | 0.9381 |
| 43 | 94.17% | 0.7616 | 0.9425 |
| 44 | 94.04% | 0.7590 | 0.9408 |
| 三种子均值 | 94.01% | 0.7380 | 0.9404 |
| Sample SD | 0.17pp | 0.0386 | 0.0022 |

相对冻结 reference macro-F1 `0.5315`，均值提高 `+0.2065`；9 个 support ≥10 的逐类
回退门槛全部通过。因此预注册 valid 决策为
`ADVANCE_SUPERVISED_V1_FREEZE_CHECKPOINT_BEFORE_HOLDOUT`，按最高 macro-F1 选择 Seed 43。

Seed 43 权重 SHA-256 为
`9a1cf3c2471dedc2ee8799ba6b8324f10e3d01ce1c0d0898da2b9aa642743e44`；
权重、config、tokenizer 与 training args 的文件级哈希均写入 valid selection 和 holdout
freeze spec。基础模型本地字节与预注册 Hugging Face revision 相符，但 trainer 通过本地
路径加载，三个 seed 的 `model_revision` 均记录为 `null`，因此只能称“运行后 artifact
字节核对通过”，不能声称运行时 revision 被 Transformers 记录。v1 trainer 也没有记录
official-valid 文件或成员 commitment，当前只能核验 1,459 行、固定标签/support 与聚合
算术；这两项限制均保留在公开 selection。

上述 pre-holdout 资产先提交并推送为
`a531c6c498602e258bf58c98ed71459a4d7b3757`，本地、upstream 与 live remote SHA
核对一致后，scorer 才建立持久独占 claim 并首次加载 holdout。唯一完成结果为：

| 分区 / 模型 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Official valid / Seed 43 | 94.17% | 0.7616 | 0.9425 |
| Frozen internal holdout / Seed 43 | 93.43% | 0.7275 | 0.9348 |
| Holdout − valid | -0.74pp | -0.0340 | -0.0077 |

holdout 的 11 类 support 总和为 2,329；逐类聚合中，被动意图 F1 `0.8685`、主动意图
`0.9176`、计划 `0.8085`，但只有 2 条的“自杀准备行为”仍为 `0.0`。因此结果支持
“模型能在未见内部样本上维持较高整体与 macro-F1”，同时也确认极稀有类仍未解决。

公开 result SHA-256 为
`65d2f716c2c59d2465affb7baaa9041060a3dd6b244476176e8d4bff77a674d2`。
私有 claim/completion receipt 权限为 `0600`，claim ID 一致；逐行文本、ID、gold、
prediction 与 logits 均未持久化。这个证据只证明本 scorer ledger namespace 内完成了
一次 claim，不能证明不存在绕过工具的历史访问，也不能用于显著胜出、统计平手或等价结论。
<!-- SUPERVISED_RESULT_END -->

### Accuracy-first v2 冻结状态（2026-07-31）

v2 没有回看或重复评分 v1 已消费的 frozen holdout，也没有使用 official valid/test 调参。
在一次明确授权下，固定 exporter 对既有 full-train 文件做了一次内容读取和分区复现：源文件
SHA-256、11,671 条保留行、9,342 条 optimization、2,329 条 holdout 以及三套 commitment
均与 Stage A 公开承诺精确一致。只有 optimization 行被写到仓库外 `0600` 文件；holdout
没有形成单独的行集合或输出，公开仓库只保存聚合审计。

optimization-only 文件随后确定性冻结为：

| v2 分区 | 行数 | commitment SHA-256 |
|---|---:|---|
| inner-train | 7,479 | `bb14eb605af3655fd951c9e59c51536e979330b814c6982efed09c8db4ebf1c8` |
| inner-dev | 1,863 | `b229ae62d034ca2926ae8e0b8cd00f4aed18493422432a08ab38b7b52b7b75d7` |

两者 row-digest 重叠为 0，11 个 inner-dev 标签分层 commitment 已写入最终预注册。最终
协议保持四个固定臂：自然采样+CE、加权采样+CE、自然采样+weighted focal、加权采样+
weighted focal；10 epochs、逐 epoch 验证并按未舍入 accuracy 保存最佳 checkpoint。

此处完成的是数据边界和预注册冻结，不是模型效果。当前尚未运行 code smoke、Seed 42
四臂 screen 或 Seeds 43/44/45 confirmation，因此不能宣称 v2 提高了 accuracy、macro-F1、
其他 benchmark 或静室路由。公开证据为
`reports/psysuicide-roberta-v2-optimization-export.audit.json`、
`reports/psysuicide-roberta-v2-accuracy-first.split-freeze.json` 与
`reports/psysuicide-roberta-v2-accuracy-first.prereg.json`。

## 7. 现在该怎么继续

优先级不是再同时发散多个 prompt，而是按证据分层：

1. 监督 v1 已按规则完成并通过内部 holdout 描述性确认。下一轮必须另建预注册和分支，
   只从 optimization 建 inner-dev，先解决训练不足、双重平衡与极稀有类；不得根据本次
   holdout 改 v1、调阈值或再次评分。
2. EmoBench 只在账户可提供至少 `$4.79` 独立批准预算时运行完整 8,000 calls；在此之前
   scoreboard 继续明确标识 proxy。
3. IMHI 不再扩大全量 LLM prompt 搜索。下一条合理路线是每任务训练判别式 encoder，或在
   独立 development split 上做阈值/类别不平衡优化；最终仍必须统一报告九任务，不能只挑
   两个改善项。
4. 后续每个候选继续使用“冻结方案 → 小 smoke → 统一晋级门槛 → 完整 valid → 唯一
   holdout/test”的顺序，并同时报告负结果、调用完整性与成本。

## 8. 发布与隐私边界

本报告、代码、预注册、aggregate metrics 与 commitments 可公开。授权数据、敏感文本、ID、
逐行 gold/prediction、原始模型输出、API key、本地模型权重和 trainer state 均保留在本地
ignored 路径，不进入 Git 或 Release。
