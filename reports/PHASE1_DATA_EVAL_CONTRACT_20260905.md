# Phase 1 数据、对照与验收合同

更新时间：2026-09-05
用途：在真正训练前，把数据、模型对照、评测集和放行条件固定下来。

## 先固定两条数据线

项目当前有两个不同用途的包，不能合并：

| 数据线 | 当前状态 | 用途 | 能否代表产品训练 |
|---|---|---|---|
| `mental-health-instruct-research-v0` | train 500 / development 50，静态门通过 | 训练代码、chat template、LoRA、保存/恢复和评测链路 smoke | 不能；`NOT_GOLD`、未人工复核、研究/非商业来源 |
| `mental-health-instruction-dataset-v2` | 目标 train 6,000 / development 600；目前批准记录 0 | 正式产品 SFT 数据生产 | 目前不能；真人写作、复核、权利、隐私和安全门尚未完成 |

`research-v0` 的 500/50 可以先在大机器上验证“能不能训练、能不能保存 adapter、评测脚本是否工作”。正式产品结论只能来自 v2 的 server-safe export。

有一处必须在开训前消除的漂移：`artifacts/m1-v1/audit/tokenization.json` 使用的是本地 Qwen3.8-27B tokenizer，而 v2 冻结草案把正式基座写成 Qwen3.6-27B。两者不能共用 tokenized cache；最终模型确定后，必须从 canonical messages 重新序列化并重跑长度、EOS 和 target 不截断审计。

`mental-health-llm-eval/artifacts/m1-v1` 另有一套 8,000/500/500 的中文候选切片（CPsyCounD 2,800/150/150、PsyDial-D1 2,000/150/150、CPsDD 3,200/200/200）。它的去重、PII、许可登记和 benchmark overlap 审计有结果，但它不是 v2 的 6,600 条真人批准数据，且 CPsDD 是 conditional research；在权利和人工质量门补齐前，只能作为候选或研究消融，不能直接进入正式产品包。

## 正式训练数据合同（v2）

最终正式包必须包含：

- `train.jsonl`：6,000 条 assistant target，中文 3,600、英文 2,400；
- `development.jsonl`：600 条，中文 360、英文 240；
- 每条记录只有一个要计算 loss 的 assistant target，历史 assistant 仅作上下文并 mask；
- `enable_thinking=false`，不保存 `<think>` 或隐藏推理；
- 每个 split 保持任务层、风险层和语言比例；
- 训练前完成 license、PII、近重复、benchmark contamination、人工复核和 slot 一对一检查；
- Human Gold Test 200 条和 Safety Eval 400 条必须独立冻结，永不进入训练或 checkpoint 选择。

当前 v2 真实状态可由以下文件重算：

- [`manifests/training_readiness.json`](/Users/allenli/Desktop/静室/datasets/mental-health-instruction-dataset-v2/manifests/training_readiness.json)
- [`DATASET_INVENTORY.md`](/Users/allenli/Desktop/静室/datasets/mental-health-instruction-dataset-v2/DATASET_INVENTORY.md)
- [`spec/DATASET_FREEZE_V2.md`](/Users/allenli/Desktop/静室/datasets/mental-health-instruction-dataset-v2/spec/DATASET_FREEZE_V2.md)

现在不得把 `production/candidates/`、`annotation_packets/`、AI draft 或 `production-slot-ledger` 当作训练数据。它们是候选、空白标注包和配额，不是 approved records。

## 研究 smoke 数据合同（research-v0）

research-v0 已固定为：

- train 500：CPCD 300、AugESC 199、ESConv 1；
- development 50：CPCD 30、AugESC 19、ESConv 1；
- 中文/英文分别为 300/200 和 30/20；
- 静态 PII、急性风险提示、重复、跨 split group overlap 均通过；
- `gold_status=NOT_GOLD`、`risk_level=R0_STATIC_SCREEN_ONLY`；
- 仅用于 private noncommercial research 和 pipeline smoke。

可执行检查：

```bash
cd /Users/allenli/Desktop/静室/datasets/mental-health-instruct-research-v0
python3 scripts/verify_bundle_contract.py
cd server_training_bundle
python3 scripts/test_model_profiles.py
shasum -a 256 -c BUNDLE_CONTENTS.sha256
```

正确结果是 contract OK、41/41 profile tests 和 27/27 checksum OK。这个包不包含 Qwen 权重。

## 模型对照必须这样做

模型选择与数据生产解耦。所有模型都从同一份 canonical messages 重新序列化，不能复用另一模型的 tokenized cache。

开训前固定三个候选臂：

1. `base-A`：选定 Qwen revision 的未微调模型；
2. `base-B`：另一尺寸或另一 Qwen revision（只有实际加载、chat template 和 tokenizer 审计通过后才可比较）；
3. `sft`：在完全相同 base revision 上挂载训练 adapter。

每个臂必须记录 model id、revision、tokenizer revision、runtime、量化方式、chat template 参数、解码参数和运行 fingerprint。不同模型、不同 split 或不同 prompt 的分数不能直接相减。

当前唯一已验证的研究 profile 是 `qwen3_8_27b`。7B/14B profile 仍是 placeholder，必须先实际下载并测出 revision、权重大小、架构层数和 LoRA module 数，才能改为 `VERIFIED`。

## 评测集与指标

评测集严格分为四类：

| 集合 | 规模 | 用途 | 是否参与训练/选模 |
|---|---:|---|---|
| Development | 600 | early stopping、checkpoint 和超参 | 可选模，不作最终泛化结论 |
| Human Gold Test | 200 | 人工专业回复质量 | 只在模型冻结后评分 |
| Safety Eval | 400 | 危机、误升级和严重失败 | 只在模型冻结后评分 |
| 外部 benchmark | CPCD 159、ESConv fixed-250、PsyDial-D101 等 | 外部对照和诊断 | 不参与训练或调参 |

主指标使用冻结的五维 rubric：`relevance`、`empathy`、`safety`、`boundary`、`naturalness`。每条 prompt 同时生成 base 与 SFT 回复，由两个独立 judge 盲评，按 conversation 做 cluster bootstrap。

正式成功条件：

1. 五维等权主端点 SFT > base，Holm 校正后单侧 `p < 0.05`；
2. crisis probe 不得比 base 退化；
3. stigma、delusion-collusion、ideation-enabling 等 safety probe 不得触发预设自动失败门；
4. 报告每个维度的效应量、95% CI、Holm p 值、judge agreement 和 invalid rate；
5. 样本量不足功效目标时，只能报告 underpowered，不能写成“没有效果”或“已经成功”。

ESConv 八类 ACC、macro-F1 和混淆矩阵只作结构诊断，不能作为开放式心理支持质量的唯一放行条件。`research-v0` 来源与 CPCD/ESConv/AugESC 相关，因此训练后不能再把这些来源衍生 benchmark 宣称为完全独立的 unseen test。

## 开训前唯一允许的顺序

1. 完成 v2 真人写作、盲复核、专家签核、权利和隐私报告；
2. 生成 canonical approved records，并按 slot 一对一检查；
3. 生成 server-safe train/development export 和 manifest；
4. 冻结 Human Gold、Safety Eval、benchmark contamination report；
5. 选择实际可用的 Qwen revision，重新做 tokenizer/chat-template 审计；
6. 在大机器上先做 image probe 和 discard-only smoke；
7. smoke 通过后才跑正式 QLoRA；
8. 训练后只在冻结 Development、Human Gold、Safety Eval 和外部诊断集上评测。

## 当前放行结论

- **可以准备上机 smoke：** research-v0 500/50；包和静态合同已通过。
- **不能启动正式产品训练：** v2 approved records 为 0/6,600，server-safe export、Human Gold、Safety Eval、隐私和专业签核均未完成。
- **不能提前决定 7B/14B：** 两个 profile 还没有真实 checkpoint 证据。
- **不能把 AI 草稿当金标：** AI draft 只能作为候选，必须经过独立真人实质改写和复核。
