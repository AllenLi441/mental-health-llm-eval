# 2026-08-28：Qwen 疗愈模型准备冲刺（基线结算 / 3.8 可得性 / 训练包完整性）

## 为什么更新

用户拍板把静室后端替换为自研微调模型，基座从 Qwen3.6-27B 改为 **Qwen3.8-27B**，并明确**不租云 GPU**（本地算力方案下周到位、机器与库尚未确定）。本周为准备冲刺：全部本机可做，不启动任何训练。清单见 `HANDOFF-qwen-therapy-model-20260828.md` §0bis。

本次记录第 1、2、5、6 条的执行结果。

## 修改范围

- 分支：`codex/cross-benchmark-accuracy-family-v1`（未新建分支；本次改动与在飞的 EmoDynamiX 干净重训互不相交）
- 新增文件：
  - `scripts/aggregate_pretest_baseline.py`
  - `reports/qwen_finetune_pretest_baseline_20260828.{md,json}`
  - `reports/qwen38_availability_20260828.md`
- 涉及数据/付费 API/模型训练：**否**。本次零 API 调用、零训练、零付费。

## 做了什么

### 1. Pre-finetune 基线结算（§0bis 第 1 条）

新增可复算聚合脚本，把 8/08–8/13 已生成的四臂输出与盲评记录结算成 pre-test 基线。ESConv fixed-250 不重算，改为读取 `reports/esconv_fixed250_deepseek_v4_pro_0813_20260813.json` 并**回验其声明的源文件 sha256**（4/4 一致）。

### 2. Qwen3.8-27B 可得性核实（§0bis 第 2 条）

直读 HF 模型 API 与两版 `config.json` 比对，不采信二手转述。结论落盘 `reports/qwen38_availability_20260828.md`。

### 5. 训练包 3.8 适配前置核查（§0bis 第 5 条）

审计 `静室/datasets/mental-health-instruct-research-v0/server_training_bundle/`，确认迁移工作量，并**发现该 bundle 的完整性冻结已破**（见下）。

### 6. 数据终验（§0bis 第 6 条）——**未通过**

## 当前指标

### CPCD 开放回复（内部 proxy judge，1–5 分，非公开榜分）

| 臂 | n | micro 均分 | macro 均分 |
|---|---:|---:|---:|
| deepseek_v4_pro | 159 | 4.281 | 4.417 |
| qwen_3_6_27b_siliconflow | 159 | 4.198 | 4.315 |
| deepseek_v4_flash | 156 | 4.155 | 4.086 |
| qwen_3_6_27b | 136 | 4.118 | 4.314 |

配对（同 task_id 交集，10,000 次 bootstrap，seed 20260828）：

- `qwen_3_6_27b_siliconflow − deepseek_v4_pro` = **−0.083，95% CI [−0.176, +0.010]**，159 对中 106 对同分 → **差异不显著**
- `qwen_3_6_27b − deepseek_v4_pro` = −0.120，CI [−0.225, −0.013]（n=136）
- `deepseek_v4_pro − deepseek_v4_flash` = +0.121，CI [−0.006, +0.253]
- `qwen_3_6_27b_siliconflow − qwen_3_6_27b` = +0.055，CI [−0.005, +0.113]（两次 Qwen 运行互相一致）

生成侧截断率：V4-Flash 33/156 = **21.15%**，V4-Pro 12/159 = 7.55%，Qwen 两臂 **0%**。

### ESConv fixed-250 八类策略（沿用 8/13 已结算报告，源哈希 4/4 复验一致）

| 臂 | Accuracy | Cluster-bootstrap 95% CI |
|---|---:|---:|
| DeepSeek V4-Flash | 16.40% | [12.13%, 21.12%] |
| DeepSeek V4-Pro-0813 | 16.00% | [12.59%, 20.00%] |
| DeepSeek V4-Pro（旧测） | 14.80% | [10.31%, 19.92%] |
| Qwen3.6-27B（未微调） | 14.00% | [9.62%, 19.20%] |

同一 frozen 协议下的工程轨基线：BlenderBot-small Joint 32.22%、EmoDynamiX 33.61%。

## 错误与失败用例

### ERR-20260828-01：research-v0 训练包完整性冻结已破（阻塞 §0bis 第 5、6 条）

三层校验全部不一致：

1. **外层 `releases/SHA256SUMS` 过期。** 记录 `f91385cb…`，实算 `d2f8c2d7…`。文件时间戳显示压缩包（07:32）在校验和文件（07:29）**之后 3 分钟**被重新生成，校验和未同步重算。
2. **压缩包内容落后于磁盘源，且缺少许可证文件。** 压缩包自身 `BUNDLE_CONTENTS.sha256` 18/18 通过（内部自洽），但与磁盘上的 `server_training_bundle/` 相比：缺 `ATTRIBUTION.md`、**缺整个 `licenses/` 目录**（AugESC 与 ESConv 均为 CC-BY-NC，随包分发许可证是权利边界要求），且 `bootstrap.sh` / `preflight.py` / `train_qlora.py` / 两个 config / 两个 README 内容较旧。
3. **磁盘源自身也不通过自己的 manifest。** `shasum -c BUNDLE_CONTENTS.sha256` → 8 项 FAILED（ATTRIBUTION.md、README.md、两个 config、data/README.md、bootstrap.sh、preflight.py、train_qlora.py），另有 1 项 `licenses/EmpatheticDialogues-CC-BY-NC-4.0.txt` **列在清单里但文件不存在**；反过来实际存在的 `licenses/{AugESC,CPCD,ESConv}-*.txt` **未列入清单**。

磁盘版的改动本身看起来是有意的改进（`bootstrap.sh` 增加了 `sha256sum --check` 强制校验；`train_qlora.py` 从 `AutoModelForCausalLM`/`qwen3_5_text` 迁移到 `AutoModelForMultimodalLM`/`qwen3_5` 多模态包装并新增视觉塔冻结与 LoRA 边界断言；eval/save steps 由 100 收紧到 20），只是**改完没有重算 manifest、也没有重打包**。

**未自行修复。** 重算冻结数据集的完整性清单属于数据治理动作，需 owner 决定；且 3.8 迁移本就要重打包，两件事应合并成一次受控重打。

## 验证

```
python3 scripts/aggregate_pretest_baseline.py          # PASS，写出 md+json
shasum -a 256 -c releases/SHA256SUMS                   # FAIL 1/1（见 ERR-20260828-01）
shasum -a 256 -c BUNDLE_CONTENTS.sha256（磁盘版）       # FAIL 8 项 + 1 项文件缺失
shasum -a 256 -c BUNDLE_CONTENTS.sha256（压缩包内）     # PASS 18/18
exports/train.jsonl                                    # PASS n=500，zh 300 / en 200
exports/development.jsonl                              # PASS n=50，zh 30 / en 20
来源分布                                                # PASS cpcd 300/30、augesc 199/19、esconv 1/1
```

## 结论边界

**已证明**

- Qwen3.8-27B 与 Qwen3.6-27B 的 `config.json` 在训练脚本断言涉及的每个字段上完全一致（`model_type=qwen3_5`、`text_config.model_type=qwen3_5_text`、64 层、48 linear + 16 full、vocab 248,320 等）；496 这个 LoRA 模块数正是该布局的算术结果，故迁移**不需要改架构断言**。
- research-v0 的 exports 行数、语言比、来源分布与冻结声明一致。
- ESConv fixed-250 已结算报告所声明的四个源文件 sha256 与磁盘实算一致。

**描述性**

- CPCD 分数是内部 proxy judge 结果，只能在本项目 pre/post 同 judge 配对中使用，不可对外。
- 未微调 Qwen 与 DeepSeek V4-Pro 在 CPCD 上差异不显著（CI 跨 0）；在 ESConv fixed-250 上四臂差异同样不显著（原报告 Holm p 全为 1）。

**不可比**

- 四臂判分数 159/159/156/136 不等，只能看配对交集，不能直接比 micro 均分名次。
- SiliconFlow Qwen 臂（8/13 判分）的 judge fingerprint 为 `a26a7955944dc5c60445bff77fac9c8e`，其余三臂（8/09 判分）为 `fp_a18b46594c_prod0820_fp8_kvcache_20260402`，格式不同一次判分，post-test 必须并列披露。
- V4-Flash 21.15% 的截断率会压低其 tcr 分，该低分不可解读为能力差距。

**未完成**

- §0bis 第 3 条（prereg addendum v2 冻结）：待 provider smoke 确认 thinking 开关行为后再冻结。
- 第 4 条（pre-test 3.8 臂重跑）：已出报价 **$2.47–2.99**（含判官），待用户确认后执行。
- 第 5、6 条：被 ERR-20260828-01 阻塞。
- 第 7 条（本地算力对接）：用户尚未确定机器与运行库，按条件预案处理。
- 第 8 条（生产端点预研）：未开始。

## GitHub 状态

- Commit SHA：未提交（本次为工作区改动）
- Remote branch：`codex/cross-benchmark-accuracy-family-v1`
- Draft PR：#5（本次改动与其内容不相交）
- 是否部署：评测仓库不涉及部署
