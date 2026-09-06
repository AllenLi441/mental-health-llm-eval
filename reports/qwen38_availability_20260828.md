# Qwen3.8-27B 可得性与成本核实

日期：2026-08-28　对应交接书 §0bis 第 2 条
用途：为 prereg addendum v2（§0bis 第 3 条）与 pre-test 重跑报价（第 4 条）提供可引用的事实底稿。
核实方式：直接读 HuggingFace 模型 API 与 `config.json`、OpenRouter 供应商页，非二手转述。

---

## 1. 模型本体

| 项 | 值 | 来源 |
|---|---|---|
| 仓库 | `Qwen/Qwen3.8-27B` | HF |
| 创建 | 2026-08-05T08:22:59Z | HF API `createdAt` |
| 最后更新（视为发布） | 2026-08-14T15:00:01Z | HF API `lastModified` |
| **main commit sha** | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | HF API `sha` |
| License | Apache-2.0，`gated=false` | HF cardData |
| 参数量 | 27,781,427,952（BF16，18 个 safetensors 分片，55.6 GB） | HF API safetensors |
| pipeline | `image-text-to-text`（原生多模态） | HF API |
| 上下文 | 262,144 原生，YaRN 可扩至 1M | 模型卡 / OpenRouter |
| 思考 | 默认开启，`reasoning_effort` 档位 xhigh / medium / low，`preserve_thinking` 默认开 | 模型卡 |

**商用可用**：Apache-2.0 + 非 gated → 权重下载、微调、自托管、商用均无授权障碍。（训练**数据**的许可边界另算：AugESC / ESConv 为 CC-BY-NC，见 §4。）

## 2. 架构：与 Qwen3.6-27B 逐字段一致

这是本次核实最关键的结论。两者 `config.json` 在训练脚本断言涉及的每个字段上完全相同：

| 字段 | Qwen3.6-27B | Qwen3.8-27B | 一致 |
|---|---|---|:--:|
| `architectures` | `Qwen3_5ForConditionalGeneration` | 同 | ✅ |
| `model_type` | `qwen3_5` | 同 | ✅ |
| `text_config.model_type` | `qwen3_5_text` | 同 | ✅ |
| `text_config.num_hidden_layers` | 64 | 64 | ✅ |
| `layer_types` | 16×(3 linear_attention + 1 full_attention) | 同 | ✅ |
| `full_attention_interval` | 4 | 4 | ✅ |
| `vocab_size` | 248,320 | 248,320 | ✅ |
| `hidden_size` / `intermediate_size` / `head_dim` | 5120 / 17408 / 256 | 同 | ✅ |
| `num_attention_heads` / `num_key_value_heads` | 24 / 4 | 同 | ✅ |
| `max_position_embeddings` | 262,144 | 262,144 | ✅ |
| 存档 `transformers_version` | `4.57.1` | `5.8.0.dev0` | 仅此项不同 |

3.8 额外可见：`mtp_num_hidden_layers: 1`、`attn_output_gate: true`、`output_gate_type: swish`、`partial_rotary_factor: 0.25`、mrope section `[11,11,10]`、`linear_num_value_heads: 48` / `linear_num_key_heads: 16`。

**对训练包的意义**（详见 §3）：`server_training_bundle/scripts/train_qlora.py` 里的硬断言——`model_type=="qwen3_5"`、`text_config.model_type=="qwen3_5_text"`、48 个 `Qwen3_5GatedDeltaNet` 层、496 个 language LoRA 模块——按此 config **应当全部继续成立**：

```
48 linear_attention 层 × (5 linear_attn + 3 mlp) = 384
16 full_attention  层 × (4 self_attn  + 3 mlp) = 112
                                        合计 = 496 ✓
```

即 496 这个数字本就是"64 层 / 48 GDN / 16 GA"布局的产物，而 3.8 布局相同。

## 3. 迁移到 3.8 需要改什么（→ §0bis 第 5 条）

**必须改（config 级）**

1. `configs/*.json` 的 `model_id`：`Qwen/Qwen3.6-27B` → `Qwen/Qwen3.8-27B`
2. `model_revision`：`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` → `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
3. `train_qlora.py` 的报错文案里写死的 "Pinned Qwen3.6"（仅措辞，不影响行为）

**必须在 GPU 上实测才能确认（三项都 fail-closed，不会静默走错）**

4. `config.language_model_only is False` 断言——该键未出现在 3.8 的公开 config 字段里；若 transformers 不再注入该键，`getattr(..., None) is not False` 会直接抛错。**这是最可能的第一个失败点。**
5. **chat template 变了**：3.8 默认开思考且 `preserve_thinking=true`，模板与 3.6 不同。研究集每行带 `chat_template_kwargs`，**completion-only loss mask 必须在 3.8 模板下重新验证**（历史 assistant token 是否仍被正确 mask、思考段是否混入 completion）。这是本次迁移唯一有实质风险的一项。
6. `is_fast_path_available` / 48 个 GatedDeltaNet 模块的绑定检查——取决于 causal_conv1d 与 fla kernel 在实际镜像里的安装情况，与模型版本无关。

**很可能不用改**

7. `transformers==5.14.1`（当前 pin）> 3.8 存档的 `5.8.0.dev0`，且 3.8 复用 `qwen3_5` 建模路径 → 现有 pin 大概率可直接加载。**但 `bootstrap.sh` 的装前装后版本比对基线仍需随任何改动同步更新。**

**显存**：4-bit NF4 QLoRA、`minimum_gpu_vram_gib: 72`；3.8 参数量 27.78B 与 3.6 同量级 → 80GB 档位不变。

## 4. API 供应商与价格（用于 pre-test 重跑）

### 4.0 【2026-08-28 当日更正，优先于本节其余内容】DashScope 第一方已上架，且实测通过

用现有 `.env` 里的 `DASHSCOPE_API_KEY` 实探三个已配置端点，结论推翻了本节原先"必须换 OpenRouter/Cloudflare"的判断：

| 端点 | key 状态 | 是否有 3.8-27B |
|---|---|---|
| **DashScope 中国站** `https://dashscope.aliyuncs.com/compatible-mode/v1` | ✅ 有效 | ✅ **`qwen3.8-27b`**（另有 3.8-flash / 3.8-max / 3.8-2.4t-a95b） |
| SiliconFlow `https://api.siliconflow.com/v1` | ✅ 有效 | ❌ 仅 `Qwen/Qwen3.8-2.4T-A95B`（大 MoE），**无 27B** |
| DeepSeek `https://api.deepseek.com/v1`（判官用） | ✅ 有效，余额 **¥4.71** | 不适用 |
| DashScope 国际站 `dashscope-intl` | ❌ 同一 key 无效 | — |

**thinking 开关实测通过**（§6 要求的 smoke，已执行）：

```
enable_thinking=false  → reasoning_content=''，usage 无 reasoning_tokens，content 正常
不传该参数（默认）      → reasoning_tokens=37，reasoning_content 有内容
两次 model 字段均回显 qwen3.8-27b
```

即 **DashScope 真正遵守 thinking 关闭**，不是仅在文档上声称。这满足 §6 判据第 1 条，且它是 Qwen 第一方。

⚠ **但 `system_fingerprint` 返回 `None`**——DashScope 不提供指纹。现有 harness 逐条落盘 `api_model` + fingerprint 作为模型出处凭据，该臂只能靠 `model` 字段与 response id 佐证。**addendum 必须显式记载这一证据降级**，不能假装与 SiliconFlow 臂同等强度。

**结论：provider 判据已闭合 → 首选 DashScope 中国站 `qwen3.8-27b` + `enable_thinking:false`，不需要新开 OpenRouter/Cloudflare 账户。** 以下 OpenRouter 价格表保留作备选与价格参照。

### 4.1 SiliconFlow / OpenRouter（备选）

**SiliconFlow 未上架 3.8-27B**（有 3.8-2.4T-A95B，但那是另一个模型，不能替代）。

OpenRouter 现有 11 家供应商（USD / 1M tokens）：

| 供应商 | 输入 | 输出 | 缓存读 |
|---|---:|---:|---:|
| Chutes | $0.35 | $2.75 | $0.035 |
| AkashML | $0.40 | $2.55 | $0.05 |
| CoreWeave | $0.40 | $3.00 | $0.15 |
| Phala | $0.40 | $3.00 | $0.15 |
| Novita | $0.42 | $3.00 | $0.085 |
| **Alibaba Cloud Int.** | $0.425 | $2.55 | $0.085（另有 5m cache create $0.5313） |
| Cloudflare | $0.45 | $3.20 | $0.05 |
| Reka AI / Venice / Parasail | $0.45 | $3.20 | $0.05 |
| io.net | $0.48 | $3.40 | $0.25 |

⚠ OpenRouter 页面**不按供应商列出 context 长度与 thinking 开关支持**。模型级说明只写"thinking 可开可关"，未说明哪些供应商真正遵守该开关——**这是协议风险，不是价格问题**（见 §6）。

## 5. pre-test 重跑报价（§0bis 第 4 条）

用量按 Qwen3.6 臂的实测 token 作代理（CPCD 提示词均长 22,324 tok 是主成本，ESConv 均长仅 546 tok）：

- 生成侧：CPCD 159 + ESConv fixed-250 = **409 条**，prompt 3,685,982 tok / completion 61,760 tok（**思考关闭**前提）
- 判官侧：CPCD 159 条盲评，prompt 3,535,765 tok / completion 40,538 tok（deepseek-v4-flash）

| 供应商 | 生成 | 判官 | **合计** | 若思考开启（输出×3 粗估） |
|---|---:|---:|---:|---:|
| Chutes | $1.46 | $1.01 | **$2.47** | ~$2.81 |
| AkashML | $1.63 | $1.01 | **$2.64** | ~$2.95 |
| Alibaba Cloud Int. | $1.72 | $1.01 | **$2.73** | ~$3.05 |
| Cloudflare | $1.86 | $1.01 | **$2.86** | ~$3.26 |
| io.net | $1.98 | $1.01 | **$2.99** | ~$3.41 |

**结论：全部供应商都在 $3 上下，价格不构成选择依据。** 判官成本按 DeepSeek V4-Flash $0.28/$0.42 每 M 估算，此单价为占位值，下单前需按账单页核实（对总额影响 <$0.5）。

## 6. 选 provider 的真实判据（价格之外）

按重要性排序，建议 addendum 里同时冻结"首选 + 备选"并记录实际命中：

1. **是否真正遵守 thinking/reasoning 关闭**——3.6 臂是 `enable_thinking:false`。若某供应商忽略该参数，3.8 臂就带着思考链参与配对比较，pre/post 链直接失效。
2. **是否返回稳定的 `response_model` 与 fingerprint**——现有 harness 逐条落盘 `api_model` + fingerprint，这是模型出处的唯一凭据。
3. 限流与 240s 超时下的稳定性（CPCD 单条 22K 输入）。

**建议首选 Alibaba Cloud International**（Qwen 第一方，最可能正确实现 `reasoning_effort`），**备选 Cloudflare Workers AI**（有独立文档化端点，便于出问题时切换）。**冻结前必须先做 3–5 条 smoke**，逐条确认返回体里没有思考内容、`reasoning` 字段为空、`response_model` 稳定——smoke 通过再冻结 addendum，不通过就换下一家。

## 7. 免费渠道调查（2026-08-28）

用户问能否零成本跑。逐条查证：

| 渠道 | 免费额度 | 对本任务是否够 | 判定 |
|---|---|---|---|
| **DashScope 新用户免费额度** | 每个模型独立 **100 万 tokens**（不跨模型合并；部分模型仅华北2有额度；用完自动转按量付费，可开"用完即停"） | 需 369 万 input → **覆盖约 27%** | 🟡 部分可用，见下方拆分 |
| OpenRouter `:free` | 20 req/min；**50 req/日**（充值过 $10 则 1000/日） | 需 409 条 → 至少 9 天 | ❌ 且**当前无 Qwen 免费端点**（Qwen 免费层已下架） |
| Cloudflare Workers AI | 10,000 neurons/日 ≈ **15–25 次文本生成** | 需 409 条，且 22K 长提示词消耗远高于均值 | ❌ 要几十天 |

⚠ **免费端点还有一条非成本的排除理由**：多家免费层的实际对价是"用你的 prompt 做训练"。本项目的 CPCD 提示词是心理咨询对话内容，AugESC/ESConv 语料为 CC-BY-NC，**把它们送进会训练的免费端点是权利与隐私问题，不是省钱问题**。DashScope 按量付费通道不在此列。

### 7.1 关键拆分：两个 benchmark 的成本差 26 倍

|  | 请求数 | input tokens | output tokens | 需要判官？ | 成本 |
|---|---:|---:|---:|:--:|---|
| **ESConv fixed-250** | 250 | **136,393** | 11,976 | ❌ 否（八类精确匹配，非判官打分） | **完全落在 100 万免费额度内** |
| CPCD | 159 | **3,549,589** | 49,784 | ✅ 是（另需 353 万 input 判官调用） | 超免费额度 3.5 倍 |

**ESConv 臂可以零成本跑，且不需要判官、不消耗 DeepSeek 余额。** 它交付的正是部署门第 2 条（八类 ACC ≥ 33.61%）的 3.8 基线——本次微调的主要目标指标。

CPCD 臂是花钱的那一半：生成 355 万 input + 判官 353 万 input。判官走 DeepSeek 现有 ¥4.71 余额，**够不够需按 V4-Flash 实际单价核算后再决定**。

### 7.2 建议执行顺序

1. 先冻结 addendum v2（provider 判据已闭合，见 §4.0）
2. 跑 **ESConv 250 条**（免费额度内，无判官）→ 拿到 3.8 的八类 ACC 基线
3. CPCD 臂单独决策：确认 DeepSeek 余额够判官后再跑，或推迟到微调完成后与 post-test 同批跑（同 provider、同日、同判官指纹，反而消除时间漂移风险）
