# 微调前基线结算（pre-finetune baseline）

> **2026-09-03 口径更新：** 本报告保留 2026-08-28 当时四臂结算；后续已完成的
> Qwen3.8-27B DashScope fixed-250 结果（ACC 13.60%，0 missing/invalid）见
> `reports/esconv_fixed250_qwen38_pretest_20260828.md`。ESConv fixed-250 现仅作
> diagnostic：四/五臂差异均无显著证据，且 32.22% / 33.61% 来自 2,775-item 监督分类
> 的另一协议，不能再据此定义 QLoRA 的能力缺口或部署门。当前执行口径见根目录
> `HANDOFF-qwen-therapy-model-20260903.md`。

生成时间：2026-08-28　生成脚本：`scripts/aggregate_pretest_baseline.py`

> **口径警告**：CPCD 分数来自内部 proxy judge（deepseek-v4-flash，非思考，盲评），
> 不是论文可比的公开榜分。只允许在本项目 pre/post 同 judge、同 rubric 的配对比较中使用。

## 1. CPCD 开放回复（proxy judge 1–5 分）

judge 记录：`outputs/cpcd_proxy_flash_judgements_20260809.jsonl`（sha256 `c1e707818ba6f16b…`，610 条）

三个 family 各用各的 rubric，维度不可跨 family 比较；`micro` = 159 条直接平均，
`macro` = 三个 family 均值再平均（消除 srg 占 62% 的权重偏斜）。

| 臂 | 判分数 | micro 均分 | macro 均分 | srg | mr | tcr |
|---|---:|---:|---:|---:|---:|---:|
| `deepseek_v4_pro` | 159 | **4.281** | 4.417 | 4.057 | 4.756 | 4.438 |
| `qwen_3_6_27b_siliconflow` | 159 | **4.198** | 4.315 | 3.990 | 4.669 | 4.287 |
| `deepseek_v4_flash` | 156 | **4.155** | 4.086 | 4.061 | 4.699 | 3.500 |
| `qwen_3_6_27b` | 136 | **4.118** | 4.314 | 3.943 | 4.655 | 4.344 |

### 分维度（各 family 内）

**单轮回复生成 (srg)**

| 臂 | coherence | empathy | professionalism |
|---|---|---|---|
| `deepseek_v4_flash` | 4.051 | 4.091 | 4.040 |
| `deepseek_v4_pro` | 4.030 | 4.091 | 4.051 |
| `qwen_3_6_27b` | 3.909 | 3.970 | 3.949 |
| `qwen_3_6_27b_siliconflow` | 3.970 | 4.010 | 3.990 |

**记忆回溯 (mr)**

| 臂 | accuracy | completeness | no_hallucination | temporal_consistency |
|---|---|---|---|---|
| `deepseek_v4_flash` | 4.692 | 4.205 | 4.974 | 4.923 |
| `deepseek_v4_pro` | 4.775 | 4.325 | 5.000 | 4.925 |
| `qwen_3_6_27b` | 4.621 | 4.241 | 5.000 | 4.759 |
| `qwen_3_6_27b_siliconflow` | 4.625 | 4.350 | 4.975 | 4.725 |

**时序因果回溯 (tcr)**

| 臂 | causal_coherence | completeness | no_hallucination | temporal_accuracy |
|---|---|---|---|---|
| `deepseek_v4_flash` | 3.222 | 3.111 | 4.444 | 3.222 |
| `deepseek_v4_pro` | 4.350 | 4.200 | 5.000 | 4.200 |
| `qwen_3_6_27b` | 4.125 | 4.125 | 5.000 | 4.125 |
| `qwen_3_6_27b_siliconflow` | 4.050 | 4.050 | 5.000 | 4.050 |

### 配对比较（同 task_id 交集，10,000 次 bootstrap，seed 20260828）

| 比较 (A−B) | 配对数 | 平均差 | 95% CI | A 胜 | B 胜 | 平 |
|---|---:|---:|---:|---:|---:|---:|
| qwen_3_6_27b_siliconflow - deepseek_v4_pro | 159 | -0.083 | [-0.176, 0.010] | 20 | 33 | 106 |
| qwen_3_6_27b - deepseek_v4_pro | 136 | -0.120 | [-0.225, -0.013] | 15 | 33 | 88 |
| deepseek_v4_pro - deepseek_v4_flash | 156 | 0.121 | [-0.006, 0.253] | 43 | 28 | 85 |
| qwen_3_6_27b_siliconflow - qwen_3_6_27b | 136 | 0.055 | [-0.005, 0.113] | 19 | 7 | 110 |

### 生成侧完成状态

| 臂 | 生成条数 | 模型 | stop | length(截断) | 截断率 |
|---|---:|---|---:|---:|---:|
| `deepseek_v4_flash` | 156 | deepseek-v4-flash | 123 | 33 | 21.15% |
| `deepseek_v4_pro` | 159 | deepseek-v4-pro | 147 | 12 | 7.55% |
| `qwen_3_6_27b` | 136 | qwen3.6-27b | 136 | 0 | 0.00% |
| `qwen_3_6_27b_siliconflow` | 159 | Qwen/Qwen3.6-27B | 159 | 0 | 0.00% |

## 2. ESConv fixed-250 八类策略（沿用已结算报告）

来源：`reports/esconv_fixed250_deepseek_v4_pro_0813_20260813.json`（sha256 `fa445dc23c5d11a9…`）

| 臂 | Accuracy | Cluster-bootstrap 95% CI | Macro-F1 | Weighted-F1 | missing/invalid | 源文件哈希一致 |
|---|---:|---:|---:|---:|---:|:--:|
| DeepSeek V4-Flash | 16.40% | [12.13%, 21.12%] | 12.26% | 14.85% | 3/4 | ✅ |
| DeepSeek V4-Pro-0813 | 16.00% | [12.59%, 20.00%] | 9.64% | 13.66% | 10/12 | ✅ |
| DeepSeek V4-Pro（旧测） | 14.80% | [10.31%, 19.92%] | 9.48% | 13.08% | 11/17 | ✅ |
| Qwen3.6-27B（未微调） | 14.00% | [9.62%, 19.20%] | 10.44% | 11.80% | 0/0 | ✅ |

## 3. 读数要点

1. **CPCD 上未微调 Qwen 已与 DeepSeek 基本持平。** V4-Pro 4.281 vs Qwen3.6-27B(SF) 4.198，配对差 -0.083，95% CI [-0.176, 0.010]——**跨 0，差异不显著**；159 对中 106 对完全同分。

2. **ESConv 八类策略上，所有 API 臂都远低于专用小模型。** 四臂区间 14.00%–16.40%，而同一 frozen 协议下 BlenderBot-small Joint 为 32.22%、EmoDynamiX 为 33.61%。**这才是微调要补的缺口**——不是补 Qwen 与 DeepSeek 之间的差距。

3. **截断率差异会污染跨臂比较。** V4-Flash 截断 33/156（21.15%），Qwen 两臂均为 0%。Flash 的 tcr 低分很可能是截断产物而非能力差距，勿据此下能力结论。

## 4. 必须随分数一起披露的口径事项

| 项 | 内容 |
|---|---|
| `deepseek_v4_flash` 的 judge 指纹 | `deepseek-v4-flash` / `proxy_non_thinking` / `fp_a18b46594c_prod0820_fp8_kvcache_20260402` |
| `deepseek_v4_pro` 的 judge 指纹 | `deepseek-v4-flash` / `proxy_non_thinking` / `fp_a18b46594c_prod0820_fp8_kvcache_20260402` |
| `qwen_3_6_27b` 的 judge 指纹 | `deepseek-v4-flash` / `proxy_non_thinking` / `fp_a18b46594c_prod0820_fp8_kvcache_20260402` |
| `qwen_3_6_27b_siliconflow` 的 judge 指纹 | `deepseek-v4-flash` / `proxy_non_thinking` / `a26a7955944dc5c60445bff77fac9c8e` |
| ⚠ 指纹不一致 | SiliconFlow Qwen 臂（8/13 判分）的 judge fingerprint 与其余三臂（8/09 判分）不同一格式。post-test 必须记录当次指纹并在报告中并列披露，不得假装同一次判分。 |
| 各臂 n 不等 | 判分数 159/159/156/136 不等，跨臂只能看配对交集列，不得直接比较各自的 micro 均分名次。 |
| 分数性质 | 内部 proxy judge 分，**不可**写进论文或对外榜单。 |
