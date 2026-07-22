# v4-pro 终端评测与既有 reasoner 结果（审计版）

> 2026-07-22 更新。本文件替代旧版“5 个硬胜项”等已失效结论。IMHI 的统一去偏结果以 `imhi_uniform_v3u.md` 为准；本文件报告已核验的 v4-pro 终端运行、CPsyExam V4 全量配对结果和模型路由边界。

## CPsyExam V4 全量确认性配对

预注册已先行冻结于 [`cpsyexam-v4-prereg-2026-07-22`](https://github.com/AllenLi441/mental-health-llm-eval/releases/tag/cpsyexam-v4-prereg-2026-07-22)。随后对同一批 3,902 题运行两臂：

| 臂 | n | 正确 | accuracy | invalid | API error | response fingerprint |
|---|---:|---:|---:|---:|---:|---|
| deepseek-v4-pro（thinking enabled，reasoning high） | 3,902 | 3,307 | **84.7514%** | 17 | 0 | `fp_9954b31ca7_prod0820_fp8_kvcache_20260402` |
| deepseek-v4-flash（thinking disabled） | 3,902 | 3,252 | **83.3419%** | 0 | 0 | `fp_8b330d02d0_prod0820_fp8_kvcache_20260402` |

配对列联为：两者都对 3,088、只有 v4-pro 对 219、只有 v4-flash 对 164、两者都错 431。v4-pro 相对 v4-flash 的差值为 **+1.4095pp**；配对 normal 95% CI `[0.4274, 2.3917]pp`，20,000 次配对 bootstrap 95% CI `[0.4357, 2.4090]pp`。预注册 exact McNemar 双侧 `p=0.00572255`，故本协议内可表述为“v4-pro 显著优于 v4-flash”。

±2pp TOST 的 `p_lower=5.086e-12`、`p_upper=0.119332`，等价性的两个单侧检验未同时通过；结论是 **未证明等价**，不能写“统计平手”或“等价”。

公开聚合证据见 `results-summary/cpsyexam-v4-full-paired.summary.json`。去敏资产已发布到 [`cpsyexam-v4-full-2026-07-22`](https://github.com/AllenLi441/mental-health-llm-eval/releases/tag/cpsyexam-v4-full-2026-07-22)；GitHub 报告 `isImmutable=true`，Release attestation 与 6/6 个资产摘要均已验证。runner commit 为 `3e6890374cb39631bb1cc8bca46ef4835df85446`，公开 case commitment manifest SHA-256 为 `1275ce7edeb55ad62500ac1692b82bef3800592decc2ece4d615fc8770232c9d`。

## 历史 v4-pro 终端运行

| 任务 | 唯一样本 | 正确 | accuracy | 分项 | 数据状态 |
|---|---:|---:|---:|---|---|
| CPsyExam pilot（非确认性，已被全量确认性结果取代） | 599 | 518 | 86.48% | KG-SCQ 322/347；KG-MAQ 89/124；CA-SCQ 90/102；CA-MAQ 17/26 | 原始 600 行含 1 个完全重复 ID；仅保留作历史审计 |
| EmoBench-EA 官方 prompt/scoring proxy | 400 | 297 | 74.25% | en 75.5%；zh 73.0% | 中英各 200，完整；temp-0 单次调用 |
| EmoBench-EU 官方 prompt/scoring proxy | 400 | 286 | 71.50% | en 74.0%；zh 69.0% | 中英各 200；temp-0 单次调用；旧 summary 仅中文半跑，已由原始 400 行重建 |

三组历史记录的 `api_model` 均为 `deepseek-v4-pro`，fingerprint 均为 `fp_9954b31ca7_prod0820_fp8_kvcache_20260402`。pilot 聚合证据 `results-summary/v4-pro-pilot-terminal.audit.json` 已显式标为 historical，并指向全量结果；其逐行结果仍受发布边界约束。

## 对照口径

- CPsyExam V4 的 3,902 条确认性结果只支持同题、同 harness 的 v4-pro 与 v4-flash 配对结论。旧 599 条抽样 pilot 不进入该检验。与论文 GPT-4 的比较仍是跨设置描述性比较：`67.43` 含少样本取优，严格零样本分组加权约为 57.6%，没有论文对照的逐行预测，不能把该点值差另写成配对显著胜负。
- EmoBench-EA/EU 各覆盖官方 400 条数据并复用官方 prompt/单次评分，但没有复现论文的 5 次采样多数票 × 4 个选项排列聚合，因此不是完全同协议。GPT-4 论文中英均值约为 74.6% / 56.9%（由论文分语言数字计算）；这里只能报告描述性点值差，benchmark 结果也不等于临床有效性。
- 旧版 reasoner 的选择性 prompt 结论已废止，不再把 IMHI-Dreaddit/DR 写成硬胜项。统一协议只引用 `imhi_uniform_v3u.md`。

## 产品与评测边界

产品 `pace=deep` 的回复生成路由到 v4-pro；隐性风险判官的 DeepSeek backup 仍为 v4-flash。因此“深度回复使用 v4-pro”与“风险检测 deep 臂使用 v4-pro”不是同一个主张。

## 复现边界

```bash
node run.mjs all --selftest
python3 scripts/audit_results.py --selftest
python3 scripts/scoreboard.py --selftest
```

以上公开命令验证代码入口、聚合 schema 和 baseline 注册表。逐行指标重算需要按许可取得数据和本地 JSONL；公共包不能把缺失的原始行伪装成已复算。

## 仍有限制的外部证据

- CPsyExam V4 全量 3,902×2 已运行并按预注册完成配对分析；结果 Release 已发布、锁定为 immutable，并完成 6/6 资产摘要验证。
- 人类金标尚未产生；模型裁判与公开 benchmark 不能替代真人安全评审。
- API 余额、历史 key 轮换和过去线上 smoke 不能由聚合文件反向证明。
