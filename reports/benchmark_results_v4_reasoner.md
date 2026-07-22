# v4-pro 终端评测与既有 reasoner 结果（审计版）

> 2026-07-21 重算。本文件替代旧版“5 个硬胜项”等已失效结论。IMHI 的统一去偏结果以 `imhi_uniform_v3u.md` 为准；本文件只报告已核验的 v4-pro 终端运行和模型路由边界。

## v4-pro 终端运行

| 任务 | 唯一样本 | 正确 | accuracy | 分项 | 数据状态 |
|---|---:|---:|---:|---|---|
| CPsyExam pilot（非确认性） | 599 | 518 | 86.48% | KG-SCQ 322/347；KG-MAQ 89/124；CA-SCQ 90/102；CA-MAQ 17/26 | 原始 600 行含 1 个完全重复 ID；聚合报告使用 canonical 去重视图 |
| EmoBench-EA 官方 prompt/scoring proxy | 400 | 297 | 74.25% | en 75.5%；zh 73.0% | 中英各 200，完整；temp-0 单次调用 |
| EmoBench-EU 官方 prompt/scoring proxy | 400 | 286 | 71.50% | en 74.0%；zh 69.0% | 中英各 200；temp-0 单次调用；旧 summary 仅中文半跑，已由原始 400 行重建 |

三组记录的 `api_model` 均为 `deepseek-v4-pro`，fingerprint 均为 `fp_9954b31ca7_prod0820_fp8_kvcache_20260402`。本公开仓库只发布聚合 pilot 证据 `results-summary/v4-pro-pilot-terminal.audit.json`，逐行结果受发布边界约束。

## 对照口径

- CPsyExam 的 599 条是抽样，不是 3,902 条全量；不能与论文全量结果写成同条件显著胜负。GPT-4 `67.43` 含少样本取优，严格零样本分组加权约为 57.6%。
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

## 尚未完成的外部证据

- CPsyExam v4-pro 全量 3,902 条尚未运行；论文主表必须写 `n=599 抽样`，或补全量。
- 人类金标尚未产生；模型裁判与公开 benchmark 不能替代真人安全评审。
- API 余额、历史 key 轮换和过去线上 smoke 不能由聚合文件反向证明。
