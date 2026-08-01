# CPsyExam V4 全量确认性配对评测预注册

状态：**在本协议提交并公开后才允许发出新的确认性模型请求**。登记日期为 2026-07-22。研究者已经看过早期 `n=599` 的 v4-pro pilot，因此这不是结果盲的首次预注册；pilot 不进入确认性分析。

## 研究问题与固定配置

- 数据：CPsyExam released answered test split 的全部 3,902 个行实例。
- 主模型：`deepseek-v4-pro`，`thinking=enabled`，`reasoning_effort=high`。
- 单一正式对照：`deepseek-v4-flash`，`thinking=disabled`。
- 两臂使用相同题目顺序、system/user prompt、严格解析器、零样本设置与官方 DeepSeek Chat Completions endpoint。
- v4-pro `max_tokens=2048`；v4-flash `max_tokens=8`。两臂均不使用抽样，`seed=42` 仅作为运行元数据。
- temperature 对非思考对照固定为 0；DeepSeek 文档说明 thinking 模式忽略 temperature，因此主模型不发送该字段。
- 每臂并发上限 8。传输层对 429/5xx 最多重试 3 次；401/402 立即中止且不得把该题写成“已完成”。

## 纳入、身份与失败门槛

1. 当前 loader 必须得到恰好 3,902 行和 3,902 个规范化唯一 ID。
2. 两臂必须各有 3,902 行，ID 集合完全一致，且每行 gold 与授权数据一致。
3. 每行必须记录 requested/response model、provider、fingerprint、attempt、UTC 时间、usage、case digest、prompt hash、dataset manifest hash、run ID 与配置。
4. `response_model` 必须分别严格等于 `deepseek-v4-pro` 和 `deepseek-v4-flash`，fingerprint 覆盖必须为 100%。若一臂出现多个 fingerprint，将如实分 cohort 披露并把该臂标成 version-mixed。
5. invalid 计为错误。任何最终 API error 都使确认性推断状态变成 `NOT_EVALUABLE`；不得剔除 error 后继续宣称显著或等价。需要重跑时必须新建完整 run，不覆盖或选择性补写旧 run。
6. 早期 599 pilot、历史 `deepseek-chat`、Kimi 与论文 GPT-4 聚合点值只可作背景或敏感性描述，不进入本次主检验。

## 主估计量与检验

- 每题结果为二元正确性；主估计量 `delta = accuracy(v4-pro) - accuracy(v4-flash)`。
- 记 `b` 为仅 v4-pro 正确、`c` 为仅 v4-flash 正确。差异检验使用双侧 exact McNemar：在 `b+c` 个 discordant pairs 上按 `Binomial(0.5)` 计算，`alpha=0.05`。
- 同时报 delta 的 paired normal 95% CI 和固定 `seed=20260722`、`B=20,000` 的 paired nonparametric bootstrap 95% CI。
- 只有 exact McNemar `p<0.05` 且 paired normal 95% CI 与 bootstrap 95% CI 均完全大于 0，才表述“v4-pro 在本协议下显著优于 v4-flash”。完全小于 0 时才允许反向表述。
- `p>=0.05` 只写“未检出差异”；不能写“统计平手”。

## 预注册等价性规则

- 实质等价界固定为 `[-0.02, +0.02]`（accuracy 的 ±2 percentage points）。
- 使用逐题差值 `d_i in {-1,0,1}` 的 paired large-sample TOST，`alpha=0.05`；等价判定等同于 paired normal 90% CI 完全落在预设界内，并要求两个 one-sided p 值都小于 0.05。
- 只有满足上述条件且两臂均为 0 API error，才可写“在 ±2pp 界内等价”。否则写“未证明等价”，不能写“平手”。

## 次要与敏感性分析

- KG-SCQ、KG-MAQ、CA-SCQ、CA-MAQ 只报告行数与点估计，不作确认性显著性声明。
- 历史旧 ID 结果若展示，必须明确标为非确认性敏感性分析；不得替代 3,902 对 3,902 主分析。
- 不增加第二个正式对照。若未来新增确认性比较，必须另行预注册并处理多重比较，不能事后塞入本协议。

## 公开产物与隐私/许可边界

CPsyExam 数据卡许可按 CC BY 4.0 处理并附来源署名。公开 Release 不包含题目、选项、gold、prediction、raw output、原始 ID、API key 或可识别 key 指纹。逐行公开文件只包含：

- 对 canonical case（含相对来源位置、行内序号与完整授权行内容）计算的 SHA-256 commitment；
- 四个题型分组；
- 两臂的 `ok`、`invalid`、error-presence、requested/response model 与 fingerprint；
- 足以重算配对统计、但不能直接还原题目和答案的 provenance。

Release 还必须包含分析 JSON、数据/代码/私有原始结果哈希、许可 NOTICE 与 SHA256SUMS。GitHub release immutability 必须在发布前开启；工作流为 draft → 上传全部 assets → publish → `gh release verify` 与逐资产 `gh release verify-asset`。

本评测只回答该 benchmark/协议下的模型差异，不构成临床有效性、安全性或医疗建议。
