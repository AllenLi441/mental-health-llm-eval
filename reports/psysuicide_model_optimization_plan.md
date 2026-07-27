# PsySUICIDE 模型优化实施计划

状态日期：2026-07-27
当前状态：离线框架已实现并通过；付费验证集实验尚未开始。

## 目标

在不修改官方标签、不删除难例、不查看 test 结果调参的前提下，回答三个独立问题：

1. 从 V4-Flash 换成 V4-Pro，本身能提高多少？
2. 在同一个 V4-Pro 上加入类别定义，能提高多少？
3. 在类别定义上再加入层级判别，能否减少主要混淆？

## 固定数据与指标

- 开发数据：PsySUICIDE 官方 `valid`，保留 1,459 条单标签样本，全量运行。
- 隐藏确认数据：官方 `test`，保留 1,464 条单标签样本。
- `train` 只允许用于后续检索示例构建，不产生调参排行榜。
- 主指标：macro-F1，防止多数类掩盖稀有高风险类别。
- 次指标：weighted-F1、accuracy、invalid rate。
- 诊断：逐类 F1、完整混淆矩阵，重点检查被动/主动/探索、计划/准备/未遂、
  自伤意图/行为和攻击主体。
- 所有臂使用相同保留 ID、seed=42、thinking=enabled、reasoning_effort=high。

## 第一轮验证矩阵

| 臂 | 模型 | prompt profile | 作用 |
|---|---|---|---|
| A | deepseek-v4-flash | baseline | 与旧 reasoner 路由最接近的显式控制 |
| B | deepseek-v4-pro | baseline | A→B 隔离“只换模型”的贡献 |
| C | deepseek-v4-pro | taxonomy | B→C 隔离 11 类工作定义的贡献 |
| D | deepseek-v4-pro | hierarchical | B→D 隔离层级判别的贡献 |

先为每个臂运行固定 seed 的 50 条 route smoke，只检查请求模型、response model、错误、
invalid、usage 和成本是否正常；不据此删题或改变 gold。route smoke 通过后运行四个完整 valid
臂。

## 选择规则

1. 以完整 valid 的 macro-F1 最高者为唯一候选。
2. macro-F1 相同到小数点后三位时，依次用 weighted-F1、更低 invalid、较低平均费用决胜。
3. 报告 A→B、B→C、B→D 三个预先指定的配对对比。
4. accuracy 使用逐题正确/错误的 exact McNemar；macro-F1 和 weighted-F1 使用同一批题的
   paired bootstrap 置信区间。多重比较使用 Holm 校正。
5. 验证结果允许表述“在 valid 上更高/更低”；只有冻结后的一次完整 test 和预注册配对检验
   才能支持确认性结论。

## Test 冻结门禁

runner 已强制执行：

- scored run 必须显式给出 `--split`；
- `train` 不能作为 scored optimization split；
- test 必须全量、非默认 run ID、`--confirm-test`；
- test 参数必须与预注册 JSON 中的 model、profile、thinking、seed、sample、prompt hash 和
  dataset hash 完全一致；
- 预注册 JSON 必须位于本仓库、已被 git 跟踪且内容与 HEAD 一致；
- resume 必须匹配 split、profile、model、run ID、seed 和两个哈希；
- 退役的 `deepseek-chat` / `deepseek-reasoner` 别名会在请求前被拒绝。

## 调用量与预算边界

四个完整 valid 臂共 5,836 次请求。当前提示字符量：

| profile | 次数 | 输入字符总数 | 平均字符 |
|---|---:|---:|---:|
| baseline | 1,459 | 254,842 | 175 |
| taxonomy | 1,459 | 860,327 | 590 |
| hierarchical | 1,459 | 1,257,175 | 862 |

按 2026-07-27 DeepSeek 官方价格，V4-Pro cache-miss 输入为 $0.435/百万 token、输出
$0.87/百万 token；V4-Flash 分别为 $0.14 和 $0.28。若把每次 thinking 输出都按 runner 的
2,048 token 上限计，并粗略把中文字符按一 token 估计，四个完整臂约为 **$9.71 的保守单次
上界**；实际通常更低，但价格、分词、缓存命中、重试和模型实际 reasoning 用量都会改变费用。
运行前重新核对官方[价格页](https://api-docs.deepseek.com/quick_start/pricing)，建议专用评测
key 的账户预算至少留出 $15，并在 50 条 smoke 后根据真实 `usage` 再决定是否继续全量。

## 当前阻塞条件

- 当前进程没有 `EVAL_API_KEY`，仓库也没有 `.env`。
- 因此目前没有产生任何新的模型预测、分数或“提高了”的结论。
- 下一步需要用户在本机配置专用、可撤销、有限额的评测 key，并确认本轮预算；密钥不得粘贴
  到聊天、代码、提交或报告中。
