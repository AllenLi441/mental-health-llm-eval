# 2026-08-01：Benchmark freeze 与 MentalHealth-Instruct v1

## 为什么更新

用户决定当前不以服务器或继续训练为优先，而是先冻结 benchmark/baseline，建立真正可形成方法贡献的训练数据结构、真人标注流程与质量控制。执行期间不重启 IMHI，不恢复 PsySUICIDE v2 B/C/D，不访问新的 benchmark test，也不由 AI 生成 human gold。

## 已完成

### Benchmark suite v1

- 冻结 8 个 benchmark family、19 个 task contract。
- 绑定 source commit `702f0b7c1fed44d94de911786d4d6edaa2e6d9ef`、registry/schema/index/bundle SHA-256 与每个 task spec SHA-256。
- 明确 freeze 只发布协议、hash 与聚合边界，不发布许可原文、逐行 ID/gold/prediction、raw output、凭据或模型权重。

### DeepSeek historical baseline v1

- 19 条 task 记录全部绑定 reviewed public artifact 与 SHA-256。
- 精确区分 exact V4/fingerprint、response exact but requested legacy alias、以及 unresolved `deepseek-chat` 三种 identity。
- CPsyExam/PsySUICIDE 使用冻结 reference arm；EmoBench 明示 temp-0 proxy；IMHI 明示历史 sampled-test reported-best；MDD/EATD/MentalManip/CBT 明示 legacy identity/protocol 缺口。
- 顶层固定 `paper_control_ready=false`，禁止把混合协议求一个跨 benchmark 平均分。

### MentalHealth-Instruct v1

- 建立 final record JSON Schema，覆盖 knowledge、emotion、risk、condition、CBT、manipulation 与 supportive-response 七类任务。
- `condition_signal` 只允许作为数据集分类信号，不得冒充临床诊断。
- final row 强制 `model_generated_gold=false`，gold 只能来自真人共识或真人裁决。
- 建立来源许可/PII/benchmark overlap/group split 政策。
- 建立 author、两个独立盲审 reviewer、expert adjudicator 的角色隔离。
- 建立 `pilot-0001` 空白卡：`UNLABELED`、无 instruction/input/target、不可 export。
- 仓库外私有 workspace 已初始化：`raw_data/`、annotation/review/manifests 目录、0-byte `train.jsonl` 与 `validation.jsonl`；manifest 为 `EMPTY_WORKSPACE`、`training_allowed=false`、0 human review。

## IMHI 状态更正

IMHI 12-job screen 已实际启动。首个 DR/CE job 训练到 epoch 4 / step 504，early stopping 后加载 best checkpoint-252 时，严格 state-key 校验因 Transformers 5 LayerNorm `gamma/beta` 与 live model `weight/bias` 命名转换失败。共同参数 shape/dtype 一致，属于 harness 兼容问题，不是数据或指标失败。

- manifest：0 `COMPLETE`、1 `FAILED`、11 `PLANNED`。
- checkpoint-252 诊断值：valid accuracy `0.9093`、weighted-F1 `0.9085`。
- 没有 `result.json` 或 `best-model`，所以诊断值不得进入正式 screen 排名。
- manifest 顶层 `PROTOCOL_PREFLIGHT` 分类不准确；真实阶段是 post-training best-load。
- 当前按用户新优先级暂停，不自动修复后重启。

## 验证

| 检查 | 结果 |
|---|---|
| `python3 scripts/check_research_freeze.py --selftest` | `PASS`：8 families、19 tasks、19 baseline records、0 public rows、unlabeled pilot |
| `python3 -O scripts/check_research_freeze.py --selftest` | `PASS`：同上 |
| workspace initializer normal / `python -O` selftest | `PASS`：dry-run、external-only、empty JSONL、no overwrite |
| 真实 workspace dry-run -> explicit execute | `PASS`：0 train / 0 validation，`training_allowed=false` |
| 全仓离线回归 | `PASS`：19/19 prompt/parser、aggregate audit、scoreboard、baseline、safety、CPsyExam、PsySUICIDE v1/v2、EmoBench、generic trainer、IMHI screen/v4 selftests 全部退出 0 |
| diff / JSON / privacy scan | `PASS`：无 whitespace error、绝对 licensed-data path、credential value、raw row、human gold、checkpoint/weight 或 row-level output；既有 IMHI runtime artifacts 保持 ignored |
| PsySUICIDE v2 B/C/D | `PAUSED`：无恢复、无选择 |
| 新训练/API/test access | `NOT RUN` |

## 下一步

1. 让真实数据负责人在空白 `pilot-0001` 上只填写来源、task family、language、instruction 与 input。
2. 由两名真实 reviewer 独立盲审，再由 expert adjudicator 处理分歧和高风险项。
3. pilot 未通过前不扩到 1k-5k，不选择 Qwen/LoRA 超参，不找服务器。
4. 另行修复 ERR-20260801-16，但只有在 schema/pilot 路线不被打断时才重新启动 IMHI。
