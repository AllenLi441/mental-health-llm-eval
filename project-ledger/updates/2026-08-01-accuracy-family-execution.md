# 2026-08-01：跨 benchmark 提分执行台账

## 为什么更新

用户要求先建立由 Codex 持续维护的执行记录，再按以下顺序推进：在 Draft PR #5 上加入 benchmark spec registry 与通用分类 trainer；运行 2026-07-31 更新后的 DeepSeek V4-Flash + taxonomy 全量 PsySUICIDE valid；随后启动 IMHI 的 DR、dreaddit、IRF、SAD 四任务单 Seed screen。PsySUICIDE accuracy-first v2 的 B/C/D 明确继续暂停。

本文件是本轮工作的追加式事实台账。每次实现、运行、失败、停止或结论变化，都新增一条记录；不回写或覆盖以前的历史记录，不把计划写成完成，不把单 Seed screen 写成最终提升。

## 修改范围

- 分支：`codex/cross-benchmark-accuracy-family-v1`
- 起始公开 HEAD：`924e7cf235b13a0f66a99a149050b0f24a055236`
- Draft PR：[AllenLi441/mental-health-llm-eval#5](https://github.com/AllenLi441/mental-health-llm-eval/pull/5)
- benchmark：PsySUICIDE valid；IMHI DR、dreaddit、IRF、SAD 的 train/inner-dev
- 可能涉及付费 API：只限经预算门禁批准的 V4-Flash taxonomy valid
- 可能涉及模型训练：只限 IMHI 四任务、每任务独立 head/checkpoint 的单 Seed screen
- 禁止范围：PsySUICIDE v2 B/C/D；任何 official/frozen test 调参；许可原文、逐行 gold/prediction、原始模型输出、凭据、权重和 private diagnostics 入 Git

## 当前状态快照

| 工作项 | 状态 | 当前事实 | 晋级前必须满足 |
|---|---|---|---|
| 持续台账与 eval 定义 | `PASS` | 本文件与 `.claude/evals/cross-benchmark-accuracy-family-v1.md` 已建立；首次 ledger checker 通过 | 实现完成后继续追加实际命令与结果；所有未运行项保持 `NOT RUN` |
| Benchmark spec registry | `PASS_LOCAL` | 19/19 task 与 runner 对齐；17/17 synthetic checks 在 normal/`-O` 均通过；bundle `d1f38dc3…6433b` | 作为 clean、committed execution base 后才可供训练绑定 |
| 通用分类 trainer | `PASS_OFFLINE` | normal/`-O` synthetic selftest 均通过；默认 metadata-only、无 external read | 真实训练仍须经已验证 adapter、冻结 registry/data manifest 和 clean Git 门禁 |
| V4-Flash 0731 + taxonomy valid | `REJECTED_BY_GATE` | smoke 50 通过；full valid `n=1,459` 完成且点值提高，但 invalid、关键类 recall、macro inference 三项失败 | 停止，不访问 test，不用本候选替换 reference |
| IMHI 四任务单 Seed screen | `VALIDATED / NOT STARTED` | 显式 `--validate-data` 生成 12-job `DRY_RUN`；0 个训练 job 启动 | 先形成 clean、committed execution base，再启动冻结的 Seed-42 matrix |
| PsySUICIDE v2 B/C/D | `PAUSED` | A 已完成；B 历史上中断；C/D 未开始 | 本轮禁止恢复、续跑或据 A 选 winner |

## 更新日志

| 时间（America/Los_Angeles） | 更新 | 命令 / 证据 | 结果 | 下一步 |
|---|---|---|---|---|
| 2026-08-01 | 冻结本轮目标、顺序、边界与 eval | `git status --short --branch`；阅读 `project-ledger/` 全部既有文档与模板 | 工作分支确认；历史账本保留；四条工作流均未被误写成完成 | 实现 registry 并记录真实文件、命令与结果 |
| 2026-08-01 | 验证新增台账结构与公开安全 marker | `git diff --check -- project-ledger`；新文件 trailing-whitespace scan；`python3 scripts/check_project_ledger.py` | `PASS`：无 whitespace error；`5 required files, 2 dated update, changelog links, public-safe markers` | 每次物质更新后重复运行并追加结果 |
| 2026-08-01 | 完成 benchmark registry 离线门禁 | `python3 scripts/check_benchmark_registry.py`；normal/`python3 -O` `--selftest`；`--print-hashes` | `PASS`：19/19 task；两种模式均 17/17；bundle SHA-256 `d1f38dc38cd3745c8c84edc12390609e35580f8304322607424e5e16d366433b` | 将 registry hash 绑定到 trainer/screen manifest |
| 2026-08-01 | 完成 generic trainer 与 IMHI adapter 自检 | normal/`python3 -O` 分别运行 `scripts/train_text_classifier.py --selftest` 与 `scripts/imhi_classification_screen.py --selftest` | `PASS`：CE/weighted-CE/focal、metrics、checkpoint/resume、post-only leakage、test-path rejection、固定 4×3 matrix；无 model download | 显式 validate 四个 train/valid pair |
| 2026-08-01 | 修复 trainer 兼容与 leakage fixture | 初始 integration check；修复后重复 normal/`-O` selftest | `FIXED`：移除 Transformers 5 不接受的 `save_safetensors` 参数；cross-split key 从完整输入改为 post-only | 保留为 `ERR-20260801-09/10` 回归 |
| 2026-08-01 | 验证 IMHI 四任务真实数据边界 | `python3 scripts/imhi_classification_screen.py --validate-data --dataset-root <repository-external-root>` | `DRY_RUN`：12 jobs；DR 1002/430、cross=1；Irf 3941/985、cross=0；SAD 5547/616、cross=0；dreaddit 2814/300、cross=1；未训练、未访问 test | 等 clean commit 后再显式 launch |
| 2026-08-01 | 运行 0731 Flash taxonomy smoke | `node run.mjs psysuicide --split valid --prompt-profile taxonomy --sample 50 --run-id flash0731-taxonomy-20260801-smoke50-exact-old-harness` | `PASS`：50/50，accuracy `0.92`，invalid `0`，errors `0`；model/fingerprint 一致 | 运行 full valid |
| 2026-08-01 | 运行 0731 Flash taxonomy full valid 并配对复核 | 同一命令去掉 bounded sample，run ID `flash0731-taxonomy-20260801-valid-full-exact-old-harness`；一次性只读 paired recomputation | `COMPLETE / REJECTED`：1,459/1,459，点值提升，但 invalid、关键类 recall 与 macro inference 失败；不碰 test | 记录负结论，停止本候选 |
| 2026-08-01 | 实际结果回写后的最终文档复验 | `python3 scripts/check_project_ledger.py`；tracked ledger diff check；新增文件 whitespace 与敏感 marker/absolute-path scan | `PASS`：checker 仍为 `5 required files, 2 dated update, changelog links, public-safe markers`；其余扫描无输出、退出码 0 | 随后每次训练状态变化继续追加，不覆盖本轮失败记录 |
| 2026-08-01 | 终审加固 shared execute registry proof | 伪造 bundle/task/profile/config/protocol/data-plan synthetic regressions；normal/`python3 -O` trainer selftest | `FIXED / PASS`：`execute_training` 独立重载 live registry、重跑 checker/hash，并核对 task/profile/config/protocol/path/rows/split SHA/plan；伪造 proof 被拒绝 | 保留 `ERR-20260801-11` 回归 |
| 2026-08-01 | 终审加固 MPS exact resume | actual MPS RNG save/restore random-tensor roundtrip；缺失/损坏高 step fixture；normal/`python3 -O` trainer selftest | `FIXED / PASS`：严格 `mps_rng_state.pth`；roundtrip tensor 一致；latest checkpoint 按 device 跳过不完整高 step | 保留 `ERR-20260801-12` 回归 |
| 2026-08-01 | 终审加固 failed preflight 与 resume mode | aggregate-only failure manifest fixture；`--resume` without `--execute` fixture；normal/`python3 -O` screen selftest | `FIXED / PASS`：`ATTEMPTED → FAILED/PREFLIGHT`、exit 1、raw message 不持久化；resume misuse exit 1 | 保留 `ERR-20260801-13` 回归；IMHI 仍不启动 |
| 2026-08-01 | 补齐 capability selftest 与 checkpoint epoch 证据 | binary/three-class loss+metrics、imbalanced weights、invalid/empty-class rejection、deterministic-repeat、checkpoint step/epoch synthetic assertions | `FIXED / PASS`：normal/`python3 -O` 均执行全部断言；result 严格交叉核对 checkpoint 名、best step、logged epoch，并持久化 chosen epoch | 保留 `ERR-20260801-14/15` 回归 |
| 2026-08-01 | 完成全仓最终回归与隐私扫描 | README 所列 prompt/parser、audit、scoreboard、baseline、harness safety、CPsyExam、PsySUICIDE v1/v2、EmoBench、IMHI offline selftests；`git diff --check` 与 tracked/new-file scan | `PASS`：全部退出 0；19/19 prompt/parser、ledger checker 与所有专项 selftest 通过；无 secret、授权原文、绝对数据路径、模型输出或权重进入 Git | 仅剩 clean commit/push、remote SHA 核验与 IMHI launch |
| 2026-08-01 | 终审修复写入后的文档复验 | ledger checker；tracked ledger diff check；新增文件 whitespace 与敏感 marker/absolute-path scan | `PASS`：checker 仍为 `5 required files, 2 dated update, changelog links, public-safe markers`；其余扫描无输出、退出码 0 | 等 clean commit/remote SHA 后启动 IMHI，并在下一次状态变化时追加 |

以后每条更新必须记录：代码/协议变化、实际执行命令、退出结果、聚合指标或错误类型、是否通过停止门槛、下一步。失败命令不得删除；重跑必须使用新的 run identity 并另起一行。

## 当前可追溯指标

2026-07-31 更新后的 V4-Flash baseline 已在完整 PsySUICIDE valid `n=1,459` 上完成：accuracy `0.8231665524`、macro-F1 `0.4801830002`、weighted-F1 `0.8365657416`、invalid `1`、API errors `0`。请求模型、响应模型与 API 模型均为 `deepseek-v4-flash`，fingerprint 为 `fp_a18b46594c_prod0820_fp8_kvcache_20260402`，dataset manifest SHA-256 为 `89bb98ae101f176c1e125a60a6d7ab23f7467c22a77698a0d9668c665dfffc59`，Seed 为 `42`。

旧冻结 Flash baseline 在同一 valid 的公开 aggregate 为 accuracy `0.8095`、macro-F1 `0.4837`、weighted-F1 `0.8250`。新 baseline accuracy 点值增加，但 macro-F1 点值略降且出现一个 invalid；因此它不能仅凭 accuracy 自动晋级。0731 taxonomy 必须与上段完整 baseline 使用相同 retained-case manifest 做逐项配对，且不访问 official test。

0731 taxonomy smoke 为 `n=50`、accuracy `0.92`、invalid `0`、errors `0`。full valid 为 `n=1,459`、accuracy `0.8779986292`、macro-F1 `0.5318369277`、weighted-F1 `0.8809387821`、invalid `16`、errors `0`；模型 identity、fingerprint、Seed 与 full-valid manifest 均和 reference 一致。

该 run 的 provenance 进一步绑定为：历史 harness commit `479e5c03d3626bccdd1d0882c161daebc173d7db`，taxonomy prompt SHA-256 `eb2180f8305316f4cc2c07a810d137901fc62ab94e18e68392b236a2b869d1c8`，response fingerprint `fp_a18b46594c_prod0820_fp8_kvcache_20260402`。这些字段只证明本次 valid run identity；official test 保持未访问。

相对同 fingerprint baseline 的配对点值差为 accuracy `+0.054832`、macro-F1 `+0.051654`、weighted-F1 `+0.044373`；accuracy McNemar `p=4.0268e-08`。但 macro-F1 paired randomization `p=0.06360`，bootstrap 95% CI `[-0.00619, 0.10574]` 跨 0；invalid rate 为 `16/1,459 ≈ 1.10%`，超过 `0.5%`；预注册关键类 recall guard 亦失败。因此这不是可靠晋级：candidate 为 `REJECTED_BY_GATE`，不进入 official test。

本次 full-valid 估算成本 `$0.0722557584`；smoke + full 合计约 `$0.074955`。这是 harness token/pricing 估算，不写成 provider 最终账单。

IMHI 仅完成数据验证：DR 使用 1,002 train / 430 valid，并删除 1 条跨 split train overlap；Irf 为 3,941 / 985、跨 split 0；SAD 为 5,547 / 616、跨 split 0；dreaddit 为 2,814 / 300，并删除 1 条跨 split train overlap。12 个 Seed-42 job 均仍为 planned；尚无训练指标。单 Seed 输出未来也只用于筛选，不构成三 Seed 稳定性、test 提升或论文优越性证据。

## 错误与失败用例

| Ledger ID | 公开安全证据 | 风险 | 当前处理 |
|---|---|---|---|
| `ERR-20260801-07` | taxonomy accuracy/macro-F1 点值均提高，但 invalid=`16/1,459`、关键类 recall guard 失败，macro randomization `p=0.06360` 且 bootstrap CI 跨 0 | 只看点值会把候选误写成可靠升级 | `REJECTED_BY_GATE`：不进入 test |
| `ERR-20260801-08` | 单 Seed 或单任务 inner-dev 点值可能高于 control | 把 screen 当最终结果，或在 test 上选择 loss/head，会制造选择偏差 | `OPEN`：单 Seed 只筛选；通过后另行预注册三 Seed confirmation，test 在最终冻结前保持未访问 |
| `ERR-20260801-09` | Transformers 5 对不支持的 `TrainingArguments.save_safetensors` 抛出 `TypeError` | 首个训练 job 会启动失败 | `FIXED_PROSPECTIVE`：移除不兼容参数并保留 checkpoint 完整性校验；normal/`-O` selftest 通过 |
| `ERR-20260801-10` | 合成场景：相同 post 搭配不同 question，完整输入 digest 不相同 | full-input leakage key 会漏掉跨 split 同帖 | `FIXED_PROSPECTIVE`：固定 post-only key；真实 data-validation 检出并从 DR/dreaddit train 各删除 1 条 overlap |
| `ERR-20260801-11` | 合成 forged bundle 由替代 caller 传给 shared execute | 非空 adapter proof 不能替代 live registry 重验证 | `FIXED_PROSPECTIVE`：shared execute 独立 checker/hash + 完整 task/profile/config/protocol/data-plan binding；伪造 proof 被拒绝 |
| `ERR-20260801-12` | MPS checkpoint 缺少或损坏 RNG state；更高 step 不完整 | resume 可能不可精确复现，或选中不可恢复 checkpoint | `FIXED_PROSPECTIVE`：strict MPS RNG artifact/save/restore；实际 tensor roundtrip；device-aware fallback |
| `ERR-20260801-13` | execute preflight 失败或 `--resume` 未配 `--execute` | 无失败留痕或 CLI 误用静默通过 | `FIXED_PROSPECTIVE`：aggregate-only `FAILED/PREFLIGHT` manifest + exit 1；raw message 不持久化；resume misuse exit 1 |
| 既有 `ERR-20260801-04` | PsySUICIDE v2 只有 A 完成，B 未完成，C/D 未开始 | 单臂不能选择 winner | `INCOMPLETE / PAUSED`：本轮不恢复 B/C/D |

真实错误样本不得写入本文件。若需要回归，只能新增不含原文、ID、gold/prediction 的合成 fixture，并在 [`../ERROR_CASES.md`](../ERROR_CASES.md) 记录。

## 验证命令与结果

| 命令 | 本轮结果 |
|---|---|
| `git status --short --branch` | `PASS`：分支为 `codex/cross-benchmark-accuracy-family-v1`；执行开始时与 origin 对齐 |
| `python3 scripts/check_project_ledger.py` | `PASS`：`5 required files, 2 dated update, changelog links, public-safe markers` |
| `git diff --check -- project-ledger` + 新文件 trailing-whitespace scan | `PASS`：无输出、退出码 0 |
| `python3 scripts/check_benchmark_registry.py` | `PASS`：19/19 task；schema 1.0；runner keys aligned；无 dataset/API read |
| registry `--selftest`，normal 与 `python3 -O` | `PASS`：两次均为 17/17；bundle `d1f38dc38cd3745c8c84edc12390609e35580f8304322607424e5e16d366433b` |
| generic trainer `--selftest`，normal 与 `python3 -O` | `PASS`：两种模式均通过，无 model download |
| IMHI screen `--selftest`，normal 与 `python3 -O` | `PASS`：两种模式均通过固定四任务、单 Seed、三 loss arm synthetic matrix |
| 终审后 trainer `--selftest` + 显式 12-job registry proof audit | `PASS`：selftest 覆盖 binary/multiclass、imbalance、invalid/empty-class、deterministic metrics、checkpoint epoch、actual MPS exact RNG resume 与 device-aware fallback；proof audit 独立验证 12 jobs 并拒绝伪造 bundle |
| 终审后 screen `--selftest`，normal 与 `python3 -O` | `PASS`：aggregate preflight failure artifact、raw-message redaction、strict resume mode |
| IMHI screen 默认 metadata-only | `PASS`：不读取 external dataset；显式 `--validate-data` 才读取声明的 train/valid |
| IMHI screen 显式 `--validate-data` | `PASS / DRY_RUN`：12 jobs、上述四组 row counts；0 训练 job、0 test/holdout access |
| V4-Flash taxonomy smoke/full-valid/paired analysis | `COMPLETE / REJECTED_BY_GATE`：smoke 通过；full 1,459 完成；invalid、critical recall、macro inference 失败；未访问 test |
| 全仓 README regression matrix | `PASS`：prompt/parser 19/19、aggregate audit、scoreboard、baseline、safety、CPsyExam、PsySUICIDE v1/v2、EmoBench 与 IMHI selftests 全部退出 0 |
| IMHI DR/dreaddit/Irf/SAD single-Seed training | `NOT STARTED`：等待 clean、committed execution base |

## 结论边界

- 已证明：registry/trainer/screen 的离线与数据边界门禁通过；0731 Flash taxonomy 在完整 valid 的点值及 paired aggregate 已完成。
- 拒绝结论：taxonomy candidate 虽提高三个总体点值，但没有通过 invalid、关键类 recall 与 macro-F1 inference 门槛；不进入 test，也不替换 reference。
- 描述性：IMHI 只有 data-validation counts 与 planned 12-job manifest，没有训练分数。
- 不可比：IMHI inner-dev 与历史 sampled test/paper test；PsySUICIDE valid 与 official test/internal holdout。
- 未完成：IMHI 四任务 single-Seed training；任何三 Seed confirmation。
- 禁止声明：一套统一模型已跨 benchmark 提高；单 Seed 已稳定提高；PsySUICIDE v2 已选出 winner；任何 Web 部署已更新。

## GitHub 状态

- 执行基线：由本文件所在 clean Git HEAD 与 PR #5 live remote SHA 共同定义；提交/推送后以 Git/PR 实际状态为准
- Remote branch：`origin/codex/cross-benchmark-accuracy-family-v1`
- Draft PR：[PR #5](https://github.com/AllenLi441/mental-health-llm-eval/pull/5)
- 是否部署：否；评测仓库不代表 Web 生产部署

## 下一步与停止规则

1. Registry、trainer 与 data-validation 已过本地门禁；下一步先完成 diff、安全扫描、commit/push 与 live remote SHA 核验，未形成 clean committed base 前不得启动 12 个训练 job。启动后只在 ignored live manifest 标记 `RUNNING`，完成或失败后再追加本账本。
2. 0731 Flash taxonomy 已按失败停止：不得修改门槛后重跑，不访问 official test；如未来研究 invalid parser，只能作为新 preregistered experiment。
3. IMHI 固定 Seed `42`、DR/dreaddit/Irf/SAD 与 CE/weighted-CE/focal，只用已验证 train/valid。至少 3/4 任务相对各自 CE control 的 weighted-F1 提高 `0.02`、且任何任务不回退超过 `0.01`，才进入另行预注册的三 Seed confirmation。
4. PsySUICIDE v2 B/C/D 在本轮始终为 `PAUSED`；任何恢复需要用户新的明确决定与单独 eval 更新。
