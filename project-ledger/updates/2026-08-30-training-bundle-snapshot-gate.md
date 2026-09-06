# 2026-08-30：训练包上机就绪核验，并修掉一道比它守卫的下载更松的磁盘门

## 为什么更新

用户要求把 Qwen3.8-27B 的训练准备一次性做到「上机即可跑」。ERR-20260828-01 记录的三层完整性
不一致已在 8/29 受控重打中收口，本次要回答的是剩下那个问题：**在没有 GPU 的这台机器上，还有
哪一条准备项是可以做完却还挂着的？**

答案是一条：`snapshot_gib_estimate` 一直沿用 ERR-20260828-03 构造的估计值 60，等着上机第一次
`fetch_weights.py --dry-run` 用真实数字取代。本会话 hub 可达（8/28 时不可达，curl 全超时），
于是把这一步补跑了——**并且发现那个值不只是不准，它比它自己前置守卫的那步下载要得更少。**

## 修改范围

- 分支：`codex/esconv-2021-faithful-reproduction`（本次改动全部落在评测仓库外的
  `静室/datasets/mental-health-instruct-research-v0/`，与在飞的 ESConv 复现不相交）
- 文件（训练包内 4 个）：
  - `server_training_bundle/model_profiles/qwen3_8_27b.json`
  - `server_training_bundle/scripts/test_model_profiles.py`
  - `server_training_bundle/scripts/fetch_weights.py`（docstring）
  - `server_training_bundle/scripts/preflight.py`（注释）
  - 随后重打：`releases/*.tar.gz` + `releases/SHA256SUMS` + `BUNDLE_CONTENTS.sha256`
  - 事故记录：`INCIDENTS.md`（新增 ERR-20260830-01，并结案 ERR-20260829-01 的遗留边界）
- benchmark：不涉及（本次不产生任何分数）
- 是否涉及数据/付费 API/模型训练：**否**。零 API 调用、零训练、零付费；冻结的 500/50 未动一个字节。

## 做了什么

### 1. 补跑 `--dry-run`，拿到 hub 对该 revision 的真实答复

```
revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
32 个文件 / 55586114863 bytes = 51.77 GiB，其中 18 个分片 > 1 GiB
status: DRY_RUN_ONLY_NOTHING_DOWNLOADED（未下载任何字节）
```

8/28 那个构造出来的 50.41 GiB 与真值差 1.4 GiB——**原始大小估得很准，错的不是它。**

### 2. 发现缺陷：前置门比它守卫的那步要得更松

`snapshot_gib_estimate = 60` 低于 `fetch_weights.py` 自己对同一个卷的要求：

```
preflight.check_snapshot_disk()   要求缓存卷 ≥ 60    GiB   ← 先跑
fetch_weights.check_room_for()    要求缓存卷 ≥ 64.53 GiB   ← 后跑
                                  = 51.77 × 1.15 + 5（全部文件 pending，即冷取）
```

那 1.15 与 +5 GiB 不是保险系数：下载器给每个分片写 `.incomplete` 暂存文件，所以「刚好等于
下载量」的缓存必然失败（ERR-20260828-03 已记过这条）。于是一个缓存卷剩 62 GiB 的服务器：
preflight **全绿**，然后死在下载中途——正是这道门存在的唯一目的所指的那个失败。

60 这个数字从来没有和 `check_room_for()` 的公式对过账；8/29 的参数化只是把它从常量搬进
profile，没有重新审过它够不够。

### 3. 修复：把值改对，并把「为什么必须这么大」写进三处

- profile：`snapshot_gib_estimate` 60.0 → **64.53**，`snapshot_gib_basis` 改成实测记录
  （文件数 / 字节数 / GiB / 大分片数 / 测量日期 / 换 revision 要重跑 `--dry-run`），
  并写明**这个字段必须 ≥ `check_room_for()` 的冷取值**。
- 测试：控制组原来钉 `== 60.0`，改成 `== 64.53` 并注明它为什么被钉住——手改回一个更松的
  数字要让测试红，而不是让 preflight 放过一个下载会填满的卷。
- `fetch_weights.py`：docstring 原写「replaces preflight's ~60 GiB estimate」，已过时，改为
  记录实测值与 64.53 的来历。
- `preflight.py`：在 `check_snapshot_disk()` 补注这条不变量——**这道门至少和它前面守的那步
  一样严；保持这样。**

### 4. 复核「上机的人实际读到什么」，又抓到两处文档说得比真话强

改完代码之后去读上机会用到的两份文档，发现两处缺陷。**与本次门缺陷是同一失效类——声明比
事实强**——所以一并修掉：

1. `server_training_bundle/README.md` 写 fast-path kernel 绑定是「剩下的**唯一** GPU-only
   项」。**不是。** 完整清单四项：kernel 绑定 + `expected_lora_modules`(496) /
   `allowed_model_types` / `allowed_text_configs` 三道 `UNTESTED_NO_GPU` 门。这句话把待验项
   从 4 说成 1，而漏掉的恰好是那三道**本该拦住错模型**的门（它们在 ERR-20260829-01 的遗留
   边界里记着，只是没同步到 README）。已列全四项并写明第一次 smoke 的注入要求。
2. 两份文档只写下载量、不写门槛：README「~51 GiB，可中断续传」、`UPLOAD_AND_TRAIN.md`
   「至少 160 GiB 可用磁盘」。**给磁盘选型的人看的数字恰好不是门要求的那个数**：真实门槛
   比下载量多约 12.8 GiB（`.incomplete` 暂存），而 160 GiB 那条在异卷时对缓存卷完全不生效
   （ERR-20260828-03）。照原文配盘就会撞上本次这个缺陷的现场。

### 5. 把门真跑一次，抓到我自己刚写错的第三处

补第 4 节那两条时，我把门槛写成一句平铺的「缓存卷需 64.53 GiB 空闲」。**同卷时不是这个数**：
`required = max(160, 64.53)`，实际执行的是 **160**。也就是说，我为了修「文档报了一个不是门在
执行的数字」，又写出了一个不是门在执行的数字——**与 ERR-20260828-03 末尾那个消息缺陷一模一样
的错**，只是这次落在文档里而不是报错里。两份文档已改成同卷 / 异卷两行的表。

发现方式是这一节的重点：**不是再读一遍文档，而是把门真跑了一次。** `preflight.py` 只 import
标准库，所以本机能直接 `check_snapshot_disk(160.0)`——它报「run-wide minimum of 160 GiB
(…snapshot alone is ~65 GiB)」，与我刚写下的那句话当场对不上。读文档读不出这个矛盾。顺带确认
了 ERR-20260828-03 修的三分支消息行为正确：点的是真正生效的阈值，另一个数字放括号里作参考。

### 6. 重打并复验三层完整性（共三次）

改的是包内文件，所以必须重打，否则外层 sha 与内容脱钩——那正是 ERR-20260828-01。
`README.md` 在 `BUNDLE_CONTENTS.sha256` 的 27 项里，所以第 4 节触发**第二次**、第 5 节触发
**第三次**；`UPLOAD_AND_TRAIN.md` 在包外，不影响归档。下表是第三次重打后的结果。

## 当前指标

本次不产生任何 benchmark 指标（零 API 调用、零训练）。可核验的是准备状态：

| 项 | 结果 |
|---|---|
| 本机反向注入测试 | **30/30** 通过（控制组已按 64.53 更新） |
| 三层完整性 | `releases/SHA256SUMS` 归档 OK；`BUNDLE_CONTENTS.sha256` **27/27** OK；归档 27/27 与磁盘逐字节一致，且**无未列入清单的多余文件** |
| 许可随包 | `licenses/` 三份在包内（AugESC / CPCD / ESConv）——ERR-20260828-01 那次漏的就是整个目录 |
| 确定性 | 连续重打 **IDENTICAL**（三轮改动都没破坏它） |
| 冻结数据 | train 500 行 `a22d0dcd2cc1148f` / dev 50 行 `1b8c237a62bd2a07` / manifest `6fbb81057ff21376`；**归档内 payload** 与 `exports/` 逐字节相同 |
| 3.8 钉定 | `model_id` + `revision` 在 4 个 config 与 profile 中一致；包内**无 3.6 残留**；profile `status=VERIFIED` |
| 门实跑 | `check_snapshot_disk(160.0)` 真调用一次：读到 profile 的 64.53，报的是真正生效的 **160**（同卷），括号里附快照 ~65 GiB 作参考 |
| 运行时清洁 | runbook 点名的 9 个脚本全部在位；9 个脚本语法干净；8 个 JSON 可解析 |

归档 sha256 因重打而变；**新值只写在 `releases/SHA256SUMS`，本文件与任何文档都不抄**——
手抄归档 sha 与 ERR-20260828-01 是同一个失效类。

## 错误与失败用例

### ERR-20260830-01：快照磁盘门比它前置守卫的那步下载要得更少

见上文第 2 节；完整记录在训练包的 `INCIDENTS.md`。值得单列成一类的原因不是数字估偏了：

**一道前置门比它守卫的那步要得更松，等于这道门不存在**，而且比不存在更坏——它会给出一份
绿色的 preflight 报告，把读的人的注意力从磁盘上引开。ERR-20260828-03 写下这道门时防的是
「量错了卷」，这次防的是「量对了卷、阈值取小了」，是同一道门的第二种失效。

顺带记两条方法论：

1. **改值让控制组变红，是控制组在干活。** 本次 `test_model_profiles.py` 先红后绿，红的那次
   提醒了我这个常量有测试在钉——如果当时选择「把测试改软」而不是「把值改对并注明」，
   下一个人手改回 60 就再没有东西会拦。
2. **是写注释时发现的（与 ERR-20260829-01 同一个来源）。** 我先写下「这道门至少和它守的那步
   一样严」，然后去核实，发现并不是。

### 同类：三处文档声明强于事实（并入 ERR-20260830-01 记录，未另编号）

见上文第 4、5 节。合成安全的复现方式就是**照文档配一台机器**：按「~51 GiB」或「训练包目录
160 GiB」去配缓存卷，preflight 全绿、下载中途失败；按「唯一 GPU-only 项是 kernel 绑定」
去安排上机验证，则三道 `UNTESTED_NO_GPU` 门一次都不会被注入测试；按我修完后那句平铺的
「缓存卷需 64.53」去配同卷机器，则真正卡住你的是 160 而不是 64.53。

三条留给自己的判据：

1. **「剩下唯一一项」这类计数式声明，写的时候必须去数一遍。** 它比一个错误的数字更容易骗过
   复核——因为它读起来像结论，不像数据。
2. **门槛只要依赖分支，就不能写成一个平铺的数字。** 第 3 处缺陷是我在修第 2 处时新写出来的：
   同一个失效类可以在修复动作里复发，所以修完要按门的**分支结构**回读一遍，而不是按句子。
3. **能跑的门就别只读。** 前两处是读文档发现的，第三处是**跑一次 `check_snapshot_disk()`**
   发现的——`preflight.py` 只 import 标准库，这类校验在本机零成本，却比再读三遍都有效。

## 验证

```
python3 scripts/fetch_weights.py --dry-run        # PASS  32 文件 / 55586114863 bytes / 51.77 GiB
                                                 #       DRY_RUN_ONLY_NOTHING_DOWNLOADED
python3 scripts/test_model_profiles.py            # FAIL 1 项（控制组钉 60.0）→ 改值后 PASS 30/30
python3 scripts/package_bundle.py（第 1 次，改代码）    # PASS  内容契约 OK，27 个 payload 重算
python3 scripts/package_bundle.py（第 2 次，改 README） # PASS  README 是 payload，必须再打
python3 scripts/package_bundle.py（第 3 次，同卷/异卷表）# PASS  同上
shasum -a 256 -c releases/SHA256SUMS              # PASS  1/1（每次重打后各验一遍）
shasum -a 256 -c BUNDLE_CONTENTS.sha256           # PASS  27/27，无 FAILED
归档 ↔ 磁盘逐文件比对                              # PASS  27/27 一致，且无未列入清单的多余文件
连续重打比对                                       # PASS  IDENTICAL（三轮改动都没破坏确定性）
tar tzf releases/*.tar.gz | grep licenses/        # PASS  三份许可在包内（ERR-20260828-01 回归检查）
归档内 data/*.jsonl ↔ exports/*.jsonl             # PASS  500 / 50 行，逐字节相同
check_snapshot_disk(160.0) 真调用                 # PASS  报真正生效的 160，附快照 ~65 GiB 作参考
9 个脚本语法 + 8 个 JSON 解析                      # PASS  全部干净
4 个 config + profile 的 model_id/revision        # PASS  全钉 3.8，包内无 3.6 残留
python3 scripts/check_project_ledger.py           # PASS  5 required files, 5 dated update, changelog links
```

失败那一项（控制组钉 60.0）不省略：它是本次改动的第一个信号，不是意外。

另有两次**我自己的检查脚本出错**，不是被检对象的问题，一并记下以免后来的人误读：
① 一轮 `cut`/`sed`/`sort` 报 `command not found`（临时 shell 状态问题），导致两个空字符串相比，
把 27/27 全报成 DIFF——已改用单个 Python 进程重跑，结果 0 差异；② 一条 `grep '64'` 断言报 False，
因为报错信息用 `:.0f` 把 64.53 渲染成 `~65`。**两次都是检查方法错，不是产物错。**

顺带记一笔：`py_compile` 会在包内留下 `scripts/__pycache__`。它在 `package_bundle.py` 的
`EXCLUDED_DIRS` 里，所以**不会进归档**，但会脏工作树、让 `git status` 与人工核对多一层噪音。
本次已删净。

## 结论边界

**已证明**

- hub 对 revision `1d4bf0f2…` 的答复是 32 文件 / 55586114863 bytes；64.53 GiB 是把它代入
  `check_room_for()` 冷取公式的结果，因此快照门现在不低于它守卫的那步下载。
- 训练包三层完整性一致、可确定性重建、冻结 500/50 未动。

**描述性**

- 「上机就绪」指的是**本机可核验的每一项都已核验**，不等于上机一定跑通。没执行过的是：真正的
  4-bit 载入、下载本身，以及**四项 GPU-only 检查**——fast-path kernel 的绑定，加上
  `expected_lora_modules`(496) / `allowed_model_types` / `allowed_text_configs` 这三道
  `UNTESTED_NO_GPU` 门。README 此前把这里写成「剩下唯一一项」，本次已改成四项全列（见第 4 节）。

**不可比**

- 51.77 GiB 只属于这个 revision。换 revision 必须重跑 `--dry-run` 并同步改 profile 与测试里
  钉的数——两处不同步就是回到本次这个缺陷。
- 14B / 7B profile 仍是 `UNVERIFIED_PLACEHOLDER`（null 门）。这个字段同样不能按 27B 等比缩放
  算出来，那会得到**看起来已验证的假数字**，而它喂的正是本该拦住错模型的那些门。

**未完成**

- 三道门本机测不了，仍标 `UNTESTED_NO_GPU`：`expected_lora_modules`（496）、
  `allowed_model_types`、`allowed_text_configs`——只在真正 4-bit 载入后才执行。上机第一次
  smoke 要各注入一次刻意的不匹配，才算从「读到了」升级为「读错了会拦」。
- fast-path kernel 的**绑定**只能上卡实测（可安装性已验）。
- 本机缓存卷余量不足以冷取，**下载这一步依然只能上机执行**；本次验的是阈值对不对。
- 三项许可核实（AntEngage / SoulChat corpus / OnCoCo）仍挂着，与本次无关。

## GitHub 状态

- Commit SHA：未提交（工作区改动；训练包位于评测仓库之外）
- Remote branch：`codex/esconv-2021-faithful-reproduction`
- Draft PR：无（本次不进 PR #5）
- 是否部署：评测仓库不涉及部署

## 下一步与停止规则

1. 上机后**先**跑 `fetch_weights.py --dry-run` 再取权重；若报的 required 与 64.53 不一致，
   说明 revision 动了，**停下来重新测量并改 profile + 测试**，不要临时放宽门槛。
2. 第一次 smoke 必须对三道 `UNTESTED_NO_GPU` 门各做一次刻意不匹配注入；任一门没拦住，
   **停止全量跑**，先修门。
3. 产出的 adapter 保持 `RESEARCH_ADAPTER_UNEVALUATED`，未经同协议评测不得接入静室生产端点。
4. 禁止用 test 集调参；48GB 路径仍属需显式确认的实验分支。

