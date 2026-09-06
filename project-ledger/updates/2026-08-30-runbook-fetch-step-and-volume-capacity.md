# 2026-08-30：runbook 漏掉取权重那一步，且两份文档的磁盘数字只够撑到下载结束

## 为什么更新

同日的[快照磁盘门修正](2026-08-30-training-bundle-snapshot-gate.md)把「文档写的和代码强制的
不一致」这一类缺陷收了一个，但收的是**门的阈值**。接着换了一个问法复核：**上机的人拿着
`UPLOAD_AND_TRAIN.md` 一条一条照敲，到底能不能走到训练？**

不能。两处断掉，而且两处都是散文级复核抓不到的形状：一处是「少了一行」，一处是「数字在某个
时点之后才变错」。合起来的效果是：GPU 已经在计费之后才暴露。

用户当日通知高性能机器要**推迟到下周**才可用，本次两条修复正好落在这段多出来的核验时间里。

## 修改范围

- 分支：`codex/esconv-2021-faithful-reproduction`（改动全部落在评测仓库外的
  `静室/datasets/mental-health-instruct-research-v0/`，与在飞的 ESConv 复现不相交）
- 文件：
  - `UPLOAD_AND_TRAIN.md`（**包外**，不触发重打）：补第 5 步、重写租用规格
  - `server_training_bundle/README.md`（**payload**，因此触发第四次重打）：规格条目 + 容量表
  - 随后重打：`releases/*.tar.gz` + `releases/SHA256SUMS` + `BUNDLE_CONTENTS.sha256`
  - 事故记录：`INCIDENTS.md`（新增 ERR-20260830-02）
- benchmark：不涉及（本次不产生任何分数）
- 是否涉及数据/付费 API/模型训练：**否**。零 API 调用、零训练、零付费、零下载；冻结的
  500/50 未动一个字节（重打后逐字节比对过）。
- **代码一行未改。** 两道门的阈值和逻辑都没动，理由见「结论边界」。

## 做了什么

### 缺陷 A：runbook 里 `fetch_weights.py` 出现 0 次

`grep -c fetch_weights UPLOAD_AND_TRAIN.md` = 0。runbook 的编号步骤是
解压 → `bootstrap.sh` → `run_smoke_80gb.sh` → `run_train_80gb.sh`，**中间没有取权重**；
README 的「80GB 启动步骤」里有这一步，runbook 没跟着同步。

要命的地方是它**不会报错**：

- `bootstrap.sh` 全程不调 `fetch_weights.py`（77 行逐行确认）。
- 两个 runner 也不调。
- `train_qlora.py:126` 的 `from_pretrained()` **不带 `local_files_only`**，所以缺快照时它不是
  失败，而是**当场联网下载**。
- preflight 里没有任何「快照是否已就位」的门。

于是照 runbook 敲的实际效果是把 51.77 GiB 下载搬进 smoke 内部——网络慢、缓存卷满、repo 变
gated 这三类问题全部推迟到计费 GPU 已经在跑之后才暴露。而 ERR-20260828-03 之所以把
`fetch_weights.py` 拆成独立一步，要的正是相反的效果。

已补成第 5 步「取权重快照（不可跳过）」（`--dry-run` + 正式下载两条命令，并写明为什么跳过它
不会报错），后续步骤重新编号 6/7/8。

### 缺陷 B：两份文档把「门每次要求的空闲」当成「要配的容量」

`check_snapshot_disk()` 不是只在 bootstrap 跑一次。它在 `verify_environment()` 里
（`preflight.py:1137`），四个 runner 全部传 `--environment-check`，`train_qlora.py:277` 还会
再调一次——也就是**下载完成之后又量一遍缓存卷空闲**，而那时快照本身正占着 51.77 GiB。门从不问
「是不是已经缓存好了」，要求一分不降。按文档给的数字配盘的后果（本机模拟验证）：

```
异卷、缓存卷配 64.53 GiB   下载后剩 12.76  <  门要 64.53   → BLOCKED
同卷、配 160 GiB           下载后剩 108.23 <  门要 160     → BLOCKED
异卷、配 120 GiB           下载后剩 68.23  ≥  门要 64.53   → PASS
同卷、配 220 GiB           下载后剩 168.23 ≥  门要 160     → PASS
```

失败时序是最坏的那种：`bootstrap.sh` 全绿 → 下载 51.77 GiB 全绿 → **smoke 起手被拦**，而且拦在
一个「你以为早就过了」的检查上。

两份文档现在都分两列写：

| 情形 | 门每次要求的**空闲** | 实际应配**容量** |
|---|---|---|
| 缓存卷与训练包**异卷**（推荐，`HF_HOME=/<大盘>/hf`） | 64.53 GiB | **≥ 120 GiB** |
| 缓存卷与训练包**同卷** | 160 GiB（`max(160, 64.53)`） | **≥ 220 GiB** |

右列 = 左列 + 51.77 GiB 常驻快照，取整留余量（`64.53 + 51.77 = 116.3 → 120`，
`160 + 51.77 = 211.8 → 220`）。README 的规格条目也从「至少 160 GiB 可用」改为指向这张表。

### 第四次重打

`README.md` 在 `BUNDLE_CONTENTS.sha256` 的 27 项里，所以改它必须重打，否则外层 sha 与内容脱钩
——那正是 ERR-20260828-01。重打前 layer 2 确实报 `README.md: FAILED`（1 项不匹配），重打后
27/27 OK。新归档 sha **只写在 `releases/SHA256SUMS` 里，本文件与任何文档都不抄**。

## 当前指标

不涉及。本次不产生任何 benchmark 分数，不改任何指标口径。

## 错误与失败用例

- `INCIDENTS.md` 新增 **ERR-20260830-02**（两条缺陷合为一条记录，因为它们是同一次复核、同一个
  失效类、同一批修复）。
- 本仓 `ERROR_CASES.md` 新增同 ID 的聚合行。
- 与 ERR-20260830-01 的关系：**同一个失效类的第二和第三例**——代码实际强制的东西比文档写的更严，
  而文档被当成了真相。三例都不是靠重读文档发现的，是靠把门真跑一遍 / 把 runbook 当作会被逐条
  照敲的脚本来推演。

## 验证

```
grep -c fetch_weights UPLOAD_AND_TRAIN.md（修前）      = 0    ← 缺陷 A 的证据
bootstrap.sh / 两个 runner 是否调 fetch_weights        无     ← 77 行 + 两脚本逐行确认
train_qlora.py:126 是否带 local_files_only             无     ← 缺陷 A 为何静默失败
容量模拟 70/120/160/220 GiB                            PASS  BLOCKED/PASS/BLOCKED/PASS 如上表
UPLOAD_AND_TRAIN.md 标题编号 1–8 连续、无残留交叉引用    PASS  （补第 5 步时曾误造两个「## 6.」，
                                                             重 grep 标题后改正）
python3 scripts/package_bundle.py（第 4 次）            PASS  内容契约 OK，27 个 payload 重算
shasum -a 256 -c releases/SHA256SUMS（在 releases/ 内）  PASS  1/1
shasum -a 256 -c BUNDLE_CONTENTS.sha256                PASS  27/27，无 FAILED
连续重打比对                                            PASS  IDENTICAL
归档 ↔ 磁盘逐文件比对                                   PASS  27/27 一致
归档内是否有未列入清单的文件                             1 项  仅 BUNDLE_CONTENTS.sha256 本身
                                                             （清单不自含，符合设计）
tar 内 licenses/                                       PASS  3 份（ERR-20260828-01 回归检查）
归档内 data/*.jsonl ↔ exports/*.jsonl                  PASS  500 / 50 行，逐字节相同
python3 scripts/check_project_ledger.py                见下
```

`package_bundle.py` 与 `check_project_ledger.py` 都必须在**各自仓库根目录**跑，不在
`scripts/` 或 `project-ledger/` 里跑——这两个路径坑本会话各踩过一次，都是检查方法错，不是产物错。

## 结论边界

- **已证明**：按修后文档配盘并按修后 runbook 执行，能一路通过所有本机可模拟的门。三层完整性在
  第四次重打后重新一致。
- **描述性**：「上机就绪」仍只表示本机可核验的每一项都已核验。未执行的仍是四项 GPU-only——
  fast-path kernel 的**绑定**，加上 `expected_lora_modules`(496) / `allowed_model_types` /
  `allowed_text_configs` 这三道 `UNTESTED_NO_GPU` 门——外加真正的 4-bit 载入与下载本身。
- **未完成 / 明确不做**：**没有从代码层消除缺陷 A 的成因。** 包里仍然没有门要求快照已就位，
  只是在 runbook 里补了字。是否加一道「快照未完整缓存则 smoke 拒绝启动」的门，留待上机后连同
  四项 GPU-only 检查一起决定；真要加必须同时在 `test_model_profiles.py` 补覆盖。
- **为什么不放宽磁盘门**：「下载完就别再要求全额空闲」需要门去判断快照是否已完整缓存，而这个
  判断只有在一台我无法实测的机器上才会真正被执行。放宽一道安全门去省 60 GiB 磁盘，风险远大于
  多买 60 GiB。所以改文档给出的容量，让照文档配的机器能通过门，而不是让门迁就文档。
  `check_snapshot_disk` 的注释里写着同一条教训的正方向（ERR-20260830-01：前置门必须至少和它挡的
  那一步一样严）；往松改它就是把那条教训反着用。
- **不可外推**：容量表里的 51.77 与 64.53 GiB 都绑定 revision `1d4bf0f2…`，换 revision 两个数
  都要重算。14B / 7B profile 仍是 `UNVERIFIED_PLACEHOLDER`，不得按 27B 等比缩放。

## GitHub 状态

- Commit SHA：未提交（改动在评测仓库外的 `静室/datasets/`，本仓只新增本记录与账本行）
- Remote branch：`codex/esconv-2021-faithful-reproduction`（未推送本次改动）
- Draft PR：不适用
- 是否部署：**否**。评测仓库不涉及部署；训练也尚未开始（机器推迟到下周）。

