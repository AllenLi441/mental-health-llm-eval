# 本地算力对接预案 + 生产端点预研

日期：2026-08-28　对应交接书 §0bis 第 7、8 条
状态：**条件预案**。用户尚未确定本地运行库的名称、所在机器、以及"训练用还是仅推理用"，本文按分支写死判据，机器一到位即可直接选路。

---

# 第一部分：训练算力（§0bis 第 7 条）

## 0. 先厘清一个常被混淆的点

"能跑 27B" 有两种截然不同的含义，本项目里差着一个数量级：

| | 4-bit 推理 | 4-bit QLoRA 训练 |
|---|---|---|
| 显存需求 | ~16–19 GB 权重 + KV cache | **≥72 GiB**（bundle `minimum_gpu_vram_gib`） |
| 24GB M4 | 勉强可行（须关掉微信/Chrome 等） | **不可行，且被本机资源纪律禁止** |
| 现成 bundle 可用 | 不适用 | 仅 CUDA |

所以"本地库能跑 Qwen3.8-27B"这句话本身**不足以决定 Phase 2 走哪条路**——必须先问清是哪一种。

## 1. 决策树

```
本地运行库到位
├─ Q1: 是训练还是仅推理？
│   ├─ 仅推理 ────────────────────► 路线 I（见 §2）
│   │                                训练仍需外部算力，Phase 2 不变
│   └─ 训练 ──► Q2: 什么加速器？
│       ├─ NVIDIA CUDA ≥80GB ────► 路线 C（见 §3）✅ 现成 bundle 直接可用
│       ├─ NVIDIA CUDA 48GB ─────► 路线 C'（实验档，须先 smoke 后决定）
│       ├─ Apple Silicon / MLX ──► 路线 M（见 §4）⚠ 需自建训练器 + 等效门禁
│       └─ 其他（ROCm / 国产卡）──► 按路线 M 的方法论处理，工作量同级或更大
```

## 2. 路线 I —— 本地库只做推理

**这是最可能的情形，也是最省事的。** 本地 4-bit 27B 推理在项目里有三处真实用途：

1. **静室开发/测试环境的后端**：不烧 API 额度地跑 `jingshi-eval` 45 例 replay 与危机管线回归。
2. **post-test 的本地复算臂**：微调完成后，本地起 OpenAI 兼容端口（Ollama / LM Studio / vLLM-metal 均可），对同一冻结 prompt 复跑一遍，与云端 API 臂交叉验证。
3. **adapter 落地验证**：merge LoRA 后先在本地确认输出不崩，再决定要不要花钱上托管。

**代价与限制**

- 权重 55.6 GB（BF16，18 分片）；4-bit 量化版约 16–19 GB。下载与磁盘占用要提前预留——注意本机 `mental-health-llm-eval` 已占 100G。
- 24GB 机器上跑 4-bit 27B 会挤掉几乎所有其他内存；属"单一重型进程"，跑之前必须先关其他应用，并且**不能与任何 build/test 同时进行**。
- **本地推理臂不能替代冻结协议的 API 臂**：量化会改变输出分布。若要进 pre/post 配对表，必须在 addendum 里作为**独立臂**登记（记录量化方式、runtime、fingerprint），不得与 BF16 API 臂混为一谈。

**结论：路线 I 下 Phase 2 训练方案不变**，仍需外部 80GB CUDA（租用或借用）。

## 3. 路线 C —— 本地有 CUDA ≥80GB

现成 bundle 原样可用，**这是唯一零改造路径**。按 `UPLOAD_AND_TRAIN.md` 执行：`sha256sum -c` → `bootstrap.sh` → `run_smoke_80gb.sh` → `run_train_80gb.sh`。

前置条件（缺一不可）：Linux x86_64、Python 3.11、torch `2.7.1+cu128`、torchvision `0.22.1+cu128`、Triton 3.3.1、CXX11 ABI enabled、≥160 GiB 可用磁盘。

**注意**：`bootstrap.sh` 会在安装前后比对 Python/Torch/CUDA/Triton/ABI，任何漂移即停。本地机器若已有别的 CUDA 环境，必须用干净 venv，不要复用。

路线 C' （48GB）是 bundle 里的 `experimental-48gb` 档，需显式确认，且必须先 smoke 再决定是否正式训练。

## 4. 路线 M —— Apple Silicon / MLX

**可行性：小模型已验证，27B 训练在 24GB 上不可行。** 社区已有 Qwen3.5 系在 mlx-lm 上做 LoRA 的完整例子，但落在 0.8B–4B 量级；把 27B QLoRA 塞进 24GB 不现实。若本地库指的是**另一台大内存 Apple Silicon**（如 M3 Ultra 512GB），路线 M 才成立。

**成立时必须自建的东西（现成 bundle 一行都用不上）**

| bundle 里的门禁 | CUDA 实现 | MLX 下要重做成什么 |
|---|---|---|
| 4-bit NF4 双量化（bitsandbytes） | `bitsandbytes==0.50.0` | mlx 量化，**量化语义不同，须记录并声明不等价** |
| GatedDeltaNet fast path 断言（48 层 + kernel 绑定） | `causal_conv1d` / `chunk_gated_delta_rule` | MLX 无对应；须换成等效的"架构确已加载正确"断言 |
| 496 个 language LoRA 模块精确计数 | transformers 模块名正则 | MLX 模块命名不同，正则与期望值都要重定 |
| 视觉塔冻结 + 可训练参数越界检查 | `named_parameters` 扫描 | 必须保留等效检查，**这是安全边界不是性能优化** |
| 两步强制 smoke + SHA-256 收据 | `run_smoke_80gb.sh` | **必须保留同等语义**：跑最长两条、NaN/Inf 即失败、产出绑定数据与代码哈希的收据 |
| 装前装后环境比对 | `bootstrap.sh` | 同等语义重写 |

**已知风险（须在 smoke 阶段专门验证）**

- mlx-lm 有一个未关闭的转换保真度 issue：Qwen3.5-4B 的 LoRA 合并后转 MLX，贪心解码前 20–30 token 与 HF 一致、之后发散，怀疑与 gated DeltaNet 状态处理有关。**合并 adapter 后必须与 HF 参考实现逐 token 对拍**，不对拍不得进 post-test。
- 该模型是视觉-文本统一模型，mlx-vlm 侧支持情况不稳定；纯文本 SFT 必须确认视觉塔被正确忽略/冻结。

**判据：路线 M 的改造成本（自建训练器 + 六项等效门禁 + 对拍验证）远高于借一台 80GB CUDA 机器跑几小时。** 除非本地就是大内存 Apple Silicon 且完全不打算用 CUDA，否则不建议走 M。

## 5. 机器到位后要问的五个问题（照抄即可）

1. 加速器型号与显存/统一内存容量？
2. 操作系统与架构（Linux x86_64？macOS arm64？）
3. 已装的 CUDA / PyTorch 版本（决定 `bootstrap.sh` 能否直接过）
4. 可用磁盘剩余（需 ≥160 GiB）
5. 这台机器是训练用、推理用，还是两者都要？

---

# 第二部分：生产端点（§0bis 第 8 条）

## 6. 硬约束：本地机器不能背生产流量

静室是 Vercel 上的公网服务。无论本地库多强，都不能作为生产后端：家宽 IP 不稳定、机器会睡眠、无 SLA、无横向扩展、且把家用机直接暴露给公网是安全问题。

**定位划分（建议写死进 Phase 5）**

- 本地库 = 开发 / 测试 / 评测复算
- 生产端点 = 托管或自托管的云端服务

## 7. 三条生产路线对比

| | 7a 供应商托管 LoRA | 7b 自托管 vLLM | 7c 合并权重后当自定义模型托管 |
|---|---|---|---|
| 做法 | 上传 adapter，供应商在其 Qwen3.8-27B 基座上挂载 | merge → 量化(AWQ/FP8) → serverless GPU 起 vLLM OpenAI 端点 | merge 后把完整 27B 权重交给支持自定义模型的平台 |
| 静室改动 | 仅 `THERAPY_*` env | 仅 `THERAPY_*` env | 仅 `THERAPY_*` env |
| 常驻成本 | 通常按 token | **有**（GPU 常驻或冷启动延迟） | 按平台，通常最贵 |
| 冷启动 | 无 | 有，可能数十秒——**对聊天体验致命** | 视平台 |
| 主要风险 | 未必有供应商支持自定义 LoRA | 运维全在自己身上 | 上传 55.6 GB 权重 + 存储费 |

**推荐顺序：7a → 7c → 7b。** 7b 只有在前两条都不可行时才选，因为静室是交互式聊天，冷启动延迟直接伤用户体验，而常驻 GPU 的月成本远高于当前 DeepSeek API 账单。

## 8. 待查（下周随本地方案一并落实）

- OpenRouter 只是聚合层，**不提供自定义 LoRA 托管**；需逐个核实底层供应商（Novita / Together / Fireworks 等）是否支持在 Qwen3.8-27B 基座上挂私有 adapter，以及是否要求 base 在其目录内。
- SiliconFlow 已在静室生产使用（判官 + embedding + reranker），**若其后续上架 3.8-27B 并支持自部署实例，是零新依赖的最优解**——需持续关注其目录。
- 各路线的实际月成本，需按静室真实日活与平均 token 数估算后再比。

## 9. 无论走哪条，Phase 5 的切换方式不变

1. 新增 `THERAPY_BASE_URL` / `THERAPY_MODEL` / `THERAPY_API_KEY`，未设置时行为与现在完全一致
2. 改 `src/lib/model-options.ts` 的 deep/fast 档位映射
3. DeepSeek 保留为超时/报错 fallback
4. 危机管线一行不动；危机快照测试禁 `-u`
5. 全量测试 + `npm run build` 绿 → push main → Vercel → 中英双语线上冒烟 + 危机 header 验证 → bump `APP_VERSION` 并报版本号
