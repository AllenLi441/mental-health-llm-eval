# Phase 1 上机前交付说明

本目录是正式产品数据线；当前不能直接训练。

## 当前可用与不可用

- `production/candidates/`：候选和空白 writer 包，不是训练数据。
- `production/canonical/approved_records.jsonl`：当前不存在，因此批准训练行数为 0。
- `production/exports/`：正式 server-safe export 尚未生成。
- `server_training_bundle/releases/mhi-v2-smoke-only.tar.gz`：只含代码和 smoke fixture，可做两步丢弃式 GPU smoke，不含训练数据。
- `manifests/training_readiness.json`：用 `tools/update_training_readiness.py` 重算；当前结论是 `NOT_READY_FOR_FORMAL_TRAINING`。

## 正式数据目标

- train 6,000：中文 3,600 / 英文 2,400；
- development 600：中文 360 / 英文 240；
- Human Gold Test 200 和 Safety Eval 400 独立冻结，绝不训练；
- 只有真人独立写作、盲复核、必要的专业签核、PII/权利/污染门均通过的记录，才能进入 canonical。

## 正式训练前顺序

```bash
cd /Users/allenli/Desktop/静室/datasets/mental-health-instruction-dataset-v2
python3 tools/update_training_readiness.py
python3 server_training_bundle/scripts/preflight.py --selftest
```

只有 readiness 变为 `READY_FOR_FORMAL_TRAINING` 后，才运行 `build_exports.py` 和 `package_server_bundle.py`，然后在大机器上做 image probe → smoke → formal QLoRA。

模型尺寸（Qwen3.8-27B、14B 或 7B）必须在实际 checkpoint 加载后再确定；不得用比例估算 profile，也不得用 AI draft 冒充 gold。

完整的数据、对照和评测合同见：
`/Users/allenli/Desktop/mental-health-llm-eval/reports/PHASE1_DATA_EVAL_CONTRACT_20260905.md`
